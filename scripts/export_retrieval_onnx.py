"""Export retrieval embeddings with dynamic batches, views, and padding masks.

The training model groups collections by their valid view count using Python.
This adapter instead masks padded attention keys, which gives the same valid
tokens and CLS embedding without freezing a particular mask during tracing.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import MultiImageHybrid  # noqa: E402


class RetrievalEmbeddingExport(nn.Module):
    """Return fused [B,D] and per-view [B,N,D] normalized embeddings.

Inputs are normalized RGB images [B,N,3,H,W] and a boolean mask [B,N].
Every row must have at least one valid view, with 1 <= N <= model.n.
Invalid per-view output embeddings are zero. Collection positions follow the
order of valid views, including when the mask has holes or leading padding.
"""

    def __init__(self, model):
        super().__init__()
        vit = model.model
        if (vit.num_prefix_tokens != 1 or vit.cls_token is None
                or vit.global_pool != "token"):
            raise ValueError("Export requires one CLS token and token pooling")
        if any(type(block).__name__ != "Block" for block in vit.blocks):
            raise ValueError("Export requires standard timm transformer Blocks")
        self.retrieval_model = model

    @staticmethod
    def _masked_attention(attention, tokens, token_mask):
        batch, length, channels = tokens.shape
        qkv = attention.qkv(tokens).reshape(
            batch, length, 3, attention.num_heads, attention.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = attention.q_norm(query) * attention.scale
        key = attention.k_norm(key)
        scores = query @ key.transpose(-2, -1)
        scores = scores.masked_fill(~token_mask[:, None, None, :], float("-inf"))
        weights = attention.attn_drop(scores.softmax(dim=-1))
        result = (weights @ value).transpose(1, 2).reshape(batch, length, channels)
        return attention.proj_drop(attention.proj(result))

    def forward(self, images, view_mask):
        model = self.retrieval_model
        vit = model.model
        batch, views = images.shape[:2]
        patches = vit.patch_embed(images.reshape(-1, *images.shape[2:]))
        tokens = vit._pos_embed(patches)
        single = F.normalize(model._encode_tokens(tokens), p=2, dim=-1)
        single = single.reshape(batch, views, model.embed_dim)
        single = single * view_mask.unsqueeze(-1).to(single.dtype)

        tokens_by_view = tokens.reshape(batch, views, tokens.shape[1], model.embed_dim)
        # In eval mode, each image's CLS token has the same value before the
        # transformer. Keeping one of them is equivalent to the packed path.
        cls_token = tokens_by_view[:, 0, :1, :]
        position_indices = (view_mask.to(torch.int64).cumsum(dim=1) - 1).clamp(min=0)
        positions = F.normalize(model.img_embed_matrix[0], p=2, dim=-1)
        image_positions = positions[position_indices]
        collection_patches = tokens_by_view[:, :, 1:, :] + image_positions[:, :, None, :]
        patch_count = collection_patches.shape[2]
        collection = torch.cat((
            cls_token, collection_patches.reshape(batch, -1, model.embed_dim)), dim=1)
        patch_mask = view_mask.unsqueeze(-1).expand(batch, views, patch_count)
        token_mask = torch.cat((
            torch.ones_like(view_mask[:, :1]), patch_mask.reshape(batch, -1)), dim=1)

        for block in vit.blocks:
            attention = self._masked_attention(block.attn, block.norm1(collection), token_mask)
            collection = collection + block.drop_path1(block.ls1(attention))
            collection = collection + block.drop_path2(block.ls2(block.mlp(block.norm2(collection))))
        collection = vit.forward_head(vit.norm(collection), pre_logits=True)
        return F.normalize(collection, p=2, dim=-1), single


def validate_export(model, adapter, session, image_size):
    """Check dynamic shapes, mixed counts, mask holes, and padding invariance."""
    generator = torch.Generator().manual_seed(20261008)
    masks = [
        [[True]],
        [[True, True], [False, True]],
        [[True, True, True]],
        [[False, False, True, False],
         [True, False, False, True],
         [True, True, False, True],
         [True, True, True, True]],
    ]
    reports = []
    for values in masks:
        mask = torch.tensor(values, dtype=torch.bool)
        images = torch.randn(*mask.shape, 3, image_size, image_size, generator=generator)
        with torch.no_grad():
            original = model.forward_embeddings(images, view_mask=mask)
            expected_joint = original["mv_collection"]["embeddings"].numpy()
            expected_views = original["single"]["embeddings"].reshape(
                *mask.shape, model.embed_dim).numpy() * mask.numpy()[..., None]
            adapted_joint, adapted_views = (x.numpy() for x in adapter(images, mask))
        inputs = {"images": images.numpy(), "view_mask": mask.numpy()}
        actual_joint, actual_views = session.run(None, inputs)
        for actual, expected in (
                (adapted_joint, expected_joint), (adapted_views, expected_views),
                (actual_joint, expected_joint), (actual_views, expected_views)):
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(actual_joint, axis=-1), 1, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(actual_views[mask.numpy()], axis=-1), 1, atol=1e-5)
        # Invalid pixels cannot alter a collection or a valid view embedding.
        changed = images.clone()
        changed[~mask] = torch.randn(changed[~mask].shape, generator=generator) * 10
        changed_joint, changed_views = session.run(None, {
            "images": changed.numpy(), "view_mask": mask.numpy()})
        np.testing.assert_allclose(changed_joint, actual_joint, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(changed_views, actual_views, rtol=1e-4, atol=1e-5)
        report = {
            "batch": mask.shape[0], "input_views": mask.shape[1],
            "valid_view_counts": mask.sum(dim=1).tolist(),
            "joint_max_absolute_error": float(np.max(np.abs(actual_joint - expected_joint))),
            "view_max_absolute_error": float(np.max(np.abs(actual_views - expected_views))),
            "padding_invariance": "passed",
        }
        reports.append(report)
        print("Validation:", json.dumps(report), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--architecture", default="vit_small_r26_s32_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    print(f"Loading {checkpoint}", flush=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    epoch = payload.get("epoch")
    if args.expected_epoch is not None and epoch != args.expected_epoch:
        raise ValueError(f"Expected epoch {args.expected_epoch}, found {epoch}")
    state = {key.removeprefix("module."): value for key, value in payload["model"].items()}
    max_views = state["img_embed_matrix"].shape[1]
    if max_views != 4:
        raise ValueError("Validation cases currently require a four-view checkpoint")
    model = MultiImageHybrid(
        args.architecture, num_classes=state["model.head.weight"].shape[0],
        n=max_views, pretrained_weights=False, image_size=args.image_size)
    model.load_state_dict(state, strict=True)
    best_score = payload.get("best_score")
    del state, payload
    model.eval()
    adapter = RetrievalEmbeddingExport(model).eval()
    print(f"Loaded epoch {epoch}, embedding size {model.embed_dim}, maximum views {max_views}", flush=True)

    with tempfile.TemporaryDirectory(prefix="retrieval-onnx-") as temporary:
        local_path = Path(temporary) / output.name
        example_images = torch.zeros(1, max_views, 3, args.image_size, args.image_size)
        example_mask = torch.ones(1, max_views, dtype=torch.bool)
        print("Exporting ONNX with embedded float32 weights", flush=True)
        with torch.no_grad():
            torch.onnx.export(
                adapter, (example_images, example_mask), str(local_path),
                input_names=["images", "view_mask"],
                output_names=["joint_embedding", "view_embeddings"],
                dynamic_axes={
                    "images": {0: "batch", 1: "views"},
                    "view_mask": {0: "batch", 1: "views"},
                    "joint_embedding": {0: "batch"},
                    "view_embeddings": {0: "batch", 1: "views"},
                },
                opset_version=17, export_params=True,
                do_constant_folding=True, dynamo=False)
        graph = onnx.load(local_path)
        stat = checkpoint.stat()
        provenance = {
            "checkpoint": str(checkpoint), "epoch": epoch,
            "checkpoint_size_bytes": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
            "best_validation_hybrid_recall_at_1": best_score,
            "architecture": args.architecture, "max_views": max_views,
            "embedding_dim": model.embed_dim, "image_size": args.image_size,
            "preprocessing": {
                "color": "RGB", "scale": "uint8 to float32 in [0,1]",
                "resize_shorter_side": round(args.image_size * 256 / 224),
                "resize_interpolation": "bilinear", "center_crop": args.image_size,
                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            },
            "inputs": {
                "images": ["batch", "views", 3, args.image_size, args.image_size],
                "view_mask": ["batch", "views"],
            },
            "input_constraints": "batch >= 1; 1 <= views <= 4; at least one true mask entry per row",
            "outputs": {
                "joint_embedding": ["batch", model.embed_dim],
                "view_embeddings": ["batch", "views", model.embed_dim],
            },
            "output_semantics": {
                "joint_embedding": "L2-normalized fused multi-image embedding for direct gallery-image matching",
                "view_embeddings": "L2-normalized single-image branch embeddings; masked views are zero",
            },
            "single_image_retrieval": "Use view_embeddings[:,0,:] with a one-view input for gallery images and single-image queries",
            "hybrid_retrieval": "Both outputs are available; hotel score aggregation runs outside this embedding model",
            "opset": 17, "precision": "float32", "external_weights": False,
            "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        }
        onnx.helper.set_model_props(graph, {
            "retrieval_metadata": json.dumps(provenance, sort_keys=True)})
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, local_path)
        del graph
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(local_path), sess_options=options, providers=["CPUExecutionProvider"])
        print("Checking original PyTorch outputs against ONNX Runtime", flush=True)
        provenance["validation"] = {
            "status": "passed", "provider": "CPUExecutionProvider",
            "rtol": 1e-4, "atol": 1e-5,
            "cases": validate_export(model, adapter, session, args.image_size),
        }
        digest = hashlib.sha256()
        with local_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        provenance["onnx_sha256"] = digest.hexdigest()
        provenance["onnx_size_bytes"] = local_path.stat().st_size
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as destination:
            destination_path = Path(destination.name)
        try:
            shutil.copyfile(local_path, destination_path)
            destination_path.chmod(stat.st_mode & 0o777)
            destination_path.replace(output)
        finally:
            destination_path.unlink(missing_ok=True)
        output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved verified export: {output} ({provenance['onnx_size_bytes']:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
