"""Check a finished run and compare direct image retrieval using existing caches.

This script never extracts images or changes checkpoints/original metrics. A
not-before time supports a one-off host job when chat scheduling is unavailable.
"""

import argparse
import ast
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def evaluation_settings(run_dir):
    values = {}
    for line in (Path(run_dir) / "retrieval.log").read_text().splitlines():
        match = re.search(r" - INFO - (\w+): (.+)$", line)
        if match:
            values[match[1]] = match[2]
    settings = {
        key: values[key] for key in ("data_dir", "architecture", "embedding_cache_dir")
    }
    for key in (
        "image_size", "model_max_views", "max_query_views", "test_min_views",
        "test_collections_per_group", "max_test_queries", "max_test_image_queries",
        "seed", "retrieval_query_chunk_size", "retrieval_gallery_chunk_size",
    ):
        settings[key] = int(values[key])
    settings["skip_missing"] = ast.literal_eval(values["skip_missing"])
    settings["recall_at"] = tuple(ast.literal_eval(values["recall_at"]))
    return settings


def direct_metrics(run_dir, settings, device, logger, *, cache_tag=None, checkpoint_epoch=None):
    import torch

    from engine.retrieval_evaluator import RetrievalEvaluator
    torch.set_float32_matmul_precision("high")
    run_dir = Path(run_dir)
    checkpoint = run_dir / "best.pth"
    if (cache_tag is None) != (checkpoint_epoch is None):
        raise ValueError("An explicit cache tag requires its checkpoint epoch")
    if cache_tag is not None:
        if not re.fullmatch(r"[a-f0-9]{16}", cache_tag) or checkpoint_epoch < 1:
            raise ValueError("Invalid explicit cache tag or checkpoint epoch")
        # A completed run's cache fingerprint can be supplied explicitly to
        # avoid rereading large checkpoints on a slow filesystem. Only cached
        # embeddings are scored; the checkpoint remains a provenance reference.
        tag, epoch = cache_tag, checkpoint_epoch
        provenance = "explicit cache fingerprint and epoch from completed run"
    else:
        from retrieval_main import _embedding_cache_tag

        tag = _embedding_cache_tag(SimpleNamespace(**settings), checkpoint)
        provenance = "checkpoint hash and saved epoch"
    logger.info("Loading cached retrieval inputs for %s (cache %s)", run_dir.name, tag)
    cache_dir = Path(settings["embedding_cache_dir"])
    required = [cache_dir / f"gallery-{tag}.pt"]
    for split in ("test_room", "test_object"):
        required.extend((cache_dir / f"{split}-{tag}.pt",
                         cache_dir / f"{split}-single-{tag}.pt"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required embedding caches missing: " + ", ".join(missing))

    if cache_tag is None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        epoch = int(payload["epoch"])
        del payload
    evaluator = RetrievalEvaluator(device=device, logger=logger, use_amp=False)
    # Empty loaders ensure this pass can only use existing cached embeddings.
    gallery = evaluator.extract_gallery([], cache_path=required[0])
    logger.info("Loaded %d cached gallery images for epoch %d", len(gallery["labels"]), epoch)
    original = json.loads((run_dir / "retrieval_test_metrics.json").read_text())
    result = {"checkpoint": str(checkpoint), "epoch": epoch, "cache_tag": tag, "provenance": provenance,
              "settings": settings, "metrics": {}}
    kwargs = dict(
        ks=settings["recall_at"], compute_image_metrics=True,
        compute_hotel_metrics=False,
        query_chunk_size=settings["retrieval_query_chunk_size"],
        gallery_chunk_size=settings["retrieval_gallery_chunk_size"],
    )
    for split in ("test_room", "test_object"):
        queries = evaluator.extract_queries([], cache_path=cache_dir / f"{split}-{tag}.pt")
        single = evaluator.extract_gallery([], cache_path=cache_dir / f"{split}-single-{tag}.pt")
        scores = {}
        for protocol, embeddings, labels in (
            ("official_single_image", single["embeddings"], single["labels"]),
            ("multi_view", queries["joint_embeddings"], queries["labels"]),
        ):
            expected = original[split][protocol]
            if (len(labels) != expected["num_queries"]
                    or len(gallery["labels"]) != expected["num_gallery_images"]):
                raise ValueError(f"Cached query/gallery counts differ from original {split}/{protocol}")
            logger.info("Scoring epoch %d %s %s (%d queries)", epoch, split, protocol, len(labels))
            scores[protocol] = evaluator.evaluate_embeddings(
                embeddings, None, labels, gallery["embeddings"], gallery["labels"], **kwargs,
            )
        result["metrics"][split] = scores
    return result


def compare_direct_results(previous, current):
    # Different sampling or preprocessing must not silently become a delta.
    ignored = {"embedding_cache_dir", "retrieval_query_chunk_size", "retrieval_gallery_chunk_size"}
    previous_settings = {k: v for k, v in previous["settings"].items() if k not in ignored}
    current_settings = {k: v for k, v in current["settings"].items() if k not in ignored}
    if previous_settings != current_settings:
        raise ValueError("Checkpoint evaluations use different query/gallery settings")
    rows = []
    for protocol in ("official_single_image", "multi_view"):
        for split in ("test_room", "test_object"):
            before = previous["metrics"][split][protocol]
            after = current["metrics"][split][protocol]
            for key in ("num_queries", "num_gallery_images", "num_gallery_hotels"):
                if before[key] != after[key]:
                    raise ValueError(f"Checkpoint evaluation counts differ for {split}/{protocol}: {key}")
            old_scores, new_scores = before["image"]["joint"], after["image"]["joint"]
            rows.append({
                "query_type": "single_image" if protocol == "official_single_image" else "fused_multi_image",
                "split": split, "num_queries": after["num_queries"],
                "previous_epoch": previous["epoch"], "current_epoch": current["epoch"],
                "previous_recall": old_scores, "current_recall": new_scores,
                "change_percentage_points": {
                    k: round(100 * (new_scores[k] - old_scores[k]), 6) for k in old_scores
                },
            })
    return rows


def check_and_evaluate(args, logger):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "comparison.json"
    status_path = Path(args.current_run) / "monitor.status"
    status = status_path.read_text() if status_path.is_file() else "Monitor status unavailable"
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "current_run": str(args.current_run), "current_monitor_status": status,
        "device": str(args.device),
        "status": "not_finished", "updated_results": {},
    }
    write_json(report_path, report)
    current_metrics = Path(args.current_run) / "retrieval_test_metrics.json"
    if "state=exited:0" not in status.splitlines()[0] or not current_metrics.is_file():
        logger.info("Current evaluation is not finished; saved status to %s", report_path)
        return report

    report["original_results"] = {
        "previous": json.loads((Path(args.previous_run) / "retrieval_test_metrics.json").read_text()),
        "current": json.loads(current_metrics.read_text()),
    }
    report["status"] = "original_results_ready"
    write_json(report_path, report)
    if args.check_only:
        return report
    try:
        if str(args.device).startswith("cuda"):
            busy = subprocess.run([
                "nvidia-smi", "-i", str(args.physical_gpu),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits",
            ], check=True, text=True, capture_output=True).stdout.strip()
            if busy:
                report["status"] = "gpu_busy"
                report["reason"] = "GPU has active compute processes; no additional evaluation started."
                write_json(report_path, report)
                return report
        report["status"] = "evaluating_direct_image_retrieval"
        write_json(report_path, report)
        for label, run_dir in (("previous", args.previous_run), ("current", args.current_run)):
            updated = direct_metrics(
                run_dir, evaluation_settings(run_dir), args.device, logger,
                cache_tag=getattr(args, f"{label}_cache_tag", None),
                checkpoint_epoch=getattr(args, f"{label}_epoch", None),
            )
            report["updated_results"][label] = updated
            write_json(output / f"epoch-{updated['epoch']:03d}-direct.json", updated)
            write_json(report_path, report)
        report["comparison"] = compare_direct_results(
            report["updated_results"]["previous"], report["updated_results"]["current"],
        )
        report["status"] = "complete"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
        logger.info("Direct single-image and fused multi-image comparison:\n%s",
                    json.dumps(report["comparison"], indent=2))
    except Exception as error:
        report["status"] = "failed"
        report["reason"] = str(error)
        write_json(report_path, report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--current-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--not-before", help="ISO timestamp with an explicit UTC offset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    for label in ("previous", "current"):
        parser.add_argument(f"--{label}-cache-tag", help="Cache fingerprint recorded by the completed run")
        parser.add_argument(f"--{label}-epoch", type=int, help="Epoch associated with that cache fingerprint")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("retrieval-evaluation-check")
    if args.not_before:
        target = datetime.fromisoformat(args.not_before)
        if target.tzinfo is None:
            parser.error("--not-before must include a UTC offset")
        logger.info("Waiting until %s for the one-off evaluation check", target.isoformat())
        while (remaining := (target - datetime.now(timezone.utc)).total_seconds()) > 0:
            time.sleep(min(30, remaining))
    report = check_and_evaluate(args, logger)
    logger.info("Check finished with status %s", report["status"])


if __name__ == "__main__":
    main()
