import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class MultiImageHybrid(nn.Module):

    def __init__(self, arch, num_classes, n, pretrained_weights=True,
                 image_size=224):
        
        super().__init__()

        self.n = n
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights

        drop_rate = .0 if 'tiny' in arch else .1
        self.model = timm.create_model(
            arch, pretrained=self.pretrained_weights,
            num_classes=self.num_classes, drop_rate=drop_rate,
            img_size=image_size)
        for block in self.model.blocks:
            block.attn.fused_attn = False
        
        self.embed_dim = self.model.embed_dim

        for block in self.model.blocks:
            block.attn.proj_drop = nn.Dropout(p=0.0)

        self.img_embed_matrix = nn.Parameter(torch.zeros(1, n, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.img_embed_matrix)

        nn.init.zeros_(self.model.head.weight)
        nn.init.zeros_(self.model.head.bias)

    def format_multi_image_tokens(self, x, batch_size, tokens_per_image,
                                  num_images=None):
        """Pack image token sequences into one collection token sequence.

        ``tokens_per_image`` is the number of patch tokens before timm adds its
        prefix token.  Keeping this value separate is important: it is also the
        number of tokens that receive each learned image-position embedding.
        ``num_images`` may be smaller than ``self.n`` for variable-view input.
        """

        if num_images is None:
            num_images = self.n
        if not 1 <= num_images <= self.n:
            raise ValueError(
                f'num_images must be in [1, {self.n}], got {num_images}')

        x = einops.rearrange(
            x, '(b n) s c -> b (n s) c', b=batch_size, n=num_images)
        first_img_token_idx = 0
        if self.model.cls_token is not None:
            # Need to remove all excess CLS tokens
            for i in range(1, num_images):
                excess_cls_index = i * tokens_per_image + 1
                x = torch.cat((x[:, :excess_cls_index], x[:, excess_cls_index + 1:]), dim=1)
            first_img_token_idx = 1

        image_embeddings = F.normalize(
            self.img_embed_matrix[:, :num_images], dim=-1)
        x[:, first_img_token_idx:] += torch.repeat_interleave(image_embeddings, tokens_per_image, dim=1)
        return x

    def _validate_inputs(self, x, view_mask):
        if x.ndim != 5:
            raise ValueError(
                'Expected images with shape [batch, views, channels, height, width], '
                f'got {tuple(x.shape)}')

        batch_size, num_images = x.shape[:2]
        if not 1 <= num_images <= self.n:
            raise ValueError(
                f'Input has {num_images} views, but this model supports 1-{self.n}')

        if view_mask is None:
            view_mask = torch.ones(
                (batch_size, num_images), dtype=torch.bool, device=x.device)
        else:
            if tuple(view_mask.shape) != (batch_size, num_images):
                raise ValueError(
                    'view_mask must have shape [batch, views], got '
                    f'{tuple(view_mask.shape)} for images {tuple(x.shape)}')
            view_mask = view_mask.to(device=x.device, dtype=torch.bool)

        num_valid_views = view_mask.sum(dim=1)
        if torch.any(num_valid_views == 0):
            raise ValueError('Every collection must contain at least one valid view')
        return view_mask, num_valid_views

    def _encode_tokens(self, tokens):
        tokens = self.model.blocks(tokens)
        tokens = self.model.norm(tokens)
        return self.model.forward_head(tokens, pre_logits=True)

    def _make_output(self, features, return_logits, return_embeddings):
        output = {}
        if return_logits:
            output['logits'] = self.model.head(features)
        if return_embeddings:
            output['embeddings'] = F.normalize(features, p=2, dim=-1)
        return output

    def _encode_collections(self, tokens_by_view, view_mask,
                            num_valid_views, tokens_per_image):
        """Fuse padded collections without exposing padding to attention.

        Samples are grouped by their valid view count.  Valid views are packed
        in their original order, each group is fused with the corresponding
        prefix of the learned image-position embeddings, and the results are
        restored to the original batch order.
        """

        feature_groups = []
        index_groups = []
        for num_images_tensor in torch.unique(num_valid_views, sorted=True):
            num_images = int(num_images_tensor.item())
            batch_indices = torch.nonzero(
                num_valid_views == num_images_tensor, as_tuple=False).flatten()
            group_tokens = tokens_by_view.index_select(0, batch_indices)
            group_mask = view_mask.index_select(0, batch_indices)
            # Boolean selection preserves row-major ordering, producing
            # [group_batch * num_images, sequence, channels].
            group_tokens = group_tokens[group_mask]
            group_tokens = self.format_multi_image_tokens(
                group_tokens, batch_size=batch_indices.numel(),
                tokens_per_image=tokens_per_image, num_images=num_images)
            feature_groups.append(self._encode_tokens(group_tokens))
            index_groups.append(batch_indices)

        features = torch.cat(feature_groups, dim=0)
        indices = torch.cat(index_groups, dim=0)
        return features.index_select(0, torch.argsort(indices))

    def forward(self, x, view_mask=None, return_embeddings=False,
                return_logits=True, return_collection=True):
        """Run the single-view and multi-view branches.

        Args:
            x: Image tensor with shape ``[B, N, C, H, W]``. ``N`` can range
                from one to the configured maximum ``self.n``.
            view_mask: Optional boolean tensor ``[B, N]``. False entries are
                padding and are excluded from the multi-view attention path.
            return_embeddings: Add L2-normalized pre-classifier embeddings to
                each branch under the ``embeddings`` key.
            return_logits: Return classifier logits (the historical default).
                Retrieval-only callers should set this to False to avoid the
                large hotel-class classifier projection.
            return_collection: Run the fused multi-view branch. Single-image
                gallery extraction can set this to False to avoid a second pass
                through the transformer blocks.

        The single-view branch remains flattened as ``[B * N, ...]`` for
        compatibility. Its flattened ``valid_mask`` identifies padded entries.
        """

        if not return_logits and not return_embeddings:
            raise ValueError('At least one of return_logits/return_embeddings must be True')

        view_mask, num_valid_views = self._validate_inputs(x, view_mask)
        batch_size, num_images = x.shape[:2]

        output_dict = {'single': {'valid_mask': view_mask.reshape(-1)}}
        use_collection = self.n > 1 and return_collection
        if use_collection:
            output_dict['mv_collection'] = {'num_views': num_valid_views}

        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.model.patch_embed(x)

        tokens_per_image = x.shape[1]
        x = self.model._pos_embed(x)

        single_features = self._encode_tokens(x.clone())
        output_dict['single'].update(self._make_output(
            single_features, return_logits, return_embeddings))

        if use_collection:
            tokens_by_view = einops.rearrange(
                x, '(b n) s c -> b n s c', b=batch_size, n=num_images)
            collection_features = self._encode_collections(
                tokens_by_view, view_mask, num_valid_views, tokens_per_image)
            output_dict['mv_collection'].update(self._make_output(
                collection_features, return_logits, return_embeddings))

        return output_dict

    def forward_embeddings(self, x, view_mask=None, return_collection=True):
        """Return normalized embeddings without evaluating the classifier."""

        return self.forward(
            x, view_mask=view_mask, return_embeddings=True,
            return_logits=False, return_collection=return_collection)
