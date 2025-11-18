# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LinearSegmentationHead(nn.Module):
    """Linear segmentation head using bilinear interpolation for upsampling."""

    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        use_batchnorm=False,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.d_encoder = d_encoder

        # Linear layer for classification
        self.head = nn.Linear(d_encoder, n_cls)

        # Optional batch normalization
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(d_encoder)

        # Initialize weights
        nn.init.normal_(self.head.weight, mean=0, std=0.01)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x, im_size):
        """
        Args:
            x: tensor of shape [B, N, D] where N is number of patches, D is embedding dim
            im_size: tuple (H, W) target output size
        Returns:
            masks: tensor of shape [B, n_cls, H, W]
        """
        H, W = im_size
        B, N, D = x.shape

        # Compute grid size from number of patches
        GS = int(N ** 0.5)

        # Apply batch normalization if enabled
        if self.use_batchnorm:
            # Reshape for batchnorm: [B, N, D] -> [B*N, D]
            x_flat = x.reshape(B * N, D)
            x_flat = self.batchnorm(x_flat)
            x = x_flat.reshape(B, N, D)

        # Apply linear classification head
        x = self.head(x)  # [B, N, n_cls]

        # Reshape to spatial grid: [B, N, n_cls] -> [B, n_cls, GS, GS]
        x = rearrange(x, "b (h w) c -> b c h w", h=GS, w=GS)

        # Upsample to target size using bilinear interpolation
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return x


def linear_segmentation_head(encoder, n_cls, use_batchnorm=False, **kwargs):
    """
    Factory function to create linear segmentation head.

    Args:
        encoder: encoder model with embed_dim and patch_embed attributes
        n_cls: number of segmentation classes
        use_batchnorm: whether to use batch normalization (default: False)
    """
    d_encoder = encoder.embed_dim
    patch_size = encoder.patch_embed.patch_size

    model = LinearSegmentationHead(
        n_cls=n_cls,
        patch_size=patch_size,
        d_encoder=d_encoder,
        use_batchnorm=use_batchnorm
    )
    return model
