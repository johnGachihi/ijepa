# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def scale_masks(masks_lr, crop_size_lr, crop_size_hr, patch_size):
    """
    Scale masks from LR to HR patch grid.
    One LR patch index maps to scale_factor^2 HR patch indices.

    :param masks_lr: list of 1D tensors containing patch indices for LR image
    :param crop_size_lr: LR image size in pixels
    :param crop_size_hr: HR image size in pixels
    :param patch_size: patch size in pixels (same for both LR and HR)
    :return: list of 1D tensors with scaled patch indices for HR image

    Example: If crop_size_lr=224, crop_size_hr=448, patch_size=16:
        - LR: 14×14 = 196 patches
        - HR: 28×28 = 784 patches
        - Scale factor: 2
        - One LR patch index → 4 HR patch indices
    """
    n_patches_lr = crop_size_lr // patch_size
    n_patches_hr = crop_size_hr // patch_size
    scale_factor = crop_size_hr // crop_size_lr

    if isinstance(masks_lr, list):
        masks_lr = masks_lr[0]

    scaled_masks = []
    for mask_lr in masks_lr:
        hr_indices = []
        for idx_lr in mask_lr:
            # Convert 1D LR index to 2D coordinates
            idx_lr_val = idx_lr.item() if isinstance(idx_lr, torch.Tensor) else int(idx_lr)
            row_lr = idx_lr_val // n_patches_lr
            col_lr = idx_lr_val % n_patches_lr

            # Each LR patch maps to scale_factor x scale_factor HR patches
            for dr in range(scale_factor):
                for dc in range(scale_factor):
                    row_hr = row_lr * scale_factor + dr
                    col_hr = col_lr * scale_factor + dc
                    idx_hr = row_hr * n_patches_hr + col_hr
                    hr_indices.append(idx_hr)

        scaled_masks.append(torch.tensor(hr_indices, device=mask_lr.device, dtype=torch.long))

    return torch.stack(scaled_masks, dim=0)
