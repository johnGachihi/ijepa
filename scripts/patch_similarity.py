#!/usr/bin/env python
"""
Patch Cosine Similarity Visualization for ijepa Vision Transformer

This script loads a pretrained ijepa ViT model from a checkpoint,
loads images from various satellite datasets (Sen2Venus, MADOS, Geobench),
computes the cosine similarity between a selected patch and all other patches,
and visualizes the result as a heatmap overlay.

Usage:
    python patch_similarity.py --model_name vit_small --checkpoint path/to/checkpoint.pth \
        --dataset sen2venus --image_index 0 --dataset_root /path/to/sen2venus --patch_idx 128

Requirements:
    - torch
    - torchvision
    - numpy
    - matplotlib
    - PIL
    - scikit-learn
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.models.vision_transformer as vit
from src.datasets.sen2venus import Sen2Venus
from src.datasets.mados_dataset import MADOSDataset
from src.datasets.geobench_dataset import GeobenchDataset


def load_model(model_name='vit_small', patch_size=14, checkpoint_path=None,
               img_size=224, in_chans=4, device='cuda'):
    """
    Load a pretrained ijepa ViT model from a checkpoint.

    Args:
        model_name: Name of the ViT model (vit_small, vit_base, vit_large, vit_giant)
        patch_size: Patch size for the ViT model
        checkpoint_path: Path to the pretrained checkpoint
        img_size: Input image size
        in_chans: Number of input channels
        device: Device to load the model on ('cuda' or 'cpu')

    Returns:
        model: Loaded model in evaluation mode
        embed_dim: Feature dimension of the model
    """
    print(f"Loading {model_name} model with patch_size={patch_size}, img_size={img_size}, in_chans={in_chans}...")

    # Build model
    model = vit.__dict__[model_name](
        img_size=[img_size],
        patch_size=patch_size,
        in_chans=in_chans
    )

    # Get embed_dim from the model
    embed_dim = model.embed_dim
    print(f"Model embed_dim: {embed_dim}")

    model.to(device)
    model.eval()

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove module prefix if it exists (from DDP)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Skip pos_embed to allow loading checkpoints with different image sizes
        state_dict = {k: v for k, v in state_dict.items() if 'pos_embed' not in k}

        # Load the state dict
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded: {msg}")
    else:
        print("No checkpoint provided, using randomly initialized weights")

    return model, embed_dim


def load_and_preprocess_image(image_index, dataset_root, dataset_name='sen2venus',
                              img_size=224, use_hr=True, split='train', in_chans=4):
    """
    Load and preprocess an image from a satellite dataset.

    Args:
        image_index: Index of the image in the dataset
        dataset_root: Root directory of the dataset
        dataset_name: Name of the dataset ('sen2venus', 'mados', or 'geobench')
        img_size: Size to resize the image (default: 224)
        use_hr: If True and dataset is sen2venus, use high-resolution (Venus) image
        split: Dataset split to use (default: 'train')
        in_chans: Number of input channels to use

    Returns:
        img_tensor: Preprocessed image tensor
        original_img: Original image as PIL image for visualization
    """
    print(f"Loading {dataset_name} dataset from {dataset_root}...")

    if dataset_name.lower() == 'sen2venus':
        dataset = Sen2Venus(
            hdf5_file=Path(dataset_root) / 'sen2venus.hdf5',
            splits_file=Path(dataset_root) / 'splits_v1.json',
            split=split,
            img_size=img_size,
            use_hr_image=use_hr,
            load_both_images=True,
            random_crop_resize=False,
            normalise=False)
        venus_img, sentinel2_img, _ = dataset[image_index]

        # Select which image to use
        img_tensor = venus_img if use_hr else sentinel2_img
        print(f"Using {'high-resolution (Venus)' if use_hr else 'low-resolution (Sentinel-2)'} image")

    elif dataset_name.lower() == 'mados':
        dataset = MADOSDataset(root=dataset_root, split=split)
        img_tensor = dataset[image_index]['image']

    elif dataset_name.lower() == 'geobench':
        dataset = GeobenchDataset(root=dataset_root, split=split)
        img_tensor = dataset[image_index]['image']

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Convert to float32 if needed
    if img_tensor.dtype != torch.float32:
        img_tensor = img_tensor.float()

    # Use only the first in_chans channels
    if img_tensor.shape[0] > in_chans:
        img_tensor = img_tensor[:in_chans]
    elif img_tensor.shape[0] < in_chans:
        # Pad with zeros if not enough channels
        padding = torch.zeros(in_chans - img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
        img_tensor = torch.cat([img_tensor, padding], dim=0)

    # Resize if needed (C, H, W)
    if img_tensor.shape[1] != img_size or img_tensor.shape[2] != img_size:
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    # Create a visualization image (use first 3 channels for RGB visualization)
    if img_tensor.shape[0] >= 3:
        # Try to use RGB channels
        vis_img = img_tensor[:3][[2, 1, 0]] if img_tensor.shape[0] >= 3 else img_tensor[:3]
    else:
        # Use available channels
        vis_img = img_tensor[:min(3, img_tensor.shape[0])]
        if vis_img.shape[0] < 3:
            # Pad with zeros to get 3 channels
            padding = torch.zeros(3 - vis_img.shape[0], vis_img.shape[1], vis_img.shape[2])
            vis_img = torch.cat([vis_img, padding], dim=0)

    vis_img = vis_img.permute(1, 2, 0).numpy()

    # Normalize to 0-1 for visualization
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-8)
    vis_img = (vis_img * 255).astype(np.uint8)
    original_img = Image.fromarray(vis_img)

    # Normalize the full tensor for model input (zero-center)
    img_tensor = img_tensor - img_tensor.mean(dim=[1, 2], keepdim=True)
    img_tensor = img_tensor / (img_tensor.std(dim=[1, 2], keepdim=True) + 1e-8)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, original_img


def extract_patch_features(model, img_tensor, device='cuda'):
    """
    Extract patch features from ViT model.

    Args:
        model: ViT model
        img_tensor: Preprocessed image tensor
        device: Device to run inference on

    Returns:
        patch_features: Tensor of shape (num_patches, feature_dim)
        num_patches_h: Number of patches along height
        num_patches_w: Number of patches along width
    """
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        # Extract features using forward method
        # VisionTransformer.forward() returns a tensor of shape (batch, num_patches, feature_dim)
        # Note: No class token is used in this model
        patch_features = model.forward(img_tensor)

    # Remove batch dimension
    patch_features = patch_features.squeeze(0)  # Shape: (num_patches, feature_dim)

    # Calculate grid dimensions
    num_patches = patch_features.shape[0]
    num_patches_side = int(np.sqrt(num_patches))
    num_patches_h = num_patches_side
    num_patches_w = num_patches_side

    print(f"Extracted {num_patches} patches ({num_patches_h}x{num_patches_w})")

    return patch_features, num_patches_h, num_patches_w


def compute_cosine_similarity_map(patch_features, reference_patch_idx):
    """
    Compute cosine similarity between a reference patch and all other patches.

    Args:
        patch_features: Tensor of shape (num_patches, feature_dim)
        reference_patch_idx: Index of the reference patch

    Returns:
        similarity_map: Cosine similarity values for all patches (num_patches,)
    """
    # Get the reference patch feature
    reference_feature = patch_features[reference_patch_idx].unsqueeze(0)  # Shape: (1, feature_dim)

    # Compute cosine similarity with all patches
    similarity_map = F.cosine_similarity(reference_feature, patch_features, dim=1)

    return similarity_map.cpu().numpy()


def visualize_similarity_map(original_img, similarity_map, num_patches_h, num_patches_w,
                             reference_patch_idx, img_size=224, alpha=0.6, save_path=None):
    """
    Visualize the cosine similarity map as a heatmap overlay on the original image.

    Args:
        original_img: Original PIL image
        similarity_map: Similarity values (num_patches,)
        num_patches_h: Number of patches along height
        num_patches_w: Number of patches along width
        reference_patch_idx: Index of the reference patch
        img_size: Size of the processed image
        alpha: Transparency of the heatmap overlay
        save_path: Path to save the visualization (optional)
    """
    # Reshape similarity map to 2D grid
    similarity_grid = similarity_map.reshape(num_patches_h, num_patches_w)

    # Resize original image to match processed size
    original_img_resized = original_img.resize((img_size, img_size))

    # Calculate reference patch position
    ref_row = reference_patch_idx // num_patches_w
    ref_col = reference_patch_idx % num_patches_w

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Original image with reference patch marker
    axes[0].imshow(original_img_resized)
    patch_size = img_size // num_patches_h
    rect_x = ref_col * patch_size
    rect_y = ref_row * patch_size
    rect = plt.Rectangle((rect_x, rect_y), patch_size, patch_size,
                         fill=False, edgecolor='red', linewidth=3)
    axes[0].add_patch(rect)
    axes[0].set_title(f'Original Image\nReference Patch: [{ref_row}, {ref_col}]', fontsize=14)
    axes[0].axis('off')

    # Plot 2: Similarity heatmap only
    im = axes[1].imshow(similarity_grid, cmap='viridis', vmin=-1, vmax=1)
    axes[1].set_title('Cosine Similarity Heatmap', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot 3: Overlay heatmap on original image
    axes[2].imshow(original_img_resized)
    # Upsample similarity grid to image size for smooth overlay
    similarity_upsampled = np.kron(similarity_grid, np.ones((patch_size, patch_size)))
    # Ensure correct size
    similarity_upsampled = similarity_upsampled[:img_size, :img_size]

    im_overlay = axes[2].imshow(similarity_upsampled, cmap='hot', alpha=alpha, vmin=0, vmax=1)
    axes[2].set_title(f'Similarity Overlay (alpha={alpha})', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im_overlay, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def interactive_patch_selection(original_img, model, img_tensor, device, img_size=224, alpha=0.6):
    """
    Interactive mode: click on the image to select a patch and compute similarity.

    Args:
        original_img: Original PIL image
        model: ViT model
        img_tensor: Preprocessed image tensor
        device: Device to run inference on
        img_size: Size of the processed image
        alpha: Transparency of the heatmap overlay
    """
    # Extract patch features once
    patch_features, num_patches_h, num_patches_w = extract_patch_features(model, img_tensor, device)
    patch_size = img_size // num_patches_h

    # Resize original image
    original_img_resized = original_img.resize((img_size, img_size))

    # Create interactive figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original_img_resized)
    ax.set_title('Click on a patch to compute similarity\n(Close window to exit)', fontsize=14)
    ax.axis('off')

    # Overlay grid
    for i in range(num_patches_h + 1):
        ax.axhline(y=i * patch_size, color='white', alpha=0.3, linewidth=0.5)
    for j in range(num_patches_w + 1):
        ax.axvline(x=j * patch_size, color='white', alpha=0.3, linewidth=0.5)

    def onclick(event):
        if event.inaxes != ax:
            return

        # Get clicked position
        x, y = int(event.xdata), int(event.ydata)

        # Convert to patch index
        patch_col = min(x // patch_size, num_patches_w - 1)
        patch_row = min(y // patch_size, num_patches_h - 1)
        patch_idx = patch_row * num_patches_w + patch_col

        print(f"\nSelected patch: [{patch_row}, {patch_col}] (index: {patch_idx})")

        # Compute similarity
        similarity_map = compute_cosine_similarity_map(patch_features, patch_idx)

        # Visualize
        visualize_similarity_map(original_img, similarity_map, num_patches_h, num_patches_w,
                                 patch_idx, img_size, alpha)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compute and visualize patch cosine similarity for ijepa ViT')
    parser.add_argument('--model_name', type=str, default='vit_small',
                        choices=['vit_small', 'vit_base', 'vit_large', 'vit_giant'],
                        help='Name of the ViT model (default: vit_small)')
    parser.add_argument('--patch_size', type=int, default=14,
                        help='Patch size for the ViT model (default: 14)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint (.pth file). If not provided, uses randomly initialized weights.')
    parser.add_argument('--dataset', type=str, default='sen2venus',
                        choices=['sen2venus', 'mados', 'geobench'],
                        help='Dataset to use (default: sen2venus)')
    parser.add_argument('--image_index', type=int, required=True,
                        help='Index of the image in the dataset')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--use_hr', action='store_true', default=True,
                        help='Use high-resolution (Venus) image for sen2venus. Use --no-use_hr for low-resolution (Sentinel-2)')
    parser.add_argument('--no-use_hr', dest='use_hr', action='store_false',
                        help='Use low-resolution (Sentinel-2) image instead of high-resolution')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use (default: train)')
    parser.add_argument('--patch_idx', type=int, default=None,
                        help='Index of reference patch (0 to num_patches-1). If not provided, enters interactive mode.')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for processing (default: 224)')
    parser.add_argument('--in_chans', type=int, default=4,
                        help='Number of input channels (default: 4)')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Transparency of heatmap overlay (0-1, default: 0.6)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the visualization (optional)')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Load model
    model, embed_dim = load_model(
        model_name=args.model_name,
        patch_size=args.patch_size,
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        in_chans=args.in_chans,
        device=args.device
    )

    # Load and preprocess image
    print(f"Loading image index {args.image_index} from {args.dataset_root}...")
    img_tensor, original_img = load_and_preprocess_image(
        args.image_index, args.dataset_root, args.dataset,
        args.img_size, args.use_hr, args.split, args.in_chans
    )

    # Extract patch features
    print("Extracting patch features...")
    patch_features, num_patches_h, num_patches_w = extract_patch_features(model, img_tensor, args.device)
    num_patches = patch_features.shape[0]

    # Interactive or single patch mode
    if args.patch_idx is None:
        print("\nEntering interactive mode...")
        print("Click on any patch in the image to compute its similarity with all other patches.")
        interactive_patch_selection(original_img, model, img_tensor, args.device, args.img_size, args.alpha)
    else:
        # Validate patch index
        if args.patch_idx < 0 or args.patch_idx >= num_patches:
            print(f"Error: patch_idx must be between 0 and {num_patches - 1}")
            sys.exit(1)

        # Compute similarity map
        print(f"Computing cosine similarity for patch {args.patch_idx}...")
        similarity_map = compute_cosine_similarity_map(patch_features, args.patch_idx)

        # Visualize
        print("Generating visualization...")
        visualize_similarity_map(original_img, similarity_map, num_patches_h, num_patches_w,
                                 args.patch_idx, args.img_size, args.alpha, args.save_path)


if __name__ == '__main__':
    main()
