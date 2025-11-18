#!/usr/bin/env python
"""
Unsupervised satellite image segmentation using hierarchical clustering on patch representations.
Visualizes clustering results with t-SNE plots.
Works with sen2venus, mados, and geobench datasets.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.models.vision_transformer as vit
from src.datasets.mados_dataset import MADOSDataset
from src.datasets.geobench_dataset import GeobenchDataset
from src.datasets.sen2venus import Sen2Venus


def load_model(model_name='vit_small', patch_size=14, pretrained_checkpoint=None, img_size=224, device='cuda'):
    """Load pretrained ViT model."""
    model = vit.__dict__[model_name](
        img_size=[img_size],
        patch_size=patch_size,
        in_chans=4  # For satellite imagery (RGBN or similar)
    )
    model.to(device)
    model.eval()

    if pretrained_checkpoint:
        if not os.path.exists(pretrained_checkpoint):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        if 'encoder' in checkpoint:
            state_dict = checkpoint['encoder']
        else:
            state_dict = checkpoint
        # Remove module prefix if it exists (from DDP)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # Remove pos_embed to allow different image sizes
        state_dict = {k: v for k, v in state_dict.items() if 'pos_embed' not in k}
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained checkpoint from {pretrained_checkpoint}")

    return model


def load_dataset_image(dataset_name, data_root, image_idx, dataset_split='val', splits_file=None, use_hr_image=True, img_size=224):
    """Load a single image from the dataset."""
    if dataset_name == 'mados':
        dataset = MADOSDataset(
            data_root=data_root,
            split=dataset_split,
            img_size=img_size
        )
        img_tensor, mask = dataset[image_idx]
        # img_tensor: (C, H, W), mask: (1, H, W)
        img_array = img_tensor.numpy()

    elif dataset_name in ['m-cashew-plantation', 'm-SA-crop-type']:
        dataset = GeobenchDataset(
            split=dataset_split,
            partition='1.00x_train',
            dataset_name=dataset_name,
            img_size=img_size
        )
        img_tensor, mask = dataset[image_idx]
        # img_tensor: (C, H, W), mask: (1, H, W)
        img_array = img_tensor.numpy()

    elif dataset_name == 'sen2venus':
        if splits_file is None:
            raise ValueError("splits_file is required for sen2venus dataset")
        dataset = Sen2Venus(
            hdf5_file=data_root,
            splits_file=splits_file,
            split=dataset_split,
            img_size=img_size,
            use_hr_image=use_hr_image,
            random_crop_resize=False
        )
        img_tensor, _ = dataset[image_idx]
        # img_tensor: (C, H, W), create dummy mask for compatibility
        img_array = img_tensor.numpy()
        mask = np.zeros((1, *img_array.shape[1:]), dtype=np.int32)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return img_array, mask.numpy() if hasattr(mask, 'numpy') else mask


def extract_patch_features(img_tensor, model, patch_size=14, device='cuda'):
    """Extract patch representations from a satellite image tensor."""
    # img_tensor: (C, H, W) numpy array
    C, H, W = img_tensor.shape

    # Resize to multiple of patch size
    h = (H // patch_size) * patch_size
    w = (W // patch_size) * patch_size

    # Crop or resize if needed
    if h != H or w != W:
        img_tensor = img_tensor[:, :h, :w]

    # Convert to tensor and move to device
    img_torch = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)  # (1, C, H, W)

    # Extract features
    with torch.no_grad():
        features = model(img_torch)  # (1, num_patches, dim)

    features = features.squeeze(0)  # (num_patches, dim)

    # Get patch grid dimensions
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    return features.cpu().numpy(), (num_patches_h, num_patches_w), (h, w)


def hierarchical_cluster(features, n_clusters=3, linkage_method='ward'):
    """Perform hierarchical clustering on features."""
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(features)
    return labels


def apply_clustering_to_image(labels, num_patches_h, num_patches_w, patch_size, h, w):
    """Convert patch labels to image segmentation."""
    segmentation = np.zeros((h, w), dtype=np.int32)

    patch_idx = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            if patch_idx < len(labels):
                y_start = i * patch_size
                y_end = min((i + 1) * patch_size, h)
                x_start = j * patch_size
                x_end = min((j + 1) * patch_size, w)
                segmentation[y_start:y_end, x_start:x_end] = labels[patch_idx]
                patch_idx += 1

    return segmentation


def reduce_to_2d(features, method='tsne', n_components=2, perplexity=100, random_state=42):
    """Reduce high-dimensional features to 2D."""
    n_features = len(features)
    perplexity = min(perplexity, (n_features - 1) / 3)

    if method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, max_iter=1000)
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(features)
    return reduced


def visualize_satellite_image(img_array):
    """Create RGB visualization from multi-band satellite image."""
    # img_array: (C, H, W)
    C, H, W = img_array.shape

    rgb = img_array[[2, 1, 0]].transpose(1, 2, 0)

    # Normalize to 0-1
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    rgb = np.clip(rgb, 0, 1)

    return rgb


def visualize_nir(img_array):
    """Create NIR (4th band) visualization from multi-band satellite image."""
    # img_array: (C, H, W), NIR is the 4th band (index 3)
    nir = img_array[3]  # Extract NIR band

    # Normalize to 0-1
    nir_normalized = (nir - nir.min()) / (nir.max() - nir.min() + 1e-6)
    nir_normalized = np.clip(nir_normalized, 0, 1)

    return nir_normalized


def plot_clustering_results(img_array, segmentation, features_2d, labels, n_clusters,
                           num_channels=4, figsize=(20, 5), overlay_alpha=0.5):
    """Create visualization with RGB, NIR, clustering overlay, and t-SNE plot."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Visualize satellite image and NIR
    rgb = visualize_satellite_image(img_array)
    nir = visualize_nir(img_array)

    # Create segmentation overlay with cluster colors
    cmap = plt.cm.get_cmap('tab20', n_clusters)
    segmentation_colored = cmap(segmentation / (n_clusters - 1))

    # Display RGB satellite image
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    # Display NIR band
    axes[1].imshow(nir, cmap='gray')
    axes[1].set_title('NIR Band')
    axes[1].axis('off')

    # Display image with segmentation overlay
    axes[2].imshow(rgb)
    axes[2].imshow(segmentation_colored, alpha=overlay_alpha)
    axes[2].set_title(f'Clustering Overlay (k={n_clusters})')
    axes[2].axis('off')

    # t-SNE scatter plot
    scatter = axes[3].scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap=cmap, s=30, alpha=0.7)
    axes[3].set_title(f't-SNE Visualization (k={n_clusters})')
    axes[3].set_xlabel('t-SNE 1')
    axes[3].set_ylabel('t-SNE 2')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_multi_cluster_visualization(img_array, segmentations_dict, features_2d, labels_dict,
                                       n_clusters_list=[3, 4, 5], num_channels=4, figsize=(24, 12), overlay_alpha=0.5):
    """Create a grid visualization comparing different numbers of clusters with RGB, NIR, overlay, and t-SNE."""
    n_rows = len(n_clusters_list)
    fig, axes = plt.subplots(n_rows, 5, figsize=figsize)

    # Visualize satellite image and NIR once
    rgb = visualize_satellite_image(img_array)
    nir = visualize_nir(img_array)

    for row, n_clusters in enumerate(n_clusters_list):
        # RGB image (same for all rows)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f'RGB' if row == 0 else '')
        axes[row, 0].axis('off')

        # NIR band (same for all rows)
        axes[row, 1].imshow(nir, cmap='gray')
        axes[row, 1].set_title(f'NIR' if row == 0 else '')
        axes[row, 1].axis('off')

        # Segmentation with overlay
        segmentation = segmentations_dict[n_clusters]
        labels = labels_dict[n_clusters]
        cmap = plt.cm.get_cmap('tab20', n_clusters)

        # Create colored segmentation overlay
        segmentation_colored = cmap(segmentation / (n_clusters - 1))

        # Display image with segmentation overlay
        axes[row, 2].imshow(rgb)
        axes[row, 2].imshow(segmentation_colored, alpha=overlay_alpha)
        axes[row, 2].set_title(f'Overlay (k={n_clusters})')
        axes[row, 2].axis('off')

        # t-SNE plot
        scatter = axes[row, 3].scatter(features_2d[:, 0], features_2d[:, 1],
                                       c=labels, cmap=cmap, s=30, alpha=0.7)
        axes[row, 3].set_title(f't-SNE (k={n_clusters})')
        axes[row, 3].set_xlabel('t-SNE 1')
        axes[row, 3].set_ylabel('t-SNE 2')
        axes[row, 3].grid(True, alpha=0.3)

        # Hide the 5th column
        axes[row, 4].axis('off')

    plt.tight_layout()
    return fig


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("Loading ViT model...")
    model = load_model(
        model_name=args.model_name,
        patch_size=args.patch_size,
        pretrained_checkpoint=args.pretrained_checkpoint,
        device=device,
        img_size=args.img_size
    )

    # Load image from dataset
    print(f"Loading image {args.image_idx} from {args.dataset} dataset...")
    use_hr_image = args.sen2venus_image_type == 'hr'
    img_array, mask = load_dataset_image(
        dataset_name=args.dataset,
        data_root=args.data_root,
        image_idx=args.image_idx,
        dataset_split=args.dataset_split,
        splits_file=args.splits_file,
        use_hr_image=use_hr_image,
        img_size=args.img_size
    )
    print(f"Loaded image with shape {img_array.shape}")

    # Extract features
    print("Extracting patch features...")
    features, (num_patches_h, num_patches_w), (h, w) = extract_patch_features(
        img_array, model, args.patch_size, device
    )
    print(f"Extracted {len(features)} patch features of dimension {features.shape[1]}")
    print(f"Image patches: {num_patches_h} x {num_patches_w}")

    # Reduce to 2D for visualization
    print("Computing t-SNE reduction...")
    features_2d = reduce_to_2d(features, method='tsne', perplexity=args.tsne_perplexity)

    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.dataset}_img{args.image_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process multiple cluster numbers
    n_clusters_list = args.n_clusters if isinstance(args.n_clusters, list) else [args.n_clusters]

    segmentations_dict = {}
    labels_dict = {}

    for n_clusters in n_clusters_list:
        print(f"\nPerforming clustering with k={n_clusters}...")
        labels = hierarchical_cluster(features, n_clusters=n_clusters, linkage_method=args.linkage)

        # Apply to image
        segmentation = apply_clustering_to_image(
            labels, num_patches_h, num_patches_w, args.patch_size, h, w
        )

        segmentations_dict[n_clusters] = segmentation
        labels_dict[n_clusters] = labels

        # Save individual plots
        fig = plot_clustering_results(img_array, segmentation, features_2d, labels, n_clusters,
                                     num_channels=img_array.shape[0])
        plot_path = output_dir / f'clustering_k{n_clusters}.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
        plt.close(fig)

    # Create multi-cluster comparison
    if len(n_clusters_list) > 1:
        print("\nCreating multi-cluster comparison visualization...")
        fig = create_multi_cluster_visualization(
            img_array, segmentations_dict, features_2d, labels_dict, n_clusters_list,
            num_channels=img_array.shape[0]
        )
        comparison_path = output_dir / 'clustering_comparison.png'
        fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {comparison_path}")
        plt.close(fig)

    # Save segmentation masks
    for n_clusters, segmentation in segmentations_dict.items():
        seg_path = output_dir / f'segmentation_k{n_clusters}.npy'
        np.save(seg_path, segmentation)
        print(f"Saved segmentation mask to {seg_path}")

    # Save features for further analysis
    features_path = output_dir / 'patch_features.npy'
    np.save(features_path, features)
    features_2d_path = output_dir / 'features_2d.npy'
    np.save(features_2d_path, features_2d)
    print(f"Saved features to {features_path} and {features_2d_path}")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical clustering visualization for satellite images')
    parser.add_argument('--dataset', '-d', required=True,
                        choices=['mados', 'm-cashew-plantation', 'm-SA-crop-type', 'sen2venus'],
                        help='Dataset name')
    parser.add_argument('--data-root', required=True, help='Root directory of the dataset')
    parser.add_argument('--image-idx', type=int, required=True, help='Index of image in dataset')
    parser.add_argument('--dataset-split', default='val', choices=['train', 'val', 'test'],
                        help='Dataset split')
    parser.add_argument('--splits-file', default=None,
                        help='Path to JSON file with dataset splits (required for sen2venus)')
    parser.add_argument('--sen2venus-image-type', choices=['hr', 'lr'], default='hr',
                        help='For sen2venus: use HR (Venus) or LR (Sentinel2) images')
    parser.add_argument('--img-size', type=int, default=224, help='Image size for loading datasets')
    parser.add_argument('--model-name', default='vit_small', help='ViT model name')
    parser.add_argument('--patch-size', type=int, default=14, help='Patch size')
    parser.add_argument('--pretrained-checkpoint', help='Path to pretrained checkpoint')
    parser.add_argument('--n-clusters', type=int, nargs='+', default=[3, 4, 5],
                        help='Number of clusters to test')
    parser.add_argument('--linkage', default='ward', help='Linkage method for clustering')
    parser.add_argument('--tsne-perplexity', type=float, default=30.0, help='t-SNE perplexity')
    parser.add_argument('--output-dir', '-o', default='clustering_results', help='Output directory')

    args = parser.parse_args()
    main(args)