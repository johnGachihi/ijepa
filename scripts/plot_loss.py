#!/usr/bin/env python3
"""
Script to plot loss curves from I-JEPA training logs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def smooth_curve(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_loss_curve(csv_path, output_path=None, smooth=True, smooth_weight=0.9):
    """
    Plot loss curve from training CSV log.

    Args:
        csv_path: Path to the CSV log file
        output_path: Path to save the plot (if None, displays plot)
        smooth: Whether to add smoothed curve
        smooth_weight: Weight for exponential moving average (0-1)
    """
    # Read CSV
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} training iterations")
    print(f"Epochs: {df['epoch'].min()} to {df['epoch'].max()}")
    print(f"Loss range: {df['loss'].min():.4f} to {df['loss'].max():.4f}")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Loss over iterations
    ax1.plot(df.index, df['loss'], alpha=0.3, color='blue', label='Raw loss')
    if smooth:
        smoothed_loss = smooth_curve(df['loss'].values, weight=smooth_weight)
        ax1.plot(df.index, smoothed_loss, color='blue', linewidth=2,
                label=f'Smoothed loss (EMA={smooth_weight})')

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss over Iterations', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss per epoch (averaged)
    epoch_stats = df.groupby('epoch')['loss'].agg(['mean', 'std', 'min', 'max'])

    ax2.plot(epoch_stats.index, epoch_stats['mean'],
            color='darkblue', linewidth=2, marker='o', label='Mean loss per epoch')
    ax2.fill_between(epoch_stats.index,
                     epoch_stats['mean'] - epoch_stats['std'],
                     epoch_stats['mean'] + epoch_stats['std'],
                     alpha=0.2, color='blue', label='±1 std')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss per Epoch', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    return df, epoch_stats


def main():
    parser = argparse.ArgumentParser(
        description='Plot loss curves from I-JEPA training logs'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        nargs='?',
        default='/home/user/ijepa/logs/sen2venus_vits14.224-bs.128-ep.100_hr/_jepa_sen2venus_r0.csv',
        help='Path to the CSV log file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for the plot (e.g., loss_curve.png). If not specified, displays plot.'
    )
    parser.add_argument(
        '--no-smooth',
        action='store_true',
        help='Disable smoothed curve'
    )
    parser.add_argument(
        '--smooth-weight',
        type=float,
        default=0.9,
        help='Smoothing weight for exponential moving average (default: 0.9)'
    )

    args = parser.parse_args()

    # Validate input file
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Set default output path if not specified
    output_path = args.output
    if output_path is None:
        # Create output in the same directory as the CSV
        output_path = csv_path.parent / f"{csv_path.stem}_loss_curve.png"

    # Plot
    plot_loss_curve(
        csv_path,
        output_path=output_path,
        smooth=not args.no_smooth,
        smooth_weight=args.smooth_weight
    )


if __name__ == '__main__':
    main()
