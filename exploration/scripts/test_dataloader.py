"""Smoke test for the data pipeline.

Loads the manifest, creates a PanelPatchDataset, iterates one batch, prints
statistics, and saves a grid of patches to /tmp/patch_grid.png.

Usage:
    uv run --with h5py --with torch --with matplotlib --with numpy \
        exploration/scripts/test_dataloader.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add exploration/ to path so we can import data_pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "exploration"))

from data_pipeline import create_dataloader, PanelPatchDataset, DiffractionTransform
from data_pipeline.manifest import load_manifest

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
OUTPUT_PATH = Path("/tmp/patch_grid.png")


def print_manifest_summary(manifest):
    """Print a quick summary of the manifest."""
    print("Manifest summary:")
    print(f"  Version: {manifest['version']}")
    print(f"  Created: {manifest['created']}")
    summary = manifest["summary"]
    print(f"  Experiments: {summary['num_experiments']}")
    print(f"  Files: {summary['num_files']}")
    print(f"  Total frames: {summary['total_frames']}")
    print()

    for exp in manifest["experiments"]:
        print(f"  {exp['experiment_id']:20s}  detector={exp['detector']:15s}  "
              f"files={exp['num_files']:5d}  frames={exp['total_frames']:8d}  "
              f"panels={exp['num_panels']}x{exp['panel_shape']}")


def test_dataset_basic(manifest_path):
    """Test basic dataset creation and indexing."""
    print("\n--- Dataset creation test ---")

    dataset = PanelPatchDataset(
        manifest_path=manifest_path,
        patch_size=256,
        transform=None,  # No augmentation for this test
    )

    print(f"  Dataset length: {len(dataset):,} (panels across all frames)")
    print(f"  Patch size: 256x256")

    # Grab a few samples to check shapes and values
    print("\n  Sampling 5 items...")
    for i in range(min(5, len(dataset))):
        # Use evenly spaced indices to sample from different experiments
        idx = i * (len(dataset) // 5)
        patch = dataset[idx]
        print(f"    [{idx:8d}] shape={list(patch.shape)}, "
              f"min={patch.min():.3f}, max={patch.max():.3f}, "
              f"mean={patch.mean():.3f}, std={patch.std():.3f}")

    return dataset


def test_dataloader(manifest_path):
    """Test DataLoader with a single batch."""
    print("\n--- DataLoader test ---")

    loader = create_dataloader(
        manifest_path=manifest_path,
        patch_size=256,
        batch_size=16,
        num_workers=0,  # Single-threaded for testing
    )

    # Get one batch
    batch = next(iter(loader))

    print(f"  Batch shape: {list(batch.shape)}")
    print(f"  Batch dtype: {batch.dtype}")
    print(f"  Batch min:   {batch.min():.3f}")
    print(f"  Batch max:   {batch.max():.3f}")
    print(f"  Batch mean:  {batch.mean():.3f}")
    print(f"  Batch std:   {batch.std():.3f}")

    # Check for degenerate patches (all zeros)
    zero_patches = (batch.abs().sum(dim=[1, 2, 3]) == 0).sum().item()
    print(f"  Zero patches: {zero_patches}/{batch.shape[0]}")

    return batch


def save_patch_grid(batch, output_path):
    """Save a grid of patches as a PNG image."""
    print(f"\n--- Saving patch grid to {output_path} ---")

    n = min(batch.shape[0], 16)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            img = batch[i, 0].numpy()  # [H, W]
            # Clip to mean +/- 3 std for display
            mu, sigma = img.mean(), img.std()
            vmin = mu - 3 * sigma if sigma > 0 else img.min()
            vmax = mu + 3 * sigma if sigma > 0 else img.max()
            ax.imshow(img, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
            ax.set_title(f"patch {i}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Sample patches from PanelPatchDataset", fontsize=12)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to: {output_path}")


def main():
    print("Data Pipeline Smoke Test")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Output:   {OUTPUT_PATH}")
    print()

    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        print("Run exploration/scripts/build_manifest.py first.")
        sys.exit(1)

    # 1. Load and summarize manifest
    manifest = load_manifest(MANIFEST_PATH)
    print_manifest_summary(manifest)

    # 2. Test dataset creation
    dataset = test_dataset_basic(MANIFEST_PATH)

    # 3. Test DataLoader
    batch = test_dataloader(MANIFEST_PATH)

    # 4. Save patch grid
    save_patch_grid(batch, OUTPUT_PATH)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
