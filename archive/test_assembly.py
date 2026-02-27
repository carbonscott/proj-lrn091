#!/usr/bin/env -S uv run --with h5py --with numpy --with regex --with matplotlib
"""Assembly proof-of-concept: load one raw frame, assemble it, and save a
side-by-side comparison (raw vs assembled) to /tmp/assembly_test.png.

Usage:
    uv run --with h5py --with numpy --with regex --with matplotlib \
        exploration/scripts/test_assembly.py
"""

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Add crystfel_stream_parser to path
# ---------------------------------------------------------------------------
CSP_ROOT = Path.home() / "codes" / "crystfel_stream_parser"
sys.path.insert(0, str(CSP_ROOT))

from crystfel_stream_parser.geom_file import read_geom_file
from crystfel_stream_parser.cheetah_converter import CheetahConverter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "geometry_registry.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
OUTPUT_PATH = Path("/tmp/assembly_test.png")

EXPERIMENT_ID = "cxi101235425"


def main():
    # 1. Load geometry registry
    registry = json.loads(REGISTRY_PATH.read_text())
    geom_path = registry["experiments"][EXPERIMENT_ID]["geom_file"]
    print(f"Geometry file: {geom_path}")

    # 2. Read .geom file and build converter
    geom_dict = read_geom_file(geom_path)
    converter = CheetahConverter(geom_dict)
    print(f"Panels: {len(converter.idx_to_panel)}")

    # 3. Load one raw frame from the manifest
    manifest = json.loads(MANIFEST_PATH.read_text())
    exp_entry = next(
        e for e in manifest["experiments"]
        if e["experiment_id"] == EXPERIMENT_ID
    )
    first_file = exp_entry["files"][0]
    hdf5_path = PROJECT_ROOT / first_file["path"]
    image_key = exp_entry["image_key"]
    print(f"HDF5 file: {hdf5_path}")
    print(f"Image key: {image_key}")

    with h5py.File(str(hdf5_path), "r") as f:
        ds = f[image_key]
        raw_frame = ds[0]  # First frame, shape (H_stacked, W)
    raw_frame = np.array(raw_frame, dtype=np.float32)
    print(f"Raw frame shape: {raw_frame.shape}")

    # 4. Assemble with CheetahConverter
    assembled = converter.convert_to_detector_img(raw_frame)
    # Squeeze out any singleton z-dimension
    if assembled.ndim == 3 and assembled.shape[2] == 1:
        assembled = assembled[:, :, 0]
    print(f"Assembled shape: {assembled.shape}")

    # 5. Save side-by-side visualization
    vmin = np.percentile(raw_frame[raw_frame != 0], 1)
    vmax = np.percentile(raw_frame[raw_frame != 0], 99)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(raw_frame, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    axes[0].set_title(f"Raw stacked ({raw_frame.shape[0]}x{raw_frame.shape[1]})")
    axes[0].set_xlabel("fast-scan (fs)")
    axes[0].set_ylabel("slow-scan (ss)")

    axes[1].imshow(assembled, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title(f"Assembled ({assembled.shape[0]}x{assembled.shape[1]})")
    axes[1].set_xlabel("x (pixels)")
    axes[1].set_ylabel("y (pixels)")

    fig.suptitle(f"{EXPERIMENT_ID} — Assembly POC", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_PATH), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
