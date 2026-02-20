#!/usr/bin/env -S uv run --with h5py --with numpy --with regex --with zarr
"""Batch assembly pipeline: convert raw HDF5 frames to assembled Zarr stores.

Pre-computes the pixel map once per experiment (geometry is fixed), then
assembles each frame via array indexing.

Output per experiment:
    data/assembled/{experiment_id}.zarr/
        images/   — (N, H, W) float32, chunked (1, H, W), zstd compressed
        pixel_map/ — (2, H, W) int64  (row/col coordinate maps)

Usage:
    uv run --with h5py --with numpy --with regex --with zarr \
        exploration/scripts/assemble_all.py

    # Process specific experiments:
    uv run ... exploration/scripts/assemble_all.py \
        --experiments cxi101235425 cxil1005322

    # Custom output directory:
    uv run ... exploration/scripts/assemble_all.py --output-dir /tmp/assembled
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import zarr

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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "geometry_registry.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "assembled"


def compute_pixel_maps(converter, raw_shape):
    """Pre-compute the pixel coordinate maps for a given geometry.

    Args:
        converter: CheetahConverter instance.
        raw_shape: (H_stacked, W) of raw cheetah images.

    Returns:
        pixel_map_row: int array, row indices into assembled image.
        pixel_map_col: int array, col indices into assembled image.
        assembled_shape: (H_assembled, W_assembled) tuple.
    """
    # Create a dummy frame to compute pixel maps
    dummy = np.zeros(raw_shape, dtype=np.float32)
    psana_img = converter.convert_to_psana_img(dummy)
    pixel_map_x, pixel_map_y, pixel_map_z = converter.calculate_pixel_map(psana_img)

    pixel_map_row = np.round(pixel_map_x).astype(np.int64)
    pixel_map_col = np.round(pixel_map_y).astype(np.int64)

    assembled_h = int(pixel_map_row.max() - pixel_map_row.min() + 1)
    assembled_w = int(pixel_map_col.max() - pixel_map_col.min() + 1)

    return pixel_map_row, pixel_map_col, (assembled_h, assembled_w)


def assemble_frame(raw_frame, converter, pixel_map_row, pixel_map_col,
                   assembled_shape):
    """Assemble a single raw frame using pre-computed pixel maps.

    Args:
        raw_frame: (H_stacked, W) float32 raw cheetah image.
        converter: CheetahConverter instance.
        pixel_map_row: Pre-computed row coordinate map.
        pixel_map_col: Pre-computed col coordinate map.
        assembled_shape: (H, W) of the output assembled image.

    Returns:
        (H, W) float32 assembled image.
    """
    psana_img = converter.convert_to_psana_img(raw_frame)
    assembled = np.zeros(assembled_shape, dtype=np.float32)
    assembled[pixel_map_row, pixel_map_col] = psana_img
    return assembled


def process_experiment(experiment_id, manifest, registry, output_dir):
    """Process all files for a single experiment into a Zarr store.

    Args:
        experiment_id: Experiment ID string.
        manifest: Parsed manifest dict.
        registry: Parsed geometry registry dict.
        output_dir: Path to output directory.

    Returns:
        Number of frames assembled.
    """
    # Look up geometry
    exp_registry = registry["experiments"][experiment_id]
    geom_path = exp_registry["geom_file"]

    # Look up manifest entry
    exp_manifest = next(
        (e for e in manifest["experiments"]
         if e["experiment_id"] == experiment_id),
        None,
    )
    if exp_manifest is None:
        print(f"  SKIP: {experiment_id} not found in manifest")
        return 0

    image_key = exp_manifest["image_key"]
    files = exp_manifest["files"]

    print(f"  Geometry: {Path(geom_path).name}")
    print(f"  Files: {len(files)}, Image key: {image_key}")

    # Build converter and pre-compute pixel maps
    geom_dict = read_geom_file(geom_path)
    converter = CheetahConverter(geom_dict)

    # Determine raw shape from first available file
    raw_shape = None
    for file_entry in files:
        fpath = PROJECT_ROOT / file_entry["path"]
        if fpath.exists():
            with h5py.File(str(fpath), "r") as f:
                ds_shape = f[image_key].shape
                raw_shape = ds_shape[-2:]  # (H_stacked, W)
            break

    if raw_shape is None:
        print(f"  SKIP: no accessible HDF5 files")
        return 0

    print(f"  Raw shape: {raw_shape}")
    pixel_map_row, pixel_map_col, assembled_shape = compute_pixel_maps(
        converter, raw_shape
    )
    print(f"  Assembled shape: {assembled_shape}")

    # Count total frames
    total_frames = sum(f["num_frames"] for f in files)
    print(f"  Total frames: {total_frames}")

    # Create Zarr store
    zarr_path = output_dir / f"{experiment_id}.zarr"
    store = zarr.open(str(zarr_path), mode="w")

    # Create images array
    images = store.create_array(
        "images",
        shape=(total_frames, assembled_shape[0], assembled_shape[1]),
        chunks=(1, assembled_shape[0], assembled_shape[1]),
        dtype="float32",
        compressors=[zarr.codecs.ZstdCodec(level=3)],
    )

    # Save pixel maps (row and col coordinate maps)
    store.create_array(
        "pixel_map_row",
        data=pixel_map_row.astype(np.int64),
    )
    store.create_array(
        "pixel_map_col",
        data=pixel_map_col.astype(np.int64),
    )

    # Store metadata
    store.attrs["experiment_id"] = experiment_id
    store.attrs["detector"] = exp_registry["detector"]
    store.attrs["geom_file"] = geom_path
    store.attrs["raw_shape"] = list(raw_shape)
    store.attrs["assembled_shape"] = list(assembled_shape)
    store.attrs["total_frames"] = total_frames

    # Process frames
    frame_offset = 0
    skipped_files = 0
    t0 = time.time()

    for file_idx, file_entry in enumerate(files):
        fpath = PROJECT_ROOT / file_entry["path"]
        num_frames = file_entry["num_frames"]

        if not fpath.exists():
            skipped_files += 1
            # Fill with zeros for missing files (maintain indexing)
            frame_offset += num_frames
            continue

        try:
            with h5py.File(str(fpath), "r") as f:
                ds = f[image_key]
                for i in range(num_frames):
                    if ds.ndim == 3:
                        raw = np.array(ds[i], dtype=np.float32)
                    else:
                        raw = np.array(ds[:], dtype=np.float32)

                    assembled = assemble_frame(
                        raw, converter, pixel_map_row, pixel_map_col,
                        assembled_shape,
                    )
                    images[frame_offset + i] = assembled
        except (OSError, KeyError) as e:
            print(f"    ERROR reading {fpath.name}: {e}")
            skipped_files += 1

        frame_offset += num_frames

        # Progress
        if (file_idx + 1) % 50 == 0 or file_idx == len(files) - 1:
            elapsed = time.time() - t0
            fps = frame_offset / max(elapsed, 1e-6)
            print(f"    [{file_idx + 1}/{len(files)}] "
                  f"{frame_offset}/{total_frames} frames, "
                  f"{fps:.1f} fps, "
                  f"{elapsed:.0f}s elapsed")

    if skipped_files:
        print(f"  Skipped {skipped_files} files (missing or error)")

    print(f"  Done: {zarr_path}")
    return frame_offset


def main():
    parser = argparse.ArgumentParser(
        description="Assemble raw HDF5 frames into Zarr stores"
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Specific experiment IDs to process (default: all in registry)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = json.loads(REGISTRY_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())

    experiment_ids = args.experiments or list(registry["experiments"].keys())

    print(f"Output dir: {output_dir}")
    print(f"Experiments to process: {len(experiment_ids)}")
    print()

    total_frames = 0
    t_start = time.time()

    for exp_id in experiment_ids:
        if exp_id not in registry["experiments"]:
            print(f"SKIP: {exp_id} not in geometry registry")
            continue

        print(f"Processing {exp_id}...")
        n = process_experiment(exp_id, manifest, registry, output_dir)
        total_frames += n
        print()

    elapsed = time.time() - t_start
    print(f"All done: {total_frames} frames in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
