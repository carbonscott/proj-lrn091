#!/usr/bin/env -S uv run --with h5py --with numpy --with regex --with zarr --with ray
"""Batch assembly pipeline: convert raw HDF5 frames to per-run chunked Zarr stores.

Pre-computes the pixel map once per experiment (geometry is fixed), then
assembles each frame via array indexing.  Output follows the peaknet10k
convention: one Zarr chunk file per ~N frames within each run.

Output naming:
    data/assembled/{exp_id}_r{run}.{chunk:04d}.zarr/
        images/                               — (N, H, W) float32, chunks (1,H,W), zstd
        shared_metadata/pixel_maps/
            {exp_id}_r{run}/
                {exp_id}_r{run}               — (2, H, W) int64

Usage:
    uv run --with h5py --with numpy --with regex --with zarr --with ray \
        exploration/scripts/assemble_all.py

    # Process specific experiments:
    uv run ... exploration/scripts/assemble_all.py \
        --experiments mfx100903824

    # Parallel with 8 workers:
    uv run ... exploration/scripts/assemble_all.py \
        --experiments mfx100903824 --num-workers 8

    # Custom chunk size and output directory:
    uv run ... exploration/scripts/assemble_all.py \
        --chunk-size 40 --output-dir /tmp/assembled
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import ray
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
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


# ---------------------------------------------------------------------------
# Run-number extraction and grouping
# ---------------------------------------------------------------------------

def extract_run_number(file_path):
    """Extract a 4-digit zero-padded run number from a manifest file path.

    Tries three strategies in order:
      1. r(\\d{4}) in the filename  (most experiments)
      2. /r(\\d{4})/ in a directory component
      3. -{digits}_ after experiment ID in the filename (mfx100903824 style)

    Returns:
        4-digit zero-padded run string, e.g. "0106".  Falls back to "0000"
        with a warning if no run number can be extracted.
    """
    p = Path(file_path)
    fname = p.name

    # Strategy 1: r(\d{4}) in filename (e.g. cxi101235425-r0106_1.cxi)
    m = re.search(r'r(\d{4})', fname)
    if m:
        return m.group(1)

    # Strategy 2: /r(\d{4})/ in directory path
    for part in p.parts:
        m = re.match(r'r(\d{4})', part)
        if m:
            return m.group(1)

    # Strategy 3: -{digits}_ after experiment ID (e.g. mfx100903824-27_0.cxi)
    m = re.search(r'-(\d+)_', fname)
    if m:
        return m.group(1).zfill(4)

    warnings.warn(f"Could not extract run number from: {file_path}")
    return "0000"


def group_files_by_run(files):
    """Group manifest file entries by extracted run number.

    Args:
        files: List of manifest file dicts (each has "path", "num_frames", ...).

    Returns:
        dict mapping run_number (str) -> list of file dicts, sorted by run.
    """
    groups = defaultdict(list)
    for entry in files:
        run = extract_run_number(entry["path"])
        groups[run].append(entry)
    return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Zarr chunk creation
# ---------------------------------------------------------------------------

def create_zarr_chunk(zarr_path, frames, pixel_map_row, pixel_map_col,
                      assembled_shape, exp_run_key, attrs):
    """Write a single Zarr chunk file in peaknet10k format.

    Args:
        zarr_path: Path for the .zarr store.
        frames: List of (H, W) float32 assembled numpy arrays.
        pixel_map_row: (panel_pixels,) int64 row coordinate map.
        pixel_map_col: (panel_pixels,) int64 col coordinate map.
        assembled_shape: (H, W) of assembled images.
        exp_run_key: e.g. "cxi101235425_r0106" for pixel_map path.
        attrs: Dict of metadata to store as root attributes.
    """
    n = len(frames)
    h, w = assembled_shape

    store = zarr.open(str(zarr_path), mode="w")

    # images/ — (N, H, W) float32, chunked (1, H, W), zstd
    images = store.create_array(
        "images",
        shape=(n, h, w),
        chunks=(1, h, w),
        dtype="float32",
        compressors=[zarr.codecs.ZstdCodec(level=3)],
    )
    for i, frame in enumerate(frames):
        images[i] = frame

    # shared_metadata/pixel_maps/{exp_run_key}/{exp_run_key} — (2, H, W) int64
    pixel_map_stack = np.stack([pixel_map_row, pixel_map_col], axis=0)
    pm_path = f"shared_metadata/pixel_maps/{exp_run_key}/{exp_run_key}"
    store.create_array(pm_path, data=pixel_map_stack.astype(np.int64))

    # Root attributes
    for k, v in attrs.items():
        store.attrs[k] = v


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run(exp_id, run_number, run_files, image_key, converter,
                pixel_map_row, pixel_map_col, assembled_shape,
                output_dir, chunk_size, detector, geom_path,
                project_root=None):
    """Assemble all frames for one run and write chunked Zarr stores.

    Args:
        exp_id: Experiment ID string.
        run_number: 4-digit zero-padded run string.
        run_files: List of manifest file dicts for this run.
        image_key: HDF5 dataset key for images.
        converter: CheetahConverter instance.
        pixel_map_row: Pre-computed row coordinate map.
        pixel_map_col: Pre-computed col coordinate map.
        assembled_shape: (H, W) of the output assembled image.
        output_dir: Path to output directory.
        chunk_size: Number of frames per Zarr chunk.
        detector: Detector name string.
        geom_path: Path to .geom file.
        project_root: Project root path (default: module-level PROJECT_ROOT).

    Returns:
        (frames_assembled, chunks_written) tuple.
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    project_root = Path(project_root)
    output_dir = Path(output_dir)

    exp_run_key = f"{exp_id}_r{run_number}"

    # Skip if this run already has assembled chunks
    existing_chunks = sorted(output_dir.glob(f"{exp_run_key}.*.zarr"))
    if existing_chunks:
        existing_frames = sum(f["num_frames"] for f in run_files)
        print(f"      SKIP: {exp_run_key} already has "
              f"{len(existing_chunks)} chunk(s), ~{existing_frames} frames")
        return existing_frames, len(existing_chunks)

    frame_buffer = []
    chunk_idx = 0
    frames_assembled = 0
    skipped_files = 0

    def flush_chunk():
        nonlocal chunk_idx
        if not frame_buffer:
            return
        zarr_name = f"{exp_run_key}.{chunk_idx:04d}.zarr"
        zarr_path = output_dir / zarr_name
        attrs = {
            "experiment_id": exp_id,
            "detector": detector,
            "geom_file": geom_path,
            "assembled_shape": list(assembled_shape),
            "run_number": run_number,
            "chunk_index": chunk_idx,
            "num_frames": len(frame_buffer),
        }
        create_zarr_chunk(
            zarr_path, frame_buffer, pixel_map_row, pixel_map_col,
            assembled_shape, exp_run_key, attrs,
        )
        chunk_idx += 1
        frame_buffer.clear()

    for file_entry in run_files:
        fpath = project_root / file_entry["path"]
        num_frames = file_entry["num_frames"]

        if not fpath.exists():
            skipped_files += 1
            continue

        try:
            with h5py.File(str(fpath), "r") as f:
                ds = f[image_key]
                indices = file_entry.get("frame_indices")
                if indices is None:
                    indices = range(num_frames)
                for i in indices:
                    if ds.ndim == 3:
                        raw = np.array(ds[i], dtype=np.float32)
                    else:
                        raw = np.array(ds[:], dtype=np.float32)

                    assembled = assemble_frame(
                        raw, converter, pixel_map_row, pixel_map_col,
                        assembled_shape,
                    )
                    frame_buffer.append(assembled)
                    frames_assembled += 1

                    if len(frame_buffer) >= chunk_size:
                        flush_chunk()
        except (OSError, KeyError) as e:
            print(f"      ERROR reading {fpath.name}: {e}")
            skipped_files += 1

    # Flush remaining frames (last chunk may be smaller)
    flush_chunk()

    if skipped_files:
        print(f"      Skipped {skipped_files} files (missing or error)")

    return frames_assembled, chunk_idx


# ---------------------------------------------------------------------------
# Ray remote wrapper
# ---------------------------------------------------------------------------

@ray.remote
def process_run_remote(exp_id, run_number, run_files, image_key,
                       geom_path, pixel_map_row, pixel_map_col,
                       assembled_shape, output_dir, chunk_size,
                       detector, project_root):
    """Ray remote wrapper: reconstructs converter in-worker and assembles a run."""
    import sys as _sys
    from pathlib import Path as _Path

    _csp = str(_Path.home() / "codes" / "crystfel_stream_parser")
    if _csp not in _sys.path:
        _sys.path.insert(0, _csp)

    from crystfel_stream_parser.geom_file import read_geom_file
    from crystfel_stream_parser.cheetah_converter import CheetahConverter

    converter = CheetahConverter(read_geom_file(geom_path))

    return process_run(
        exp_id, run_number, run_files, image_key, converter,
        pixel_map_row, pixel_map_col, assembled_shape,
        output_dir, chunk_size, detector, geom_path,
        project_root=project_root,
    )


# ---------------------------------------------------------------------------
# Per-experiment processing
# ---------------------------------------------------------------------------

def prepare_experiment(experiment_id, manifest, registry):
    """Set up geometry and pixel maps for an experiment.

    Args:
        experiment_id: Experiment ID string.
        manifest: Parsed manifest dict.
        registry: Parsed geometry registry dict.

    Returns:
        Dict with geom_path, detector, image_key, converter,
        pixel_map_row, pixel_map_col, assembled_shape, run_groups.
        Returns None if experiment should be skipped.
    """
    exp_registry = registry["experiments"][experiment_id]
    geom_path = exp_registry["geom_file"]
    detector = exp_registry["detector"]

    exp_manifest = next(
        (e for e in manifest["experiments"]
         if e["experiment_id"] == experiment_id),
        None,
    )
    if exp_manifest is None:
        print(f"  SKIP: {experiment_id} not found in manifest")
        return None

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
                raw_shape = ds_shape[-2:]
            break

    if raw_shape is None:
        print(f"  SKIP: no accessible HDF5 files")
        return None

    print(f"  Raw shape: {raw_shape}")
    pixel_map_row, pixel_map_col, assembled_shape = compute_pixel_maps(
        converter, raw_shape
    )
    print(f"  Assembled shape: {assembled_shape}")

    # Group files by run
    run_groups = group_files_by_run(files)
    total_frames = sum(f["num_frames"] for f in files)
    print(f"  Total frames: {total_frames} across {len(run_groups)} run(s)")

    return {
        "geom_path": geom_path,
        "detector": detector,
        "image_key": image_key,
        "converter": converter,
        "pixel_map_row": pixel_map_row,
        "pixel_map_col": pixel_map_col,
        "assembled_shape": assembled_shape,
        "run_groups": run_groups,
    }


def process_experiment_sequential(experiment_id, manifest, registry,
                                  output_dir, chunk_size):
    """Process all files for a single experiment sequentially (no Ray).

    Args:
        experiment_id: Experiment ID string.
        manifest: Parsed manifest dict.
        registry: Parsed geometry registry dict.
        output_dir: Path to output directory.
        chunk_size: Frames per Zarr chunk.

    Returns:
        Total number of frames assembled.
    """
    prep = prepare_experiment(experiment_id, manifest, registry)
    if prep is None:
        return 0

    total_assembled = 0
    total_chunks = 0
    t0 = time.time()

    for run_number, run_files in prep["run_groups"].items():
        run_frames_expected = sum(f["num_frames"] for f in run_files)
        print(f"    Run {run_number}: {len(run_files)} files, "
              f"{run_frames_expected} frames")

        n_assembled, n_chunks = process_run(
            experiment_id, run_number, run_files, prep["image_key"],
            prep["converter"], prep["pixel_map_row"], prep["pixel_map_col"],
            prep["assembled_shape"], output_dir, chunk_size,
            prep["detector"], prep["geom_path"],
        )
        total_assembled += n_assembled
        total_chunks += n_chunks

        elapsed = time.time() - t0
        fps = total_assembled / max(elapsed, 1e-6)
        print(f"      -> {n_assembled} frames, {n_chunks} chunk(s), "
              f"{fps:.1f} fps cumulative")

    print(f"  Done: {total_assembled} frames in {total_chunks} chunk(s)")
    return total_assembled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Assemble raw HDF5 frames into per-run chunked Zarr stores"
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Specific experiment IDs to process (default: all in registry)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=40,
        help="Number of frames per Zarr chunk (default: 40)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=os.cpu_count(),
        help="Number of Ray workers for parallel run processing "
             "(default: all CPUs; 1 = sequential, no Ray)"
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to curated manifest JSON. When provided, only files/frames "
             "listed in this manifest are assembled. Files with 'frame_indices' "
             "assemble only those indices; others assemble all frames."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = json.loads(REGISTRY_PATH.read_text())
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
    else:
        manifest = json.loads(MANIFEST_PATH.read_text())

    experiment_ids = args.experiments or list(registry["experiments"].keys())

    # Estimate uncompressed size for selected experiments
    bytes_per_pixel = 4  # float32
    total_frames = 0
    for exp_id in experiment_ids:
        exp_manifest = next(
            (e for e in manifest["experiments"]
             if e["experiment_id"] == exp_id),
            None,
        )
        if exp_manifest:
            for f in exp_manifest["files"]:
                total_frames += f.get("num_selected", len(f["frame_indices"])) \
                    if "frame_indices" in f else f["num_frames"]

    # Use a rough assembled size (2000x2000 average) for the estimate
    est_bytes = total_frames * 2000 * 2000 * bytes_per_pixel
    est_tb = est_bytes / (1024 ** 4)

    num_workers = args.num_workers

    print(f"Output dir: {output_dir}")
    print(f"Chunk size: {args.chunk_size} frames")
    print(f"Workers: {num_workers}" + (" (sequential)" if num_workers <= 1 else " (Ray)"))
    print(f"Experiments to process: {len(experiment_ids)}")
    print(f"Estimated uncompressed size: {total_frames} frames, ~{est_tb:.1f} TB")
    print(f"  WARNING: Use --experiments to process selectively if space is limited.")
    print()

    total_assembled = 0
    t_start = time.time()

    if num_workers > 1:
        ray.init(num_cpus=num_workers)
        print(f"Ray initialized with {num_workers} CPUs")
        print()

        # Phase 1: prepare all experiments (fast, sequential)
        all_futures = []
        for exp_id in experiment_ids:
            if exp_id not in registry["experiments"]:
                print(f"SKIP: {exp_id} not in geometry registry")
                continue

            print(f"Preparing {exp_id}...")
            prep = prepare_experiment(exp_id, manifest, registry)
            if prep is None:
                print()
                continue

            pm_row_ref = ray.put(prep["pixel_map_row"])
            pm_col_ref = ray.put(prep["pixel_map_col"])

            for run_number, run_files in prep["run_groups"].items():
                run_frames_expected = sum(f["num_frames"] for f in run_files)
                print(f"    Run {run_number}: {len(run_files)} files, "
                      f"{run_frames_expected} frames (queued)")
                fut = process_run_remote.remote(
                    exp_id, run_number, run_files, prep["image_key"],
                    prep["geom_path"], pm_row_ref, pm_col_ref,
                    prep["assembled_shape"], str(output_dir),
                    args.chunk_size, prep["detector"], str(PROJECT_ROOT),
                )
                all_futures.append((exp_id, run_number, fut))
            print()

        # Phase 2: collect results as they complete
        print(f"All {len(all_futures)} run(s) queued, collecting results...")
        for i, (exp_id, run_number, fut) in enumerate(all_futures):
            n_assembled, n_chunks = ray.get(fut)
            total_assembled += n_assembled
            elapsed = time.time() - t_start
            fps = total_assembled / max(elapsed, 1e-6)
            print(f"  [{i+1}/{len(all_futures)}] {exp_id} r{run_number} "
                  f"-> {n_assembled} frames, {n_chunks} chunk(s), "
                  f"{fps:.1f} fps cumulative")

        ray.shutdown()
    else:
        print()
        # Sequential: process one experiment at a time
        for exp_id in experiment_ids:
            if exp_id not in registry["experiments"]:
                print(f"SKIP: {exp_id} not in geometry registry")
                continue

            print(f"Processing {exp_id}...")
            n = process_experiment_sequential(
                exp_id, manifest, registry, output_dir, args.chunk_size,
            )
            total_assembled += n
            print()

    elapsed = time.time() - t_start
    print(f"All done: {total_assembled} frames in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
