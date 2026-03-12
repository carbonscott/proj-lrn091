"""
Generate Parquet manifests for the SFX data broker.

Scans assembled Zarr chunks and peaknet10k Zarr chunks, computes per-frame
statistics, and produces entity/artifact manifest Parquet files + dataset YAML
configs for each run.

Usage:
    # All runs (needs a compute node for ~440K frames):
    uv run --with zarr --with numpy --with pandas --with pyarrow --with ray \
        --with 'ruamel.yaml' python scripts/generate_manifests.py

    # Specific runs (for testing):
    uv run --with zarr --with numpy --with pandas --with pyarrow --with ray \
        --with 'ruamel.yaml' python scripts/generate_manifests.py \
        --runs cxilw5019_r0017 mfx100903824_r0027

    # Sequential (no Ray, for debugging):
    uv run --with zarr --with numpy --with pandas --with pyarrow \
        --with 'ruamel.yaml' python scripts/generate_manifests.py \
        --runs mfx100903824_r0027 --num-workers 1
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
ASSEMBLED_DIR = PROJECT_DIR / "data" / "assembled"
PEAKNET_DIR = PROJECT_DIR / "data" / "peaknet10k"
OUTPUT_DIR = PROJECT_DIR / "data" / "broker"
MANIFESTS_DIR = OUTPUT_DIR / "manifests"
DATASETS_DIR = OUTPUT_DIR / "datasets"

# ---------------------------------------------------------------------------
# Experiment metadata lookup (not stored in Zarr)
# ---------------------------------------------------------------------------
EXPERIMENT_META = {
    "cxi101235425": {"instrument": "CXI", "pi": "van Thor",  "sample": "Frenkel-CT exciton SFX",  "proposal": "1012354"},
    "cxil1005322":  {"instrument": "CXI", "pi": "Pollack",   "sample": "Nucleic acid dynamics",   "proposal": "L-10053"},
    "cxil1015922":  {"instrument": "CXI", "pi": "Standfuss", "sample": "Beta-2 receptor",         "proposal": "L-10159"},
    "cxilw5019":    {"instrument": "CXI", "pi": "Cherezov",  "sample": "GPCR SFX in LCP",         "proposal": "LW50"},
    "mfx100903824": {"instrument": "MFX", "pi": "Schleiss",  "sample": "Cryptochrome",            "proposal": "1009038"},
    "mfx101211025": {"instrument": "MFX", "pi": "Lane",      "sample": "CPD photolyase",          "proposal": "1012110"},
    "mfxp22421":    {"instrument": "MFX", "pi": "Cherezov",  "sample": "Fixed-target GPCR SFX",   "proposal": "P224"},
    "mfxl1025422":  {"instrument": "MFX", "pi": "Unknown",   "sample": "PeakNet labeled",         "proposal": "L-10254"},
    "mfxl1027522":  {"instrument": "MFX", "pi": "Unknown",   "sample": "PeakNet labeled",         "proposal": "L-10275"},
    "mfx13016":     {"instrument": "MFX", "pi": "Unknown",   "sample": "PeakNet labeled",         "proposal": "13016"},
}

# Regex patterns for parsing Zarr filenames
ASSEMBLED_PATTERN = re.compile(r'^(.+)_r(\d+)\.(\d+)\.zarr$')
PEAKNET_PATTERN = re.compile(r'^(.+)_r(\d+)_peaknet\.(\d+)\.v3\.zarr$')
# mfx13016 has a slightly different naming: mfx13016_0036.NNNN.v3.zarr
MFX13016_PATTERN = re.compile(r'^(mfx13016)_(\d+)\.(\d+)\.v3\.zarr$')


def discover_runs(assembled_dir, peaknet_dir):
    """Scan directories and group Zarr files by run.

    Returns:
        dict: {run_key: {"chunks": [sorted zarr paths], "source": "assembled"|"peaknet"}}
    """
    runs = {}

    # Assembled data
    if assembled_dir.exists():
        for name in sorted(os.listdir(assembled_dir)):
            m = ASSEMBLED_PATTERN.match(name)
            if not m:
                continue
            exp, run_num, chunk_idx = m.group(1), m.group(2), m.group(3)
            run_key = f"{exp}_r{run_num}"
            if run_key not in runs:
                runs[run_key] = {"chunks": [], "source": "assembled", "exp": exp, "run_num": run_num}
            runs[run_key]["chunks"].append(assembled_dir / name)

    # Peaknet10k data (v3.zarr only)
    if peaknet_dir.exists():
        for name in sorted(os.listdir(peaknet_dir)):
            m = PEAKNET_PATTERN.match(name)
            if not m:
                m = MFX13016_PATTERN.match(name)
            if not m:
                continue
            exp, run_num, chunk_idx = m.group(1), m.group(2), m.group(3)
            run_key = f"{exp}_r{run_num}"
            if run_key not in runs:
                runs[run_key] = {"chunks": [], "source": "peaknet", "exp": exp, "run_num": run_num}
            runs[run_key]["chunks"].append(peaknet_dir / name)

    # Sort chunks within each run by chunk index
    for run_key in runs:
        runs[run_key]["chunks"].sort()

    return runs


def process_chunk(zarr_path, source, global_frame_offset, run_key):
    """Process one Zarr chunk: read frames, compute stats, build manifest rows.

    Args:
        zarr_path: Path to the Zarr store.
        source: "assembled" or "peaknet".
        global_frame_offset: Starting frame index for this chunk within the run.
        run_key: Run identifier (e.g., "cxilw5019_r0017").

    Returns:
        (entity_rows, artifact_rows, num_frames, detector, assembled_shape)
    """
    import zarr

    store = zarr.open(str(zarr_path), mode='r')
    attrs = dict(store.attrs)

    images = store['images']
    num_frames = images.shape[0]

    # Extract metadata from zarr attrs
    detector = attrs.get('detector', 'unknown')
    assembled_shape = list(images.shape[1:])

    # For peaknet, get per-frame peak metadata
    peaknet_meta = None
    if source == "peaknet" and 'metadata_migrated' in attrs:
        peaknet_meta = attrs['metadata_migrated']

    chunk_filename = zarr_path.name
    # Make file path relative to the data source directory
    if source == "assembled":
        rel_file = chunk_filename
    else:
        rel_file = chunk_filename

    entity_rows = []
    artifact_rows = []

    for i in range(num_frames):
        frame = images[i]
        global_idx = global_frame_offset + i
        uid = f"{run_key}_{global_idx:06d}"
        key = f"f_{uid}"

        # Per-frame stats
        mean_val = float(np.mean(frame))
        max_val = float(np.max(frame))
        std_val = float(np.std(frame))
        frac_zero = float(np.mean(frame == 0))

        # Peak count (from peaknet metadata if available)
        npeaks = -1
        if peaknet_meta is not None and i < len(peaknet_meta):
            frame_meta = peaknet_meta[i]
            good_peaks = frame_meta.get('good_peaks', [])
            npeaks = len(good_peaks)

        entity_rows.append({
            "uid": uid,
            "key": key,
            "frame_index": global_idx,
            "chunk_file": chunk_filename,
            "chunk_frame_index": i,
            "mean_intensity": mean_val,
            "max_intensity": max_val,
            "std_intensity": std_val,
            "fraction_zero": frac_zero,
            "npeaks": npeaks,
        })

        # Image artifact
        artifact_rows.append({
            "uid": uid,
            "type": "image",
            "file": rel_file,
            "dataset": "images",
            "index": i,
        })

        # Label artifact (peaknet only)
        if source == "peaknet" and 'labels' in store:
            artifact_rows.append({
                "uid": uid,
                "type": "label",
                "file": rel_file,
                "dataset": "labels",
                "index": i,
            })

    return entity_rows, artifact_rows, num_frames, detector, assembled_shape


def process_run(run_key, run_info):
    """Process all chunks for a single run.

    Returns:
        (run_key, ent_df, art_df, run_metadata)
    """
    chunks = run_info["chunks"]
    source = run_info["source"]
    exp = run_info["exp"]
    run_num = run_info["run_num"]

    all_entity_rows = []
    all_artifact_rows = []
    global_frame_offset = 0
    detector = "unknown"
    assembled_shape = [0, 0]

    t0 = time.time()
    for chunk_path in chunks:
        ent_rows, art_rows, n_frames, det, shape = process_chunk(
            chunk_path, source, global_frame_offset, run_key
        )
        all_entity_rows.extend(ent_rows)
        all_artifact_rows.extend(art_rows)
        global_frame_offset += n_frames
        detector = det
        assembled_shape = shape

    elapsed = time.time() - t0
    total_frames = global_frame_offset

    ent_df = pd.DataFrame(all_entity_rows)
    art_df = pd.DataFrame(all_artifact_rows)

    # Determine data_type
    data_type = "peaknet_labeled" if source == "peaknet" else "assembled"

    # Build run-level metadata for the dataset YAML
    exp_meta = EXPERIMENT_META.get(exp, {})
    run_metadata = {
        "experiment_id": exp,
        "run_number": int(run_num),
        "instrument": exp_meta.get("instrument", "Unknown"),
        "detector": detector,
        "sample_name": exp_meta.get("sample", "Unknown"),
        "pi": exp_meta.get("pi", "Unknown"),
        "proposal_id": exp_meta.get("proposal", "Unknown"),
        "data_type": data_type,
        "assembled_shape": assembled_shape,
        "num_frames": total_frames,
        "num_chunks": len(chunks),
    }

    # Determine base_dir from actual chunk location (respects CLI overrides)
    run_metadata["base_dir"] = str(chunks[0].parent)

    fps = total_frames / elapsed if elapsed > 0 else 0
    print(f"  {run_key}: {total_frames} frames, {len(chunks)} chunks, "
          f"{elapsed:.1f}s ({fps:.0f} fps)")

    return run_key, ent_df, art_df, run_metadata


def process_run_ray(run_key, run_info):
    """Ray-compatible wrapper for process_run."""
    return process_run(run_key, run_info)


def write_dataset_yaml(run_key, run_metadata, output_dir):
    """Write a dataset YAML config for one run."""
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.default_flow_style = False

    config = {
        "key": run_key,
        "generator": None,
        "base_dir": run_metadata.pop("base_dir"),
        "metadata": run_metadata,
    }

    yaml_path = output_dir / f"{run_key}.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Parquet manifests for the SFX data broker."
    )
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Specific run keys to process (e.g., cxilw5019_r0017). Default: all.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of Ray workers (0=auto, 1=sequential/no Ray).",
    )
    parser.add_argument(
        "--assembled-dir", type=Path, default=ASSEMBLED_DIR,
        help="Path to assembled Zarr directory.",
    )
    parser.add_argument(
        "--peaknet-dir", type=Path, default=PEAKNET_DIR,
        help="Path to peaknet10k Zarr directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for manifests and configs.",
    )
    args = parser.parse_args()

    manifests_dir = args.output_dir / "manifests"
    datasets_dir = args.output_dir / "datasets"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Discover runs
    print("Discovering runs...")
    all_runs = discover_runs(args.assembled_dir, args.peaknet_dir)
    print(f"Found {len(all_runs)} runs total")

    # Filter to requested runs
    if args.runs:
        filtered = {k: v for k, v in all_runs.items() if k in args.runs}
        missing = set(args.runs) - set(filtered.keys())
        if missing:
            print(f"Warning: runs not found: {missing}")
        all_runs = filtered

    if not all_runs:
        print("No runs to process.")
        return

    total_chunks = sum(len(r["chunks"]) for r in all_runs.values())
    print(f"Processing {len(all_runs)} runs ({total_chunks} chunks)...\n")

    # Process runs
    use_ray = args.num_workers != 1
    if use_ray:
        try:
            import ray
        except ImportError:
            print("Ray not available, falling back to sequential processing.")
            use_ray = False

    t_start = time.time()

    if use_ray:
        import ray

        num_cpus = args.num_workers if args.num_workers > 0 else None
        ray.init(num_cpus=num_cpus, log_to_driver=False)

        remote_fn = ray.remote(process_run_ray)
        futures = []
        for run_key, run_info in sorted(all_runs.items()):
            futures.append(remote_fn.remote(run_key, run_info))

        results = ray.get(futures)
        ray.shutdown()
    else:
        results = []
        for run_key, run_info in sorted(all_runs.items()):
            result = process_run(run_key, run_info)
            results.append(result)

    # Write outputs
    print(f"\nWriting manifests and configs...")
    total_entities = 0
    total_artifacts = 0

    for run_key, ent_df, art_df, run_metadata in results:
        # Save parquet manifests
        ent_path = manifests_dir / f"{run_key}_entities.parquet"
        art_path = manifests_dir / f"{run_key}_artifacts.parquet"
        ent_df.to_parquet(ent_path, index=False)
        art_df.to_parquet(art_path, index=False)

        # Save dataset YAML config
        write_dataset_yaml(run_key, run_metadata, datasets_dir)

        total_entities += len(ent_df)
        total_artifacts += len(art_df)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Runs:      {len(results)}")
    print(f"  Entities:  {total_entities:,}")
    print(f"  Artifacts: {total_artifacts:,}")
    print(f"  Manifests: {manifests_dir}")
    print(f"  Configs:   {datasets_dir}")


if __name__ == "__main__":
    main()
