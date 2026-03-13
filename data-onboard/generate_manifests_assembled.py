"""
Generate Parquet manifests for assembled SFX Zarr data.

Scans assembled Zarr chunks, computes per-frame statistics, and produces
entity/artifact manifest Parquet files + dataset YAML configs for each run.

Usage:
    # All runs (needs a compute node for ~440K frames):
    python data-onboard/generate_manifests_assembled.py \
        --data-dir /lustre/orion/lrn091/proj-shared/data

    # Specific runs:
    python data-onboard/generate_manifests_assembled.py \
        --data-dir /lustre/orion/lrn091/proj-shared/data \
        --runs cxilw5019_r0017 mfx100903824_r0027

    # Sequential (no Ray, for debugging):
    python data-onboard/generate_manifests_assembled.py \
        --data-dir /lustre/orion/lrn091/proj-shared/data \
        --runs mfx100903824_r0027 --num-workers 1
"""

import argparse
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "data" / "broker"

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

# Assembled Zarr filename: {exp}_r{run}.{chunk}.zarr
ASSEMBLED_PATTERN = re.compile(r'^(.+)_r(\d+)\.(\d+)\.zarr$')


def discover_runs(data_dir):
    """Scan directory and group assembled Zarr files by run.

    Returns:
        dict: {run_key: {"chunks": [sorted zarr paths], "exp": str, "run_num": str}}
    """
    runs = {}
    for name in sorted(os.listdir(data_dir)):
        m = ASSEMBLED_PATTERN.match(name)
        if not m:
            continue
        exp, run_num, chunk_idx = m.group(1), m.group(2), m.group(3)
        run_key = f"{exp}_r{run_num}"
        if run_key not in runs:
            runs[run_key] = {"chunks": [], "exp": exp, "run_num": run_num}
        runs[run_key]["chunks"].append(data_dir / name)

    for run_key in runs:
        runs[run_key]["chunks"].sort()

    return runs


def process_chunk(zarr_path, global_frame_offset, run_key):
    """Process one assembled Zarr chunk: read frames, compute stats.

    Returns:
        (entity_rows, artifact_rows, num_frames, detector, assembled_shape)
    """
    import zarr

    store = zarr.open(str(zarr_path), mode='r')
    attrs = dict(store.attrs)

    images = store['images']
    num_frames = images.shape[0]
    detector = attrs.get('detector', 'unknown')
    assembled_shape = list(images.shape[1:])

    chunk_filename = zarr_path.name
    entity_rows = []
    artifact_rows = []

    for i in range(num_frames):
        frame = images[i]
        global_idx = global_frame_offset + i
        uid = f"{run_key}_{global_idx:06d}"
        key = f"f_{uid}"

        mean_val = float(np.mean(frame))
        max_val = float(np.max(frame))
        std_val = float(np.std(frame))
        frac_zero = float(np.mean(frame == 0))

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
        })

        artifact_rows.append({
            "uid": uid,
            "type": "image",
            "file": chunk_filename,
            "dataset": "images",
            "index": i,
        })

    return entity_rows, artifact_rows, num_frames, detector, assembled_shape


def process_run(run_key, run_info):
    """Process all chunks for a single run.

    Returns:
        (run_key, ent_df, art_df, run_metadata)
    """
    chunks = run_info["chunks"]
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
            chunk_path, global_frame_offset, run_key
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

    exp_meta = EXPERIMENT_META.get(exp, {})
    run_metadata = {
        "experiment_id": exp,
        "run_number": int(run_num),
        "instrument": exp_meta.get("instrument", "Unknown"),
        "detector": detector,
        "sample_name": exp_meta.get("sample", "Unknown"),
        "pi": exp_meta.get("pi", "Unknown"),
        "proposal_id": exp_meta.get("proposal", "Unknown"),
        "data_type": "assembled",
        "assembled_shape": assembled_shape,
        "num_frames": total_frames,
        "num_chunks": len(chunks),
        "base_dir": str(chunks[0].parent),
    }

    fps = total_frames / elapsed if elapsed > 0 else 0
    print(f"  {run_key}: {total_frames} frames, {len(chunks)} chunks, "
          f"{elapsed:.1f}s ({fps:.0f} fps)")

    return run_key, ent_df, art_df, run_metadata


def process_chunk_remote_fn(zarr_path, run_key):
    """Ray remote unit: process a single chunk with offset=0 (fixed up later)."""
    return process_chunk(zarr_path, 0, run_key)


def aggregate_chunk_results(chunk_results, run_key, run_info):
    """Aggregate per-chunk results into a single run result with correct offsets.

    Args:
        chunk_results: list of (entity_rows, artifact_rows, num_frames, detector, shape)
                       in chunk order.
        run_key: e.g. "mfx100903824_r0027"
        run_info: dict with "chunks", "exp", "run_num"

    Returns:
        (run_key, ent_df, art_df, run_metadata) — same shape as process_run().
    """
    all_entity_rows = []
    all_artifact_rows = []
    global_frame_offset = 0
    detector = "unknown"
    assembled_shape = [0, 0]

    for ent_rows, art_rows, n_frames, det, shape in chunk_results:
        if global_frame_offset > 0:
            # Build old_uid -> new_uid mapping for this chunk
            uid_map = {}
            for row in ent_rows:
                old_uid = row["uid"]
                new_idx = row["frame_index"] + global_frame_offset
                new_uid = f"{run_key}_{new_idx:06d}"
                uid_map[old_uid] = new_uid
                row["uid"] = new_uid
                row["key"] = f"f_{new_uid}"
                row["frame_index"] = new_idx
            for row in art_rows:
                row["uid"] = uid_map[row["uid"]]

        all_entity_rows.extend(ent_rows)
        all_artifact_rows.extend(art_rows)
        global_frame_offset += n_frames
        detector = det
        assembled_shape = shape

    ent_df = pd.DataFrame(all_entity_rows)
    art_df = pd.DataFrame(all_artifact_rows)

    exp = run_info["exp"]
    run_num = run_info["run_num"]
    exp_meta = EXPERIMENT_META.get(exp, {})
    run_metadata = {
        "experiment_id": exp,
        "run_number": int(run_num),
        "instrument": exp_meta.get("instrument", "Unknown"),
        "detector": detector,
        "sample_name": exp_meta.get("sample", "Unknown"),
        "pi": exp_meta.get("pi", "Unknown"),
        "proposal_id": exp_meta.get("proposal", "Unknown"),
        "data_type": "assembled",
        "assembled_shape": assembled_shape,
        "num_frames": global_frame_offset,
        "num_chunks": len(run_info["chunks"]),
        "base_dir": str(run_info["chunks"][0].parent),
    }

    return run_key, ent_df, art_df, run_metadata


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
        description="Generate Parquet manifests for assembled SFX Zarr data."
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
        "--data-dir", type=Path, required=True,
        help="Path to directory containing assembled Zarr files.",
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

    print("Discovering assembled runs...")
    all_runs = discover_runs(args.data_dir)
    print(f"Found {len(all_runs)} assembled runs")

    if args.runs:
        filtered = {k: v for k, v in all_runs.items() if k in args.runs}
        missing = set(args.runs) - set(filtered.keys())
        if missing:
            print(f"Warning: runs not found: {missing}")
        all_runs = filtered

    if not all_runs:
        print("No runs to process.")
        return

    # Skip runs that already have manifests
    existing = {f.stem.replace("_entities", "")
                for f in manifests_dir.glob("*_entities.parquet")}
    skipped = {k for k in all_runs if k in existing}
    if skipped:
        print(f"Skipping {len(skipped)} runs with existing manifests: "
              f"{sorted(skipped)}")
        all_runs = {k: v for k, v in all_runs.items() if k not in existing}

    if not all_runs:
        print("All runs already have manifests. Nothing to do.")
        return

    total_chunks = sum(len(r["chunks"]) for r in all_runs.values())
    print(f"Processing {len(all_runs)} runs ({total_chunks} chunks)...\n")

    use_ray = args.num_workers != 1
    if use_ray:
        try:
            import ray
        except ImportError:
            print("Ray not available, falling back to sequential processing.")
            use_ray = False

    t_start = time.time()
    total_entities = 0
    total_artifacts = 0
    runs_written = 0

    if use_ray:
        import ray

        num_cpus = args.num_workers if args.num_workers > 0 else None
        ray.init(num_cpus=num_cpus, log_to_driver=False)

        remote_fn = ray.remote(process_chunk_remote_fn)

        # Submit all chunks as individual Ray tasks
        future_to_info = {}  # ray_future -> (run_key, chunk_index)
        run_tracker = {}     # run_key -> {"total": int, "results": {chunk_idx: result}}
        for run_key, run_info in sorted(all_runs.items()):
            chunks = run_info["chunks"]
            run_tracker[run_key] = {"total": len(chunks), "results": {}}
            for chunk_idx, chunk_path in enumerate(chunks):
                fut = remote_fn.remote(chunk_path, run_key)
                future_to_info[fut] = (run_key, chunk_idx)

        pending = list(future_to_info.keys())
        print(f"Submitted {len(pending)} chunk tasks across "
              f"{len(run_tracker)} runs\n")

        # Incremental collection with ray.wait()
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            for fut in done:
                result = ray.get(fut)
                run_key, chunk_idx = future_to_info[fut]
                tracker = run_tracker[run_key]
                tracker["results"][chunk_idx] = result

                # Check if this run is now complete
                if len(tracker["results"]) == tracker["total"]:
                    run_info = all_runs[run_key]
                    # Sort by chunk index to get correct order
                    sorted_results = [tracker["results"][i]
                                      for i in range(tracker["total"])]
                    rk, ent_df, art_df, run_meta = aggregate_chunk_results(
                        sorted_results, run_key, run_info
                    )
                    # Write immediately and free memory
                    ent_df.to_parquet(
                        manifests_dir / f"{rk}_entities.parquet", index=False)
                    art_df.to_parquet(
                        manifests_dir / f"{rk}_artifacts.parquet", index=False)
                    write_dataset_yaml(rk, run_meta, datasets_dir)

                    total_entities += len(ent_df)
                    total_artifacts += len(art_df)
                    runs_written += 1
                    print(f"  {rk}: {run_meta['num_frames']} frames, "
                          f"{tracker['total']} chunks — written")
                    del run_tracker[run_key]

        ray.shutdown()
    else:
        for run_key, run_info in sorted(all_runs.items()):
            rk, ent_df, art_df, run_meta = process_run(run_key, run_info)

            ent_df.to_parquet(
                manifests_dir / f"{rk}_entities.parquet", index=False)
            art_df.to_parquet(
                manifests_dir / f"{rk}_artifacts.parquet", index=False)
            write_dataset_yaml(rk, run_meta, datasets_dir)

            total_entities += len(ent_df)
            total_artifacts += len(art_df)
            runs_written += 1

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Runs:      {runs_written}")
    print(f"  Entities:  {total_entities:,}")
    print(f"  Artifacts: {total_artifacts:,}")
    print(f"  Manifests: {manifests_dir}")
    print(f"  Configs:   {datasets_dir}")


if __name__ == "__main__":
    main()
