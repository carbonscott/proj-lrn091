"""
Check class distribution in PeakNet training data.

Samples zarr stores from both experiments (mfxl1025422, mfxl1027522)
and reports the distribution of non-zero (peak) pixels per frame.
Only samples from stores that actually contain 'labels'.
"""

import os
import random
import zarr
import numpy as np
from collections import defaultdict

DATA_DIR = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet/peaknet10k"

# Collect all zarr store paths grouped by experiment, filtering for labels
print("Scanning stores for 'labels' arrays...")
stores_by_exp = defaultdict(list)
stores_without_labels = defaultdict(int)
for name in sorted(os.listdir(DATA_DIR)):
    if not name.endswith(".zarr"):
        continue
    exp = name.split("_r")[0]
    path = os.path.join(DATA_DIR, name)
    try:
        z = zarr.open(path, mode="r")
        if "labels" in z:
            stores_by_exp[exp].append(path)
        else:
            stores_without_labels[exp] += 1
    except Exception:
        stores_without_labels[exp] += 1

total_with = sum(len(v) for v in stores_by_exp.values())
total_without = sum(v for v in stores_without_labels.values())
print(f"Stores WITH labels: {total_with}")
print(f"Stores WITHOUT labels: {total_without}")
for exp in sorted(set(list(stores_by_exp.keys()) + list(stores_without_labels.keys()))):
    w = len(stores_by_exp.get(exp, []))
    wo = stores_without_labels.get(exp, 0)
    print(f"  {exp}: {w} with labels, {wo} without labels")
print()

# Sample 5 stores from each experiment
random.seed(42)
sampled = {}
for exp, paths in stores_by_exp.items():
    n = min(5, len(paths))
    sampled[exp] = random.sample(paths, n)

# Define bins for peak pixel counts
BIN_EDGES = [0, 1, 101, 1001, 5001, float("inf")]
BIN_LABELS = ["0 (no peaks)", "1-100", "101-1000", "1001-5000", "5000+"]

# Analyze
overall_counts = defaultdict(int)
per_exp_counts = {exp: defaultdict(int) for exp in sampled}
total_frames = 0
per_exp_frames = {exp: 0 for exp in sampled}
all_nonzero_counts = []

for exp, paths in sampled.items():
    print(f"--- Experiment: {exp} ---")
    for path in paths:
        store_name = os.path.basename(path)
        z = zarr.open(path, mode="r")
        labels = z["labels"]
        n_frames = labels.shape[0]
        print(f"  {store_name}: shape={labels.shape}, dtype={labels.dtype}")

        for i in range(n_frames):
            frame = labels[i]
            nonzero = int(np.count_nonzero(frame))
            all_nonzero_counts.append((exp, store_name, i, nonzero))
            total_frames += 1
            per_exp_frames[exp] += 1

            # Bin it
            for b_idx in range(len(BIN_LABELS)):
                if BIN_EDGES[b_idx] <= nonzero < BIN_EDGES[b_idx + 1]:
                    overall_counts[BIN_LABELS[b_idx]] += 1
                    per_exp_counts[exp][BIN_LABELS[b_idx]] += 1
                    break
    print()

# Print summary
n_stores = sum(len(v) for v in sampled.values())
print("=" * 70)
print(f"OVERALL CLASS DISTRIBUTION ({total_frames} frames from {n_stores} stores)")
print("=" * 70)
for label in BIN_LABELS:
    count = overall_counts[label]
    pct = 100.0 * count / total_frames if total_frames else 0
    bar = "#" * int(pct / 2)
    print(f"  {label:>15s}: {count:6d} frames ({pct:5.1f}%) {bar}")

print()
print("PER-EXPERIMENT BREAKDOWN:")
for exp in sampled:
    n = per_exp_frames[exp]
    if n == 0:
        print(f"\n  {exp}: no frames analyzed")
        continue
    print(f"\n  {exp} ({n} frames):")
    for label in BIN_LABELS:
        count = per_exp_counts[exp][label]
        pct = 100.0 * count / n if n else 0
        print(f"    {label:>15s}: {count:6d} ({pct:5.1f}%)")

# Basic stats
nonzero_vals = [x[3] for x in all_nonzero_counts]
nonzero_only = [v for v in nonzero_vals if v > 0]
print()
print("STATISTICS ON NON-ZERO PIXEL COUNTS (per frame):")
print(f"  All frames:   min={min(nonzero_vals)}, max={max(nonzero_vals)}, "
      f"mean={np.mean(nonzero_vals):.1f}, median={np.median(nonzero_vals):.1f}")
if nonzero_only:
    print(f"  Hits only:    min={min(nonzero_only)}, max={max(nonzero_only)}, "
          f"mean={np.mean(nonzero_only):.1f}, median={np.median(nonzero_only):.1f}")
    print(f"  Hit rate:     {len(nonzero_only)}/{len(nonzero_vals)} "
          f"= {100.0*len(nonzero_only)/len(nonzero_vals):.1f}%")
else:
    print("  No frames with non-zero peak pixels found.")

# Percentile distribution
print()
print("PERCENTILE DISTRIBUTION (non-zero pixel counts, hits only):")
if nonzero_only:
    for p in [5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(nonzero_only, p)
        print(f"  P{p:02d}: {val:.0f}")

# Check unique label values
print()
print("UNIQUE LABEL VALUES (sample from each experiment):")
for exp, paths in sampled.items():
    z = zarr.open(paths[0], mode="r")
    frame0 = z["labels"][0]
    unique_vals = np.unique(frame0)
    print(f"  {exp} ({os.path.basename(paths[0])}, frame 0):")
    print(f"    Unique values: {unique_vals}")
    for v in unique_vals:
        cnt = np.sum(frame0 == v)
        pct = 100.0 * cnt / frame0.size
        print(f"    value={v}: {cnt:>10d} pixels ({pct:.4f}%)")

# Class imbalance at pixel level
print()
print("PIXEL-LEVEL CLASS IMBALANCE (across all analyzed frames):")
total_pixels = total_frames * 1920 * 1920
total_peak_pixels = sum(nonzero_vals)
total_bg_pixels = total_pixels - total_peak_pixels
ratio = total_bg_pixels / total_peak_pixels if total_peak_pixels > 0 else float('inf')
print(f"  Total pixels:      {total_pixels:>15,d}")
print(f"  Background pixels: {total_bg_pixels:>15,d} ({100.0*total_bg_pixels/total_pixels:.4f}%)")
print(f"  Peak pixels:       {total_peak_pixels:>15,d} ({100.0*total_peak_pixels/total_pixels:.4f}%)")
print(f"  BG:Peak ratio:     {ratio:,.0f}:1")
