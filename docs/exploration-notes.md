# Exploration

Early prototyping code for understanding the LCLS diffraction data and testing
a data pipeline concept. This is **not production code** — it's a learning
exercise to inform architecture decisions.


## Scripts (logical order)

| Script | What it does | Dependencies |
|---|---|---|
| `scripts/explore_hdf5.py` | Walk symlinks, catalog HDF5 structure | h5py |
| `scripts/visualize_samples.py` | Render sample frames with multiple clipping strategies | h5py, matplotlib, numpy |
| `scripts/build_manifest.py` | Scan all files, validate shapes, write `data/manifest.json` | h5py |
| `scripts/test_dataloader.py` | Smoke test the data pipeline prototype | h5py, torch, matplotlib, numpy |

Run any script from the project root:
```
uv run --with h5py exploration/scripts/explore_hdf5.py
```


## Data pipeline prototype

`data_pipeline/` contains a minimal PyTorch Dataset that reads the manifest,
indexes (file, frame, panel) triples, and extracts random patches from detector
panels. Built to test whether panel-wise patching is mechanically feasible.

Not yet validated for actual training.


## Lessons learned

### 1. Data landscape is messier than expected

The `prjcwang31--userdata-psocake` symlink contains files from 4+ different
detector geometries (4096x1024, 1480x1552, 5632x384, 1920x1920), not just
assembled 1920x1920 images as initially assumed. Shape validation during
manifest building is essential — 2,564 files were filtered out due to
mismatches.

### 2. Many empty/sentinel frames

`cxilw5019` has 2.3M frames, many of which are all-zero (empty background or
sentinel values). Random sampling from the full index yields ~75% blank patches.
Any real training pipeline will need upfront filtering or a smarter sampler
that skips empty frames.

### 3. Corrupt files exist in the wild

Several HDF5 files have truncated data or bad headers. The pipeline must handle
read errors gracefully rather than crashing mid-epoch.

### 4. Scale is significant

4M frames across 3,209 files. The flat (file, frame, panel) index creates
36.7M entries in memory. Worth considering lazy/chunked indexing for the real
pipeline, or filtering experiments at manifest level.

### 5. Panel-wise patching is mechanically feasible

Splitting stacked-panel images along the H dimension into physical panels and
extracting random 256x256 patches works. Real evaluation needs actual model
training to see if patches carry meaningful signal for self-supervised learning.

### 6. Too early for production packaging

Building `src/lrn091/` implied more certainty than we have. Architecture choices
(MAE vs DINOv2, patch size, normalization strategy, how to handle empty frames)
are still open. The exploration informed these questions but didn't answer them.
