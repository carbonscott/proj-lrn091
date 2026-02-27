# Project Memory — proj-lrn091

## Script Classification

See [script-inventory.md](script-inventory.md) for full one-off vs reusable classification.

## Key Decisions

- Dataset = run (not experiment) — uniform conditions within a run
- Entity = frame — per-frame queryability for ML training selection
- Dual access: Mode A (direct Zarr, ~2ms) / Mode B (Tiled HTTP, ~60ms)
- Tiled server requires `PYTHONPATH=.` when launched from `data/broker/`

## Directory Structure

```
pipeline/          — 6-step reusable onboarding pipeline + config files
training/          — PyTorch dataset classes (assembled_dataset, panel_dataset, transforms, manifest)
notebooks/         — Marimo notebooks (explore_catalog.py)
archive/           — One-off scripts kept for reference
data/broker/       — Tiled config, catalog.db, manifests/, datasets/, custom adapter
docs/              — Architecture and design docs
externals/         — tiled-catalog-broker submodule
exploration/       — README.md only (historical)
```

## Pipeline Steps

1. `pipeline/create_symlinks.sh --config pipeline/symlinks.yml` — link raw data
2. `pipeline/build_manifest.py --detector-map pipeline/experiment_detector_map.yml` — discover & validate HDF5
3. `pipeline/assemble_all.py` — HDF5 → Zarr (Ray-parallel)
4. `pipeline/tar_assembled.sh` — Zarr → tarballs (for ORNL shipping)
5. `pipeline/generate_manifests.py` — Zarr → Parquet manifests
6. `pipeline/ingest_all.py` — Parquet → Tiled catalog
