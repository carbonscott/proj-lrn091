# Project Memory — proj-lrn091

## Script Classification

See [script-inventory.md](script-inventory.md) for full one-off vs reusable classification.

## Key Decisions

- Dataset = run (not experiment) — uniform conditions within a run
- Entity = frame — per-frame queryability for ML training selection
- Dual access: Mode A (direct Zarr, ~2ms) / Mode B (Tiled HTTP, ~60ms)
- Tiled server runs from `data/broker/` with config at `broker/config.yml`

## Directory Structure

```
data-onboard/      — 6-step reusable pipeline + dataloader/ + notebooks/
broker/            — Tiled server config + custom adapter (committed)
data/              — Raw + processed data + broker runtime (gitignored)
docs/              — Design docs + memory/
externals/         — tiled-catalog-broker (not in git)
archive/           — One-off scripts (reference)
```

## Pipeline Steps

1. `data-onboard/create_symlinks.sh --config data-onboard/symlinks.yml` — link raw data
2. `data-onboard/build_manifest.py --detector-map data-onboard/experiment_detector_map.yml` — discover & validate HDF5
3. `data-onboard/assemble_all.py` — HDF5 → Zarr (Ray-parallel)
4. `data-onboard/tar_assembled.sh` — Zarr → tarballs (for ORNL shipping)
5. `data-onboard/generate_manifests.py` — Zarr → Parquet manifests
6. `data-onboard/ingest_all.py` — Parquet → Tiled catalog
