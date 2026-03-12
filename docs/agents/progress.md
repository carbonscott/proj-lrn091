# Agent Progress Log

## 2026-03-12: Data Broker Migration to Frontier

### Context

Migrating the SFX crystallography Tiled data broker from SLAC S3DF to OLCF
Frontier. The assembled Zarr data (~4.8 TB) and peaknet10k (~100 GB) were
already transferred to `/lustre/orion/lrn091/proj-shared/data/`.

### Completed

1. **Gap analysis** — Identified ~40 hardcoded `/sdf/` paths, missing broker
   infrastructure, HDF5 assumptions in the broker module, and confirmed all
   Zarr files are v3 format (zarr v2 cannot open them).

2. **Forked broker repo** — Copied `tiled-catalog-broker` to
   `deps/lcls-data-broker/` as an independent repo with its own git history.

3. **Added Zarr v3 support to lcls-data-broker** (3 changes):
   - `broker/utils.py`: `get_artifact_shape` falls back to zarr when h5py
     fails; added `get_artifact_dtype` for dynamic dtype detection.
   - `broker/bulk_register.py`: `mimetype` and `is_directory` parameters
     (defaults preserve HDF5 backwards compatibility). Replaced hardcoded
     float64 dtype with actual dtype from data files.

4. **Fixed `data-onboard/ingest_all.py`**:
   - Broker path uses `LCLS_DATA_BROKER_DIR` env var → `deps/lcls-data-broker/`
   - `readable_storage` uses `XTAL_DATA_ASSEMBLED` env var → single flat
     data directory
   - Passes `mimetype="application/x-zarr"` and `is_directory=1` to
     `bulk_register()`

5. **Fixed `data-onboard/generate_manifests.py`**:
   - `base_dir` in generated YAML configs now derived from actual chunk path
     (was using module-level constant that didn't respect CLI `--assembled-dir`)

6. **Updated `broker/config.yml`**:
   - SQLite URI: relative `catalog.db` (server runs from `data/broker/`)
   - `readable_storage`: `/lustre/orion/lrn091/proj-shared/data`
   - Adapter: `application/x-zarr` → `sfx_zarr_adapter:SFXZarrAdapter`

7. **Created pre-installed venv** at `.venv-broker/` with all dependencies
   (compute nodes have no internet, so `uv run` fails on them).

8. **End-to-end validation** with 1 test run (`mfx100903824_r0027`, 25 frames):
   - Manifest generation: 1.9s
   - Catalog ingestion: 0.2s
   - Tiled server: correct 4-layer hierarchy (root → dataset → frame → image),
     correct float32 dtype, data retrieval works.

### In Progress

- **Full manifest generation** for all 30 runs (~440K frames) running on
  compute node `frontier10430` (job 4204437, extended partition, debug QOS).
- Once manifests are done: full catalog ingestion and server verification.

### Data Layout Decision

No reorganization of the flat Zarr layout. Both assembled and peaknet files
coexist in `/lustre/orion/lrn091/proj-shared/data/` and are distinguished by
filename regex patterns. The broker's `readable_storage` points to this single
directory.

### Key Learnings

- Frontier compute nodes have **no internet access** — `uv run --with` fails.
  Must pre-install packages in a venv on the login node.
- `salloc` requires an interactive shell to hold the allocation. From a
  non-interactive process, use `sbatch` or `srun --jobid=<ID>` against a
  user-held allocation.
- For small 1-node jobs on Frontier: `extended -q debug` provides up to 2h
  walltime with higher scheduling priority than plain `batch`.
