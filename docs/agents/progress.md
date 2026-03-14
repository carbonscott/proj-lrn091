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

9. **Installed Ray** into `.venv-broker/` for parallel manifest generation.

10. **Switched to Ray-parallel manifest generation** — initial sequential run
    (`--num-workers 1`) completed only 1 small run in ~25 min. Killed it and
    re-launched with `--num-workers 0` (Ray auto, ~56 cores) on compute node
    `frontier10430` (job 4204437, extended partition, debug QOS).

### In Progress

- Server verification with Tiled client against the full catalog.

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

## 2026-03-13: Manifest Generation, Catalog Ingestion, Broker Cleanup & Tests

### Context

Completed the full data onboarding pipeline: manifest generation for all 30
assembled runs, catalog ingestion, broker codebase cleanup, and added a test
suite with CI.

### Completed

1. **Ray-parallel manifest generation finished** — All 30 assembled runs
   processed on compute node `frontier10430` (job 4204437, extended partition).
   Produced 30 entity + 30 artifact Parquet manifests and 30 dataset YAML
   configs under `data/broker/manifests/` and `data/broker/datasets/`.

2. **Full catalog ingestion** — All 30 runs bulk-registered into
   `data/broker/catalog.db` (235 MB) via `bulk_register` with Zarr mimetype
   and `is_directory=1`.

3. **Split manifest generators** — Replaced the single `generate_manifests.py`
   with two separate scripts (`generate_manifests_assembled.py` and
   `generate_manifests_peaknet.py`) in `data-onboard/`. Both support
   Ray-parallel processing and produce per-run Parquet + YAML outputs.

4. **lcls-data-broker cleanup** (commits `ceeeb41`→`0ecb242`):
   - Removed upstream VDP/MAIQMag leftovers: `docs/`, `examples/`, `demo/`,
     `extra/`, old `tests/` directories
   - Deleted `query_manifest.py` (MAIQMag-specific, imported non-existent fn)
   - Removed manifest generation from broker scope — manifests are the data
     provider's responsibility, not the broker's

5. **Added test suite & CI** (commit `88318c4` in lcls-data-broker):
   - 21 pytest tests across 3 modules (`test_utils`, `test_config`,
     `test_bulk_register`) using synthetic Zarr fixtures (no real data needed)
   - Full round-trip test: `init_database` → `prepare_node_data` →
     `bulk_register` → SQL verification
   - GitHub Actions CI on Python 3.11/3.12 with uv, triggered on push/PR
     to main

### Key Learnings

- Tiled's `init_database` creates a hidden root node (id=0) automatically.
  Node count assertions must account for this extra node.
- The `_load_artifact_info` cache in `utils.py` uses a mutable default arg.
  Tests must clear it between runs to avoid cross-test interference.

## 2026-03-13: Egress Showcase Marimo Notebook

### Context

Created a marimo notebook to showcase the data broker's egress capabilities —
demonstrating the different ways users can retrieve data from the Tiled catalog.

### Completed

1. **Created `notebooks/demo_egress.py`** — 8-section marimo notebook covering:
   catalog overview, full frame retrieval (Mode B), sliced/ROI reads, direct
   Zarr access (Mode A), data equivalence check, query-driven fetch by frame
   statistics, multi-artifact (image + label), and cross-run uniform access.

2. **Fixed rendering issues** — marimo cells using `mo.md()` inside `if/else`
   blocks don't display output (not a top-level expression). Converted all
   guard checks to `mo.stop()` for idiomatic marimo early-exit with output.

3. **Added shape-mismatch resilience** — Some `mfxl*` runs have stale catalog
   metadata (expected_shape 2203x2299 vs actual 1920x1920). Added try/except
   around `.read()` calls in multi-artifact and cross-run cells.

### Key Learnings

- In marimo, `mo.md()` must be a **top-level expression** to display as cell
  output. Inside `if/else` blocks it's discarded. Use `mo.stop(condition, output)`
  for guard patterns.
- PEP 723 script headers only work with `uv run --script`. Running
  `uv run --with marimo marimo edit` ignores the header deps.
- `fowld` (from the `fowl` package) is the non-TUI alternative for port
  forwarding on systems without curses support.
