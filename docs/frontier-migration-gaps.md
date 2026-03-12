# Frontier Migration Gap Analysis

Analysis of gaps for moving the Tiled data broker service from SLAC S3DF
to OLCF Frontier.

## 1. What Is Being Migrated

The Tiled data broker serves assembled SFX crystallography data over HTTP,
enabling frame-level queries across 44 runs (~440K frames). The service
consists of:

- **Tiled server** — HTTP API backed by SQLite catalog (`catalog.db`)
- **Custom Zarr adapter** — navigates into Zarr groups with `dataset`/`slice`
  parameters
- **Assembled Zarr data** — pre-assembled detector images (float32, zstd)
- **Parquet manifests** — per-frame statistics for catalog ingestion
- **6-step onboarding pipeline** — scripts that build everything from raw HDF5

## 2. Data Inventory

| Component | Location | Size | Transfer? |
|-----------|----------|------|-----------|
| Assembled Zarr chunks | `data/assembled/` | ~4.8 TB (10,528 chunks) | Yes |
| PeakNet labeled data | `data/peaknet10k/` | ~50–100 GB (490 chunks) | Yes |
| Broker catalog | `data/broker/catalog.db` | ~1 GB | No (regenerate) |
| Parquet manifests | `data/broker/manifests/` | ~200 MB (88 files) | Optional |
| Dataset YAML configs | `data/broker/datasets/` | <1 MB (44 files) | Optional |
| Geometry files | LCLS filesystem | <1 MB (8 `.geom` files) | Yes (copy) |
| Metadata JSONs | `data/*.json` | ~1.2 MB | Yes |
| Sample images | `data/sample-images/` | ~32 MB | Optional |
| Raw HDF5 (via symlinks) | LCLS data federation | ~9.7 TB | No |
| **Total to transfer** | | **~5 TB** | |

The raw HDF5 files (~9.7 TB) live on the LCLS data federation and are
accessed via symlinks. They are NOT needed on Frontier if we transfer the
pre-assembled Zarr data (~4.8 TB assembled + ~50–100 GB labeled). The
catalog.db should be regenerated on-site (steps 5–6) to embed correct
paths.

## 3. Critical Gaps

### 3.1 Hardcoded S3DF Absolute Paths

Every `/sdf/` path must be updated for Frontier. Here is the complete
inventory:

**`broker/config.yml`** (3 paths):
- Line 9: SQLite URI — `sqlite:////sdf/.../data/broker/catalog.db`
- Line 15: readable_storage — `/sdf/.../data/assembled`
- Line 16: readable_storage — `/sdf/.../data/peaknet10k`

**`data-onboard/notebooks/explore_catalog.py`** (2 paths):
- Line 214: `_BASE_DIRS["assembled"]` — `/sdf/.../data/assembled`
- Line 215: `_BASE_DIRS["peaknet_labeled"]` — `/sdf/.../data/peaknet10k`

**`data/geometry_registry.json`** (8 paths):
- Lines 8, 12, 16, 20, 24, 28, 32, 36 — each experiment's CrystFEL
  `.geom` file path on the LCLS filesystem

**`data-onboard/symlinks.yml`** (13 paths):
- Lines 5–21 — raw HDF5 source directories on the LCLS data federation
  (only relevant if re-assembling from raw data)

**`data-onboard/assemble_all.py`** (2 paths):
- Line 50: `CSP_ROOT = Path.home() / "codes" / "crystfel_stream_parser"`
- Line 334: duplicate `Path.home()` reference in Ray remote function
  (only relevant if re-assembling)

**`data-onboard/ingest_all.py`** (1 path):
- Line 33: `BROKER_DIR` pointing to `externals/data-broker/tiled-catalog-broker/tiled_poc`
- Line 34: `sys.path.insert(0, str(BROKER_DIR))`

**`CLAUDE.md`** (11 env vars):
- Lines 6–16: all environment variable definitions reference `/sdf/` paths

**Total: ~40 hardcoded `/sdf/` references across 7 files.**

### 3.2 External Dependency: `tiled-catalog-broker`

`ingest_all.py` imports bulk registration functions from
`externals/data-broker/tiled-catalog-broker/tiled_poc/`:
```python
from broker.bulk_register import init_database, prepare_node_data, bulk_register
```

This submodule must be cloned on Frontier and the path in `ingest_all.py`
updated. Alternatively, install it as a proper Python package.

### 3.3 External Dependency: `crystfel_stream_parser`

`assemble_all.py` loads `crystfel_stream_parser` from the user's home
directory (`~/codes/crystfel_stream_parser/`). This is only needed for
re-assembling raw HDF5 → Zarr. If transferring pre-assembled data, this
dependency can be skipped.

### 3.4 No Dependency Lockfile

All Python dependencies are specified inline via `uv run --with`:
```bash
uv run --with h5py --with zarr --with numpy --with ray python ...
```

There is no `pyproject.toml` or `requirements.txt` with pinned versions.
Package versions may differ on Frontier, causing breakage.

## 4. Moderate Gaps

### 4.1 Compute Environment Differences

| Aspect | S3DF | Frontier |
|--------|------|----------|
| CPU Architecture | x86_64 (Intel) | x86_64 (AMD EPYC) |
| GPU | NVIDIA A100 (Ada partition) | AMD MI250X |
| Scheduler | Slurm (account `lcls:prjdat21`) | Slurm (different accounts) |
| CPU partition | `milano` | TBD |
| Python tooling | `uv` available | `uv` may need installation |
| Parallel framework | Ray | Ray should work, but node topology differs |

Key concerns:
- Slurm `srun` commands in the README use S3DF-specific `--partition` and
  `-A` flags
- `uv` is not a standard HPC module — may need manual install or
  `pip install uv`
- Ray worker counts (e.g., `--num-workers 40`) are tuned for S3DF Milano
  nodes (128 cores) — Frontier nodes have 64 cores + 8 GPUs

### 4.2 SQLite Catalog Non-Portability

`catalog.db` stores absolute Zarr paths in its artifact records. Simply
copying the file will leave it pointing to `/sdf/` paths. Options:
1. **Regenerate** — run steps 5–6 after data lands at new paths (recommended)
2. **Path rewrite** — SQL UPDATE on the `data_uri` column (fragile)

### 4.3 Network/Firewall for Tiled HTTP

The Tiled server binds to `127.0.0.1:8007` (localhost only). On Frontier:
- Compute nodes may not allow inbound HTTP connections
- May need SSH tunnel or reverse proxy for remote access
- Check OLCF network policies for user-facing services on compute nodes

### 4.4 Filesystem Differences

S3DF uses a GPFS-like parallel filesystem. Frontier uses Orion (Lustre).
Zarr I/O patterns (many small files per chunk) may perform differently on
Lustre. Consider:
- Stripe settings for Zarr directories
- Potential benefit of consolidated Zarr metadata
- Whether Zarr-over-tar (ZipStore) would be more Lustre-friendly

## 5. Low-Priority Gaps

### 5.1 No Site Configuration Abstraction

Paths are scattered across multiple files with no single configuration
point. A `site.yml` or environment-variable-based resolution would make
multi-site deployment straightforward.

### 5.2 Tar Packaging Exists but May Be Stale

`data-onboard/tar_assembled.sh` creates tarballs in
`data/assembled_tarballs/` for shipping. However, this directory is
currently empty (8K) — tarballs have not been generated yet. They would
need to be created before transfer, or use direct Globus transfer of the
Zarr directories instead.

### 5.3 Marimo Notebook Server

`data-onboard/notebooks/explore_catalog.py` requires a Marimo server.
Frontier may not support interactive notebook sessions on compute nodes
without additional setup (e.g., Open OnDemand, JupyterHub).

## 6. Recommended Actions

Ordered by priority:

1. **Create `pyproject.toml`** — pin all dependencies with exact versions
   for reproducibility across sites

2. **Introduce `site.yml` configuration** — centralize all site-specific
   paths (project root, data dirs, Slurm accounts) into one file that all
   scripts read from

3. **Transfer data via Globus** — use existing tarballs or direct Globus
   transfer for `data/assembled/` and `data/peaknet10k/`

4. **Copy geometry files** — extract the 8 `.geom` files from LCLS
   filesystem and bundle them into the repo (they're small, <1 MB total)

5. **Update `broker/config.yml`** — point to new Frontier paths after data
   lands

6. **Regenerate catalog** — run steps 5–6 on Frontier to build
   `catalog.db` with correct local paths

7. **Test Tiled server on Frontier** — verify HTTP binding, network
   access, and Zarr I/O performance on Lustre

8. **Document Frontier-specific setup** — Slurm accounts, module loads,
   `uv` installation, Ray worker counts
