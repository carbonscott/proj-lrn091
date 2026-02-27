# Data Onboarding Pipeline

Six-step pipeline for onboarding new SFX experiment data.

## Steps

### 1. Create symlinks to raw data

```bash
bash data-onboard/create_symlinks.sh --config data-onboard/symlinks.yml
bash data-onboard/create_symlinks.sh --config data-onboard/symlinks.yml --dry-run
```

To add a new experiment, add an entry to `symlinks.yml`.

### 2. Build HDF5 manifest

```bash
uv run --with h5py python data-onboard/build_manifest.py
uv run --with h5py python data-onboard/build_manifest.py \
    --detector-map data-onboard/experiment_detector_map.yml \
    --experiments cxilw5019--psocake
```

To add a new experiment, add an entry to `experiment_detector_map.yml`.

### 3. Assemble images (HDF5 → Zarr)

```bash
srun uv run --with h5py --with zarr --with numpy --with ray \
    python data-onboard/assemble_all.py \
    --experiments cxilw5019 --num-workers 40 --chunk-size 40
```

Requires a compute node (Milano or Ada). Outputs to `data/assembled/`.

### 4. Package for shipping (Zarr → tar)

```bash
bash data-onboard/tar_assembled.sh --jobs 4
```

Outputs to `data/assembled_tarballs/`. Idempotent (tracks completed
runs in `tarred_runs.txt`).

### 5. Generate Parquet manifests

```bash
srun uv run --with zarr --with numpy --with pandas --with pyarrow \
    --with ray --with 'ruamel.yaml' \
    python data-onboard/generate_manifests.py --num-workers 40
```

Requires a compute node. Outputs to `data/broker/manifests/` and
`data/broker/datasets/`.

### 6. Ingest into Tiled catalog

```bash
uv run --with 'tiled[server]' --with pandas --with pyarrow --with h5py \
    --with zarr --with numpy --with 'ruamel.yaml' --with canonicaljson \
    --with sqlalchemy python data-onboard/ingest_all.py --fresh
```

Populates `data/broker/catalog.db`. Use `--fresh` to rebuild from
scratch, or omit to add new runs incrementally.

## Subdirectories

- `dataloader/` — PyTorch dataset classes for training (AssembledPatchDataset, PanelPatchDataset)
- `notebooks/` — Marimo notebooks for interactive catalog exploration
