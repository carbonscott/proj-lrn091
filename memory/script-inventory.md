# Script Inventory — One-off vs Reusable

## One-off scripts

| Script | Purpose | Notes |
|--------|---------|-------|
| `exploration/scripts/explore_hdf5.py` | Cataloged HDF5 structure | Discovery — done once |
| `exploration/scripts/build_manifest.py` | Built manifest.json from raw HDF5 | Superseded by generate_manifests.py |
| `exploration/scripts/test_assembly.py` | POC: assemble one frame | Proof-of-concept |
| `exploration/scripts/test_dataloader.py` | Smoke test PanelPatchDataset | One-time validation |
| `exploration/scripts/visualize_samples.py` | Render sample frames per experiment | Visual inspection |
| `exploration/scripts/create_symlinks.sh` | Set up experiment symlinks | Setup — run once per environment |
| `scripts/check_class_distribution.py` | Analyze PeakNet label distributions | Hardcoded paths, one-time analysis |

## Reusable scripts/modules

| Script | Purpose | Notes |
|--------|---------|-------|
| `exploration/scripts/assemble_all.py` | HDF5 → Zarr (Ray-parallel, geometry-aware) | Rerun for new experiments |
| `exploration/scripts/tar_assembled.sh` | Package Zarr → tarballs by run | Rerun per ORNL shipment |
| `scripts/generate_manifests.py` | Scan Zarr, compute stats, output Parquet | Rerun when data changes |
| `scripts/ingest_all.py` | Bulk-insert manifests into Tiled catalog | Rerun after manifest regen |
| `scripts/explore_catalog.py` | Marimo notebook for catalog browsing | Ongoing tool |
| `data/broker/sfx_zarr_adapter.py` | Custom Tiled adapter for Zarr groups | Infrastructure |
| `exploration/data_pipeline/*.py` | PyTorch dataset classes | Reusable for training |

## Gray area

| Script | Situation |
|--------|-----------|
| `exploration/scripts/build_manifest.py` | Superseded, but targets raw HDF5 (different from generate_manifests.py which targets assembled Zarr) |
