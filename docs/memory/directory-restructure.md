# Directory Restructure

Status: **Implemented**

## Layout

```
proj-lrn091/
├── data-onboard/                      # All data pipeline code
│   ├── create_symlinks.sh             # Step 1 (--config, --dry-run)
│   ├── symlinks.yml
│   ├── build_manifest.py              # Step 2 (--data-dir, --output, --detector-map)
│   ├── experiment_detector_map.yml
│   ├── assemble_all.py                # Step 3 (HDF5 → Zarr)
│   ├── tar_assembled.sh               # Step 4 (Zarr → tar)
│   ├── generate_manifests.py          # Step 5 (Zarr → Parquet)
│   ├── ingest_all.py                  # Step 6 (Parquet → catalog)
│   ├── dataloader/                    # PyTorch dataset classes
│   └── notebooks/                     # Marimo notebooks
├── broker/                            # Tiled server config + adapter (committed)
├── data/                              # Raw + processed data (gitignored)
│   └── broker/                        # Runtime: catalog.db, manifests/, datasets/
├── docs/                              # Design docs + memory/
├── externals/                         # tiled-catalog-broker (not in git)
├── archive/                           # One-off scripts (reference)
└── CLAUDE.md
```
