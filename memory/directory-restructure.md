# Directory Restructure

Status: **Implemented**

## Layout

```
proj-lrn091/
├── pipeline/                          # 6-step reusable onboarding pipeline
│   ├── create_symlinks.sh             # Step 1: link raw data (--config, --dry-run)
│   ├── symlinks.yml                   # Config: 13 experiment symlink entries
│   ├── build_manifest.py              # Step 2: discover & validate HDF5 (--data-dir, --output, --detector-map)
│   ├── experiment_detector_map.yml    # Config: experiment-to-detector mapping
│   ├── assemble_all.py                # Step 3: HDF5 → Zarr (Ray-parallel)
│   ├── tar_assembled.sh               # Step 4: Zarr → tarballs
│   ├── generate_manifests.py          # Step 5: Zarr → Parquet manifests
│   └── ingest_all.py                  # Step 6: Parquet → Tiled catalog
├── training/                          # PyTorch data loading
│   ├── __init__.py
│   ├── assembled_dataset.py
│   ├── panel_dataset.py
│   ├── transforms.py
│   └── manifest.py
├── notebooks/                         # Interactive tools
│   └── explore_catalog.py             # Marimo notebook
├── archive/                           # One-off scripts (reference only)
│   ├── explore_hdf5.py
│   ├── visualize_samples.py
│   ├── test_assembly.py
│   ├── test_dataloader.py
│   └── check_class_distribution.py
├── data/broker/                       # Tiled server + catalog (unchanged)
├── docs/                              # Design docs (unchanged)
├── externals/                         # tiled-catalog-broker (unchanged)
├── exploration/                       # README.md only (historical)
└── memory/                            # Project memory
```
