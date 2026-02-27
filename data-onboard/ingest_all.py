"""
Ingest all SFX manifests into the Tiled catalog.

Reads pre-generated Parquet manifests and dataset YAML configs, then
bulk-inserts into a SQLite catalog using the broker's bulk_register module.

Usage:
    # Ingest all runs:
    uv run --with 'tiled[server]' --with pandas --with pyarrow --with h5py \
        --with zarr --with numpy --with 'ruamel.yaml' --with canonicaljson \
        --with sqlalchemy \
        python scripts/ingest_all.py

    # Ingest specific runs:
    uv run --with 'tiled[server]' --with pandas --with pyarrow --with h5py \
        --with zarr --with numpy --with 'ruamel.yaml' --with canonicaljson \
        --with sqlalchemy \
        python scripts/ingest_all.py --runs cxilw5019_r0017 mfx100903824_r0027

    # Fresh database (delete existing):
    uv run ... python scripts/ingest_all.py --fresh
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add the broker code to the Python path
BROKER_DIR = Path(__file__).resolve().parent.parent / "externals" / "data-broker" / "tiled-catalog-broker" / "tiled_poc"
sys.path.insert(0, str(BROKER_DIR))

PROJECT_DIR = Path(__file__).resolve().parent.parent
BROKER_DATA_DIR = PROJECT_DIR / "data" / "broker"
MANIFESTS_DIR = BROKER_DATA_DIR / "manifests"
DATASETS_DIR = BROKER_DATA_DIR / "datasets"
CATALOG_DB = BROKER_DATA_DIR / "catalog.db"


def load_dataset_config(yaml_path):
    """Load a dataset YAML config."""
    from ruamel.yaml import YAML
    yaml = YAML()
    with open(yaml_path) as f:
        return yaml.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest SFX manifests into the Tiled catalog."
    )
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Specific run keys to ingest. Default: all available manifests.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete existing catalog.db and start fresh.",
    )
    parser.add_argument(
        "--catalog-db", type=Path, default=CATALOG_DB,
        help="Path to the SQLite catalog database.",
    )
    args = parser.parse_args()

    from broker.bulk_register import (
        init_database, prepare_node_data, bulk_register,
    )
    from broker.config import get_base_dir

    # Discover available manifests
    available_runs = []
    for f in sorted(MANIFESTS_DIR.glob("*_entities.parquet")):
        run_key = f.name.replace("_entities.parquet", "")
        available_runs.append(run_key)

    if args.runs:
        runs_to_ingest = [r for r in args.runs if r in available_runs]
        missing = set(args.runs) - set(runs_to_ingest)
        if missing:
            print(f"Warning: manifests not found for: {missing}")
    else:
        runs_to_ingest = available_runs

    if not runs_to_ingest:
        print("No manifests to ingest.")
        return

    print(f"Ingesting {len(runs_to_ingest)} runs into {args.catalog_db}")

    # Initialize database
    if args.fresh or not args.catalog_db.exists():
        print("\nInitializing fresh database...")
        # Set up minimal config for the broker's init_database
        os.environ.setdefault("TILED_CATALOG_DB", str(args.catalog_db))

        from sqlalchemy import create_engine
        from tiled.catalog import from_uri as catalog_from_uri

        if args.catalog_db.exists():
            args.catalog_db.unlink()

        readable_storage = [
            str(PROJECT_DIR / "data" / "assembled"),
            str(PROJECT_DIR / "data" / "peaknet10k"),
        ]

        catalog_from_uri(
            f"sqlite:///{args.catalog_db}",
            writable_storage=str(BROKER_DATA_DIR / "storage"),
            readable_storage=readable_storage,
            init_if_not_exists=True,
        )
        print(f"  Created: {args.catalog_db}")

    # Get engine for bulk operations
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{args.catalog_db}")

    # Ingest each run
    t_start = time.time()
    total_entities = 0
    total_artifacts = 0

    for run_key in runs_to_ingest:
        ent_path = MANIFESTS_DIR / f"{run_key}_entities.parquet"
        art_path = MANIFESTS_DIR / f"{run_key}_artifacts.parquet"
        config_path = DATASETS_DIR / f"{run_key}.yaml"

        if not config_path.exists():
            print(f"  Skipping {run_key}: no dataset config found")
            continue

        config = load_dataset_config(config_path)
        dataset_key = config["key"]
        dataset_metadata = dict(config.get("metadata", {}))
        base_dir = config["base_dir"]

        ent_df = pd.read_parquet(ent_path)
        art_df = pd.read_parquet(art_path)

        print(f"\n{'='*60}")
        print(f"Ingesting {run_key}: {len(ent_df)} entities, {len(art_df)} artifacts")

        # Prepare node data
        ent_nodes, art_nodes, art_data_sources = prepare_node_data(
            ent_df, art_df, max_entities=len(ent_df), base_dir=base_dir
        )

        # Bulk register
        bulk_register(
            engine, ent_nodes, art_nodes, art_data_sources,
            dataset_key=dataset_key,
            dataset_metadata=dataset_metadata,
        )

        total_entities += len(ent_df)
        total_artifacts += len(art_df)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Runs:      {len(runs_to_ingest)}")
    print(f"  Entities:  {total_entities:,}")
    print(f"  Artifacts: {total_artifacts:,}")
    print(f"  Catalog:   {args.catalog_db}")


if __name__ == "__main__":
    main()
