# Tiled Data Broker

## Starting the Server

```bash
cd data/broker
PYTHONPATH=../../broker uv run --with 'tiled[server]' \
    tiled serve config ../../broker/config.yml --api-key secret
```

The server runs from `data/broker/` (where `catalog.db` and runtime files
live). `PYTHONPATH=../../broker` lets Tiled import the custom adapter.

## File Layout

- `broker/config.yml` — Tiled server config (committed to git)
- `broker/sfx_zarr_adapter.py` — Custom adapter (committed to git)
- `data/broker/catalog.db` — SQLite catalog (runtime, gitignored)
- `data/broker/manifests/` — Parquet manifests (runtime, gitignored)
- `data/broker/datasets/` — Dataset YAML configs (runtime, gitignored)

## Custom Adapter

`sfx_zarr_adapter.py` — Handles `application/x-zarr` assets registered with
`dataset` and `slice` parameters. The built-in Zarr adapter ignores these
parameters and crashes on Zarr groups. This adapter navigates into the group
(e.g. `group["images"][0]`) before handing the array to Tiled.
