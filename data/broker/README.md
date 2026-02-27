# Tiled Data Broker

## Starting the Server

```bash
cd data/broker
PYTHONPATH=. uv run --with 'tiled[server]' tiled serve config config.yml --api-key secret
```

`PYTHONPATH=.` is required so Tiled can import the custom adapter (`sfx_zarr_adapter.py`).

## Custom Adapter

`sfx_zarr_adapter.py` — Handles `application/x-zarr` assets registered with
`dataset` and `slice` parameters. The built-in Zarr adapter ignores these
parameters and crashes on Zarr groups. This adapter navigates into the group
(e.g. `group["images"][0]`) before handing the array to Tiled.
