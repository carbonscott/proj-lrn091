# Data Assembly Pipeline

Date: 2026-02-20

## Overview

Converts raw stacked-panel HDF5 frames into geometrically assembled Zarr stores,
suitable for model training on physically accurate detector images. Uses CrystFEL
geometry files to map raw panel pixels to their correct 2D positions.

Script: `exploration/scripts/assemble_all.py`

## How it works

1. **Geometry setup** (once per experiment): reads the `.geom` file from
   `data/geometry_registry.json`, builds a `CheetahConverter`, and pre-computes
   pixel coordinate maps (`pixel_map_row`, `pixel_map_col`).

2. **Frame assembly**: for each raw frame, converts from stacked-panel layout to
   psana panel ordering, then scatter-writes pixel values into an assembled 2D
   image using the pre-computed coordinate maps.

3. **Chunked Zarr output**: assembled frames are grouped into Zarr stores of
   `--chunk-size` frames each (default 40). One Zarr store per chunk per run.

4. **Parallelism**: runs within an experiment are dispatched in parallel via Ray.
   Each Ray worker reconstructs its own `CheetahConverter` from the geometry
   file. Pixel maps are shared across workers via `ray.put()` (zero-copy).

5. **Resume support**: if a run already has `.zarr` chunks in the output
   directory, it is skipped entirely. To re-process a run, delete its chunks
   first (see below).


## Output format

```
data/assembled/
    {exp_id}_r{run}.{chunk:04d}.zarr/
        images/                          (N, H, W) float32, chunks (1,H,W), zstd
        shared_metadata/pixel_maps/
            {exp_id}_r{run}/
                {exp_id}_r{run}          (2, H, W) int64
```

Root attributes per chunk: `experiment_id`, `detector`, `geom_file`,
`assembled_shape`, `run_number`, `chunk_index`, `num_frames`.


## Assembled image sizes

| Detector | Raw shape | Assembled shape | Experiments |
|---|---|---|---|
| Jungfrau 4M | 4096 x 1024 | ~2203 x 2299 | cxi101235425, cxil1005322, cxil1015922, cxilw5019 |
| ePix10k 2M | 5632 x 384 | varies | mfx100903824, mfxp22421, mfxx49820 |
| Jungfrau 16M | 16384 x 1024 | varies | mfx101211025 |


## Usage

### Parallel (default, uses all CPUs)

```bash
nohup uv run --with h5py --with numpy --with regex --with zarr --with ray \
    python3 -u exploration/scripts/assemble_all.py \
    --experiments cxil1015922 mfx101211025 \
    > assemble_medium.log 2>&1 &
```

### Explicit worker count

```bash
uv run --with h5py --with numpy --with regex --with zarr --with ray \
    python3 -u exploration/scripts/assemble_all.py \
    --experiments mfx100903824 --num-workers 8
```

### Sequential (no Ray, for debugging)

```bash
uv run --with h5py --with numpy --with regex --with zarr --with ray \
    python3 -u exploration/scripts/assemble_all.py \
    --experiments mfx100903824 --num-workers 1
```

### All experiments

```bash
uv run --with h5py --with numpy --with regex --with zarr --with ray \
    python3 -u exploration/scripts/assemble_all.py
```


## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--experiments` | all in registry | Space-separated experiment IDs to process |
| `--output-dir` | `data/assembled` | Output directory for Zarr stores |
| `--chunk-size` | 40 | Number of frames per Zarr chunk |
| `--num-workers` | all CPUs | Ray workers for parallel run processing (1 = sequential) |


## Resuming after interruption

The script skips runs that already have `.zarr` chunks in the output directory.
If interrupted mid-run, the last run's chunks may be incomplete. Delete them
before re-running:

```bash
# Example: run 0037 of cxil1015922 was interrupted
rm -rf data/assembled/cxil1015922_r0037.*.zarr

# Re-run; completed runs (0033-0036) will be skipped automatically
nohup uv run --with h5py --with numpy --with regex --with zarr --with ray \
    python3 -u exploration/scripts/assemble_all.py \
    --experiments cxil1015922 \
    > assemble_medium.log 2>&1 &
```


## Performance

Sequential throughput observed: ~4.7 fps (Jungfrau 4M, 2203x2299 assembled).
With Ray parallelism across runs, throughput scales roughly linearly with worker
count up to the I/O bandwidth limit.


## Dependencies

- `crystfel_stream_parser` (from `~/codes/crystfel_stream_parser/`)
- Python packages (managed by `uv run --with`): h5py, numpy, regex, zarr, ray
- Data inputs: `data/geometry_registry.json`, `data/manifest.json`, raw HDF5 files
