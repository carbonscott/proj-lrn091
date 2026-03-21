## Environment Setup

Set these environment variables before running any commands:

```bash
export PEAKNET_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/peaknet
export PEAKNET_PIPELINE_CODE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/peaknet-pipeline-ray
export PEAKNET_PIPELINE_PROJ_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/exp-peaknet-pipeline-ray
export PROJ_PEAKNET_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/exp-peaknet
export MAXIE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/maxie
export XTAL_DATA_ASSEMBLED=/lustre/orion/lrn091/proj-shared/data
export TILED_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/tiled
export UV_CACHE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/.UV_CACHE
export DATA_BROKER_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/tiled-catalog-broker
export LCLS_DATA_BROKER_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/lcls-data-broker
export DATA_BROKER_EXAMPLE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/MAIQMag-data-broker-examples
export DEEPLEARNING_DOCS_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deeplearning-docs
export AGENT_DOCS_DIR=/lustre/orion/lrn091/proj-shared/cwang31/proj-lrn091/docs/agents
export RESEARCH_DIR=/lustre/orion/lrn091/proj-shared/cwang31/research-lrn091
```

### Python

Use uv to run python programs. The UV_CACHE_DIR avoids repeated package downloads.

## HPC Resources

You are on OLCF's Frontier supercomputer. Frontier nodes have AMD EPYC 64-core
CPUs (56 allocatable, 8 reserved by system) and 4x AMD MI250X GPUs (8 GCDs per
node, 64 GB HBM2e each). Each node has 512 GB DDR4 memory.

**Interactive GPU Jobs**

```bash
salloc -A lrn091 -J interactive -t 2:00:00 -p batch -N 1
```

Caveat: Do NOT specify a shell (e.g., avoid `salloc ... /bin/bash`) as this
causes the job to start on a login node instead of a compute node.

**Partitions**

- `batch` — default production partition
- `extended` — for smaller long-running jobs (max 64 nodes, 24h walltime)

**Submitting jobs**

Once allocated, use `srun` to launch on compute nodes. Serial tasks do not
need `srun`. Do NOT launch parallel or threaded tasks from login nodes.

## Data Broker

- Original (MAIQMag): `/lustre/orion/lrn091/proj-shared/cwang31/deps/tiled-catalog-broker`
- LCLS/SFX fork (with Zarr v3 support): `/lustre/orion/lrn091/proj-shared/cwang31/deps/lcls-data-broker`

**Launching the broker** (from this repo):

```bash
cd data/broker
PYTHONPATH=../../broker uv run --with 'tiled[server]' \
    tiled serve config ../../broker/config.yml --api-key secret
```

Server runs on `http://127.0.0.1:8007`. The CWD must be `data/broker/`
(where `catalog.db` lives) because `config.yml` uses a relative SQLite path.

**Access modes:**

- **Mode A (Expert):** Query entity metadata for file paths, load data directly (fast, for ML pipelines)
- **Mode B (Visualizer):** Browse `dataset → entity → artifact` as Tiled children via HTTP; arrays served chunked. Client example:
  ```python
  from tiled.client import from_uri
  c = from_uri("http://127.0.0.1:8007", api_key="secret")
  arr = c["mfx100903824_r0027"]["f_mfx100903824_r0027_000000"]["image"].read()
  ```

## Research

- Research repo: `$RESEARCH_DIR` (`/lustre/orion/lrn091/proj-shared/cwang31/research-lrn091`)
  - `phase0/data-exploration` — Zarr loader + exploratory analysis (85K frames, 4 experiments)
  - `phase0/data-loading` — Manifest-based PyTorch Dataset + transforms
- Broker catalog: `data/broker/catalog.db` (104K frames, 30 runs, 7 experiments)
- Manifest guide: `$RESEARCH_DIR/docs/manifest-guide.md` — how to query catalog and export training manifests
- Playbook: `docs/agents/README-PLAYBOOK.md` — entry point for MAXIE training documentation

## Agent Documentation

- `docs/agents/progress.md` — Running log of work done by agents. Update only
  when the user explicitly asks to "document our progress". Git-tracked.
- `docs/agents/notes.md` — Technical notes, caveats, and improvement ideas
  discovered during agent work. Git-tracked.
