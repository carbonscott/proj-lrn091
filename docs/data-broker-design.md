# Data Broker Design for SFX Training Data

Date: 2026-02-26


## Why a data broker before model training

We have ~420K curated frames across 7 experiments, 3 detector types, and 33
runs stored as 10,528 Zarr chunks, plus ~19K labeled frames from peaknet10k.
Before feeding any of this into a model, we need to answer: *what data are we
actually training on?*

Without a broker, training data selection happens ad-hoc — hardcoded paths,
manual experiment lists, implicit filtering. This creates problems:

- **Reproducibility**: which frames went into training run X?
- **Metadata-driven selection**: "give me all Jungfrau 4M frames with >15
  peaks" requires scanning thousands of Zarr files.
- **Provenance**: when we add new experiments or re-curate existing ones, how do
  we track what changed?

The data broker gives us a queryable catalog with standardized metadata at every
level. We register data once, then query by any metadata field — experiment,
detector, sample, peak count, quality class — without touching the filesystem.


## The tiled-catalog-broker

**Path**: `$DATA_BROKER_DIR` (`externals/data-broker/tiled-catalog-broker`)

A config-driven system built on [Tiled](https://blueskyproject.io/tiled/) that
registers scientific datasets into a queryable HTTP catalog backed by SQLite.

**Core contract**: data providers describe their data with two Parquet manifest
files — one for entities (the queryable units) and one for artifacts (the data
arrays). The broker reads these generically: all non-standard columns become
queryable metadata automatically, with no hardcoded field names.

**Dual access modes**:

| Mode | How | Speed | Best for |
|------|-----|-------|----------|
| **A (Expert)** | Extract `(file, dataset, index)` locator from entity metadata, open with h5py/zarr | ~2ms/artifact | ML training (bulk loading) |
| **B (HTTP)** | `client["Dataset"]["Entity"]["Artifact"][:]` via Tiled server | ~60ms/artifact | Interactive exploration, slicing |

**Registration pipeline**: write a manifest generator → create dataset YAML
config → `broker-generate` (produces Parquet) → `broker-ingest` (loads into
SQLite catalog) → `tiled serve config` (starts HTTP server).

**Reference deployment**: `$DATA_BROKER_EXAMPLE_DIR` serves 6 datasets (27K
entities, 166K artifacts, 151 GB) for the MAIQMag project with the same broker.


## The four-level hierarchy

```
Root                                              Level 0 (implicit)
├── Dataset_A/                                    Level 1 — discovery scope
│   metadata: {data_type, material, detector, ...}
│   ├── Entity_001/                               Level 2 — queryable unit
│   │   metadata: {physics params, locators, ...}
│   │   ├── artifact_a                            Level 3 — data array (leaf)
│   │   └── artifact_b
│   └── Entity_002/ ...
└── Dataset_B/ ...
```

### Design rationale at each level

**Root (Level 0)**: entry point. Queries here scan only dataset containers (few
rows), enabling cross-dataset discovery like "find all experimental datasets" or
"find all Jungfrau data".

**Dataset (Level 1)**: a logically distinct collection of data. The key design
principle from the broker docs: *"Promote a descriptor to structure only when it
partitions the data into roughly equal, stable groups that users routinely
navigate by."* This level carries discovery metadata — what kind of data, what
material, what facility.

**Entity (Level 2)**: the queryable unit. This is where all the science metadata
lives — the parameters you filter on when selecting training data. In the
MAIQMag example, each entity is one Hamiltonian (simulation) or one experimental
run.

**Artifact (Level 3)**: a data array. The terminal leaf. Each artifact has a
self-contained locator `(file, dataset, index)` that lets you load it directly
without knowing the storage layout.


## Data model decision: Dataset = run

### Why not experiment?

In LCLS experiments, conditions can change between runs within the same
experiment — scientists may switch detectors, toggle laser on/off, or introduce
different perturbations from run to run. Grouping by experiment would mix
heterogeneous data under one container, defeating the purpose of scoped queries.

### Why run?

Each run has stable, uniform conditions throughout:

- Same detector
- Same laser on/off state
- Same sample injection setup
- Same photon energy

This makes run the natural unit for the dataset level — it is the finest
granularity at which experimental conditions are guaranteed to be homogeneous.

### Data inventory

**Assembled SFX data** (`data/assembled/`): 10,528 Zarr v3 chunks

| Experiment | Instrument | Detector | Runs | Chunks | ~Frames |
|------------|-----------|----------|------|--------|---------|
| cxi101235425 | CXI | jungfrau_4m | 3 (r0100, r0105, r0106) | 808 | 32K |
| cxil1005322 | CXI | jungfrau_4m | 1 (r0007) | 179 | 7K |
| cxil1015922 | CXI | jungfrau_4m | 10 (r0033-r0043) | 1,323 | 53K |
| cxilw5019 | CXI | jungfrau_4m | 5 (r0017, r0022, r0025, r0079, r0084) | 2,479 | 99K |
| mfx100903824 | MFX | epix10k_2m | 1 (r0027) | 1 | 40 |
| mfx101211025 | MFX | jungfrau_16m | 5 (r0074, r0075, r0077, r0079, r0080) | 266 | 11K |
| mfxp22421 | MFX | epix10k_2m | 8 (r0017-r0025) | 5,472 | 219K |

**Peaknet10k labeled data** (`data/peaknet10k/`): 490 Zarr v3 chunks

| Experiment | Instrument | Detector | Runs | Chunks | ~Frames |
|------------|-----------|----------|------|--------|---------|
| mfxl1025422 | MFX | epix10k_2m | 5 (r0309-r0313) | 179 | ~7K |
| mfxl1027522 | MFX | epix10k_2m | 5 (r0026-r0030) | 300 | ~12K |
| mfx13016 | MFX | epix10k_2m | 1 (0036) | 11 | ~440 |

Key difference: peaknet10k chunks include `labels/` arrays (peak segmentation
masks, int64) and per-frame peak metadata (good_peaks, bad_peaks, fit
parameters) in root attributes.

**Totals**: 44 runs → 44 datasets, ~11,018 Zarr chunks, ~440K frames.


## Data model: Dataset = run, Entity = frame

Each frame is an individually queryable entity. The broker's `index` field in
the artifact manifest supports selecting individual slices from multi-frame
Zarr chunks, so we register one entity per frame even though frames are stored
in batched Zarr files (typically 40 frames per chunk).

Per-frame statistics (mean, max, std, fraction_zero) are computed during
manifest generation. For peaknet10k runs, peak counts (`npeaks`) are also
extracted from the per-frame metadata stored in Zarr root attributes.

```
Root
├── cxilw5019_r0017/                              Dataset (run)
│   metadata: {experiment_id: "cxilw5019", run_number: 17,
│              detector: "jungfrau_4m", instrument: "CXI",
│              sample_name: "GPCR SFX in LCP", pi: "Cherezov",
│              data_type: "assembled", assembled_shape: [2203, 2299],
│              num_frames: 47560, num_chunks: 1189}
│   │
│   ├── f_cxilw5019_r0017_000000/                Entity (one per frame)
│   │   metadata: {frame_index: 0, chunk_file: "cxilw5019_r0017.0000.zarr",
│   │              chunk_frame_index: 0, mean_intensity: 12.3,
│   │              max_intensity: 8542.0, std_intensity: 45.6,
│   │              fraction_zero: 0.42, npeaks: -1}
│   │   │
│   │   └── image                                 Artifact (H, W) via index
│   │
│   ├── f_cxilw5019_r0017_000001/ ...
│   └── ... (~47K entities for this run)
│
├── mfxl1027522_r0029/                            Dataset (peaknet run)
│   metadata: {experiment_id: "mfxl1027522", run_number: 29,
│              detector: "epix10k_2m", instrument: "MFX",
│              data_type: "peaknet_labeled", assembled_shape: [1667, 1665],
│              num_frames: 3080, num_chunks: 77}
│   │
│   ├── f_mfxl1027522_r0029_000000/              Entity
│   │   metadata: {frame_index: 0, ..., npeaks: 23}
│   │   │
│   │   ├── image                                 Artifact (H, W)
│   │   └── label                                 Artifact (peak mask, H, W)
│   │
│   └── ... (~3K entities)
│
└── ... (44 datasets total)
```

**Totals**: 44 datasets, ~440K entities, ~440K–880K artifacts (assembled runs
have 1 artifact/entity; peaknet runs have 2 — image + label).

**What this enables**:

```python
from tiled.client import from_uri
from tiled.queries import Key

client = from_uri("http://localhost:8007", api_key="secret")

# Run-level queries (scan 44 dataset containers, sub-second)
jf4m = client.search(Key("detector") == "jungfrau_4m")
labeled = client.search(Key("data_type") == "peaknet_labeled")
cherezov = client.search(Key("pi") == "Cherezov")

# Frame-level queries (within a dataset)
run = client["cxilw5019_r0017"]
bright = run.search(Key("max_intensity") > 5000)
sparse = run.search(Key("fraction_zero") > 0.8)

# Direct loading via locator (Mode A, ~2ms)
entity = run["f_cxilw5019_r0017_000042"]
# entity.metadata gives {chunk_file, chunk_frame_index, ...}
# → open zarr, read images[chunk_frame_index]
```

### Why entity = frame (not run)

- **Frame-level selection is the core use case**: ML training needs filtering
  by per-frame statistics, not just run-level properties.
- **The broker supports batched-file indexing**: the `index` column in the
  artifact manifest selects individual slices from multi-frame Zarr chunks, so
  no data restructuring is needed.
- **Per-frame stats are computed during manifest generation**: mean, max, std,
  fraction_zero, and npeaks are available from day one.
- **Chunks are transparent**: they are a storage detail. Entity metadata carries
  `chunk_file` + `chunk_frame_index` as a locator through the chunk to the
  individual frame.


## Metadata fields

### Dataset-level (Level 1)

| Field | Type | Example | Purpose |
|-------|------|---------|---------|
| `experiment_id` | string | `"cxilw5019"` | LCLS experiment identifier |
| `run_number` | int | `17` | Run number within experiment |
| `instrument` | string | `"CXI"` | LCLS instrument/hutch |
| `detector` | string | `"jungfrau_4m"` | Detector model |
| `sample_name` | string | `"GPCR SFX in LCP"` | Sample description |
| `pi` | string | `"Cherezov"` | Principal investigator |
| `proposal_id` | string | `"LW50"` | LCLS proposal number |
| `data_type` | string | `"assembled"` or `"peaknet_labeled"` | Distinguishes labeled vs unlabeled data |
| `assembled_shape` | list[int] | `[2203, 2299]` | Image dimensions |
| `num_frames` | int | `47560` | Total frames in the run |
| `num_chunks` | int | `1189` | Number of Zarr chunks |

### Entity-level (Level 2) — one per frame

| Field | Type | Purpose |
|-------|------|---------|
| `frame_index` | int | Global frame index within the run |
| `chunk_file` | string | Zarr chunk filename (locator) |
| `chunk_frame_index` | int | Frame index within the chunk (locator) |
| `mean_intensity` | float | Frame mean pixel value |
| `max_intensity` | float | Frame max pixel value |
| `std_intensity` | float | Frame standard deviation |
| `fraction_zero` | float | Fraction of zero-valued pixels |
| `npeaks` | int | Peak count from peaknet metadata (-1 if unavailable) |

### Artifact-level (Level 3) — one or two per frame

| Artifact | Description | Shape | Present in |
|----------|-------------|-------|------------|
| `image` | Single assembled diffraction frame | `(H, W)` float32 | All runs |
| `label` | Peak segmentation mask | `(H, W)` int64 | peaknet10k only |

Artifacts reference multi-frame Zarr chunks via the `(file, dataset, index)`
locator pattern. For example: `file=cxilw5019_r0017.0042.zarr`,
`dataset=images`, `index=15` selects `images[15]` from that chunk.


## Implementation

### Scripts

- **`scripts/generate_manifests.py`** — scans Zarr directories, computes
  per-frame stats, outputs Parquet manifests + dataset YAML configs.
  Supports Ray parallelization for processing ~440K frames.
- **`scripts/ingest_all.py`** — reads pre-generated manifests and bulk-inserts
  into the SQLite catalog using the broker's `bulk_register` module.

### Broker modifications

- **`broker/utils.py`** — `get_artifact_shape()` extended to handle Zarr
  stores alongside HDF5. Cache key changed from `dataset_path` to
  `(file_path, dataset_path)` to avoid shape collisions across detectors.

### Output structure

```
data/broker/
├── config.yml              # Tiled server config (port 8007)
├── catalog.db              # SQLite catalog (after ingestion)
├── manifests/              # Per-run Parquet files (44 × 2 = 88 files)
│   ├── {run_key}_entities.parquet
│   └── {run_key}_artifacts.parquet
└── datasets/               # Per-run YAML configs (44 files)
    └── {run_key}.yaml
```

### Running

```bash
# Generate manifests (on a Milano compute node):
uv run --with zarr --with numpy --with pandas --with pyarrow --with ray \
    --with 'ruamel.yaml' python scripts/generate_manifests.py

# Ingest into catalog:
uv run --with 'tiled[server]' --with pandas --with pyarrow --with h5py \
    --with zarr --with numpy --with 'ruamel.yaml' --with canonicaljson \
    --with sqlalchemy python scripts/ingest_all.py

# Serve:
cd data/broker && uv run --with 'tiled[server]' tiled serve config config.yml --api-key secret
```


## Open questions

1. **Additional metadata extraction.** The raw HDF5 source files contain
   `/LCLS/photon_energy_eV`, `/LCLS/timestamp`, and
   `/entry_1/result_1/nPeaks`. These are not in the assembled Zarr stores.
   Could be extracted and added as entity metadata in a future enrichment pass.

2. **Curation pipeline integration.** The data curation strategy
   (`data-curation-strategy.md`) proposes a three-class taxonomy. Once frames
   are classified, those labels should flow back into the broker as entity
   metadata — enabling queries like "give me all class-3 frames from Jungfrau
   4M runs." The broker supports re-registration to add new metadata fields.
