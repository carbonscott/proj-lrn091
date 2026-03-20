# Data Pipeline for MAXIE on Frontier

Date: 2026-03-19
Context: ALCC project Group 1 -- MAXIE foundation model (ViT-MAE for X-ray
diffraction). This document covers the end-to-end data pipeline: raw detector
images to model input tensors, data loading at HPC scale, and format
considerations for Frontier.

Companion documents:
- `training-playbook-for-maxie.md` -- optimizer, LR, loss, augmentation
- `research-loop-frontier-strategy.md` -- Frontier launch patterns, sbcast, RCCL
- `ssl-candidates-for-maxie.md` -- SSL method selection
- `research-loop-brainstorm.md` -- design space exploration

---

## 1. Current MAXIE Data Pipeline

### 1.1 Dataset classes

MAXIE provides several dataset implementations in `$MAXIE_DIR/maxie/datasets/`:

| Class | File | Pattern | Status |
|-------|------|---------|--------|
| `IPCDistributedSegmentedDataset` | `ipc_segmented_dataset_dist.py` | IPC socket to psana server; segmented for distributed training | **Active** -- used by `train.fsdp.py` |
| `IPCDistributedSegmentedDataset` | `ipc_dataset_dist.py` | Earlier version; flat list instead of JSON-driven round-robin | Legacy |
| `StreamingDataset` | `streaming_dataset.py` | IterableDataset pulling via pynng sockets; multi-node queue per node | Alternative streaming path |
| `DistributedZarrDataset` | `zarr_dataset.py` | Reads directly from Zarr stores on disk via Parquet manifest | **Zarr path** -- reads from `$XTAL_DATA_ASSEMBLED` |
| `PsanaDataset` | `psana_dataset.py` | Direct psana access with lazy init; for LCLS-local use | LCLS only |
| `RemotePsanaDataset` | `remote_psana_dataset.py` | HTTP+msgpack to remote psana server | Prototype |
| `DummyImageData` / `DistributedSegmentedDummyImageData` | `dummy_dataset.py` | Random tensors for throughput testing | Testing |

### 1.2 Active training path: IPC segmented dataset

The production training script (`$MAXIE_DIR/train/train.fsdp.py`) uses
`IPCDistributedSegmentedDataset`. The data flow is:

```
JSON config (exp, run, detector, events)
    |
    v
IPCDistributedSegmentedDataset.__getitem__(idx)
    |-- Looks up (exp, run, access_mode, detector_name, event) from round-robin generator
    |-- Sends socket request to IPC server (TCP, localhost:5000)
    |-- IPC server (server.ipc.py) calls psana to read XTC data
    |-- Server returns assembled image via shared memory (numpy ndarray)
    |-- Dataset applies pre_transforms (Pad to 2048x2048)
    |-- Returns tensor of shape (B=1, C=1, H, W)
    v
DataLoader (DistributedSampler, custom_collate filters None)
    |
    v
Training loop applies transforms (Pad, PolarCenterCrop, MergeBatchPatchDims,
    BatchSampler, RandomPatch, RandomRotate, RandomShift)
    |
    v
ViTMAEForPreTraining.forward() -- expects (B, C=1, 224, 224)
```

Key observations:
- The psana IPC server is a **separate process** that must be launched alongside
  training. It caches `PsanaImg` objects per (exp, run) and serves assembled
  images via TCP socket + shared memory.
- The `entry_per_cycle` parameter controls round-robin interleaving across
  experiments (default: 1 = strict round-robin).
- Segmented loading: the dataset is divided into segments of size
  `seg_size * world_size`. Each segment is loaded, a `DistributedSampler`
  distributes indices, and training proceeds. After exhausting a segment, the
  next segment is loaded.

### 1.3 Zarr dataset path (for Frontier)

`DistributedZarrDataset` reads directly from Zarr stores using a Parquet
manifest. This is the path designed for Frontier, where psana is not available.

```
Parquet manifest (absolute_path, shape per Zarr store)
    |
    v
DistributedZarrDataset.__init__()
    |-- Reads Parquet -> cumulative sizes -> total_size
    |-- Global shuffle via torch.Generator (seeded, reproducible)
    |-- Segment management (start_idx, end_idx, global_seg_size)
    |
    v
__getitem__(idx) -> _fetch_image(original_idx)
    |-- Maps segment idx -> shuffled global idx -> (file_idx, zarr_idx)
    |-- Opens Zarr store (LRU cache, default 100 stores)
    |-- Reads z['data'][zarr_idx] -> numpy (H, W)
    |-- Converts to tensor (1, C=1, H, W)
    |-- Applies transforms
    |-- Returns (C, H, W)
```

### 1.4 Transforms pipeline

Defined in `$MAXIE_DIR/maxie/tensor_transforms.py`:

| Transform | Purpose | Operates on |
|-----------|---------|-------------|
| `Pad(H_pad, W_pad)` | Zero-pad image to target size (e.g. 2048x2048) | (B, C, H, W) |
| `Norm(detector_norm_params)` | Per-detector z-score normalization using precomputed mean/std | (B, C, H, W) |
| `InstanceNorm(eps)` | Per-image zero-mean unit-variance normalization | (B, C, H, W) |
| `DownscaleLocalMean(factors)` | Average pooling downscale (e.g. 2x2) | (B, C, H, W) |
| `PolarCenterCrop(Hv, Wv, sigma, num_crop)` | Random polar-coordinate crop around image center | (B, C, H, W) -> (B, N, C, Hv, Wv) |
| `MergeBatchPatchDims` | Reshape (B, N, C, H, W) -> (B*N, C, H, W) | After PolarCenterCrop |
| `BatchSampler(sampling_fraction)` | Subsample along batch dim (for crop diversity) | (B, ...) |
| `Patchify(patch_size, stride)` | Unfold into non-overlapping patches | (B, C, H, W) -> (B, N, C, ps, ps) |
| `RandomPatch(num_patch, H_patch, W_patch)` | Random rectangular zero-masking (data augmentation) | (B, C, H, W) |
| `RandomRotate(angle_max)` | Random rotation (bilinear interpolation) | (B, C, H, W) |
| `RandomShift(frac_y_max, frac_x_max)` | Random translation (zero-fill boundary) | (B, C, H, W) |
| `NoTransform` | Identity (placeholder when transform is disabled) | Any |

The training script applies transforms in two phases:
1. **pre_transforms** (inside dataset `__getitem__`): `Pad(2048, 2048)`
2. **transforms** (in training loop, after DataLoader): `Pad` -> `PolarCenterCrop`
   -> `MergeBatchPatchDims` -> `BatchSampler` -> `RandomPatch` -> `RandomRotate`
   -> `RandomShift`

Note: `Norm` is currently **commented out** in `train.fsdp.py` (line 377:
`## Norm(detector_norm_params)`). The `DownscaleLocalMean` and `Patchify`
transforms are also commented out.

### 1.5 Hydra config defaults (dataset section)

Source: `$MAXIE_DIR/train/hydra_config/train_config/base.yaml`

```yaml
dataset:
  batch_size:      1
  num_workers:     2
  seg_size:        4
  entry_per_cycle: 1
  pin_memory:      true     # (not in base.yaml, but used in code)
  prefetch_factor: null     # (not in base.yaml, but used in code)
  transforms:
    norm:
      Rayonix:     { mean: 116.92, std: 22.89 }
      epix10k2M:   { mean: 46.6,   std: 98.3  }
      jungfrau4M:  { mean: 593.17, std: 204.13 }
    H_pad: 2048
    W_pad: 2048
    num_patch: 100
    size_patch: 20
    angle_max: 360
    frac_shift_max: 0.1
    downscale_factors: [2, 2]
    var_size_patch: 0.2
    patch_size: 224
    stride: 224
```

The ViT model config specifies `image_size: 224`, `patch_size: 16`,
`num_channels: 1`, confirming the model expects single-channel 224x224 input.

---

## 2. Data Sources and Formats

### 2.1 LCLS serial femtosecond crystallography (SFX) data

The primary data source is LCLS beamline experiments. Each experiment is
identified by an experiment ID (e.g., `cxi101235425`), run number, and detector.

**Detectors in the dataset:**

| Detector | Typical resolution | Norm mean | Norm std |
|----------|-------------------|-----------|----------|
| Rayonix | varies | 116.92 | 22.89 |
| epix10k2M | ~2M pixels | 46.6 | 98.3 |
| jungfrau4M | 2203 x 2299 (assembled) | 593.17 | 204.13 |

### 2.2 Data on Lustre ($XTAL_DATA_ASSEMBLED)

Location: `/lustre/orion/lrn091/proj-shared/data/`

Contents (as of 2026-03-19):
- **2621 Zarr v3 stores** (`.zarr` directories)
- **21 tar archives** (`.tar` files)
- Naming convention: `{experiment_id}_r{run:04d}.{chunk_index:04d}.zarr`

**Experiments present:**

| Experiment ID | Instrument | Detector | Approximate stores |
|---------------|-----------|----------|-------------------|
| cxi101235425 | CXI | jungfrau4M | ~2600 |
| cxil1005322 | CXI | unknown | ~1 |
| cxil1015922 | CXI | unknown | ~10+ |
| mfx100903824 | MFX | unknown | ~1 |
| mfx101211025 | MFX | unknown | ~5 |
| mfxl1025422 | MFX | unknown | small |
| mfxl1027522 | MFX | unknown | small |

The bulk of the data is from experiment `cxi101235425` with jungfrau4M detector.

### 2.3 Zarr v3 store structure

Each Zarr store (e.g., `cxi101235425_r0100.0000.zarr/`) has:

```
zarr.json                     # Group metadata (zarr_format: 3)
  attributes:
    experiment_id: "cxi101235425"
    detector: "jungfrau_4m"
    assembled_shape: [2203, 2299]
    run_number: "0100"
    chunk_index: 0
    num_frames: 40
images/
  zarr.json                   # Array metadata
    shape: [40, 2203, 2299]   # (num_frames, H, W)
    data_type: float32
    chunk_grid: regular, chunk_shape: [1, 2203, 2299]
    codecs: bytes(little-endian) + zstd(level=3)
  c/                          # Chunk directory
    0/, 1/, 2/, ...           # Individual frame chunks
shared_metadata/              # (if present)
```

Key facts:
- **Float32** data type (assembled, calibrated images)
- **Per-frame chunking**: each chunk = 1 frame = 2203 x 2299 x 4 bytes = ~20.3 MB raw
- **zstd compression** at level 3 (good compression/speed balance)
- **40 frames** per store is typical (varies)
- **Total estimated size**: 2621 stores x 40 frames x ~10 MB compressed ~ 1 TB

### 2.4 Data broker system

Two data broker implementations manage metadata and provide catalog access:

**Original (MAIQMag):** `$DATA_BROKER_DIR` (`tiled-catalog-broker`)
- Config-driven HDF5 registration into Tiled catalog
- Hierarchy: Dataset -> Entity -> Artifact
- Dual-mode: expert (file paths) or visualizer (HTTP chunked)
- Parquet manifest is the contract

**LCLS/SFX fork:** `$LCLS_DATA_BROKER_DIR` (`lcls-data-broker`)
- Adds Zarr v3 support for LCLS/SFX crystallography data
- CLI flags: `--mimetype application/x-zarr --is-directory`
- Dynamic dtype detection (reads from data files)
- Bulk ingest: `python ingest.py --mimetype application/x-zarr --is-directory datasets/*.yaml`

The broker enables Mode A access (query for file paths, load directly) which is
the pattern `DistributedZarrDataset` uses via Parquet manifests.

---

## 3. Preprocessing Strategy

### 3.1 Raw detector to assembled image

The psana pipeline (used at LCLS, implemented in `psana_utils.py`) provides
multiple read modes:

| Mode | Output | Description |
|------|--------|-------------|
| `raw` | (N_panels, H_panel, W_panel) | Raw ADU values per panel |
| `calib` | (N_panels, H_panel, W_panel) | Calibrated (pedestal-subtracted, gain-corrected) |
| `image` | (H_assembled, W_assembled) | Geometrically assembled into single image |

The IPC server (`server.ipc.py`, line 59) calls `psana_img.get_masked()` with
`returns_assemble=True, edge_width=1`, which:
1. Reads calibrated data
2. Masks edge pixels (1 pixel border set to 0)
3. Applies bad pixel mask (dead/hot/unbonded pixels set to 0)
4. Assembles panels into a single image using detector geometry

The Zarr stores on Frontier contain **pre-assembled float32 images** (already
calibrated and assembled), so steps 1-4 are already done.

### 3.2 Dead pixel masking

`PsanaImg.create_bad_pixel_mask()` creates a comprehensive mask using psana's
detector mask system:

```python
mask = detector.mask(
    run,
    calib=True,        # Calibration-derived bad pixels
    status=True,       # Pixel status from calibration DB
    edges=True,        # Detector edge pixels
    central=True,      # Central pixels (between ASICs)
    unbond=True,       # Unbonded pixels
    unbondnbrs=True,   # Neighbors of unbonded pixels
    unbondnbrs8=False  # 8-connected neighbors (disabled)
)
```

`apply_mask()` in `utils.py` sets masked pixels to 0 (or configurable value):
```python
data_masked = np.where(mask, data, mask_value)
```

**For Zarr data on Frontier:** The assembled images likely already have bad pixels
handled during assembly. However, this should be verified -- if the Zarr stores
contain raw assembled data without masking, a mask array should be stored
alongside the images and applied during loading.

### 3.3 Background subtraction

Not currently implemented in the MAXIE pipeline. For diffraction data, background
subtraction options include:

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| Static background | Subtract per-run average of non-hit frames | Simple, well-understood | Requires hit/no-hit classification |
| Radial average | Subtract azimuthal average from each frame | Removes powder rings | May remove weak Bragg peaks |
| None | Use calibrated data as-is | Preserves all signal | Background variance may dominate loss |

**Recommendation:** Start with no background subtraction. The MAE's reconstruction
loss with `norm_pix_loss=true` naturally handles per-patch mean removal, which
effectively removes local background within each 16x16 patch.

### 3.4 Dynamic range handling

Diffraction data has extreme dynamic range:
- Background pixels: 0-100 ADU
- Weak Bragg peaks: 100-1,000 ADU
- Strong Bragg peaks: 1,000-100,000+ ADU

The data is float32 in the Zarr stores. The detector norm statistics from the
config suggest typical means of 47-593 with stds of 23-204, indicating the
data is in calibrated ADU units.

---

## 4. Normalization for SSL

### 4.1 Normalization options

| Strategy | Formula | When to use |
|----------|---------|-------------|
| **Per-detector z-score** | `(x - mu_det) / sigma_det` | When mixing detectors; precomputed global stats |
| **Per-image z-score (instance norm)** | `(x - mu_img) / sigma_img` | When images vary widely in exposure/intensity |
| **Log transform** | `log(1 + x)` or `log(1 + x/x_median)` | Compresses extreme dynamic range |
| **Quantile normalization** | Map to [0, 1] via empirical CDF | Robustly handles outliers |
| **Patch-level normalization** | `(patch - mean(patch)) / std(patch)` | MAE's `norm_pix_loss` handles this in the loss |

### 4.2 Current implementation

The `Norm` class in `tensor_transforms.py` applies per-detector z-score
normalization using `torchvision.transforms.functional.normalize`:
```python
def __call__(self, img, detector_name, **kwargs):
    mean, std = self.detector_norm_params[detector_name]["mean"], ...
    return normalize(img, [mean]*C, [std]*C)
```

However, **Norm is currently commented out** in the training script. The `InstanceNorm`
class (per-image normalization) is defined but also not used.

**Precomputed statistics** (from base.yaml):
```
Rayonix:     mean=116.92, std=22.89
epix10k2M:   mean=46.6,   std=98.3
jungfrau4M:  mean=593.17, std=204.13
```

These were computed by `preprocess/get_global_stats.py`, which samples 10% of
events per run via psana and computes per-image mean/std, then aggregates.

### 4.3 Interaction with SSL methods

**MAE with `norm_pix_loss=true` (recommended):**
- The loss target is per-patch normalized: `(patch - patch.mean()) / (patch.std() + eps)`
- This means the model learns to predict relative structure within patches,
  not absolute intensity values
- **Input normalization is still important** -- it affects the encoder's
  representation even though the loss is patch-normalized
- Recommendation: apply per-detector z-score normalization at input. This puts
  all detectors on comparable scales, which matters for a foundation model
  that should generalize across detectors.

**VQ-VAE codebook:**
- The codebook learns discrete representation vectors
- Input scale directly affects codebook entry magnitudes and commitment loss
- Without normalization, codebook entries for different detectors live in different
  regions of embedding space, wasting capacity
- Recommendation: normalize inputs so all detectors produce similar activation
  distributions. Per-detector z-score is the minimum; log-transform + z-score
  may be better for the extreme dynamic range.

**I-JEPA representations:**
- Loss is MSE in the representation space (predicted vs. EMA teacher output)
- Input normalization affects both student and teacher identically (EMA)
- Less sensitive to input scale than VQ-VAE, but consistent normalization
  improves training stability
- Recommendation: same as MAE -- per-detector z-score normalization.

### 4.4 Statistics to compute from the dataset

Before full-scale training, compute and record:

| Statistic | Scope | Method | Use |
|-----------|-------|--------|-----|
| Mean, std | Per detector | Sample 10% of frames, aggregate | `Norm` transform |
| Min, max | Per detector | Full scan | Clipping / outlier detection |
| Histogram | Per detector | Bin counts over sampled frames | Verify distribution, choose log vs linear |
| Fraction of zero pixels | Per detector | Count zeros | Understand masking coverage |
| Per-run statistics | Per (experiment, run) | Mean/std per run | Detect anomalous runs |

The existing `preprocess/get_global_stats.py` computes mean/std but should be
extended for the Zarr data path on Frontier (currently only works via psana).

---

## 5. Dataset Design

### 5.1 Train/val/test splits

The current pipeline uses `generate_dataset_in_json.py` with `--train_frac 0.8`
and a seed for reproducibility. This creates an 80/20 train/eval split at the
per-experiment level, with optional shuffling.

### 5.2 Split strategies

| Strategy | How | Pros | Cons |
|----------|-----|------|------|
| **Random frame split** | Shuffle all frames, split 80/20 | Simple, maximum data usage | Frames from same run in both sets -> data leakage |
| **Random run split** | Split at run level | No within-run leakage | Run-level variation may be small |
| **Experiment holdout** | Hold out entire experiments for test | True generalization test | May lose diversity in training |
| **Stratified by detector** | Ensure each detector in train and val | Balanced evaluation per detector | More complex bookkeeping |
| **Temporal split** | Train on earlier runs, test on later | Tests temporal generalization | May not be meaningful for crystallography |

### 5.3 Recommendations

1. **Primary split: by run** -- group all frames from the same run together in
   either train or val. This prevents data leakage from temporal correlation
   within a run.

2. **Stratify by experiment and detector** -- ensure each experiment contributes
   proportionally to train and val. With the current data dominated by
   `cxi101235425` (jungfrau4M), this is critical to avoid the val set being
   entirely one experiment.

3. **Held-out experiments for generalization testing** -- reserve at least one
   complete experiment per detector type for final evaluation. This tests whether
   the foundation model generalizes to unseen experimental conditions.

4. **For scaling law studies** -- use a fixed split with a fixed seed. The
   existing `--seed 42` parameter in `generate_dataset_in_json.py` supports this.

### 5.4 Current limitation

The experiment distribution is highly skewed (cxi101235425 dominates with ~2600
Zarr stores out of 2621 total). This means:
- Training is effectively on one experiment with one detector
- Generalization to other detectors/experiments is untested
- Multi-detector normalization (`Norm` with per-detector stats) is defined but
  not exercised at scale

This should be addressed as more data becomes available.

---

## 6. Data Loading at Scale

### 6.1 Current DataLoader configuration

From `train.fsdp.py` (lines 856-871):

```python
sampler = torch.utils.data.DistributedSampler(
    dataset_train,
    shuffle=True,
    seed=base_seed,
    drop_last=drop_last_in_sampler
)
dataloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    collate_fn=custom_collate,
    drop_last=drop_last_in_loader,
    pin_memory=pin_memory,
    prefetch_factor=prefetch_factor,
)
```

The `DistributedSampler` is recreated for each segment and has `set_epoch(epoch)`
called for proper shuffling across epochs.

### 6.2 num_workers tuning for Frontier

Each Frontier GCD gets 7 CPU cores (56 cores / 8 GCDs per node). The current
default is `num_workers=2`.

| num_workers | CPU usage per GCD | I/O concurrency | Notes |
|-------------|-------------------|-----------------|-------|
| 0 | Main thread only | None | Data loading blocks training |
| 2 (current) | 2 workers + main | 2 concurrent reads | Conservative, leaves 4 cores free |
| 4 | 4 workers + main | 4 concurrent reads | Good balance for I/O-bound loads |
| 6 | 6 workers + main | 6 concurrent reads | Near maximum, may cause CPU contention |

**Recommendation:** Start with `num_workers=4` for Zarr loading on Lustre.
Profile to determine if data loading is the bottleneck (check if GPUs are idle
waiting for data). If so, increase to 5-6. Note that each DataLoader worker
forks the process and may duplicate Zarr cache memory.

### 6.3 Prefetching

`prefetch_factor` (default: `None` in base config, which means PyTorch default
of 2) controls how many batches each worker prefetches. For large images
(2203x2299 float32 = ~20 MB per frame), aggressive prefetching increases memory:

```
memory_per_worker = prefetch_factor * batch_size * frame_size
                  = 2 * 1 * 20 MB = 40 MB per worker
                  = 40 * 4 workers = 160 MB total prefetch buffer
```

This is well within the 64 GB HBM2e per GCD. Can increase `prefetch_factor` to
4-8 if data loading is the bottleneck.

### 6.4 Distributed sampler and sharding across 4096 GCDs

With `DistributedSampler(shuffle=True, drop_last=True)`:
- Each GCD sees `total_frames / world_size` frames per epoch
- No overlap between GCDs (each frame seen by exactly one GCD per epoch)
- `drop_last=True` ensures all GCDs process the same number of batches (important
  for FSDP synchronization)

For the segmented dataset pattern:
- Segment size = `seg_size * world_size` frames
- With `seg_size=4` and 4096 GCDs: 16,384 frames per segment
- With ~2621 stores x 40 frames = ~104,840 total frames
- Number of segments: `ceil(104,840 / 16,384)` = 7 segments per epoch

**Scaling consideration:** At 4096 GCDs with `seg_size=4`, each GCD sees only 4
frames per segment before synchronization. This is very fine-grained. Consider
increasing `seg_size` to 100-500 for Frontier to reduce segment switching
overhead.

### 6.5 Lustre I/O optimization

**Stripe count:** Zarr stores with per-frame chunks (~10 MB compressed) should
use moderate stripe counts. For the data directory:

```bash
# Check current striping
lfs getstripe /lustre/orion/lrn091/proj-shared/data/

# Recommended: set stripe count to 4-8 for the data directory
# (done once by the project admin)
lfs setstripe -c 8 -S 4M /lustre/orion/lrn091/proj-shared/data/
```

**Read-ahead:** The Linux kernel read-ahead can be tuned, but on Lustre the
default is usually acceptable. More important is to avoid small random reads --
Zarr's per-frame chunking already provides sequential access patterns.

**Avoid metadata storms:** At 4096 GCDs, simultaneous `zarr.open()` calls can
overwhelm the Lustre metadata server. The `DistributedZarrDataset` LRU cache
(`cache_size=100`) helps by keeping stores open. Ensure the cache is large enough
to cover all stores accessed in a segment.

### 6.6 sbcast data to NVMe

For datasets that fit on NVMe (480 GB per node on Frontier), broadcasting data
avoids Lustre contention entirely:

```bash
# In batch script
#SBATCH -C nvme

# Create tar of dataset (offline, once)
tar -cf maxie_data.tar -C /lustre/orion/lrn091/proj-shared/data/ .

# Broadcast and extract
sbcast -pf ./maxie_data.tar /mnt/bb/${USER}/data.tar
srun -N ${SLURM_NNODES} --ntasks-per-node 1 \
    tar -xf /mnt/bb/${USER}/data.tar -C /mnt/bb/${USER}/data/
```

**Feasibility:** The dataset is ~1 TB, which exceeds the 480 GB NVMe per node.
Options:
1. **Shard the data** -- each node gets a subset. Requires custom sampling to
   ensure each GCD only reads from its node's NVMe. The `StreamingDataset` with
   per-node queues could support this.
2. **Compress more aggressively** -- the Zarr stores use zstd level 3. Higher
   levels (e.g., 10-15) could reduce size by 30-50%, potentially fitting on NVMe.
3. **Subset for scaling studies** -- for early experiments with fewer frames,
   NVMe is viable and dramatically faster.

See `research-loop-frontier-strategy.md` Section 2.1 for the full sbcast setup.

---

## 7. Data Format Performance

### 7.1 HDF5 vs Zarr vs raw files

| Format | Random access | Parallel read | Compression | Lustre friendliness |
|--------|--------------|---------------|-------------|-------------------|
| **HDF5** | Per-dataset | h5py parallel mode (MPI-IO) | Multiple codecs | Single file -> hot OST |
| **Zarr v3** (current) | Per-chunk (directory) | Native (each chunk is a file) | zstd, blosc, etc. | Excellent -- chunks are files, distributed across OSTs |
| **Raw numpy** (.npy) | Per-file | Trivial | None | Good -- one file per frame |
| **TFRecord/WebDataset** | Sequential only | Shardable | Optional | Good -- sequential shards |

**Why Zarr is a good fit for Frontier:**
- Each chunk is an independent file on Lustre, naturally distributed across OSTs
- No file locking (read-only after creation)
- Metadata is lightweight JSON (no HDF5 metadata server bottleneck)
- zstd compression provides 2-3x reduction with fast decompression
- Python `zarr` library is pure-Python (no compiled dependencies beyond numpy)

### 7.2 Chunking strategy

Current chunking: `[1, 2203, 2299]` -- one frame per chunk.

This is optimal for MAXIE's access pattern (random access to individual frames).
The per-chunk overhead is:
- Compressed chunk: ~10 MB (zstd level 3)
- Decompression to float32: ~20 MB
- Lustre minimum stripe unit: typically 1 MB

### 7.3 Parallel I/O considerations

With 4096 GCDs reading simultaneously:
- **Best case:** Each GCD reads from a different Zarr store/chunk, distributing
  load across Lustre OSTs
- **Worst case:** Many GCDs read from the same store simultaneously, creating a
  hot spot on the OST(s) holding that store

The shuffled DistributedSampler naturally distributes reads across stores. The
segmented loading pattern further reduces contention by limiting the active set
of stores per segment.

**Monitoring I/O performance:**
```bash
# On a compute node during training
lfs getstripe <zarr_store_path>  # Check OST distribution
```

### 7.4 Potential optimization: convert to single-chunk stores

Currently, each Zarr store holds ~40 frames. An alternative is to create one
Zarr store per frame (or use a flat directory of numpy files). This eliminates
the need for the `zarr` library at read time:

```python
# Direct numpy memmap read (no zarr dependency)
data = np.fromfile(chunk_path, dtype=np.float32).reshape(2203, 2299)
```

This trades metadata richness for read speed. Worth profiling if Zarr open/read
overhead is significant.

---

## 8. Patch Extraction

### 8.1 How 224x224 patches are produced

The ViT-MAE model expects 224x224 single-channel images. The detector images are
much larger (e.g., 2203x2299 for jungfrau4M). The current pipeline extracts
patches via `PolarCenterCrop`:

```
Raw image (2203, 2299)
    |-- Pad to (2048, 2048) or larger
    |-- PolarCenterCrop(Hv=224, Wv=224, sigma=0.33, num_crop=N)
    |      Samples N crop centers from a Gaussian distribution
    |      centered on the image, with radial std = sigma * half-image
    |-- MergeBatchPatchDims: (B, N, C, 224, 224) -> (B*N, C, 224, 224)
    v
Multiple 224x224 patches per original image
```

### 8.2 PolarCenterCrop mechanics

The `PolarCenterCrop` class (tensor_transforms.py, line 260):
1. Samples random angles uniformly: `theta ~ U(0, 2*pi)`
2. Samples random radii from half-normal: `radius ~ |N(0, sigma)|`
3. Converts polar to Cartesian offsets from image center
4. Clamps to valid range
5. Uses advanced indexing for efficient batched cropping

With `sigma=0.33`, about 68% of crops are within 1/3 of the image radius from
center, and 99.7% within the full image radius. This biases crops toward the
beam center where Bragg peaks are most likely, while allowing occasional crops
of outer detector regions.

### 8.3 Alternative tiling strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **PolarCenterCrop** (current) | Random polar crops, center-biased | Covers full detector, center-heavy | Non-uniform coverage, may miss edges |
| **Grid tiling** | Non-overlapping 224x224 grid | Complete coverage, deterministic | Fixed positions, no augmentation effect |
| **Sliding window with overlap** | 224x224 with stride < 224 | Complete coverage with context sharing | More patches per image, slower |
| **Random crop** | Uniform random position | Simple, unbiased | May waste time on empty detector regions |
| **Patchify** (available in codebase) | Non-overlapping unfold | Exact coverage, efficient via F.unfold | Fixed grid, `patch_size` must divide image |

**Patchify transform details** (available but not currently used):
```python
# From tensor_transforms.py
# patches = F.unfold(img_padded, kernel_size=(224,224), stride=224)
# For a 2048x2048 padded image: ceil(2048/224) = 10 patches per axis
# Total: 10 x 10 = 100 patches per image (after padding to 2240x2240)
```

### 8.4 Overlap considerations

For a foundation model, patch overlap is generally not needed during pre-training
(each 224x224 patch is an independent training example). However:
- **Stride < patch_size** provides data augmentation via overlapping views
- At inference time (e.g., peak finding on full images), overlapping patches
  with averaging may improve reconstruction quality
- The `Patchify` transform supports arbitrary stride

---

## 9. Data Augmentation

### 9.1 Safe augmentations for diffraction data

Cross-referencing `training-playbook-for-maxie.md` Section 9:

| Augmentation | Safe? | Rationale |
|-------------|-------|-----------|
| **Random rotation** | Yes (with caveats) | Diffraction patterns from randomly oriented crystals naturally appear at all rotations. Safe for SFX; may need constraints for single-crystal experiments. |
| **Random translation / shift** | Yes | Beam center jitter is a real experimental effect. Small shifts (frac_shift_max=0.1) are physically reasonable. |
| **Horizontal/vertical flip** | Likely safe | Detector geometry is usually symmetric. Verify per detector. |
| **Intensity scaling** | Yes | Exposure time variation is common. Scale by 0.5-2.0x. |
| **Gaussian noise addition** | Yes | Simulates detector noise. Use detector-specific noise levels. |
| **Random rectangular masking** (RandomPatch) | Yes | Simulates dead regions, detector gaps. Already implemented. |

### 9.2 Dangerous augmentations for diffraction data

| Augmentation | Danger | Why |
|-------------|--------|-----|
| **Color jitter** | N/A | Single-channel data |
| **RandomResizedCrop with aggressive scale** | Moderate | Scale < 0.5 may create artificial resolution changes; Bragg peak sizes are physically meaningful |
| **Mixup / CutMix** | High | Mixing two diffraction patterns creates physically impossible images |
| **Elastic deformation** | High | Would distort the reciprocal-space geometry |
| **Gaussian blur** | Moderate | Would change the apparent peak width, which carries physics |
| **Cutout of beam center** | High | Removing the beam center (most information-dense region) could degrade representation quality |

### 9.3 Current augmentation pipeline

From `train.fsdp.py`, the active augmentations (controlled by config flags) are:
- `RandomPatch`: 100 random zero-masked rectangles of ~20x20 pixels (with 20% size variation)
- `RandomRotate`: 0-360 degree rotation (bilinear interpolation)
- `RandomShift`: up to 10% of image width/height in each direction

These are applied in the training loop (after DataLoader, on GPU tensors).

### 9.4 MAE-specific augmentation philosophy

From `training-playbook-for-maxie.md` Section 9.1:

> The MAE paper uses very minimal augmentation during pre-training -- the masking
> itself is the primary source of training signal.

Canonical MAE pre-training uses only `RandomResizedCrop(224)` and
`RandomHorizontalFlip`. The existing MAXIE augmentations (RandomPatch,
RandomRotate, RandomShift) are more aggressive than the original MAE recipe.

**Recommendation for baseline:** Start with minimal augmentation (just the
PolarCenterCrop for patch extraction) to establish a clean baseline. Add
augmentations incrementally and measure their effect on reconstruction quality
and downstream task performance.

---

## 10. Key Files Reference

### 10.1 MAXIE codebase ($MAXIE_DIR)

| File | Purpose |
|------|---------|
| `maxie/datasets/ipc_segmented_dataset_dist.py` | Production dataset: IPC socket + segmented distributed loading |
| `maxie/datasets/zarr_dataset.py` | Zarr-based dataset with Parquet manifest (for Frontier) |
| `maxie/datasets/streaming_dataset.py` | pynng streaming IterableDataset with per-node queues |
| `maxie/datasets/psana_dataset.py` | Direct psana dataset (LCLS-local) |
| `maxie/datasets/psana_utils.py` | `PsanaImg` class: psana interface, bad pixel mask, assembly |
| `maxie/datasets/dummy_dataset.py` | Random tensor datasets for throughput testing |
| `maxie/datasets/utils.py` | `apply_mask()` helper |
| `maxie/datasets/server.ipc.py` | TCP IPC server that wraps psana for serving images |
| `maxie/tensor_transforms.py` | All transforms: Pad, Norm, InstanceNorm, DownscaleLocalMean, PolarCenterCrop, MergeBatchPatchDims, Patchify, RandomPatch, RandomRotate, RandomShift, BatchSampler |
| `train/train.fsdp.py` | Main FSDP training script -- dataset config, transforms, training loop |
| `train/hydra_config/train_config/base.yaml` | Default config: dataset params, model params, optimizer |
| `train/generate_dataset_in_json.py` | Generates train/eval JSON splits from YAML experiment files |
| `preprocess/get_global_stats.py` | Computes per-detector mean/std via psana sampling |
| `preprocess/preprocess_psana_runs.py` | Validates psana events, outputs YAML per experiment/run |

### 10.2 Data locations

| Path | Content |
|------|---------|
| `/lustre/orion/lrn091/proj-shared/data/` | Assembled Zarr v3 stores (~2621 stores, ~105K frames) |
| `$LCLS_DATA_BROKER_DIR` | LCLS data broker (Zarr v3 ingest, Tiled catalog) |
| `$DATA_BROKER_DIR` | Original data broker (HDF5, Tiled catalog) |

### 10.3 Environment variables

```bash
export MAXIE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/maxie
export XTAL_DATA_ASSEMBLED=/lustre/orion/lrn091/proj-shared/data
export LCLS_DATA_BROKER_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/lcls-data-broker
export DATA_BROKER_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/tiled-catalog-broker
```

### 10.4 Companion documents

| Document | Relevant sections |
|----------|------------------|
| `training-playbook-for-maxie.md` | Section 9 (augmentation/masking), Section 3 (norm_pix_loss) |
| `research-loop-frontier-strategy.md` | Section 2.1 (sbcast/NVMe), Section 2.2 (srun launch), Section 3 (batch script) |
| `ssl-candidates-for-maxie.md` | SSL method selection and their data requirements |
| `research-loop-brainstorm.md` | Design space axes including data/preprocessing choices |
