# Open Questions and Design Discussions

Date: 2026-02-19

These topics emerged from brainstorming after the initial architecture survey
and data exploration. They are not yet resolved — this document captures the
trade-offs and proposed starting points to revisit as we gain empirical results.


## 1. Image decomposition: panels to patches

### The problem

Panel-wise patching (see `architecture-research.md` Section 2) handles extreme
aspect ratios by splitting stacked-panel images along the H dimension into
physical detector panels. But the panels themselves have different sizes across
detector types, and extracting uniform patches from non-uniform panels is not
straightforward.

### Panel dimensions recap

```
Jungfrau 4M:  8 panels of 512 x 1024
ePix10k2M:   16 panels of 352 x 384
Jungfrau 16M: 32 panels of 512 x 1024
Assembled:    1 panel  of 1920 x 1920
```

### Patch size trade-offs

| Patch size | Jungfrau 512x1024 | ePix 352x384 | Pros | Cons |
|---|---|---|---|---|
| **128x128** | 4x8 = 32 patches/panel | 2x3 = 6 patches/panel (with 96x0 leftover H) | Many patches, fast training per patch | Small context window; Bragg peak clusters may span patches |
| **192x192** | 2x5 = 10 patches (with 128x64 leftover) | 1x2 = 2 patches (with 160x0 leftover H) | Reasonable context | Awkward tiling on both detector types |
| **256x256** | 2x4 = 8 patches/panel (clean tiling) | 1x1 = 1 patch (96x128 leftover) | Clean tiling on Jungfrau; standard ViT size | Wastes most of each ePix panel |
| **384x384** | 1x2 = 2 patches (with 128x256 leftover) | 0 full patches (panel is 352x384) | Large context per patch | Few patches per panel; ePix panels don't fit at all |

No single patch size tiles cleanly on both Jungfrau (512x1024) and ePix
(352x384) panels.

### Approaches for the ePix mismatch

Given a 256x256 target patch size and ePix panels of 352x384:

1. **Random crop**: Randomly sample a 256x256 region from the 352x384 panel.
   Simple, provides spatial augmentation for free, but discards edge pixels on
   average.

2. **Pad to 384x384, then crop**: Add 32 rows of zeros (or mirror-pad) to make
   the panel square, then extract a 256x256 random crop. Slightly cleaner
   geometry but introduces artificial content at boundaries.

3. **Use a different patch size for ePix**: e.g., 176x192 tiles 2x2 on a
   352x384 panel exactly. But now the ViT sees different-sized inputs depending
   on detector type, which complicates the model.

4. **Resize the panel**: Resize 352x384 → 256x256. Simple but destroys the
   physical pixel scale — a pixel in ePix covers a different solid angle than
   in Jungfrau, and resizing makes this inconsistency worse.

### Assembled images (1920x1920)

These are already geometry-corrected (panels assembled into a physically
accurate layout). Patching strategy is simpler:

- **Random 256x256 crops**: Standard approach for large images. Each crop spans
  a contiguous region that may cross physical panel boundaries — this is fine
  because geometry correction has already been applied.
- **Grid-based tiling**: 7x7 grid of 256x256 patches with ~37-pixel overlap,
  or 8x8 grid of 240x240 patches. Useful for inference (cover the full image)
  but not needed for self-supervised training where random crops suffice.

### Panel boundary handling

Stacked-panel images contain dark horizontal bands at inter-panel gaps. These
are physical gaps in the detector tiling and carry no signal.

Options:
- **Crop panels to exclude gap rows**: Cleanest approach, requires knowing the
  gap location per detector type (typically a few rows of zeros at panel
  boundaries).
- **Let the model learn to ignore them**: Lazy but may work — the model will
  see consistent dark bands and learn they're uninformative.
- **Panel-boundary-aware masking**: In I-JEPA or MAE, never mask gap regions
  (they're trivially predictable). This avoids wasting prediction capacity on
  zeros.

### Physical scale inconsistency

A 256x256 patch at Jungfrau resolution covers a different solid angle than at
ePix resolution (different pixel sizes, different sample-to-detector distances).
For self-supervised pretraining this is acceptable — the model should learn
detector-agnostic features. For physics-sensitive downstream tasks, the detector
type metadata should be available as conditioning information.

### Suggested starting approach (Phase 1)

- **256x256 patches** for Jungfrau panels (clean 2x4 tiling of 512x1024)
- **Random 256x256 crop** from ePix panels (352x384 → random crop)
- **Random 256x256 crop** from assembled images (1920x1920)
- **Exclude inter-panel gap rows** during panel splitting
- Revisit patch size empirically once we have a working training pipeline


## 2. Training data composition: signal vs. noise vs. artifacts

### The problem

The dataset contains a wide spectrum of frame content:

| Category | Examples | Estimated prevalence |
|---|---|---|
| **Bragg peaks** (signal) | Discrete bright spots from microcrystal diffraction | Minority of frames in most experiments |
| **Smooth scattering** (background) | Solvent ring, LCP scattering, powder-like rings | Majority in mfx experiments |
| **Artifacts** | Beam stop shadow, nozzle shadow, gain variation, hot pixels | Present in most frames |
| **Empty/sentinel** | All-zero frames, -1 sentinel values | ~75% of random samples in cxilw5019; varies by experiment |

Naive random sampling across the full dataset yields mostly empty or
low-information frames. This wastes compute and may bias the model toward
trivial representations.

### Is this a problem for self-supervised learning?

It depends on the method:

**MAE**: Empty frames are clearly harmful. Reconstructing zeros from zeros
teaches nothing. If most training batches are blank, the model learns a trivial
"predict low values everywhere" strategy.

**I-JEPA / DINOv2**: Less clear-cut. Empty frames still waste compute, but
frames with artifact scattering (solvent rings, Kapton arcs, shadows) are *not*
useless — the model can learn to recognize and represent these patterns. The
model needs to see background to distinguish signal from background.

### Proposed tiered filtering strategy

#### Tier 1 — Hard filter (precompute once, store in manifest)

Remove before training:
- All-zero frames (sentinel/empty)
- Frames where `max_value < threshold` (e.g., 10 ADU above frame mean —
  effectively dark frames with no signal)
- Frames from corrupt HDF5 files (already identified during manifest building)

Implementation: add a `valid` boolean flag per frame in the manifest or stats
sidecar file. Compute once during manifest building.

#### Tier 2 — Soft sampling (adjust during training)

Use per-frame statistics to assign sampling weights:
- Frames with high max/std ratio (likely Bragg peaks) → **higher weight**
- Frames with moderate std (background scattering, artifacts) → **normal weight**
- Frames near the Tier 1 threshold (dim but not empty) → **lower weight**

This does not require labels. The `nPeaks` field from Cheetah/psocake (available
in many HDF5 files at `/entry_1/result_1/nPeaks`) can serve as a proxy for
signal content where available.

Implementation: a weighted `RandomSampler` in PyTorch that reads weights from
the stats sidecar.

#### Tier 3 — Smart sampling (future, after Phase 1)

After initial MAE training:
1. Use the trained encoder to embed all valid frames
2. Cluster embeddings (e.g., k-means in embedding space)
3. Identify underrepresented or interesting clusters
4. Oversample rare clusters in Phase 2 training

This creates a feedback loop between model quality and data curation.

### The case for keeping "uninteresting" frames

Frames with smooth scattering and no peaks serve a purpose:
- They teach the model what "normal background" looks like
- A model that only sees Bragg peak frames won't learn to distinguish peaks
  from background — it will overfit to high-intensity features
- Analogous to ImageNet pretraining benefiting from "boring" images (sky,
  grass, textures) alongside object-centric images

### Suggested starting ratios

A reasonable initial target (adjustable after Phase 1 results):

| Category | Target fraction | Source |
|---|---|---|
| Bragg peak frames (nPeaks > 0 or high max/std) | ~30% | Oversample from peak-rich experiments |
| Scattering/background (moderate signal) | ~50% | Natural prevalence after Tier 1 filtering |
| Artifact-dominated / edge cases | ~20% | Shadows, gain variation, partial frames |

These ratios should be treated as a starting hypothesis, not a requirement. The
right ratio depends on the training objective and will be tuned empirically.


## 3. Data access: manifest vs. broker

### What metadata does each panel/patch need?

| Metadata | Source | Why it matters |
|---|---|---|
| Detector type | Manifest (per-file) | Determines panel layout, pixel size, gain characteristics |
| Panel index | Derived from H-offset during patching | Position on detector → scattering angle range |
| Experiment ID | Manifest (per-file) | Groups frames by experimental conditions |
| Run number | File path / manifest | Groups by time, sample, conditions within an experiment |
| Photon energy (eV) | HDF5 `/LCLS/photon_energy_eV` | Affects scattering geometry; needed for physics-aware normalization |
| nPeaks | HDF5 `/entry_1/result_1/nPeaks` | Enables Tier 2 sampling by signal content |
| Frame statistics | Precomputed sidecar | min/max/mean/std for filtering and normalization |
| Gain mode | Detector-specific metadata | ePix10k has multi-gain pixels; affects intensity calibration |
| Valid flag | Computed during manifest build | Whether frame passes Tier 1 hard filter |

### Current approach: manifest.json

The manifest stores file-level metadata (path, shape, detector type, panel
config). The PyTorch Dataset reads this at init time and builds an in-memory
index of `(file, frame, panel)` triples.

**Strengths**: Simple, inspectable, version-controllable, no infrastructure.

**Limitations**: File-level only. Per-frame metadata (nPeaks, statistics, valid
flags) must live elsewhere.

### Proposed extension: manifest + stats sidecar

Add a sidecar file (`data/frame_stats.h5` or `.parquet`) with one row per frame:

```
file_index  |  frame_index  |  valid  |  mean  |  std  |  max  |  nPeaks  |  photon_energy_eV
------------|---------------|---------|--------|-------|-------|----------|-------------------
0           |  0            |  True   |  22.3  |  35.1 | 12745 |  47      |  9500.0
0           |  1            |  True   |  1.3   |  3.0  |  489  |  0       |  9500.0
0           |  2            |  False  |  0.0   |  0.0  |  0    |  0       |  9500.0
...
```

The PyTorch Dataset loads both the manifest and the sidecar at init time. The
sidecar enables:
- Tier 1 filtering (skip `valid=False` frames)
- Tier 2 weighted sampling (weight by nPeaks or max/std ratio)
- Per-frame normalization using precomputed statistics (faster than computing
  on-the-fly)

### When does the manifest approach break down?

| Trigger | Why it's painful |
|---|---|
| New experiments added frequently | Must re-run stats pipeline and rebuild sidecar manually |
| Cross-experiment queries | "All Jungfrau 4M frames with >10 peaks" requires loading full sidecar + manifest and filtering in Python |
| Changing access patterns mid-training | Curriculum learning that shifts sampling weights requires recomputing the sampler |
| Multiple consumers | Training, evaluation, and visualization all need different views of the same data |

### When to build a broker

**Trigger**: When you find yourself re-running the stats pipeline manually for
a 3rd experiment, or when you need per-frame embeddings queryable by
experiment/detector/content for smart sampling (Tier 3).

**What it should look like**: A thin Python API (not a service) wrapping the
manifest + sidecar:

```python
# Query for specific frames
triples = broker.query(detector="Jungfrau4M", min_peaks=5, experiments=["cxil1015922"])

# Get a patch with its metadata
patch, metadata = broker.get_patch(file_idx=42, frame_idx=7, panel_idx=3, patch_size=256)
# metadata = {"detector": "Jungfrau4M", "panel_index": 3, "nPeaks": 12, ...}

# Add a new experiment (scan, validate, update manifest + sidecar)
broker.add_experiment("/path/to/new/experiment/hdf5/")
```

This is the manifest approach with a Python API on top — not a database or web
service. It can grow into a Tiled broker later when/if we need multi-format
support (ZARR), caching, or remote access.

### Recommendation

- **Phases 1-2**: Manifest + stats sidecar. Build the sidecar during manifest
  generation. Keep it as plain files (JSON + HDF5 or Parquet).
- **Phase 3 or when triggered**: Wrap in a thin Python broker class. The
  manifest and sidecar become the broker's backing store.
- **Long-term**: Migrate to Tiled if LCLS infrastructure moves that direction
  or if we need cloud/remote access.
