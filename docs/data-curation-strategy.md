# Data Curation Strategy

Date: 2026-02-21


## The problem

We want to train a large vision model on LCLS X-ray scattering/diffraction
images. The full dataset across all experiments is ~4M frames and could exceed
30 TB when assembled. Most of these frames contain no meaningful sample
diffraction — they are either empty, dominated by background noise, or show only
artifact scattering (solvent rings, Kapton arcs, beam stop shadows).

Manually inspecting and labeling millions of frames is not feasible. We need
automated or semi-automated methods to classify frames before investing storage
and compute on assembly and training.


## Three-class taxonomy

| Class | Description | Examples |
|---|---|---|
| **No diffraction** | Empty, dark, sentinel values, negligible signal | All-zero frames (cxilw5019 r0017), -1 sentinel frames (mfxx49820 r0017), very low max/std |
| **Artifact scattering** | Signal present but not from the biochemical sample | Solvent/water rings, Kapton/mylar arcs, LCP scattering, beam stop halo, nozzle shadow scatter |
| **Sample diffraction** | Bragg peaks from microcrystal diffraction | Discrete bright spots above background (cxil1015922 lysozyme), powder-ring spots (prjcwang31 mfx13016) |

The boundary between classes 2 and 3 is the hardest — some frames have both
artifact scattering and weak Bragg peaks. For training the vision model,
class 2 frames are still useful (the model needs to learn what background looks
like), but class 1 frames are pure waste.


## Approach 1: Statistics baseline

Use metadata already available in the HDF5 files — no model required.

**What's available**: Many Cheetah-processed HDF5 files contain
`/entry_1/result_1/nPeaks` (peak count from Cheetah's built-in peak finder),
plus we can compute per-frame `min`, `max`, `mean`, `std` from the image data
itself.

**Classification logic**:
- `max == 0` or frame is all-zeros → class 1 (no diffraction)
- `max < mean + threshold` (e.g., threshold = 10 ADU) → class 1
- `nPeaks > 0` → class 3 (sample diffraction)
- `nPeaks == 0` with moderate `max/std` → class 2 (artifact scattering)

**Strengths**: Instant, free, no model needed. Covers the easy cases
(empty/dark frames). The nPeaks field is physics-informed — Cheetah applies
peak-finding algorithms that distinguish peaks from smooth background.

**Weaknesses**: Cheetah thresholds vary per experiment — nPeaks may
over/under-count depending on the configuration used during hit-finding. Some
experiments may not have the nPeaks field at all. Cannot distinguish subtle
Bragg peaks from strong artifact scatter.

**When to use**: Always, as a first pass. This should be computed and stored in
a stats sidecar file regardless of what other approaches we adopt.


## Approach 2: PeakNet feature extraction

PeakNet is a neural network trained for Bragg peak segmentation on ~20k images.
Source: `$PEAKNET_DIR` (`/sdf/home/c/cwang31/codes/peaknet`).

### 2a. Direct peak counting

Run PeakNet inference on frames, count predicted peak pixels above a confidence
threshold. This gives a "learned nPeaks" that may generalize better than
Cheetah's handcrafted peak-finding, especially on detectors/experiments not
well-covered by Cheetah's configuration.

### 2b. Feature map classifier

Extract intermediate feature maps from PeakNet's encoder (before the
segmentation head). These feature maps encode spatial patterns that distinguish
peaks from background. Train a lightweight classifier (linear probe or small
MLP) on top of the frozen features to predict {class 1, 2, 3}.

**Strengths**: Domain-specific features already tuned for X-ray diffraction.
Efficient inference (single forward pass). The feature maps already encode
"what kind of scattering pattern is this?"

**Weaknesses**: Trained on 20k images — generalizability across all detector
types (especially ePix10k if training was Jungfrau-heavy) is unknown. Need to
determine whether PeakNet was trained on raw (stacked-panel) or assembled
images, since this affects how we use it.

**Open question**: What is PeakNet's input format (raw vs assembled)? What
detector types were in its 20k training set?


## Approach 3: Vision-language model (VLM)

Use a multimodal LLM (Claude, GPT-4V, Gemini) to classify sample images via
visual inspection + text prompt.

**Workflow**:
1. Export 500-1000 representative frames as PNG across all experiments and runs
   (diverse sampling: different detectors, different signal levels, different
   experiments).
2. Prompt the VLM: "Classify this X-ray diffraction image as: (A) empty/no
   signal, (B) artifact scattering only (smooth rings, shadows, no discrete
   peaks), (C) sample diffraction (discrete Bragg peaks visible)."
3. Validate VLM labels against known nPeaks data where available.
4. If accuracy is acceptable, use VLM to label a larger calibration set.

**Strengths**: Zero training required. Can understand visual nuance and spatial
context. Excellent for creating a ground truth benchmark set that calibrates all
other approaches.

**Weaknesses**: Expensive at scale (~$0.01-0.05 per image at current API
pricing). Too slow for millions of frames (minutes per batch, not
milliseconds). VLMs may not understand X-ray physics subtleties (e.g.,
distinguishing weak Bragg peaks from hot pixels).

**Best use**: Create a benchmark set of ~500-1000 labeled images. This set
then serves as ground truth for training and evaluating all other approaches.

**Estimated cost**: ~$25-50 for 1000 images.


## Approach 4: CLIP/SigLIP embeddings + clustering

Use a pretrained vision encoder (CLIP ViT-L, SigLIP, or DINOv2) to embed
frames, then cluster the embedding space and label clusters.

**Workflow**:
1. Resize assembled images to 224x224 or 384x384.
2. Extract embeddings with pretrained model (single forward pass per image).
3. Cluster with k-means or HDBSCAN (~50-200 clusters).
4. Manually inspect 3-5 example images per cluster and assign a class label.
5. Propagate cluster labels to all frames in each cluster.

**Strengths**: Scales to millions of frames by labeling ~50-200 clusters
instead of millions of images. Pretrained models are surprisingly good at
capturing texture and spatial structure even for out-of-domain images. Fast
inference (~100 frames/sec on GPU).

**Weaknesses**: Pretrained models may group frames by detector type or
background level rather than by diffraction content — a Jungfrau empty frame
and a Jungfrau peak frame might cluster together because they share the same
panel geometry. May need per-detector-type clustering to avoid this. Resizing
destroys physical pixel scale.

**Mitigation**: Cluster within each detector type separately. Use
high-dimensional clustering (UMAP + HDBSCAN) to preserve fine structure.


## Approach 5: Azimuthal profile analysis

Since assembled images have correct geometry, compute the azimuthal (radial)
intensity average around the beam center.

**How it works**: For each frame, compute a 1D intensity profile I(q) by
averaging intensity in concentric annular bins. Smooth scattering produces a
smooth I(q). Bragg peaks produce sharp spikes above the smooth baseline.

**Classification logic**:
- Fit a smooth baseline to I(q) (e.g., polynomial or Savitzky-Golay filter).
- Count the number of bins where I(q) exceeds the baseline by >N*sigma.
- Many spikes → class 3 (sample diffraction).
- Smooth profile with moderate amplitude → class 2 (artifact scattering).
- Flat near-zero → class 1 (empty).

**Strengths**: Fast (no ML needed), physics-grounded, works identically across
all detector types once assembled. This is fundamentally what crystallographers
do when visually inspecting diffraction patterns.

**Weaknesses**: Requires knowing the beam center (available from geometry
files). Cannot distinguish artifact rings from sample powder rings (both are
smooth azimuthally). Works best for sharp Bragg peaks, less for diffuse or
streaky sample scatter. Requires assembled images (not raw stacked-panel).


## Approach 6: Active learning loop

Combine multiple approaches into an iterative refinement pipeline.

**Workflow**:
1. **Bootstrap**: Run stats baseline (Approach 1) to create noisy initial labels.
   Easily separates class 1 from classes 2+3.
2. **Benchmark**: Use VLM (Approach 3) to label 500-1000 diverse images as
   ground truth, calibrating the stats-based labels.
3. **Train**: Train a small CNN (ResNet-18 or EfficientNet-B0) on the
   VLM-validated labels. This model runs at ~1000 frames/sec on GPU.
4. **Uncertainty sampling**: Run the CNN on all frames. Find the 200-500 most
   uncertain predictions (highest entropy).
5. **Refine**: VLM or human labels the uncertain cases.
6. **Retrain**: Update the CNN with the new labels. Repeat steps 4-6.

**Strengths**: Minimizes human/VLM labeling effort by focusing on the hard
cases. Produces a fast, cheap classifier that can handle millions of frames.
Each iteration improves both the classifier and the labeled dataset.

**Weaknesses**: More complex pipeline with multiple moving parts. Requires
initial investment in infrastructure (export frames, set up VLM API, train CNN).
Several iterations may be needed before convergence.


## Recommended tiered plan

| Step | Approach | Purpose | Effort | Dependency |
|---|---|---|---|---|
| **1** | Stats baseline | Filter obvious empties, compute per-frame stats | Low (hours) | None |
| **2** | VLM benchmark | Create ground truth labels on 500-1000 images | Low-medium (~$25-50 API cost) | Step 1 (to sample diverse frames) |
| **3** | PeakNet feature probe | Test domain-specific features as classifier | Medium | Step 2 (for evaluation) |
| **4** | Lightweight CNN | Fast classifier for millions of frames | Medium | Step 2 (for training labels) |
| **5** | Active learning | Iterate on hard cases | Ongoing | Steps 2 + 4 |

Steps 1 and 2 are independent and can start immediately. Step 2 is the most
interesting research question — how well do VLMs actually understand X-ray
diffraction images? Step 3 leverages existing infrastructure (PeakNet) and can
run in parallel with Step 2.


## Open questions

1. **PeakNet input format**: Was PeakNet trained on raw (stacked-panel) or
   assembled images? This determines whether we run it before or after assembly.

2. **VLM accuracy**: How well do vision-language models classify X-ray
   diffraction images? This needs empirical testing on a pilot set before
   committing to VLM as a labeling strategy.

3. **Unconverted experiments**: Should we assemble cxilw5019, mfxp22421, and
   mfxx49820 now, or first classify their raw HDF5 frames (using stats or
   PeakNet on raw data) and only assemble the non-empty frames?

4. **Class 2 value**: For training the vision model, artifact scattering frames
   (class 2) may be valuable — the model needs to learn what background looks
   like. Should we keep a controlled fraction of class 2 frames, or filter
   them aggressively?

5. **prjcwang31 geometry**: The personal project directory contains data from
   multiple sub-experiments with potentially different geometries. Needs
   separate geometry discovery before assembly is possible.
