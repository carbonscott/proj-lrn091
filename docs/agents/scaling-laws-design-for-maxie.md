# Scaling Laws Experiment Design for MAXIE

Date: 2026-03-19
Context: ALCC project Group 1 -- Phase 2 of the MAXIE training playbook. This document
designs the scaling law study that will determine how to allocate 121,680 GPU-hours
across 10 pre-training campaigns of the MAXIE foundation model (ViT-MAE for X-ray
crystallography diffraction images on Frontier).

Companion documents:
- `training-playbook-for-maxie.md` -- training recipe, optimizer, batch scaling, Phase 2 overview
- `ssl-candidates-for-maxie.md` -- model architecture details and SSL candidates
- `research-loop-frontier-strategy.md` -- Frontier resource details, Flux ensemble patterns
- `data-pipeline-for-maxie.md` -- dataset sizes, Zarr stores, data skew
- `monitoring-protocol-for-maxie.md` -- metrics to track during scaling runs

---

## 1. Goal

The purpose of this scaling law study is to answer a single question before committing
the full 121,680 GPU-hour ALCC budget:

**For a given compute budget C, what is the optimal combination of model size N,
dataset size D, and training duration T that minimizes the pre-training loss?**

Without this study, the project risks two expensive mistakes:

1. **Undertrained large model.** Training ViT-Huge (632M params) on all data but for
   too few steps -- the model never converges, and the GPU-hours are wasted.
2. **Overtrained small model.** Training ViT-Base (86M params) to completion many
   times -- the model saturates and additional compute yields diminishing returns.

Scaling laws let us fit power-law relationships from cheap, short experiments and
extrapolate to predict the optimal allocation for the full budget. The study itself
should consume 10-15% of the total budget (12,168-18,252 GPU-hours), leaving 85-90%
for the actual campaigns.

---

## 2. Scaling Law Frameworks

### 2.1 Kaplan et al. (2020) -- "Scaling Laws for Neural Language Models"

The foundational work that established power-law relationships between loss and three
factors: model parameters (N), dataset size (D), and compute (C).

**Core equation:**

```
L(N, D) = (N_c / N)^alpha_N + (D_c / D)^alpha_D + L_inf
```

Where:
- `L_inf` is the irreducible loss (entropy of the data)
- `N_c`, `D_c` are characteristic scales
- `alpha_N ~ 0.076`, `alpha_D ~ 0.095` (for language models)

**Key finding:** For a fixed compute budget, there is an optimal model size. Larger
models are more sample-efficient -- they extract more information per training token.
Kaplan recommended scaling model size faster than dataset size (roughly N^0.74 : D^0.27
for a compute increase).

**Limitation:** Later work (Chinchilla) showed Kaplan's recommendations led to
undertrained models because the experiments used too-short training runs.

### 2.2 Hoffmann et al. (2022) -- Chinchilla / "Compute-Optimal Training"

The Chinchilla paper corrected Kaplan's findings by training models to near-convergence
at each scale point and using three complementary fitting approaches.

**Core finding:** For compute-optimal training, model parameters and training tokens
should be scaled roughly equally. Specifically:

```
N_opt ~ C^0.50    (optimal parameters scale as sqrt of compute)
D_opt ~ C^0.50    (optimal tokens scale as sqrt of compute)
```

This means: **for every doubling of compute, double both the model size AND the
amount of training data.** Most models at the time (including Gopher 280B) were
significantly undertrained relative to their size.

**Chinchilla scaling law (parametric form):**

```
L(N, D) = E + A / N^alpha + B / D^beta
```

Where `E` is the irreducible loss, and `alpha ~ 0.34`, `beta ~ 0.28` (fitted values
from the paper; domain-specific values will differ).

**Why this fits MAXIE best:** The ALCC allocation is a fixed compute budget (121,680
GPU-hours). The binding constraint is compute, not data availability or researcher
time. The Chinchilla framework directly answers "given C FLOPs, what N and D are
optimal?" -- exactly the question MAXIE faces.

### 2.3 CLIPA -- Inverse Scaling for CLIP Training

Li et al. (2023) "An Inverse Scaling Law for CLIP Training" found that for CLIP
models, larger encoders can be trained effectively with shorter input sequences
(fewer image/text tokens). This is an inverse relationship: bigger model, less
data per sample.

**Relevance to MAXIE:** Limited. CLIPA applies specifically to contrastive
image-text pre-training where sequence length (number of tokens per sample) is a
free variable. In MAE pre-training:
- The image is always 224x224 with patch_size=16, giving a fixed 196 tokens
- The "data" axis is the number of training images, not tokens per image
- The mask ratio varies the effective input but does not change the token count

CLIPA's insight could become relevant if MAXIE experiments with variable image
resolution or patch size (see Section 3, secondary axes), but for the primary
scaling study, Chinchilla is the better framework.

### 2.4 Framework Selection for MAXIE

| Framework | Primary use case | Fits MAXIE? | Why |
|-----------|-----------------|-------------|-----|
| Kaplan (2020) | Initial exploration | Partially | Established methodology, but recommendations biased |
| Chinchilla (2022) | Compute-optimal allocation | **Yes** | Fixed compute budget, need optimal N:D ratio |
| CLIPA (2023) | Sequence length optimization | No | Applies to contrastive CLIP, not MAE |

**Decision: Use the Chinchilla framework** (Approach 3 from Hoffmann et al.) --
fit a parametric loss function `L(N, D)` from a grid of (model_size, data_size,
steps) experiments, then solve for the compute-optimal frontier.

---

## 3. Axes of Variation

### 3.1 Primary axes (must sweep)

#### Model size

| Variant | Hidden | Layers | Heads | MLP | Encoder params | Decoder params | Total params |
|---------|--------|--------|-------|-----|---------------|---------------|-------------|
| ViT-Base | 768 | 12 | 12 | 3072 | ~86M | ~24M | ~110M |
| ViT-Large | 1024 | 24 | 16 | 4096 | ~304M | ~24M | ~328M |
| ViT-Huge | 1280 | 32 | 16 | 5120 | ~632M | ~24M | ~656M |

The MAE decoder is fixed across all encoder sizes (hidden=512, 8 layers, 16 heads,
mlp=2048). Only encoder parameters N count for scaling law purposes, since the
decoder is discarded after pre-training.

Source: `training-playbook-for-maxie.md` Section 13.

#### Dataset size

Available data: ~2,621 Zarr stores x ~40 frames each = ~104,840 frames total.
All from experiment `cxi101235425` with jungfrau4M detector (2203x2299 assembled
images, float32, zstd compressed).

| Data fraction | Frames | Zarr stores | Approximate size |
|--------------|--------|-------------|------------------|
| 10% | ~10,484 | ~262 | ~100 GB |
| 25% | ~26,210 | ~655 | ~250 GB |
| 50% | ~52,420 | ~1,311 | ~500 GB |
| 100% | ~104,840 | ~2,621 | ~1 TB |

Data fractions should be drawn as contiguous run-level subsets (not random frame
sampling) to avoid data leakage within runs. Use a fixed seed for reproducibility.

#### Training steps

| Steps | Purpose | Approx. cost at 8 nodes (ViT-Base) |
|-------|---------|-------------------------------------|
| 25K | Quick signal -- does the model learn at all? | ~50 GPU-hours |
| 50K | Early scaling trend visible | ~100 GPU-hours |
| 100K | Solid scaling curve, near inflection point | ~200 GPU-hours |
| 200K | Approaching convergence for small models | ~400 GPU-hours |

Step costs are approximate and depend on batch size, model size, and throughput
(detailed in Section 4).

### 3.2 Secondary axes (sweep after primary)

#### Mask ratio

| Mask ratio | Visible tokens (of 196) | Encoder input | Reconstruction difficulty |
|-----------|------------------------|---------------|--------------------------|
| 0.50 | 98 | 50% of image | Moderate -- many peaks visible |
| 0.75 | 49 | 25% of image | Standard MAE -- high difficulty |
| 0.90 | 20 | 10% of image | Very hard -- almost blind reconstruction |

For diffraction data with sparse Bragg peaks, the optimal mask ratio may differ from
the natural-image default of 0.75. A 75% mask has a high probability of occluding
most peaks, which could make reconstruction either too hard (model gives up) or force
useful semantic learning (model must infer peak locations from context).

#### Patch size

| Patch size | Tokens per image (224x224) | Params change? | Notes |
|-----------|--------------------------|---------------|-------|
| 16 | 196 | No | Standard, fine-grained patches |
| 32 | 49 | ~25% fewer (smaller embedding) | Coarser, fewer tokens, faster training |

Larger patch size reduces sequence length (fewer tokens), which reduces compute per
step quadratically (attention cost) but may lose spatial resolution needed for Bragg
peak detection.

---

## 4. Compute Budget Allocation

### 4.1 Total budget

| Item | GPU-hours |
|------|-----------|
| Full ALCC allocation | 121,680 |
| Scaling law study (target: 12.5%) | **15,210** |
| Remaining for 10 campaigns | 106,470 |

Reserving 12.5% for scaling studies is conservative. Chinchilla used ~10% of the
total Gopher compute budget for their scaling experiments. We budget slightly more
because this is a novel domain (X-ray diffraction) where we cannot rely on
published scaling exponents from NLP/vision.

### 4.2 GPU-hour calculation

On Frontier, each node has 8 GCDs (Graphics Compute Dies from 4 MI250X GPUs).

```
GPU-hours = nodes * 8 GCDs * walltime_hours
```

Example: 8 nodes for 2 hours = 8 * 8 * 2 = 128 GPU-hours.

### 4.3 Throughput estimates

Throughput depends on model size, batch size, and hardware efficiency. Using the
current MAXIE configuration (float16 AMP, FSDP, activation checkpointing, 224x224
images, patch_size=16) and conservative estimates for Frontier MI250X:

| Model | Batch/GCD | Tokens/sec/GCD | Steps/hour (8 nodes, batch=1/GCD) | Steps/hour (effective) |
|-------|-----------|---------------|-----------------------------------|----------------------|
| ViT-Base (86M) | 16 | ~4,000 | ~1,300 | ~1,300 |
| ViT-Large (304M) | 8 | ~1,500 | ~475 | ~475 |
| ViT-Huge (632M) | 4 | ~600 | ~190 | ~190 |

Notes:
- Tokens/sec/GCD is an estimate based on MAE paper throughput scaled for MI250X
  (~52 TFLOPS fp16 effective per GCD)
- Steps/hour assumes batch_per_GCD samples consumed per step (no gradient accumulation)
- Effective batch size = nodes * 8 * batch_per_GCD

These numbers must be validated with a short benchmark run (see Section 9.3).

### 4.4 Experiment table

The table below defines the complete set of scaling law experiments. Each row is one
training run.

**Tier 1: Core scaling grid (model size x data fraction x steps)**

| Run ID | Model | Data % | Steps | Nodes | Est. walltime | Est. GPU-hours |
|--------|-------|--------|-------|-------|---------------|---------------|
| S01 | Base (86M) | 10% | 25K | 4 | 0.6h | 19 |
| S02 | Base (86M) | 10% | 50K | 4 | 1.2h | 38 |
| S03 | Base (86M) | 10% | 100K | 4 | 2.4h | 77 |
| S04 | Base (86M) | 25% | 25K | 4 | 0.6h | 19 |
| S05 | Base (86M) | 25% | 50K | 4 | 1.2h | 38 |
| S06 | Base (86M) | 25% | 100K | 4 | 2.4h | 77 |
| S07 | Base (86M) | 50% | 50K | 4 | 1.2h | 38 |
| S08 | Base (86M) | 50% | 100K | 4 | 2.4h | 77 |
| S09 | Base (86M) | 100% | 50K | 4 | 1.2h | 38 |
| S10 | Base (86M) | 100% | 100K | 4 | 2.4h | 77 |
| S11 | Base (86M) | 100% | 200K | 4 | 4.8h | 154 |
| S12 | Large (304M) | 10% | 25K | 8 | 0.8h | 51 |
| S13 | Large (304M) | 10% | 50K | 8 | 1.6h | 102 |
| S14 | Large (304M) | 25% | 25K | 8 | 0.8h | 51 |
| S15 | Large (304M) | 25% | 50K | 8 | 1.6h | 102 |
| S16 | Large (304M) | 25% | 100K | 8 | 3.3h | 211 |
| S17 | Large (304M) | 50% | 50K | 8 | 1.6h | 102 |
| S18 | Large (304M) | 50% | 100K | 8 | 3.3h | 211 |
| S19 | Large (304M) | 100% | 50K | 8 | 1.6h | 102 |
| S20 | Large (304M) | 100% | 100K | 8 | 3.3h | 211 |
| S21 | Large (304M) | 100% | 200K | 8 | 6.6h | 422 |
| S22 | Huge (632M) | 10% | 25K | 16 | 1.0h | 128 |
| S23 | Huge (632M) | 10% | 50K | 16 | 2.1h | 269 |
| S24 | Huge (632M) | 25% | 25K | 16 | 1.0h | 128 |
| S25 | Huge (632M) | 25% | 50K | 16 | 2.1h | 269 |
| S26 | Huge (632M) | 50% | 50K | 16 | 2.1h | 269 |
| S27 | Huge (632M) | 50% | 100K | 16 | 4.1h | 525 |
| S28 | Huge (632M) | 100% | 50K | 16 | 2.1h | 269 |
| S29 | Huge (632M) | 100% | 100K | 16 | 4.1h | 525 |
| S30 | Huge (632M) | 100% | 200K | 16 | 8.2h | 1,050 |

**Tier 1 subtotal: ~5,714 GPU-hours (30 runs)**

**Tier 2: Mask ratio sweep (ViT-Base, 50% data, 50K steps)**

| Run ID | Model | Mask ratio | Steps | Nodes | Est. GPU-hours |
|--------|-------|-----------|-------|-------|---------------|
| M01 | Base (86M) | 0.50 | 50K | 4 | 38 |
| M02 | Base (86M) | 0.75 | 50K | 4 | 38 |
| M03 | Base (86M) | 0.90 | 50K | 4 | 38 |

Note: M02 overlaps with S07; reuse results. Effective cost: 76 GPU-hours.

**Tier 2 subtotal: ~76 GPU-hours**

**Tier 3: Patch size sweep (ViT-Base, 50% data, 50K steps)**

| Run ID | Model | Patch size | Steps | Nodes | Est. GPU-hours |
|--------|-------|-----------|-------|-------|---------------|
| P01 | Base (86M) | 16 | 50K | 4 | 38 |
| P02 | Base (86M) | 32 | 50K | 4 | 20 |

Note: P01 overlaps with S07; reuse results. Effective cost: 20 GPU-hours.

**Tier 3 subtotal: ~20 GPU-hours**

**Throughput benchmark (run first, before committing to the grid):**

| Run ID | Model | Purpose | Nodes | Est. GPU-hours |
|--------|-------|---------|-------|---------------|
| B01 | Base (86M) | Measure tokens/sec, validate batch size | 4 | 6 |
| B02 | Large (304M) | Measure tokens/sec, validate batch size | 8 | 13 |
| B03 | Huge (632M) | Measure tokens/sec, validate batch size | 16 | 26 |

Each benchmark runs for 500 steps (~10-20 min) to get stable throughput numbers.

**Benchmark subtotal: ~45 GPU-hours**

### 4.5 Total scaling study budget

| Tier | Runs | GPU-hours |
|------|------|-----------|
| Benchmarks | 3 | 45 |
| Tier 1: Core grid | 30 | 5,714 |
| Tier 2: Mask ratio | 2 (net) | 76 |
| Tier 3: Patch size | 1 (net) | 20 |
| Buffer (20% contingency) | -- | 1,171 |
| **Total** | **36** | **7,026** |

This is approximately **5.8%** of the total ALCC budget -- well within the 12.5%
ceiling, leaving ample room for rerunning failed experiments or adding follow-up
investigations.

---

## 5. FLOP Estimation

### 5.1 FLOPs per training step (forward + backward)

For a transformer model, the standard approximation for FLOPs per training step is:

```
FLOPs_step = 6 * N * B * S
```

Where:
- `6` accounts for forward pass (2x multiplies per multiply-add) + backward pass (2x forward)
- `N` = number of model parameters (encoder + decoder for MAE)
- `B` = batch size (per step, across all devices)
- `S` = sequence length (number of tokens per image)

For MAE specifically, the encoder only processes visible tokens (25% at mask_ratio=0.75),
while the decoder processes all tokens. A more precise formula:

```
FLOPs_step = 6 * B * [N_enc * S_visible + N_dec * S_total]
```

Where:
- `S_total = (image_size / patch_size)^2 = (224/16)^2 = 196`
- `S_visible = S_total * (1 - mask_ratio) = 196 * 0.25 = 49` (at mask_ratio=0.75)

### 5.2 Concrete FLOP estimates per model

Using the simplified formula with effective batch size and the MAE-adjusted token counts:

| Model | N_enc | N_dec | S_vis | S_total | FLOPs/step (B=1) | FLOPs/step (B=256) |
|-------|-------|-------|-------|---------|-----------------|-------------------|
| ViT-Base | 86M | 24M | 49 | 196 | 5.34e10 | 1.37e13 |
| ViT-Large | 304M | 24M | 49 | 196 | 1.18e11 | 3.02e13 |
| ViT-Huge | 632M | 24M | 49 | 196 | 2.14e11 | 5.47e13 |

Calculation example for ViT-Base, B=1:

```
FLOPs = 6 * 1 * (86e6 * 49 + 24e6 * 196)
      = 6 * (4.214e9 + 4.704e9)
      = 6 * 8.918e9
      = 5.35e10
```

### 5.3 Total FLOPs per scaling run

```
FLOPs_run = FLOPs_step * num_steps
```

Example for S11 (ViT-Base, 100% data, 200K steps, effective batch = 4 nodes * 8 GCDs * 16 batch/GCD = 512):

```
FLOPs_run = 6 * 512 * (86e6*49 + 24e6*196) * 200,000
          = 6 * 512 * 8.918e9 * 200,000
          = 5.48e18 FLOPs
          = 5.48 PFLOPs
```

### 5.4 Converting FLOPs to GPU-hours

Each MI250X GCD achieves approximately 26 TFLOPS (fp16, measured, ~50% of 52 TFLOPS
theoretical peak, accounting for memory bandwidth limitations in ViT workloads):

```
GPU-hours = FLOPs_run / (26e12 * 3600)
          = FLOPs_run / 9.36e16
```

For S11: `5.48e18 / 9.36e16 = 58.5 GPU-hours`

This is lower than the 154 GPU-hours estimated in Section 4.4 because the Section 4
estimates include I/O overhead, data loading latency, communication overhead (FSDP
AllReduce), and checkpoint saving. The FLOP-based estimate represents the compute
lower bound; multiply by ~2.5x for realistic wall-clock estimates.

### 5.5 Total FLOPs for the full ALCC budget

```
Total_FLOPs = 121,680 GPU-hours * 26e12 FLOPS * 3600 sec
            = 1.14e22 FLOPs
            = 11.4 ZettaFLOPs
```

This is the "compute envelope" within which the scaling law must identify the optimal
(N, D, T) configuration.

---

## 6. Fitting Scaling Curves

### 6.1 Data collection

From each scaling experiment, extract:

| Metric | Source | When to record |
|--------|--------|---------------|
| Validation loss | `LOSS:EVAL` log event | Every 1K steps |
| Training loss | `LOSS:TRAIN` log event | Every step (use smoothed: EMA with alpha=0.01) |
| Total FLOPs consumed | Computed from step count + model size + batch size | Every 1K steps |
| Total tokens seen | `step * effective_batch_size` | Every 1K steps |
| Wall-clock time | Step timestamps | Every step |

### 6.2 Plotting

Generate three families of plots:

**Plot A: Loss vs. compute (FLOPs) -- one curve per model size**

```
                    Loss
                      |
                  2.0 |  x  Base
                      |   \   x Large
                  1.5 |    \   \  x Huge
                      |     \   \ \
                  1.0 |      ----\-\---
                      |           --------
                  0.5 |
                      +-----|------|------|---> log(FLOPs)
                           1e17  1e18  1e19
```

Each model size traces a curve. The compute-optimal frontier is the lower envelope.

**Plot B: Loss vs. tokens seen -- one curve per model size**

Reveals whether models are data-limited (curves flatten) or compute-limited (curves
still decreasing).

**Plot C: Loss vs. model size at fixed compute**

Intersect Plot A at specific FLOP budgets to extract the optimal model size for each
compute level.

### 6.3 Fitting the parametric model

Fit the Chinchilla parametric form:

```
L(N, D) = E + A / N^alpha + B / D^beta
```

Where:
- `N` = encoder parameters (86M, 304M, 632M)
- `D` = total tokens seen = steps * effective_batch_size
- `L` = final validation loss (or loss at a fixed FLOP budget)
- `E, A, B, alpha, beta` = parameters to fit

**Fitting procedure (nonlinear least squares):**

```python
import numpy as np
from scipy.optimize import curve_fit

def chinchilla_loss(ND, E, A, alpha, B, beta):
    """Chinchilla parametric scaling law."""
    N, D = ND
    return E + A / (N ** alpha) + B / (D ** beta)

# Collect data points from all Tier 1 runs
# N_values: encoder param counts
# D_values: total tokens seen at evaluation point
# L_values: validation loss at that point
N_values = np.array([...])  # shape (num_points,)
D_values = np.array([...])  # shape (num_points,)
L_values = np.array([...])  # shape (num_points,)

# Initial guess
p0 = [0.5, 1.0, 0.3, 1.0, 0.3]

# Fit
popt, pcov = curve_fit(
    chinchilla_loss,
    (N_values, D_values),
    L_values,
    p0=p0,
    bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, 1, np.inf, 1]),
    maxfev=10000,
)

E_fit, A_fit, alpha_fit, B_fit, beta_fit = popt
perr = np.sqrt(np.diag(pcov))  # 1-sigma confidence intervals
```

### 6.4 Confidence intervals

Report 95% confidence intervals on all fitted parameters. If the confidence interval
on alpha or beta spans more than 0.1 (e.g., alpha = 0.30 +/- 0.08), the fit is
informative. If it spans more than 0.3 (e.g., alpha = 0.30 +/- 0.20), more data
points are needed -- run additional scaling experiments.

### 6.5 Determining the compute-optimal frontier

Given the fitted parameters, the compute-optimal allocation for a FLOP budget C is:

```
C = 6 * N * D  (approximate compute in FLOPs)

Minimize L(N, D) subject to 6*N*D = C

Using Lagrange multipliers:
    N_opt = (C / 6)^(beta / (alpha + beta)) * (alpha * A / (beta * B))^(1/(alpha+beta))
    D_opt = C / (6 * N_opt)
```

For the full budget C_full = 1.14e22 FLOPs (from Section 5.5), solve for N_opt and
D_opt to determine:
- Which ViT variant to use (Base / Large / Huge)
- How many tokens to train on (which determines steps given the batch size)

### 6.6 Alternative: IsoFLOP profiles

If the parametric fit is poor (high residuals), use the non-parametric approach from
Chinchilla Approach 1:

1. Group experiments by approximate FLOP budget
2. Within each group, find the model size with the lowest loss
3. Plot `N_opt` vs `C` -- this should follow a power law
4. Extrapolate to C_full

---

## 7. Checkpoint Strategy for Scaling

### 7.1 Principle: extract trends from partial runs

Running every scaling experiment to full convergence is wasteful. Instead, log the
loss at regular intervals and use the learning curve shape to extrapolate.

### 7.2 Checkpoint frequency

| Run duration | Checkpoint interval | Loss logging interval |
|-------------|--------------------|-----------------------|
| 25K steps | Every 2,500 steps | Every 100 steps |
| 50K steps | Every 5,000 steps | Every 100 steps |
| 100K steps | Every 10,000 steps | Every 200 steps |
| 200K steps | Every 10,000 steps | Every 200 steps |

Keep the last 3 checkpoints plus the best (lowest val loss) to manage disk space.
Each ViT-Base checkpoint is ~350 MB; ViT-Large ~1.2 GB; ViT-Huge ~2.5 GB.

### 7.3 Extrapolating convergence from partial training

Fit a power-law decay to the loss curve of each run:

```
L(t) = a * t^(-gamma) + L_inf
```

Where `t` is the step number. If the curve is well-fit (R^2 > 0.95), you can predict
the loss at steps you have not run. This lets you:

1. Predict the final loss of a 200K-step run from only 50K steps of data
2. Identify early whether a run is promising (steep gamma) or hopeless (flat gamma)
3. Decide to terminate bad runs early and reallocate GPU-hours

**Early stopping rule:** If after 25K steps the loss has not decreased by at least 5%
from the initial value (after warmup), terminate the run. This prevents wasting
compute on broken configurations.

### 7.4 What to save at each checkpoint

| Artifact | Size | Purpose |
|----------|------|---------|
| Model state_dict | 350MB-2.5GB | Resume training, analysis |
| Optimizer state | 700MB-5GB | Resume training only |
| Training log (full) | <10MB | Loss curves, gradient norms, LR |
| 5 reconstruction samples | <5MB | Visual quality assessment |
| Validation loss on 1000 images | Logged in training log | Scaling curve data point |

For scaling law experiments, **save only the model state_dict at the final checkpoint
and the last checkpoint**. Skip optimizer state to save disk space (we will not resume
these runs).

---

## 8. Hydra Multirun Configuration

### 8.1 Scaling sweep config

Create a new Hydra config group for scaling experiments. This overrides the base
config with model-size-specific parameters.

**File: `$MAXIE_DIR/train/hydra_config/train_config/scaling_base.yaml`**

```yaml
# Base config for scaling law experiments
# Override model-specific params via multirun

checkpoint:
  chkpt_saving_iterations: 5000
  directory: experiments/scaling/chkpts/${model_variant}/${data_fraction}
  prefix: scaling

dataset:
  batch_size: ${batch_per_gcd}
  num_workers: 4
  seg_size: 100
  drop_last_in_sampler: true
  drop_last_in_loader: true
  transforms:
    H_pad: 2048
    W_pad: 2048
    patch_size: 224
    stride: 224

dist:
  backend: nccl
  uses_unique_world_seed: true
  dtype: float16

logging:
  directory: experiments/scaling/logs/${model_variant}/${data_fraction}
  prefix: scaling
  level: info

loss:
  grad_accum_steps: 1

lr_scheduler:
  min_lr: !!float 1e-6
  # total_iterations set per run
  # warmup = 5% of total_iterations
  scheduler_update_iterations: 1

misc:
  max_epochs: 999999  # use step-based termination
  max_eval_iter: 20
  compiles_model: false
  data_dump_on: false
  peak_flops_per_sec: !!float 52e12
  monitors_dynamics: false

model:
  hf_config:
    image_size: 224
    patch_size: 16
    num_channels: 1
    mask_ratio: 0.75
    norm_pix_loss: true
    # hidden_size, num_hidden_layers, etc. set per model variant
  from_scratch: true

optim:
  grad_clip: 1.0
  # lr set per run via linear scaling rule
  weight_decay: !!float 0.05
  beta1: 0.9
  beta2: 0.95
  fused: false
```

**File: `$MAXIE_DIR/train/hydra_config/train_config/scaling_vit_base.yaml`**

```yaml
defaults:
  - scaling_base

model_variant: vit_base
batch_per_gcd: 16

model:
  hf_config:
    hidden_size: 768
    num_hidden_layers: 12
    num_attention_heads: 12
    intermediate_size: 3072
    hidden_act: gelu
    hidden_dropout_prob: 0.0
    attention_probs_dropout_prob: 0.0
    initializer_range: 0.02
    layer_norm_eps: 1.0e-12
    qkv_bias: true
    decoder_num_attention_heads: 16
    decoder_hidden_size: 512
    decoder_num_hidden_layers: 8
    decoder_intermediate_size: 2048
```

**File: `$MAXIE_DIR/train/hydra_config/train_config/scaling_vit_large.yaml`**

```yaml
defaults:
  - scaling_base

model_variant: vit_large
batch_per_gcd: 8

model:
  hf_config:
    hidden_size: 1024
    num_hidden_layers: 24
    num_attention_heads: 16
    intermediate_size: 4096
    hidden_act: gelu
    hidden_dropout_prob: 0.0
    attention_probs_dropout_prob: 0.0
    initializer_range: 0.02
    layer_norm_eps: 1.0e-12
    qkv_bias: true
    decoder_num_attention_heads: 16
    decoder_hidden_size: 512
    decoder_num_hidden_layers: 8
    decoder_intermediate_size: 2048
```

**File: `$MAXIE_DIR/train/hydra_config/train_config/scaling_vit_huge.yaml`**

```yaml
defaults:
  - scaling_base

model_variant: vit_huge
batch_per_gcd: 4

model:
  hf_config:
    hidden_size: 1280
    num_hidden_layers: 32
    num_attention_heads: 16
    intermediate_size: 5120
    hidden_act: gelu
    hidden_dropout_prob: 0.0
    attention_probs_dropout_prob: 0.0
    initializer_range: 0.02
    layer_norm_eps: 1.0e-12
    qkv_bias: true
    decoder_num_attention_heads: 16
    decoder_hidden_size: 512
    decoder_num_hidden_layers: 8
    decoder_intermediate_size: 2048
```

### 8.2 Hydra multirun command

To sweep across data fractions and step counts for a given model size:

```bash
# ViT-Base sweep (runs S01-S11)
python train.fsdp.py --multirun \
    train_config=scaling_vit_base \
    data_fraction=0.10,0.25,0.50,1.00 \
    lr_scheduler.total_iterations=25000,50000,100000,200000 \
    optim.lr=2.4e-3 \
    hydra.sweep.dir=experiments/scaling/sweeps/vit_base \
    hydra.launcher.partition=batch \
    hydra.launcher.nodes=4

# ViT-Large sweep (runs S12-S21)
python train.fsdp.py --multirun \
    train_config=scaling_vit_large \
    data_fraction=0.10,0.25,0.50,1.00 \
    lr_scheduler.total_iterations=25000,50000,100000,200000 \
    optim.lr=1.2e-3 \
    hydra.sweep.dir=experiments/scaling/sweeps/vit_large \
    hydra.launcher.partition=batch \
    hydra.launcher.nodes=8

# ViT-Huge sweep (runs S22-S30)
python train.fsdp.py --multirun \
    train_config=scaling_vit_huge \
    data_fraction=0.10,0.25,0.50,1.00 \
    lr_scheduler.total_iterations=25000,50000,100000,200000 \
    optim.lr=6.0e-4 \
    hydra.sweep.dir=experiments/scaling/sweeps/vit_huge \
    hydra.launcher.partition=batch \
    hydra.launcher.nodes=16
```

Note: Not all combinations of data_fraction x total_iterations are in the experiment
table. Some (e.g., 10% data with 200K steps) are omitted to save compute because the
model will have seen the full 10% subset many times over. The Hydra command above
generates the full Cartesian product; use a Hydra config to skip unwanted combinations,
or use a Flux-based launcher (Section 9) for finer control.

### 8.3 LR calculation

The learning rate for each run follows the linear scaling rule from the training playbook:

```
lr = base_lr * effective_batch_size / 256
```

Where `base_lr = 1.5e-4` and `effective_batch_size = nodes * 8 * batch_per_gcd`.

| Model | Nodes | Batch/GCD | Effective batch | LR |
|-------|-------|-----------|----------------|-----|
| ViT-Base | 4 | 16 | 512 | 3.0e-4 |
| ViT-Large | 8 | 8 | 512 | 3.0e-4 |
| ViT-Huge | 16 | 4 | 512 | 3.0e-4 |

All three configurations happen to have the same effective batch size (512) and
therefore the same learning rate. This is by design -- it isolates model size as the
variable by keeping the optimization landscape comparable.

Warmup: 5% of total_iterations (e.g., 1,250 steps for a 25K-step run, 10,000 steps
for a 200K-step run).

---

## 9. Flux Ensemble Execution

### 9.1 Why Flux

Scaling law experiments are independent runs that vary in size (4-16 nodes each). The
Frontier user guide (and `research-loop-frontier-strategy.md` Section 4) recommends
Flux for multi-node ensemble jobs within a single Slurm allocation. Flux provides:

- Local job queue (no Slurm controller overload)
- Automatic load-balancing (next job starts when a node frees)
- Tested at 500 nodes on Frontier
- Native GPU affinity support

### 9.2 Allocation strategy

To run the Tier 1 grid efficiently, group runs by model size and submit one large
Flux allocation per tier:

| Allocation | Nodes | Walltime | Runs | Purpose |
|-----------|-------|----------|------|---------|
| A1 | 16 | 6h | B01, S01-S11 | ViT-Base benchmarks + full sweep |
| A2 | 24 | 8h | B02, S12-S21 | ViT-Large benchmarks + full sweep |
| A3 | 32 | 10h | B03, S22-S30 | ViT-Huge benchmarks + full sweep |
| A4 | 8 | 3h | M01-M03, P01-P02 | Tier 2+3 secondary sweeps |

Total node-hours: 16*6 + 24*8 + 32*10 + 8*3 = 96 + 192 + 320 + 24 = 632 node-hours =
5,056 GPU-hours. Well within the 7,026 GPU-hour budget (the difference is contingency).

### 9.3 Flux job script: ViT-Base sweep (Allocation A1)

```bash
#!/bin/bash
#SBATCH -A lrn091
#SBATCH -J maxie_scaling_base
#SBATCH -o logs/scaling_base-%j.o
#SBATCH -e logs/scaling_base-%j.e
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH -N 16
#SBATCH -C nvme
#SBATCH --network=disable_rdzv_get

# --- Modules ---
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
module load hwloc/2.9.1-gpu
module load flux

# --- sbcast env to NVMe ---
sbcast -pf ./maxie_env.tar.gz /mnt/bb/${USER}/maxie_env.tar.gz
if [ ! "$?" == "0" ]; then
    echo "SBCAST failed!"
    exit 1
fi
srun -N ${SLURM_NNODES} --ntasks-per-node 1 mkdir -p /mnt/bb/${USER}/maxie_env
srun -N ${SLURM_NNODES} --ntasks-per-node 1 -c56 \
    tar --use-compress-program=pigz -xf /mnt/bb/${USER}/maxie_env.tar.gz \
    -C /mnt/bb/${USER}/maxie_env

# --- Environment variables ---
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDZV_PROTO=alt_read
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

TRAIN_SCRIPT="${MAXIE_DIR}/train/train.fsdp.py"
SCALING_DIR="experiments/scaling"

# --- Launch Flux scheduler across all 16 nodes ---
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 56 --gpus-per-node=8 flux start \
    "flux resource list;

    # -----------------------------------------------------------
    # B01: Throughput benchmark (4 nodes, 500 steps, ViT-Base)
    # -----------------------------------------------------------
    flux submit -N 4 -n 32 -c 7 --gpus-per-task=1 \
        -o gpu-affinity=per-task \
        --output=${SCALING_DIR}/logs/B01_benchmark.log \
        bash -c '
            export ROCR_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES
            unset CUDA_VISIBLE_DEVICES
            export MASTER_ADDR=\$(hostname -i)
            export MASTER_PORT=3442
            rm -rf ${MIOPEN_USER_DB_PATH} && mkdir -p ${MIOPEN_USER_DB_PATH}
            conda activate /mnt/bb/${USER}/maxie_env
            python3 -W ignore -u ${TRAIN_SCRIPT} \
                train_config=scaling_vit_base \
                data_fraction=0.10 \
                lr_scheduler.total_iterations=500 \
                --master_addr=\$MASTER_ADDR \
                --master_port=\$MASTER_PORT
        '

    # -----------------------------------------------------------
    # S01-S11: ViT-Base scaling grid
    # Each run uses 4 nodes (32 GCDs)
    # -----------------------------------------------------------
    declare -A RUNS
    RUNS[S01]=\"0.10 25000\"
    RUNS[S02]=\"0.10 50000\"
    RUNS[S03]=\"0.10 100000\"
    RUNS[S04]=\"0.25 25000\"
    RUNS[S05]=\"0.25 50000\"
    RUNS[S06]=\"0.25 100000\"
    RUNS[S07]=\"0.50 50000\"
    RUNS[S08]=\"0.50 100000\"
    RUNS[S09]=\"1.00 50000\"
    RUNS[S10]=\"1.00 100000\"
    RUNS[S11]=\"1.00 200000\"

    for RUN_ID in \${!RUNS[@]}; do
        read DATA_FRAC STEPS <<< \${RUNS[\$RUN_ID]}
        WARMUP=\$(( STEPS / 20 ))

        flux submit -N 4 -n 32 -c 7 --gpus-per-task=1 \
            -o gpu-affinity=per-task \
            --output=${SCALING_DIR}/logs/\${RUN_ID}_base_d\${DATA_FRAC}_s\${STEPS}.log \
            bash -c \"
                export ROCR_VISIBLE_DEVICES=\\\$CUDA_VISIBLE_DEVICES
                unset CUDA_VISIBLE_DEVICES
                export MASTER_ADDR=\\\$(hostname -i)
                export MASTER_PORT=3442
                rm -rf ${MIOPEN_USER_DB_PATH} && mkdir -p ${MIOPEN_USER_DB_PATH}
                conda activate /mnt/bb/${USER}/maxie_env
                python3 -W ignore -u ${TRAIN_SCRIPT} \
                    train_config=scaling_vit_base \
                    data_fraction=\${DATA_FRAC} \
                    lr_scheduler.total_iterations=\${STEPS} \
                    lr_scheduler.warmup_iterations=\${WARMUP} \
                    optim.lr=3.0e-4 \
                    --master_addr=\\\$MASTER_ADDR \
                    --master_port=\\\$MASTER_PORT
            \"
    done

    flux jobs -a;
    flux queue drain;"
```

### 9.4 Flux job script: ViT-Large sweep (Allocation A2)

The script follows the same pattern as A1 but with:
- `scaling_vit_large` config
- 8 nodes per run (-N 8 -n 64)
- Total allocation: 24 nodes
- Flux can run up to 3 concurrent ViT-Large jobs (24/8 = 3)

### 9.5 Flux job script: ViT-Huge sweep (Allocation A3)

Same pattern with:
- `scaling_vit_huge` config
- 16 nodes per run (-N 16 -n 128)
- Total allocation: 32 nodes
- Flux can run up to 2 concurrent ViT-Huge jobs (32/16 = 2)

### 9.6 GPU binding caveat

Per the Frontier user guide (Section 4.2 of `research-loop-frontier-strategy.md`),
Flux's `--gpus-per-task=1` sets `CUDA_VISIBLE_DEVICES` but may not correctly restrict
GPU visibility. The workaround in all scripts above copies `CUDA_VISIBLE_DEVICES` to
`ROCR_VISIBLE_DEVICES` and then unsets `CUDA_VISIBLE_DEVICES`.

---

## 10. From Scaling Laws to Campaign Design

### 10.1 Decision framework

After the scaling law study completes, the fitted parameters provide a decision
function:

```
Given: C_campaign = 106,470 GPU-hours (remaining after scaling study)
       10 campaigns planned
       C_per_run = 10,647 GPU-hours each

Step 1: Convert C_per_run to FLOPs
        C_flops = 10,647 * 8 GCDs * 26e12 FLOPS * 3600
        (adjust 26e12 using actual measured throughput from benchmarks B01-B03)

Step 2: Solve for N_opt, D_opt using fitted scaling law
        N_opt = f(C_flops, alpha, beta, A, B)
        D_opt = C_flops / (6 * N_opt)

Step 3: Map N_opt to a concrete model variant
        If N_opt < 150M  -> ViT-Base
        If 150M < N_opt < 450M -> ViT-Large
        If N_opt > 450M -> ViT-Huge

Step 4: Determine training steps
        steps = D_opt / effective_batch_size
        walltime = steps / measured_steps_per_hour

Step 5: Verify walltime fits in Frontier's 24h limit (extended partition)
        or 2h limit (batch partition)
```

### 10.2 How to use 10 campaigns

The 10 campaigns do not all have to use the same configuration. A recommended split:

| Campaigns | Purpose | Configuration |
|-----------|---------|---------------|
| 1-2 | Validate scaling prediction | Run the compute-optimal config predicted by the scaling law. Compare actual loss to predicted loss. |
| 3-5 | Explore the frontier | Vary model size by +/- 1 tier from optimal (e.g., if optimal is Large, run one Base and one Huge). Test sensitivity. |
| 6-7 | SSL method comparison | Run the best non-MAE candidate (likely I-JEPA or VQ-VAE) at the optimal model size for head-to-head comparison. |
| 8-9 | Hyperparameter refinement | Mask ratio, patch size, contrast-aware loss variants at the optimal model size. |
| 10 | Final production run | Best configuration from campaigns 1-9, trained to maximum tokens budget. This becomes the released checkpoint. |

### 10.3 What if scaling law predicts "need more data"?

The Chinchilla framework may reveal that the optimal D for a given compute budget
exceeds the available 104,840 frames. In this case:

1. **Reduce model size** to match available data (accept higher irreducible loss)
2. **Use data augmentation** to effectively increase D (random rotation, intensity
   scaling, crop variation -- but validate these do not degrade representation quality)
3. **Increase mask ratio** to create a harder task from the same data (each mask
   configuration is effectively a different training example)
4. **Acquire more data** -- flag to the PI that the data collection campaign should be
   accelerated

### 10.4 What if scaling law is inconclusive?

If the fitted power-law has large confidence intervals (alpha or beta confidence
interval > 0.3), the 30 runs in the core grid were insufficient. Options:

1. **Add interpolation points** -- run additional experiments at intermediate model
   sizes (e.g., ViT-Base with wider hidden dim) or intermediate data fractions (15%,
   35%, 75%)
2. **Run longer** -- extend the step counts for existing configurations to get loss
   values closer to convergence
3. **Use the IsoFLOP approach** (Section 6.6) instead of parametric fitting, which
   requires fewer assumptions
4. **Fall back to the Kaplan approach** -- use the model size that achieves the lowest
   loss at the largest tested FLOP budget, without trying to extrapolate

---

## 11. Key Files Reference

### Companion documents

| Document | Path | Relevant sections |
|----------|------|-------------------|
| Training playbook | `docs/agents/training-playbook-for-maxie.md` | Phase 2 (Section 11), model sizes (Section 13), batch scaling (Section 5) |
| SSL candidates | `docs/agents/ssl-candidates-for-maxie.md` | MAE architecture, model variants, HuggingFace checkpoints |
| Frontier strategy | `docs/agents/research-loop-frontier-strategy.md` | Flux ensembles (Section 4), sbcast (Section 2.1), RCCL tuning (Section 2.3) |
| Data pipeline | `docs/agents/data-pipeline-for-maxie.md` | Dataset sizes (Section 2.2), Zarr structure (Section 2.3), normalization (Section 4) |
| Monitoring protocol | `docs/agents/monitoring-protocol-for-maxie.md` | Per-step metrics (Section 2), validation metrics (Section 3), checkpoint diagnostics (Section 4) |

### MAXIE source files

| File | What it contains |
|------|-----------------|
| `$MAXIE_DIR/train/train.fsdp.py` | Main training loop (adapt for scaling experiments) |
| `$MAXIE_DIR/train/hydra_config/train_config/base.yaml` | Default Hydra config (base for scaling configs) |
| `$MAXIE_DIR/maxie/modeling/adapted_mae.py` | MAE model adapted for 1-channel input |
| `$MAXIE_DIR/maxie/datasets/zarr_dataset.py` | DistributedZarrDataset for Frontier |
| `$MAXIE_DIR/maxie/lr_scheduler.py` | CosineLRScheduler |
| `$MAXIE_DIR/maxie/utils/monitor.py` | ActivationMonitor, param update metrics |

### Reference implementations

| Resource | Path | Use |
|----------|------|-----|
| NeMo FLOPS formulas | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/NeMo/nemo/utils/flops_formulas.py` | FLOP calculation reference for GPT/BERT/LLaMA |
| CLIPA inverse scaling | `$DEEPLEARNING_DOC_DIR/clip/open_clip/docs/clipa.md` | Alternative scaling strategy for reference |
| Scaling tutorial (UvA) | `$DEEPLEARNING_DOC_DIR/uvadlc_notebooks/docs/tutorial_notebooks/scaling/JAX/overview.md` | Educational scaling context |
| d2l scaling chapter | `$DEEPLEARNING_DOC_DIR/dl-repos/d2l-en/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.md` | GPT-3 scaling law figures and context |
| Hydra multirun (NeMo) | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/NeMo/docs/source/core/exp_manager.rst` | Hydra multirun + sweep configuration patterns |
| HuggingFace MAE example | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/examples/pytorch/image-pretraining/run_mae.py` | MAE training script with LR scaling |

### Data

| Resource | Path | Size |
|----------|------|------|
| Assembled Zarr stores | `$XTAL_DATA_ASSEMBLED` (`/lustre/orion/lrn091/proj-shared/data/`) | ~2,621 stores, ~104K frames, ~1 TB |

---

## Appendix A: Quick Reference -- Scaling Study Execution Checklist

```
Phase 0: Preparation
  [ ] Create Hydra configs (scaling_base, scaling_vit_{base,large,huge})
  [ ] Implement data_fraction parameter in DistributedZarrDataset
  [ ] Implement step-based termination (max_steps instead of max_epochs)
  [ ] Set norm_pix_loss=true in all configs
  [ ] Pack conda environment for sbcast
  [ ] Create output directory structure on Lustre

Phase 1: Benchmarks (B01-B03)
  [ ] Run 500-step benchmarks for each model size
  [ ] Record: tokens/sec, step time, GPU memory, max batch_per_gcd
  [ ] Update throughput estimates in Section 4.3
  [ ] Verify RCCL environment variables work at target node counts

Phase 2: Tier 1 Core Grid (S01-S30)
  [ ] Submit Flux allocation A1 (ViT-Base, 16 nodes, 6h)
  [ ] Submit Flux allocation A2 (ViT-Large, 24 nodes, 8h)
  [ ] Submit Flux allocation A3 (ViT-Huge, 32 nodes, 10h)
  [ ] Monitor all runs for NaN/divergence, terminate bad runs early
  [ ] Collect validation loss at every checkpoint

Phase 3: Analysis
  [ ] Plot loss vs FLOPs curves (one per model size)
  [ ] Plot loss vs tokens seen curves
  [ ] Fit Chinchilla parametric model: L(N,D) = E + A/N^alpha + B/D^beta
  [ ] Report confidence intervals on alpha, beta
  [ ] Compute N_opt, D_opt for C = 10,647 GPU-hours per campaign
  [ ] If fit is poor: run additional experiments or use IsoFLOP approach

Phase 4: Tier 2+3 Secondary Sweeps (if budget allows)
  [ ] Submit Flux allocation A4 (mask ratio + patch size, 8 nodes, 3h)
  [ ] Analyze mask ratio impact on loss and reconstruction quality
  [ ] Analyze patch size impact on loss and throughput

Phase 5: Decision
  [ ] Choose model variant for campaigns
  [ ] Choose training duration for campaigns
  [ ] Choose data fraction for campaigns
  [ ] Document decision rationale
  [ ] Update training-playbook-for-maxie.md with chosen configuration
```

---

## Appendix B: Parameter Count Verification

To verify the parameter counts used in FLOP calculations, run:

```python
from transformers import ViTMAEConfig, ViTMAEForPreTraining

configs = {
    "ViT-Base": dict(hidden_size=768, num_hidden_layers=12,
                     num_attention_heads=12, intermediate_size=3072,
                     image_size=224, patch_size=16, num_channels=1),
    "ViT-Large": dict(hidden_size=1024, num_hidden_layers=24,
                      num_attention_heads=16, intermediate_size=4096,
                      image_size=224, patch_size=16, num_channels=1),
    "ViT-Huge": dict(hidden_size=1280, num_hidden_layers=32,
                     num_attention_heads=16, intermediate_size=5120,
                     image_size=224, patch_size=16, num_channels=1),
}

for name, kwargs in configs.items():
    cfg = ViTMAEConfig(**kwargs, decoder_hidden_size=512,
                       decoder_num_hidden_layers=8,
                       decoder_num_attention_heads=16,
                       decoder_intermediate_size=2048)
    model = ViTMAEForPreTraining(cfg)
    total = sum(p.numel() for p in model.parameters())
    encoder = sum(p.numel() for n, p in model.named_parameters()
                  if 'decoder' not in n)
    decoder = total - encoder
    print(f"{name}: encoder={encoder/1e6:.1f}M, "
          f"decoder={decoder/1e6:.1f}M, total={total/1e6:.1f}M")
```

Run this on a login node to get exact counts before starting the study.
