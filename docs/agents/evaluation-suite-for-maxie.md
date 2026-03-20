# Downstream Evaluation Suite for MAXIE

Date: 2026-03-19
Context: ALCC project Group 1 -- MAXIE foundation model (ViT-MAE pre-trained on
X-ray crystallography diffraction images). This document specifies the full
downstream evaluation suite: task definitions, evaluation protocols, metric
formulas, the 400-run campaign design, statistical analysis plan, and script
specifications.

Companion documents:
- `ssl-candidates-for-maxie.md` -- SSL method candidates and evaluation metrics
- `training-playbook-for-maxie.md` -- training recipe, fine-tuning hyperparameters
- `monitoring-protocol-for-maxie.md` -- per-checkpoint diagnostics, JSONL format
- `research-loop-frontier-strategy.md` -- Flux ensemble jobs, Frontier launch patterns
- `data-pipeline-for-maxie.md` -- dataset design, Zarr stores, transforms

---

## 1. Evaluation Goals

The evaluation suite answers three questions in decreasing order of priority:

1. **Which SSL method produces the best representations for X-ray diffraction?**
   Compare MAE, VQ-VAE, and I-JEPA (the three Tier-1 candidates from
   `ssl-candidates-for-maxie.md`) on identical downstream tasks.

2. **How does representation quality scale with compute and data?**
   Track downstream performance across the scaling law study (Phase 2 of the
   training playbook): model size (ViT-Base, Large, Huge), dataset fraction
   (10%, 25%, 50%, 100%), and training checkpoint (early, mid, late).

3. **Are the learned representations useful for real crystallography tasks?**
   Validate that pre-trained encoders transfer to tasks scientists actually
   care about: finding Bragg peaks, classifying crystal systems, detecting
   anomalies, and denoising detector images.

### What "good representations" means for diffraction data

A good encoder should produce embeddings where:
- Images from the same crystal system cluster together (semantic structure)
- Bragg peak spatial structure is preserved (spatial precision)
- The representation generalizes across detectors and experimental conditions
- Few labeled examples suffice for high downstream accuracy (label efficiency)

---

## 2. Downstream Tasks

### 2.1 Peak Finding (Bragg Peak Detection)

**Definition:** Given a diffraction image, produce a binary mask indicating
which pixels belong to Bragg peaks.

**Task type:** Dense prediction (semantic segmentation, 2 classes: peak vs background).

**Label source:** PeakNet labels (`$PEAKNET_DIR`). PeakNet is a trained peak
finder that produces per-pixel binary masks. These serve as pseudo-ground-truth
for evaluating MAXIE representations.

**Evaluation head:** Attach a lightweight decoder (1x1 conv or small UPN) to
the frozen ViT encoder's patch embeddings. Each patch embedding maps to a
peak/no-peak prediction for that spatial region.

**Metrics:**
- Pixel-level precision, recall, F1
- Peak-level detection rate: fraction of true peaks that overlap with at
  least one predicted peak pixel
- False discovery rate per image

**Why it matters:** Peak finding is the central downstream task in serial
femtosecond crystallography. If the encoder learns meaningful spatial
structure, a simple head should locate peaks from frozen features.

### 2.2 Crystal System Classification

**Definition:** Classify a diffraction image into one of 7 crystal systems
(triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic)
based on the symmetry of its diffraction pattern.

**Task type:** Image-level classification (7 classes).

**Label source:** Derived from experiment metadata. Each experiment in the
Zarr store has an `experiment_id` that maps to a known crystal structure.
Labels must be curated from the experimental database.

**Evaluation head:** Linear classifier on CLS token or mean-pooled patch
embeddings.

**Metrics:**
- Top-1 accuracy
- Macro-averaged F1 (accounts for class imbalance)
- Per-class accuracy (some crystal systems are rarer)
- Confusion matrix

**Why it matters:** Crystal system determination is a fundamental step in
structure solution. If the encoder captures lattice symmetry from the
diffraction pattern geometry, classification should be possible with minimal
labeled data.

### 2.3 Hit/Miss Classification

**Definition:** Binary classification -- does the image contain usable
diffraction signal (hit) or not (miss)?

**Task type:** Image-level binary classification.

**Label source:** Derived from peak count thresholds. An image is a "hit" if
it contains more than N Bragg peaks (typical threshold: N >= 15). PeakNet peak
counts or manual hit lists from beamline scientists provide labels.

**Evaluation head:** Linear classifier on CLS token or mean-pooled patch
embeddings.

**Metrics:**
- Accuracy, precision, recall, F1
- AUC-ROC (the threshold-free metric)
- Hit rate at fixed false positive rate (e.g., hit rate at FPR=1%)

**Why it matters:** Hit/miss filtering is the first processing step at
high-repetition-rate XFEL beamlines. A fast, accurate classifier reduces data
volumes by 10-100x. This is the simplest downstream task and serves as a
sanity check -- any reasonable encoder should achieve high accuracy here.

### 2.4 Anomaly Detection

**Definition:** Identify images with detector artifacts (hot pixels, dead
regions, ice rings, beam stop shadows, saturation, powder rings) vs clean
diffraction patterns.

**Task type:** Can be framed as binary classification (anomalous vs clean) or
as out-of-distribution detection using embedding distances.

**Label source:** Requires manual curation. A subset of images should be
labeled as anomalous with the specific anomaly type. Alternatively, train on
clean images only and flag images whose embeddings are far from the learned
distribution.

**Evaluation head (supervised):** Linear classifier on frozen embeddings.

**Evaluation head (unsupervised):** Compute Mahalanobis distance of each
test embedding from the clean-image embedding distribution. Threshold for
anomaly detection.

**Metrics:**
- AUC-ROC
- Precision at recall=0.95 (high recall is critical -- missing anomalies
  corrupts downstream analysis)
- Per-anomaly-type detection rate

**Why it matters:** Detector artifacts contaminate structure determination.
Automated anomaly detection prevents bad data from propagating through the
analysis pipeline.

### 2.5 Denoising / Reconstruction Quality

**Definition:** Evaluate how well the encoder-decoder pipeline reconstructs
the original diffraction image from a corrupted or masked input.

**Task type:** Image-to-image regression. Applicable only to MAE (pixel
reconstruction) and VQ-VAE (encode-decode). I-JEPA does not reconstruct
pixels, so this task is skipped for I-JEPA.

**Evaluation input:** Original images with synthetic corruption:
- Masking: 50%, 75%, 90% patch masking (for MAE)
- Noise injection: Gaussian noise at sigma = 0.1, 0.5, 1.0 of image std
- Partial occlusion: simulate beam stop or detector gaps

**Metrics:** See Section 6 for the full metric specification.

---

## 3. Linear Probe Protocol

The linear probe is the standard evaluation for self-supervised representation
quality. It measures what information is linearly accessible in the frozen
encoder's output.

### 3.1 Procedure

```
Step 1: Load pre-trained encoder checkpoint
Step 2: Freeze all encoder parameters (requires_grad = False)
Step 3: Extract features for all images in the labeled dataset
        - For ViT: use CLS token (shape: embed_dim)
        - Alternative: mean-pool all patch embeddings (shape: embed_dim)
        - For VQ-VAE: flatten the quantized spatial grid to a vector
Step 4: Train a single linear layer on top of frozen features
Step 5: Evaluate on held-out test set
Step 6: Report accuracy and log to JSONL
```

### 3.2 Feature Extraction

```python
@torch.no_grad()
def extract_features(encoder, dataloader, device, pool="cls"):
    """Extract frozen features from a pre-trained encoder.

    Args:
        encoder: Pre-trained ViT encoder (eval mode, frozen).
        dataloader: Yields (images, labels) batches.
        device: Target device.
        pool: "cls" for CLS token, "mean" for mean-pooled patches.

    Returns:
        features: Tensor of shape (N, embed_dim).
        labels: Tensor of shape (N,).
    """
    encoder.eval()
    all_features = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = encoder(images, output_hidden_states=True)

        if pool == "cls":
            features = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif pool == "mean":
            features = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        else:
            raise ValueError(f"Unknown pool method: {pool}")

        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)
```

### 3.3 Linear Head Training

```python
class LinearProbe(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(x.detach())  # stop gradient to encoder
```

### 3.4 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | SGD with momentum | Standard for linear probes (Adam gives similar results but SGD is conventional) |
| Learning rate | 0.1 | Standard starting point; grid search over {0.01, 0.1, 1.0} if needed |
| Momentum | 0.9 | Standard |
| Weight decay | 0.0 | No regularization on the linear head (the encoder is frozen) |
| Epochs | 100 | Sufficient for convergence of a single linear layer |
| Batch size | 256 | Standard; adjust for memory |
| LR schedule | Cosine decay to 0 | Standard |
| Data augmentation | None | The probe should evaluate raw representation quality |
| Normalization | L2-normalize features before probe (optional; report both) | Common practice in DINO/DINOv2 evaluations |
| Train/val/test split | 60% / 20% / 20% stratified by class | Ensure all classes are represented |

### 3.5 Running for Each SSL Method

| SSL Method | Encoder | Feature location | embed_dim (Base/Large/Huge) |
|------------|---------|------------------|-----------------------------|
| MAE | `model.vit` (the encoder, without decoder) | CLS token or mean-pooled patches | 768 / 1024 / 1280 |
| VQ-VAE | CNN encoder (before quantization) | Global average pool of spatial grid | Depends on architecture |
| I-JEPA | Context encoder | CLS token or mean-pooled patches | 768 / 1024 / 1280 |

For MAE, the decoder is discarded after pre-training. Only the ViT encoder
is used. Load the checkpoint, extract `model.vit`, freeze, and probe.

### 3.6 What the Linear Probe Tells You

- **High linear probe accuracy:** The representation has linearly separable
  structure for the downstream task. Good for simple classifiers.
- **Low linear probe, high k-NN accuracy:** The representation has structure,
  but it is non-linear. May still be useful with a non-linear head.
- **Low linear probe, low k-NN accuracy:** The representation does not encode
  the task-relevant information. The pre-training may need more data, more
  compute, or a different pretext task.

---

## 4. k-NN Evaluation Protocol

k-NN evaluation complements the linear probe. It requires no training, has no
hyperparameters to tune (beyond k and distance metric), and directly measures
the geometric structure of the embedding space.

### 4.1 Procedure

```
Step 1: Extract features for all images (same as linear probe Step 3)
Step 2: L2-normalize all feature vectors
Step 3: For each test image:
        a. Compute cosine similarity (or L2 distance) to all training images
        b. Find the k nearest neighbors in the training set
        c. Predict the majority label among k neighbors
Step 4: Report accuracy
```

### 4.2 Hyperparameters

| Parameter | Values to report | Rationale |
|-----------|-----------------|-----------|
| k | 1, 5, 10, 20, 200 | k=1 is most sensitive to local structure; k=20 is more robust; k=200 is the DINO default |
| Distance metric | Cosine similarity (after L2 normalization) | Standard for ViT features; equivalent to L2 distance on unit sphere |
| Feature normalization | L2-normalize all features | Required for cosine similarity |
| Weighted voting | Yes (weight by 1/distance) | Reduces noise from distant neighbors |

### 4.3 Implementation

```python
@torch.no_grad()
def knn_evaluate(train_features, train_labels, test_features, test_labels,
                 k_values=(1, 5, 10, 20, 200), temperature=0.07):
    """Evaluate representations using weighted k-NN.

    Args:
        train_features: (N_train, D) L2-normalized.
        train_labels: (N_train,) integer labels.
        test_features: (N_test, D) L2-normalized.
        test_labels: (N_test,) integer labels.
        k_values: tuple of k values to evaluate.
        temperature: Softmax temperature for weighted voting.

    Returns:
        dict mapping k -> accuracy.
    """
    num_classes = train_labels.max().item() + 1
    results = {}

    # Compute similarity matrix: (N_test, N_train)
    similarity = test_features @ train_features.T

    for k in k_values:
        # Top-k neighbors
        topk_sim, topk_idx = similarity.topk(k, dim=1)  # (N_test, k)
        topk_labels = train_labels[topk_idx]              # (N_test, k)

        # Weighted voting: weight by exp(sim / temperature)
        weights = torch.exp(topk_sim / temperature)       # (N_test, k)

        # Accumulate weighted votes per class
        votes = torch.zeros(test_features.size(0), num_classes,
                            device=test_features.device)
        votes.scatter_add_(1, topk_labels, weights)

        # Predict majority class
        predictions = votes.argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean().item()
        results[k] = accuracy

    return results
```

### 4.4 What k-NN Tells You vs Linear Probe

| | Linear Probe | k-NN |
|---|---|---|
| Requires training | Yes (100 epochs SGD) | No |
| Captures non-linear structure | No | Partially (local geometry) |
| Sensitive to embedding scale | No (learns scaling) | Yes (needs L2 norm) |
| Cost | ~10 min per evaluation | ~1 min (just matrix multiply) |
| Use case | Main evaluation metric | Quick sanity check; monitor during training |

---

## 5. Few-Shot Evaluation

Few-shot evaluation measures label efficiency: how well does the representation
perform when only a handful of labeled examples are available? This directly
addresses the reality of X-ray crystallography, where labeled datasets are
small and expert annotation is expensive.

### 5.1 Protocol

For each few-shot setting (1-shot, 5-shot, 10-shot):

```
Step 1: Extract features (same as linear probe)
Step 2: For each of R episodes (R = 100 for statistical reliability):
        a. Randomly sample N examples per class from the training set
           (N = 1, 5, or 10)
        b. The remaining training examples are discarded for this episode
        c. Train a linear probe (or use nearest-centroid) on the N*C samples
        d. Evaluate on the full test set
        e. Record accuracy
Step 3: Report mean and 95% confidence interval across R episodes
```

### 5.2 Classification Methods for Few-Shot

| Method | When to use | Hyperparameters |
|--------|-------------|-----------------|
| **Nearest centroid** | 1-shot (too few samples for SGD) | None; compute class mean, predict nearest |
| **Linear probe (SGD)** | 5-shot and above | Same as Section 3.4 but fewer epochs (50) and lower LR (0.01) |
| **Logistic regression (sklearn)** | All settings; serves as reference | C=1.0, max_iter=1000, L-BFGS |

### 5.3 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Episodes (R) | 100 |
| Shots (N) per class | 1, 5, 10 |
| Test set | Full test split (fixed across episodes) |
| Feature normalization | L2-normalize |

### 5.4 Why This Matters for X-ray Data

In practice, a new beamline experiment may produce millions of unlabeled
diffraction images but only a few hundred manually labeled ones. A
representation that achieves 90% accuracy with 10 labels per class is far more
valuable than one that needs 10,000 labels. Few-shot performance is the most
operationally relevant metric for MAXIE.

### 5.5 Reporting Format

```
Task: hit_miss_classification
SSL Method: MAE (ViT-Base, checkpoint epoch 200)
1-shot: 78.3% +/- 3.2% (100 episodes)
5-shot: 89.1% +/- 1.4% (100 episodes)
10-shot: 92.7% +/- 0.8% (100 episodes)
```

---

## 6. Reconstruction Quality Metrics

These metrics apply only to SSL methods that reconstruct pixels: **MAE** and
**VQ-VAE**. I-JEPA predicts in representation space and does not produce pixel
reconstructions.

### 6.1 Standard Image Quality Metrics

**Mean Squared Error (MSE):**

```
MSE = (1/N) * sum((x_i - x_hat_i)^2)
```

where x is the original image and x_hat is the reconstruction. Lower is better.

**Peak Signal-to-Noise Ratio (PSNR):**

```
PSNR = 10 * log10(MAX^2 / MSE)    [in dB]
     = -10 * log10(MSE / MAX^2)
```

where MAX is the maximum possible pixel value. For normalized images (range
[0, 1]), MAX = 1.0, so `PSNR = -10 * log10(MSE)`.

Interpretation for diffraction images:
- PSNR < 20 dB: Poor reconstruction; major structural loss
- PSNR 20-30 dB: Moderate; global structure preserved, fine detail lost
- PSNR 30-40 dB: Good; most Bragg peaks correctly reconstructed
- PSNR > 40 dB: Excellent; near-lossless

**Structural Similarity Index (SSIM):**

```
SSIM(x, y) = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
             / ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))
```

where C1 = (0.01 * L)^2, C2 = (0.03 * L)^2, L = dynamic range. Computed
in a sliding window (default 11x11). Range: [-1, 1], where 1 = identical.

SSIM captures structural (not just pixel-level) similarity, which is important
for diffraction patterns where the spatial arrangement of peaks matters more
than exact intensity values.

### 6.2 Domain-Specific Metrics for Diffraction

**Bragg Peak SNR Preservation:**

```
peak_snr(img) = mean(intensity[peak_pixels]) / std(intensity[bg_pixels])

snr_ratio = peak_snr(reconstruction) / peak_snr(original)
```

A snr_ratio near 1.0 means the reconstruction preserves the signal-to-noise
characteristics of Bragg peaks. Values < 0.8 indicate the reconstruction
is blurring or attenuating peaks.

**Peak Detection Rate:**

```
Run PeakNet (or threshold-based peak finder) on both the original and
reconstructed image. Count matched peaks.

peak_detection_rate = |peaks_recon intersect peaks_orig| / |peaks_orig|
```

A peak is "matched" if a predicted peak in the reconstruction falls within
r pixels of a true peak in the original (r = 5 pixels, matching PeakNet's
typical peak radius).

This directly measures whether the reconstruction preserves the features
that matter for downstream crystallography.

**Intensity Fidelity (per peak):**

```
For each matched peak p:
    fidelity_p = |I_recon(p) - I_orig(p)| / I_orig(p)

mean_intensity_fidelity = 1 - mean(fidelity_p)
```

Range: [0, 1], where 1 = perfect intensity preservation. Intensity fidelity
matters because integrated peak intensities are the input to structure
determination algorithms (e.g., CrystFEL's `process_hkl`).

**Reconstruction Error Decomposition:**

As specified in `monitoring-protocol-for-maxie.md` Section 4.3:

```python
def decomposed_recon_error(original, reconstructed):
    error = (original - reconstructed) ** 2
    threshold = original.mean() + 3 * original.std()
    peak_mask = original > threshold
    bg_mask = ~peak_mask

    peak_mse = error[peak_mask].mean().item() if peak_mask.any() else 0.0
    bg_mse = error[bg_mask].mean().item() if bg_mask.any() else 0.0
    return {"recon_err_peak": peak_mse, "recon_err_bg": bg_mse}
```

### 6.3 Implementation

```python
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_skimage

def compute_psnr(original, reconstructed, max_val=1.0):
    """Compute PSNR in dB."""
    mse = F.mse_loss(reconstructed, original).item()
    if mse == 0:
        return float('inf')
    return 10.0 * torch.log10(torch.tensor(max_val**2 / mse)).item()

def compute_ssim(original, reconstructed):
    """Compute SSIM using skimage (handles single-channel)."""
    orig_np = original.squeeze().cpu().numpy()
    recon_np = reconstructed.squeeze().cpu().numpy()
    return ssim_skimage(orig_np, recon_np, data_range=orig_np.max() - orig_np.min())

def compute_all_recon_metrics(original, reconstructed):
    """Compute all reconstruction quality metrics."""
    mse = F.mse_loss(reconstructed, original).item()
    psnr = compute_psnr(original, reconstructed)
    ssim_val = compute_ssim(original, reconstructed)
    decomposed = decomposed_recon_error(original, reconstructed)

    return {
        "recon_mse": mse,
        "recon_psnr_db": psnr,
        "recon_ssim": ssim_val,
        **decomposed,
    }
```

### 6.4 Summary Table

| Metric | Formula | Range | Best | Applicable to |
|--------|---------|-------|------|---------------|
| MSE | mean((x - x_hat)^2) | [0, inf) | 0 | MAE, VQ-VAE |
| PSNR | -10*log10(MSE) | (0, inf) dB | Higher | MAE, VQ-VAE |
| SSIM | structural similarity | [-1, 1] | 1 | MAE, VQ-VAE |
| Bragg Peak SNR Ratio | snr(recon) / snr(orig) | [0, inf) | 1.0 | MAE, VQ-VAE |
| Peak Detection Rate | matched_peaks / total_peaks | [0, 1] | 1.0 | MAE, VQ-VAE |
| Intensity Fidelity | 1 - mean(abs(I_err)/I_orig) | [0, 1] | 1.0 | MAE, VQ-VAE |
| Peak MSE | MSE on peak pixels only | [0, inf) | 0 | MAE, VQ-VAE |
| Background MSE | MSE on background pixels | [0, inf) | 0 | MAE, VQ-VAE |

---

## 7. The 400-Run Evaluation Campaign

The ALCC proposal allocates a 400-run downstream evaluation campaign. This
section specifies the sweep design, run allocation, and Flux job orchestration.

### 7.1 Sweep Axes

| Axis | Values | Count |
|------|--------|-------|
| SSL method | MAE, VQ-VAE, I-JEPA | 3 |
| Model size | ViT-Base, ViT-Large, ViT-Huge | 3 |
| Training checkpoint | Early (25%), Mid (50%), Late (100%) | 3 |
| Downstream task | Peak finding, Crystal system, Hit/miss, Anomaly detection | 4 |
| Evaluation protocol | Linear probe, k-NN, 10-shot | 3 |

Full Cartesian product: 3 x 3 x 3 x 4 x 3 = **324 runs**.

Remaining budget: 400 - 324 = **76 runs** reserved for:
- Reconstruction quality evaluation: 3 methods x 3 sizes x 3 checkpoints =
  27 runs (only MAE + VQ-VAE = 18 runs, since I-JEPA has no reconstruction)
- Ablations: feature pooling (CLS vs mean), L2 normalization effect = ~20 runs
- Reruns for statistical confidence on key comparisons = ~38 runs

### 7.2 Run Allocation Summary

| Category | Runs | Purpose |
|----------|------|---------|
| Core sweep (324) | 324 | Full method x model x checkpoint x task x protocol |
| Reconstruction metrics | 18 | MAE + VQ-VAE pixel quality assessment |
| Feature extraction ablations | 20 | CLS vs mean pool, L2 norm on/off |
| Statistical reruns | 38 | Repeat top-3 configs 5x each + buffer |
| **Total** | **400** | |

### 7.3 Resource Estimate per Run

Each evaluation run is a feature extraction + probe training job:

| Phase | GPU time (1 GCD) | Memory |
|-------|-------------------|--------|
| Feature extraction (full dataset, ~100K images) | ~10 min | ~20 GB |
| Linear probe training (100 epochs, features in RAM) | ~5 min | ~4 GB |
| k-NN evaluation (matrix multiply) | ~2 min | ~8 GB |
| Few-shot evaluation (100 episodes) | ~10 min | ~4 GB |
| **Total per run** | **~30 min** | **~20 GB peak** |

Each run fits on 1 node (8 GCDs). Feature extraction is the bottleneck and
can use all 8 GCDs in data-parallel mode. Probe training uses 1 GCD.

### 7.4 Flux Ensemble Job Template

Based on `research-loop-frontier-strategy.md` Section 4.2, using Flux to
manage 400 jobs within a single Slurm allocation.

```bash
#!/bin/bash
#SBATCH -A lrn091
#SBATCH -J maxie_eval_400
#SBATCH -o logs/eval_sweep-%j.o
#SBATCH -e logs/eval_sweep-%j.e
#SBATCH -t 06:00:00
#SBATCH -p batch
#SBATCH -N 50

# --- Modules ---
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0
module load hwloc/2.9.1-gpu
module load flux

# --- Environment ---
export MAXIE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/maxie
export PEAKNET_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/peaknet
export UV_CACHE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/.UV_CACHE
export EVAL_OUTPUT_DIR=/lustre/orion/lrn091/proj-shared/cwang31/eval_results
mkdir -p $EVAL_OUTPUT_DIR/logs

# --- MIOpen cache ---
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

# --- Generate sweep configs ---
# sweep_configs.json contains 400 entries, each specifying:
#   ssl_method, model_size, checkpoint_path, downstream_task, eval_protocol
SWEEP_CONFIG=$EVAL_OUTPUT_DIR/sweep_configs.json

# --- Launch Flux scheduler across all 50 nodes ---
srun -N $SLURM_NNODES -n $SLURM_NNODES -c 56 --gpus-per-node=8 flux start \
    "flux resource list;

    # Submit 400 evaluation runs, each using 1 node
    for config_id in \$(seq 0 399); do
        flux submit -N 1 -n 8 -c 7 --gpus-per-task=1 \
            -o gpu-affinity=per-task \
            --output=$EVAL_OUTPUT_DIR/logs/eval_\${config_id}.log \
            --error=$EVAL_OUTPUT_DIR/logs/eval_\${config_id}.err \
            bash -c '
                export ROCR_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES
                unset CUDA_VISIBLE_DEVICES

                rm -rf /tmp/my-miopen-cache
                mkdir -p /tmp/my-miopen-cache
                export MIOPEN_USER_DB_PATH=/tmp/my-miopen-cache
                export MIOPEN_CUSTOM_CACHE_DIR=/tmp/my-miopen-cache

                python3 $EVAL_OUTPUT_DIR/eval_maxie_downstream.py \
                    --sweep-config $SWEEP_CONFIG \
                    --config-id '\$config_id' \
                    --output-dir $EVAL_OUTPUT_DIR/results
            '
    done;

    echo 'All 400 jobs submitted. Waiting for completion...';
    flux queue drain;
    echo 'All jobs completed.';"
```

**Timing estimate:** 400 runs / 50 nodes = 8 rounds of 50 concurrent jobs.
At ~30 min per run, total wall time is ~4 hours (within the 6-hour allocation,
leaving 2 hours of buffer).

### 7.5 Sweep Configuration Generator

```python
"""Generate sweep_configs.json for the 400-run evaluation campaign."""
import json
import itertools

SSL_METHODS = ["mae", "vqvae", "ijepa"]
MODEL_SIZES = ["base", "large", "huge"]
CHECKPOINTS = ["early", "mid", "late"]  # mapped to actual paths at runtime
TASKS = ["peak_finding", "crystal_system", "hit_miss", "anomaly_detection"]
PROTOCOLS = ["linear_probe", "knn", "few_shot_10"]

# Core sweep: 324 configs
configs = []
for ssl, size, ckpt, task, protocol in itertools.product(
    SSL_METHODS, MODEL_SIZES, CHECKPOINTS, TASKS, PROTOCOLS
):
    configs.append({
        "config_id": len(configs),
        "ssl_method": ssl,
        "model_size": size,
        "checkpoint": ckpt,
        "task": task,
        "protocol": protocol,
        "category": "core_sweep",
    })

# Reconstruction metrics: 18 configs (MAE + VQ-VAE only)
for ssl in ["mae", "vqvae"]:
    for size in MODEL_SIZES:
        for ckpt in CHECKPOINTS:
            configs.append({
                "config_id": len(configs),
                "ssl_method": ssl,
                "model_size": size,
                "checkpoint": ckpt,
                "task": "reconstruction",
                "protocol": "recon_metrics",
                "category": "reconstruction",
            })

# Feature ablations: 20 configs
for ssl in SSL_METHODS:
    for pool in ["cls", "mean"]:
        for l2_norm in [True, False]:
            if len(configs) < 362:  # budget cap
                configs.append({
                    "config_id": len(configs),
                    "ssl_method": ssl,
                    "model_size": "base",
                    "checkpoint": "late",
                    "task": "hit_miss",
                    "protocol": "linear_probe",
                    "pool_method": pool,
                    "l2_normalize": l2_norm,
                    "category": "ablation",
                })

# Remaining budget -> statistical reruns
# (filled at runtime based on top configs from core sweep)
while len(configs) < 400:
    configs.append({
        "config_id": len(configs),
        "category": "reserved_rerun",
        "ssl_method": None,
        "model_size": None,
        "checkpoint": None,
        "task": None,
        "protocol": None,
    })

with open("sweep_configs.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Generated {len(configs)} configs")
```

---

## 8. Statistical Analysis

### 8.1 How Many Runs for Significance

Each evaluation config produces a single accuracy number. To determine whether
method A is better than method B, we need confidence intervals.

**For linear probe and k-NN:** These are deterministic given the data split.
Variance comes from the data split itself. Use 5-fold cross-validation to
estimate variance:

```
For each of 5 folds:
    Train probe on 4 folds, evaluate on held-out fold
Report: mean accuracy +/- std across 5 folds
```

**For few-shot:** Variance comes from the random episode sampling. 100
episodes give tight confidence intervals (typical std: 1-3% for 5-shot).

### 8.2 Confidence Intervals

Report 95% confidence intervals using the bootstrap or normal approximation:

```
CI_95 = mean +/- 1.96 * (std / sqrt(n))
```

where n = number of folds (5 for cross-validation) or episodes (100 for
few-shot).

### 8.3 Statistical Tests for Method Comparison

**Paired comparison (Method A vs Method B on same tasks):**

Use the **paired t-test** or **Wilcoxon signed-rank test** across tasks:

```python
from scipy import stats

# accuracies_A and accuracies_B: arrays of accuracy for each task
# (e.g., 4 tasks x 3 model sizes = 12 paired observations)
t_stat, p_value = stats.ttest_rel(accuracies_A, accuracies_B)

# Non-parametric alternative (fewer assumptions):
w_stat, p_value = stats.wilcoxon(accuracies_A, accuracies_B)
```

**Multi-method comparison (MAE vs VQ-VAE vs I-JEPA):**

Use the **Friedman test** (non-parametric repeated-measures ANOVA) followed by
**Nemenyi post-hoc test** for pairwise comparisons:

```python
from scipy.stats import friedmanchisquare

# Each row = one task/config, each column = one method
stat, p_value = friedmanchisquare(mae_accs, vqvae_accs, ijepa_accs)
# If p < 0.05, proceed to Nemenyi post-hoc
```

### 8.4 Multiple Comparisons Correction

With 3 methods compared across 4 tasks and 3 model sizes, there are many
pairwise comparisons. Apply the **Bonferroni correction** or the less
conservative **Benjamini-Hochberg** (FDR) procedure:

```python
from statsmodels.stats.multitest import multipletests

# p_values: array of p-values from all pairwise comparisons
reject, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
```

### 8.5 Effect Size

Beyond p-values, report **Cohen's d** for practical significance:

```
d = (mean_A - mean_B) / pooled_std
```

Interpretation:
- d < 0.2: negligible difference
- 0.2 <= d < 0.5: small effect
- 0.5 <= d < 0.8: medium effect
- d >= 0.8: large effect

### 8.6 Reporting Template

For each SSL method comparison, produce a table:

```
| Task              | MAE (Base)    | VQ-VAE (Base) | I-JEPA (Base) | p-value (Friedman) |
|-------------------|---------------|---------------|---------------|--------------------|
| Peak finding (F1) | 0.82 +/- 0.03| 0.79 +/- 0.04| 0.84 +/- 0.02| 0.023*             |
| Crystal system    | 0.71 +/- 0.05| 0.68 +/- 0.06| 0.73 +/- 0.04| 0.041*             |
| Hit/miss (AUC)    | 0.96 +/- 0.01| 0.94 +/- 0.02| 0.97 +/- 0.01| 0.008**            |
| Anomaly (AUC)     | 0.88 +/- 0.04| 0.85 +/- 0.05| 0.90 +/- 0.03| 0.034*             |
```

* p < 0.05, ** p < 0.01 after Bonferroni correction.

---

## 9. Evaluation Scripts Specification

### 9.1 Script Inventory

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `generate_sweep_configs.py` | Generate sweep_configs.json | Command-line args for checkpoint paths | `sweep_configs.json` |
| `extract_features.py` | Extract frozen encoder features | Checkpoint path, dataset path, pool method | `features_{config_id}.pt` (features + labels) |
| `eval_linear_probe.py` | Train and evaluate linear probe | Features file, task labels, hyperparams | JSONL metrics record |
| `eval_knn.py` | k-NN evaluation | Features file, k values | JSONL metrics record |
| `eval_few_shot.py` | Few-shot evaluation | Features file, N shots, R episodes | JSONL metrics record |
| `eval_reconstruction.py` | Reconstruction quality metrics | Checkpoint, test images | JSONL metrics record |
| `eval_peak_finding.py` | Peak finding with segmentation head | Checkpoint, PeakNet labels | JSONL metrics record |
| `eval_maxie_downstream.py` | Dispatcher: reads sweep config, calls appropriate eval script | sweep_configs.json, config_id | JSONL metrics record |
| `aggregate_results.py` | Collect all JSONL results into a summary table | Results directory | `eval_summary.csv`, plots |
| `statistical_analysis.py` | Run statistical tests on aggregated results | `eval_summary.csv` | `statistical_report.md` |

### 9.2 JSONL Output Format

All evaluation scripts write results in the JSONL format specified by
`monitoring-protocol-for-maxie.md` Section 6.1. Each evaluation run produces
one JSONL record:

```json
{
  "event": "downstream_eval",
  "timestamp_utc": "2026-03-19T14:30:00Z",
  "config_id": 42,
  "ssl_method": "mae",
  "model_size": "base",
  "checkpoint": "epoch_200",
  "checkpoint_path": "/path/to/checkpoint.pt",
  "task": "hit_miss",
  "protocol": "linear_probe",
  "metrics": {
    "accuracy": 0.953,
    "f1": 0.948,
    "auc_roc": 0.987,
    "precision": 0.961,
    "recall": 0.936
  },
  "hyperparameters": {
    "lr": 0.1,
    "epochs": 100,
    "batch_size": 256,
    "pool_method": "cls",
    "l2_normalize": true
  },
  "runtime_sec": 1847.3,
  "device": "AMD MI250X GCD",
  "num_train_samples": 5400,
  "num_test_samples": 1800
}
```

### 9.3 Dispatcher Script Structure

```python
"""eval_maxie_downstream.py -- Main dispatcher for evaluation runs."""
import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", required=True, help="Path to sweep_configs.json")
    parser.add_argument("--config-id", type=int, required=True, help="Config index")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    args = parser.parse_args()

    # Load config
    with open(args.sweep_config) as f:
        configs = json.load(f)
    config = configs[args.config_id]

    if config["category"] == "reserved_rerun":
        print(f"Config {args.config_id} is reserved. Skipping.")
        sys.exit(0)

    # Resolve checkpoint path from ssl_method + model_size + checkpoint
    checkpoint_path = resolve_checkpoint(config)

    # Step 1: Extract features (shared across protocols for same checkpoint)
    features_path = extract_features(checkpoint_path, config, args.output_dir)

    # Step 2: Run evaluation protocol
    if config["protocol"] == "linear_probe":
        results = run_linear_probe(features_path, config)
    elif config["protocol"] == "knn":
        results = run_knn(features_path, config)
    elif config["protocol"] == "few_shot_10":
        results = run_few_shot(features_path, config, n_shot=10)
    elif config["protocol"] == "recon_metrics":
        results = run_reconstruction_eval(checkpoint_path, config)
    else:
        raise ValueError(f"Unknown protocol: {config['protocol']}")

    # Step 3: Write JSONL result
    write_jsonl_result(results, config, args.output_dir)

if __name__ == "__main__":
    main()
```

### 9.4 Integration with Monitoring Protocol

The evaluation JSONL files are stored alongside the training JSONL metrics
under a unified directory structure:

```
$EVAL_OUTPUT_DIR/
  sweep_configs.json
  results/
    eval_000.jsonl
    eval_001.jsonl
    ...
    eval_399.jsonl
  features/
    mae_base_early_cls.pt
    mae_base_early_mean.pt
    ...
  logs/
    eval_000.log
    ...
  eval_summary.csv
  statistical_report.md
```

The `aggregate_results.py` script reads all `results/eval_*.jsonl` files and
produces `eval_summary.csv` with columns:

```
config_id, ssl_method, model_size, checkpoint, task, protocol, accuracy, f1,
auc_roc, precision, recall, recon_mse, recon_psnr_db, recon_ssim, runtime_sec
```

---

## 10. Key Files Reference

### Companion Documents

| Document | Path | Relevant sections |
|----------|------|-------------------|
| SSL candidates | `docs/agents/ssl-candidates-for-maxie.md` | Evaluation metrics (bottom), Tier-1 candidates |
| Training playbook | `docs/agents/training-playbook-for-maxie.md` | Phase 2 scaling study, Section 4.4 fine-tuning recipe |
| Monitoring protocol | `docs/agents/monitoring-protocol-for-maxie.md` | Section 4.2 linear probe, Section 6 JSONL format |
| Frontier strategy | `docs/agents/research-loop-frontier-strategy.md` | Section 4 Flux ensemble jobs |
| Data pipeline | `docs/agents/data-pipeline-for-maxie.md` | Dataset classes, Zarr store structure |

### Source Code

| Component | Path | Notes |
|-----------|------|-------|
| MAXIE adapted MAE | `$MAXIE_DIR/maxie/modeling/adapted_mae.py` | ViT-MAE encoder with 1-channel adaptation |
| MAXIE eval utility | `$MAXIE_DIR/maxie/utils/eval.py` | `estimate_loss()` -- loss-only evaluation, no downstream metrics |
| MAXIE training script | `$MAXIE_DIR/train/train.fsdp.py` | Reference for data loading, transforms, FSDP setup |
| PeakNet eval utility | `$PEAKNET_DIR/peaknet/utils/eval.py` | `estimate_loss()` with criterion support |
| PeakNet models | `$PEAKNET_DIR/peaknet/modeling/` | ConvNextV2-BiFPN, Hiera -- peak finding architectures |
| Exp-PeakNet configs | `$PROJ_PEAKNET_DIR/hydra_config/` | Experiment configurations for peak finding |
| Exp-PeakNet eval docs | `$PROJ_PEAKNET_DIR/docs/evaluation-optimization-80gpu.md` | Scaling eval iterations with world size |

### External References in Docs Collection

| Resource | Path | Use for |
|----------|------|---------|
| PSNR/SSIM implementations | `$DEEPLEARNING_DOC_DIR/computer-vision/mmagic/mmagic/evaluation/metrics/` | Reference PSNR and SSIM metric code |
| ViT MSN (low-shot SSL) | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/docs/source/en/model_doc/vit_msn.md` | Low-shot evaluation methodology |
| SSL evaluation tutorial | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/PyTorch-Deep-Learning-Minicourse/docs/en/week10/10-1.md` | Linear probe vs fine-tuning comparison |
| ClusterFit / PIRL | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/PyTorch-Deep-Learning-Minicourse/docs/en/week10/10-2.md` | Transfer learning evaluation patterns |
| k-NN for GANs | `$DEEPLEARNING_DOC_DIR/computer-vision/PyTorch-StudioGAN/` | k-NN analysis implementation reference |
| VQ library | `$DEEPLEARNING_DOC_DIR/lucidrain-repos/vector-quantize-pytorch/` | Codebook evaluation utilities |

### Model Dimensions Quick Reference

| Model | embed_dim | Layers | Heads | Params | HuggingFace ID |
|-------|-----------|--------|-------|--------|----------------|
| ViT-MAE-Base | 768 | 12 | 12 | 86M | `facebook/vit-mae-base` |
| ViT-MAE-Large | 1024 | 24 | 16 | 304M | `facebook/vit-mae-large` |
| ViT-MAE-Huge | 1280 | 32 | 16 | 632M | `facebook/vit-mae-huge` |
