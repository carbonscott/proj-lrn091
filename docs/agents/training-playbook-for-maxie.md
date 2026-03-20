# Training Playbook for MAXIE on Frontier

Date: 2026-03-19
Context: ALCC project Group 1 — training recipe, optimizer, LR schedule, loss design,
batch size scaling, and failure mode diagnostics for MAXIE pre-training campaigns on
Frontier (up to 512 nodes / 4096 GCDs).

Companion to: `ssl-candidates-for-maxie.md` (what to train), `research-loop-frontier-strategy.md`
(where/how to run), `research-loop-brainstorm.md` (how to explore the design space).

---

## 1. Current MAXIE Training Configuration

Source: `$MAXIE_DIR/train/train.fsdp.py`, `$MAXIE_DIR/train/hydra_config/train_config/base.yaml`,
`$MAXIE_DIR/maxie/lr_scheduler.py`

| Parameter | Current Default | Notes |
|-----------|----------------|-------|
| Optimizer | AdamW | `torch.optim.AdamW` |
| Learning rate | 1.5e-4 | Base LR |
| Betas | (0.9, 0.95) | Standard MAE recipe |
| Weight decay | 0.05 | Decoupled (AdamW-style) |
| Gradient clipping | 1.0 | Max norm (MAE paper uses none — see Section 2.4) |
| LR schedule | Cosine with linear warmup | Custom `CosineLRScheduler` |
| Warmup iterations | 5 | Very short — likely needs tuning |
| Total iterations | 1,000,000 | |
| Min LR | 1e-7 | |
| Batch size | 1 | Per-device; effective depends on world size |
| Gradient accumulation | 2 | |
| Max epochs | 5 | |
| Precision | float16 | AMP with `ShardedGradScaler` |
| Distributed strategy | FSDP (PyTorch) | Wraps `ViTMAELayer`; supports zero2/zero3 |
| Activation checkpointing | Yes | On `ViTMAELayer` modules |
| Loss | MSE (pixel reconstruction) | Via `ViTMAEForPreTraining` |
| norm_pix_loss | false | Should be true (see Section 3) |
| Mask ratio | 0.75 | Standard MAE |
| Model | ViT-MAE-Base (768d, 12L, 12H) | From `facebook/vit-mae-base` |
| Input channels | 1 (adapted from 3) | Averaged RGB weights |
| Image size | 224x224 | |
| Patch size | 16x16 | |

---

## 2. Optimizer Selection

### 2.1 AdamW (current, recommended as default)

AdamW is the standard choice for ViT/MAE pre-training across all major implementations
(original MAE paper, Scenic, HuggingFace, ViTDet).

**Implementation subtlety:** The original MAE paper and Scenic config actually use
**Adam** (not AdamW) with *explicit/decoupled* weight decay applied separately. The
Scenic config shows `optimizer='adam'` with `weight_decay=0` and
`explicit_weight_decay=0.05`. This is functionally equivalent to AdamW — PyTorch's
`AdamW` implements the same decoupled weight decay. MAXIE using `torch.optim.AdamW`
with `weight_decay=0.05` achieves the same effect.

**Canonical MAE hyperparameters** (from He et al. 2021 and Scenic config):

| Parameter | Value | Source |
|-----------|-------|--------|
| beta1 | 0.9 | MAE paper |
| beta2 | 0.95 | MAE paper (not the usual 0.999) |
| weight_decay | 0.05 | MAE paper (decoupled; exclude bias and norm params) |
| eps | 1e-8 | Default |

**Why beta2=0.95 instead of 0.999:** The MAE paper uses a lower beta2, which gives
faster adaptation to changing gradient magnitudes. This is important during the early
phase of self-supervised pre-training where the loss landscape changes rapidly as the
model transitions from random predictions to meaningful reconstructions. Note that
fine-tuning switches back to beta2=0.999 (see Section 4.4).

**Weight decay exclusions:** Bias parameters and layer norm parameters should be
excluded from weight decay. This is standard practice across ViT recipes (confirmed in
torchvision Swin config: `norm-weight-decay=0.0, bias-weight-decay=0.0`).

**Reference supervised ViT recipes** (from torchvision):

| Model | Optimizer | LR | Weight Decay | Grad Clip | Epochs |
|-------|-----------|-----|-------------|-----------|--------|
| ViT-B/16 | AdamW | 0.003 | 0.3 | 1.0 | 300 |
| ViT-B/32 | AdamW | 0.003 | 0.3 | 1.0 | 300 |
| ViT-L/16 | AdamW | 0.5 | 0.00002 | 1.0 | 600 |
| Swin-T | AdamW | 0.001 | 0.05 | 5.0 | 300 |

Note the dramatically different weight decay between supervised (0.3) and self-supervised
(0.05) ViT training. Self-supervised methods need less regularization because the pretext
task itself acts as a regularizer.

**MAXIE status:** Already using correct values. No change needed.

### 2.2 LAMB (for large-batch scaling beyond 4096)

LAMB (Layer-wise Adaptive Moments) was designed specifically for large-batch training
("Large Batch Optimization for Deep Learning: Training BERT in 76 minutes").

**When to consider for MAXIE:**
- If scaling to effective batch sizes > 4096 (e.g., 512 nodes x 8 GCDs x batch_per_gpu)
- LAMB applies per-layer trust ratios that stabilize training at large batch sizes
  where AdamW may diverge

**Available implementations:**
- `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/pytorch-image-models/timm/optim/lamb.py` —
  pure PyTorch, supports XLA/TPU
- NVIDIA APEX `FusedLamb` — GPU-optimized, faster but requires APEX
- `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/apex/apex/optimizers/fused_lamb.py`

**LAMB hyperparameters (starting point):**
| Parameter | Value |
|-----------|-------|
| lr | Scale with batch size (see Section 4) |
| betas | (0.9, 0.999) |
| weight_decay | 0.01 |
| eps | 1e-6 |

**Recommendation:** Start with AdamW. Only switch to LAMB if you observe training
instability at the target batch size (4096+ effective). LAMB adds complexity (per-layer
trust ratio computation) and the ViT/MAE literature predominantly uses AdamW successfully
at batch size 4096.

### 2.3 Gradient Clipping

**MAE paper uses NO gradient clipping** during pre-training. The Scenic MAE config
explicitly sets `max_grad_norm = None`. MAXIE currently uses `grad_clip=1.0`.

This is worth investigating:

| Setting | Used by | Notes |
|---------|---------|-------|
| No clipping | MAE paper, Scenic MAE pre-training | Canonical recipe |
| max_norm=1.0 | MAXIE, torchvision ViT supervised, AV-MAE fine-tuning | Conservative safety net |
| max_norm=5.0 | Swin Transformer | Looser clipping |

**Recommendation:** For the baseline fix phase, keep `grad_clip=1.0` as a safety net.
Once training is stable, experiment with removing it to match the canonical recipe.
If gradient norms are consistently < 1.0, the clipping is doing nothing and can be
removed. If they spike, the clipping is saving you.

**AV-MAE fine-tuning detail:** Uses `grad_clip_after_pmean=True`, meaning gradients
are clipped *after* averaging across devices. This is the correct order for distributed
training — clip the synchronized gradient, not the per-device gradient.

### 2.4 Other optimizers to be aware of

| Optimizer | When to consider | Notes |
|-----------|-----------------|-------|
| **8-bit AdamW** (bitsandbytes) | Memory-constrained | Quantizes optimizer states, ~2x memory savings. Less tested on AMD/ROCm. |
| **Lion** | Faster convergence claimed | Sign-based momentum optimizer. Simpler than Adam. Emerging results, not yet standard for MAE. |
| **Distributed FSDP optimizer** | Already available | MAXIE already supports FSDP's distributed optimizer via Hydra config. |

---

## 3. Loss Function Design

### 3.1 MSE Pixel Reconstruction (current)

The standard MAE loss: MSE between predicted and original pixel values, computed only
on masked patches.

**Critical setting: `norm_pix_loss=True`**

The MAE paper (Table 3) shows that normalizing the target pixel values *per patch*
improves representation quality on downstream tasks. Specifically:

```
target_patch = (patch - patch.mean()) / (patch.std() + eps)
```

This makes the loss focus on *relative structure within each patch* rather than absolute
intensity — crucial for diffraction data where absolute intensity varies enormously
across the detector face.

**MAXIE currently has `norm_pix_loss=false`.** Switching to `true` is likely the single
highest-value change for representation quality.

Source: `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/src/transformers/models/vit_mae/configuration_vit_mae.py` —
"Using normalized pixels improved representation quality in the experiments of the authors."

HuggingFace MAE example also defaults to `--norm_pix_loss` enabled.

### 3.2 Contrast-Aware Reconstruction Loss (proposed)

Standard MSE treats all pixels equally, but in diffraction images:
- Bragg peaks: <5% of pixels, carry >90% of the information
- Background: >95% of pixels, mostly detector noise

A contrast-aware loss would weight Bragg peak regions higher. Design axes from
`research-loop-brainstorm.md` Candidate 2:

| Variant | Formula | Tradeoff |
|---------|---------|----------|
| **Local contrast weighting** | `w_ij = (local_max - local_min) / local_mean` within NxN window | Emphasizes high-contrast regions (peaks) |
| **Local SNR weighting** | `w_ij = local_mean / local_std` | Emphasizes high-signal regions |
| **Threshold-based** | `w_ij = alpha if pixel > threshold else 1` | Simple, requires threshold selection |
| **Hybrid** | `loss = (1-lambda) * MSE_global + lambda * MSE_peak_weighted` | Balances global and peak fidelity |

**Implementation approach:** Replace the loss computation in `ViTMAEForPreTraining`
or wrap it:

```python
def contrast_aware_loss(pred, target, mask, window_size=9, alpha=5.0):
    # Standard MSE on masked patches
    mse = (pred - target) ** 2

    # Compute local contrast weights from target
    # (unfold target into windows, compute per-window contrast)
    weights = compute_local_contrast(target, window_size)

    # Weight the MSE
    weighted_mse = mse * (1.0 + alpha * weights)
    return weighted_mse[mask].mean()
```

**Window sizes to explore:** 5x5, 9x9, 15x15, 21x21 (patch-relative)

### 3.3 Loss functions for other SSL candidates

| SSL Method | Loss | Key difference from MAE |
|------------|------|------------------------|
| **VQ-VAE** | Reconstruction MSE + codebook loss + commitment loss (beta * ‖sg[z_e] - e‖²) | Three-term loss; beta controls encoder-codebook coupling |
| **I-JEPA** | MSE in representation space (predicted vs. EMA teacher output) | Not pixel-level — loss magnitude is harder to interpret |
| **BEiT** | Cross-entropy over codebook token vocabulary | Classification loss, not regression |
| **DINOv2** | Cross-entropy between student/teacher distributions + centering | Self-distillation loss |
| **Contrastive** | InfoNCE: -log(exp(sim(z_i,z_j)/τ) / Σ exp(sim(z_i,z_k)/τ)) | Temperature τ is critical |

---

## 4. Learning Rate Schedule

### 4.1 Linear Scaling Rule

The MAE paper and Scenic implementation both use the **linear scaling rule** from
Goyal et al. ("Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"):

```
effective_lr = base_lr * effective_batch_size / 256
```

Where `base_lr = 1.5e-4` (the canonical MAE value).

| Effective batch size | LR | Scenario |
|---------------------|-----|----------|
| 256 | 1.5e-4 | Single-GPU reference |
| 1024 | 6.0e-4 | 8 nodes x 8 GCDs x batch_per_gpu=16 |
| 4096 | 2.4e-3 | 512 nodes x 8 GCDs x batch_per_gpu=1 |
| 8192 | 4.8e-3 | With gradient accumulation = 2 |

**MAXIE note:** The current config uses `lr=1.5e-4` without scaling by batch size. At
scale (4096 GCDs), the effective batch size is much larger, so the LR should be scaled
accordingly.

**Square root scaling alternative:** For very large batches (>8192), some practitioners
use `lr = base_lr * sqrt(batch_size / 256)` which is more conservative. If linear
scaling causes divergence, try sqrt scaling.

### 4.2 Warmup

**Canonical MAE:** 40 epochs of linear warmup (from the Scenic config).

**Current MAXIE:** 5 iterations of warmup — this is almost certainly too short.

**Recommended warmup:**
- For scaling law studies (small runs): 5-10% of total training steps
- For full pre-training campaigns: 40 epochs (matching the original MAE paper)
- At large batch sizes: longer warmup stabilizes training. Rule of thumb: scale warmup
  linearly with batch size.

**Warmup formula (linear):**

```
if step < warmup_steps:
    lr = base_lr * step / warmup_steps
```

### 4.3 Cosine Decay

After warmup, decay to a minimum LR following a cosine schedule:

```
lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * (step - warmup_steps) / (total_steps - warmup_steps)))
```

**Canonical values:**
| Parameter | MAE paper | Scenic config | MAXIE current |
|-----------|-----------|---------------|---------------|
| min_lr | 0 | 0 | 1e-7 |
| Warmup epochs | 40 | 40 | ~0 (5 iters) |
| Total epochs | 800 | 800 | 5 |
| Schedule shape | Cosine | Cosine | Cosine |

**Recommendation for MAXIE scaling studies:** For short runs (few epochs), use a
proportional warmup (5-10% of total steps) and cosine decay to 1e-6. For full
pre-training, match the MAE paper's 40-epoch warmup with cosine decay to 0.

### 4.4 Layer-wise LR Decay (for fine-tuning)

When fine-tuning the pre-trained encoder on downstream tasks, apply layer-wise LR decay:
earlier layers get smaller LR, later layers get larger LR.

**Formula:** Each layer `i` gets `lr_scale = decay_rate^(num_layers - i)`.

**Decay rates scale with model depth** (gentler decay for deeper models):

| Model | Layer decay rate | Source |
|-------|-----------------|--------|
| ViT-Base (12L) | 0.75 | Scenic MAE fine-tune config |
| ViT-Base (12L) | 0.7 | ViTDet (detectron2) |
| ViT-Huge (32L) | 0.9 | ViTDet (detectron2) |

**Fine-tuning recipe** (from Scenic MAE fine-tune config):

| Parameter | Pre-training | Fine-tuning |
|-----------|-------------|-------------|
| Optimizer | Adam(W) | AdamW |
| Betas | (0.9, 0.95) | (0.9, 0.999) |
| Base LR | 1.5e-4 | 4e-3 (no batch scaling) |
| Weight decay | 0.05 | 0.05 |
| Warmup | 40 epochs | 5 epochs |
| Total epochs | 800 | 50 |
| End LR | 0 | 1e-6 |
| Layer-wise decay | N/A | 0.75 |
| Grad clip | None | 1.0 |

**Stochastic depth (drop path)** is used only during fine-tuning, scaled by model size:

| Model | Drop path rate |
|-------|---------------|
| ViT-Base | 0.1 |
| ViT-Large | 0.2 |
| ViT-Huge | 0.5 |

**Additional fine-tuning details:**
- Positional embeddings excluded from weight decay (`pos_embed` wd=0.0 in ViTDet)
- Layer norm and bias excluded from weight decay

This is not needed during pre-training, but worth building into the recipe for
downstream evaluation.

---

## 5. Batch Size Scaling Strategy

### 5.1 Effective batch size on Frontier

| Nodes | GCDs | Batch/GCD | Grad accum | Effective batch |
|-------|------|-----------|------------|----------------|
| 2 | 16 | 16 | 1 | 256 |
| 8 | 64 | 16 | 1 | 1,024 |
| 32 | 256 | 16 | 1 | 4,096 |
| 512 | 4,096 | 1 | 1 | 4,096 |
| 512 | 4,096 | 1 | 2 | 8,192 |
| 512 | 4,096 | 4 | 1 | 16,384 |

**Memory constraint:** Each MI250X GCD has 64 GB HBM2e. With ViT-MAE-Base (768d, 12L),
float16, and activation checkpointing, batch_per_gpu of 16-32 should be feasible for
224x224 images. Larger ViT variants (Large, Huge) will reduce this.

### 5.2 Scaling approach

1. **Start small:** Validate the training recipe on 2-8 nodes with batch_per_gpu=16
2. **Scale batch size with LR:** Use `lr = 1.5e-4 * batch_size / 256`
3. **Monitor training loss curves:** If loss diverges at larger batch sizes:
   - Increase warmup length
   - Switch to sqrt LR scaling
   - Try LAMB optimizer
4. **Gradient accumulation:** Use when batch_per_gpu is memory-limited. MAXIE already
   supports `grad_accum_steps` with `model.no_sync()` for efficient FSDP accumulation.

### 5.3 Large-batch training stability checklist

- [ ] Warmup length proportional to batch size
- [ ] LR scaled from base_lr via linear or sqrt rule
- [ ] Gradient clipping enabled (max_norm=1.0)
- [ ] Loss scaled by `1 / grad_accum_steps` before backward (MAXIE already does this)
- [ ] GradScaler enabled for float16 (MAXIE already does this)
- [ ] Monitor gradient norm per step — spikes indicate instability
- [ ] Monitor loss for NaN/Inf — MAXIE tracks NaN losses

---

## 6. Mixed Precision

### 6.1 float16 vs bfloat16 on Frontier

| Precision | Pros | Cons | Frontier support |
|-----------|------|------|-----------------|
| **float16** | 2x throughput, well-tested | Requires GradScaler, overflow risk | MI250X supports natively |
| **bfloat16** | No GradScaler needed, larger dynamic range | Slightly less precision in mantissa | MI250X supports natively |
| **float32** | Most stable | 2x slower, 2x memory | Always supported |

**MAXIE currently uses float16** with `ShardedGradScaler`. This is fine.

**Consider bfloat16 for Frontier:** AMD MI250X supports bf16 natively. bfloat16
eliminates the need for GradScaler (no overflow risk due to larger exponent range),
simplifying the training loop and potentially improving stability at scale. The
PyTorch Lightning docs note bf16 "maintains more of the dynamic range that FP32 offers."

**To switch:** Change `dist.dtype` in Hydra config from `float16` to `bfloat16` and
disable GradScaler.

### 6.2 TF32

MAXIE already enables TF32 on Ampere+ GPUs. On AMD MI250X (ROCm), the equivalent is
the default float32 behavior which uses reduced-precision tensor cores. No action needed.

---

## 7. Training Duration and Checkpointing

### 7.1 How long to train

**MAE paper recipe:** 800 epochs on ImageNet-1K (1.28M images, batch 4096).

**MAXIE's dataset is smaller** (X-ray diffraction images — likely 10K-100K range). With
fewer images, each epoch is shorter, but you may need more epochs to see the same number
of gradient updates.

**Recommendation: think in terms of total gradient steps, not epochs.**

| Dataset size | Batch size | Steps/epoch | Target steps | Equivalent epochs |
|-------------|-----------|-------------|-------------|------------------|
| 10K | 256 | 39 | 250K | ~6,400 |
| 50K | 1,024 | 49 | 250K | ~5,100 |
| 100K | 4,096 | 24 | 250K | ~10,400 |

For scaling law studies (short runs), 50K-100K steps should reveal trends. For full
pre-training, 200K-500K steps depending on dataset size.

### 7.2 Checkpoint frequency

- **During scaling studies:** Every 1K-5K steps (frequent, for fine-grained analysis)
- **During full pre-training:** Every epoch or every 10K steps
- **Always checkpoint at end of training** (MAXIE does this)
- **Keep best + last + periodic** — use `save_total_limit` to avoid filling Lustre

### 7.3 Checkpoint diagnostics (per checkpoint)

For the monitoring protocol, log these at every checkpoint:

| Metric | What it tells you | Action threshold |
|--------|------------------|-----------------|
| Training loss | Overall learning progress | Plateau for >10K steps = LR too low or capacity saturated |
| Validation loss | Generalization | Val loss diverging from train loss = overfitting |
| Gradient norm (mean) | Training stability | Spikes > 10x mean = instability |
| Gradient norm (max) | Explosion risk | > 100 = likely divergence soon |
| Learning rate | Schedule correctness | Should match expected cosine curve |
| GPU memory utilization | Resource efficiency | < 60% = can increase batch size |
| Throughput (samples/sec) | Scaling efficiency | Should scale ~linearly with nodes |

**For VQ-VAE specifically** (from `research-loop-brainstorm.md`):
| Metric | What it tells you | Action threshold |
|--------|------------------|-----------------|
| Codebook utilization | Fraction of codes used | < 50% = too many dead codes |
| Codebook perplexity | Effective vocabulary size | Should be close to codebook_size |
| Dead code fraction | Codes never selected | > 20% after 10K steps = reset strategy needed |
| Reconstruction loss | Encoding quality | Plateau = codebook capacity saturated |

---

## 8. Failure Modes and Mitigations

### 8.1 Loss divergence / NaN

**Symptoms:** Loss suddenly spikes to large values or becomes NaN.

**Common causes at scale:**
1. LR too high for batch size → reduce LR or use sqrt scaling
2. Warmup too short → increase warmup to 5-10% of total steps
3. float16 overflow → switch to bfloat16 or reduce LR
4. Gradient explosion → reduce grad_clip threshold (try 0.5 or 0.3)
5. Bad data batch (corrupted image, extreme values) → add input validation

**MAXIE already tracks NaN losses** in the training loop. Consider adding early stopping
if NaN count exceeds a threshold.

### 8.2 Loss plateau (no improvement)

**Symptoms:** Loss stops decreasing well before expected convergence.

**Common causes:**
1. LR too low (not scaled for batch size) → apply linear scaling rule
2. Model capacity too small for data → scale to ViT-Large/Huge
3. Mask ratio wrong for data → try 0.5 or 0.9 (diffraction may need different ratio)
4. `norm_pix_loss=false` → switch to true (let loss focus on structure, not absolute intensity)
5. Data too easy / too homogeneous → check data diversity

### 8.3 Codebook collapse (VQ-VAE specific)

**Symptoms:** Only a few codebook entries are used, most have zero utilization.

**Causes and mitigations:**
| Cause | Mitigation |
|-------|-----------|
| beta (commitment loss) too high | Reduce beta; try 0.1 → 0.25 |
| Codebook too large for data | Start smaller (256-512), scale up |
| No EMA update | Enable exponential moving average codebook update |
| No dead code reset | Enable threshold-based reset (reinit from data) |
| Encoder too strong | Add bottleneck before quantization |

### 8.4 Slow convergence at scale (distributed)

**Symptoms:** Wall-clock time per step doesn't decrease linearly with more nodes.

**Frontier-specific causes:**
1. RCCL AllReduce bottleneck → ensure `aws-ofi-rccl` plugin is built and loaded
2. Python env loading from Lustre → use sbcast to NVMe (see `research-loop-frontier-strategy.md`)
3. Data loading bottleneck → ensure data is on Lustre (Orion), not NFS; use adequate num_workers
4. NCCL/RCCL misconfiguration → check env vars in Section 2.3-2.4 of strategy doc
5. torchrun instead of srun → use srun with `--gpu-bind=closest` (see strategy doc Section 2.2)

### 8.5 Overfitting on small datasets

**Symptoms:** Train loss keeps decreasing, val loss increases.

**Mitigations for self-supervised pre-training:**
- Increase mask ratio (more masking = harder task = more regularization)
- Increase weight decay (0.05 → 0.1)
- Add dropout (MAE default is 0, can add 0.1)
- Data augmentation: random horizontal flip, random rotation (if symmetry allows)
- For diffraction: random intensity scaling, random detector region masking

---

## 9. Data Augmentation and Masking

### 9.1 MAE pre-training augmentation (minimal)

The MAE paper uses very minimal augmentation during pre-training — the masking itself
is the primary source of training signal.

**Pre-training augmentations (canonical MAE):**
- `RandomResizedCrop(224)` with scale=(0.2, 1.0), bicubic interpolation
- `RandomHorizontalFlip`
- ImageNet normalization

**Not used during pre-training:** RandAugment, color jitter, mixup, cutmix, random
erasing, label smoothing. These are fine-tuning only.

**For MAXIE / diffraction data:**
- `RandomResizedCrop` needs care — diffraction patterns have a beam center, and
  aggressive cropping may remove it. Consider constraining crop scale to (0.5, 1.0)
  or using center-aware cropping.
- `RandomHorizontalFlip` is likely safe if detector geometry is symmetric.
- `RandomRotation` may be valid depending on crystal symmetry — domain-dependent.
- Color jitter: meaningless for single-channel data.
- Intensity scaling: potentially valid (exposure time invariance).
- No normalization to ImageNet stats — compute mean/std from your diffraction dataset.

### 9.2 Fine-tuning augmentation (aggressive)

Fine-tuning adds substantially more augmentation:

| Augmentation | Setting |
|-------------|---------|
| RandAugment | n=2, m=15 |
| Random erasing | prob=0.25 |
| Mixup | alpha=0.8 |
| CutMix | alpha=1.0, switch_prob=0.5 |
| Label smoothing | 0.1 |
| Stochastic depth | 0.1-0.5 (see Section 4.4) |

**For diffraction fine-tuning:** Mixup and CutMix may not make physical sense for
diffraction patterns. Consider domain-specific alternatives or skip these.

### 9.3 Mask ratio

| Method | Mask ratio | Why |
|--------|-----------|-----|
| MAE (images) | 75% | High ratio forces semantic understanding |
| VideoMAE (video) | 90-95% | Higher due to temporal redundancy between frames |
| SimMIM | 32-64% | Lower masking, larger mask patches |

**For diffraction data:** The optimal mask ratio may differ from natural images. Bragg
peaks are spatially sparse — a 75% random mask has a high probability of masking most
peaks, which could make the task too hard or force the model to learn peak-free
background reconstruction. Consider:
- **Lower ratio (50-60%):** If peaks need to be visible for reconstruction
- **Higher ratio (85-90%):** If the data has high spatial redundancy (smooth background)
- **Structured masking:** Block masking (like I-JEPA) rather than random patch masking,
  to ensure some complete peaks remain visible

This is a high-priority axis for the scaling law study (Phase 2).

---

## 10. Interaction Table: SSL Method x Training Recipe

How the training recipe changes depending on which SSL candidate is used:

| Recipe component | MAE (current) | VQ-VAE | I-JEPA |
|-----------------|---------------|--------|--------|
| **Optimizer** | AdamW (beta2=0.95) | AdamW (beta2=0.99) | AdamW (beta2=0.95) |
| **Base LR** | 1.5e-4 | 1e-4 to 4e-4 | 1.5e-4 |
| **LR scaling** | Linear with batch | Linear with batch | Linear with batch |
| **Weight decay** | 0.05 | 0.0 on codebook, 0.05 on rest | 0.05 |
| **Warmup** | 40 epochs | 10-20 epochs | 40 epochs |
| **Total epochs** | 800 | 300-500 (faster convergence) | 300-600 |
| **Loss** | MSE (norm_pix_loss=true) | MSE + codebook + commitment | MSE in repr space |
| **Extra hyperparam** | mask_ratio=0.75 | beta, codebook_size, EMA decay | EMA teacher decay, mask block size |
| **Failure mode** | Blurry reconstruction | Codebook collapse | Representation collapse |
| **Key diagnostic** | Reconstruction visual quality | Codebook utilization histogram | Loss in repr space + linear probe |

---

## 11. Recommended Experiments: Priority Order

### Phase 1: Fix the baseline (2-8 nodes, 1-2 hours each)

These are changes to the current MAE setup that should improve results immediately:

| Experiment | Change | Expected impact | Effort |
|-----------|--------|----------------|--------|
| 1a | Enable `norm_pix_loss=true` | Better representation quality | Config change only |
| 1b | Increase warmup to 5-10% of total steps | More stable early training | Config change only |
| 1c | Scale LR with batch size: `lr = 1.5e-4 * batch/256` | Correct LR for distributed training | Config change only |
| 1d | Try bfloat16 instead of float16 | Simpler numerics, no GradScaler needed | Config + minor code change |

### Phase 2: Scaling law study (8-32 nodes)

Systematic sweep to understand model size vs. data size vs. compute tradeoffs:

| Axis | Values to sweep |
|------|----------------|
| Model size | ViT-Base (86M), ViT-Large (304M), ViT-Huge (632M) |
| Dataset size | 10%, 25%, 50%, 100% of available data |
| Training steps | 50K, 100K, 200K |
| Mask ratio | 0.5, 0.75, 0.9 |

Use Hydra multirun + Flux for parallel execution.

### Phase 3: SSL method comparison (8 nodes, Flux ensemble)

Run Tier 1 candidates (MAE, VQ-VAE, I-JEPA) head-to-head on the same data:

| Metric | How to compare |
|--------|---------------|
| Reconstruction quality | MSE, PSNR, SSIM on held-out data (MAE, VQ-VAE only) |
| Bragg peak preservation | Peak SNR in reconstructed vs. original |
| Linear probe | Freeze encoder, train linear classifier |
| k-NN retrieval | Embed images, check nearest-neighbor quality |

### Phase 4: Full pre-training campaign (512 nodes)

Apply the winning SSL method + optimized recipe from Phases 1-3 at full scale.

---

## 12. Reference: Canonical MAE Training Recipe

From the original MAE paper (He et al. 2021) and Scenic implementation, for
reproducibility reference:

```yaml
# Model
model: ViT-Large/16
image_size: 224
patch_size: 16
mask_ratio: 0.75
norm_pix_loss: true  # normalize target per patch (crucial)
dropout: 0.0         # no dropout during pre-training
attention_dropout: 0.0
positional_embedding: sinusoidal_2d  # fixed, not learned

# Decoder (same for all encoder sizes)
decoder_hidden_size: 512
decoder_num_layers: 8
decoder_num_heads: 16
decoder_mlp_dim: 2048

# Optimizer
optimizer: Adam  # with explicit decoupled weight decay (= AdamW)
base_lr: 1.5e-4  # scaled by batch_size/256
betas: [0.9, 0.95]  # note: beta2=0.95 for pre-training, 0.999 for fine-tuning
weight_decay: 0.05   # decoupled; exclude bias and layernorm params
grad_clip: null      # MAE paper doesn't clip during pre-training

# Schedule
total_epochs: 800
warmup_epochs: 40   # linear warmup
lr_schedule: cosine
min_lr: 0

# Data
batch_size: 4096  # global
augmentation: RandomResizedCrop(224, scale=(0.2, 1.0)) + HorizontalFlip
# NOT used during pre-training: RandAugment, color jitter, mixup, cutmix

# Precision
dtype: float32  # original paper; float16/bf16 at scale
```

---

## 13. Model Architecture Reference

**MAE decoder is the same size regardless of encoder size.** This is a key design
choice — the decoder is lightweight and discarded after pre-training:

| Encoder | Encoder dims | Decoder dims |
|---------|-------------|-------------|
| ViT-Base | hidden=768, 12L, 12H, mlp=3072 | hidden=512, 8L, 16H, mlp=2048 |
| ViT-Large | hidden=1024, 24L, 16H, mlp=4096 | hidden=512, 8L, 16H, mlp=2048 |
| ViT-Huge | hidden=1280, 32L, 16H, mlp=5120 | hidden=512, 8L, 16H, mlp=2048 |

**Other architecture details (MAE defaults):**
- Activation: GELU
- Dropout: 0.0 (no dropout during pre-training)
- Attention dropout: 0.0
- QKV bias: enabled
- Positional embedding: sinusoidal 2D (fixed, not learned)
- Initializer range: 0.02

**HuggingFace `run_mae.py` LR note:** The HuggingFace training script uses
`base_learning_rate=1e-3` (not 1.5e-4). This is because their linear scaling formula
is `lr = base_learning_rate * total_batch_size / 256`, and their default per-device
batch size results in a smaller total batch — the formula compensates. The effective
LR at batch_size=4096 works out the same either way. When using MAXIE's own LR
scheduler, use `base_lr=1.5e-4` and scale by `batch_size/256`.

---

## 14. Key Files Reference

| File | What it contains |
|------|-----------------|
| `$MAXIE_DIR/train/train.fsdp.py` | Main training loop, FSDP setup, optimizer, gradient handling |
| `$MAXIE_DIR/maxie/lr_scheduler.py` | Custom `CosineLRScheduler` |
| `$MAXIE_DIR/maxie/modeling/adapted_mae.py` | MAE model adapted for 1-channel input |
| `$MAXIE_DIR/train/hydra_config/train_config/base.yaml` | Default training hyperparameters |
| `$DEEPLEARNING_DOC_DIR/computer-vision/scenic/scenic/projects/av_mae/configs/imagenet/pretrain.py` | Canonical MAE pre-training recipe (Scenic/JAX) |
| `$DEEPLEARNING_DOC_DIR/computer-vision/scenic/scenic/projects/av_mae/configs/imagenet/finetune.py` | Canonical MAE fine-tuning recipe (Scenic/JAX) |
| `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/examples/pytorch/image-pretraining/README.md` | HuggingFace MAE training example |
| `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/examples/pytorch/image-pretraining/run_mae.py` | HuggingFace MAE training script (LR scaling implementation) |
| `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/vision/references/classification/README.md` | Torchvision supervised ViT/Swin training recipes |
| `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py` | ViTDet config (layer-wise LR decay, AMP, fine-tuning recipe) |
| `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/src/transformers/models/vit_mae/configuration_vit_mae.py` | norm_pix_loss documentation |
| `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/pytorch-image-models/timm/optim/lamb.py` | LAMB optimizer (pure PyTorch) |
| `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/apex/apex/optimizers/fused_lamb.py` | NVIDIA FusedLamb (GPU-optimized) |
| `$DEEPLEARNING_DOC_DIR/lucidrain-repos/vector-quantize-pytorch/` | VQ library for VQ-VAE candidate |
| `docs/agents/ssl-candidates-for-maxie.md` | SSL approach comparison |
| `docs/agents/research-loop-frontier-strategy.md` | Frontier execution strategy |
| `docs/agents/research-loop-brainstorm.md` | Research loop design for exploration |
