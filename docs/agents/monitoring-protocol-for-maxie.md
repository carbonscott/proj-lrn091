# Monitoring and Diagnostics Protocol for MAXIE Pre-Training

Date: 2026-03-19
Context: Operational protocol for monitoring ViT/MAE self-supervised pre-training runs
in the MAXIE project (X-ray diffraction foundation model on Frontier). Covers per-step
metrics, per-epoch aggregates, per-checkpoint diagnostics, SSL-method-specific signals,
structured logging format, decision trees, cross-run comparison, and integration with
the automated research loop.

Companion documents:
- `training-playbook-for-maxie.md` -- training recipe, optimizer, failure modes
- `research-loop-brainstorm.md` -- research loop design and diagnostic signals
- `research-loop-frontier-strategy.md` -- Frontier resource strategy
- `ssl-candidates-for-maxie.md` -- SSL method comparison

---

## 1. Current MAXIE Logging

Source: `$MAXIE_DIR/train/train.fsdp.py`, `$MAXIE_DIR/maxie/utils/monitor.py`,
`$MAXIE_DIR/train/hydra_config/train_config/base.yaml`

### 1.1 What is logged today

MAXIE uses Python's `logging` module (initialized via `maxie.utils.misc.init_logger`)
with file and console handlers. All log output goes to a timestamped file under the
configured `logging.directory` (default: `experiments/logs`). Logging happens only
from rank 0 for most events.

**Per-iteration (after each parameter update), rank 0 logs:**

| Field | Format | Source line |
|-------|--------|-------------|
| `rank` | int | Always 0 |
| `logevent` | `"LOSS:TRAIN"` | Fixed string |
| `iteration` | int | Global iteration counter |
| `segment` | `"start-end"` | Dataset segment range |
| `learning_rate` | comma-separated floats | From `scheduler.get_lr()` |
| `grad_norm` | float (6 decimals) | From `clip_grad_norm_` return value |
| `mean_train_loss` | float (6 decimals) | All-reduced average across ranks |
| `tokens_per_sec` | scientific notation | `total_num_tokens / t_delta` |
| `mfu` | float (3 decimals) | Model FLOPS utilization vs peak |
| `grad_nosync_counter` | int | Grad accumulation step counter |

Format: pipe-delimited key=value pairs in a single log line:
```
rank=0 | logevent=LOSS:TRAIN | iteration=42 | segment=0-4 | learning_rate=1.5e-04 | grad_norm=0.123456 | mean_train_loss=0.456789 | tokens_per_sec=1.2e+05 | mfu=0.012 | grad_nosync_counter=2
```

**Per-iteration dynamics monitoring (when `monitors_dynamics=true`):**

| Log event | Fields | What it captures |
|-----------|--------|-----------------|
| `DYNAMICS:ACT` | `name`, `preact.mean`, `preact.std`, `act.mean`, `act.std` | Pre/post activation statistics per monitored module |
| `DYNAMICS:PARAMS` | `part` (encoder/decoder), `name`, `update` | log10(update_std / param_std) per layer |

The `ActivationMonitor` class hooks into forward passes of modules matching
`ACT2CLS[model.config.hidden_act]` (typically GELU activations). The
`monitor_param_update_metrics` function computes the percent parameter update metric
(Karpathy's "update ratio") for encoder and decoder layers separately.

**Per-checkpoint (at `chkpt_saving_iterations` intervals):**

| Log event | What is logged |
|-----------|---------------|
| `LOSS:EVAL` (train split) | Mean training loss over `max_eval_iter` batches |
| `LOSS:EVAL` (val split) | Mean validation loss over `max_eval_iter` batches |
| Checkpoint save | Path to saved checkpoint |
| NaN detection | Error log if NaN encountered in eval losses |

**At initialization (rank 0):**

| Log event | Content |
|-----------|---------|
| `INIT` | Per-module weight mean and std after initialization |
| Config dump | Full YAML config |
| Parameter count | Total and sharded parameter counts |
| Timestamp | Current run timestamp |

### 1.2 What is NOT logged today

| Missing metric | Impact |
|----------------|--------|
| GPU memory utilization | Cannot detect OOM risk or optimize batch size |
| Wall-clock time per iteration | `tokens_per_sec` exists but raw step time is not logged |
| Loss standard deviation across batches | Cannot detect noisy vs stable training |
| Validation loss trend (delta from previous) | Must be computed offline |
| Gradient norm histogram / per-layer breakdown | Only aggregate norm is logged |
| No structured (JSON) output | Logs are human-readable but hard to parse programmatically |
| No reconstruction visualization | No sample outputs saved at checkpoints |
| No embedding analysis | No t-SNE/PCA of encoder representations |
| No downstream probe metrics | No linear probe accuracy tracked |
| No wandb/tensorboard integration | All logging is to flat text files |

### 1.3 Config knobs

From `base.yaml`:
- `logging.directory`: Log file output directory (default: `experiments/logs`)
- `logging.prefix`: Log file name prefix (default: `fsdp`)
- `logging.level`: Python log level (default: `debug`)
- `misc.monitors_dynamics`: Enable activation/param monitoring (default: `false`)
- `misc.data_dump_on`: Save raw tensor dumps during eval (default: `false`)
- `misc.peak_flops_per_sec`: Reference for MFU calculation (default: `112e12`)
- `checkpoint.chkpt_saving_iterations`: How often to run eval + checkpoint (default: `1`)

---

## 2. Per-Step Metrics

These metrics should be logged every training step (one parameter update). All are
already partially present in MAXIE; the table marks what exists and what to add.

### 2.1 Required per-step metrics

| Metric | Name | Units | Already logged | Priority |
|--------|------|-------|---------------|----------|
| Training loss | `train_loss` | float | Yes (`mean_train_loss`) | -- |
| Learning rate | `learning_rate` | float | Yes | -- |
| Gradient norm (global) | `grad_norm` | float | Yes | -- |
| Throughput | `tokens_per_sec` | tokens/s | Yes | -- |
| MFU | `mfu` | ratio | Yes | -- |
| Step wall time | `step_time_sec` | seconds | No (derivable from tokens_per_sec) | High |
| GPU memory allocated | `gpu_mem_allocated_gb` | GB | No | High |
| GPU memory reserved | `gpu_mem_reserved_gb` | GB | No | Medium |
| Loss scale (float16) | `loss_scale` | float | No | Medium |
| Effective batch size | `effective_batch_size` | int | No (implicit) | Low |

### 2.2 Implementation for missing metrics

**Step wall time** -- Already computed internally as `t_delta`. Just add to the log dict:
```python
"step_time_sec": f"{t_delta:.4f}",
```

**GPU memory** -- Add after `torch.cuda.synchronize()`:
```python
if device_type == "cuda":
    gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024**3)
    gpu_mem_reserved  = torch.cuda.memory_reserved(device) / (1024**3)
```

**Loss scale** -- Add when using float16:
```python
"loss_scale": f"{scaler.get_scale():.1f}" if scaler.is_enabled() else "N/A",
```

### 2.3 Optional per-step metrics (when `monitors_dynamics=true`)

| Metric | Name | Frequency | Notes |
|--------|------|-----------|-------|
| Per-layer activation mean/std | `act.{layer}.mean`, `act.{layer}.std` | Every step | Already implemented |
| Per-layer param update ratio | `param_update.{layer}` | Every step | Already implemented (log10 scale) |
| Per-layer gradient mean/std | `grad.{layer}.mean`, `grad.{layer}.std` | Every step | Available via `monitor_param_update_metrics` |
| Gradient norm per layer group | `grad_norm.encoder`, `grad_norm.decoder` | Every step | Not implemented; compute separately |

---

## 3. Per-Epoch Metrics

An "epoch" in MAXIE is one pass over the full dataset (across all segments). These
should be computed at the end of each epoch (or at a configurable interval like every
N segments).

### 3.1 Aggregated training metrics

| Metric | Name | Computation |
|--------|------|-------------|
| Mean training loss | `epoch_train_loss_mean` | Average of all per-step `train_loss` values in the epoch |
| Std training loss | `epoch_train_loss_std` | Std dev of per-step losses (indicates training stability) |
| Min training loss | `epoch_train_loss_min` | Best single-step loss in epoch |
| Mean gradient norm | `epoch_grad_norm_mean` | Average gradient norm across all steps |
| Max gradient norm | `epoch_grad_norm_max` | Worst-case gradient norm (spike detector) |
| Grad norm spike count | `epoch_grad_spikes` | Count of steps where grad_norm > 5x running mean |
| Mean throughput | `epoch_tokens_per_sec_mean` | Average throughput (detect IO bottlenecks) |
| Total tokens seen | `epoch_total_tokens` | Running count of all tokens processed |
| Mean MFU | `epoch_mfu_mean` | Hardware utilization efficiency |
| NaN loss count | `epoch_nan_count` | Number of NaN losses encountered |

### 3.2 Validation metrics

Run at end of each epoch (or at checkpoint intervals):

| Metric | Name | Notes |
|--------|------|-------|
| Validation loss (mean) | `val_loss_mean` | Already computed at checkpoints |
| Validation loss (std) | `val_loss_std` | Add: std across eval batches |
| Train-val loss gap | `overfit_gap` | `val_loss_mean - epoch_train_loss_mean` |
| Overfit ratio | `overfit_ratio` | `val_loss_mean / epoch_train_loss_mean` |

### 3.3 Reconstruction samples

At each epoch boundary (or checkpoint), save reconstruction visualizations:

```python
# Save K random reconstruction samples
for i in range(K):
    # original: (1, C, H, W) input image
    # reconstructed: model output (reprojected to pixel space)
    # mask: which patches were masked
    save_reconstruction_triplet(
        path=f"diagnostics/epoch{epoch}_sample{i}.png",
        original=original[i],
        reconstructed=pred[i],
        mask=mask[i],
    )
```

Save as PNG files in a per-run diagnostics directory. For diffraction data, use a
log-scale colormap to show both Bragg peaks and diffuse scattering.

---

## 4. Per-Checkpoint Diagnostics

These are deeper analyses run at each checkpoint boundary (controlled by
`chkpt_saving_iterations`). They are more expensive than per-step metrics but provide
critical insight into representation quality.

### 4.1 Checkpoint diagnostic checklist

| Diagnostic | Metric names | Cost | Priority |
|-----------|-------------|------|----------|
| Eval loss (train split) | `chkpt_train_loss` | Low | Already done |
| Eval loss (val split) | `chkpt_val_loss` | Low | Already done |
| Reconstruction visualization | (PNG files) | Low | High |
| Gradient norm distribution | `chkpt_grad_norm_hist` | Low | Medium |
| Weight norm per layer | `chkpt_weight_norm.{layer}` | Low | Medium |
| Attention map entropy | `chkpt_attn_entropy.{layer}` | Medium | Medium |
| Linear probe accuracy | `chkpt_linear_probe_acc` | High | High (periodic) |
| Embedding PCA variance | `chkpt_pca_var_explained` | Medium | Low |
| Reconstruction error by region | `chkpt_recon_err_peak`, `chkpt_recon_err_bg` | Medium | High |

### 4.2 Linear probe evaluation

Run a lightweight linear probe at periodic checkpoints (not every checkpoint -- every
5th or 10th is sufficient for trend detection).

**Protocol:**
1. Freeze the encoder
2. Extract CLS token or mean-pooled patch embeddings for a held-out labeled subset
3. Train a single linear layer (SGD, 100 epochs, lr=0.1, no augmentation)
4. Report top-1 accuracy on the evaluation set

For MAXIE/diffraction data, the downstream task could be:
- Crystal system classification (if labels available)
- Peak/no-peak binary classification per patch
- Detector type classification

**Implementation reference:** The Scenic `linear_probe_utils.py` provides a clean
pattern: `LinearProbe(nn.Module)` with `stop_gradient`, trained for a fixed number
of steps, evaluated on a held-out set. Adapt to PyTorch:

```python
class LinearProbe(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(x.detach())  # stop gradient to encoder
```

### 4.3 Reconstruction error decomposition

For diffraction data, decompose reconstruction error into:

| Region | Definition | Why it matters |
|--------|-----------|---------------|
| Bragg peak regions | Patches with intensity > mean + 3*std | Model must reconstruct sparse bright features |
| Background regions | Patches with intensity < mean + 1*std | Diffuse scattering, easier to reconstruct |
| Beam stop region | Central patches (detector-dependent) | Often masked in real data |

```python
def decomposed_recon_error(original, reconstructed, mask):
    error = (original - reconstructed) ** 2
    peak_mask = original > (original.mean() + 3 * original.std())
    bg_mask = ~peak_mask

    peak_error = error[mask & peak_mask].mean()
    bg_error = error[mask & bg_mask].mean()
    return peak_error, bg_error
```

### 4.4 Attention map analysis

At checkpoint boundaries, extract attention maps from the last encoder layer and
compute:
- **Entropy per head:** `H = -sum(attn * log(attn))`. Low entropy means the head
  attends to few patches (specialized). Very low entropy across all heads suggests
  attention collapse.
- **Attention to peak regions vs background:** Are heads attending to Bragg peaks?

---

## 5. SSL-Method-Specific Diagnostics

MAXIE may use different SSL methods (see `ssl-candidates-for-maxie.md`). Each has
unique failure modes requiring specific diagnostics.

### 5.1 MAE (Masked Autoencoder) -- Current Method

**Key metrics:**

| Metric | Name | Healthy range | Failure signal |
|--------|------|--------------|----------------|
| Pixel reconstruction loss (MSE) | `mae_recon_loss` | Monotonically decreasing | Plateau or increase |
| Masked patch loss vs visible patch loss | `mae_masked_loss`, `mae_visible_loss` | Masked >> visible | If equal, model is not learning from context |
| Reconstruction PSNR | `mae_psnr_db` | Increasing (20-40 dB typical) | Stagnation < 15 dB |
| Per-frequency reconstruction error | `mae_freq_err_low`, `mae_freq_err_high` | Both decreasing | High-freq stagnation = model learns only smooth features |

**MAE-specific diagnostics:**

1. **Mask ratio sensitivity:** At checkpoints, evaluate with mask ratios 0.5, 0.75,
   0.9. If performance degrades sharply above 0.75, the model has not learned strong
   semantic representations yet.

2. **Norm pix loss check:** If `norm_pix_loss=false` (current MAXIE default), the
   loss is dominated by high-intensity patches. The training-playbook recommends
   switching to `norm_pix_loss=true`. Monitor whether loss scale changes after
   switching.

3. **Reconstruction quality by mask pattern:** Compare random masking vs block masking
   to assess whether the model relies on local or global context.

### 5.2 VQ-VAE (Vector Quantized VAE)

**Key metrics (from `research-loop-brainstorm.md`):**

| Metric | Name | Healthy range | Failure signal |
|--------|------|--------------|----------------|
| Codebook utilization | `vqvae_codebook_util` | > 80% of codes used | < 50% = dead codes |
| Codebook perplexity | `vqvae_codebook_perplexity` | Close to `codebook_size` | < 50% of `codebook_size` = collapse |
| Dead code fraction | `vqvae_dead_code_frac` | < 5% after warmup | > 20% after 10K steps = reset needed |
| Commitment loss | `vqvae_commit_loss` | Decreasing then stable | Increasing = encoder diverging from codebook |
| Reconstruction loss | `vqvae_recon_loss` | Decreasing | Plateau = codebook capacity saturated |
| VQ loss (codebook loss) | `vqvae_vq_loss` | Decreasing then stable | Increasing = codebook drift |
| Codebook embedding norm | `vqvae_embed_norm_mean`, `_std` | Stable | Growing unboundedly = normalization issue |

**VQ-VAE-specific diagnostics:**

1. **Codebook utilization histogram:** At each checkpoint, compute a histogram of
   code usage counts across the evaluation set. Plot as a bar chart. Healthy = roughly
   uniform. Collapsed = a few bars dominate.

   ```python
   def codebook_utilization(indices, codebook_size):
       counts = torch.bincount(indices.flatten(), minlength=codebook_size)
       utilization = (counts > 0).float().mean().item()
       perplexity = torch.exp(-torch.sum(
           (counts.float() / counts.sum()) *
           torch.log(counts.float() / counts.sum() + 1e-10)
       )).item()
       dead_fraction = (counts == 0).float().mean().item()
       return utilization, perplexity, dead_fraction
   ```

2. **Dead code tracking over time:** Plot dead code fraction vs training step. Should
   decrease. If it increases, the reset strategy is failing.

3. **Commitment loss balance:** The total VQ-VAE loss is
   `recon_loss + vq_loss + beta * commit_loss`. Log each component separately.
   If `commit_loss` dominates, reduce beta. If `vq_loss` dominates, the codebook
   is not being updated fast enough (increase EMA decay or switch to straight-through).

**Techniques for improving codebook utilization** (from `vector-quantize-pytorch`):
- Lower codebook dimension (`codebook_dim=16-32`) to increase usage
- Cosine similarity matching (`use_cosine_sim=True`) for better code distribution
- Expiring stale codes (`threshold_ema_dead_code=2`) to replace dead entries
- K-means initialization (`kmeans_init=True`) for better starting codebook

### 5.3 I-JEPA (Image Joint Embedding Predictive Architecture)

**Key metrics:**

| Metric | Name | Healthy range | Failure signal |
|--------|------|--------------|----------------|
| Prediction loss (MSE in repr space) | `ijepa_pred_loss` | Decreasing | Plateau = predictor capacity issue |
| EMA teacher-student divergence | `ijepa_teacher_student_dist` | Small, stable | Growing = EMA rate too slow |
| Target representation variance | `ijepa_target_var` | Stable, non-zero | Collapsing to zero = representation collapse |
| Predictor output variance | `ijepa_pred_var` | Close to target variance | Much smaller = predictor is averaging |
| Feature std across batch | `ijepa_feat_std` | > 0.01 | Near zero = embedding collapse |

**I-JEPA-specific diagnostics:**

1. **Representation collapse detection:** The primary failure mode. Monitor the
   standard deviation of encoder outputs across a batch. If std approaches zero,
   the model is collapsing to a constant representation.

   ```python
   def check_collapse(encoder_output):
       # encoder_output: (B, N, D) -- batch, patches, embed_dim
       batch_std = encoder_output.std(dim=0).mean()  # std across batch for each feature
       feature_std = encoder_output.std(dim=-1).mean()  # std across features
       return batch_std.item(), feature_std.item()
   ```

2. **EMA momentum schedule verification:** I-JEPA uses an EMA teacher with momentum
   increasing during training (e.g., 0.996 to 1.0). Log the current EMA momentum and
   verify it follows the expected schedule.

3. **Context-target mask overlap:** I-JEPA masks context and target regions differently.
   Log the fraction of overlap -- zero overlap is typical. If the training code has a
   bug, overlap could leak information.

---

## 6. Logging Format Specification

### 6.1 Structured JSON Lines format

All new monitoring output should be in JSON Lines format (one JSON object per line),
written to a separate file from the existing Python logger output. This enables
programmatic parsing by the research loop agent.

**File naming convention:**
```
{log_directory}/{prefix}.metrics.{timestamp}.jsonl
```

**Schema for per-step metrics:**

```json
{
  "event": "step",
  "timestamp_utc": "2026-03-19T14:30:00.123Z",
  "iteration": 42,
  "epoch": 0,
  "segment": "0-4",
  "metrics": {
    "train_loss": 0.456789,
    "learning_rate": 1.5e-4,
    "grad_norm": 0.123456,
    "tokens_per_sec": 120000.0,
    "mfu": 0.012,
    "step_time_sec": 1.234,
    "gpu_mem_allocated_gb": 48.2,
    "gpu_mem_reserved_gb": 52.1,
    "loss_scale": 65536.0,
    "effective_batch_size": 16
  }
}
```

**Schema for per-step dynamics (optional):**

```json
{
  "event": "dynamics",
  "iteration": 42,
  "activations": {
    "vit.encoder.layer.0.intermediate.act_fn": {
      "pre_mean": 0.001, "pre_std": 0.45,
      "post_mean": 0.12, "post_std": 0.33
    }
  },
  "param_updates": {
    "encoder.layer.0.attention.self.query.weight": -2.5,
    "decoder.layers.0.self_attn.q_proj.weight": -2.8
  }
}
```

**Schema for checkpoint diagnostics:**

```json
{
  "event": "checkpoint",
  "iteration": 1000,
  "epoch": 1,
  "metrics": {
    "train_loss": 0.321,
    "val_loss": 0.345,
    "overfit_gap": 0.024,
    "linear_probe_acc": 0.72,
    "recon_psnr_db": 28.5,
    "recon_err_peak": 0.089,
    "recon_err_bg": 0.012
  },
  "checkpoint_path": "/path/to/checkpoint",
  "reconstruction_samples": [
    "diagnostics/iter1000_sample0.png",
    "diagnostics/iter1000_sample1.png"
  ]
}
```

**Schema for VQ-VAE diagnostics:**

```json
{
  "event": "vqvae_diagnostics",
  "iteration": 1000,
  "metrics": {
    "codebook_utilization": 0.85,
    "codebook_perplexity": 435.2,
    "dead_code_fraction": 0.15,
    "recon_loss": 0.321,
    "commit_loss": 0.045,
    "vq_loss": 0.012,
    "embed_norm_mean": 1.23,
    "embed_norm_std": 0.15
  },
  "codebook_histogram": [12, 45, 3, 0, 78, ...]
}
```

### 6.2 Implementation pattern

```python
import json
import time

class JSONMetricsLogger:
    """Structured metrics logger for agent-parseable output."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.fh = open(filepath, 'a')

    def log(self, event_type, iteration, epoch=None, **kwargs):
        record = {
            "event": event_type,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "iteration": iteration,
        }
        if epoch is not None:
            record["epoch"] = epoch
        record.update(kwargs)
        self.fh.write(json.dumps(record) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()
```

### 6.3 Backward compatibility

The existing pipe-delimited log format should be preserved for human readability.
The JSON Lines logger runs in parallel, writing to a separate `.jsonl` file. Only
rank 0 writes to the JSON logger.

---

## 7. Decision Trees

These tables map observed metric patterns to recommended actions. They are designed
to be consumed by both human operators and the automated research loop agent.

### 7.1 Loss Divergence

```
IF train_loss becomes NaN or Inf:
  1. Check loss_scale -- if it dropped to 0 or near 0:
     -> float16 overflow. Switch to bfloat16 or reduce learning rate by 2x.
  2. Check grad_norm at the step before NaN:
     -> If grad_norm > 100: gradient explosion. Reduce grad_clip to 0.5.
     -> If grad_norm was normal: likely bad data batch. Enable input validation.
  3. Check if NaN appeared during warmup:
     -> Yes: warmup too aggressive. Increase warmup_iterations by 4x.
     -> No: mid-training instability. Reduce LR by 2x and resume from last checkpoint.

IF train_loss increases for > 500 consecutive steps:
  -> Likely diverging. Reduce LR by 50% or increase warmup.
  -> If already at low LR: check data pipeline for corruption.
```

### 7.2 Loss Plateau

```
IF train_loss has not decreased by > 1% in the last 10K steps:
  1. Check learning_rate:
     -> If LR is near min_lr (< 1e-6): training is in late cosine decay.
        This is expected. Check if val_loss is also plateaued.
     -> If LR is still high (> 1e-4): model capacity may be saturated.
        Action: scale model (ViT-Base -> ViT-Large).
  2. Check if norm_pix_loss is false:
     -> Switch to true. Loss landscape changes; plateau may break.
  3. Check mask_ratio:
     -> If 0.75 and data is spatially redundant (smooth background):
        increase to 0.85 to make the task harder.
     -> If 0.75 and data has sparse peaks:
        decrease to 0.5 to keep some peaks visible.
  4. Check effective_batch_size vs LR:
     -> If batch_size changed without LR scaling: apply linear scaling rule.
```

### 7.3 Gradient Explosion

```
IF grad_norm > 10x running_mean(grad_norm) for any single step:
  -> Likely a spike. If isolated (< 3 occurrences per epoch): usually benign.
     Gradient clipping handles it.
  -> If frequent (> 10 spikes per epoch):
     1. Reduce grad_clip from 1.0 to 0.5
     2. If spikes persist: reduce LR by 2x
     3. If spikes correlate with specific data segments: inspect those segments

IF grad_norm is consistently > 10.0 (even after clipping):
  -> Clipping is not sufficient. The model is in an unstable region.
  -> Reduce LR by 4x and increase warmup.
  -> If using float16: switch to bfloat16 (wider dynamic range).
```

### 7.4 Codebook Collapse (VQ-VAE)

```
IF codebook_utilization < 0.5 after 5K steps:
  1. Check commitment loss weight (beta):
     -> If beta > 0.5: too high. Reduce to 0.1-0.25.
  2. Check if dead code reset is enabled:
     -> If no: enable threshold_ema_dead_code = 2
  3. Check codebook_size:
     -> If > 2048 and dataset is small: reduce to 512-1024.

IF codebook_perplexity < 0.3 * codebook_size after 10K steps:
  -> Codebook is collapsing. Immediate actions:
     1. Enable cosine similarity matching (use_cosine_sim=True)
     2. Reduce codebook dimension (codebook_dim=16)
     3. Initialize with k-means on first batch (kmeans_init=True)
     4. If all else fails: reduce codebook_size by 2x

IF dead_code_fraction is increasing over time:
  -> Reset strategy is not working.
  -> Switch reset strategy: EMA -> threshold-based -> random reinit from data
  -> Check if encoder outputs are collapsing (low variance)
```

### 7.5 Overfitting

```
IF val_loss increases while train_loss decreases for > 3 consecutive checkpoints:
  -> Overfitting detected.
  1. Check overfit_ratio (val_loss / train_loss):
     -> If > 1.5: severe. Increase mask_ratio by 0.1 (more regularization).
     -> If 1.1-1.5: moderate. Increase weight_decay from 0.05 to 0.1.
  2. Check dataset size:
     -> If < 10K images: add data augmentation (random flip, rotation).
  3. Check model size:
     -> If ViT-Large or larger on < 50K images: switch to ViT-Base.
  4. For MAE: increase mask ratio (0.75 -> 0.85).
     For VQ-VAE: no easy fix; reduce model size.
```

### 7.6 Underfitting

```
IF both train_loss and val_loss are high and plateaued:
  -> Model is not learning.
  1. Check learning_rate:
     -> If very low (< 1e-5): LR not scaled for batch size. Apply linear scaling.
  2. Check warmup:
     -> If warmup_iterations < 100: too short. Increase to 5-10% of total steps.
  3. Check data pipeline:
     -> Are images normalized? Check transform mean/std vs actual data distribution.
     -> Is the dataloader returning correct data? Enable data_dump_on for one step.
  4. Check model initialization:
     -> If from_scratch=true and INIT logs show large std: initialization may be wrong.
     -> If from_scratch=false: pretrained weights may not transfer to 1-channel data.
```

### 7.7 Decision Summary Table

| Signal | Pattern | Severity | Action |
|--------|---------|----------|--------|
| `train_loss = NaN` | Any step | Critical | Halt, diagnose, resume from checkpoint |
| `grad_norm > 100` | Single step | Warning | Monitor; if repeated, reduce LR |
| `grad_norm > 10x mean` | > 10/epoch | High | Reduce grad_clip to 0.5 |
| `train_loss flat > 10K steps` | Sustained | Medium | Check LR, norm_pix_loss, mask_ratio |
| `val_loss increasing` | > 3 checkpoints | Medium | Increase regularization |
| `codebook_util < 0.5` | After 5K steps | High | Adjust beta, enable reset, reduce size |
| `mfu < 0.01` | Sustained | Low | Performance issue; check data loading |
| `gpu_mem > 90%` | Any step | Warning | Reduce batch_size or enable more aggressive checkpointing |
| `tokens_per_sec dropping` | Over time | Medium | Data pipeline bottleneck; check IO |

---

## 8. Cross-Run Comparison

### 8.1 Comparison scenarios

| Scenario | What varies | What to compare | How many runs |
|----------|-----------|-----------------|---------------|
| Scaling law study | Model size, data size, compute budget | Loss vs compute (chinchilla curves) | 10-30 |
| SSL method comparison | MAE vs VQ-VAE vs I-JEPA | Val loss, linear probe acc, recon quality | 3-9 |
| Hyperparameter sweep | LR, weight_decay, mask_ratio, etc. | Final val loss, convergence speed | 4-16 |
| Batch size scaling | Node count, effective batch size | Loss curve, throughput, convergence | 4-8 |
| Codebook exploration | codebook_size, beta, reset strategy | Utilization, perplexity, recon loss | 4-8 |

### 8.2 Comparison dashboard specification

Build a Python script `compare_runs.py` that:
1. Accepts a list of JSONL metric files (one per run)
2. Parses all `event=step` records
3. Produces comparison plots:

**Essential plots:**

| Plot | X-axis | Y-axis | Lines |
|------|--------|--------|-------|
| Loss curves | iteration | train_loss | One line per run |
| Validation loss | iteration | val_loss | One line per run |
| Learning rate schedule | iteration | learning_rate | One line per run |
| Gradient norm | iteration | grad_norm | One line per run |
| Throughput scaling | num_nodes | tokens_per_sec | One point per run |
| MFU scaling | num_nodes | mfu | One point per run |
| Loss vs compute | total_flops | val_loss | Scaling law curve |

**For SSL method comparison, add:**

| Plot | Data source |
|------|------------|
| Reconstruction samples grid | Side-by-side: original, MAE recon, VQ-VAE recon, I-JEPA (N/A) |
| Linear probe accuracy over training | One curve per SSL method |
| Codebook diagnostics (VQ-VAE only) | Utilization, perplexity, dead codes over time |

### 8.3 Comparison table output

The script should also produce a markdown summary table:

```markdown
| Run ID | SSL Method | Model | Batch Size | Final Train Loss | Final Val Loss | Linear Probe Acc | Tokens/sec | MFU |
|--------|-----------|-------|-----------|-----------------|---------------|-----------------|-----------|-----|
| run-001 | MAE | ViT-B | 1024 | 0.234 | 0.256 | 0.72 | 1.2e5 | 0.012 |
| run-002 | VQ-VAE | ViT-B | 1024 | 0.198 | 0.241 | 0.68 | 1.1e5 | 0.011 |
```

### 8.4 Scaling law analysis

For scaling law studies, fit the standard power law:

```
L(C) = a * C^(-alpha) + L_inf
```

Where `L` is validation loss, `C` is total compute (FLOPs), and `L_inf` is the
irreducible loss floor. Compute C as:

```
C = 6 * N * D
```

Where `N` is model parameters and `D` is tokens seen. Log both `N`, `D`, and `C`
in the JSONL output.

### 8.5 Implementation sketch

```python
#!/usr/bin/env python3
"""compare_runs.py -- Cross-run comparison dashboard for MAXIE."""

import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(jsonl_path):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def extract_step_metrics(records):
    steps = [r for r in records if r["event"] == "step"]
    return {
        "iterations": [s["iteration"] for s in steps],
        "train_loss": [s["metrics"]["train_loss"] for s in steps],
        "grad_norm":  [s["metrics"]["grad_norm"] for s in steps],
        "lr":         [s["metrics"]["learning_rate"] for s in steps],
        "mfu":        [s["metrics"].get("mfu", 0) for s in steps],
    }

def plot_comparison(run_data, metric_name, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_id, data in run_data.items():
        ax.plot(data["iterations"], data[metric_name], label=run_id, alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    run_files = sys.argv[1:]
    run_data = {}
    for f in run_files:
        run_id = Path(f).stem
        run_data[run_id] = extract_step_metrics(load_metrics(f))
    for metric in ["train_loss", "grad_norm", "lr", "mfu"]:
        plot_comparison(run_data, metric, f"comparison_{metric}.png")
```

---

## 9. Integration with Research Loop

The research loop agent (described in `research-loop-brainstorm.md`) needs to read
diagnostic outputs to make decisions about the next experimental configuration. This
section defines the contract between the training pipeline and the agent.

### 9.1 What the agent reads

The agent parses the JSONL metrics file after each training run completes. It reads:

| Data | File | Agent action |
|------|------|-------------|
| Final metrics | Last `event=checkpoint` record | Compare against baseline |
| Loss trajectory | All `event=step` records | Detect divergence, plateau, instability |
| Codebook diagnostics | `event=vqvae_diagnostics` records | Decide codebook config changes |
| Linear probe accuracy | `chkpt_linear_probe_acc` in checkpoint records | Determine representation quality |
| Reconstruction samples | PNG files listed in checkpoint records | Visual quality check (if multimodal agent) |

### 9.2 Agent decision interface

The agent uses the following decision procedure after reading run diagnostics:

```
1. PARSE metrics from JSONL file
2. COMPUTE summary statistics:
   - final_train_loss, final_val_loss
   - convergence_speed (steps to reach 90% of final loss)
   - stability_score (1 - fraction of grad_norm spikes)
   - codebook_health (utilization * perplexity / codebook_size)  [VQ-VAE only]
3. COMPARE against baseline (previous best run):
   - If val_loss < baseline_val_loss * 0.98: KEEP (>2% improvement)
   - If val_loss within 2% of baseline: KEEP-DEFERRED (marginal)
   - If val_loss > baseline_val_loss * 1.02: DISCARD
4. DIAGNOSE any anomalies using Decision Trees (Section 7)
5. HYPOTHESIZE next configuration based on diagnosis
6. LOG result to experiments.db
```

### 9.3 Required outputs for the agent

The training script MUST produce these files for the agent to function:

| File | Format | Content |
|------|--------|---------|
| `{run_dir}/metrics.jsonl` | JSON Lines | All per-step and per-checkpoint metrics |
| `{run_dir}/config.yaml` | YAML | Full training configuration (reproducibility) |
| `{run_dir}/diagnostics/` | Directory | Reconstruction PNGs, histograms |
| `{run_dir}/summary.json` | JSON | Single file with final metrics for quick parsing |

**Summary file schema:**

```json
{
  "run_id": "exp-1-1",
  "ssl_method": "mae",
  "model": "vit-base",
  "total_iterations": 50000,
  "total_tokens": 1234567890,
  "final_train_loss": 0.234,
  "final_val_loss": 0.256,
  "linear_probe_acc": 0.72,
  "mean_mfu": 0.012,
  "mean_throughput": 120000.0,
  "total_wall_time_sec": 3600,
  "grad_norm_spikes": 3,
  "nan_count": 0,
  "codebook_utilization": null,
  "codebook_perplexity": null,
  "status": "completed"
}
```

### 9.4 Failure recovery

If the agent detects a failed run (NaN losses, OOM, NCCL timeout):

1. Read the last valid checkpoint iteration from `metrics.jsonl`
2. Read the error log to classify the failure:
   - `NaN loss` -> Reduce LR, increase warmup
   - `CUDA OOM` -> Reduce batch_size or enable more activation checkpointing
   - `NCCL timeout` -> Infrastructure issue; retry with same config
   - `KeyboardInterrupt` -> Preemption; resume from preemptive checkpoint
3. Generate a recovery config and resubmit

### 9.5 Batch mode integration

In batch mode (K experiments in parallel, per `research-loop-brainstorm.md`), the
agent reads K summary files and performs pairwise comparison:

```
FOR each experiment in batch:
    PARSE summary.json
    COMPARE against baseline
    CLASSIFY as KEEP / KEEP-DEFERRED / DISCARD

SELECT winner = experiment with lowest val_loss among KEEP results
MERGE winner into research branch
LOG all K results to experiments.db with shared batch_id
UPDATE insights.md with diagnostic observations
```

---

## 10. Key Files Reference

### 10.1 MAXIE source files

| File | Path | Relevance |
|------|------|-----------|
| Training script | `$MAXIE_DIR/train/train.fsdp.py` | Main training loop, all current logging |
| Monitor utilities | `$MAXIE_DIR/maxie/utils/monitor.py` | `ActivationMonitor`, `GradientMonitor`, `monitor_param_update_metrics` |
| Base config | `$MAXIE_DIR/train/hydra_config/train_config/base.yaml` | All training hyperparameters |
| LR scheduler | `$MAXIE_DIR/maxie/lr_scheduler.py` | `CosineLRScheduler` implementation |
| ViT-MAE model | HuggingFace `transformers` (`modeling_vit_mae.py`) | Loss computation, forward pass |

### 10.2 Agent documentation

| File | Path | Relevance |
|------|------|-----------|
| Training playbook | `$PROJ_DIR/docs/agents/training-playbook-for-maxie.md` | Sections 7.3, 8 (checkpoint diagnostics, failure modes) |
| Research loop brainstorm | `$PROJ_DIR/docs/agents/research-loop-brainstorm.md` | Diagnostic signals, batch mode design |
| Frontier strategy | `$PROJ_DIR/docs/agents/research-loop-frontier-strategy.md` | Resource allocation, Flux integration |
| SSL candidates | `$PROJ_DIR/docs/agents/ssl-candidates-for-maxie.md` | SSL method comparison |
| This document | `$PROJ_DIR/docs/agents/monitoring-protocol-for-maxie.md` | Monitoring and diagnostics protocol |

### 10.3 Reference implementations from docs collection

| Implementation | Path (under `$DEEPLEARNING_DOC_DIR`) | What to reference |
|---------------|--------------------------------------|-------------------|
| HuggingFace ViT-MAE | `pytorch-proj-repos/transformers/src/transformers/models/vit_mae/modeling_vit_mae.py` | `ViTMAEForPreTrainingOutput` loss/logits/mask structure |
| Scenic linear probe | `computer-vision/scenic/scenic/train_lib/transfer/linear_probe_utils.py` | `LinearEvaluator.run_all()` pattern |
| vector-quantize-pytorch | `lucidrain-repos/vector-quantize-pytorch/README.md` | Codebook utilization techniques, dead code reset |
| PyTorch Lightning EarlyStopping | `pytorch-proj-repos/pytorch-lightning/src/lightning/pytorch/callbacks/early_stopping.py` | Divergence threshold, patience pattern |
| Accelerate tracking | `pytorch-proj-repos/accelerate/docs/source/usage_guides/tracking.md` | Multi-process logging pattern |

### 10.4 Scripts to build

These scripts do not exist yet. They should be created as part of implementing this
monitoring protocol:

| Script | Purpose | Priority |
|--------|---------|----------|
| `maxie/utils/json_logger.py` | `JSONMetricsLogger` class (Section 6.2) | P0 -- required for agent |
| `scripts/compare_runs.py` | Cross-run comparison dashboard (Section 8.5) | P1 -- needed for scaling studies |
| `scripts/codebook_diagnostics.py` | VQ-VAE codebook analysis (Section 5.2) | P1 -- needed for VQ-VAE experiments |
| `scripts/linear_probe_eval.py` | Linear probe evaluation (Section 4.2) | P1 -- needed for representation quality |
| `scripts/recon_visualize.py` | Reconstruction visualization (Section 3.3) | P2 -- useful but not blocking |
| `scripts/attention_analysis.py` | Attention map entropy analysis (Section 4.4) | P2 -- nice to have |
| `scripts/generate_summary.py` | Generate `summary.json` from JSONL (Section 9.3) | P0 -- required for agent |

---

## Appendix A: Metric Naming Convention

All metric names follow this hierarchy:

```
{scope}.{component}.{measurement}
```

Examples:
- `train.loss` -- training loss
- `val.loss` -- validation loss
- `grad.norm` -- global gradient norm
- `grad.encoder.layer.0.norm` -- per-layer gradient norm
- `act.encoder.layer.0.gelu.mean` -- activation statistic
- `vqvae.codebook.utilization` -- VQ-VAE codebook metric
- `ijepa.target.variance` -- I-JEPA target representation variance

For JSONL output, use flat keys with underscores (no dots) to simplify JSON parsing:
`train_loss`, `grad_norm`, `vqvae_codebook_utilization`.

## Appendix B: Thresholds Quick Reference

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| `grad_norm` | < 1.0 | 1.0 - 10.0 | > 10.0 |
| `train_loss` trend | Decreasing | Flat > 5K steps | Increasing > 500 steps |
| `val_loss - train_loss` | < 10% of train_loss | 10-50% | > 50% |
| `mfu` | > 0.3 | 0.1 - 0.3 | < 0.1 |
| `gpu_mem_allocated` | < 80% of HBM | 80-90% | > 90% |
| `loss_scale` (float16) | > 1024 | 64 - 1024 | < 64 or 0 |
| `codebook_utilization` | > 0.8 | 0.5 - 0.8 | < 0.5 |
| `codebook_perplexity` | > 0.7 * codebook_size | 0.3 - 0.7 * size | < 0.3 * size |
| `dead_code_fraction` | < 0.05 | 0.05 - 0.2 | > 0.2 |
| `ijepa_feat_std` | > 0.1 | 0.01 - 0.1 | < 0.01 (collapse) |
| `nan_count` per epoch | 0 | 1-3 | > 3 |
