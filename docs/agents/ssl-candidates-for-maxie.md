# Self-Supervised Learning Candidates for MAXIE

Date: 2026-03-19
Context: ALCC project Group 1 — evaluating SSL pretraining paradigms for the MAXIE
foundation model on X-ray crystallography diffraction images (single-channel, Bragg
peaks, variable dynamic range).

Current approach: **MAE** (`ViTMAEForPreTraining` from HuggingFace, adapted to 1-channel
input in `maxie/modeling/adapted_mae.py`).

---

## Domain Constraints

These properties of X-ray diffraction data shape which SSL methods transfer well:

| Property | Implication for SSL |
|----------|-------------------|
| Single channel (grayscale) | Color jitter augmentation is meaningless |
| Bragg peaks are spatially sparse, high-contrast | Random crop risks cutting peaks; spatial relationships to beam center matter |
| Quantitative intensity matters | Perceptual losses (LPIPS) don't apply; need MSE/L1 |
| Dynamic range: 16-bit or 32-bit float | Pre-trained natural-image models assume 8-bit RGB |
| Downstream tasks: peak finding, classification, denoising | Representations need both spatial precision and semantic understanding |

---

## Candidate 1: MAE (Masked Autoencoder) — Current Baseline

**Paper:** He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021)

**Pretext task:** Mask 75% of image patches, reconstruct missing pixels from the
remaining 25% using an asymmetric encoder-decoder.

**Architecture:** ViT encoder (processes only visible patches) + lightweight ViT decoder
(reconstructs all patches). Encoder output is the learned representation.

**Strengths for diffraction:**
- Pixel-level reconstruction forces the model to learn spatial structure of diffraction
  patterns, including Bragg peak geometry and intensity profiles
- High mask ratio (75%) creates a non-trivial task — the model must understand the
  global diffraction pattern to infer missing regions
- Asymmetric design is compute-efficient (encoder sees only 25% of tokens)
- Already adapted in the MAXIE codebase; HuggingFace checkpoints available
  (`facebook/vit-mae-base`, `facebook/vit-mae-large`, `facebook/vit-mae-huge`)
- No augmentation dependency — the masking itself is the pretext task

**Weaknesses:**
- Pixel reconstruction may over-invest in low-level detail (exact background noise)
  at the expense of higher-level semantic structure
- Reconstruction loss (MSE) treats all pixels equally unless weighted — Bragg peaks
  occupy a tiny fraction of pixels but carry most of the information
- Decoder is discarded after pre-training; its capacity is "wasted"

**Scaling:** ViT-Huge achieves 87.8% on ImageNet-1K. MAXIE proposal plans Base -> Large
-> Huge scaling study.

**What to explore:**
- Contrast-aware reconstruction loss (weight Bragg peak regions higher) — see
  `research-loop-brainstorm.md` Candidate 2
- Patch size tuning (16x16, 32x32, 64x64) for diffraction-appropriate receptive fields
- Mask ratio tuning (75% may not be optimal for sparse diffraction patterns)

**Key implementations:**
- MAXIE: `$MAXIE_DIR/maxie/modeling/adapted_mae.py`
- HuggingFace: `transformers.ViTMAEForPreTraining`
- Scenic (JAX): `scenic/projects/av_mae/`
- Docs: `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/mmsegmentation/configs/mae/README.md`

---

## Candidate 2: VQ-VAE (Vector Quantized Variational Autoencoder)

**Paper:** van den Oord et al., "Neural Discrete Representation Learning" (2017)

**Pretext task:** Encode image to spatial grid of continuous vectors, quantize each to
the nearest entry in a learned codebook, reconstruct from the quantized representation.

**Architecture:** CNN encoder -> vector quantization layer (codebook lookup) -> CNN
decoder. The codebook is a set of K learned embedding vectors of dimension D.

**Training losses:**
- Reconstruction loss (MSE between input and output)
- Codebook loss (move codebook vectors toward encoder outputs)
- Commitment loss (move encoder outputs toward codebook vectors, weight = beta)

**Strengths for diffraction:**
- Discrete codebook naturally learns a *vocabulary of diffraction motifs*: different
  peak shapes, background textures, ring patterns, dead regions
- Codebook health is directly interpretable — utilization histogram, dead code fraction,
  perplexity — giving diagnostic signals beyond a single loss number
- Discrete tokens enable downstream tokenized modeling (BEiT-style, or even
  autoregressive generation of diffraction patterns)
- Well-suited to the research loop exploration pattern (see `research-loop-brainstorm.md`
  Candidate 1) — codebook size, beta, reset strategy are structural decisions that
  benefit from agent-driven diagnosis

**Weaknesses:**
- Codebook collapse: codes go unused ("dead codes"), effective vocabulary shrinks
- Commitment loss beta requires tuning — too low = encoder ignores codebook, too high =
  encoder outputs collapse to few codes
- CNN encoder/decoder (standard VQ-VAE) lacks the global receptive field of ViT — may
  miss long-range spatial correlations in diffraction patterns
- Two moving parts (encoder + codebook) make training dynamics more complex than MAE

**Variants to consider:**
- **Residual VQ (RVQ):** Multiple codebooks applied sequentially to the residual,
  capturing progressively finer detail. Better reconstruction at same total codebook size.
- **Finite Scalar Quantization (FSQ):** Replaces codebook lookup with per-dimension
  scalar quantization. No codebook collapse possible. Simpler.
- **VQ-VAE-2:** Hierarchical — coarse codebook for global structure, fine codebook for
  local detail. Natural fit for diffraction (global: lattice symmetry; local: peak shape).

**Decision axes for exploration (from brainstorm doc):**
- Codebook size: 256, 512, 1024, 2048, 4096
- Commitment loss weight (beta): 0.1, 0.25, 0.5, 1.0
- EMA decay rate for codebook update
- Dead code reset strategy: EMA, random reinit from data, threshold-based pruning
- Codebook dimensionality (embedding dim)

**Key implementations:**
- `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/PyTorch-VAE/models/vq_vae.py`
- `$DEEPLEARNING_DOC_DIR/lucidrain-repos/vector-quantize-pytorch/` (ResidualVQ,
  GroupedResidualVQ, FSQ, and more)
- HuggingFace: used as tokenizer in BEiT, GLM-Image, DAC (audio)

---

## Candidate 3: I-JEPA (Image Joint Embedding Predictive Architecture)

**Paper:** Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture" (Meta AI, 2023)

**Pretext task:** Given visible patches, predict the *abstract representation* of
masked target patches. Crucially, prediction happens in representation space (output of
an EMA teacher encoder), not in pixel space.

**Architecture:**
- Context encoder: processes visible (unmasked) patches -> context representations
- Predictor: takes context representations + target position embeddings -> predicted
  target representations
- Target encoder: EMA copy of context encoder, processes target patches -> target
  representations (the prediction target)
- No decoder needed — only the context encoder is used downstream

**Strengths for diffraction:**
- **No augmentation dependency.** This is the biggest advantage. SimCLR/DINO/BYOL all
  require defining "what transformations should the representation be invariant to" —
  a question with no good answer for diffraction data. I-JEPA sidesteps this entirely.
- Predicts semantic representations, not pixels — avoids wasting capacity on
  reconstructing exact noise patterns or background
- The model learns "given the peaks I can see, what kind of pattern should exist in
  this masked region?" — a semantically meaningful task for crystallography
- ViT-based, same architecture family as MAE — easy to compare

**Weaknesses:**
- The prediction target (EMA teacher output) is an internal bootstrap — harder to
  diagnose than pixel MSE or codebook utilization. If training goes wrong, it's less
  obvious why.
- Pre-trained checkpoints (`facebook/ijepa_vith14_1k`) are on natural images — transfer
  to single-channel diffraction is unproven. Likely needs from-scratch training.
- Less mature ecosystem than MAE. Fewer reference implementations for custom domains.
- The multi-block masking strategy (predicting large contiguous regions) may interact
  differently with sparse diffraction patterns than with dense natural images.

**What makes it different from MAE:**

| | MAE | I-JEPA |
|---|---|---|
| Prediction target | Pixels | Abstract representations |
| Loss | MSE in pixel space | MSE in representation space |
| What it avoids learning | Nothing — reconstructs everything | Low-level noise, exact textures |
| Augmentation needed | No (masking is the task) | No (masking is the task) |
| Diagnostic clarity | High (look at reconstructions) | Lower (representation similarity) |

**Key implementations:**
- HuggingFace: `IJepaModel`, `IJepaForImageClassification`
- Checkpoints: `facebook/ijepa_vith14_1k`, `facebook/ijepa_vitg16_22k`
- Docs: `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/docs/source/en/model_doc/ijepa.md`

---

## Candidate 4: DINOv2 (Self-Distillation with No Labels, v2)

**Paper:** Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
(Meta AI, 2023); Darcet et al., "Vision Transformers Need Registers" (2024)

**Pretext task:** Self-distillation — a student network (receiving masked/augmented views)
learns to match the output of a teacher network (EMA, receiving full/augmented views).
Combines masked image modeling with self-distillation.

**Architecture:** ViT student + ViT teacher (EMA). DINOv2 with registers adds extra
learnable tokens that absorb high-norm artifacts, producing cleaner attention maps.

**Strengths for diffraction:**
- Produces exceptionally good *dense* per-patch features — useful for spatial downstream
  tasks (peak finding, region segmentation, anomaly detection)
- Clean attention maps (with registers) could reveal which spatial regions the model
  considers important — interpretability for diffraction analysis
- State-of-the-art linear probe performance on natural images
- Strong at learning features that transfer across tasks without fine-tuning

**Weaknesses for diffraction:**
- **Augmentation-dependent.** DINOv2 uses multi-crop augmentation (global crops + local
  crops at different scales) and color jitter. For single-channel diffraction:
  - Color jitter: meaningless
  - Random crop: risky (may break beam-center-relative spatial relationships)
  - Multi-scale crops: may work if local vs global structure is meaningful
- Need to design domain-specific augmentation pipeline — non-trivial and may not
  capture the right invariances
- More complex training setup than MAE (teacher-student, centering, sharpening)

**When to consider:** If downstream tasks are primarily about *feature extraction*
(linear probes, nearest-neighbor retrieval) rather than reconstruction, and if a
reasonable augmentation strategy can be designed for diffraction data.

**Key implementations:**
- HuggingFace: `Dinov2Model`, `Dinov2WithRegistersModel`
- Docs: `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/docs/source/en/model_doc/dinov2_with_registers.md`

---

## Candidate 5: BEiT (BERT Pre-Training of Image Transformers)

**Paper:** Bao et al., "BEiT: BERT Pre-Training of Image Transformers" (Microsoft, 2021)

**Pretext task:** Mask image patches, predict *discrete visual tokens* from a
pre-trained tokenizer (originally DALL-E's VQ-VAE codebook). Combines masked image
modeling with discrete targets.

**Architecture:** Standard ViT encoder. The tokenizer (VQ-VAE) is trained separately
and frozen. During BEiT pre-training, the encoder learns to predict which codebook
token each masked patch corresponds to (cross-entropy loss over codebook vocabulary).

**Strengths for diffraction:**
- Predicts *what kind* of pattern (discrete token), not exact pixel values — more
  semantic than MAE, potentially more robust to noise
- Could train a domain-specific VQ-VAE tokenizer on diffraction data, then use
  BEiT-style pre-training on top — the tokenizer captures diffraction-specific motifs,
  the BEiT encoder learns to reason about them
- Uses relative position embeddings — better generalization to different image sizes
- Outperforms supervised ViT pre-training (83.2% vs 81.8% on ImageNet base-size)

**Weaknesses:**
- Two-stage pipeline: (1) train VQ-VAE tokenizer, (2) train BEiT encoder. Adds
  complexity and the tokenizer quality becomes a bottleneck.
- If you're training a VQ-VAE anyway (Candidate 2), you could just use its
  representations directly — BEiT adds a second training phase
- The discrete token prediction may lose fine-grained intensity information that
  matters for quantitative diffraction analysis

**When to consider:** If VQ-VAE (Candidate 2) shows promising codebook structure but
you want better downstream transfer than the VQ-VAE encoder provides. BEiT + VQ-VAE
is a natural two-stage pipeline.

**Key implementations:**
- HuggingFace: `BeitModel`, `BeitForMaskedImageModeling`, `BeitForImageClassification`
- Docs: `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/docs/source/en/model_doc/beit.md`

---

## Candidate 6: Contrastive Methods (SimCLR / MoCo / BYOL)

**Papers:** Chen et al., "SimCLR" (2020); He et al., "MoCo" (2020); Grill et al.,
"BYOL" (2020)

**Pretext task:** Learn representations that are invariant across different augmented
views of the same image (positive pairs) and different from other images (negative pairs,
except BYOL which uses no negatives).

**Strengths in general:**
- Simple, well-understood training dynamics
- BYOL eliminates the large batch size requirement of SimCLR
- Excellent low-shot performance on natural images
- Clean implementations available (`$DEEPLEARNING_DOC_DIR/lucidrain-repos/byol-pytorch/`,
  `$DEEPLEARNING_DOC_DIR/uvadlc_notebooks/docs/tutorial_notebooks/tutorial17/SimCLR.md`)

**Fundamental problem for diffraction:**
Contrastive methods are *entirely defined* by what augmentations you use. The
representation learns to be invariant to those augmentations. For diffraction data:

| Augmentation | Natural images | Diffraction data |
|-------------|---------------|-----------------|
| Random crop + resize | Core of SimCLR's success | Breaks beam-center-relative geometry |
| Color jitter | Invariance to lighting | Meaningless (single channel) |
| Horizontal flip | Usually harmless | May break lattice symmetry |
| Gaussian blur | Mild invariance | Destroys peak sharpness |
| Rotation | Sometimes used | Possibly valid (detector rotation), domain-dependent |
| Intensity scaling | N/A | Possibly valid (exposure invariance) |

**Verdict:** Deprioritize unless domain-specific augmentations can be rigorously
validated. The risk of learning wrong invariances is high.

---

## Candidate 7: Denoising Diffusion (DDPM / DiT)

**Papers:** Ho et al., "DDPM" (2020); Peebles & Xie, "DiT" (2023)

**Pretext task:** Predict noise added to images at various timesteps. The model learns
to reverse a gradual noising process.

**Strengths for diffraction:**
- Multi-scale representations emerge from different noise levels — low noise captures
  fine detail (peak shape), high noise captures global structure (lattice symmetry)
- Generative: can produce synthetic diffraction patterns for data augmentation
- DiT architecture (transformer-based diffusion) aligns with the ViT backbone plan
- Detailed architecture notes available in
  `$DEEPLEARNING_DOC_DIR/notes/large-image-diffusion-transformers.md`

**Weaknesses:**
- Primarily a generative model, not a representation learner — representations need
  extraction from intermediate layers at specific timesteps
- Much more expensive to train than MAE/I-JEPA (1000 timestep forward/reverse)
- Slow sampling (generation requires many denoising steps)
- Overkill if the goal is embeddings for downstream tasks, not generation

**When to consider:** If synthetic data generation is itself a project goal (augmenting
limited experimental data), or as a future direction after establishing a strong
representation learner.

**Key implementations:**
- `$DEEPLEARNING_DOC_DIR/lucidrain-repos/denoising-diffusion-pytorch/`
- NeMo DiT framework: `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/NeMo/nemo/collections/diffusion/`

---

## Candidate 8: Vanilla VAE

**Paper:** Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)

**Pretext task:** Encode image to continuous latent distribution (mu, sigma), sample,
decode. Loss = reconstruction + KL divergence (regularizes latent space).

**Why it's the weakest candidate:**
- Blurry reconstructions compared to VQ-VAE (the KL regularization trades off against
  reconstruction fidelity)
- For scientific data where quantitative accuracy matters, the reconstruction-
  regularization tradeoff is particularly painful
- The continuous latent space is less interpretable than VQ-VAE's discrete codebook
- No masking or prediction task — the representation learning signal is weaker

**When to consider:** Only as a component of a larger pipeline (e.g., VAE as the latent
space compressor for a DiT, per the latent diffusion paradigm). Not recommended as the
primary SSL approach.

**Key implementations:**
- `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/PyTorch-VAE/` (20+ variants)

---

## Recommended Evaluation Strategy

### Tier 1: Evaluate first (highest expected value for MAXIE)

| Approach | Why | Effort to add |
|----------|-----|--------------|
| **MAE** (current) | Already working. Establish strong baseline with contrast-aware loss and patch size tuning. | Low |
| **VQ-VAE** | Discrete codebook maps to diffraction motifs. Rich diagnostics. Research loop exploration is well-designed for codebook tuning. | Medium |
| **I-JEPA** | No augmentation dependency, semantic representations. Best candidate for learning beyond pixel reconstruction. | Medium-high |

### Tier 2: Consider after Tier 1 results

| Approach | Trigger to explore |
|----------|-------------------|
| **BEiT** | If VQ-VAE produces a good codebook but its encoder underperforms MAE on downstream tasks |
| **DINOv2** | If a domain-specific augmentation strategy can be validated |
| **Diffusion (DiT)** | If synthetic data generation becomes a project goal |

### Tier 3: Deprioritize

| Approach | Why |
|----------|-----|
| **SimCLR / MoCo / BYOL** | Augmentation problem is fundamental for diffraction |
| **Vanilla VAE** | Strictly dominated by VQ-VAE for this domain |

### Evaluation metrics

For fair comparison across candidates, evaluate on:

1. **Reconstruction quality** (where applicable): MSE, PSNR, SSIM on held-out data
2. **Bragg peak preservation**: SNR of peaks in reconstructed vs. original images
3. **Linear probe accuracy**: freeze encoder, train linear classifier on downstream
   task (peak/no-peak classification, crystal system identification)
4. **k-NN retrieval**: embed images, check if nearest neighbors share the same crystal
   system or experimental conditions
5. **Fine-tuning transfer**: full fine-tuning on downstream tasks with limited labels

---

## References to Local Documentation

| Topic | Path |
|-------|------|
| MAE architecture | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/mmsegmentation/configs/mae/README.md` |
| VQ-VAE implementation | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/PyTorch-VAE/models/vq_vae.py` |
| Vector quantization library | `$DEEPLEARNING_DOC_DIR/lucidrain-repos/vector-quantize-pytorch/` |
| I-JEPA docs | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/docs/source/en/model_doc/ijepa.md` |
| DINOv2 with registers | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/docs/source/en/model_doc/dinov2_with_registers.md` |
| BEiT docs | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/transformers/docs/source/en/model_doc/beit.md` |
| SimCLR tutorial | `$DEEPLEARNING_DOC_DIR/uvadlc_notebooks/docs/tutorial_notebooks/tutorial17/SimCLR.md` |
| BYOL implementation | `$DEEPLEARNING_DOC_DIR/lucidrain-repos/byol-pytorch/` |
| Diffusion models | `$DEEPLEARNING_DOC_DIR/lucidrain-repos/denoising-diffusion-pytorch/` |
| DiT / latent diffusion survey | `$DEEPLEARNING_DOC_DIR/notes/large-image-diffusion-transformers.md` |
| VAE for scientific imaging | `$DEEPLEARNING_DOC_DIR/notes/large-image-diffusion-transformers.md` (Section 8) |
| PyTorch-VAE collection | `$DEEPLEARNING_DOC_DIR/pytorch-proj-repos/PyTorch-VAE/` |
| MAXIE current model | `$MAXIE_DIR/maxie/modeling/adapted_mae.py` |
| Research loop brainstorm | `docs/agents/research-loop-brainstorm.md` |
| Frontier strategy | `docs/agents/research-loop-frontier-strategy.md` |
