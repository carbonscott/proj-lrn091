# Architecture Decisions

Date: 2026-02-19

## Goal

Self-supervised representation learning on LCLS X-ray diffraction/scattering
images. Train a general-purpose visual encoder that understands detector
physics, then fine-tune for downstream tasks (peak finding, hit classification,
anomaly detection).


## 1. Model architecture survey

We surveyed several self-supervised and generative approaches:

| Approach | Type | Input handling | Strengths for our data | Weaknesses for our data |
|---|---|---|---|---|
| **MAE** | Masked autoencoder | Fixed-size patches (e.g., 224x224) | Simple, well-understood; validates pipeline fast; learns local texture | Reconstruction objective may focus on low-level features |
| **DINOv2** | Self-distillation (student-teacher) | Fixed-size patches | Learns semantic features; strong transfer; proven at scale | More complex training loop (EMA teacher, multi-crop) |
| **DINOv3** | DINO + registers | Same as DINOv2 | Fixes artifact attention maps; cleaner features | Very recent, less community tooling |
| **Hiera** | Hierarchical MAE | Multi-scale patches | Efficient; good for detection tasks | Less proven for scientific imaging |
| **HIPT** | Hierarchical image pyramid ViT | Tile-level then slide-level | Designed for gigapixel images (pathology) | Over-engineered for our panel sizes |
| **MAR** | Masked autoregressive | Token sequences | Strong generation quality | Generative focus, not representation learning |
| **DiT** | Diffusion transformer | Latent diffusion + ViT | State-of-art generation | Needs pretrained VAE; generation not our goal |
| **VQ-VAE** | Vector-quantized autoencoder | Encoder-decoder with codebook | Discrete representations; good compression | Codebook collapse risk; reconstruction focus |
| **I-JEPA** | Joint-embedding predictive | Fixed-size patches | Predicts in latent space (not pixels); minimal augmentation; efficient | No visual debugging; newer, less proven in scientific domains |
| **V-JEPA** | Video JEPA | 3D spatio-temporal patches | Extends I-JEPA to video; multi-scale masking | Video-focused; overkill for single-frame data |

### Why representation learning over generative modeling

- Our downstream tasks (peak finding, classification) need discriminative
  features, not generated images.
- Generative models (diffusion, VAE) are more complex to train and evaluate
  without ground-truth generation targets.
- Representation learning (MAE, DINO) directly produces embeddings usable for
  fine-tuning.

### Why I-JEPA is a strong candidate

I-JEPA (Image Joint-Embedding Predictive Architecture) predicts representations
of masked regions from visible regions in **latent space**, not pixel space. This
has several advantages for our X-ray data:

- **No pixel reconstruction bias**: MAE's pixel-level objective may overfit to
  low-level detector artifacts (gain variation, hot pixels, panel gaps). I-JEPA
  learns in latent space, potentially capturing more physics-relevant features.
- **Minimal augmentation**: Only needs spatial masking — no color jitter, Gaussian
  blur, or multi-crop. This matters because aggressive augmentations may not be
  physically meaningful for diffraction patterns.
- **Multi-block prediction**: Predicts multiple small target blocks from one large
  context block. Naturally maps to learning spatial relationships between Bragg
  peaks, scattering rings, and background.
- **Efficient**: Single forward pass through encoder per image. No decoder
  overhead (MAE) or multi-view pipeline (DINOv2).

Concerns:
- **No visual debugging**: Cannot inspect reconstructions to catch pipeline bugs,
  unlike MAE where you can literally see bad patches.
- **Less community adoption**: MAE and DINO have been tried in cryo-EM, pathology,
  etc. I-JEPA is newer for domain adaptation.

### Why MAE first, then I-JEPA, then DINOv2

- **MAE first**: Simpler training loop (single network, no EMA). Validates the
  entire data pipeline end-to-end. If patches look wrong, reconstruction loss
  makes it immediately visible.
- **I-JEPA next**: Same data pipeline, swap objective. Tests whether latent
  prediction works better than pixel reconstruction for our data. Simpler than
  DINOv2 (no multi-crop augmentation), but introduces EMA target encoder.
- **DINOv2 last**: Produces strongest, most transferable features. Student-teacher
  distillation with multi-crop augmentation captures both local and global
  structure. Most complex training loop but best expected downstream performance.


## 2. Input handling decision

### The challenge: extreme aspect ratios

Our 4 detector geometries have very non-square raw shapes:

| Detector | Raw shape (H x W) | Aspect ratio | Experiments |
|---|---|---|---|
| Jungfrau 4M | 4096 x 1024 | 4:1 | cxi101235425, cxil1005322, cxil1015922, cxilw5019 |
| ePix10k2M | 5632 x 384 | ~15:1 | mfx100903824, mfxp22421, mfxx49820, prjcwang31-cheetah |
| Jungfrau 16M | 16384 x 1024 | 16:1 | mfx101211025 |
| Assembled | 1920 x 1920 | 1:1 | prjcwang31-psocake |

Standard ViTs expect square inputs (224x224, 384x384). Naively resizing a
16384x1024 image to 224x224 would destroy spatial structure.

### Three approaches considered

1. **Panel-wise patching** (chosen): Split the stacked-panel image along the H
   dimension into physical detector panels, then extract fixed-size patches
   (e.g., 256x256) from each panel.

2. **NaFlex variable resolution**: Use NaFlexViT (available in `timm`) to handle
   variable aspect ratios natively by packing different numbers of patches per
   image. Elegant but adds complexity and couples the model architecture to our
   detector geometry.

3. **Fixed resize**: Resize all images to a standard size. Simple but destroys
   the physical pixel scale that matters for diffraction analysis.

### Why panel-wise patching

- **Preserves physics**: Each panel is a physically contiguous detector module.
  Patches within a panel correspond to real spatial neighborhoods.
- **Uniform patch sizes**: All patches are the same size regardless of detector
  type. Standard ViT works without modification.
- **Natural data augmentation**: Different panels from the same frame provide
  multiple training samples with the same experimental conditions but different
  scattering angles.
- **Scalable**: Works identically whether we have 8 panels (Jungfrau 4M) or 32
  (Jungfrau 16M).

### Panel layout per detector type

```
Jungfrau 4M:  4096 x 1024  →  8 panels of 512 x 1024
ePix10k2M:    5632 x 384   → 16 panels of 352 x 384
Jungfrau 16M: 16384 x 1024 → 32 panels of 512 x 1024
Assembled:    1920 x 1920  →  1 panel  of 1920 x 1920 (direct patching)
```


## 3. Data layer decision

### Short-term: Manifest + PyTorch DataLoader

A JSON manifest lists all available HDF5 files with their shapes, detector
types, and panel configurations. A PyTorch `Dataset` reads the manifest at init
time, builds an index of `(file, frame, panel)` triples, and lazily loads data
via h5py during training.

This gets training running fast with minimal infrastructure.

### Long-term: Tiled broker

Inspired by the `proj-vdp-generic-broker` manifest pattern at LCLS. A Tiled
data broker would provide a unified API for accessing data across experiments,
handle caching, and support additional formats (ZARR, etc.).

### Future extensibility

- ZARR support for cloud-friendly storage
- New experiments added by updating the manifest
- Multi-resolution patching (different patch sizes per training stage)


## 4. Training roadmap

### Phase 1: ViT-Base + MAE + manifest DataLoader
- Validate end-to-end training pipeline
- Patch size 256x256, single channel (float32 intensity)
- Per-frame normalization (subtract mean, divide by std)
- Basic augmentations (90-degree rotations, flips)

### Phase 1.5: I-JEPA training loop
- Same data pipeline, swap training objective
- Encoder + lightweight predictor + EMA target encoder
- Multi-block masking (large context → predict small targets)
- Compare representation quality against MAE via linear probing

### Phase 2: DINOv2 training loop
- Same data pipeline, swap training objective
- Add multi-crop augmentation (global + local crops)
- Student-teacher with EMA

### Phase 3: Scale up
- ViT-Large → ViT-Huge
- Add Tiled broker for data access
- Multi-node distributed training


## 5. Key references

- **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners" (2022).
  Reference repo at `externals/reference-repos/mae/`.
- **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features" (2024).
  Reference repo at `externals/reference-repos/dinov2/`.
- **NaFlexViT**: Available in `timm` library. Variable-resolution ViT that packs
  different numbers of patches per image. Future option if panel-wise patching
  proves limiting.
- **DiT/diffusion notes**: See `docs/large-image-diffusion-transformers.md` in
  `externals/deeplearning-docs/` for VAE considerations if we ever need
  generation.
- **VDP broker**: `proj-vdp-generic-broker` for the Tiled manifest pattern we
  plan to adopt long-term.
- **DINOv2 `XChannelsDecoder`**: Pattern for handling multi-channel inputs in
  the DINOv2 codebase (relevant if we add multi-channel support later).
- **HIPT `dataset_h5.py`**: Reference for HDF5-backed PyTorch Dataset
  implementation.
- **I-JEPA**: Assran et al., "Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture" (2023). Reference repo at
  `externals/reference-repos/ijepa/`.
- **V-JEPA**: Bardes et al., "Revisiting Feature Prediction for Learning Visual
  Representations from Video" (2024). Reference repo at
  `externals/reference-repos/jepa/`. Video variant — architecture patterns are
  relevant but the temporal aspect is not needed for our single-frame data.
