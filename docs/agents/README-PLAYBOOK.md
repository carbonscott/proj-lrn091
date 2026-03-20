# MAXIE Training Campaign Playbook

Entry point for the complete operational documentation for the MAXIE foundation model
project — self-supervised pre-training on X-ray crystallography diffraction images,
running on OLCF Frontier (ALCC allocation lrn091, 121,680 GPU-hours).

**Start here.** This document tells you where to find everything else.

---

## Project Summary

MAXIE (Masked X-Ray Image AutoEncoder) is a ViT-based foundation model for X-ray
diffraction data analysis. The ALCC project "A Codebook Language and Digital Twin
Framework for Diffraction Data Analysis and Accelerator Operations" allocates compute
on Frontier (up to 512 nodes / 4096 AMD MI250X GCDs) for:

1. A scaling law study across model sizes (ViT-Base/Large/Huge) and data fractions
2. Comparison of three self-supervised learning approaches (MAE, VQ-VAE, I-JEPA)
3. 10 full pre-training campaigns at 512 nodes
4. A 400-run downstream evaluation campaign

---

## Document Map

| Document | Purpose | Lines |
|----------|---------|-------|
| [ssl-candidates-for-maxie.md](ssl-candidates-for-maxie.md) | Which SSL method to use — 8 paradigms evaluated, 3 recommended | ~400 |
| [training-playbook-for-maxie.md](training-playbook-for-maxie.md) | How to train — optimizer, LR, loss, batch scaling, failure modes | ~600 |
| [data-pipeline-for-maxie.md](data-pipeline-for-maxie.md) | What to feed — data formats, normalization, Lustre I/O, patches | ~810 |
| [monitoring-protocol-for-maxie.md](monitoring-protocol-for-maxie.md) | What to watch — metrics, JSONL logging, decision trees | ~980 |
| [evaluation-suite-for-maxie.md](evaluation-suite-for-maxie.md) | How to judge — linear probe, k-NN, few-shot, 400-run campaign | ~1060 |
| [scaling-laws-design-for-maxie.md](scaling-laws-design-for-maxie.md) | How to allocate compute — 36-experiment grid, Chinchilla fitting | ~1210 |
| [research-loop-frontier-strategy.md](research-loop-frontier-strategy.md) | Where to run — Frontier, Slurm, Flux, sbcast, RCCL tuning | ~400 |
| [research-loop-brainstorm.md](research-loop-brainstorm.md) | How to explore — agent-driven loops, batch mode, worktrees | ~225 |

**Other files in this directory** (not part of the playbook):
- `progress.md` — Running log of work done by agents
- `notes.md` — Technical notes and caveats
- `update-claude-md-for-frontier.md` — Notes on CLAUDE.md configuration

---

## Reading Guides

### "I'm starting a pre-training run"

1. **[training-playbook-for-maxie.md](training-playbook-for-maxie.md)** — Sections 1-6: optimizer, LR schedule, loss, batch scaling, mixed precision
2. **[data-pipeline-for-maxie.md](data-pipeline-for-maxie.md)** — Sections 1-3, 6: current pipeline, preprocessing, data loading at scale
3. **[research-loop-frontier-strategy.md](research-loop-frontier-strategy.md)** — Section 2-3: OLCF best practices, batch script template
4. **[monitoring-protocol-for-maxie.md](monitoring-protocol-for-maxie.md)** — Sections 2-3: what to log per step and per epoch

### "I'm designing experiments"

1. **[scaling-laws-design-for-maxie.md](scaling-laws-design-for-maxie.md)** — Full document: experiment grid, compute budget, Hydra configs
2. **[evaluation-suite-for-maxie.md](evaluation-suite-for-maxie.md)** — Sections 2-7: downstream tasks, evaluation protocols, 400-run campaign
3. **[research-loop-brainstorm.md](research-loop-brainstorm.md)** — Candidates 1-4: what structural decisions to explore

### "I'm choosing an SSL method"

1. **[ssl-candidates-for-maxie.md](ssl-candidates-for-maxie.md)** — Full document: 8 candidates ranked by fit for diffraction data
2. **[training-playbook-for-maxie.md](training-playbook-for-maxie.md)** — Section 10: how the training recipe changes per SSL method
3. **[evaluation-suite-for-maxie.md](evaluation-suite-for-maxie.md)** — Section 7: how to compare methods in the 400-run campaign

### "I'm debugging a failed run"

1. **[monitoring-protocol-for-maxie.md](monitoring-protocol-for-maxie.md)** — Section 7: decision trees (if X then Y)
2. **[training-playbook-for-maxie.md](training-playbook-for-maxie.md)** — Section 8: failure modes and mitigations
3. **[monitoring-protocol-for-maxie.md](monitoring-protocol-for-maxie.md)** — Appendix B: metric threshold quick-reference (green/yellow/red)

### "I'm an agent running a research loop"

1. **[research-loop-brainstorm.md](research-loop-brainstorm.md)** — Batch mode design, worktree layout, Flux integration
2. **[monitoring-protocol-for-maxie.md](monitoring-protocol-for-maxie.md)** — Section 6: JSONL format spec (what you parse), Section 9: research loop integration
3. **[scaling-laws-design-for-maxie.md](scaling-laws-design-for-maxie.md)** — Section 8-9: Hydra multirun configs, Flux job scripts

---

## Critical Action Items

These are the highest-impact findings aggregated across all documents. Address these
before starting any pre-training campaign.

### P0: Fix before any run

| Finding | Source | Action |
|---------|--------|--------|
| `norm_pix_loss` is `false` | training-playbook Section 3.1, data-pipeline Section 4 | Set to `true` — normalizes target per patch, improves representation quality |
| `Norm` transform is commented out | data-pipeline Section 1 | Investigate and enable if appropriate |
| Warmup is 5 iterations | training-playbook Section 4.2 | Increase to 5-10% of total steps (scaling studies) or 40 epochs (full campaign) |
| LR not scaled by batch size | training-playbook Section 4.1 | Apply `lr = 1.5e-4 * batch_size / 256` |

### P1: Implement before scaling study

| Finding | Source | Action |
|---------|--------|--------|
| `data_fraction` parameter doesn't exist | scaling-laws Section 3 | Implement in dataset class for data scaling experiments |
| `max_steps` parameter needed | scaling-laws Section 3 | Implement step-based training termination |
| JSONL structured logger needed | monitoring-protocol Section 6 | Create `maxie/utils/json_logger.py` for agent-parseable output |
| Benchmark runs needed first | scaling-laws Section 4, Appendix A | Run B01-B03 to calibrate throughput estimates before the full grid |
| bfloat16 should be tested | training-playbook Section 6.1 | Test bf16 vs fp16 on MI250X; bf16 eliminates GradScaler |

### P2: Implement before evaluation campaign

| Finding | Source | Action |
|---------|--------|--------|
| Label curation is the critical path | evaluation-suite Section 2 | Curate crystal system labels and anomaly labels from experiment metadata |
| All downstream eval code is new | evaluation-suite Section 9 | Write: linear probe, k-NN, few-shot, peak detection, reconstruction metrics scripts |
| Feature caching needed | evaluation-suite Section 9 | Extract features once per checkpoint, reuse across eval protocols |
| Gradient clipping may be unnecessary | training-playbook Section 2.3 | MAE paper uses none; monitor grad norms and consider removing |

---

## Cross-Document Dependencies

```
ssl-candidates-for-maxie.md
    |
    v
training-playbook-for-maxie.md  <------>  data-pipeline-for-maxie.md
    |                                          |
    v                                          v
monitoring-protocol-for-maxie.md          (data loading at scale)
    |                    |
    v                    v
evaluation-suite     scaling-laws-design
-for-maxie.md        -for-maxie.md
    |                    |
    +--------------------+
    |
    v
research-loop-frontier-strategy.md  <-->  research-loop-brainstorm.md
```

**Reading the diagram:**
- Arrows point from "read first" to "read after"
- `ssl-candidates` informs `training-playbook` (recipe varies by method)
- `training-playbook` and `data-pipeline` inform each other (augmentation, normalization)
- `monitoring-protocol` feeds both `evaluation-suite` (same metrics) and `scaling-laws` (what to track)
- `research-loop-*` docs tie everything together for automated exploration

---

## Execution Phases

From the training playbook, the project proceeds in four phases:

### Phase 1: Fix the baseline (2-8 nodes, days)
Apply P0 action items. Validate on small runs. No new code beyond config changes.

### Phase 2: Scaling law study (8-32 nodes, 1-2 weeks)
Run the 36-experiment grid from `scaling-laws-design`. Uses ~7,000 GPU-hours (5.8% of
budget). Fit Chinchilla curves. Determine optimal model size + data size.

### Phase 3: SSL method comparison (8 nodes, 1 week)
Run Tier 1 candidates (MAE, VQ-VAE, I-JEPA) head-to-head. Evaluate with linear probe
and k-NN from `evaluation-suite`. Select winning method.

### Phase 4: Full pre-training campaign (512 nodes, weeks)
Apply winning method + optimized recipe. 10 runs at full scale. Run 400-evaluation
campaign afterward.

---

## Environment Variables

These are defined in the project's `CLAUDE.md` and used throughout the playbook:

```bash
export MAXIE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/maxie
export DEEPLEARNING_DOC_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deeplearning-docs
export XTAL_DATA_ASSEMBLED=/lustre/orion/lrn091/proj-shared/data
export PEAKNET_DIR=/lustre/orion/lrn091/proj-shared/cwang31/deps/peaknet
export AGENT_NOTES_DIR=/lustre/orion/lrn091/proj-shared/cwang31/proj-lrn091/docs/agents
```

---

## Dates and Versioning

All playbook documents were initially created on 2026-03-19. They reference the MAXIE
codebase as of that date. If the codebase changes significantly (e.g., new dataset
classes, new training script), the relevant sections should be updated.

| Document | Created |
|----------|---------|
| ssl-candidates-for-maxie.md | 2026-03-19 |
| training-playbook-for-maxie.md | 2026-03-19 |
| data-pipeline-for-maxie.md | 2026-03-19 |
| monitoring-protocol-for-maxie.md | 2026-03-19 |
| evaluation-suite-for-maxie.md | 2026-03-19 |
| scaling-laws-design-for-maxie.md | 2026-03-19 |
| research-loop-frontier-strategy.md | 2026-03-19 |
| research-loop-brainstorm.md | 2026-03-19 |
| **README-PLAYBOOK.md** | 2026-03-20 |
