# Research Loop Brainstorm: Where It Beats Bayesian Optimization

Date: 2026-03-19
Status: scratch notes / brainstorming

## Core Principle

Bayesian Optimization works well on **continuous numerical axes** — learning rate,
weight decay, momentum, etc. A research-loop with an agent adds value where decisions
are **structural, discrete, or require interpreting qualitative/diagnostic outputs**.

The agent's advantage: it can *look at what's happening* and reason about *why*,
rather than treating the system as a black box with scalar output.

---

## Candidate 1: Codebook Design (VQ-VAE component)

**Why BO struggles here:**
- Codebook size is discrete and each value changes training dynamics qualitatively
- Codebook health requires interpreting utilization distributions, not a single number
- Failure modes (codebook collapse, dead codes) need diagnosis, not just detection

**What the agent would do:**
1. Run short training (few epochs, small node count)
2. Inspect codebook utilization histogram — are codes evenly used or collapsing?
3. Count dead codes (codes never selected)
4. Decide: increase codebook size? Change commitment loss weight? Switch reset strategy?
5. Repeat

**Decision axes:**
- Codebook size: 256, 512, 1024, 2048, 4096
- Commitment loss weight (β): interacts with codebook utilization non-trivially
- EMA decay rate for codebook update
- Dead code reset strategy: EMA, random reinit from data, threshold-based pruning
- Codebook dimensionality (embedding dim)

**Diagnostic signals the agent reads:**
- Codebook utilization histogram (should be roughly uniform)
- Perplexity of codebook usage (higher = more codes active)
- Dead code fraction over training steps (should decrease or stabilize)
- Reconstruction loss trajectory
- Codebook embedding space visualization (PCA/t-SNE) if available

---

## Candidate 2: Contrast-Aware Loss Function Design

**Why BO struggles here:**
- Not a parameter sweep — it's structural design choices
- Window size for local statistics, weighting scheme, which statistics to compute
- Need to visually verify Bragg peak preservation in reconstructions

**What the agent would do:**
1. Train with a loss variant
2. Inspect reconstruction error maps — are Bragg peaks reconstructed accurately?
3. Compare peak SNR in input vs reconstruction
4. Decide: adjust window size? Change weighting between peak regions and diffuse?
5. Repeat

**Design axes:**
- Local statistics window size: 5×5, 9×9, 15×15, 21×21
- Weighting function: linear, exponential, threshold-based
- What local statistic: contrast (max-min)/mean, local SNR, local variance
- Balance between global MSE and local contrast-aware term
- Whether to apply per-patch normalization before or after loss computation

---

## Candidate 3: Patch Size and Tokenization Strategy

**Why BO struggles here:**
- Each patch size fundamentally changes what the embedding layer must learn
- Tradeoffs aren't smooth — 16×16 sees fine detail but has long sequences,
  128×128 captures more context but may blur Bragg peaks
- Evaluation requires checking downstream task quality, not just reconstruction loss

**What the agent would do:**
1. Train with a patch size configuration
2. Evaluate reconstruction quality (especially at Bragg peak locations)
3. Run a lightweight downstream probe (e.g., linear classifier on frozen embeddings)
4. Check sequence length feasibility (memory, attention cost)
5. Decide: is finer patching worth the compute? Is the embedding capturing peaks?

**Decision axes:**
- Patch size: 16×16, 32×32, 64×64, 128×128
- Overlap between patches (0%, 25%, 50%)
- Embedding projection: linear vs small CNN
- Position encoding type for different sequence lengths

---

## Candidate 4: Pre-training → Downstream Transfer Decision Point

**Why BO struggles here:**
- No single scalar tells you "stop pre-training and start fine-tuning"
- Requires comparing trends across multiple metrics simultaneously
- Involves a judgment call about diminishing returns

**What the agent would do:**
1. At each checkpoint (every 200 iterations), run a quick downstream probe
2. Track: reconstruction loss, codebook perplexity, downstream probe accuracy
3. Detect when downstream probe accuracy plateaus while reconstruction still improves
   (signal that representation is saturating for the task)
4. Decide: continue pre-training? Scale up model? Switch to fine-tuning?

---

## Batch Mode + Git Worktrees (from updated research-loop skill)

The research-loop skill now supports **batch mode (K > 1)** using git worktrees to run
K experiments in parallel. This is a natural fit for Frontier's Flux ensemble pattern.

### How batch mode works

```
1. READ       insights.md
2. HYPOTHESIZE K times (diversify across change_type categories)
3. SETUP      K worktrees from current research branch HEAD
4. MODIFY     each worktree independently
5. COMMIT     in each worktree
6. RUN        K experiments in parallel
7. WAIT       for all K to complete
8. EVALUATE   each independently
9. LOG        all K to SQLite with shared batch_id
10. RECONCILE merge best keep into research branch, clean up worktrees
11. CHECK     if total experiments >= N → outer loop, else next batch
```

Key design rules:
- **Diversify hypotheses**: spread across categories (codebook size, loss design,
  reset strategy), not grid search on one axis
- **Winner takes main**: only the best keep gets merged (fast-forward). Avoids
  untested combinations from merging multiple changes.
- **keep-deferred**: experiments that beat baseline but weren't the batch winner.
  The outer loop can suggest combining promising deferred directions in future batches.
- **Plateau detection**: counts at batch level (batch with 0 keeps = 1 batch discard)

### Integration with Flux on Frontier

The batch mode maps directly to Flux inside a Slurm allocation:

```
research-loop/
├── experiments.db
├── insights.md
├── protocol.md
└── worktrees/
    ├── exp-1-1/   # codebook_size=512, β=0.25
    ├── exp-1-2/   # codebook_size=1024, β=0.25
    ├── exp-1-3/   # codebook_size=512, β=0.5, EMA reset
    └── exp-1-4/   # codebook_size=1024, threshold reset
```

**Workflow:**

1. Agent creates K worktrees with diversified codebook configs
2. Submit one Slurm allocation (e.g., 8 nodes, 2 hours)
3. Inside the allocation, Flux runs K training jobs in parallel:
   ```bash
   srun -N $SLURM_NNODES -n $SLURM_NNODES -c 56 --gpus-per-node=8 flux start \
       "for i in \$(seq 1 $K); do
           flux submit -N 2 -n 16 -c 7 --gpus-per-task=1 \
               -o gpu-affinity=per-task \
               --output=research-loop/worktrees/exp-\${BATCH_ID}-\${i}/run.log \
               bash -c '
                   export ROCR_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES
                   unset CUDA_VISIBLE_DEVICES
                   cd research-loop/worktrees/exp-'\$BATCH_ID'-'\$i'
                   python3 train_maxie.py --config config.yaml
               '
       done;
       flux queue drain;"
   ```
4. Agent evaluates all K results, reconciles, logs to SQLite
5. Outer loop distills after N total experiments

**Why this combination works:**
- One queue wait (the Slurm allocation), not K separate waits
- Flux auto-schedules: if K > available nodes, it queues remaining jobs internally
- Worktrees give each experiment isolated code state (different configs, potentially
  different code changes) without branch conflicts
- The reconcile step (merge winner, log deferred) feeds cleanly into the outer loop

### Example: Codebook Exploration Batch

A concrete batch for Candidate 1 (codebook design):

| Worktree | change_type | Description |
|---|---|---|
| exp-1-1 | codebook_size | size=512, β=0.25 (baseline params, smaller codebook) |
| exp-1-2 | codebook_size | size=2048, β=0.25 (larger codebook, same β) |
| exp-1-3 | commitment_loss | size=1024, β=0.5 (default size, higher commitment) |
| exp-1-4 | reset_strategy | size=1024, β=0.25, threshold reset at 10 uses |

Each runs 3 epochs on 2 nodes. Agent reads:
- Codebook utilization histogram from each
- Dead code fraction
- Reconstruction loss

Winner gets merged. If exp-1-2 (large codebook) and exp-1-4 (threshold reset) both
beat baseline, the winner merges and the other is `keep-deferred`. Outer loop might
suggest: "try large codebook + threshold reset together in next batch."

---

## Practical Considerations for Running on Frontier

- These loops should use **small allocations** (2-8 nodes, 1-2 hours)
- Each iteration is a short training run (few epochs), not a full 512-node campaign
- Results analysis happens between jobs (on login node or in the batch script)
- The agent needs structured log output to parse (JSON/CSV metrics, not just stdout)
- Checkpoint diagnostic scripts should be part of the training codebase
- Batch mode with K=4 on 8 nodes = 2 nodes per experiment, fits in one allocation
- Flux handles scheduling if K > available node groups

## What Needs to Exist First

- [ ] Training script that outputs structured metrics (JSON lines or CSV)
- [ ] Codebook diagnostic script (utilization histogram, dead codes, perplexity)
- [ ] Reconstruction visualization script (input vs output vs error, with peak overlay)
- [ ] Lightweight downstream probe script (linear eval on frozen embeddings)
- [ ] Config system (Hydra) that makes it easy to swap these design choices
- [ ] Wrapper script that bridges worktree layout → Flux submission
- [ ] Evaluation script that parses all K run logs and produces a comparison table
