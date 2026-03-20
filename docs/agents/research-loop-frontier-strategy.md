# Research Loop Strategy for MAXIE on Frontier

Date: 2026-03-19
Context: ALCC project "A Codebook Language and Digital Twin Framework for Diffraction
Data Analysis and Accelerator Operations" — Group 1 (MAXIE foundation model training).

## 1. Research Loop Applicability Assessment

### What the research-loop pattern offers

The research-loop skill sets up autonomous cycles of expansion (run experiments) and
compression (analyze results, decide next config). It fits well for:

- ML training with parameter sweeps
- Systematic experiment exploration
- Configurations that learn from prior results

### Where it aligns with MAXIE (Group 1)

| Task | Fit | Why |
|------|-----|-----|
| Scaling laws study (ViT-Base → Huge, dataset size) | Strong | Classic parameter sweep: vary model/data size, track reconstruction quality |
| Loss function / architecture exploration | Strong | Iterative: try contrast-aware loss variants, evaluate downstream |
| Downstream head evaluation (400 runs) | Strong | Many small independent runs, need automated collection |
| Pre-training campaigns (512 nodes) | Weak | Too expensive per iteration for loose exploration |

### Where it breaks down on Frontier

1. **Job queue latency** — 512-node jobs can wait hours in queue. An autonomous agent
   loop would mostly be idle, and Claude Code sessions aren't designed to persist
   across days of wall-clock time.

2. **Cost per iteration** — Each pre-training run is ~12,168 GPU-hours (121,680 total
   for 10 runs). Decisions at this scale need to be deliberate, not automated.

3. **Hydra already manages configs** — The proposal uses Hydra-formatted config
   tracking. An agent loop would overlap with that infrastructure.

4. **Session persistence** — Claude Code runs interactively. There's no built-in way to
   keep an agent running across multiple Slurm jobs spanning days.

### Recommended approach: human-in-the-loop with automation support

Instead of a fully autonomous research loop, build infrastructure that makes a
**human-driven research cycle fast**:

- **Sweep config generators** — Hydra multirun configs for scaling law experiments
- **Analysis scripts** — Auto-parse training logs, plot loss/codebook utilization/
  reconstruction metrics across runs
- **Decision dashboards** — Summarize completed runs so you can quickly pick next configs
- **Job management scripts** — Submit, monitor, and chain dependent Slurm jobs

### Parsl as a practical alternative

Parsl is natively supported on Frontier (`module load parsl/2024.12.2`) and can
orchestrate the evaluation campaign (400 downstream runs):

- Submits jobs via `SlurmProvider` with `SrunLauncher`
- Tracks completion, collects results
- Runs from a login node, survives across job completions
- Good fit for the many-small-jobs pattern of downstream evaluation

This is closer to what a "research loop" would actually look like on Frontier: Parsl
managing the sweep, with analysis scripts between rounds, and the researcher making
high-level decisions.

---

## 2. OLCF Best Practices for MAXIE Training

Source: OLCF official docs (`olcf-user-docs` repository), specifically
`software/analytics/pytorch_frontier.rst` and `software/python/sbcast_conda.rst`.

### 2.1 Environment: sbcast to NVMe (critical at scale)

At 512 nodes, Python environment initialization from Orion (Lustre) is very slow. The
docs demonstrate NVMe is dramatically faster (2s vs 51s vs 4min at just 8 nodes — the
gap widens at scale).

**Setup once (login node):**
```bash
conda install conda-pack -c conda-forge
conda pack --format tar.gz --n-threads -1 \
    --prefix /path/to/maxie_env \
    --output ./maxie_env.tar.gz
```

**In every batch script:**
```bash
#SBATCH -C nvme

# Broadcast to NVMe on every node
sbcast -pf ./maxie_env.tar.gz /mnt/bb/${USER}/maxie_env.tar.gz
if [ ! "$?" == "0" ]; then
    echo "SBCAST failed!"
    exit 1
fi

# Untar (1 task per node, use all cores for decompression)
srun -N ${SLURM_NNODES} --ntasks-per-node 1 mkdir /mnt/bb/${USER}/maxie_env
srun -N ${SLURM_NNODES} --ntasks-per-node 1 -c56 \
    tar --use-compress-program=pigz -xf /mnt/bb/${USER}/maxie_env.tar.gz \
    -C /mnt/bb/${USER}/maxie_env

# Activate and unpack
conda activate /mnt/bb/${USER}/maxie_env
srun -N ${SLURM_NNODES} --ntasks-per-node 1 conda-unpack
```

Performance ranking: **NVMe > Orion (Lustre) >> NFS**

### 2.2 Use srun, NOT torchrun

The OLCF docs explicitly warn that `torchrun` causes **order-of-magnitude slowdowns**
on Frontier due to NUMA domain mapping issues. The recommended launch pattern:

```bash
srun -N512 -n4096 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python3 -W ignore -u train_maxie.py \
    --master_addr=$MASTER_ADDR --master_port=3442
```

This maps 8 tasks per node (one per GCD), 7 CPU cores each, with proper NUMA affinity
via `--gpu-bind=closest`.

**Never nest** `torchrun` inside `srun` — the two task managers will clash.

### 2.3 RCCL Environment Variables (required at scale)

These are recommended by HPE and AMD for best performance on Frontier:

```bash
# Required — avoid deadlock with libfabric memory registration
export FI_MR_CACHE_MONITOR=kdreg2

# Network stack tuning
export FI_CXI_DEFAULT_CQ_SIZE=131072   # Additional space for message completions
export FI_CXI_DEFAULT_TX_SIZE=2048      # Additional space for pending outgoing messages
export FI_CXI_RX_MATCH_MODE=hybrid     # Allow transition to software mode if needed

# RCCL/NCCL tuning
export NCCL_NET_GDR_LEVEL=3            # Improves performance (remove if hang/crash)
export NCCL_CROSS_NIC=1                # Improves performance on large systems
export NCCL_SOCKET_IFNAME=hsn0         # Use high-speed network for coordination
```

### 2.4 Alternative Rendezvous Protocol (for RCCL-heavy jobs)

MAXIE training is AllReduce-heavy (gradient sync across 4096 GCDs). The alternative
rendezvous protocol improves RCCL performance at large scales:

```bash
#SBATCH --network=disable_rdzv_get
export FI_CXI_RDZV_PROTO=alt_read
```

Note: This may negatively impact MPI performance, so best for jobs that primarily use
RCCL for communication (which MAXIE DDP training is).

### 2.5 Master Address and Port

```bash
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0
```

### 2.6 MIOpen Cache (bypass disk I/O errors)

```bash
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}
```

### 2.7 AWS-OFI-RCCL Plugin

The `aws-ofi-rccl` plugin enables libfabric as a network provider for RCCL. Worth
building and testing for MAXIE's AllReduce-heavy gradient synchronization at 4096 GCDs.

Build instructions are in the OLCF PyTorch guide. After building:
```bash
export LD_LIBRARY_PATH=${PLUGIN_PATH}/lib/:${LD_LIBRARY_PATH}
```

### 2.8 Flash Attention (ROCm fork)

For scaling MAXIE to larger ViT variants, the ROCm fork of flash-attention is available:
```bash
git clone https://github.com/ROCm/flash-attention
cd flash-attention/
git checkout v2.7.4-cktile
git submodule init && git submodule update
python3 setup.py bdist_wheel
pip install dist/*.whl
```

### 2.9 Proxy for Compute Nodes

Compute nodes are isolated from the internet. If needed (e.g., downloading checkpoints):
```bash
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
```

### 2.10 Import Order Matters

When using `mpi4py` with PyTorch, import `mpi4py` **before** `torch` to avoid NUMA
domain errors (`MPICH ERROR: Unable to use a NIC_POLICY of 'NUMA'`).

---

## 3. Reference Batch Script Template for MAXIE Pre-training

```bash
#!/bin/bash
#SBATCH -A lrn091
#SBATCH -J maxie_pretrain
#SBATCH -o logs/maxie-%j.o
#SBATCH -e logs/maxie-%j.e
#SBATCH -t 05:00:00
#SBATCH -p batch
#SBATCH -N 512
#SBATCH -C nvme
#SBATCH --network=disable_rdzv_get

# --- Modules ---
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

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
conda activate /mnt/bb/${USER}/maxie_env
srun -N ${SLURM_NNODES} --ntasks-per-node 1 conda-unpack

# --- Network / RCCL tuning ---
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_RDZV_PROTO=alt_read
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1

# --- MIOpen cache ---
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# --- Launch (8 tasks/node = 4096 total, 7 cores each, 1 GPU each) ---
srun --unbuffered -l \
    -N 512 -n 4096 -c7 \
    --ntasks-per-node=8 --gpus-per-node=8 \
    --gpus-per-task=1 --gpu-bind=closest \
    python3 -W ignore -u train_maxie.py \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT
```

---

## 4. Ensemble Jobs: Running Many Small Jobs in a Big Allocation

Source: `systems/frontier_user_guide.rst`, section "Ensemble Jobs".

A common HPC pattern: instead of submitting hundreds of individual `sbatch` jobs (each
waiting in queue), grab one large allocation and run many jobs inside it. Frontier
supports three approaches, in order of preference.

### 4.1 Single srun with wrapper (best for single-GCD jobs)

If each job fits on 1 GCD, launch all of them in one `srun` and use a wrapper script
to vary parameters per rank. This is the fastest, most reliable approach and scales to
the entirety of Frontier.

```bash
srun -N $SLURM_NNODES -n $((SLURM_NNODES*8)) -c 7 \
    --gpus-per-task=1 --gpu-bind=closest ./wrapper.sh
```

Where `wrapper.sh` uses `$SLURM_PROCID` or `$PMI_RANK` to select different configs
per rank.

### 4.2 Flux inside Slurm (recommended for multi-node or multi-GCD jobs)

**Flux** is a lightweight scheduler that runs inside your Slurm allocation. It creates
a private job queue that only you can submit to. This is the recommended approach for
ensemble runs that don't fit on a single GCD.

**Why Flux over background srun:**

| | Background `srun &` | Flux |
|---|---|---|
| Scales beyond 100 jobs | No — can overload Slurm controller | Yes — manages steps locally |
| Auto load-balancing | No — must `wait` for full batch | Yes — starts next job as node frees |
| Reliability | Unreliable at scale | Tested at 500 nodes |
| Overhead | Minimal per-job | ~2 min to submit 500 jobs |

**Example: 400 evaluation runs on 50 nodes**

```bash
#!/bin/bash
#SBATCH -A lrn091
#SBATCH -J maxie_eval_sweep
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 50

module load rocm/6.4.1
module load hwloc/2.9.1-gpu   # Flux requires GPU-enabled hwloc
module load flux

# Launch Flux scheduler across all nodes
srun -N $SLURM_NNODES -n $SLURM_NNODES -c 56 --gpus-per-node=8 flux start \
    "flux resource list;

    # Submit 400 evaluation runs, each using 1 node
    for config_id in \$(seq 1 400); do
        flux submit -N 1 -n 8 -c 7 --gpus-per-task=1 \
            -o gpu-affinity=per-task \
            --output=logs/eval_\${config_id}.log \
            bash -c '
                export ROCR_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES
                unset CUDA_VISIBLE_DEVICES
                python3 eval_maxie_head.py --config-id '\$config_id'
            '
    done;

    flux jobs -a;
    flux queue drain;"
```

Flux will run up to 50 jobs concurrently (one per node) and automatically start the
next job as each node finishes. No manual batching or `wait` needed.

**Flux GPU binding caveat:** The `--gpus-per-task=1` flag in Flux sets
`CUDA_VISIBLE_DEVICES` but may not correctly restrict GPU visibility on Frontier. The
workaround is to copy `CUDA_VISIBLE_DEVICES` to `ROCR_VISIBLE_DEVICES` and then unset
`CUDA_VISIBLE_DEVICES`, as shown above.

### 4.3 Background srun (limited, not recommended at scale)

The `srun ... &` pattern works but has hard limits:

```bash
for node in $(scontrol show hostnames); do
    srun -N 1 -n 8 -c 7 --gpus-per-task=1 --gpu-bind=closest <executable> &
done
wait
```

OLCF explicitly warns: *"unreliable and does not scale to a significant portion of
Frontier... We do not yet recommend this approach beyond 100 simultaneous srun's."*

The Slurm `--stepmgr` flag (introduced in Slurm 24.05, Aug 2024) was intended to fix
this by managing steps on the first node instead of the Slurm controller, but it does
not currently work reliably on Frontier.

### 4.4 How this maps to MAXIE

| MAXIE task | Approach | Why |
|---|---|---|
| 400 downstream eval runs (1 node each) | Flux | Many independent jobs, auto load-balances |
| Scaling law sweep (varying model size) | Flux | Each config is independent, different resource needs |
| Pre-training (512 nodes) | Standard sbatch | Single large job, not an ensemble |
| Hyperparameter search on loss function | Flux or single-srun wrapper | Depends on per-run resource needs |

**Flux replaces the need for Parsl** for the evaluation campaign. Parsl adds value when
you need cross-facility orchestration (e.g., Frontier + Perlmutter), but for runs
within a single Frontier allocation, Flux is simpler and native to the system.

---

## 5. Next Steps

- [ ] Validate sbcast workflow with a small test job (2-4 nodes)
- [ ] Build Hydra multirun configs for scaling law sweep
- [ ] Set up analysis scripts for cross-run metric comparison
- [ ] Evaluate Parsl for orchestrating the 400 downstream evaluation runs
- [ ] Test AWS-OFI-RCCL plugin impact on AllReduce at scale
- [ ] Build Flash Attention for ROCm and benchmark with MAXIE's ViT
