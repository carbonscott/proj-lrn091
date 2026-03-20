# Task: Update CLAUDE.md for OLCF Frontier

## Background

We migrated from SLAC S3DF to OLCF Frontier. The file
`/lustre/orion/lrn091/proj-shared/cwang31/proj-lrn091/CLAUDE.md` contains
environment variables and HPC instructions that still point to SDF paths and
SLAC-specific resources. This document describes all changes needed.

## Prerequisites

- Access to Frontier (OLCF)
- The repos have already been cloned to `/lustre/orion/lrn091/proj-shared/cwang31/codes/`
- OLCF user docs are at `/lustre/orion/lrn091/proj-shared/cwang31/deps/olcf-user-docs`

---

## Part 1: Update Environment Variables

Replace the entire `export` block (lines 6-16) with:

```bash
export PEAKNET_DIR=/lustre/orion/lrn091/proj-shared/cwang31/codes/peaknet
export PEAKNET_PIPELINE_CODE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/codes/peaknet-pipeline-ray
export PEAKNET_PIPELINE_PROJ_DIR=/lustre/orion/lrn091/proj-shared/cwang31/codes/exp-peaknet-pipeline-ray
export PROJ_PEAKNET_DIR=/lustre/orion/lrn091/proj-shared/cwang31/codes/exp-peaknet
export MAXIE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/codes/maxie
export XTAL_DATA_ASSEMBLED=/lustre/orion/lrn091/proj-shared/data
export TILED_DIR=/lustre/orion/lrn091/proj-shared/cwang31/codes/tiled
export UV_CACHE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/.UV_CACHE
export DATA_BROKER_DIR=/lustre/orion/lrn091/proj-shared/cwang31/codes/tiled-catalog-broker
export DATA_BROKER_EXAMPLE_DIR=/lustre/orion/lrn091/proj-shared/cwang31/codes/MAIQMag-data-broker-examples
```

### What changed and why

| Variable | Change | Reason |
|---|---|---|
| `PEAKNET_DIR` | SDF path → codes/peaknet | Repo cloned to new location |
| `PEAKNET_PIPELINE_CODE_DIR` | SDF path → codes/peaknet-pipeline-ray | Repo cloned to new location |
| `PEAKNET_PIPELINE_PROJ_DIR` | SDF path → codes/exp-peaknet-pipeline-ray | Now points to the experiment repo |
| `PROJ_PEAKNET_DIR` | SDF path → codes/exp-peaknet | Now points to the experiment repo |
| `MAXIE_DIR` | SDF path → codes/maxie | Repo cloned to new location |
| `XTAL_DATA_ORIGINAL` | **Removed** | Not needed on Frontier |
| `XTAL_DATA_ASSEMBLED` | SDF path → /lustre/orion/.../proj-shared/data | Data location on Frontier |
| `TILED_DIR` | SDF path → codes/tiled | Now uses bluesky/tiled clone |
| `UV_CACHE_DIR` | SDF path → cwang31/.UV_CACHE | Cache location on Frontier |
| `DATA_BROKER_DIR` | SDF path → codes/tiled-catalog-broker | Repo cloned to new location |
| `DATA_BROKER_EXAMPLE_DIR` | SDF path → codes/MAIQMag-data-broker-examples | Repo cloned to new location |

---

## Part 2: Update HPC Resources Section

The current HPC section (lines 21-50) describes SLAC S3DF. It needs to be
rewritten for OLCF Frontier.

### Research step

Use the OLCF docs to get accurate details:

```bash
export OLCF_DOCS_ROOT=/lustre/orion/lrn091/proj-shared/cwang31/deps/olcf-user-docs
docs-index search "$OLCF_DOCS_ROOT" "frontier batch job slurm partition" --limit 5
```

Then read the Frontier user guide:

```bash
# Key file to read:
cat "$OLCF_DOCS_ROOT/systems/frontier_user_guide.rst"
```

### What to look for

Replace the following SLAC-specific details with Frontier equivalents:

| SLAC S3DF (current) | Frontier (find equivalent) |
|---|---|
| "SLAC's S3DF computing facility" | "OLCF's Frontier supercomputer" |
| Partition `ada` (GPU) | Frontier GPU partition name (likely `batch`) |
| Partition `milano` (CPU) | Frontier CPU-only partition if available |
| Account `lcls:prjdat21` | Account `lrn091` (verify in docs) |
| `--gpus=10` | Frontier uses AMD MI250X (8 GCDs per node) — adjust GPU flags |
| `salloc` syntax | Update flags for Frontier's SLURM configuration |
| ECC error caveat | Remove or replace with Frontier-specific caveats |
| CPU count caveat | Update for Frontier node specs |

### Template for new HPC section

```markdown
## HPC Resources

You are on OLCF's Frontier supercomputer. Frontier nodes have AMD EPYC 7A53
CPUs and 4x AMD MI250X GPUs (8 GCDs per node, 64 GB HBM2e each).

**GPU Nodes**

\```bash
salloc -A lrn091 -J interactive -t 2:00:00 -p batch -N 1
\```

**Submitting jobs**

Once allocated, use `srun` to launch on compute nodes.
```

**Important:** Verify partition names, account flags, and node specs from the
Frontier user guide before writing the final version.

---

## Part 3: Update Data Broker Section

Line 54 has a hardcoded SDF path:

```
Path: `/sdf/data/lcls/ds/prj/prjdat21/results/cwang31/proj-lrn091/externals/data-broker/tiled-catalog-broker`
```

Replace with:

```
Path: `/lustre/orion/lrn091/proj-shared/cwang31/codes/tiled-catalog-broker`
```

---

## Part 4: Verification Checklist

After making all edits, verify:

- [ ] No SDF paths remain: `grep -n "sdf" CLAUDE.md` should return nothing
- [ ] No SLAC references remain: `grep -in "slac\|s3df\|prjdat21\|lcls" CLAUDE.md` should return nothing
- [ ] All repo paths exist: run each `export` line, then `ls $PEAKNET_DIR $MAXIE_DIR $TILED_DIR` etc.
- [ ] HPC section references Frontier, not S3DF
- [ ] `XTAL_DATA_ORIGINAL` is removed (not just commented out)
- [ ] UV_CACHE_DIR path is writable: `mkdir -p $UV_CACHE_DIR && echo OK`
