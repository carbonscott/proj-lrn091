## Environment Setup

Set these environment variables before running any commands:

```bash
export PEAKNET_DIR=/sdf/home/c/cwang31/codes/peaknet
export PEAKNET_PIPELINE_CODE_DIR=/sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray
export PEAKNET_PIPELINE_PROJ_DIR=/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet-1m/
export PROJ_PEAKNET_DIR=/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet
export MAXIE_DIR=/sdf/home/c/cwang31/codes/maxie
export XTAL_DATA_ORIGINAL=/sdf/data/lcls/ds/prj/prjdat21/results/cwang31/proj-lrn091/data
export XTAL_DATA_ASSEMBLED=/sdf/data/lcls/ds/prj/prjdat21/results/cwang31/proj-lrn091/data/assembled
export TILED_DIR=/sdf/data/lcls/ds/prj/prjcwang31/results/software/tiled
export UV_CACHE_DIR=/sdf/data/lcls/ds/prj/prjdat21/scratch/cwang31/.UV_CACHE
```

Use uv to run python programs. The UV_CACHE_DIR avoids repeated package downloads.

## HPC Resources

You are on SLAC's S3DF computing facility.  There are GPU nodes and CPU nodes
that you can access to.

**GPU Nodes**

```bash
node.ada.srun ()
{
    salloc --partition ada --exclusive -n 10 -A lcls:prjdat21 --gpus=10 --time=24:00:00
}
```

Caveat: GPU nodes often have ECC errors.  Use `CUDA_VISIBLE_DEVICES` to
exclude failing GPUs.

**CPU Nodes**

```bash
node.milano.srun ()
{
    salloc --partition milano --exclusive -A lcls:prjdat21 --time=24:00:00
}
```

Caveat: Total number of CPUs differ by nodes.  Check resources after gaining
access to a node.

Once you gain access to any of the compute nodes, you can then submit jobs through `srun`.
