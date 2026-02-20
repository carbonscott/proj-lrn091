#!/bin/bash
# Create symlinks under data/ for easy access to experiment HDF5 data.
# Naming: <experiment_id>--<subdir_label>
# Run from the project root: bash scripts/01_create_symlinks.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"

mkdir -p "${DATA_DIR}/sample-images"

# --- MFX experiments ---

# mfx100845525: OnDA Monitor (OM) output with .cxi files
ln -sfn /sdf/data/lcls/ds/mfx/mfx100845525/results/tgrant/om/hdf5 \
        "${DATA_DIR}/mfx100845525--om-hdf5"

# mfxx49820: LUTE DrComp pipeline output (targeted subdir to avoid 1.3M+ file tree)
ln -sfn /sdf/data/lcls/ds/mfx/mfxx49820/results/nclaret/lute_output/DrComp_f32/baseline \
        "${DATA_DIR}/mfxx49820--lute-drcomp-f32"

# mfx101211025: Cheetah hit-finding output
ln -sfn /sdf/data/lcls/ds/mfx/mfx101211025/results/Ctah2/cheetah/hdf5 \
        "${DATA_DIR}/mfx101211025--cheetah-hdf5"

# mfx101211025: LUTE T-jump pipeline output
ln -sfn /sdf/data/lcls/ds/mfx/mfx101211025/results/tjump/lute_output \
        "${DATA_DIR}/mfx101211025--lute-tjump"

# mfxp22421: Cheetah output (fixed-target SFX)
ln -sfn /sdf/data/lcls/ds/mfx/mfxp22421/results/tony/backup_except_cheetah/cheetah/cheetah/hdf5 \
        "${DATA_DIR}/mfxp22421--cheetah-hdf5"

# mfx100903824: Direct .cxi results (small experiment, 5 files)
ln -sfn /sdf/data/lcls/ds/mfx/mfx100903824/results \
        "${DATA_DIR}/mfx100903824--results-cxi"

# --- CXI experiments ---

# cxil1005322: Cheetah output
ln -sfn /sdf/data/lcls/ds/cxi/cxil1005322/results/tgrant/cheetah/hdf5 \
        "${DATA_DIR}/cxil1005322--cheetah-hdf5"

# cxil1005322: Reborn pipeline output
ln -sfn /sdf/data/lcls/ds/cxi/cxil1005322/results/cxil1005322/hdf5 \
        "${DATA_DIR}/cxil1005322--reborn-hdf5"

# cxi101235425: Cheetah output
ln -sfn /sdf/data/lcls/ds/cxi/cxi101235425/results/cheetah/hdf5 \
        "${DATA_DIR}/cxi101235425--cheetah-hdf5"

# cxil1015922: Cheetah output (ligand dissociation dynamics)
ln -sfn /sdf/data/lcls/ds/cxi/cxil1015922/results/qbertran/cheetah/hdf5 \
        "${DATA_DIR}/cxil1015922--cheetah-hdf5"

# cxilw5019: psocake output (SFX of GPCRs)
ln -sfn /sdf/data/lcls/ds/cxi/cxilw5019/results/cgati/cgati/psocake \
        "${DATA_DIR}/cxilw5019--psocake"

# --- Project directories ---

# prjcwang31: userdata with psocake-style data across sub-experiments
ln -sfn /sdf/data/lcls/ds/prj/prjcwang31/userdata \
        "${DATA_DIR}/prjcwang31--userdata-psocake"

# prjcwang31: cheetah output within userdata
ln -sfn /sdf/data/lcls/ds/prj/prjcwang31/userdata/cheetah \
        "${DATA_DIR}/prjcwang31--userdata-cheetah"

echo "Created $(ls -1d "${DATA_DIR}"/*--* 2>/dev/null | wc -l) symlinks in ${DATA_DIR}/"
echo "Created sample-images directory at ${DATA_DIR}/sample-images/"
