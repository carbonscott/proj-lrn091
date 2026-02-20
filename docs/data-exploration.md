# Data Exploration Findings

Exploration date: 2026-02-19

## Overview

We surveyed 10 LCLS experiments (CXI and MFX instruments) to understand the
X-ray diffraction/scattering image data before choosing a model architecture.
The data spans serial femtosecond crystallography (SFX), solution scattering
(SAXS/WAXS), and related techniques.

| Experiment | Instrument | PI | Title (short) |
|---|---|---|---|
| cxi101235425 | CXI | van Thor | Frenkel-CT exciton SFX |
| cxil1005322 | CXI | Pollack | Nucleic acid structural dynamics |
| cxil1015922 | CXI | Standfuss | Ligand dissociation in beta-2 receptor |
| cxilw5019 | CXI | Cherezov | SFX of GPCRs in LCP |
| mfx100903824 | MFX | Schleiss | Cryptochrome magnetoreception |
| mfx101211025 | MFX | Lane | CPD photolyase charge separation |
| mfxp22421 | MFX | Cherezov | Fixed-target SFX of GPCRs in LCP |
| mfxx49820 | MFX | Hunter | Automated droplet on demand |
| prjcwang31 | N/A | (project) | Personal project directory |

**Excluded**: `prjlute22` (timetool benchmark data, not diffraction images).


## Symlink layout

All symlinks live under `data/` with naming `<experiment>--<subdir-label>`.

| Symlink | Target | Notes |
|---|---|---|
| `cxi101235425--cheetah-hdf5` | `.../cxi/cxi101235425/results/cheetah/hdf5` | Cheetah hit-finding output |
| `cxil1005322--cheetah-hdf5` | `.../cxi/cxil1005322/results/tgrant/cheetah/hdf5` | Cheetah output |
| `cxil1005322--reborn-hdf5` | `.../cxi/cxil1005322/results/cxil1005322/hdf5` | Reborn stats (mean/sdev/sum) |
| `cxil1015922--cheetah-hdf5` | `.../cxi/cxil1015922/results/qbertran/cheetah/hdf5` | Cheetah output |
| `cxilw5019--psocake` | `.../cxi/cxilw5019/results/cgati/cgati/psocake` | psocake output |
| `mfx100903824--results-cxi` | `.../mfx/mfx100903824/results` | Direct .cxi results (5 files) |
| `mfx101211025--cheetah-hdf5` | `.../mfx/mfx101211025/results/Ctah2/cheetah/hdf5` | Cheetah output |
| `mfxp22421--cheetah-hdf5` | `.../mfx/mfxp22421/results/tony/.../cheetah/hdf5` | Cheetah output |
| `mfxx49820--lute-drcomp-f32` | `.../mfx/mfxx49820/results/nclaret/.../baseline` | LUTE DrComp pipeline |
| `prjcwang31--userdata-cheetah` | `.../prj/prjcwang31/userdata/cheetah` | Cheetah user data |
| `prjcwang31--userdata-psocake` | `.../prj/prjcwang31/userdata` | psocake user data |

Created by: `scripts/01_create_symlinks.sh` (original location)


## HDF5 structure

The standard image key across nearly all experiments is:

```
/entry_1/data_1/data    shape: (N_events, H, W)    dtype: float32
```

This follows the CXIDB convention used by Cheetah, psocake, and OM.
Other datasets in the same files include peak-finding results
(`/entry_1/result_1/nPeaks`, `peakXPosRaw`, `peakYPosRaw`, etc.)
and metadata (`/LCLS/photon_energy_eV`, `/LCLS/timestamp`, etc.).

Full structure per experiment: `data/hdf5_structure_summary.json`
Explored by: `exploration/scripts/explore_hdf5.py`


## Detector types

Four distinct detector geometries appear in the data:

| Detector | Raw shape (H x W) | Experiments |
|---|---|---|
| Jungfrau 4M | 4096 x 1024 | cxi101235425, cxil1005322, cxil1015922, cxilw5019 |
| ePix10k2M | 5632 x 384 | mfx100903824, mfxp22421, mfxx49820, prjcwang31 (cheetah) |
| Jungfrau 16M | 16384 x 1024 | mfx101211025 |
| Assembled (square) | 1920 x 1920 | prjcwang31 (psocake, mfx13016 sub-experiment) |

All multi-panel detectors store panels stacked vertically in the raw array
(i.e., panels are concatenated along the H dimension). Geometry files are
needed to assemble panels into a physically accurate image, but the stacked
format is usable for model training as-is.


## Image content categories

### 1. Bragg peaks (signal of interest)

Discrete bright spots from microcrystal diffraction. These are the primary
signal for SFX experiments and the feature a model should learn to detect.

**Clearly present in:**
- **cxil1015922** (lysozyme): Very prominent, many bright spots/streaks across
  panels. Best example of strong crystallographic diffraction.
- **cxil1005322** (nucleic acid): Visible bright spots on some frames.
- **prjcwang31 psocake** (mfx13016, assembled): Discrete spots visible on
  scattering rings, especially in log scale.
- **cxi101235425** (Frenkel-CT SFX): Present in some frames.

**Absent or rare in:**
- **mfxp22421**, **mfx100903824**, **mfx101211025**: Dominated by smooth
  continuous scattering (likely non-hit frames or solution/powder scattering).

### 2. Artifact scattering (not from the biochemical sample)

Smooth rings, partial rings, or arcs from non-sample materials in the beam
path. These are nuisance features the model should learn to ignore or segment.

- **Water/solvent ring**: Broad ring from the liquid jet or surrounding
  solvent. Visible in ePix10k2M experiments (mfx100903824, mfxx49820,
  mfxp22421) as smooth ring patterns in log scale.
- **Kapton/mylar arcs**: Partial rings or arcs from beamline windows.
- **LCP scattering**: In LCP-based SFX experiments (cxilw5019, mfxp22421),
  the lipidic cubic phase carrier contributes its own characteristic ring.
- **Upstream scatter**: Bright vertical stripe at beam center column visible
  in mfx100903824 (possibly direct beam leak or upstream component scatter).

### 3. Shadows

Dark regions caused by physical obstructions between the sample and detector.

- **Beam stop**: Dark circle at image center (clearly visible in prjcwang31
  assembled 1920x1920 images).
- **Nozzle/injector shadow**: Asymmetric dark region cutting across panels.
  Visible in cxil1005322 and cxilw5019.
- **Varying per run**: Shadow position and shape change depending on injector
  alignment, which varies between runs and experiments.

### 4. Detector artifacts

Features intrinsic to the detector hardware, not the X-ray signal.

- **Panel gaps**: Dark horizontal bands between panels in all stacked-panel
  images. These are physical gaps in the detector tiling.
- **Gain variation**: Different panels have noticeably different baseline
  intensity levels (especially visible in mfxp22421 log-scale images).
- **Hot/dead pixels**: Individual outlier pixels, not systematically explored
  yet but expected.


## Dynamic range statistics

Per-frame statistics from sampled images (single frame, not averaged):

| Experiment | Shape | Min | Max | Mean | Std | Notes |
|---|---|---|---|---|---|---|
| cxi101235425 | 4096x1024 | -38 | 127,451 | 22-47 | 35-122 | Large max on some frames |
| cxil1005322 | 4096x1024 | -8 | 1,489 | 1.3-2.0 | 3.0-5.5 | Low background |
| cxil1015922 | 4096x1024 | -28 | 68,896 | 3.8-8.3 | 47-117 | Strong Bragg peaks |
| cxilw5019 | 4096x1024 | -149,890 | 101,253 | 13-21 | 284-286 | Very wide range, some frames all-zero |
| mfx100903824 | 5632x384 | -50 | 39,386 | 543-600 | 271-299 | High baseline |
| mfx101211025 | 16384x1024 | -54 | 2,967 | 87-108 | 112-132 | Moderate range |
| mfxp22421 | 5632x384 | -293 | 16,675 | 69-876 | 20-676 | Highly variable across frames |
| mfxx49820 | 5632x384 | -3 | 1,795 | 33 | 23 | Many frames are -1 (empty/sentinel) |
| prjcwang31 (cheetah) | 5632x384 | -6 | 13,628 | 29-56 | 31-37 | Moderate range |
| prjcwang31 (psocake) | 1920x1920 | -17k | 97,290 | 19-71 | 34-123 | Assembled, very wide range |

Key observations:
- Dynamic range spans 4-6 orders of magnitude within a single frame.
- Background levels vary significantly between experiments and detectors.
- Some datasets contain sentinel values (-1 in mfxx49820, all-zeros in
  cxilw5019 r0017/r0022) indicating empty or unprocessed frames.


## Visualization

Three clipping methods were tested:

1. **mean + 4*std** (`vmin=mean, vmax=mean+4*std`): Best general-purpose
   method. Shows Bragg peaks clearly against background. Recommended default.
2. **Log scale** (`log10(image - min + 1)`): Reveals faint features (artifact
   rings, shadows, panel gain differences). Best for diagnostics.
3. **Percentile** (`1st-99th percentile`): Good balance, less sensitive to
   extreme outliers than mean+4*std.

Sample images: `data/sample-images/<experiment>/`
All images (3 methods): `/tmp/lcls_sample_images/<experiment>/`
Generated by: `exploration/scripts/visualize_samples.py`


## Excluded data

| Source | Reason |
|---|---|
| `prjlute22` | Contains timetool benchmark data (TMO), not diffraction images |
| `mfx100845525--om-hdf5` | 1D azimuthal-integrated data (Nx3019), not 2D images |
| `mfx101211025--lute-tjump` | Smalldata (scalar metadata per event), no images |
| `cxil1005322--reborn-hdf5` | Statistical summaries only (mean/sdev/sum), not per-event images |
| All `hdf5/smalldata/` dirs | Contain reduced scalar data, not detector images |
