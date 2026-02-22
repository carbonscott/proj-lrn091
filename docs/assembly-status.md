# Assembly Status

Date: 2026-02-21


## Overview

This document tracks which experiments have been assembled into Zarr, which
remain unconverted, and what the assembled images look like. Assembly converts
raw stacked-panel HDF5 frames into geometrically correct 2D images using
CrystFEL geometry files (see `data-assembly.md` for pipeline details).


## Manifest summary

Frame counts and file counts from `data/manifest.json`. Two entries for
prjcwang31 correspond to two sub-datasets (cheetah and psocake).

| Experiment | HDF5 files | Total frames | Raw data size | Detector |
|---|---|---|---|---|
| cxilw5019 | 225 | 2,373,974 | 3.7 TB | Jungfrau 4M |
| mfxx49820 | 18 | 640,095 | 1.4 TB | ePix10k 2M |
| prjcwang31 (psocake) | 1,969 | 533,409 | — | Assembled (1920x1920) |
| mfxp22421 | 88 | 218,750 | 1.8 TB | ePix10k 2M |
| prjcwang31 (cheetah) | 44 | 146,190 | — | ePix10k 2M |
| cxil1015922 | 243 | 52,622 | 824 GB | Jungfrau 4M |
| cxi101235425 | 418 | 32,233 | 759 GB | Jungfrau 4M |
| mfx101211025 | 188 | 10,551 | 660 GB | Jungfrau 16M |
| cxil1005322 | 11 | 7,154 | 377 GB | Jungfrau 4M |
| mfx100903824 | 5 | 25 | 0.2 GB | ePix10k 2M |
| **Total** | **3,209** | **4,015,003** | | |


## Assembly progress

Five of eight experiments in `data/geometry_registry.json` have been assembled.

### Assembled (5 experiments)

| Experiment | Detector | Assembled shape | ZARR chunks | ~Frames (40/chunk) |
|---|---|---|---|---|
| cxil1015922 | Jungfrau 4M | 2203 x 2299 | 1,323 | ~52,920 |
| cxi101235425 | Jungfrau 4M | 2203 x 2299 | 808 | ~32,320 |
| mfx101211025 | Jungfrau 16M | 4216 x 4432 | 266 | ~10,640 |
| cxil1005322 | Jungfrau 4M | 2203 x 2299 | 179 | ~7,160 |
| mfx100903824 | ePix10k 2M | 1692 x 1692 | 1 | 25 |

Output: `data/assembled/{exp}_r{run}.{chunk:04d}.zarr/`

### Not yet assembled (3 experiments)

| Experiment | Detector | Frames | Why not assembled |
|---|---|---|---|
| cxilw5019 | Jungfrau 4M | 2,373,974 | Largest dataset. Many frames may be empty (r0017 all-zeros). |
| mfxx49820 | ePix10k 2M | 640,095 | Many frames are sentinel value -1. |
| mfxp22421 | ePix10k 2M | 218,750 | Not yet attempted. |

### Excluded from geometry registry

| Experiment | Reason |
|---|---|
| prjcwang31 (cheetah) | Needs separate geometry discovery per sub-experiment |
| prjcwang31 (psocake) | Already assembled at source (1920x1920 square images) |

These three unconverted experiments represent ~3.2M additional frames. At the
assembled Zarr compression ratios observed so far, this could amount to 10-20 TB
of assembled data.


## Visual inspection of assembled ZARRs

Spot-checked one frame from each assembled experiment. Images exported to
`/tmp/zarr_spot_check/`. Assembly validation comparisons (raw vs assembled) at
`/tmp/assembly_validation/`.

### cxil1015922 (Jungfrau 4M, 2203x2299)

Bright concentric scattering ring with clear **Bragg peaks** visible as discrete
bright spots scattered across the ring. This is the strongest crystallography
dataset — the lysozyme beta-2 receptor experiment with prominent SFX hits.
Panel boundaries visible as dark gaps in the assembled image. Mean ~139,
std ~146.

### cxi101235425 (Jungfrau 4M, 2203x2299)

Strong **diffuse scattering** pattern centered around the beam position. Smooth
concentric intensity falloff. Some possible peaks but the frame is dominated by
background scatter. The Frenkel-CT exciton SFX experiment — signal may be
present in other frames. Mean ~41, std ~42.

### cxil1005322 (Jungfrau 4M, 2203x2299)

Very **weak signal** — mostly dark with faint panel-to-panel gain differences.
One panel region (lower-left) shows slightly elevated counts, possibly a nozzle
shadow or localized scatter. The nucleic acid dynamics experiment has low
background overall. Mean ~1.3, std ~3.

### mfx101211025 (Jungfrau 16M, 4216x4432)

Large assembled image (16M detector). Shows strong **smooth diffuse scattering**
rings — no Bragg peaks visible. The CPD photolyase experiment appears dominated
by solution/powder-like scattering. Panel gaps are prominent in the assembled
layout. Mean ~78, std ~125.

### mfx100903824 (ePix10k 2M, 1692x1692)

Clear concentric **artifact scattering rings** (likely solvent + Kapton/mylar).
Prominent bright center stripe artifact visible in both raw and assembled views.
Only 25 frames total — this is a small screening experiment. The panel layout
shows characteristic ePix10k 2M tile pattern with large gaps. Mean ~423,
std ~342.


## Assembled image sizes

| Detector | Raw shape | Assembled shape | Bytes/frame (float32) |
|---|---|---|---|
| Jungfrau 4M | 4096 x 1024 | 2203 x 2299 | ~19.3 MB |
| Jungfrau 16M | 16384 x 1024 | 4216 x 4432 | ~71.2 MB |
| ePix10k 2M | 5632 x 384 | 1692 x 1692 | ~10.9 MB |
| Assembled (psocake) | 1920 x 1920 | 1920 x 1920 (already assembled) | ~14.1 MB |
