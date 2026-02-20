## My random thoughts

What am I doing here?

I'm going to train a large vision model in this project.  I'm not going to reinvent models, but adapt existing models to my domain - X-ray diffraction/scattering imaging.  Why am I training these models?  Great questions, 

Now, should I train it using VAE, VQ-VAE, DINO, MAE or diffusion?  I don't have a good sesnse.  I need your help here.


## Explain the past work

**Path**: `./externals/past-work`

- **peaknet**: source code of peaknet, where the model is defined
- **maxie**: Masked AutoEncoder (MAE) repo, where MAE model is define
- **proj-peaknet**:
  - It's a huge repo, so be careful about searching through this repo.
  - `README.md` talks about how to launch a training job, but it's likely things
    are out-dated.
  - `train_convnext_seg.py`: it might be the training file you need
  - I recommand you check out the recent development documented in git to gain
    more understanding about this repo.

## Reference repos

**Path**: `./externals/reference-repos/`

## Deep learning docs

**Path**: `./externals/deeplearning-docs/`

## Data

### Caveats

Images can come from different detectors, and have different dynamic ranges.
CXI or HDF5 uses Cheetah tile that is a stacked image with a large
height-to-width aspect ratio in most cases.  The detector geometry information
is thus not present in images in CXI or HDF5.

### Source

Okay, I will spend time to curate datasets.

| Experiment ID | Instrument | Spokesperson (PI) | Proposal | Start Date | CXI Files | Total Size (GB) | Experiment Title |
|--------------|------------|-------------------|----------|------------|-----------|-----------------|------------------|
| prjcwang31 | N/A | Project Directory | N/A | N/A | 8,747 | 13,478.2 | Personal project directory (cwang31) |
| mfx100845525 | MFX | Alexandra Ros | 1008455 | 2025-10-06 | 5,855 | 664.9 | Towards deciphering the redox mechanism of the human flavoenzyme NQO1 using mix-and-inject segmented droplet injection |
| mfxx49820 | MFX | Mark Hunter | X498 | 2022-07-30 | 1,488 | 1,433.8 | Automated Droplet on Demand for Macromolecular Crystallography and SAXS/WAXS |
| cxil1005322 | CXI | Lois Pollack | L-10053 | 2024-05-02 | 1,082 | 376.7 | Completing studies of time-resolved nucleic acid structural dynamics with solution scattering and mixing injector |
| cxi101235425 | CXI | Jasper van Thor | 1012354 | 2025-11-06 | 561 | 759.4 | Femtosecond serial chemical crystallography of Frenkel-CT exciton transition charge dynamics for optoelectronics |
| cxil1015922 | CXI | Joerg Standfuss | L-10159 | 2024-04-13 | 244 | 824.0 | Ligand Dissociation Dynamics in the beta-2 adrenergic receptor |
| cxilw5019 | CXI | Vadim Cherezov | LW50 | 2021-10-30 | 225 | 3,710.8 | Serial Femtosecond Crystallography of G protein-Coupled Receptors in Lipidic Cubic Phase |
| mfx101211025 | MFX | Thomas Lane | 1012110 | 2025-09-27 | 188 | 659.5 | Charge separation and photoreduction of a CPD photolyase |
| mfxp22421 | MFX | Vadim Cherezov | P224 | 2022-10-02 | 88 | 1,763.6 | Fixed Target SFX of G protein-Coupled Receptors in Lipidic Cubic Phase |
| mfx100903824 | MFX | Pamela Schleiss | 1009038 | 2025-05-24 | 5 | 0.2 | Investigating the Quantum Biological Basis of Magnetoreception in Cryptochromes: Preliminary Screening for Optimization |
| prjlute22 | N/A | Project Directory | N/A | N/A | 1 | 0.03 | Project directory (unknown user) |

### Retrieval (Data broker)

It's possible we will need data broker, since there are too many types of data
we will have to handle.
