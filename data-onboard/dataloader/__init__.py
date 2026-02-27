"""Data loading pipeline for LCLS X-ray diffraction/scattering images.

Public API:
    create_dataloader(manifest_path, ...) -> DataLoader
    create_assembled_dataloader(zarr_dir, ...) -> DataLoader
    PanelPatchDataset  — panel-wise patches from raw HDF5
    AssembledPatchDataset — patches from assembled Zarr images
    DiffractionTransform — augmentation transforms
    load_manifest, list_files — manifest utilities
"""

import torch
from torch.utils.data import DataLoader

from .assembled_dataset import AssembledPatchDataset
from .manifest import list_files, load_manifest
from .panel_dataset import PanelPatchDataset
from .transforms import DiffractionTransform


def create_dataloader(manifest_path, patch_size=256, batch_size=64,
                      num_workers=4, transform=None, experiments=None,
                      detectors=None, **kwargs):
    """Create a DataLoader for MAE/DINOv2 pre-training.

    Args:
        manifest_path: Path to data/manifest.json.
        patch_size: Size of square patches (default 256).
        batch_size: Batch size (default 64).
        num_workers: DataLoader workers (default 4).
        transform: Optional augmentation callable. If None, uses
            DiffractionTransform with default settings.
        experiments: Optional list of experiment IDs to include.
        detectors: Optional list of detector types to include.
        **kwargs: Additional arguments passed to DataLoader.

    Returns:
        torch.utils.data.DataLoader
    """
    if transform is None:
        transform = DiffractionTransform()

    dataset = PanelPatchDataset(
        manifest_path=manifest_path,
        patch_size=patch_size,
        transform=transform,
        experiments=experiments,
        detectors=detectors,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **kwargs,
    )


def create_assembled_dataloader(zarr_dir, patch_size=256, batch_size=64,
                                num_workers=4, transform=None,
                                experiments=None, **kwargs):
    """Create a DataLoader from assembled Zarr stores.

    Args:
        zarr_dir: Directory containing {experiment_id}.zarr/ stores.
        patch_size: Size of square patches (default 256).
        batch_size: Batch size (default 64).
        num_workers: DataLoader workers (default 4).
        transform: Optional augmentation callable. If None, uses
            DiffractionTransform with default settings.
        experiments: Optional list of experiment IDs to include.
        **kwargs: Additional arguments passed to DataLoader.

    Returns:
        torch.utils.data.DataLoader
    """
    if transform is None:
        transform = DiffractionTransform()

    dataset = AssembledPatchDataset(
        zarr_dir=zarr_dir,
        patch_size=patch_size,
        transform=transform,
        experiments=experiments,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **kwargs,
    )
