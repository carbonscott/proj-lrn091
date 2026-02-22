"""PyTorch Dataset that reads assembled Zarr images and extracts patches.

Each item in the dataset corresponds to one assembled frame. At __getitem__
time, a random patch of size (patch_size x patch_size) is cropped from the
assembled image, with retries to avoid patches that fall entirely in panel
gaps (zeros).

Same interface as PanelPatchDataset: same patch_size, transform, experiments
params; same output tensor shape [1, H, W].

Supports both naming conventions:
  - Legacy: {experiment_id}.zarr
  - peaknet10k: {experiment_id}_r{run}.{chunk}.zarr
"""

import re
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


def _extract_experiment_id(zarr_path):
    """Extract experiment ID from a Zarr store filename.

    Handles both naming conventions:
      - Legacy: "cxi101235425.zarr" -> "cxi101235425"
      - peaknet10k: "cxi101235425_r0106.0001.zarr" -> "cxi101235425"
    """
    stem = Path(zarr_path).stem  # removes .zarr
    # Remove .NNNN chunk suffix if present (e.g. "cxi101235425_r0106.0001" -> "cxi101235425_r0106")
    stem = re.sub(r'\.\d{4}$', '', stem)
    # Remove _rNNNN run suffix if present
    stem = re.sub(r'_r\d{4}$', '', stem)
    return stem


class AssembledPatchDataset(Dataset):
    """Reads assembled Zarr stores and extracts random patches.

    Args:
        zarr_dir: Directory containing {experiment_id}.zarr/ stores.
        patch_size: Size of square patches to extract (default 256).
        transform: Optional callable for augmentation (applied after
            normalization, receives a [1, H, W] tensor).
        experiments: Optional list of experiment IDs to include.
            If None, includes all .zarr stores found in zarr_dir.
        max_retries: Maximum number of retries to avoid all-zero patches
            (panel gaps). Default 5.
    """

    def __init__(self, zarr_dir, patch_size=256, transform=None,
                 experiments=None, max_retries=5):
        self.patch_size = patch_size
        self.transform = transform
        self.max_retries = max_retries

        zarr_dir = Path(zarr_dir)

        # Build index: list of (zarr_path, frame_idx)
        self.index = []
        self._build_index(zarr_dir, experiments)

    def _build_index(self, zarr_dir, experiments):
        """Scan zarr stores and build (zarr_path, frame_idx) index."""
        zarr_paths = sorted(zarr_dir.glob("*.zarr"))

        if experiments is not None:
            exp_set = set(experiments)
            zarr_paths = [p for p in zarr_paths
                          if _extract_experiment_id(p) in exp_set]

        skipped = 0
        for zpath in zarr_paths:
            try:
                store = zarr.open(str(zpath), mode="r")
                images = store["images"]
                num_frames = images.shape[0]
                img_h, img_w = images.shape[1], images.shape[2]
            except (KeyError, Exception):
                skipped += 1
                continue

            # Check assembled image is large enough for patching
            if img_h < self.patch_size and img_w < self.patch_size:
                skipped += 1
                continue

            for frame_idx in range(num_frames):
                self.index.append((str(zpath), frame_idx))

        if skipped > 0:
            print(f"AssembledPatchDataset: skipped {skipped} stores "
                  f"(not found or too small for patch_size={self.patch_size})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        zarr_path, frame_idx = self.index[idx]

        try:
            store = zarr.open(zarr_path, mode="r")
            frame = np.array(store["images"][frame_idx], dtype=np.float32)
        except (OSError, KeyError, ValueError):
            patch = np.zeros((self.patch_size, self.patch_size),
                             dtype=np.float32)
            return torch.from_numpy(patch).unsqueeze(0)

        # Try to extract a non-gap patch (retry if patch is all zeros)
        patch = None
        for _ in range(self.max_retries):
            candidate = self._extract_patch(frame)
            if candidate.max() != candidate.min():
                patch = candidate
                break

        if patch is None:
            # All retries hit gap regions; use last candidate
            patch = candidate

        # Per-patch normalization
        mean = patch.mean()
        std = patch.std()
        if std > 0:
            patch = (patch - mean) / std
        else:
            patch = patch - mean

        tensor = torch.from_numpy(patch).unsqueeze(0)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

    def _extract_patch(self, frame):
        """Extract a random patch_size x patch_size crop from an assembled frame."""
        fh, fw = frame.shape
        ps = self.patch_size

        if fh >= ps and fw >= ps:
            top = np.random.randint(0, fh - ps + 1)
            left = np.random.randint(0, fw - ps + 1)
            return frame[top:top + ps, left:left + ps].copy()

        # Frame smaller than patch in one or both dims: pad with zeros
        patch = np.zeros((ps, ps), dtype=np.float32)
        crop_h = min(fh, ps)
        crop_w = min(fw, ps)
        top = np.random.randint(0, max(fh - crop_h + 1, 1))
        left = np.random.randint(0, max(fw - crop_w + 1, 1))
        patch[:crop_h, :crop_w] = frame[top:top + crop_h, left:left + crop_w]
        return patch
