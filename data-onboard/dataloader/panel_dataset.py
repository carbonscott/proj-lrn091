"""PyTorch Dataset that reads HDF5 frames, splits into panels, and extracts patches.

Each item in the dataset corresponds to one panel from one frame. At __getitem__
time, a random patch of size (patch_size x patch_size) is cropped from the panel.
HDF5 files are opened lazily (per-access) so we don't hold file handles open
across workers.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import load_manifest, list_files


class PanelPatchDataset(Dataset):
    """Lazily reads HDF5 frames, splits into panels, extracts random patches.

    Args:
        manifest_path: Path to the manifest.json file.
        patch_size: Size of square patches to extract (default 256).
        transform: Optional callable for augmentation (applied after
            normalization, receives a [1, H, W] tensor).
        experiments: Optional list of experiment IDs to include.
        detectors: Optional list of detector types to include.
        project_root: Root directory for resolving relative paths in the
            manifest. Defaults to the manifest's parent's parent directory.
    """

    def __init__(self, manifest_path, patch_size=256, transform=None,
                 experiments=None, detectors=None, project_root=None):
        self.patch_size = patch_size
        self.transform = transform

        manifest_path = Path(manifest_path)
        self.manifest = load_manifest(manifest_path)

        if project_root is None:
            # manifest is at data/manifest.json, project root is parent of data/
            self.project_root = manifest_path.parent.parent
        else:
            self.project_root = Path(project_root)

        # Get filtered file list
        files = list_files(self.manifest, experiments=experiments,
                           detectors=detectors)

        # Build index: list of (file_path, frame_idx, panel_idx, panel_h, panel_w, image_key)
        self.index = []
        self._build_index(files)

    def _build_index(self, files):
        """Pre-compute all (file, frame, panel) triples."""
        skipped = 0
        for file_info in files:
            file_path = self.project_root / file_info["path"]

            if not file_path.exists():
                skipped += 1
                continue

            num_frames = file_info["num_frames"]
            num_panels = file_info["num_panels"]
            panel_h = file_info["panel_shape"][0]
            panel_w = file_info["panel_shape"][1]
            image_key = file_info["image_key"]

            # Check that panel_size is large enough for patch extraction
            if panel_h < self.patch_size and panel_w < self.patch_size:
                skipped += 1
                continue

            for frame_idx in range(num_frames):
                for panel_idx in range(num_panels):
                    self.index.append((
                        str(file_path),
                        frame_idx,
                        panel_idx,
                        panel_h,
                        panel_w,
                        image_key,
                    ))

        if skipped > 0:
            print(f"PanelPatchDataset: skipped {skipped} files "
                  f"(not found or too small for patch_size={self.patch_size})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_path, frame_idx, panel_idx, panel_h, panel_w, image_key = self.index[idx]

        # Read the frame from HDF5 (with error handling for corrupt files)
        try:
            with h5py.File(file_path, "r") as f:
                ds = f[image_key]
                if len(ds.shape) == 3:
                    frame = ds[frame_idx]  # (H, W)
                else:
                    frame = ds[:]  # (H, W) for 2D datasets
            frame = np.array(frame, dtype=np.float32)
        except (OSError, KeyError, ValueError):
            # Corrupt file or read error — return zero patch
            patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            return torch.from_numpy(patch).unsqueeze(0)

        # Split into panels along H dimension
        panel_start = panel_idx * panel_h
        panel_end = panel_start + panel_h
        panel = frame[panel_start:panel_end, :panel_w]

        # Skip sentinel frames (all -1 or all 0)
        panel_min = panel.min()
        panel_max = panel.max()
        if panel_max == panel_min:
            # Return a zero patch (will be filtered by the training loop or
            # handled gracefully — better than crashing)
            patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        else:
            # Extract a random patch
            patch = self._extract_patch(panel)

            # Per-patch normalization: subtract mean, divide by std
            mean = patch.mean()
            std = patch.std()
            if std > 0:
                patch = (patch - mean) / std
            else:
                patch = patch - mean

        # Convert to tensor: [1, patch_size, patch_size]
        tensor = torch.from_numpy(patch).unsqueeze(0)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

    def _extract_patch(self, panel):
        """Extract a random patch_size x patch_size crop from a panel.

        If the panel is smaller than patch_size in either dimension, the patch
        is zero-padded.
        """
        ph, pw = panel.shape
        ps = self.patch_size

        if ph >= ps and pw >= ps:
            # Normal case: random crop
            top = np.random.randint(0, ph - ps + 1)
            left = np.random.randint(0, pw - ps + 1)
            return panel[top:top + ps, left:left + ps].copy()

        # Panel smaller than patch in one or both dims: pad with zeros
        patch = np.zeros((ps, ps), dtype=np.float32)
        crop_h = min(ph, ps)
        crop_w = min(pw, ps)

        # Random offset within panel (for the smaller dimension)
        top = np.random.randint(0, max(ph - crop_h + 1, 1))
        left = np.random.randint(0, max(pw - crop_w + 1, 1))

        patch[:crop_h, :crop_w] = panel[top:top + crop_h, left:left + crop_w]
        return patch
