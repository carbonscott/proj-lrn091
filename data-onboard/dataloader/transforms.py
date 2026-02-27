"""Augmentations appropriate for X-ray diffraction images.

Diffraction images are single-channel intensity data where pixel values have
physical meaning. We avoid color jitter and random erasing (patches already
have natural masking from panel gaps). Rotations are limited to 90-degree
increments since panels have a physical orientation.
"""

import torch


class DiffractionTransform:
    """Augmentations for diffraction image patches.

    Applies random 90-degree rotations and random flips. These are physically
    valid symmetry operations for detector panels.
    """

    def __init__(self, rotate=True, flip=True):
        self.rotate = rotate
        self.flip = flip

    def __call__(self, x):
        """Apply augmentations to a tensor.

        Args:
            x: Tensor of shape [C, H, W].

        Returns:
            Augmented tensor of same shape.
        """
        if self.rotate:
            # Random 90-degree rotation (0, 90, 180, or 270 degrees)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                x = torch.rot90(x, k, dims=[1, 2])

        if self.flip:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[2])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[1])

        return x
