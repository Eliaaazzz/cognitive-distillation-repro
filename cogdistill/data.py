"""Data utilities: a NumPy BadNets trigger injector + optional CIFAR-10 loader."""
from __future__ import annotations
import numpy as np


def add_badnets_trigger(images, size: int = 3, value: float = 1.0) -> np.ndarray:
    """Stamp a bright square trigger into the bottom-right corner (BadNets style).

    Args:
        images: array of shape (N, C, H, W) with values in [0, 1].
        size:   side length of the square trigger in pixels.
        value:  pixel value to write into the trigger region.

    Returns:
        A poisoned copy; the input array is not modified.
    """
    x = np.array(images, dtype=np.float32, copy=True)
    x[:, :, -size:, -size:] = value
    return x


def load_cifar10(root: str = "data", train: bool = False):
    """Return ``(images_uint8 NCHW, labels)`` for CIFAR-10 via torchvision (optional)."""
    import torchvision

    ds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
    images = np.transpose(ds.data, (0, 3, 1, 2))   # NHWC -> NCHW
    labels = np.array(ds.targets)
    return images, labels
