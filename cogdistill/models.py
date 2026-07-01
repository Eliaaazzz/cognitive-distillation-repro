"""A small CNN for CIFAR-scale experiments (PyTorch)."""
from __future__ import annotations


def small_cnn(num_classes: int = 10, in_channels: int = 3):
    """Return a compact conv net suitable for CIFAR-10 backdoor experiments."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        nn.Linear(64 * 16, num_classes),
    )
