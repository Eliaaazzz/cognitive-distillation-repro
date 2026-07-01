"""Cognitive Distillation (reproduction).

A minimal, faithful re-implementation of Cognitive Distillation for detecting
backdoor / trojan samples in image classifiers.

Reference:
    Huang, Ma, Erfani, Bailey. "Distilling Cognitive Backdoor Patterns within an
    Image." ICLR 2023. https://arxiv.org/abs/2301.10908
    Official code: https://github.com/HanxunH/CognitiveDistillation

Not affiliated with the original authors; this is an educational reproduction.
The detection logic + a NumPy reference of the mask-optimisation run without any
heavy dependencies; the full optimiser (`distiller`) uses PyTorch.
"""
from .detect import detection_scores, flag_outliers
from .metrics import auroc

__all__ = ["detection_scores", "flag_outliers", "auroc"]
__version__ = "0.1.0"
