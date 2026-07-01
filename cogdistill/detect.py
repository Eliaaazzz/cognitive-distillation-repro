"""Backdoor-sample detection from Cognitive Distillation mask norms (NumPy)."""
from __future__ import annotations
import numpy as np


def detection_scores(mask_norms) -> np.ndarray:
    """Turn mask L1 norms into detection scores where HIGHER = more suspicious.

    A backdoored sample yields an unusually *small* mask, so we simply negate the
    norms; the result plugs straight into ``auroc`` / thresholding.
    """
    return -np.asarray(mask_norms, dtype=np.float64)


def flag_outliers(mask_norms, k: float = 3.0) -> np.ndarray:
    """Flag samples whose mask norm is abnormally small using a robust MAD rule.

    Returns a boolean array; ``True`` marks a suspected backdoor (lower-tail
    outlier in the distribution of mask norms).
    """
    x = np.asarray(mask_norms, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    robust_z = (x - med) / (1.4826 * mad)
    return robust_z < -k
