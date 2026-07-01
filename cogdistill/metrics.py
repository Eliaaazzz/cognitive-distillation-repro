"""Metrics (NumPy). For detection, the backdoored class is positive."""
from __future__ import annotations
import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    sr = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(scores_pos, scores_neg) -> float:
    """AUROC via Mann-Whitney U = P(score(pos) > score(neg))."""
    p = np.asarray(scores_pos, dtype=np.float64)
    n = np.asarray(scores_neg, dtype=np.float64)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    s = np.concatenate([p, n])
    y = np.concatenate([np.ones(len(p)), np.zeros(len(n))])
    ranks = _rankdata(s)
    return float((ranks[y == 1].sum() - len(p) * (len(p) + 1) / 2.0) / (len(p) * len(n)))
