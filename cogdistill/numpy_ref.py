"""NumPy reference for the Cognitive Distillation core idea (CPU, no torch).

This is a *self-contained* demonstration of the paper's central claim on a
controllable toy problem, so the idea can be verified without a GPU or datasets.

We build a deliberately "backdoored" linear classifier whose target-class logit
is driven entirely by a small corner trigger. Cognitive Distillation optimises a
per-pixel mask ``m in [0,1]`` so that the model's (softmax) output on the masked
image matches its output on the full image, with an L1 penalty pushing the mask
to be as small as possible.

Expected result (the paper's claim): a **triggered** image can reproduce its
output with a *tiny* mask (just the trigger), whereas a **clean** image needs a
*large* mask -> the mask's L1 norm cleanly separates backdoored from clean data.
"""
from __future__ import annotations
import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _optimize_mask(Wt: np.ndarray, x: np.ndarray, steps: int, lr: float,
                   alpha: float) -> np.ndarray:
    """Projected-gradient distillation of a minimal mask (softmax-output matching)."""
    p_ref = _softmax(Wt @ x)
    m = np.ones_like(x)
    for _ in range(steps):
        p = _softmax(Wt @ (x * m))
        s = np.sign(p - p_ref)                 # subgradient of L1 match term
        dz = p * s - p * (p @ s)               # softmax Jacobian applied: (diag(p)-pp^T)^T s
        grad = (dz @ Wt) * x + alpha           # + d(alpha * sum(m))/dm
        m = np.clip(m - lr * grad, 0.0, 1.0)
    return m


def run_smoke(seed: int = 0, n_samples: int = 20, steps: int = 250,
              lr: float = 0.5, alpha: float = 0.02) -> dict:
    """Return mean mask-L1 for clean vs. triggered images (triggered should be smaller)."""
    rng = np.random.default_rng(seed)
    H = Wd = 8
    D = H * Wd
    n_classes = 3
    trigger_idx = [r * Wd + c for r in (6, 7) for c in (6, 7)]  # bottom-right 2x2

    Wt = np.zeros((n_classes, D))
    Wt[0, trigger_idx] = 5.0                     # target-class logit = 5 * (trigger brightness)
    Wt[1] = 0.3 * rng.standard_normal(D)         # other classes read diffuse pixels
    Wt[2] = 0.3 * rng.standard_normal(D)

    clean_l1, trig_l1 = [], []
    for _ in range(n_samples):
        x = rng.uniform(0.0, 1.0, size=D)
        x[trigger_idx] = 0.0                     # clean image: trigger region dark
        clean_l1.append(_optimize_mask(Wt, x, steps, lr, alpha).mean())

        xt = x.copy()
        xt[trigger_idx] = 1.0                    # same image + bright trigger
        trig_l1.append(_optimize_mask(Wt, xt, steps, lr, alpha).mean())

    return {
        "clean_mask_l1": float(np.mean(clean_l1)),
        "trigger_mask_l1": float(np.mean(trig_l1)),
    }


if __name__ == "__main__":
    r = run_smoke()
    print("mean mask L1  (clean)   = %.4f" % r["clean_mask_l1"])
    print("mean mask L1  (trigger) = %.4f" % r["trigger_mask_l1"])
    print("ratio trigger/clean     = %.3f  (smaller => backdoor stands out)"
          % (r["trigger_mask_l1"] / r["clean_mask_l1"]))
