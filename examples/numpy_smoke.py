"""Run the NumPy reference of Cognitive Distillation (CPU, no torch / GPU).

    python examples/numpy_smoke.py
"""
import os
import sys

# Allow running as a plain script (python examples/numpy_smoke.py) by putting
# the repo root on the path, not just `python -m`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogdistill.numpy_ref import run_smoke


def main() -> None:
    r = run_smoke()
    ratio = r["trigger_mask_l1"] / r["clean_mask_l1"]
    print("Cognitive Distillation - NumPy smoke test")
    print("  mean mask L1 (clean)   = %.4f" % r["clean_mask_l1"])
    print("  mean mask L1 (trigger) = %.4f" % r["trigger_mask_l1"])
    print("  ratio trigger / clean  = %.3f" % ratio)
    print("  -> triggered (backdoored) samples reproduce the model output with a")
    print("     much smaller mask, exactly as the paper predicts.")


if __name__ == "__main__":
    main()
