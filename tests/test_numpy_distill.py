from cogdistill.numpy_ref import run_smoke


def test_trigger_mask_smaller_than_clean():
    # The paper's core claim: a backdoor trigger reproduces the model output with a
    # much smaller distilled mask than a clean image requires.
    r = run_smoke(seed=0, alpha=0.02)
    assert r["trigger_mask_l1"] < 0.7 * r["clean_mask_l1"]


def test_masks_are_positive():
    r = run_smoke(seed=0, alpha=0.02)
    assert r["clean_mask_l1"] > 0.0 and r["trigger_mask_l1"] > 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
