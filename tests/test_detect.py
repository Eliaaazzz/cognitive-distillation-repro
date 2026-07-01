import numpy as np

from cogdistill.detect import detection_scores, flag_outliers
from cogdistill.metrics import auroc


def test_detection_scores_rank_small_norms_high():
    norms = np.array([0.5, 0.5, 0.05, 0.5])   # index 2 is a backdoor (tiny mask)
    scores = detection_scores(norms)
    assert int(scores.argmax()) == 2


def test_flag_outliers_catches_small():
    norms = np.array([0.5, 0.52, 0.48, 0.51, 0.05])
    flags = flag_outliers(norms, k=3.0)
    assert flags[-1] and not flags[:-1].any()


def test_auroc_perfect_when_backdoors_have_smaller_norms():
    clean = np.full(40, 0.5)
    backdoor = np.full(10, 0.05)
    labels = np.array([1] * 10 + [0] * 40)
    scores = detection_scores(np.concatenate([backdoor, clean]))
    assert auroc(scores[labels == 1], scores[labels == 0]) == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
