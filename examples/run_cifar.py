"""Real Cognitive Distillation on CIFAR-10 (requires torch + torchvision, GPU recommended).

Pipeline:
    1. Load a trained classifier (clean or backdoored).
    2. Build an evaluation set mixing clean and triggered images.
    3. Run Cognitive Distillation to get a mask per image.
    4. Use the mask L1 norm as a detection score; report AUROC (backdoor vs clean).

    python examples/run_cifar.py --checkpoint model.pt
"""
import argparse
import os
import sys

import numpy as np

# Allow running as a plain script by putting the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="trained (possibly backdoored) weights")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--num-eval", type=int, default=1000)
    ap.add_argument("--poison-frac", type=float, default=0.1)
    ap.add_argument("--trigger-size", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1e-2)
    args = ap.parse_args()

    import torch
    from cogdistill.models import small_cnn
    from cogdistill.distiller import CognitiveDistillation
    from cogdistill.data import load_cifar10, add_badnets_trigger
    from cogdistill.detect import detection_scores
    from cogdistill.metrics import auroc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = small_cnn(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    images, _ = load_cifar10(args.data_root, train=False)
    images = images[: args.num_eval].astype(np.float32) / 255.0
    n_poison = int(args.poison_frac * len(images))
    labels = np.zeros(len(images), dtype=int)
    labels[:n_poison] = 1
    images[:n_poison] = add_badnets_trigger(images[:n_poison], size=args.trigger_size)

    cd = CognitiveDistillation(alpha=args.alpha, steps=args.steps, device=device)
    x = torch.tensor(images, dtype=torch.float32, device=device)
    norms = []
    for i in range(0, len(x), 128):
        masks = cd.distill(model, x[i:i + 128])
        norms.append(cd.mask_l1(masks).cpu().numpy())
    norms = np.concatenate(norms)

    scores = detection_scores(norms)
    print("Backdoor-detection AUROC = %.4f" % auroc(scores[labels == 1], scores[labels == 0]))


if __name__ == "__main__":
    main()
