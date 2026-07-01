# Cognitive Distillation (reproduction)

A minimal, faithful re-implementation of **Cognitive Distillation (CD)** — a method
that detects backdoor / trojan samples in image classifiers by distilling the minimal
input pattern a model relies on.

> **Reference.** Huang, Ma, Erfani, Bailey, *"Distilling Cognitive Backdoor Patterns
> within an Image"*, **ICLR 2023**. [arXiv:2301.10908](https://arxiv.org/abs/2301.10908) ·
> official code: [HanxunH/CognitiveDistillation](https://github.com/HanxunH/CognitiveDistillation)
>
> This is an independent **educational reproduction**, not affiliated with the authors.

## Idea in one paragraph

For a trained model `f` and an input image `x`, CD optimises a small spatial mask
`m ∈ [0,1]^{1×H×W}` so that the model's output on the masked image `f(x·m)` matches its
output on the original `f(x)`, while keeping the mask as **small (L1)** and **smooth (TV)**
as possible. For a *clean* image the minimal sufficient pattern is roughly the whole
object, so the mask is large. For a *backdoored* image the trigger alone drives the
prediction, so a **tiny** mask suffices. The **L1 norm of the distilled mask** therefore
separates poisoned samples from clean ones.

## Why this repo is structured the way it is

The mask optimiser (`distiller.py`) uses **PyTorch**, because it must backprop through a
real network. To make the central claim **verifiable on a CPU with no downloads**, the
repo also ships a **NumPy reference** (`numpy_ref.py`) that reproduces the effect on a
controllable toy "backdoored" linear model, plus NumPy detection + metrics.

```
cogdistill/
  distiller.py   # Cognitive Distillation mask optimiser (PyTorch)
  detect.py      # mask-norm -> backdoor score + robust outlier flagging (NumPy)
  metrics.py     # AUROC (NumPy)
  models.py      # small CNN for CIFAR (PyTorch)
  data.py        # BadNets trigger injector (NumPy) + CIFAR-10 loader (torchvision)
  numpy_ref.py   # NumPy reference of the mask optimisation (CPU, no torch)
examples/
  numpy_smoke.py # verifies the core claim on CPU (no torch, no data)
  run_cifar.py   # real pipeline: distill masks -> detect backdoors on CIFAR-10
tests/           # NumPy-only unit tests (no torch, no network)
```

## Quickstart (CPU, no downloads)

```bash
pip install -r requirements.txt          # just numpy
python examples/numpy_smoke.py           # reproduces the core claim on a toy model
bash scripts/run_tests.sh                # runs the unit tests
```

Output of the smoke test (a triggered image needs a far smaller mask than a clean one —
the paper's central claim):

```
mean mask L1 (clean)   = 0.0545
mean mask L1 (trigger) = 0.0217
ratio trigger / clean  = 0.398
```

All 5 unit tests pass under Python 3.10 with only NumPy installed. On a mixed batch of
40 clean + 10 triggered toy samples, ranking by `-mask_l1` gives **detection AUROC = 1.0**.

## Running the real thing (CIFAR-10, needs torch + torchvision)

```bash
pip install -r requirements-torch.txt
# You supply a trained (optionally backdoored) checkpoint for cogdistill.models.small_cnn:
python examples/run_cifar.py --checkpoint model.pt --poison-frac 0.1 --trigger-size 3
```

This mixes clean and BadNets-triggered CIFAR-10 images, distills a mask per image, and
reports the backdoor-detection AUROC from the mask L1 norms.

## What was verified here vs. what needs a GPU

| Part | Status |
|------|--------|
| Mask optimisation core idea (softmax-output matching + L1) | ✅ NumPy reference, verified on CPU |
| Detection (mask-norm score, MAD outlier flag) + AUROC | ✅ unit-tested on CPU |
| PyTorch distiller on real CNNs / CIFAR-10 backdoor benchmark | ⏳ code provided; needs `torch`, data, GPU |

## Citation

```bibtex
@inproceedings{huang2023cognitive,
  title     = {Distilling Cognitive Backdoor Patterns within an Image},
  author    = {Huang, Hanxun and Ma, Xingjun and Erfani, Sarah Monazam and Bailey, James},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}
```

## License

MIT (this reproduction). The original method and its results are the authors'.
