"""Cognitive Distillation mask optimiser (PyTorch)."""
from __future__ import annotations


class CognitiveDistillation:
    """Extract the minimal 'cognitive pattern' a model uses for its prediction.

    For each input ``x`` we optimise a mask ``m in [0,1]^{1xHxW}`` minimising::

        || f(x * m) - f(x) ||_p   +   alpha * ||m||_1   +   beta * TV(m)

    where ``f`` is the frozen model's output (logits). A backdoored input needs
    only a tiny mask (the trigger alone drives the prediction), so ``mask_l1`` is
    a strong signal for detecting poisoned samples.

    Args:
        alpha: weight of the L1 sparsity penalty on the mask.
        beta:  weight of the total-variation smoothness penalty.
        steps: number of optimisation iterations.
        lr:    Adam learning rate for the mask.
        p:     norm for the output-matching term (the paper uses p=1).
        device: optional device override.
    """

    def __init__(self, alpha: float = 1e-2, beta: float = 1.0, steps: int = 100,
                 lr: float = 0.1, p: int = 1, device=None) -> None:
        self.alpha = alpha
        self.beta = beta
        self.steps = steps
        self.lr = lr
        self.p = p
        self.device = device

    def distill(self, model, x):
        """Return the distilled masks (N, 1, H, W) for a batch of inputs ``x``."""
        import torch

        device = self.device or next(model.parameters()).device
        x = x.to(device)
        model.eval()
        with torch.no_grad():
            ref = model(x).detach()

        n, _, h, w = x.shape
        # Parameterise the mask through a sigmoid so it stays in [0, 1].
        theta = torch.zeros(n, 1, h, w, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=self.lr)

        for _ in range(self.steps):
            optimizer.zero_grad()
            mask = torch.sigmoid(theta)
            out = model(x * mask)
            match = torch.norm((out - ref).flatten(1), p=self.p, dim=1)
            l1 = mask.flatten(1).mean(1)
            tv = ((mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().flatten(1).mean(1)
                  + (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().flatten(1).mean(1))
            loss = (match + self.alpha * l1 + self.beta * tv).mean()
            loss.backward()
            optimizer.step()

        return torch.sigmoid(theta).detach()

    @staticmethod
    def mask_l1(mask):
        """Per-sample average mask value -- the Cognitive Distillation score."""
        return mask.flatten(1).mean(1)
