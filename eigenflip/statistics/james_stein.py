"""
Canonical James-Stein shrinkage for the per-channel activation mean.

This replaces the multiple near-duplicate implementations scattered across the
RTN-heuristic and SmartFlip code. All encoders (CLC, EigenFlip, EigenFlip
Solve) use this one.

    mu_JS = grand_mean + (1 - c) (mu_hat - grand_mean),
    c = (p - 2) sigma^2 / sum_j (mu_hat_j - grand_mean)^2,   c clamped to [0,1]

Falls through unchanged when p < 3 or deviations are degenerate.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def james_stein_mean(raw_mean: torch.Tensor,
                     variance_estimate: torch.Tensor | None = None) -> torch.Tensor:
    p = raw_mean.numel()
    if p < 3:
        return raw_mean
    grand = raw_mean.mean()
    dev = raw_mean - grand
    ss = (dev * dev).sum()
    if ss < 1e-10:
        return raw_mean
    if variance_estimate is None:
        variance_estimate = (dev.abs().mean()) ** 2
        variance_estimate = variance_estimate.clamp_min(1e-8)
    c = ((p - 2) * variance_estimate) / ss
    c = c.clamp(0.0, 1.0)
    return grand + (1.0 - c) * dev
