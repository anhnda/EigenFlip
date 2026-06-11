"""
Distortion scoring and lambda tuning for the Section 6.6 comparison.

Distortion metric (paper Eq. 3 / Eq. 7): for a layer with original weight W and
quantized weight Wq, residual E = (Wq - W) [C, d], the layer distortion under a
second moment H' is

    L(H') = tr(E H' E^T) = sum_j e_j^T H' e_j.

For tuning we need L on a HELD-OUT H' (a disjoint calibration split, never the
eval distribution). We compute tr(E H' E^T) without forming d x d products
larger than necessary:

    tr(E H' E^T) = sum over rows j of  e_j^T H' e_j
                 = || L_chol^T E^T ||_F^2   if H' = L_chol L_chol^T,
    but simplest stable form for moderate C is  (E @ H') elementwise* E summed.

We use  tr(E H' E^T) = ((E @ H') * E).sum()  -- one [C,d]@[d,d] plus a reduce.
For d up to ~10k and C up to ~rows this is one dense matmul; acceptable since
tuning runs on a small grid for a subset of probe layers (the paper tunes the
GLOBAL lambda on held-out distortion, not per element).

The held-out H' itself is built from a disjoint calibration split's streamed
statistics (mu_B, Sigma_B). We do NOT need to re-encode under B; we score the
A-fit encoder's residual E (computed once) against H'(B).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


@torch.no_grad()
def residual(state: IntegerQuantizedTensorState, corrected_W: torch.Tensor,
             work_dtype=torch.float64) -> torch.Tensor:
    """
    E = (Wq - W) on the unpadded width [C, d]. `corrected_W` is the encoder
    output (dequantized) [C, d]; W is the original weight [C, d].
    """
    Worig = state.float_weights.to(work_dtype)
    if state.padded_in_features > state.in_features:
        Worig = Worig[:, : state.in_features]
    return corrected_W.to(work_dtype) - Worig


@torch.no_grad()
def distortion(E: torch.Tensor, Hprime: torch.Tensor) -> float:
    """
    tr(E H' E^T) = sum_j e_j^T H' e_j, via ((E @ H') * E).sum().
    E: [C, d]; Hprime: [d, d]. Both on the same device/dtype.
    """
    EH = E @ Hprime                # [C, d]
    return (EH * E).sum().item()


@torch.no_grad()
def build_heldout_H(mu_B: torch.Tensor, Sigma_B: torch.Tensor,
                    work_dtype=torch.float64) -> torch.Tensor:
    """H'(B) = mu_B mu_B^T + Sigma_B from the held-out split's statistics."""
    mu = mu_B.to(work_dtype)
    return Sigma_B.to(work_dtype) + torch.outer(mu, mu)


# ----------------------------------------------------------------------------
# Global lambda tuning (instantiation i): one lambda for the whole model,
# chosen by total held-out distortion summed over probe layers.
# ----------------------------------------------------------------------------

LAMBDA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)


@torch.no_grad()
def tune_global_lambda(
    encoder_factory: Callable[[float], object],
    probe_layers: list,                 # list of (W, state, stats_A, Hprime_B)
    grid: tuple = LAMBDA_GRID,
    work_dtype=torch.float64,
) -> tuple[float, dict]:
    """
    For each lambda in grid, encode every probe layer with encoder_factory(lam)
    using its A-fit stats, score the residual under that layer's held-out
    H'(B), sum over layers. Return the argmin lambda and the full score table.

    probe_layers entries:
      W       : original weight [C, d]
      state   : IntegerQuantizedTensorState (base codes) for that layer
      stats_A : LayerStats fit on split A (carries Sigma_A for the encoder)
      Hprime_B: held-out second moment [d, d] (built from split B)
    """
    scores = {lam: 0.0 for lam in grid}
    for lam in grid:
        enc = encoder_factory(lam)
        total = 0.0
        for (W, state, stats_A, Hprime_B) in probe_layers:
            corrected, _ = enc.apply(state, stats_A)
            E = residual(state, corrected, work_dtype)
            total += distortion(E.to(Hprime_B), Hprime_B)
            del corrected, E
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        scores[lam] = total
    best_lam = min(scores, key=scores.get)
    return best_lam, scores


@torch.no_grad()
def tune_perlayer_lambda(
    encoder_factory: Callable[[float], object],
    W: torch.Tensor, state, stats_A, Hprime_B,
    grid: tuple = LAMBDA_GRID,
    work_dtype=torch.float64,
) -> tuple[float, dict]:
    """
    Instantiation (iii): per-layer grid-tuned lambda (oracle-flavored upper
    bound). Same scoring, but the argmin is chosen per layer.
    """
    scores = {}
    for lam in grid:
        enc = encoder_factory(lam)
        corrected, _ = enc.apply(state, stats_A)
        E = residual(state, corrected, work_dtype)
        scores[lam] = distortion(E.to(Hprime_B), Hprime_B)
        del corrected, E
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best_lam = min(scores, key=scores.get)
    return best_lam, scores
