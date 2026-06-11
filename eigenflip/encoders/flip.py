"""
Budgeted-flip encoders.

CLC (rung 1): anti-residual flips minimizing the first-moment drift b = mu^T e.
EigenFlip (rung 2): the same machinery with the scalar b replaced by the
(k+1)-vector state z = V^T e; the prefix rule minimizes ||z||^2.

Both start from the base RTN/AWQ codes and choose a budgeted set of single-step
flips (round the other way by one level) in the anti-residual direction. Flips
never move the continuous target -- that is EigenFlip Solve's job -- they only
pick among adjacent lattice points.

Knee / budget knobs match the existing CLC template:
  knee_tolerance  : Kneedle offset on sorted |coupling|; larger -> more channels
                    masked as outliers (never flipped). -10 / 'no-knee' disables.
  max_flip_frac   : per-row cap as a fraction of in_features. 1.0 = full budget.
"""

from __future__ import annotations

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


@torch.no_grad()
def _find_knee(values: torch.Tensor, tol_offset: float) -> int:
    n = values.numel()
    if n < 3:
        return n // 2
    y = values.detach().float()
    ymin, ymax = y.min(), y.max()
    if (ymax - ymin) < 1e-10:
        return n // 2
    yn = (y - ymin) / (ymax - ymin)
    xn = torch.linspace(0, 1, n, device=y.device)
    yline = yn[0] + (yn[-1] - yn[0]) * xn
    knee = int(torch.argmax((yn - yline).abs()).item())
    if knee < n - 1:
        knee = max(0, min(knee + int(tol_offset * n), n - 1))
    return knee


class FlipEncoder:
    """
    Shared budgeted prefix-flip encoder. With k=0 (V has one column = mu) it is
    CLC; with k>0 it is EigenFlip. `name` is set by the factory functions below.
    """

    def __init__(self, name: str, knee_tolerance: float = -10.0,
                 max_flip_frac: float = 1.0, use_knee: bool = False,
                 work_dtype: torch.dtype = torch.float32):
        self.name = name
        self.knee_tolerance = knee_tolerance
        self.max_flip_frac = max_flip_frac
        self.use_knee = use_knee
        self.work_dtype = work_dtype

    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        V = stats.V.to(device=dev, dtype=wdt)            # [d, k+1]
        kp1 = V.shape[1]
        if pin > d:
            Vp = torch.zeros(pin, kp1, device=dev, dtype=wdt)
            Vp[:d] = V
            V = Vp

        scale = state.scale.to(wdt)                      # [C, pin]
        zp = state.zero_point.to(wdt)
        Wint = state.integer_weights.to(wdt).clone()     # [C, pin] codes
        C = Wint.shape[0]
        lo, hi = float(state.min_int), float(state.max_int)
        max_int = state.max_int

        # current dequant error e = w_dq - w_target  (target = float_weights)
        w_dq = (Wint - zp) * scale
        e = w_dq - state.float_weights.to(wdt)           # [C, pin]
        # row state z = V^T e  -> [C, k+1]
        z = e @ V                                        # [C, k+1]

        # flip direction: anti-residual. Flipping i by +/-1 changes e by
        # +/- scale_i. Choose the sign that reduces |e_i| toward zero, i.e.
        # opposite to current rounding residual. Round residual sign:
        pre = state.pre_round.to(wdt)
        resid = pre - Wint                               # in (-0.5, 0.5] roughly
        flip_dir = torch.sign(resid)                     # +1 -> increment code
        flip_dir = torch.where(flip_dir == 0, torch.ones_like(flip_dir), flip_dir)

        # proposed code must stay in range
        proposed = Wint + flip_dir
        in_range = (proposed >= 0) & (proposed <= max_int)

        # delta to z from flipping (i): de_i = flip_dir_i * scale_i ; dz = de_i * V_i
        de = flip_dir * scale                            # [C, pin]
        # per-flip change to ||z||^2 evaluated greedily against current z below.

        # outlier mask via knee on coupling magnitude ||V_i|| (proxy for how
        # much a flip perturbs the coupled state). knee disabled if not use_knee.
        valid = in_range.clone()
        if self.use_knee:
            coupling = V.norm(dim=1)                     # [pin]
            sdesc, _ = torch.sort(coupling, descending=True)
            half = sdesc[: pin // 2]
            knee = _find_knee(half, self.knee_tolerance)
            thresh = sdesc[knee]
            is_outlier = coupling > thresh
            valid = valid & (~is_outlier).unsqueeze(0)

        # rounding regret = closeness to 0.5 (most "flippable" first)
        regret = (resid.abs())                           # [C, pin]
        regret = torch.where(valid, regret, torch.full_like(regret, -1.0))
        order = torch.argsort(regret, dim=1, descending=True)   # [C, pin]

        # gather per-row sorted flip contributions
        de_sorted = torch.gather(de, 1, order)           # [C, pin]
        Vi_sorted = V[order]                             # [C, pin, k+1]
        valid_sorted = torch.gather(valid.expand(C, pin).long()
                                    if valid.dim() == 1 else valid.long(),
                                    1, order)

        # greedily accept a prefix that minimizes ||z||^2. We evaluate the
        # cumulative state z + cumsum(dz) and pick the best prefix length.
        dz = de_sorted.unsqueeze(2) * Vi_sorted          # [C, pin, k+1]
        dz = dz * valid_sorted.unsqueeze(2)
        z_path = z.unsqueeze(1) + torch.cumsum(dz, dim=1)  # [C, pin, k+1]
        norm_path = (z_path * z_path).sum(dim=2)         # [C, pin]
        z0 = (z * z).sum(dim=1, keepdim=True)            # [C, 1]
        all_norms = torch.cat([z0, norm_path], dim=1)    # [C, pin+1]
        best_m = torch.argmin(all_norms, dim=1)          # [C] in [0, pin]

        # budget cap
        cap = max(1, int(self.max_flip_frac * state.in_features))
        best_m = torch.clamp(best_m, max=cap)

        idx = torch.arange(pin, device=dev).unsqueeze(0)
        accept_sorted = (idx < best_m.unsqueeze(1)) & valid_sorted.bool()

        # scatter accepted flips back, apply to codes
        flip_dir_sorted = torch.gather(flip_dir, 1, order)
        applied = torch.where(accept_sorted, flip_dir_sorted,
                              torch.zeros_like(flip_dir_sorted))
        Wint.scatter_add_(1, order, applied)
        Wint.clamp_(0, max_int)

        out = (Wint - zp) * scale
        if pin > d:
            out = out[:, :d]
        info = {
            "encoder": self.name, "k": stats.k,
            "total_flips": int(accept_sorted.sum().item()),
            "per_row_mean": float(accept_sorted.float().sum(dim=1).mean().item()),
            "cap": cap, "use_knee": self.use_knee, "backend": stats.backend,
        }
        del V, scale, zp, e, z, dz, z_path, norm_path, Vi_sorted, de_sorted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), info


def make_clc(knee_tolerance: float = -10.0, max_flip_frac: float = 1.0,
             use_knee: bool = False) -> FlipEncoder:
    """Rung-1: requires stats built with k=0 (V = [mu])."""
    return FlipEncoder("clc", knee_tolerance, max_flip_frac, use_knee)


def make_eigenflip(knee_tolerance: float = -10.0, max_flip_frac: float = 1.0,
                   use_knee: bool = False) -> FlipEncoder:
    """Rung-2: requires stats built with k>0 (V = [mu | U Lam^{1/2}])."""
    return FlipEncoder("eigenflip", knee_tolerance, max_flip_frac, use_knee)
