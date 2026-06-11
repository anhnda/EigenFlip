"""
Budgeted-flip encoders -- FAITHFUL port of the working RTN+JS heuristic CLC.

CLC (rung 1): start from base RTN/AWQ codes, greedily flip rounding decisions
(+/-1 quant level) in the anti-residual direction to drive the per-output
expected error  E[dY_i] = sum_j E[X_j] (W_ij - Wq_ij)  toward 0 per row.

Direct port of quantize_weight_heuristic_groupwise from the proven RTN-JS code:
SAME current_error, SAME validity guard (only flip if it reduces |current_error|),
SAME cumsum-residual best-k, SAME per-row cap.

EigenFlip (rung 2): same machinery, scalar b=mu^T e replaced by z=V^T e.
V=[mu] (k=0) reduces EXACTLY to the scalar CLC.

Knobs:
  knee_tolerance / use_knee : Kneedle outlier mask on |E[X]|. Default OFF.
  max_flip_frac             : per-row cap (fraction of in_features). 0.05 default.
"""
from __future__ import annotations
import torch
from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


@torch.no_grad()
def _find_knee(values, tol_offset):
    n = values.numel()
    if n < 3: return n // 2
    y = values.detach().float()
    ymin, ymax = y.min(), y.max()
    if (ymax - ymin) < 1e-10: return n // 2
    yn = (y - ymin) / (ymax - ymin)
    xn = torch.linspace(0, 1, n, device=y.device)
    yline = yn[0] + (yn[-1] - yn[0]) * xn
    knee = int(torch.argmax((yn - yline).abs()).item())
    if knee < n - 1:
        knee = max(0, min(knee + int(tol_offset * n), n - 1))
    return knee


class FlipEncoder:
    def __init__(self, name, knee_tolerance=-10.0, max_flip_frac=0.05,
                 use_knee=False, work_dtype=torch.float32):
        self.name = name
        self.knee_tolerance = knee_tolerance
        self.max_flip_frac = max_flip_frac
        self.use_knee = use_knee
        self.work_dtype = work_dtype

    @torch.no_grad()
    def apply(self, state, stats):
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        V = stats.V.to(device=dev, dtype=wdt)            # [d, k+1], col0 = mu
        kp1 = V.shape[1]
        if pin > d:
            Vp = torch.zeros(pin, kp1, device=dev, dtype=wdt); Vp[:d] = V; V = Vp
        ex_mean = V[:, 0]                                 # [pin]

        scale = state.scale.to(wdt)
        zp = state.zero_point.to(wdt)
        pre = state.pre_round.to(wdt)
        Wint = state.integer_weights.to(wdt).clone()
        C = Wint.shape[0]
        max_int = state.max_int

        Wf = state.float_weights.to(wdt)
        W_quant = (Wint - zp) * scale
        W_diff = Wf - W_quant                            # (W - Wq)
        z = W_diff @ V                                   # [C, k+1]; col0 = mean error

        flip_dir = torch.sign(pre - Wint)
        flip_dir = torch.where(flip_dir == 0, torch.ones_like(flip_dir), flip_dir)
        de = flip_dir * scale                            # change in Wq is +de => (W-Wq) changes by -de

        proposed = Wint + flip_dir
        valid = (proposed >= 0) & (proposed <= max_int)

        # GUARD: only flip if it reduces |scalar mean error z0|.
        # flipping (c,j): d(W-Wq) = -de  => d z0 = -de*mu. Reduce |z0| iff
        # sign(-de*mu) == -sign(z0)  <=>  sign(de*mu) == sign(z0).
        z0 = z[:, 0:1]
        impact0 = de * ex_mean.unsqueeze(0)
        valid = valid & (torch.sign(impact0) == torch.sign(z0))

        if self.use_knee:
            sdesc, _ = torch.sort(ex_mean.abs(), descending=True)
            half = sdesc[: pin // 2]
            knee = _find_knee(half, self.knee_tolerance)
            thresh = sdesc[knee]
            valid = valid & (~(ex_mean.abs() > thresh)).unsqueeze(0)

        regret = (pre - Wint).abs()
        regret = torch.where(valid, regret, torch.full_like(regret, -1.0))
        order = torch.argsort(regret, dim=1, descending=True)

        de_sorted = torch.gather(de, 1, order)
        V_sorted = V[order]                              # [C, pin, k+1]
        valid_sorted = torch.gather(valid.long(), 1, order)
        dz = (-de_sorted).unsqueeze(2) * V_sorted        # d z = (W-Wq) change . V
        dz = dz * valid_sorted.unsqueeze(2)
        z_path = z.unsqueeze(1) + torch.cumsum(dz, dim=1)
        norm_path = (z_path * z_path).sum(dim=2)
        z0n = (z * z).sum(dim=1, keepdim=True)
        all_norms = torch.cat([z0n, norm_path], dim=1)
        best_m = torch.argmin(all_norms, dim=1)

        cap = max(1, int(self.max_flip_frac * state.in_features))
        idx = torch.arange(pin, device=dev).unsqueeze(0)
        accept = (idx < best_m.unsqueeze(1)) & valid_sorted.bool()
        accept = accept & (accept.long().cumsum(dim=1) <= cap)

        fd_sorted = torch.gather(flip_dir, 1, order)
        applied = torch.where(accept, fd_sorted, torch.zeros_like(fd_sorted))
        Wint.scatter_add_(1, order, applied)
        Wint.clamp_(0, max_int)

        out = (Wint - zp) * scale
        if pin > d:
            out = out[:, :d]
        info = {"encoder": self.name, "k": stats.k,
                "total_flips": int(accept.sum().item()), "cap": cap}
        del V, scale, zp, pre, Wf, W_quant, W_diff, z, de, dz, z_path
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), info


def make_clc(knee_tolerance=-10.0, max_flip_frac=0.05, use_knee=False):
    return FlipEncoder("clc", knee_tolerance, max_flip_frac, use_knee)


def make_eigenflip(knee_tolerance=-10.0, max_flip_frac=0.05, use_knee=False):
    return FlipEncoder("eigenflip", knee_tolerance, max_flip_frac, use_knee)
