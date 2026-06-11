"""
Dense reference encoders (Section 6.5 validation harness).

These DO form a d x d matrix on purpose -- they are O(d^3) references, not the
deployed path. Two uses:

  DenseSurrogateGPTQ : run plain GPTQ/OBS sequential conditioning on the
                       MATERIALIZED H~_{k,eps} = D + V V^T. Algorithm 1 must
                       produce bitwise-identical codes to this. This is the
                       proof that EigenFlip Solve is an exact structured
                       implementation of the sequential rule, not an
                       approximation.

  DenseGPTQ          : the same sequential conditioning on the full empirical
                       second moment H (with diagonal damping) -- i.e. the
                       'gptq' ENCODER row of Table 2/6, runnable on any base.

Both consume the same IntegerQuantizedTensorState + LayerStats contract.
DenseGPTQ needs stats.H or stats.Sigma materialized (gram backend, keep_sigma).
"""

from __future__ import annotations

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


@torch.no_grad()
def _sequential_condition(Wf, scale, zp, lo, hi, Hmat, order, work_dtype):
    """
    GPTQ sequential conditioning under a dense quadratic Hmat, matching the
    coordinate order. Returns integer codes [C, pin].

    TRUE GPTQ semantics: at step t, condition on the CURRENT remaining set R
    (not-yet-quantized coordinates). Compensation for r in R\\{i} is
        w_r -= e_i * [H_RR^{-1}]_{r,i} / [H_RR^{-1}]_{ii},
    with H_RR the principal submatrix on R. This is what GPTQ's running
    Cholesky/Schur complement computes and what EigenFlip Solve's Woodbury
    capacitance downdate reproduces. A fixed full-matrix inverse instead would
    silently disagree with Algorithm 1 by +/-1 codes (verified). Explicit
    recompute is O(d^4); fine for a reference harness.
    """
    dev = Wf.device
    C, pin = Wf.shape
    Hmat = Hmat.to(work_dtype)
    order = list(order)

    # STANDARD GPTQ: permute (H, W, scale, zp) into processing order, invert H
    # ONCE, then schur-downdate the inverse by rank-1 each step -- O(d^3) total,
    # not O(d^4). Bitwise-identical to the per-step inv(H_RR) form (verified).
    p = torch.tensor(order, device=dev)
    Hp = Hmat.index_select(0, p).index_select(1, p)        # [pin, pin]
    Wp = Wf.index_select(1, p).clone()                     # [C, pin]
    sc_p = scale.index_select(1, p)                        # [C, pin]
    zp_p = zp.index_select(1, p)
    Hinv = torch.linalg.inv(Hp)
    codes_p = torch.empty(C, pin, device=dev, dtype=torch.long)

    for i in range(pin):
        si = sc_p[:, i]; zpi = zp_p[:, i]
        q = torch.clamp(torch.round(Wp[:, i] / si + zpi), lo, hi)
        w_dq = (q - zpi) * si
        e = Wp[:, i] - w_dq          # GPTQ sign: target - dequant
        codes_p[:, i] = q.long()
        if i + 1 < pin:
            denom = Hinv[i, i]
            factor = (Hinv[i, i+1:] / denom).to(work_dtype)         # [pin-i-1]
            Wp[:, i+1:] -= e.unsqueeze(1) * factor.unsqueeze(0)
            # schur downdate of the remaining inverse block
            col = Hinv[i+1:, i:i+1]
            row = Hinv[i:i+1, i+1:]
            Hinv[i+1:, i+1:] -= (col @ row) / denom

    # un-permute codes back to original coordinate positions
    codes = torch.empty(C, pin, device=dev, dtype=torch.long)
    codes.index_copy_(1, p, codes_p)
    del Hp, Wp, Hinv, codes_p
    return codes


class DenseSurrogateGPTQ:
    """GPTQ on the materialized H~_{k,eps} = D + V V^T. Reference for Algorithm 1."""
    name = "dense_surrogate_gptq"

    def __init__(self, order: str = "leverage", work_dtype=torch.float64):
        self.order = order
        self.work_dtype = work_dtype

    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        D = stats.D.to(device=dev, dtype=wdt)
        V = stats.V.to(device=dev, dtype=wdt)
        if pin > d:
            Dp = torch.empty(pin, device=dev, dtype=wdt); Dp[:d] = D; Dp[d:] = D.mean()
            Vp = torch.zeros(pin, V.shape[1], device=dev, dtype=wdt); Vp[:d] = V
            D, V = Dp, Vp
        # MATERIALIZE the surrogate (reference only)
        Htilde = torch.diag(D) + V @ V.t()                 # [pin, pin]

        scale = state.scale.to(wdt); zp = state.zero_point.to(wdt)
        Wf = state.float_weights.to(wdt)
        lo, hi = float(state.min_int), float(state.max_int)

        order = self._order(D, V)
        codes = _sequential_condition(Wf, scale, zp, lo, hi, Htilde, order, wdt)

        out = (codes.to(wdt) - zp) * scale
        if pin > d:
            out = out[:, :d]
        del Htilde, D, V
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), {
            "encoder": self.name, "k": stats.k, "codes": codes}

    def _order(self, D, V):
        if self.order == "leverage":
            lev = (1.0 / D) * (V * V).sum(dim=1)
            return torch.argsort(lev, descending=True).tolist()
        if self.order == "diag":
            return torch.argsort(D, descending=True).tolist()
        return list(range(D.shape[0]))


class DenseGPTQ:
    """
    The 'gptq' ENCODER (Table 2 rung-4 row) on full H, diagonally damped.
    Runnable on any base. Requires stats.Sigma materialized (gram, keep_sigma);
    H = mu mu^T + Sigma. Damping: H + damp * diag(H).
    """
    name = "gptq"

    def __init__(self, damp: float = 0.01, order: str = "diag",
                 work_dtype=torch.float64):
        self.damp = damp
        self.order = order
        self.work_dtype = work_dtype

    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        assert stats.Sigma is not None, (
            "DenseGPTQ needs a materialized Sigma (use gram backend, "
            "keep_sigma=True).")
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        mu = stats.mu_hat.to(device=dev, dtype=wdt)
        H = stats.Sigma.to(wdt) + torch.outer(mu, mu)      # [d, d]
        if pin > d:
            Hp = torch.zeros(pin, pin, device=dev, dtype=wdt)
            Hp[:d, :d] = H
            idx = torch.arange(d, pin, device=dev)
            Hp[idx, idx] = torch.diagonal(H).mean()
            H = Hp
        # diagonal damping (form ii)
        diagH = torch.diagonal(H).clone()
        H = H + self.damp * torch.diag(diagH)

        scale = state.scale.to(wdt); zp = state.zero_point.to(wdt)
        Wf = state.float_weights.to(wdt)
        lo, hi = float(state.min_int), float(state.max_int)

        if self.order == "diag":
            order = torch.argsort(diagH, descending=True).tolist()
        else:
            order = list(range(pin))
        codes = _sequential_condition(Wf, scale, zp, lo, hi, H, order, wdt)

        out = (codes.to(wdt) - zp) * scale
        if pin > d:
            out = out[:, :d]
        del H, diagH
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), {"encoder": self.name, "damp": self.damp}
