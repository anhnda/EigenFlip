"""
EigenFlip Solve  --  Algorithm 1 of the paper.

Sequential conditioning (the GPTQ/OBS update) applied to the trust-region
surrogate H~_{k,eps} = D + V V^T, executed in closed form via Woodbury so that
NO d x d matrix is ever formed and the only inverse is the (k+1) x (k+1)
capacitance M = I + V_R^T D_R^{-1} V_R.

Per coordinate i (in order pi), after rounding with error e_i, GPTQ would do
    w_rest <- w_rest - e_i * [H~^{-1}]_{rest,i} / [H~^{-1}]_{ii}.
Woodbury gives, for r in R, r != i,
    [H~^{-1}]_{r,i} = - D_rr^{-1} V_r M^{-1} V_i^T D_ii^{-1}
    [H~^{-1}]_{ii}  =   D_ii^{-1} (1 - D_ii^{-1} V_i M^{-1} V_i^T),
so the compensation of every remaining r is a rank-one update in R^{k+1}:
    w_r <- w_r + D_rr^{-1} <V_r, g_i>,   g_i = e_i * (M^{-1} V_i^T D_ii^{-1}) / [H~^{-1}]_{ii}.
The direction (M^{-1} V_i^T) is SHARED across all C rows; only e_i is per-row.
Lazy accumulator per row:  G_j = sum_{i processed} g_{i,j} in R^{k+1};
compensated weight at coordinate r is  w_r + D_rr^{-1} <V_r, G_j>.

Capacitance maintenance: as i leaves R, M <- M - D_ii^{-1} V_i^T V_i (rank-one
downdate, O(k^2)); maintain M directly and M^{-1} by Sherman-Morrison; refresh
M^{-1} by reinverting the maintained M every T steps (O(k^3), no d-rescan).

Memory: V [d,k+1], M & M^{-1} [k+1,k+1], G [C,k+1], plus the codes. All O(kd)
or smaller. Never O(d^2).

Order pi: descending leverage D_ii^{-1} ||V_i||^2 by default (rung-2 analog of
activation ordering); 'diag' falls back to descending D_ii.
"""

from __future__ import annotations

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


def _round_clamp(x, lo, hi):
    return torch.clamp(torch.round(x), lo, hi)


class EigenFlipSolve:
    name = "eigenflip_solve"

    def __init__(self, order: str = "leverage", refresh_T: int | None = None,
                 work_dtype: torch.dtype = torch.float64,
                 guard_no_dxd: bool = True):
        self.order = order
        self.refresh_T = refresh_T
        self.work_dtype = work_dtype
        self.guard_no_dxd = guard_no_dxd

    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        if self.guard_no_dxd:
            assert stats.Sigma is None, (
                "EigenFlip Solve must run without a materialized Sigma; "
                "free_sigma() before deploying.")

        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        # Trust-region pieces, padded to padded_in_features if needed.
        D = stats.D.to(device=dev, dtype=wdt)            # [d]
        V = stats.V.to(device=dev, dtype=wdt)            # [d, k+1]
        kp1 = V.shape[1]
        if pin > d:
            Dp = torch.empty(pin, device=dev, dtype=wdt)
            Dp[:d] = D
            Dp[d:] = D.mean()                            # inert padding, PD
            Vp = torch.zeros(pin, kp1, device=dev, dtype=wdt)
            Vp[:d] = V
            D, V = Dp, Vp
            del Dp, Vp

        scale = state.scale.to(wdt)                      # [C, pin]
        zp = state.zero_point.to(wdt)
        Wf = state.float_weights.to(wdt)                 # [C, pin] continuous target
        C = Wf.shape[0]
        lo, hi = float(state.min_int), float(state.max_int)

        Dinv = 1.0 / D                                   # [pin]

        # Capacitance over the full set R = all coordinates initially.
        # M = I + V^T diag(Dinv) V
        VtDinv = V.t() * Dinv.unsqueeze(0)               # [k+1, pin]
        M = torch.eye(kp1, device=dev, dtype=wdt) + VtDinv @ V
        Minv = torch.linalg.inv(M)
        del VtDinv

        # Per-row lazy accumulator G [C, k+1].
        G = torch.zeros(C, kp1, device=dev, dtype=wdt)
        codes = torch.empty_like(state.integer_weights)

        order = self._order(D, V, Dinv)
        T = self.refresh_T or max(1, pin // 10)

        total_comp = 0.0
        for step, i in enumerate(order):
            Vi = V[i]                                    # [k+1]
            Di_inv = Dinv[i].item()
            si = scale[:, i]                             # [C]
            zpi = zp[:, i]

            # lazy compensation: w~_i = w_i + D_ii^{-1} <V_i, G_j>
            comp_i = Di_inv * (G @ Vi)                   # [C]
            w_tilde = Wf[:, i] + comp_i                  # [C] continuous weight
            # round in quantized domain
            q = _round_clamp(w_tilde / si + zpi, lo, hi) # [C]
            w_dq = (q - zpi) * si
            e = w_tilde - w_dq                           # [C] error: target - dequant (GPTQ sign)
            codes[:, i] = q.to(codes.dtype)

            # H~^{-1}_{ii} = Dinv_i (1 - Dinv_i V_i M^{-1} V_i^T)
            MinvVi = Minv @ Vi                           # [k+1]
            quad = (Vi * MinvVi).sum().item()
            Hinv_ii = Di_inv * (1.0 - Di_inv * quad)
            # numerical floor: Hinv_ii must stay positive
            if Hinv_ii <= 1e-30:
                Hinv_ii = Di_inv  # degenerate; fall back to diagonal-only step

            # g_i direction (shared across rows up to scalar e):
            #   g_i = e * (Minv V_i^T Dinv_i) / Hinv_ii
            dir_vec = (MinvVi * (Di_inv / Hinv_ii))      # [k+1]
            G += e.unsqueeze(1) * dir_vec.unsqueeze(0)   # [C, k+1]
            total_comp += comp_i.abs().mean().item()

            # downdate capacitance: remove coordinate i from R
            # M <- M - Dinv_i V_i V_i^T   (rank-one)
            M -= Di_inv * torch.outer(Vi, Vi)
            # Sherman-Morrison downdate of Minv for A' = A - alpha u u^T:
            #   A'^{-1} = A^{-1} + alpha (A^{-1} u)(u^T A^{-1}) / (1 - alpha u^T A^{-1} u)
            u = Vi
            Ainv_u = Minv @ u
            denom = 1.0 - Di_inv * (u * Ainv_u).sum()
            if denom.abs() > 1e-12:
                Minv = Minv + (Di_inv / denom) * torch.outer(Ainv_u, Ainv_u)
            del Ainv_u, MinvVi

            if (step + 1) % T == 0:
                # periodic refresh: reinvert the maintained small M
                Minv = torch.linalg.inv(M)

        # write codes, dequantize to final weights
        out = (codes.to(wdt) - zp) * scale
        if pin > d:
            out = out[:, :d]
        info = {
            "encoder": self.name, "k": stats.k, "order": self.order,
            "mean_abs_compensation": total_comp / max(1, len(order)),
            "backend": stats.backend,
        }
        del D, V, M, Minv, G, Wf, scale, zp, Dinv, codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), info

    @torch.no_grad()
    def _order(self, D, V, Dinv):
        if self.order == "leverage":
            lev = Dinv * (V * V).sum(dim=1)      # D_ii^{-1} ||V_i||^2
            return torch.argsort(lev, descending=True).tolist()
        if self.order == "diag":
            return torch.argsort(D, descending=True).tolist()
        return list(range(D.shape[0]))           # natural
