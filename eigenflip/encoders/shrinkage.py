"""
Shrinkage-GPTQ baselines for the decisive comparison (Section 6.6).

These are the competitors the trust-region thesis must beat. Each is GPTQ
sequential conditioning run on a SHRUNK dense quadratic, sharing the base and
calibration of EigenFlip Solve.

Two families (paper Section 2, Section 6.6):

  (cov)  mean-preserving covariance shrinkage  -- the PRIMARY falsification baseline
         H_cov_lambda = mu mu^T + (1 - lambda) Sigma + lambda diag(Sigma)   (Eq. 13)
         Leaves the (stable, rung-1) mean untouched; shrinks only covariance
         off-diagonals toward the diagonal. Concedes the mean exactly as
         EigenFlip Solve does, so a remaining gap isolates the value of
         SELECTING the stable eigenspace over uniformly down-weighting all
         covariance off-diagonals.

  (2m)   second-moment blend  -- the SECONDARY (weaker, more commonly deployed) baseline
         H_lambda = (1 - lambda) H + lambda diag(H),  H = mu mu^T + Sigma   (form i)
         Shrinks the mean's off-diagonal contribution too. Beating this is
         necessary but not sufficient: it can be won by mean handling alone.

Three lambda instantiations per family (fixed before running):
  (i)   global lambda on a grid, tuned by HELD-OUT distortion on a disjoint
        calibration split (never the eval distribution) -- the deployable competitor.
  (ii)  per-layer analytic diagonal-target shrinkage (Ledoit-Wolf-style, diagonal
        target rather than scaled identity) -- principled, no tuning cost.
  (iii) per-layer grid-tuned lambda -- an oracle-flavored upper bound, clearly labeled.

All run through the same shrinking-set sequential conditioner used by DenseGPTQ
(encoders/dense_reference.py), so the ONLY difference from gptq is the matrix.

Requires stats.Sigma materialized (gram backend, keep_sigma=True for the layer).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats
from .dense_reference import _sequential_condition


# ----------------------------------------------------------------------------
# H builders (no padding; caller pads). All [d, d].
# ----------------------------------------------------------------------------

@torch.no_grad()
def build_H_full(mu, Sigma):
    """H = mu mu^T + Sigma."""
    return Sigma + torch.outer(mu, mu)


@torch.no_grad()
def build_H_cov_shrink(mu, Sigma, lam):
    """
    Eq. 13: mu mu^T + (1-lam) Sigma + lam diag(Sigma).
    The Sigma diagonal is preserved exactly: (1-lam) Sigma_ii + lam Sigma_ii =
    Sigma_ii. So we scale off-diagonals by (1-lam) and keep the diagonal.
    """
    d = Sigma.shape[0]
    idx = torch.arange(d, device=Sigma.device)
    diagSig = torch.diagonal(Sigma).clone()
    Sig_blend = (1.0 - lam) * Sigma
    Sig_blend[idx, idx] = diagSig                  # restore exact diagonal
    return torch.outer(mu, mu) + Sig_blend


@torch.no_grad()
def build_H_2m_shrink(mu, Sigma, lam):
    """
    form (i): (1-lam) H + lam diag(H), H = mu mu^T + Sigma.
    Diagonal of H is preserved; off-diagonals (which include mu mu^T's
    off-diagonals) are scaled by (1-lam).
    """
    H = Sigma + torch.outer(mu, mu)
    d = H.shape[0]
    idx = torch.arange(d, device=H.device)
    dH = torch.diagonal(H).clone()
    Hb = (1.0 - lam) * H
    Hb[idx, idx] = dH                              # restore exact diagonal
    return Hb


# ----------------------------------------------------------------------------
# Analytic per-layer lambda (instantiation ii): Ledoit-Wolf with diagonal target.
# ----------------------------------------------------------------------------

@torch.no_grad()
def ledoit_wolf_lambda_diag_target(Sigma_sample_second_moment: torch.Tensor,
                                   n_eff: int) -> float:
    """
    Ledoit-Wolf-style optimal shrinkage intensity toward the DIAGONAL target
    (not the scaled-identity target of vanilla LW). Estimates

        lambda* = E|| S - Sigma ||^2 / E|| S - T ||^2   (clamped to [0,1])

    where S is the sample covariance, T = diag(S) the target. We use the
    standard plug-in: numerator ~ sum_t || x_t x_t^T - S ||_F^2 / n^2 (the
    estimation variance of the off-diagonals), denominator = || S - diag(S) ||_F^2
    (the off-diagonal energy being shrunk). Diagonal target shrinks ONLY
    off-diagonals, so only off-diagonal terms enter.

    We approximate the numerator from second-moment structure without storing
    per-token outer products: a coarse but standard estimate uses
        pi_hat ~ (1/n) * || S ||_F^2-style dispersion.
    For a defensible per-layer value we use the off-diagonal Frobenius energy
    and an O(1/n_eff) variance proxy; the exact LW estimator needs per-token
    fourth moments we do not stream, so this is documented as an approximation.

    Argument here is the sample covariance S itself (pass Sigma).
    """
    S = Sigma_sample_second_moment
    d = S.shape[0]
    idx = torch.arange(d, device=S.device)
    off = S.clone()
    off[idx, idx] = 0.0
    denom = (off * off).sum().clamp_min(1e-12)        # ||S - diag(S)||_F^2
    # variance proxy: off-diagonal energy / n_eff (each off-diag entry has
    # sampling variance ~ (S_ii S_jj + S_ij^2)/n_eff; we use the cheap
    # trace-based upper proxy sum_ij S_ii S_jj / n_eff restricted to off-diag).
    diagS = torch.diagonal(S)
    outer_diag = torch.outer(diagS, diagS)
    outer_diag[idx, idx] = 0.0
    num = (outer_diag.sum() + denom) / max(1, n_eff)  # ~ E||S - Sigma||^2 (off-diag)
    lam = (num / denom).clamp(0.0, 1.0).item()
    return lam


# ----------------------------------------------------------------------------
# Encoder.
# ----------------------------------------------------------------------------

class ShrinkageGPTQ:
    """
    GPTQ on a shrunk H. family in {'cov', '2m'}. lambda is either a fixed float
    (instantiations i / iii supply the tuned value) or 'analytic' for (ii).

    The tuning of the global/per-layer lambda lives in the harness
    (pipeline/section66.py), which calls this encoder at candidate lambdas and
    scores held-out distortion. This class just encodes at a GIVEN lambda.
    """
    def __init__(self, family: str = "cov", lam: float | str = 0.01,
                 order: str = "diag", work_dtype=torch.float64,
                 n_eff: Optional[int] = None):
        assert family in ("cov", "2m")
        self.family = family
        self.lam = lam
        self.order = order
        self.work_dtype = work_dtype
        self.n_eff = n_eff
        self.name = f"shr_gptq_{family}"

    @torch.no_grad()
    def _build_H(self, mu, Sigma):
        if isinstance(self.lam, str) and self.lam == "analytic":
            lam = ledoit_wolf_lambda_diag_target(Sigma, self.n_eff or Sigma.shape[0])
        else:
            lam = float(self.lam)
        if self.family == "cov":
            return build_H_cov_shrink(mu, Sigma, lam), lam
        return build_H_2m_shrink(mu, Sigma, lam), lam

    @torch.no_grad()
    def apply(self, state: IntegerQuantizedTensorState, stats: LayerStats):
        assert stats.Sigma is not None, (
            "ShrinkageGPTQ needs a materialized Sigma (gram backend, "
            "keep_sigma=True).")
        dev = state.scale.device
        wdt = self.work_dtype
        d = stats.d
        pin = state.padded_in_features

        mu = stats.mu_hat.to(device=dev, dtype=wdt)
        Sigma = stats.Sigma.to(device=dev, dtype=wdt)
        H, lam_used = self._build_H(mu, Sigma)

        if pin > d:
            Hp = torch.zeros(pin, pin, device=dev, dtype=wdt)
            Hp[:d, :d] = H
            idx = torch.arange(d, pin, device=dev)
            Hp[idx, idx] = torch.diagonal(H).mean()
            H = Hp
        # numerical floor so the shrinking-set inverses are stable
        diagH = torch.diagonal(H).clone()
        H = H + 1e-8 * torch.diag(diagH)

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
        del H, Sigma, diagH
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out.to(state.original_dtype), {
            "encoder": self.name, "family": self.family, "lambda": lam_used,
            "backend": stats.backend}
