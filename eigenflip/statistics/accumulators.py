"""
Streaming activation-statistics accumulators.

Design constraints (from the trust-region encoder requirements):
  * NEVER materialize the full activation matrix  X in [N_tokens, d]
    (e.g. (128*2048) x 4096 ~ 8.6 GB, or x 10000 ~ 21 GB). We fold each
    hook fire into a persistent accumulator and drop the activations.
  * The deployed encoders (EigenFlip / EigenFlip Solve) consume only
        mu        = E[x]            [d]
        diag(H)   = E[x^2]          [d]   -> diag(Sigma) = E[x^2] - mu^2
        U_k,Lam_k = top-k eig(Sigma)      (rung-2 only)
    so we stream exactly enough to recover those, nothing d x d unless the
    chosen backend needs it.
  * Heavy layers: d can reach ~10k. A d x d Gram in fp64 at d=10000 is
    ~800 MB; eigh on it is non-trivial. dtype and device are therefore
    chosen PER LAYER against a VRAM budget, not fixed globally.
  * GPU-first; free dead tensors immediately.

Two covariance backends:
  StreamingGram   -- accumulate G = sum_t x_t x_t^T  (O(d^2) memory, exact
                     top-k via a single eigh). Preferred when it fits.
  StreamingSketch -- randomized range sketch Y = (sum_t x_t x_t^T) Omega,
                     Omega in [d, k+l]  (O(d(k+l)) memory, never forms d x d).
                     The fallback for heavy layers / tight VRAM.

Moments (mu, diag) are always streamed in StreamingMoments at O(d), shared
by both backends and by the rung-0/1 encoders that need no covariance.

Numerical note. diag(Sigma) = E[x^2] - mu^2 and Sigma = G/n - mu mu^T are
cancellation-prone when the mean is large relative to the spread (precisely
the high-mean outlier channels AWQ cares about). We accumulate moments and
Gram in fp64 when the VRAM budget allows, else fp32. The eigenstructure that
the paper's stability claims rest on (lambda_{k+1}, U_k) is sensitive to this,
so fp64 is preferred and the realized dtype is recorded for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


# ----------------------------------------------------------------------------
# VRAM budgeting: decide Gram dtype/device per layer instead of globally.
# ----------------------------------------------------------------------------

@dataclass
class GramPlan:
    """How to hold the d x d Gram for one layer."""
    feasible: bool          # can we afford a Gram at all (either dtype)?
    dtype: torch.dtype      # torch.float64 preferred, else torch.float32
    device: torch.device    # cuda preferred, else cpu
    bytes_needed: int
    note: str


def plan_gram(d: int,
              device: torch.device | str = "cuda",
              vram_fraction: float = 0.5,
              prefer_exact: bool = True,
              safety_bytes: int = 512 * 1024 * 1024) -> GramPlan:
    """
    Choose dtype/device for a d x d Gram.

    Strategy: prefer fp64 on GPU; if that exceeds the free-VRAM budget, try
    fp32 on GPU; if that still doesn't fit, fall back to CPU fp64 (slow but
    correct). If even a Gram is judged infeasible (caller wants the sketch),
    feasible=False is returned and the caller should use StreamingSketch.

    `vram_fraction` caps the Gram at that fraction of *free* VRAM, leaving
    room for the model forward pass and the eigh workspace (eigh needs a
    second d x d buffer, so a Gram occupying >~40% of free VRAM is risky).
    """
    device = torch.device(device)
    elems = d * d
    need64 = elems * 8
    need32 = elems * 4

    if device.type != "cuda" or not torch.cuda.is_available():
        # CPU: assume host RAM is ample relative to one d x d; prefer fp64.
        return GramPlan(True, torch.float64, torch.device("cpu"), need64,
                        "cpu fp64 (no cuda)")

    free, _total = torch.cuda.mem_get_info(device)
    budget = max(0, int(free * vram_fraction) - safety_bytes)
    # eigh needs roughly another d x d workspace; require room for ~2 buffers.
    if prefer_exact and 2 * need64 <= budget:
        return GramPlan(True, torch.float64, device, need64, "gpu fp64 (exact)")
    if 2 * need32 <= budget:
        return GramPlan(True, torch.float32, device, need32,
                        "gpu fp32 (budget too tight for fp64)")
    # GPU can't hold it with eigh headroom; try CPU fp64 before giving up.
    # (Caller may still prefer the sketch for speed; see collect.py.)
    return GramPlan(False, torch.float64, torch.device("cpu"), need64,
                    "gram infeasible on gpu within budget; use sketch or cpu")


# ----------------------------------------------------------------------------
# First/second moments: always cheap, always streamed.
# ----------------------------------------------------------------------------

class StreamingMoments:
    """Running E[x], E[x^2] over tokens. O(d) memory."""

    def __init__(self, d: int, device: torch.device | str = "cuda",
                 dtype: torch.dtype = torch.float64):
        self.d = d
        self.device = torch.device(device)
        self.dtype = dtype
        self.s1 = torch.zeros(d, dtype=dtype, device=self.device)
        self.s2 = torch.zeros(d, dtype=dtype, device=self.device)
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        # x: [..., d]; fold to [tokens, d] and accumulate, then drop x.
        xf = x.reshape(-1, x.shape[-1]).to(self.dtype)
        self.s1 += xf.sum(dim=0)
        self.s2 += (xf * xf).sum(dim=0)
        self.n += xf.shape[0]
        del xf

    @torch.no_grad()
    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, diag_H, diag_Sigma), each [d]."""
        if self.n == 0:
            z = torch.zeros(self.d, dtype=self.dtype, device=self.device)
            return z, z.clone(), z.clone()
        mu = self.s1 / self.n
        diag_H = self.s2 / self.n
        diag_Sigma = (diag_H - mu * mu).clamp_min(0)  # guard fp cancellation
        return mu, diag_H, diag_Sigma


# ----------------------------------------------------------------------------
# Backend A: streaming Gram (exact top-k via one eigh).
# ----------------------------------------------------------------------------

class StreamingGram:
    """
    Accumulate G = sum_t x_t x_t^T without ever forming the activation matrix.
    Combined with StreamingMoments for mu, yields Sigma = G/n - mu mu^T.

    Memory: one d x d buffer (dtype/device from GramPlan) + the moments.
    """

    def __init__(self, d: int, plan: GramPlan):
        self.d = d
        self.plan = plan
        self.G = torch.zeros(d, d, dtype=plan.dtype, device=plan.device)
        self.moments = StreamingMoments(
            d, device=plan.device, dtype=torch.float64
            if plan.dtype == torch.float64 else torch.float32)
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        xf = x.reshape(-1, x.shape[-1]).to(self.plan.dtype)
        if xf.device != self.G.device:
            xf = xf.to(self.G.device)
        # G += X^T X  -- the [tokens, d] block is transient; G is persistent.
        self.G += xf.t() @ xf
        self.moments.s1 += xf.sum(dim=0)
        self.moments.s2 += (xf * xf).sum(dim=0)
        self.moments.n += xf.shape[0]
        self.n += xf.shape[0]
        del xf

    @torch.no_grad()
    def finalize_covariance(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (mu, diag_H, Sigma). Sigma is d x d in the Gram's dtype/device.
        Caller is responsible for freeing Sigma after eigendecomposition.
        """
        mu, diag_H, _ = self.moments.finalize()
        H = self.G / max(1, self.n)
        Sigma = H - torch.outer(mu.to(H.dtype), mu.to(H.dtype))
        del H
        # symmetrize to kill fp asymmetry before eigh
        Sigma = 0.5 * (Sigma + Sigma.t())
        return mu, diag_H, Sigma

    def free(self) -> None:
        self.G = None
        self.moments.s1 = None
        self.moments.s2 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Backend B: streaming randomized sketch (never forms d x d).
# ----------------------------------------------------------------------------

class StreamingSketch:
    """
    Single-pass randomized range finder for the top-k eigenspace of Sigma,
    without ever materializing Sigma or the Gram.

    We maintain Y = (sum_t x_t x_t^T) Omega for a fixed Gaussian test matrix
    Omega in [d, r], r = k + oversample. Each block contributes
        Y += X_b^T (X_b Omega)
    where X_b Omega is [tokens, r] (small), so no d x d object ever exists.

    After the pass: Sigma Omega = Y/n - mu (mu^T Omega). Orthonormalize the
    range Q = qr(Sigma_Omega), form the small projected matrix
    B = Q^T Sigma Q (computed via a second cheap statistic, below), eigh B,
    and lift. To get Q^T Sigma Q without Sigma we also stream
        Z = (sum_t x_t x_t^T) Q  -- but Q is only known post-pass.
    So we use the standard one-pass Nystrom-style estimate:
        B ~= (Q^T Sigma_Omega)(Q^T Omega)^{-1}  ... numerically fragile.
    Instead we run a cheap SECOND statistic that IS one-pass-friendly:
    accumulate the small r x r and d x r pieces needed for a Nystrom
    eigenestimate. See finalize_eigs for the exact formulas.

    Memory: O(d r) for Y and Omega, plus O(r^2). For d=10000, k=16,
    oversample=8 -> r=24: ~ 2 * 10000 * 24 * 8 B ~ 3.8 MB. Negligible.

    Accuracy caveat: a single fixed-Omega pass is less accurate on the
    trailing retained directions than backend A. Validate against StreamingGram
    on a few layers (paper Section 6.5) before trusting it for stability claims.
    """

    def __init__(self, d: int, k: int, oversample: int = 8,
                 device: torch.device | str = "cuda",
                 dtype: torch.dtype = torch.float32,
                 seed: int = 0):
        self.d = d
        self.k = k
        self.r = k + oversample
        self.device = torch.device(device)
        self.dtype = dtype
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        self.Omega = torch.randn(d, self.r, generator=g,
                                 device=self.device, dtype=dtype)
        self.Y = torch.zeros(d, self.r, dtype=dtype, device=self.device)
        # second-moment pieces for the Nystrom lift:
        #   C = sum_t (Omega^T x_t)(Omega^T x_t)^T = Omega^T G Omega  [r, r]
        self.C = torch.zeros(self.r, self.r, dtype=dtype, device=self.device)
        self.moments = StreamingMoments(d, device=self.device, dtype=dtype)
        self.n = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        xf = x.reshape(-1, x.shape[-1]).to(self.dtype)
        if xf.device != self.Y.device:
            xf = xf.to(self.Y.device)
        xo = xf @ self.Omega                 # [tokens, r] -- small
        self.Y += xf.t() @ xo                # [d, r]
        self.C += xo.t() @ xo                # [r, r]
        self.moments.s1 += xf.sum(dim=0)
        self.moments.s2 += (xf * xf).sum(dim=0)
        self.moments.n += xf.shape[0]
        self.n += xf.shape[0]
        del xf, xo

    @torch.no_grad()
    def finalize_eigs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (mu, diag_H, U_k, Lam_k) with U_k in [d, k], Lam_k in [k].

        Lift: with G ~ n^{-1} sum x x^T, we have, using mu-centering,
            Sigma Omega = Y/n - mu (mu^T Omega)
            Omega^T Sigma Omega = C/n - (Omega^T mu)(mu^T Omega)
        Orthonormalize Q = qr(Sigma Omega). The projected operator is
            B = Q^T Sigma Q
        We do not have Sigma Q directly in one pass; we use the standard
        randomized-PCA approximation that, for a good range Q, the leading
        eigenpairs of  (Q^T Sigma Omega)(Q^T Omega)^+  approximate those of B.
        This is the one-pass HMT estimator; for moderate oversampling and a
        decaying spectrum it is accurate on the leading k. We then symmetrize.
        """
        n = max(1, self.n)
        mu, diag_H, _ = self.moments.finalize()
        Om_mu = self.Omega.t() @ mu          # [r]
        SigOm = self.Y / n - torch.outer(mu, Om_mu)          # [d, r]
        # range
        Q, _ = torch.linalg.qr(SigOm)        # [d, r]
        QtOm = Q.t() @ self.Omega            # [r, r]
        QtSigOm = Q.t() @ SigOm              # [r, r]
        # B ~= QtSigOm @ pinv(QtOm); symmetrize
        B = QtSigOm @ torch.linalg.pinv(QtOm)
        B = 0.5 * (B + B.t())
        evals, evecs = torch.linalg.eigh(B)  # ascending
        idx = torch.argsort(evals, descending=True)[: self.k]
        Lam_k = evals[idx].clamp_min(0)
        U_k = Q @ evecs[:, idx]              # [d, k]
        del SigOm, Q, QtOm, QtSigOm, B, evals, evecs
        return mu, diag_H, U_k, Lam_k

    def free(self) -> None:
        for attr in ("Omega", "Y", "C"):
            setattr(self, attr, None)
        self.moments.s1 = None
        self.moments.s2 = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
