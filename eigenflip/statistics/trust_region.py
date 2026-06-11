"""
Trust-region statistics for the encoder.

LayerStats holds exactly what the encoders consume and -- critically -- never
materializes the d x d surrogate H~_{k,eps} = D_{k,eps} + V V^T. The deployed
EigenFlip Solve path allocates only:
    D      [d]            residual diagonal (floored)
    V      [d, k+1]       = [mu_hat | U_k Lam_k^{1/2}]
    M      [k+1, k+1]     capacitance (built inside the encoder)
and per-row (k+1) accumulators. `.Sigma` is kept ONLY when an exact backend
produced it AND the dense reference harness (Section 6.5) asks for it; the
deployed encoder asserts Sigma is None.

Rung map (paper Eq. 6):
    rung 0 (RTN)            : needs diag only
    rung 1 (CLC, k=0 Solve) : needs diag + mu_hat              -> V has 1 col
    rung 2 (EigenFlip[,Solve]): + top-k eig(Sigma)             -> V has k+1 cols
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .james_stein import james_stein_mean


@dataclass
class LayerStats:
    d: int
    mu_hat: torch.Tensor                      # [d], JS-shrunk
    diag_H: torch.Tensor                      # [d]
    diag_Sigma: torch.Tensor                  # [d]
    U_k: Optional[torch.Tensor] = None        # [d, k] or None (rung<2)
    Lam_k: Optional[torch.Tensor] = None      # [k]   or None
    eps: float = 1e-6
    # filled by build():
    D: torch.Tensor = field(default=None, repr=False)      # [d] floored residual diag
    V: torch.Tensor = field(default=None, repr=False)      # [d, k+1]
    Sigma: Optional[torch.Tensor] = field(default=None, repr=False)  # dense, harness-only
    backend: str = "unknown"

    @property
    def k(self) -> int:
        return 0 if self.U_k is None else self.U_k.shape[1]

    @torch.no_grad()
    def build(self) -> "LayerStats":
        """
        Construct D_{k,eps} and V. Definition 1 of the paper:
            D_k      = diag(Sigma - U_k Lam_k U_k^T)
            D_{k,eps}= max(D_k, eps * diag(H))    elementwise
            V        = [mu_hat | U_k Lam_k^{1/2}]
        D_{k,eps} >= eps*diag(H) > 0 so the Woodbury capacitance is PD.
        """
        device = self.diag_Sigma.device
        dtype = self.diag_Sigma.dtype
        floor = self.eps * self.diag_H

        if self.U_k is None or self.k == 0:
            Dk = self.diag_Sigma
            V = self.mu_hat.reshape(-1, 1).to(device=device, dtype=dtype)
        else:
            U = self.U_k.to(device=device, dtype=dtype)
            Lam = self.Lam_k.to(device=device, dtype=dtype).clamp_min(0)
            # diag(U Lam U^T) = sum_j Lam_j U[:,j]^2  -- no d x d formed
            diag_lowrank = (U * U) @ Lam                      # [d]
            Dk = self.diag_Sigma - diag_lowrank
            V_eig = U * Lam.sqrt().unsqueeze(0)               # [d, k]
            V = torch.cat([self.mu_hat.reshape(-1, 1).to(V_eig), V_eig], dim=1)
            del diag_lowrank, U, Lam, V_eig

        self.D = torch.maximum(Dk, floor)
        self.V = V
        return self

    @torch.no_grad()
    def free_sigma(self) -> None:
        if self.Sigma is not None:
            self.Sigma = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# Builders from each accumulator backend.
# ----------------------------------------------------------------------------

@torch.no_grad()
def stats_from_gram(gram, k: int, eps: float = 1e-6,
                    keep_sigma: bool = False,
                    eig_device: Optional[torch.device] = None) -> LayerStats:
    """
    Build LayerStats from a StreamingGram via one exact eigh.

    k=0 -> no eigendecomposition (rung 0/1). Otherwise eigh(Sigma) and take
    the top-k. Sigma is freed unless keep_sigma (dense-reference harness).
    For heavy layers, eig_device can move the eigh to CPU if GPU is tight.
    """
    mu, diag_H, Sigma = gram.finalize_covariance()
    diag_Sigma = torch.diagonal(Sigma).clone()
    mu_js = james_stein_mean(mu)

    U_k = Lam_k = None
    if k > 0:
        S = Sigma if eig_device is None else Sigma.to(eig_device)
        evals, evecs = torch.linalg.eigh(S)          # ascending
        topk = torch.argsort(evals, descending=True)[:k]
        Lam_k = evals[topk].clamp_min(0).to(Sigma.device)
        U_k = evecs[:, topk].to(Sigma.device)
        del evals, evecs
        if S is not Sigma:
            del S
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    st = LayerStats(
        d=gram.d, mu_hat=mu_js, diag_H=diag_H, diag_Sigma=diag_Sigma,
        U_k=U_k, Lam_k=Lam_k, eps=eps,
        Sigma=Sigma if keep_sigma else None, backend="gram",
    ).build()
    if not keep_sigma:
        del Sigma
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return st


@torch.no_grad()
def stats_from_sketch(sketch, eps: float = 1e-6) -> LayerStats:
    """Build LayerStats from a StreamingSketch (never forms Sigma)."""
    mu, diag_H, U_k, Lam_k = sketch.finalize_eigs()
    diag_Sigma = (diag_H - mu * mu).clamp_min(0)
    mu_js = james_stein_mean(mu)
    return LayerStats(
        d=sketch.d, mu_hat=mu_js, diag_H=diag_H, diag_Sigma=diag_Sigma,
        U_k=U_k, Lam_k=Lam_k, eps=eps, Sigma=None, backend="sketch",
    ).build()


@torch.no_grad()
def stats_from_moments(moments, eps: float = 1e-6) -> LayerStats:
    """Rung 0/1 only: diag + mu, no covariance backend at all."""
    mu, diag_H, diag_Sigma = moments.finalize()
    mu_js = james_stein_mean(mu)
    return LayerStats(
        d=moments.d, mu_hat=mu_js, diag_H=diag_H, diag_Sigma=diag_Sigma,
        U_k=None, Lam_k=None, eps=eps, Sigma=None, backend="moments",
    ).build()
