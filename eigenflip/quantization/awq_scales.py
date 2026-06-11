"""
AWQ scale extraction.

The `AWQ` base in the comparison needs per-input-channel scales s[j] for each
layer. These come from the AWQ grid search (your awq_*_xl.py): for each layer,
s = salience^alpha with the alpha that minimizes reconstruction MSE, where
salience[j] = E[X[:,j]^2] (L2). We re-implement only the *scale selection* here
(not a new quantizer), reading activation second moments we already stream.

Two ways to obtain AWQ scales:

  (A) Reuse your existing awq_*_xl.py run: it stores `layer_scales[name]['scales']`.
      Pass that dict straight into the runner as awq_scales -- no recompute.

  (B) compute_awq_scales(): a self-contained grid search using the SAME
      streaming E[X^2] salience plus a small calibration activation sample for
      the MSE objective. Use this when you don't want to run the full awq script.

For the paper's base-blocking to be clean, prefer (A) so the AWQ base in
EigenFlip's table is byte-identical to your standalone AWQ baseline.
"""

from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def _groupwise_asym_quant(W, bits, group_size):
    C, in_f = W.shape
    n_groups = (in_f + group_size - 1) // group_size
    pin = n_groups * group_size
    if pin > in_f:
        Wp = torch.zeros(C, pin, device=W.device, dtype=W.dtype)
        Wp[:, :in_f] = W
    else:
        Wp = W
    Wg = Wp.reshape(C, n_groups, group_size)
    wmin = Wg.min(2, keepdim=True)[0]
    wmax = Wg.max(2, keepdim=True)[0]
    mi = 2 ** bits - 1
    sc = ((wmax - wmin) / mi).clamp_min(1e-8)
    zp = torch.round(-wmin / sc).clamp(0, mi)
    q = torch.round(Wg / sc + zp).clamp(0, mi)
    dq = ((q - zp) * sc).reshape(C, pin)
    return dq[:, :in_f] if pin > in_f else dq


@torch.no_grad()
def compute_awq_scales(W: torch.Tensor, salience_l2: torch.Tensor,
                       X_sample: torch.Tensor, bits: int, group_size: int,
                       n_grid: int = 20) -> tuple[torch.Tensor, float, float]:
    """
    AWQ grid search for one layer.

    W           [C, d]              layer weight
    salience_l2 [d]                 E[X[:,j]^2] (from streaming s2 / n)
    X_sample    [m, d]              a small activation sample for the MSE objective
    Returns (scales [d], best_alpha, best_error).
    """
    device = W.device
    dtype = W.dtype
    sal = salience_l2.to(device=device, dtype=torch.float32)
    Xs = X_sample.to(device=device, dtype=dtype)
    Y_orig = Xs @ W.t()

    best_err = float("inf"); best_alpha = 0.0
    best_scales = torch.ones(W.shape[1], device=device, dtype=dtype)
    for g in range(n_grid + 1):
        alpha = g / n_grid
        scales = sal.pow(alpha).clamp_min(1e-5).to(dtype)
        W_scaled = W * scales.unsqueeze(0)
        W_q = _groupwise_asym_quant(W_scaled, bits, group_size)
        X_comp = Xs / scales.unsqueeze(0)
        Y_q = X_comp @ W_q.t()
        err = (Y_orig - Y_q).pow(2).mean().item()
        if err < best_err:
            best_err, best_alpha = err, alpha
            best_scales = scales.clone()
        del W_scaled, W_q, X_comp, Y_q
    del Y_orig, Xs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_scales, best_alpha, best_err


def scales_from_awq_run(layer_scales: dict) -> dict:
    """
    Adapter for path (A): convert an awq_*_xl.py `layer_scales` dict into the
    {name: scales_tensor} mapping the runner expects.
    """
    out = {}
    for name, info in layer_scales.items():
        s = info.get("scales")
        if s is not None:
            out[name] = s if torch.is_tensor(s) else torch.as_tensor(s)
    return out
