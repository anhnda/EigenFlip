"""
collect_fast.py -- AWQ-STYLE batched collection. The FAST path.

Copies the exact structure of the fast AWQ XL quantizer's
quantize_model_sequential:
  * batch `layer_batch_size` Linear layers
  * register hooks on all of them
  * run calibration data through the model ONCE for the whole batch
  * quantize each layer in the batch
  * move to next batch
=> 224 layers / 16 = ~14 calibration passes total. NOT one-per-layer.

The ONLY change vs AWQ: the hook does NOT store activations on CPU
(`self.activation_data[name].append(...)` was 8.6 GB of RAM for d=4096).
Instead it FOLDS each fire into a streaming accumulator and drops the tensor:
  * rtn  -> nothing
  * clc  -> mean E[X] only            (O(d) running sum, no Gram)
  * eigenflip / solve / gptq / shr   -> H = E[xx^T]  (streaming Gram, fp32
                                        matmul into fp64 buffer)

Activations are never materialized; the giant list is gone.
"""

from __future__ import annotations

import gc
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .trust_region import LayerStats, james_stein_mean


def is_lm_head(name: str) -> bool:
    return name.lower().endswith("lm_head") or "lm_head" in name.lower()


# ---------------------------------------------------------------------------
# Streaming accumulator: mean-only (O(d)) or full Gram (d x d). fp32 matmul,
# fp64 buffer. Never stores activations.
# ---------------------------------------------------------------------------

class _Acc:
    def __init__(self, d, need_H, device):
        self.d = d
        self.need_H = need_H
        self.s1 = torch.zeros(d, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(d, dtype=torch.float64, device=device)
        self.n = 0
        self.G = torch.zeros(d, d, dtype=torch.float64, device=device) if need_H else None

    @torch.no_grad()
    def add(self, x):
        xf = x.reshape(-1, x.shape[-1]).float()
        if xf.device != self.s1.device:
            xf = xf.to(self.s1.device, non_blocking=True)
        self.s1 += xf.sum(0).double()
        self.s2 += (xf * xf).sum(0).double()
        self.n += xf.shape[0]
        if self.G is not None:
            self.G += (xf.t() @ xf).double()
        del xf

    @torch.no_grad()
    def to_stats(self, k, eps, keep_sigma, eig_device):
        n = max(1, self.n)
        mu = self.s1 / n
        diag_H = self.s2 / n
        if not self.need_H:
            diag_Sigma = (diag_H - mu * mu).clamp_min(0)
            return LayerStats(d=self.d, mu_hat=james_stein_mean(mu),
                              diag_H=diag_H, diag_Sigma=diag_Sigma,
                              U_k=None, Lam_k=None, eps=eps,
                              Sigma=None, backend="mean").build()
        Sigma = self.G / n - torch.outer(mu, mu)
        Sigma = 0.5 * (Sigma + Sigma.t())
        diag_Sigma = torch.diagonal(Sigma).clone()
        U_k = Lam_k = None
        if k > 0:
            S = Sigma if eig_device is None else Sigma.to(eig_device)
            evals, evecs = torch.linalg.eigh(S)
            topk = torch.argsort(evals, descending=True)[:k]
            Lam_k = evals[topk].clamp_min(0).to(Sigma.device)
            U_k = evecs[:, topk].to(Sigma.device)
            del evals, evecs
            if S is not Sigma:
                del S
        st = LayerStats(d=self.d, mu_hat=james_stein_mean(mu),
                        diag_H=diag_H, diag_Sigma=diag_Sigma,
                        U_k=U_k, Lam_k=Lam_k, eps=eps,
                        Sigma=Sigma if keep_sigma else None, backend="gram").build()
        if not keep_sigma:
            del Sigma
        return st

    def free(self):
        self.s1 = self.s2 = self.G = None


# ---------------------------------------------------------------------------
# AWQ-style batched driver.
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_and_encode_awq_style(
    model, tokenizer, calib, device, *,
    need_H, k, eps, callback,
    layer_batch_size=16,
    keep_sigma=False,
    skip_lm_head=True,
    eig_on_cpu=False,
    max_length=2048,
):
    """
    calib: list of pre-tokenized [1,L] tensors OR text strings.
    callback(name, module, LayerStats) encodes + writes module.weight.
    """
    eig_device = torch.device("cpu") if eig_on_cpu else None

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and not (skip_lm_head and is_lm_head(n))]
    n_layers = len(layers)

    # KEY SPEED FIX: mean-only (rtn/clc) costs O(d) per layer -> hook ALL layers
    # at once and calibrate in ONE pass. Batching only matters for need_H (the
    # d x d Gram is what costs RAM); with mean-only, batching just forces the
    # model to be re-run from scratch for every batch (the slow bug). So when
    # need_H is False we override the batch size to cover all layers.
    if not need_H:
        layer_batch_size = n_layers

    n_batches = (n_layers + layer_batch_size - 1) // layer_batch_size
    print(f"  {n_layers} layers, {n_batches} batch(es) of {layer_batch_size}, "
          f"need_H={need_H}"
          + ("  [mean-only: single pass]" if not need_H else ""))

    for bi in range(n_batches):
        s = bi * layer_batch_size
        e = min(s + layer_batch_size, n_layers)
        batch = layers[s:e]
        print(f"\n[batch {bi+1}/{n_batches}] layers {s}-{e-1}")

        # one accumulator per layer in batch
        accs = {n: _Acc(m.weight.shape[1], need_H, device) for n, m in batch}

        def mk_hook(nm):
            def hook(_m, inp, _o):
                x = inp[0] if isinstance(inp, tuple) else inp
                accs[nm].add(x)
            return hook

        handles = [m.register_forward_hook(mk_hook(n)) for n, m in batch]

        # ONE calibration pass for the whole batch (AWQ-style)
        for sample in tqdm(calib, desc="  calib", leave=False):
            try:
                if torch.is_tensor(sample):
                    ids = sample.to(device, non_blocking=True)
                    if ids.dim() == 1:
                        ids = ids.unsqueeze(0)
                    model(input_ids=ids, use_cache=False)
                    del ids
                else:
                    enc = tokenizer(sample, return_tensors="pt",
                                    truncation=True, max_length=max_length)
                    enc = {kk: vv.to(device, non_blocking=True)
                           for kk, vv in enc.items()}
                    model(**enc, use_cache=False)
                    del enc
            except Exception:
                continue

        for h in handles:
            h.remove()

        # encode each layer in the batch (THIS is the quantize step)
        from tqdm import tqdm as _tq
        for n, m in _tq(batch, desc="  quantize", leave=False):
            st = accs[n].to_stats(k, eps, keep_sigma, eig_device)
            callback(n, m, st)
            st.free_sigma()
            accs[n].free()
            del st
        print(f"  quantized {len(batch)} layers")
        accs.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
