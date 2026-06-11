"""
sequential.py -- FAST block-paged collection + encoding, GPTQ-style.

Replaces the slow per-batch collector. ONE calibration pass total:
  * Catcher captures the first decoder block's input (all kwargs generically).
  * Then block-by-block: page block to GPU, run nsamples forwards THROUGH THAT
    BLOCK ONLY, stream stats per sublayer, encode, then  inps, outs = outs, inps
    feeds this block's output to the next. The full model is never re-run.

CONDITIONAL STREAMING -- collect only what the chosen encoder needs:
    rtn               -> nothing (no calibration needed at all)
    clc               -> mean E[X] only          (O(d) running sum)
    eigenflip / solve -> H = E[xx^T]             (streaming Gram, GPTQ-style)
    gptq / shrinkage  -> H = E[xx^T]             (streaming Gram)
Activations are NEVER stored; H is built incrementally in add_batch exactly
like the official GPTQ (H *= n/(n+t); H += sqrt(2/n) x x^T -> here we use the
plain Gram sum and normalize at the end, equivalent up to the 2/n scaling
which we fold into finalize).

The streamed H feeds the EXISTING, validated builders (stats_from_gram /
stats_from_moments) and the EXISTING encoders -- none of that code changes.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .trust_region import LayerStats, james_stein_mean


# ---------------------------------------------------------------------------
# Per-sublayer streaming accumulator. Two modes, picked by need_H.
# ---------------------------------------------------------------------------

class _SublayerStat:
    """
    Streams either mean-only (O(d)) or full Gram (d x d), fp32 matmul into an
    fp64 buffer. Never stores activations.
    """
    def __init__(self, d: int, need_H: bool, device, gram_dtype=torch.float64):
        self.d = d
        self.need_H = need_H
        self.device = device
        self.s1 = torch.zeros(d, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(d, dtype=torch.float64, device=device)
        self.n = 0
        self.G = (torch.zeros(d, d, dtype=gram_dtype, device=device)
                  if need_H else None)

    @torch.no_grad()
    def add(self, x: torch.Tensor):
        # x: [..., d] -> [tokens, d]
        xf = x.reshape(-1, x.shape[-1]).float()
        if xf.device != self.s1.device:
            xf = xf.to(self.s1.device, non_blocking=True)
        self.s1 += xf.sum(0).double()
        self.s2 += (xf * xf).sum(0).double()
        self.n += xf.shape[0]
        if self.G is not None:
            block = xf.t() @ xf                       # fp32 matmul (fast)
            self.G += block.to(self.G.dtype)
            del block
        del xf

    @torch.no_grad()
    def finalize_mean_diag(self):
        n = max(1, self.n)
        mu = self.s1 / n
        diag_H = self.s2 / n
        diag_Sigma = (diag_H - mu * mu).clamp_min(0)
        return mu, diag_H, diag_Sigma

    @torch.no_grad()
    def finalize_sigma(self):
        n = max(1, self.n)
        mu = self.s1 / n
        diag_H = self.s2 / n
        H = self.G / n
        Sigma = H - torch.outer(mu, mu)
        Sigma = 0.5 * (Sigma + Sigma.t())
        del H
        return mu, diag_H, Sigma

    def free(self):
        self.s1 = self.s2 = self.G = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Build LayerStats from a finalized accumulator (reuses validated math).
# ---------------------------------------------------------------------------

@torch.no_grad()
def _stats_from_sublayer(stat: _SublayerStat, k: int, eps: float,
                         keep_sigma: bool, eig_device=None) -> LayerStats:
    if not stat.need_H:
        mu, diag_H, diag_Sigma = stat.finalize_mean_diag()
        return LayerStats(d=stat.d, mu_hat=james_stein_mean(mu),
                          diag_H=diag_H, diag_Sigma=diag_Sigma,
                          U_k=None, Lam_k=None, eps=eps,
                          Sigma=None, backend="mean").build()
    mu, diag_H, Sigma = stat.finalize_sigma()
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
    st = LayerStats(d=stat.d, mu_hat=james_stein_mean(mu),
                    diag_H=diag_H, diag_Sigma=diag_Sigma,
                    U_k=U_k, Lam_k=Lam_k, eps=eps,
                    Sigma=Sigma if keep_sigma else None, backend="gram").build()
    if not keep_sigma:
        del Sigma
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return st


def find_linear(module, name=""):
    if isinstance(module, nn.Linear):
        return {name: module}
    res = {}
    for n, child in module.named_children():
        res.update(find_linear(child, name + "." + n if name else n))
    return res


# ---------------------------------------------------------------------------
# The fast driver. callback(name, module, stats) does the encoding/writeback.
# ---------------------------------------------------------------------------

@torch.no_grad()
def sequential_collect_and_encode(
    model, dataloader, device, *,
    need_H: bool, k: int, eps: float,
    callback,                          # callback(layer_key, module, LayerStats)
    keep_sigma: bool = False,
    true_sequential: bool = True,
    skip_lm_head: bool = True,
    eig_on_cpu: bool = False,
):
    """
    GPTQ-style single-pass. dataloader: iterable of (input_ids[1,L],) tensors.
    `need_H=False` streams mean-only (rtn/clc); True streams Gram (eigenflip/
    gptq/shrinkage). `callback` receives finalized LayerStats per sublayer and
    is responsible for encoding + writing module.weight.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(device)
    if getattr(model.model, "rotary_emb", None) is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    samples = list(dataloader)
    nsamples = len(samples)
    seqlen = samples[0][0].shape[1]
    hidden = model.config.hidden_size
    inps = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device=device)
    cache = {"i": 0, "kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.detach()
            cache["i"] += 1
            cache["kwargs"] = {kk: (vv.detach() if isinstance(vv, torch.Tensor)
                                    else vv) for kk, vv in kwargs.items()}
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in samples:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.cpu()
    if getattr(model.model, "rotary_emb", None) is not None:
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = {kk: vv for kk, vv in cache["kwargs"].items()
                    if kk != "use_cache"}
    cache.clear()
    gc.collect()

    eig_device = torch.device("cpu") if eig_on_cpu else None

    for li in range(len(layers)):
        layer = layers[li].to(device)
        full = find_linear(layer)
        if skip_lm_head:
            full = {n: m for n, m in full.items() if "lm_head" not in n.lower()}

        if true_sequential:
            groups = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            groups = [list(full.keys())]

        for names in groups:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue
            stats = {n: _SublayerStat(m.weight.shape[1], need_H, device)
                     for n, m in subset.items()}

            def mk_hook(nm):
                def hook(_m, inp, _out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    stats[nm].add(x)
                return hook

            handles = [subset[n].register_forward_hook(mk_hook(n)) for n in subset]
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
            for h in handles:
                h.remove()

            for n in subset:
                key = f"model.layers.{li}.{n}"
                st = _stats_from_sublayer(stats[n], k, eps, keep_sigma, eig_device)
                callback(key, subset[n], st)
                st.free_sigma()
                stats[n].free()
                del st
            del stats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # feed this block's output forward (quantized weights now in place)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]

        layers[li] = layer.cpu()
        del layer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(f"  block {li+1}/{len(layers)} done")

    model.config.use_cache = use_cache
