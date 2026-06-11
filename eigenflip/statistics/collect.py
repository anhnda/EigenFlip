"""
Calibration collection: AWQ-style layer batching with streaming fold-in.

Key differences from the AWQ-XL template:
  * The hook does NOT append activations to a list. It folds each fire into a
    streaming accumulator (StreamingGram or StreamingSketch) and drops the
    activation immediately. Per-batch memory is therefore
        batch_size x (one accumulator), not batch_size x (all activations).
  * Backend per layer is chosen against a VRAM budget (plan_gram): exact fp64
    Gram if it fits, else fp32 Gram, else the randomized sketch.
  * lm_head is skipped by default.
  * Aggressive freeing: accumulators are finalized to LayerStats and dropped
    as soon as a layer is quantized.

This module only COLLECTS statistics and builds LayerStats. Encoding/writeback
is the runner's job.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from .accumulators import (StreamingGram, StreamingSketch, StreamingMoments,
                           plan_gram, GramPlan)
from .trust_region import (LayerStats, stats_from_gram, stats_from_sketch,
                           stats_from_moments)


def is_lm_head(name: str) -> bool:
    return name.lower().endswith("lm_head") or "lm_head" in name.lower()


class StatsCollector:
    """
    Collects LayerStats for a batch of Linear layers via one calibration pass.

    backend selection:
      k == 0                -> moments only (no covariance)
      k > 0, gram feasible  -> StreamingGram (exact eigh)
      k > 0, gram infeasible-> StreamingSketch (O(dk))
    `force_backend` overrides: 'gram' | 'sketch' | 'moments'.
    """

    def __init__(self, model, tokenizer, device="cuda", k: int = 16,
                 eps: float = 1e-6, oversample: int = 8,
                 vram_fraction: float = 0.4, prefer_exact: bool = True,
                 force_backend: Optional[str] = None,
                 max_tokens_per_sample: Optional[int] = None,
                 keep_sigma_for: Optional[set] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.k = k
        self.eps = eps
        self.oversample = oversample
        self.vram_fraction = vram_fraction
        self.prefer_exact = prefer_exact
        self.force_backend = force_backend
        self.max_tokens_per_sample = max_tokens_per_sample
        self.keep_sigma_for = keep_sigma_for or set()
        self.accumulators: dict = {}
        self._kinds: dict = {}

    # ---- backend choice --------------------------------------------------

    def _make_accumulator(self, name: str, d: int):
        if self.k == 0 or self.force_backend == "moments":
            self._kinds[name] = "moments"
            return StreamingMoments(d, device=self.device,
                                    dtype=torch.float64)
        if self.force_backend == "sketch":
            self._kinds[name] = "sketch"
            return StreamingSketch(d, self.k, self.oversample,
                                   device=self.device, dtype=torch.float32)
        if self.force_backend == "gram":
            plan = plan_gram(d, self.device, self.vram_fraction,
                             self.prefer_exact)
            plan.feasible = True  # forced
            self._kinds[name] = "gram"
            return StreamingGram(d, plan)
        # auto
        plan = plan_gram(d, self.device, self.vram_fraction, self.prefer_exact)
        if plan.feasible:
            self._kinds[name] = "gram"
            return StreamingGram(d, plan)
        self._kinds[name] = "sketch"
        return StreamingSketch(d, self.k, self.oversample,
                               device=self.device, dtype=torch.float32)

    # ---- hook ------------------------------------------------------------

    def _hook(self, name):
        def hook(_m, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            if (self.max_tokens_per_sample and x.dim() == 3
                    and x.shape[1] > self.max_tokens_per_sample):
                seq = x.shape[1]
                idx = torch.randperm(seq, device=x.device)[:self.max_tokens_per_sample]
                idx = idx.sort()[0]
                x = x[:, idx, :]
            self.accumulators[name].update(x)
        return hook

    # ---- one batch -------------------------------------------------------

    @torch.no_grad()
    def collect_batch(self, layer_batch, calib_texts, n_samples,
                      max_length=512):
        """
        layer_batch: list of (name, module) Linear layers.
        Returns {name: LayerStats}. Accumulators are freed before return.
        """
        self.accumulators = {}
        self._kinds = {}
        for name, module in layer_batch:
            d = module.weight.shape[1]
            self.accumulators[name] = self._make_accumulator(name, d)

        handles = [m.register_forward_hook(self._hook(n))
                   for n, m in layer_batch]
        for i, sample in enumerate(calib_texts[:n_samples]):
            try:
                # sample may be a text string OR a pre-tokenized [1, L] tensor
                # (calibration_utils with return_tensors=True). Tensors preserve
                # the full sliced seqlen (e.g. 2048) instead of truncating to
                # max_length, keeping n/d honest -- preferred for encoder stats.
                if torch.is_tensor(sample):
                    input_ids = sample.to(self.device)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    self.model(input_ids=input_ids, use_cache=False,
                               return_dict=True)
                    del input_ids
                else:
                    enc = self.tokenizer(sample, return_tensors="pt",
                                         truncation=True, max_length=max_length)
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    self.model(**enc, use_cache=False, return_dict=True)
                    del enc
                if (i + 1) % 16 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                continue
        for h in handles:
            h.remove()

        stats = {}
        for name, _ in layer_batch:
            acc = self.accumulators[name]
            kind = self._kinds[name]
            keep_sigma = name in self.keep_sigma_for
            if kind == "moments":
                stats[name] = stats_from_moments(acc, self.eps)
            elif kind == "gram":
                stats[name] = stats_from_gram(acc, self.k, self.eps,
                                              keep_sigma=keep_sigma)
            else:
                stats[name] = stats_from_sketch(acc, self.eps)
            acc.free()
            del acc
        self.accumulators = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return stats
