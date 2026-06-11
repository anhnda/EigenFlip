"""
Runner: produce the base x encoder comparison matrix of the paper (Table 6).

For each (base, encoder) cell:
  1. Collect calibration statistics ONCE per layer batch (StatsCollector),
     reused across encoders so the encoder is the only varying factor
     (paper Section 6.6: base is a blocking variable).
  2. For each base in {rtn, awq}: produce the IntegerQuantizedTensorState once.
  3. For each encoder: apply to the SAME state + stats, write the corrected
     weights into a fresh copy of the layer, save / evaluate.

This driver focuses on the per-layer mechanics; orchestration over the whole
model (load, iterate batches, save each base x encoder model variant) is left
to a thin script that the user runs. We expose `quantize_layer_variants` as the
reusable unit and `EIGEN_REGISTRY` as the encoder set.

Important: the AWQ base needs per-input-channel AWQ scales. Those come from a
separate AWQ scale search (your existing awq_*_xl grid search), passed in as
`awq_scales[name]`. The runner does not re-implement the AWQ search; it consumes
its output so AWQ+encoder cells use AWQ scales, not RTN min/max -- otherwise the
base comparison is confounded.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats
from ..encoders.base_encoder import IdentityEncoder
from ..encoders.flip import make_clc, make_eigenflip
from ..encoders.eigenflip_solve import EigenFlipSolve
from ..encoders.dense_reference import DenseGPTQ, DenseSurrogateGPTQ


def default_registry(clc_knee=-10.0, clc_budget=1.0,
                     ef_knee=-10.0, ef_budget=1.0,
                     gptq_damp=0.01, solve_order="leverage"):
    """
    The encoder set for the comparison. 'none' = base only.
    clc/eigenflip defaults: no-knee (-10), full budget (1.0), per your template.
    """
    return {
        "none": IdentityEncoder(),
        "clc": make_clc(clc_knee, clc_budget, use_knee=False),
        "eigenflip": make_eigenflip(ef_knee, ef_budget, use_knee=False),
        "eigenflip_solve": EigenFlipSolve(order=solve_order),
        "gptq": DenseGPTQ(damp=gptq_damp),
        # 'dense_surrogate_gptq' is the validation reference, not a table row;
        # add it explicitly when running Section 6.5 checks.
    }


def make_base_state(W: torch.Tensor, base: str, bits: int, group_size: int,
                    awq_scales: Optional[torch.Tensor] = None
                    ) -> IntegerQuantizedTensorState:
    if base == "rtn":
        return IntegerQuantizedTensorState.from_rtn(W, bits, group_size)
    if base == "awq":
        if awq_scales is None:
            raise ValueError("AWQ base requires awq_scales for this layer.")
        return IntegerQuantizedTensorState.from_awq(W, awq_scales, bits, group_size)
    raise ValueError(f"unknown base {base!r}")


@torch.no_grad()
def quantize_layer_variants(W: torch.Tensor, stats: LayerStats,
                            bases: list[str], encoders: dict,
                            bits: int, group_size: int,
                            awq_scales: Optional[torch.Tensor] = None
                            ) -> dict:
    """
    Return {(base, encoder_name): (corrected_W [C,d], info)}.

    The base state is built once per base and shared across encoders. The
    encoder that needs a materialized Sigma (DenseGPTQ) requires stats.Sigma to
    be present (collect with keep_sigma for that layer); otherwise it asserts.
    """
    out = {}
    for base in bases:
        state = make_base_state(W, base, bits, group_size, awq_scales)
        for enc_name, enc in encoders.items():
            corrected, info = enc.apply(state, stats)
            out[(base, enc_name)] = (corrected, {**info, "base": base})
        # state is cheap to rebuild; drop before next base
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out
