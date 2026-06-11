"""
Encoder interface. An encoder takes a base-produced IntegerQuantizedTensorState
plus LayerStats (trust region) and returns corrected dequantized weights ready
to write back into the module, plus a diagnostics dict.

This is the SAME contract GPTQ's _run_post_correction already used, generalized
so RTN/AWQ bases feed it too. Encoders never form a d x d matrix unless they are
the explicit dense reference.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ..quantization.state import IntegerQuantizedTensorState
from ..statistics.trust_region import LayerStats


class Encoder(Protocol):
    name: str

    def apply(self, state: IntegerQuantizedTensorState,
              stats: LayerStats) -> tuple[torch.Tensor, dict]:
        ...


class IdentityEncoder:
    """'base only' -- just dequantize the base codes, no correction."""
    name = "none"

    @torch.no_grad()
    def apply(self, state, stats):
        return state.dequantize().to(state.original_dtype), {"encoder": "none"}
