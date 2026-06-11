"""
IntegerQuantizedTensorState: the base-produced integer state that every
encoder consumes. Normalized so an RTN or AWQ base can feed the same encoders
as GPTQ.

Layout (group-wise asymmetric, the common case):
    float_weights   [C, d]      original W (pre-quant), padding-stripped view
    integer_weights [C, d]      rounded codes in [0, max_int]
    pre_round       [C, d]      W/scale + zp  (the continuous pre-round target)
    scale           [C, d]      per-element scale (replicated within group)
    zero_point      [C, d]      per-element zero point (replicated within group)
    max_int, min_int            code range
    in_features, padded_in_features
    group_size                  (informational; scale/zp already expanded)

We store scale/zp already expanded to [C, d] (GPTQ-style) so encoders never
need to know the grouping -- they read scale[i,j] directly. dequantize() and
the per-coordinate helpers below are the only quant arithmetic the encoders
touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class IntegerQuantizedTensorState:
    float_weights: torch.Tensor
    pre_round: torch.Tensor
    integer_weights: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    max_int: int
    min_int: int
    in_features: int
    padded_in_features: int
    original_dtype: torch.dtype
    group_size: int = -1

    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """(q - zp) * scale, stripped to in_features."""
        W = (self.integer_weights - self.zero_point) * self.scale
        if self.padded_in_features > self.in_features:
            W = W[:, : self.in_features]
        return W

    @torch.no_grad()
    def clone_codes(self) -> torch.Tensor:
        return self.integer_weights.clone()

    # ---- constructors from each base -------------------------------------

    @staticmethod
    @torch.no_grad()
    def _expand_group(t_grouped: torch.Tensor, group_size: int,
                      padded_in: int) -> torch.Tensor:
        # t_grouped: [C, n_groups, 1] -> [C, padded_in]
        C, n_groups, _ = t_grouped.shape
        return t_grouped.repeat(1, 1, group_size).reshape(C, padded_in)

    @classmethod
    @torch.no_grad()
    def from_rtn(cls, W: torch.Tensor, bits: int, group_size: int
                 ) -> "IntegerQuantizedTensorState":
        """
        Group-wise asymmetric RTN. Produces the full integer state so any
        encoder (CLC/EigenFlip/Solve/gptq-encoder) can run on top.
        """
        C, in_features = W.shape
        device, dtype = W.device, W.dtype
        n_groups = (in_features + group_size - 1) // group_size
        padded_in = n_groups * group_size

        if padded_in > in_features:
            Wp = torch.zeros(C, padded_in, device=device, dtype=dtype)
            Wp[:, :in_features] = W
        else:
            Wp = W

        Wg = Wp.reshape(C, n_groups, group_size)
        wmin = Wg.min(dim=2, keepdim=True)[0]
        wmax = Wg.max(dim=2, keepdim=True)[0]
        max_int = 2 ** bits - 1
        scale_g = ((wmax - wmin) / max_int).clamp_min(1e-8)
        zp_g = torch.round(-wmin / scale_g).clamp(0, max_int)

        scale = cls._expand_group(scale_g, group_size, padded_in)
        zp = cls._expand_group(zp_g, group_size, padded_in)
        pre_round = Wp / scale + zp
        integer = torch.round(pre_round).clamp(0, max_int)

        st = cls(
            float_weights=Wp, pre_round=pre_round, integer_weights=integer,
            scale=scale, zero_point=zp, max_int=max_int, min_int=0,
            in_features=in_features, padded_in_features=padded_in,
            original_dtype=dtype, group_size=group_size,
        )
        del Wg, wmin, wmax, scale_g, zp_g
        return st

    @classmethod
    @torch.no_grad()
    def from_awq(cls, W: torch.Tensor, awq_scales: torch.Tensor,
                 bits: int, group_size: int) -> "IntegerQuantizedTensorState":
        """
        AWQ base: per-input-channel scales `awq_scales` [in_features] already
        chosen by the AWQ grid search. We quantize W * s group-wise, and store
        the state in the *unscaled* coordinate so encoders and dequant compose
        with the AWQ division. Concretely the deployed weight is
        Q(W*s)/s; we keep integer codes of W*s and fold 1/s into `scale`.
        """
        C, in_features = W.shape
        device, dtype = W.device, W.dtype
        s = awq_scales.to(device=device, dtype=dtype).reshape(1, in_features)
        Ws = W * s

        n_groups = (in_features + group_size - 1) // group_size
        padded_in = n_groups * group_size
        if padded_in > in_features:
            Wp = torch.zeros(C, padded_in, device=device, dtype=dtype)
            Wp[:, :in_features] = Ws
            sp = torch.ones(1, padded_in, device=device, dtype=dtype)
            sp[:, :in_features] = s
        else:
            Wp, sp = Ws, s

        Wg = Wp.reshape(C, n_groups, group_size)
        wmin = Wg.min(dim=2, keepdim=True)[0]
        wmax = Wg.max(dim=2, keepdim=True)[0]
        max_int = 2 ** bits - 1
        scale_g = ((wmax - wmin) / max_int).clamp_min(1e-8)
        zp_g = torch.round(-wmin / scale_g).clamp(0, max_int)

        scale_q = cls._expand_group(scale_g, group_size, padded_in)
        zp = cls._expand_group(zp_g, group_size, padded_in)
        pre_round = Wp / scale_q + zp
        integer = torch.round(pre_round).clamp(0, max_int)

        # Fold AWQ division into the effective scale so dequantize() yields
        # (q - zp) * scale_q / s == Q(W*s)/s in the unscaled coordinate.
        eff_scale = scale_q / sp

        st = cls(
            float_weights=W if padded_in == in_features else
            torch.cat([W, torch.zeros(C, padded_in - in_features,
                                      device=device, dtype=dtype)], dim=1),
            pre_round=pre_round, integer_weights=integer,
            scale=eff_scale, zero_point=zp, max_int=max_int, min_int=0,
            in_features=in_features, padded_in_features=padded_in,
            original_dtype=dtype, group_size=group_size,
        )
        del Wg, wmin, wmax, scale_g, zp_g, scale_q, Ws
        return st
