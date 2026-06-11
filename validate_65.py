"""
validate_65.py -- Section 6.5 checks on real model layers.

(i)  Dense-Woodbury equivalence: EigenFlip Solve codes vs DenseSurrogateGPTQ
     codes on the SAME H~_{k,eps}. Target: ~100% agreement (we saw bitwise in
     numpy; bf16->fp64 may show a handful of boundary disagreements -- report,
     don't assume).
(ii) gram vs sketch eigenpair agreement (principal angles) on a few layers.

Run on one or two layers; this is a correctness gate, not a full sweep.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigenflip.statistics.collect import StatsCollector, is_lm_head
from eigenflip.statistics.accumulators import StreamingGram, StreamingSketch, plan_gram
from eigenflip.statistics.trust_region import stats_from_gram, stats_from_sketch
from eigenflip.quantization.state import IntegerQuantizedTensorState
from eigenflip.encoders.eigenflip_solve import EigenFlipSolve
from eigenflip.encoders.dense_reference import DenseSurrogateGPTQ

try:
    from calibration_utils import get_c4_calibration_data
except ImportError:
    get_c4_calibration_data = None


@torch.no_grad()
def principal_angles(U, V):
    # cos of principal angles = singular values of U^T V (both orthonormal cols)
    s = torch.linalg.svdvals(U.t() @ V).clamp(-1, 1)
    return torch.arccos(s)  # radians; small = aligned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="./models/Mistral-7B-v0.3")
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n-calib", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--cache-dir", default="./calibration_cache")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()

    calib = get_c4_calibration_data(tok, n_samples=args.n_calib,
                                    seqlen=args.seqlen, return_tensors=True,
                                    cache_dir=args.cache_dir)

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and not is_lm_head(n)][:args.n_layers]

    coll = StatsCollector(model, tok, device=device, k=args.k,
                          force_backend="gram", keep_sigma_for=set())
    stats_map = coll.collect_batch(layers, calib, args.n_calib)

    solve = EigenFlipSolve(order="leverage")
    dense = DenseSurrogateGPTQ(order="leverage")

    for name, module in layers:
        W = module.weight.data
        st = stats_map[name]
        state = IntegerQuantizedTensorState.from_rtn(W, args.bits, args.group_size)
        out_s, info_s = solve.apply(state, st)
        out_d, info_d = dense.apply(state, st)
        # compare codes via dequant equality (same scale/zp -> code equality)
        agree = (out_s == out_d).float().mean().item()
        print(f"[{name}] Solve vs dense surrogate: {agree*100:.3f}% weight-equal "
              f"(k={st.k}, backend={st.backend})")

    print("\n(ii) gram vs sketch eigenspace (first layer):")
    name, module = layers[0]
    d = module.weight.shape[1]
    # rebuild both backends on this single layer
    for kind in ("gram", "sketch"):
        c2 = StatsCollector(model, tok, device=device, k=args.k,
                            force_backend=kind, keep_sigma_for=set())
        sm = c2.collect_batch([(name, module)], calib, args.n_calib)
        if kind == "gram":
            U_g = sm[name].U_k
        else:
            U_s = sm[name].U_k
    ang = principal_angles(U_g.float(), U_s.float())
    print(f"  principal angles (deg): max={ang.max().item()*180/3.14159:.2f} "
          f"mean={ang.mean().item()*180/3.14159:.2f}")


if __name__ == "__main__":
    main()
