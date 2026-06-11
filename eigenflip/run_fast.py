"""
run_fast.py -- one base x one encoder per invocation, single calibration pass.

Encoder decides what to collect:
    rtn               -> need_H=False, mean unused (RTN needs no stats)
    clc               -> need_H=False (mean only)
    eigenflip/solve   -> need_H=True  (top-k eig of Sigma)
    gptq/shrinkage    -> need_H=True  + keep_sigma

Block-paged: the encoded weight is written into the module immediately so the
block re-run propagates quantized activations to the next block (matches GPTQ).

Run one cell per process to bound memory:
  python -m eigenflip.run_fast --base rtn --encoder eigenflip_solve --k 16 ...
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigenflip.statistics.sequential import sequential_collect_and_encode
from eigenflip.quantization.state import IntegerQuantizedTensorState
from eigenflip.encoders.base_encoder import IdentityEncoder
from eigenflip.encoders.flip import make_clc, make_eigenflip
from eigenflip.encoders.eigenflip_solve import EigenFlipSolve
from eigenflip.encoders.dense_reference import DenseGPTQ
from eigenflip.encoders.shrinkage import ShrinkageGPTQ
from eigenflip.quantization.awq_scales import scales_from_awq_run

try:
    from calibration_utils import (get_c4_calibration_data,
                                   get_wikitext2_calibration_data)
except ImportError:
    get_c4_calibration_data = get_wikitext2_calibration_data = None


# which encoders need the full Gram H (and which need Sigma materialized)
NEED_H = {"none": False, "clc": False,
          "eigenflip": True, "eigenflip_solve": True,
          "gptq": True, "shr_gptq_cov": True, "shr_gptq_2m": True}
KEEP_SIGMA = {"gptq", "shr_gptq_cov", "shr_gptq_2m"}


def build_encoder(name, args):
    if name == "none":
        return IdentityEncoder()
    if name == "clc":
        return make_clc(args.clc_knee, args.clc_budget, use_knee=False)
    if name == "eigenflip":
        return make_eigenflip(args.ef_knee, args.ef_budget, use_knee=False)
    if name == "eigenflip_solve":
        return EigenFlipSolve(order=args.solve_order)
    if name == "gptq":
        return DenseGPTQ(damp=args.gptq_damp)
    if name == "shr_gptq_cov":
        return ShrinkageGPTQ(family="cov", lam=args.shr_lambda)
    if name == "shr_gptq_2m":
        return ShrinkageGPTQ(family="2m", lam=args.shr_lambda)
    raise ValueError(name)


def build_dataloader(calib, tokenizer, seqlen, nsamples, seed):
    # accept pre-tokenized tensors (return_tensors=True) or text
    out = []
    if calib and torch.is_tensor(calib[0]):
        for t in calib[:nsamples]:
            ids = t if t.dim() == 2 else t.unsqueeze(0)
            out.append((ids,))
        return out
    # text -> tokenize+concat+slice
    import random
    rng = random.Random(seed)
    ids = []
    for txt in calib:
        ids.extend(tokenizer(txt, return_tensors="pt").input_ids[0].tolist())
        if len(ids) > seqlen * nsamples * 4:
            break
    while len(ids) < seqlen * nsamples:
        ids = ids + ids
    mx = len(ids) - seqlen - 1
    for _ in range(nsamples):
        s = rng.randint(0, mx)
        out.append((torch.tensor([ids[s:s + seqlen]], dtype=torch.long),))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="./models/Mistral-7B-v0.3")
    p.add_argument("--output-dir", default="./quantized_models/eigenflip")
    p.add_argument("--base", default="rtn", choices=["rtn", "awq"])
    p.add_argument("--encoder", required=True,
                   choices=list(NEED_H.keys()))
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--calib-dataset", default="c4", choices=["c4", "wikitext2"])
    p.add_argument("--cache-dir", default="./calibration_cache")
    p.add_argument("--awq-scales-pt", default=None)
    p.add_argument("--eig-on-cpu", action="store_true",
                   help="run eigh on CPU for heavy layers (saves VRAM)")
    p.add_argument("--no-true-sequential", dest="true_sequential",
                   action="store_false", default=True)
    p.add_argument("--clc-knee", type=float, default=-10.0)
    p.add_argument("--clc-budget", type=float, default=1.0)
    p.add_argument("--ef-knee", type=float, default=-10.0)
    p.add_argument("--ef-budget", type=float, default=1.0)
    p.add_argument("--gptq-damp", type=float, default=0.01)
    p.add_argument("--shr-lambda", type=float, default=0.01)
    p.add_argument("--solve-order", default="leverage")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, device_map="cpu",
        trust_remote_code=True).eval()

    if get_c4_calibration_data is None:
        raise RuntimeError("calibration_utils.py not importable")
    loader = (get_c4_calibration_data if args.calib_dataset == "c4"
              else get_wikitext2_calibration_data)
    kw = dict(n_samples=args.nsamples, seqlen=args.seqlen, seed=args.seed,
              cache_dir=args.cache_dir)
    if args.calib_dataset == "c4":
        kw["return_tensors"] = True
    calib = loader(tok, **kw)
    dataloader = build_dataloader(calib, tok, args.seqlen, args.nsamples, args.seed)

    awq_scales = {}
    if args.base == "awq":
        if not args.awq_scales_pt:
            raise ValueError("AWQ base needs --awq-scales-pt")
        raw = torch.load(args.awq_scales_pt, map_location="cpu")
        awq_scales = (scales_from_awq_run(raw)
                      if raw and isinstance(next(iter(raw.values())), dict)
                      else {k: torch.as_tensor(v) for k, v in raw.items()})

    enc = build_encoder(args.encoder, args)
    need_H = NEED_H[args.encoder]
    keep_sigma = args.encoder in KEEP_SIGMA

    # RTN needs no stats at all -> still goes through the loop but ignores them.
    def callback(key, module, stats):
        W = module.weight.data
        s = awq_scales.get(key.replace("model.layers.", "").replace(".", "_")) \
            if args.base == "awq" else None
        if args.base == "rtn":
            state = IntegerQuantizedTensorState.from_rtn(W, args.bits, args.group_size)
        else:
            sc = awq_scales.get(key)  # exact key match from awq run
            if sc is None:
                raise KeyError(f"no AWQ scales for {key}")
            state = IntegerQuantizedTensorState.from_awq(W, sc, args.bits, args.group_size)
        corrected, _ = enc.apply(state, stats)
        module.weight.data = corrected.to(module.weight.dtype)
        del state, corrected

    print(f"base={args.base} encoder={args.encoder} need_H={need_H} "
          f"keep_sigma={keep_sigma} k={args.k}")
    sequential_collect_and_encode(
        model, dataloader, device,
        need_H=need_H, k=args.k, eps=args.eps, callback=callback,
        keep_sigma=keep_sigma, true_sequential=args.true_sequential,
        skip_lm_head=True, eig_on_cpu=args.eig_on_cpu)

    out = os.path.join(args.output_dir, f"{args.base}_{args.encoder}")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
