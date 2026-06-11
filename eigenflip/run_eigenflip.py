"""
run_eigenflip.py -- build the base x encoder comparison from one calibration pass.

Pipeline
--------
1. Load model + tokenizer, load calibration data via calibration_utils
   (C4 by default, return_tensors=True so the full seqlen is used for stats).
2. For each layer batch (AWQ-style): collect LayerStats ONCE (streaming Gram or
   sketch, lm_head skipped), reused across all encoders.
3. For each requested (base, encoder) cell: apply the encoder to the base state
   and stash the corrected weight. We keep one model per cell by deep-copying
   the corrected weights into a fresh state_dict, written at the end.

Memory: one streaming accumulator per layer in a batch; corrected weights for
the active layer only; we do NOT hold C output-row activations. Each base x
encoder variant is materialized as a separate saved model so downstream PPL eval
loads them independently.

Because holding K full model copies in RAM is wasteful, the default strategy is
ONE ENCODER (or one base x encoder set) per process invocation: pass
--bases and --encoders to pick the cells, run the script once per row of the
table, save each to its own --output-dir/<base>_<encoder>/. This keeps peak
memory at a single model. See the command list in the README.

This script imports your calibration_utils.py and (optionally) an awq run's
layer_scales (a .pt dict) for the AWQ base.
"""

from __future__ import annotations

import argparse
import copy
import os
import gc

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigenflip.statistics.collect import StatsCollector, is_lm_head
from eigenflip.pipeline.runner import (default_registry, make_base_state,
                                       quantize_layer_variants)
from eigenflip.quantization.awq_scales import scales_from_awq_run

# calibration_utils.py must be importable (same dir or PYTHONPATH)
try:
    from calibration_utils import (get_c4_calibration_data,
                                    get_wikitext2_calibration_data)
except ImportError:
    get_c4_calibration_data = get_wikitext2_calibration_data = None


def linear_layers(model, skip_lm_head=True):
    out = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if skip_lm_head and is_lm_head(name):
                continue
            out.append((name, module))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="./models/Mistral-7B-v0.3")
    p.add_argument("--output-dir", default="./quantized_models/eigenflip")
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--k", type=int, default=16, help="trust-region rank")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--bases", nargs="+", default=["rtn"],
                   choices=["rtn", "awq"])
    p.add_argument("--encoders", nargs="+",
                   default=["none", "clc", "eigenflip", "eigenflip_solve", "gptq"])
    p.add_argument("--eig-backend", default="auto",
                   choices=["auto", "gram", "sketch", "moments"])
    p.add_argument("--vram-fraction", type=float, default=0.4)
    p.add_argument("--no-prefer-exact", action="store_true",
                   help="skip fp64 Gram; go straight to fp32/sketch")
    p.add_argument("--layer-batch-size", type=int, default=16)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-tokens-per-sample", type=int, default=None,
                   help="subsample tokens in the hook (None = use all)")
    p.add_argument("--calib-dataset", default="c4",
                   choices=["c4", "wikitext2"])
    p.add_argument("--cache-dir", default="./calibration_cache")
    p.add_argument("--awq-scales-pt", default=None,
                   help=".pt dict {name: scales} or an awq run's layer_scales")
    # encoder knobs (match CLC template defaults)
    p.add_argument("--clc-knee", type=float, default=-10.0)
    p.add_argument("--clc-budget", type=float, default=1.0)
    p.add_argument("--ef-knee", type=float, default=-10.0)
    p.add_argument("--ef-budget", type=float, default=1.0)
    p.add_argument("--gptq-damp", type=float, default=0.01)
    p.add_argument("--solve-order", default="leverage",
                   choices=["leverage", "diag", "natural"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model:", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    # calibration: tensors preserve full seqlen for honest n/d
    print(f"Loading calibration data from {args.calib_dataset} ...")
    if get_c4_calibration_data is None:
        raise RuntimeError("calibration_utils.py not importable; put it on PYTHONPATH")
    if args.calib_dataset == "c4":
        calib = get_c4_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=args.seqlen,
            seed=args.seed, return_tensors=True, cache_dir=args.cache_dir)
    else:
        # wikitext2 loader returns text; collect will tokenize (truncates to 512)
        calib = get_wikitext2_calibration_data(
            tokenizer, n_samples=args.n_calib, seqlen=args.seqlen,
            seed=args.seed, cache_dir=args.cache_dir)
    print(f"Calibration samples: {len(calib)} from {args.calib_dataset}")
    # AWQ scales (path A) if AWQ base requested
    awq_scales = {}
    if "awq" in args.bases:
        if args.awq_scales_pt is None:
            raise ValueError("AWQ base needs --awq-scales-pt (from your awq run)")
        raw = torch.load(args.awq_scales_pt, map_location="cpu")
        # accept either {name: tensor} or {name: {'scales': ...}}
        if raw and isinstance(next(iter(raw.values())), dict):
            awq_scales = scales_from_awq_run(raw)
        else:
            awq_scales = {k: (v if torch.is_tensor(v) else torch.as_tensor(v))
                          for k, v in raw.items()}

    registry = default_registry(
        clc_knee=args.clc_knee, clc_budget=args.clc_budget,
        ef_knee=args.ef_knee, ef_budget=args.ef_budget,
        gptq_damp=args.gptq_damp, solve_order=args.solve_order)
    encoders = {n: registry[n] for n in args.encoders}
    # gptq encoder needs materialized Sigma -> mark those layers keep_sigma
    needs_sigma = "gptq" in encoders

    layers = linear_layers(model, skip_lm_head=True)
    print(f"Linear layers (lm_head skipped): {len(layers)}")

    # one saved model per (base, encoder); start from fresh copies of base sd.
    # NOTE ON MEMORY: this holds len(bases)*len(encoders) CPU state_dicts. For a
    # full 2x5 table that's 10 copies of the 7B weights in CPU RAM (~130 GB fp32
    # / ~65 GB if kept bf16). For large tables, run ONE cell per invocation
    # (e.g. --bases rtn --encoders eigenflip_solve) so peak memory is a single
    # model -- see the command list in README. The multi-cell path here is for
    # small selections / machines with ample RAM.
    base_state_dicts = {(b, e): copy.deepcopy(model.state_dict())
                        for b in args.bases for e in encoders}

    collector = StatsCollector(
        model, tokenizer, device=device, k=args.k, eps=args.eps,
        vram_fraction=args.vram_fraction,
        prefer_exact=not args.no_prefer_exact,
        force_backend=None if args.eig_backend == "auto" else args.eig_backend,
        max_tokens_per_sample=args.max_tokens_per_sample)

    LBS = args.layer_batch_size
    for b0 in range(0, len(layers), LBS):
        batch = layers[b0:b0 + LBS]
        names = [n for n, _ in batch]
        print(f"\n[batch {b0//LBS + 1}] layers {b0}-{b0+len(batch)-1}")
        # keep Sigma only for this batch's layers if gptq is requested
        collector.keep_sigma_for = set(names) if needs_sigma else set()
        stats_map = collector.collect_batch(batch, calib, args.n_calib)

        for name, module in batch:
            W = module.weight.data
            stats = stats_map[name]
            variants = quantize_layer_variants(
                W, stats, args.bases, encoders, args.bits, args.group_size,
                awq_scales=awq_scales.get(name))
            for (base, enc_name), (corrected, info) in variants.items():
                key = f"{name}.weight"
                base_state_dicts[(base, enc_name)][key] = corrected.cpu()
            stats.free_sigma()
            del stats, variants, W
        del stats_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # write each variant
    os.makedirs(args.output_dir, exist_ok=True)
    for (base, enc_name), sd in base_state_dicts.items():
        sub = os.path.join(args.output_dir, f"{base}_{enc_name}")
        os.makedirs(sub, exist_ok=True)
        print(f"saving {base}+{enc_name} -> {sub}")
        model.load_state_dict(sd)
        model.save_pretrained(sub)
        tokenizer.save_pretrained(sub)
    print("\ndone.")


if __name__ == "__main__":
    main()
