"""
section66.py -- the decisive comparison (paper Section 6.6).

Builds, on a FIXED base, the full encoder set for the gating experiment and
runs the shrinkage baselines with the pre-registered lambda protocol:

  EigenFlip Solve (selected rank k)        -- the method under test
  gptq                                      -- full-H reference
  shr_gptq (cov), lambda (i)  global tuned  -- PRIMARY falsification baseline
  shr_gptq (cov), lambda (ii) analytic      -- PRIMARY, no-tuning
  shr_gptq (cov), lambda (iii) per-layer    -- PRIMARY, oracle upper bound
  shr_gptq (2m),  lambda (i)/(ii)/(iii)     -- SECONDARY baseline (3 variants)
  clc, eigenflip, none                      -- context rows

Lambda tuning uses HELD-OUT distortion on a disjoint calibration split, never
the eval distribution (Section 6.6). We split the calibration set into A and B:
  * encoder statistics (Sigma_A) are fit on A,
  * the held-out scoring metric H'(B) is built from B,
  * the GLOBAL lambda is the argmin of summed held-out distortion over probe
    layers; per-layer lambda (iii) is the per-layer argmin.

Falsification check (printed): does EigenFlip Solve at rank k beat shr_gptq(cov)
under (i) and (ii) on held-out distortion, and stay competitive with (iii)?

Memory: gram backend with keep_sigma is REQUIRED (shrinkage + gptq need Sigma).
For d up to ~10k this is one d x d per layer in the active batch; the harness
frees Sigma immediately after a layer's encoders are done. Run ONE base per
invocation to bound peak memory.

This harness writes one saved model per method into
  --output-dir/<base>_<method>/   ready for eval_ppl.py.
It also writes a JSON of the held-out distortion table and the chosen lambdas.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigenflip.statistics.collect import StatsCollector, is_lm_head
from eigenflip.quantization.state import IntegerQuantizedTensorState
from eigenflip.pipeline.runner import make_base_state
from eigenflip.encoders.base_encoder import IdentityEncoder
from eigenflip.encoders.flip import make_clc, make_eigenflip
from eigenflip.encoders.eigenflip_solve import EigenFlipSolve
from eigenflip.encoders.dense_reference import DenseGPTQ
from eigenflip.encoders.shrinkage import ShrinkageGPTQ
from eigenflip.pipeline.distortion import (residual, distortion, build_heldout_H,
                                           tune_global_lambda,
                                           tune_perlayer_lambda, LAMBDA_GRID)
from eigenflip.quantization.awq_scales import scales_from_awq_run

try:
    from calibration_utils import (get_c4_calibration_data,
                                   get_wikitext2_calibration_data)
except ImportError:
    get_c4_calibration_data = get_wikitext2_calibration_data = None


def linear_layers(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and not is_lm_head(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="./models/Mistral-7B-v0.3")
    p.add_argument("--output-dir", default="./quantized_models/section66")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4, 8])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--k", type=int, default=16, help="EigenFlip Solve rank")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--base", default="awq", choices=["rtn", "awq"],
                   help="fixed base (blocking variable); run once per base")
    p.add_argument("--awq-scales-pt", default=None)
    p.add_argument("--layer-batch-size", type=int, default=8)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--calib-dataset", default="c4", choices=["c4", "wikitext2"])
    p.add_argument("--cache-dir", default="./calibration_cache")
    p.add_argument("--gptq-damp", type=float, default=0.01)
    p.add_argument("--solve-order", default="leverage")
    # which families/instantiations to run
    p.add_argument("--families", nargs="+", default=["cov", "2m"],
                   choices=["cov", "2m"])
    p.add_argument("--lambda-modes", nargs="+",
                   default=["global", "analytic", "perlayer"],
                   choices=["global", "analytic", "perlayer"])
    p.add_argument("--probe-frac", type=float, default=1.0,
                   help="fraction of layers used to tune the GLOBAL lambda "
                        "(1.0 = all; smaller speeds tuning)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True).eval()

    # calibration, split into disjoint halves A and B
    if get_c4_calibration_data is None:
        raise RuntimeError("calibration_utils.py not importable")
    if args.calib_dataset == "c4":
        calib = get_c4_calibration_data(tok, n_samples=args.n_calib,
                                        seqlen=args.seqlen, seed=args.seed,
                                        return_tensors=True,
                                        cache_dir=args.cache_dir)
    else:
        calib = get_wikitext2_calibration_data(tok, n_samples=args.n_calib,
                                               seqlen=args.seqlen, seed=args.seed,
                                               cache_dir=args.cache_dir)
    half = len(calib) // 2
    calib_A, calib_B = calib[:half], calib[half:]
    print(f"calibration split: |A|={len(calib_A)} |B|={len(calib_B)}")

    # AWQ scales if needed
    awq_scales = {}
    if args.base == "awq":
        if args.awq_scales_pt is None:
            raise ValueError("AWQ base needs --awq-scales-pt")
        raw = torch.load(args.awq_scales_pt, map_location="cpu")
        if raw and isinstance(next(iter(raw.values())), dict):
            awq_scales = scales_from_awq_run(raw)
        else:
            awq_scales = {k: (v if torch.is_tensor(v) else torch.as_tensor(v))
                          for k, v in raw.items()}

    # method set ---------------------------------------------------------------
    # fixed-rank / no-lambda encoders applied directly
    fixed_methods = {
        "none": IdentityEncoder(),
        "clc": make_clc(-10.0, 1.0, use_knee=False),
        "eigenflip": make_eigenflip(-10.0, 1.0, use_knee=False),
        "eigenflip_solve": EigenFlipSolve(order=args.solve_order),
        "gptq": DenseGPTQ(damp=args.gptq_damp),
    }
    # shrinkage methods are built per-lambda at encode time; enumerate the
    # (family, mode) combinations we will produce.
    shr_combos = [(fam, mode) for fam in args.families
                  for mode in args.lambda_modes]

    method_names = list(fixed_methods.keys()) + \
        [f"shr_gptq_{fam}_{mode}" for fam, mode in shr_combos]

    # one CPU state_dict per method (run one base per invocation -> manageable)
    sds = {m: copy.deepcopy(model.state_dict()) for m in method_names}

    layers = linear_layers(model)
    print(f"layers (lm_head skipped): {len(layers)}")
    n_probe = max(1, int(args.probe_frac * len(layers)))
    probe_names = set(n for n, _ in layers[:n_probe])  # first n_probe layers

    # collectors for split A and split B (gram + keep_sigma for ALL layers,
    # since shrinkage/gptq need Sigma)
    collA = StatsCollector(model, tok, device=device, k=args.k, eps=args.eps,
                           force_backend="gram")
    collB = StatsCollector(model, tok, device=device, k=args.k, eps=args.eps,
                           force_backend="gram")

    # global-lambda tuning needs A-fit stats + held-out H'(B) for probe layers.
    # We collect per batch; tuning is done after all probe layers seen, so we
    # stash probe-layer encode inputs. To bound memory we keep only Sigma_A and
    # Hprime_B (both d x d) for probe layers -- acceptable for a probe subset.
    chosen_lambda = {}     # (family, mode) -> lambda (global: one; perlayer: dict by name)
    heldout_table = {}     # diagnostics

    # ---- PASS 1: per batch, collect A & B, tune per-layer lambdas + cache probes
    probe_cache = []       # list of (name, W, state, stats_A, Hprime_B)
    LBS = args.layer_batch_size
    print("\n=== PASS 1: statistics + per-layer tuning ===")
    for b0 in range(0, len(layers), LBS):
        batch = layers[b0:b0 + LBS]
        names = [n for n, _ in batch]
        collA.keep_sigma_for = set(names)
        collB.keep_sigma_for = set(names)
        print(f"[batch {b0//LBS+1}] {b0}-{b0+len(batch)-1}  (split A)")
        statsA = collA.collect_batch(batch, calib_A, len(calib_A))
        print(f"           (split B)")
        statsB = collB.collect_batch(batch, calib_B, len(calib_B))

        for name, module in batch:
            W = module.weight.data
            stA = statsA[name]
            stB = statsB[name]
            state = make_base_state(W, args.base, args.bits, args.group_size,
                                    awq_scales.get(name))
            Hprime_B = build_heldout_H(stB.mu_hat, stB.Sigma)

            # per-layer lambda (instantiation iii) for each shrinkage combo
            for fam in args.families:
                if "perlayer" in args.lambda_modes:
                    fac = lambda lam, fam=fam: ShrinkageGPTQ(family=fam, lam=lam)
                    best, scores = tune_perlayer_lambda(
                        fac, W, state, stA, Hprime_B, grid=LAMBDA_GRID)
                    chosen_lambda.setdefault((fam, "perlayer"), {})[name] = best
                    heldout_table.setdefault(f"{fam}_perlayer", {})[name] = scores

            # cache probe inputs for GLOBAL tuning (keep Sigma_A + Hprime_B)
            if name in probe_names:
                probe_cache.append((name, W.clone(), state, stA, Hprime_B))
            else:
                stA.free_sigma(); stB.free_sigma()
                del Hprime_B, state

            del stB
        del statsA, statsB
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- GLOBAL lambda (instantiation i): argmin summed held-out distortion
    print("\n=== global lambda tuning (held-out distortion) ===")
    for fam in args.families:
        if "global" not in args.lambda_modes:
            continue
        fac = lambda lam, fam=fam: ShrinkageGPTQ(family=fam, lam=lam)
        probe_layers = [(W, state, stA, HB)
                        for (nm, W, state, stA, HB) in probe_cache]
        best, scores = tune_global_lambda(fac, probe_layers, grid=LAMBDA_GRID)
        chosen_lambda[(fam, "global")] = best
        heldout_table[f"{fam}_global"] = {str(k): v for k, v in scores.items()}
        print(f"  family={fam}: global lambda* = {best}   scores={scores}")

    # ---- PASS 2: encode every method into its state_dict
    print("\n=== PASS 2: encode all methods ===")
    # re-walk layers; we need A-fit stats again. For probe layers we still have
    # them cached; for non-probe layers we re-collect split A only (with Sigma).
    probe_by_name = {nm: (W, state, stA, HB)
                     for (nm, W, state, stA, HB) in probe_cache}

    for b0 in range(0, len(layers), LBS):
        batch = layers[b0:b0 + LBS]
        names = [n for n, _ in batch]
        need_recollect = [(n, m) for n, m in batch if n not in probe_by_name]
        statsA2 = {}
        if need_recollect:
            collA.keep_sigma_for = set(n for n, _ in need_recollect)
            statsA2 = collA.collect_batch(need_recollect, calib_A, len(calib_A))

        for name, module in batch:
            W = module.weight.data
            if name in probe_by_name:
                _, state, stA, _ = probe_by_name[name]
            else:
                stA = statsA2[name]
                state = make_base_state(W, args.base, args.bits,
                                        args.group_size, awq_scales.get(name))

            key = f"{name}.weight"
            # fixed methods
            for mname, enc in fixed_methods.items():
                corrected, _ = enc.apply(state, stA)
                sds[mname][key] = corrected.cpu()
                del corrected
            # shrinkage methods
            for fam, mode in shr_combos:
                if mode == "global":
                    lam = chosen_lambda[(fam, "global")]
                    enc = ShrinkageGPTQ(family=fam, lam=lam)
                elif mode == "analytic":
                    enc = ShrinkageGPTQ(family=fam, lam="analytic",
                                        n_eff=len(calib_A))
                else:  # perlayer
                    lam = chosen_lambda[(fam, "perlayer")][name]
                    enc = ShrinkageGPTQ(family=fam, lam=lam)
                corrected, _ = enc.apply(state, stA)
                sds[f"shr_gptq_{fam}_{mode}"][key] = corrected.cpu()
                del corrected

            stA.free_sigma()
            del stA, state, W
        del statsA2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- save models + diagnostics
    os.makedirs(args.output_dir, exist_ok=True)
    for mname, sd in sds.items():
        sub = os.path.join(args.output_dir, f"{args.base}_{mname}")
        os.makedirs(sub, exist_ok=True)
        print(f"saving {args.base}+{mname} -> {sub}")
        model.load_state_dict(sd)
        model.save_pretrained(sub)
        tok.save_pretrained(sub)

    diag = {
        "base": args.base, "bits": args.bits, "k": args.k,
        "chosen_lambda": {f"{fam}_{mode}":
                          (chosen_lambda.get((fam, mode))
                           if mode != "perlayer"
                           else "per-layer dict (see heldout_table)")
                          for fam in args.families for mode in args.lambda_modes},
        "heldout_distortion": heldout_table,
    }
    with open(os.path.join(args.output_dir, f"{args.base}_section66.json"), "w") as f:
        json.dump(diag, f, indent=2, default=str)

    # ---- falsification check on held-out distortion (probe layers)
    print("\n=== FALSIFICATION CHECK (held-out distortion, probe layers) ===")
    falsification_check(probe_cache, args, chosen_lambda)

    print("\ndone.")


@torch.no_grad()
def falsification_check(probe_cache, args, chosen_lambda):
    """
    Compare EigenFlip Solve vs shr_gptq(cov) (i)/(ii)/(iii) on summed held-out
    distortion over probe layers. The thesis requires Solve <= cov(i) and
    Solve <= cov(ii); competitive with cov(iii).
    """
    solve = EigenFlipSolve(order=args.solve_order)
    totals = {"eigenflip_solve": 0.0,
              "cov_global": 0.0, "cov_analytic": 0.0, "cov_perlayer": 0.0}

    for (name, W, state, stA, HB) in probe_cache:
        # solve
        out, _ = solve.apply(state, stA)
        E = residual(state, out); totals["eigenflip_solve"] += distortion(E.to(HB), HB)
        del out, E
        # cov global
        if ("cov", "global") in chosen_lambda:
            enc = ShrinkageGPTQ(family="cov", lam=chosen_lambda[("cov", "global")])
            out, _ = enc.apply(state, stA); E = residual(state, out)
            totals["cov_global"] += distortion(E.to(HB), HB); del out, E
        # cov analytic
        enc = ShrinkageGPTQ(family="cov", lam="analytic", n_eff=args.n_calib // 2)
        out, _ = enc.apply(state, stA); E = residual(state, out)
        totals["cov_analytic"] += distortion(E.to(HB), HB); del out, E
        # cov perlayer
        if ("cov", "perlayer") in chosen_lambda:
            lam = chosen_lambda[("cov", "perlayer")].get(name)
            if lam is not None:
                enc = ShrinkageGPTQ(family="cov", lam=lam)
                out, _ = enc.apply(state, stA); E = residual(state, out)
                totals["cov_perlayer"] += distortion(E.to(HB), HB); del out, E
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    s = totals["eigenflip_solve"]
    print(f"  EigenFlip Solve         : {s:.6e}")
    for key in ("cov_global", "cov_analytic", "cov_perlayer"):
        v = totals[key]
        if v > 0:
            verdict = "PASS" if s <= v else "FAIL"
            print(f"  shr_gptq(cov,{key[4:]:8s}): {v:.6e}   "
                  f"Solve {'<=' if s<=v else '>'} baseline  [{verdict}]")
    print("  (PASS on global & analytic supports the trust-region thesis; "
          "competitive-with-perlayer is the oracle bound.)")


if __name__ == "__main__":
    main()
