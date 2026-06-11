"""
Direction 2 (cross-eval): does the rung-2 eigenspace gain SURVIVE distribution
shift between calibration and evaluation?
==============================================================================

Rung 2 fits an encoder against the top-k eigenspace of the CALIBRATION block
covariance Sigma_cal. The hierarchy (doc Eq.5) predicts that beyond the
eigenvalue-separation point those eigendirections are calibration NOISE and will
not transfer. The clean test:

    FIT phase  (calibration H_cal, mu_cal):  pick flips  -> integer codes
    SCORE phase (evaluation   H_eval):        distortion = e^T H_eval e

So every encoder (RTN / CLC / eigflip-k) is BUILT using only calib statistics,
then its resulting codes are scored under a DIFFERENT distribution's H. The
exact-CVP reference is computed two ways:
    - exact_cal  : the block optimum under H_cal   (what the fit "aimed at")
    - exact_eval : the block optimum under H_eval  (the honest floor we score against)
The recoverable gap for the headroom fractions is taken w.r.t. exact_eval, since
that is the real attainable distortion on the eval distribution.

Prediction to look for in the output:
    in-distribution (cal==eval): E2-CLC rises monotonically with k (as you saw).
    cross  (cal!=eval):          E2-CLC rises, PEAKS at some k*, then DROPS.
    that k* is the recommended operating point (eigenvalue-separation knee).

Depends on the Direction-1 file (`d1`) and the Direction-2 file (`d2`) for the
shared machinery and the eigflip encoder. OFFLINE / encoder-side only.
"""

import torch
import numpy as np
import argparse
import json
import random
import gc
import importlib.util as _ilu
import os as _os


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Collect TWO second-moment estimates (one per dataset) for the same layers
# =============================================================================

def collect_dual_H(d1, model, tokenizer, layer_batch, calib_a, calib_b,
                   device, n_samples, max_tokens_per_sample):
    """
    Run the Direction-1 collector twice over the same layer batch, once per
    dataset. Returns dict name -> (H_a, mu_a, H_b, mu_b), all fp32 CPU,
    James-Stein-shrunk means (rung-1 statistic), matching d1 exactly.
    """
    out = {}

    coll_a = d1.SecondMomentCollector(max_tokens_per_sample=max_tokens_per_sample)
    d1.calibrate_and_collect(model, tokenizer, layer_batch, calib_a,
                             coll_a, device, n_samples, max_len=512)
    finals_a = {}
    for name, _ in layer_batch:
        Ha, mua = coll_a.finalize(name, use_james_stein=True)
        finals_a[name] = (Ha, mua)
    coll_a.clear(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    coll_b = d1.SecondMomentCollector(max_tokens_per_sample=max_tokens_per_sample)
    d1.calibrate_and_collect(model, tokenizer, layer_batch, calib_b,
                             coll_b, device, n_samples, max_len=512)
    for name, _ in layer_batch:
        Hb, mub = coll_b.finalize(name, use_james_stein=True)
        Ha, mua = finals_a[name]
        if Ha is None or Hb is None:
            out[name] = None
        else:
            out[name] = (Ha, mua, Hb, mub)
    coll_b.clear(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# =============================================================================
# Per-layer cross-eval sweep
# =============================================================================

@torch.no_grad()
def measure_layer_crosseval(d1, d2, name, module,
                            H_cal, mu_cal, H_eval, mu_eval,
                            bits, group_size, dprimes, n_rows, block_method,
                            k_list, enum_threshold, device, seed=0):
    """
    FIT encoders on (H_cal, mu_cal); SCORE distortion under H_eval.
    Two directions are measured in one pass by calling this with the (cal, eval)
    pair and, separately, with the (eval, cal) pair from the driver.

    Block selection uses the CALIBRATION statistics (what the practitioner has).
    """
    W = module.weight.data.to(device).float()
    out_f, in_f = W.shape
    H_cal = H_cal.to(device).float();  mu_cal = mu_cal.to(device).float()
    H_eval = H_eval.to(device).float(); mu_eval = mu_eval.to(device).float()
    Sigma_cal = H_cal - torch.outer(mu_cal, mu_cal)

    _, W_int, scale_flat, zp_flat = d1.groupwise_asym_rtn(W, bits, group_size)
    max_int = 2 ** bits - 1
    rng = random.Random(seed)
    rows = rng.sample(range(out_f), min(n_rows, out_f))

    records = []
    for dprime in dprimes:
        # block chosen on CALIBRATION stats (honest: that's all we have at fit time)
        block = d1.select_block_indices(mu_cal, H_cal, dprime,
                                        method=block_method).to(device)
        Hcal_II = H_cal[block][:, block]
        Heval_II = H_eval[block][:, block]
        Sigcal_II = Sigma_cal[block][:, block]
        mucal_I = mu_cal[block]

        kmax = min(max(k_list), dprime)
        U_full, sqrt_full = (d2.block_topk_eig(Sigcal_II, kmax)
                             if kmax > 0 else (None, None))

        for r in rows:
            w_I = W[r, block]
            wint_I = W_int[r, block]
            s_I = scale_flat[r, block]
            zp_I = zp_flat[r, block]

            w_dq = (wint_I - zp_I) * s_I
            e_rtn = w_dq - w_I
            dist_rtn_eval = float(e_rtn @ (Heval_II @ e_rtn))

            # exact floor on the EVAL distribution (honest attainable optimum)
            _, _, dist_ex_eval = d1.exact_cvp_block(
                w_I, s_I, zp_I, max_int, Heval_II, enum_threshold=enum_threshold)
            # exact under CAL (what the fit aimed at) -- for reference only
            _, _, dist_ex_cal = d1.exact_cvp_block(
                w_I, s_I, zp_I, max_int, Hcal_II, enum_threshold=enum_threshold)

            # CLC fit on CAL stats, scored on EVAL
            q_clc, _, _ = d1.clc_block(
                w_I, wint_I, s_I, zp_I, max_int, mucal_I,
                knee_tolerance=-10.0, max_flip_percent=1.0)
            e_clc = q_clc - w_I
            dist_clc_eval = float(e_clc @ (Heval_II @ e_clc))

            gap_eval = dist_rtn_eval - dist_ex_eval
            for k in k_list:
                kk = min(k, dprime)
                if kk > 0:
                    U = U_full[:, :kk]; sqrt_lam = sqrt_full[:kk]
                else:
                    U = sqrt_lam = None
                # eigflip FIT on CAL eigenspace + CAL mu
                q_e2, _, n_e2, _ = d2.eigflip_block(
                    w_I, wint_I, s_I, zp_I, max_int, mucal_I, Sigcal_II, kk,
                    max_flip_percent=1.0, knee_tolerance=-10.0,
                    U=U, sqrt_lam=sqrt_lam)
                e_e2 = q_e2 - w_I
                dist_e2_eval = float(e_e2 @ (Heval_II @ e_e2))

                clc_rec = dist_rtn_eval - dist_clc_eval
                e2_rec = dist_rtn_eval - dist_e2_eval
                records.append({
                    "layer": name, "row": r, "dprime": dprime, "k": k,
                    "dist_rtn_eval": dist_rtn_eval,
                    "dist_exact_eval": dist_ex_eval,
                    "dist_exact_cal": dist_ex_cal,
                    "dist_clc_eval": dist_clc_eval,
                    "dist_eigflip_eval": dist_e2_eval,
                    "gap_eval": gap_eval,
                    "clc_recovered": clc_rec,
                    "eigflip_recovered": e2_rec,
                    "eigflip_minus_clc": e2_rec - clc_rec,
                    "n_flips_eigflip": n_e2,
                })
    del W, H_cal, H_eval, mu_cal, mu_eval, Sigma_cal
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def summarize_crosseval(records, tag):
    by = {}
    for rec in records:
        by.setdefault((rec["dprime"], rec["k"]), []).append(rec)
    curve = []
    for (dprime, k) in sorted(by):
        rs = by[(dprime, k)]
        gap = np.array([r["gap_eval"] for r in rs])
        clc = np.array([r["clc_recovered"] for r in rs])
        e2 = np.array([r["eigflip_recovered"] for r in rs])
        delta = np.array([r["eigflip_minus_clc"] for r in rs])
        sgap = float(gap.sum())
        curve.append({
            "tag": tag, "dprime": dprime, "k": k, "n": len(rs),
            "clc_frac_of_gap": float(clc.sum()) / sgap if sgap > 0 else 0.0,
            "eigflip_frac_of_gap": float(e2.sum()) / sgap if sgap > 0 else 0.0,
            "eigflip_minus_clc_frac_of_gap": (float(delta.sum()) / sgap
                                              if sgap > 0 else 0.0),
            "rows_eigflip_beats_clc": int((delta > 1e-15).sum()),
            "rows_tie": int((np.abs(delta) <= 1e-15).sum()),
            "rows_eigflip_worse": int((delta < -1e-15).sum()),
            "delta_mean_abs": float(delta.mean()),
        })
    return curve


def _print_curve(title, curve):
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)
    print(f"{'d':>4} {'k':>4} {'n':>6} {'CLC/gap':>8} {'E2/gap':>8} "
          f"{'E2-CLC':>8} {'win':>5} {'tie':>5} {'lose':>5}")
    for c in curve:
        print(f"{c['dprime']:>4} {c['k']:>4} {c['n']:>6} "
              f"{c['clc_frac_of_gap']*100:>7.1f}% "
              f"{c['eigflip_frac_of_gap']*100:>7.1f}% "
              f"{c['eigflip_minus_clc_frac_of_gap']*100:>+7.2f}% "
              f"{c['rows_eigflip_beats_clc']:>5} {c['rows_tie']:>5} "
              f"{c['rows_eigflip_worse']:>5}")
    # peak-k report per d'
    by_d = {}
    for c in curve:
        by_d.setdefault(c["dprime"], []).append(c)
    print("\n  peak E2-CLC (recommended k*) per d':")
    for d, cs in sorted(by_d.items()):
        best = max(cs, key=lambda x: x["eigflip_minus_clc_frac_of_gap"])
        print(f"    d'={d:>3}:  k*={best['k']:>3}  "
              f"E2-CLC={best['eigflip_minus_clc_frac_of_gap']*100:+.2f}%")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Direction 2 cross-eval: rung-2 gain under distribution shift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--d1-path", type=str, default="./direction1_headroom.py")
    p.add_argument("--d2-path", type=str, default="./direction2_eigflip.py")
    p.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    p.add_argument("--output-json", type=str, default="./eigflip_crosseval_results.json")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-calib", type=int, default=128)
    # the two distributions
    p.add_argument("--calib-a", type=str, default="c4",
                   choices=["c4", "wikitext2", "wikitext2-simple"])
    p.add_argument("--calib-b", type=str, default="wikitext2",
                   choices=["c4", "wikitext2", "wikitext2-simple"])
    p.add_argument("--cache-dir", type=str, default="./calibration_cache")
    p.add_argument("--max-tokens-per-sample", type=int, default=2048)
    p.add_argument("--layer-batch-size", type=int, default=8)

    p.add_argument("--dprimes", type=int, nargs="+", default=[24])
    p.add_argument("--k-list", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--n-rows", type=int, default=64)
    p.add_argument("--block-method", type=str, default="abs_mu",
                   choices=["abs_mu", "eigvec"])
    p.add_argument("--enum-threshold", type=int, default=24)
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--sample-layers", type=int, default=0)
    p.add_argument("--layer-filter", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not _os.path.exists(args.d1_path):
        raise FileNotFoundError(f"Direction-1 file not found: {args.d1_path}")
    if not _os.path.exists(args.d2_path):
        raise FileNotFoundError(f"Direction-2 file not found: {args.d2_path}")
    d1 = _load("d1", args.d1_path)
    d2 = _load("d2", args.d2_path)

    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 84)
    print("Direction 2 CROSS-EVAL: rung-2 eigenspace gain under distribution shift")
    print(f"Model: {args.model_path} | bits={args.bits} | g={args.group_size}")
    print(f"calib-A={args.calib_a}  calib-B={args.calib_b}")
    print(f"d' grid: {args.dprimes} | k grid: {args.k_list} | rows/layer: {args.n_rows}")
    print("FIT on calib stats; SCORE distortion on the OTHER distribution's H.")
    print("=" * 84)

    if args.calib_a == args.calib_b:
        print("WARNING: calib-a == calib-b; this reduces to the in-distribution run.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    def load_calib(which):
        if which == "c4":
            return d1.get_c4_calibration_data(tokenizer, n_samples=args.n_calib,
                                              seqlen=2048, seed=args.seed,
                                              cache_dir=args.cache_dir)
        elif which == "wikitext2-simple":
            return d1.load_wikitext2_simple(n_samples=args.n_calib)
        else:
            return d1.get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib,
                                                     seqlen=2048, seed=args.seed,
                                                     cache_dir=args.cache_dir)

    print(f"\nLoading calibration A: {args.calib_a}")
    calib_a = load_calib(args.calib_a)
    print(f"Loading calibration B: {args.calib_b}")
    calib_b = load_calib(args.calib_b)

    all_linear = [(n, m) for n, m in model.named_modules()
                  if isinstance(m, nn.Linear)
                  and not (('lm_head' in n.lower()) or n.endswith('lm_head'))]
    if args.layer_filter:
        all_linear = [(n, m) for n, m in all_linear if args.layer_filter in n]
    if args.sample_layers > 0 and args.sample_layers < len(all_linear):
        idx = sorted(random.Random(args.seed).sample(range(len(all_linear)),
                                                      args.sample_layers))
        all_linear = [all_linear[i] for i in idx]
        print(f"  Sampled {len(all_linear)} layers across the model.")
    elif args.max_layers > 0:
        all_linear = all_linear[:args.max_layers]
    print(f"\nTarget layers: {len(all_linear)}")

    recs_AtoB = []   # fit A, score B  (cross)
    recs_BtoA = []   # fit B, score A  (cross, other direction)
    recs_AtoA = []   # fit A, score A  (in-distribution control)

    bs = args.layer_batch_size
    n_batches = (len(all_linear) + bs - 1) // bs
    for b in range(n_batches):
        batch = all_linear[b * bs:(b + 1) * bs]
        print(f"\n[Batch {b+1}/{n_batches}] {len(batch)} layers "
              f"(collecting H for both distributions)")
        dual = collect_dual_H(d1, model, tokenizer, batch, calib_a, calib_b,
                              device, args.n_calib, args.max_tokens_per_sample)
        for name, module in batch:
            if dual.get(name) is None:
                print(f"    no activations for {name}, skipping")
                continue
            Ha, mua, Hb, mub = dual[name]

            # cross A->B
            recs_AtoB.extend(measure_layer_crosseval(
                d1, d2, name, module, Ha, mua, Hb, mub,
                args.bits, args.group_size, args.dprimes, args.n_rows,
                args.block_method, args.k_list, args.enum_threshold, device,
                seed=args.seed))
            # cross B->A
            recs_BtoA.extend(measure_layer_crosseval(
                d1, d2, name, module, Hb, mub, Ha, mua,
                args.bits, args.group_size, args.dprimes, args.n_rows,
                args.block_method, args.k_list, args.enum_threshold, device,
                seed=args.seed))
            # control A->A
            recs_AtoA.extend(measure_layer_crosseval(
                d1, d2, name, module, Ha, mua, Ha, mua,
                args.bits, args.group_size, args.dprimes, args.n_rows,
                args.block_method, args.k_list, args.enum_threshold, device,
                seed=args.seed))
            del Ha, mua, Hb, mub
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()

    cur_AtoB = summarize_crosseval(recs_AtoB, f"fit_{args.calib_a}->eval_{args.calib_b}")
    cur_BtoA = summarize_crosseval(recs_BtoA, f"fit_{args.calib_b}->eval_{args.calib_a}")
    cur_AtoA = summarize_crosseval(recs_AtoA, f"fit_{args.calib_a}->eval_{args.calib_a}")

    with open(args.output_json, "w") as f:
        json.dump({
            "config": vars(args),
            "curve_in_distribution": cur_AtoA,
            "curve_cross_AtoB": cur_AtoB,
            "curve_cross_BtoA": cur_BtoA,
        }, f, indent=2)

    _print_curve(f"IN-DISTRIBUTION CONTROL  (fit {args.calib_a} -> eval {args.calib_a})",
                 cur_AtoA)
    _print_curve(f"CROSS  (fit {args.calib_a} -> eval {args.calib_b})", cur_AtoB)
    _print_curve(f"CROSS  (fit {args.calib_b} -> eval {args.calib_a})", cur_BtoA)

    print("\nRead: if E2-CLC keeps rising with k in-distribution but PEAKS then")
    print("DROPS in the cross curves, that peak k* is the eigenvalue-separation")
    print("knee and your recommended operating point. A flat/monotone cross curve")
    print("means k is still under the noise floor -- you can push k higher.")
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()