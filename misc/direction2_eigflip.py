"""
Direction 2: Rung-2 Eigenspace-Aware Flip Selection
===================================================

Tests the doc's §4.2 hypothesis:

    Scoring flips against the projection of the residual onto the top-k eigenspace
    of Sigma (in addition to mu-hat) strictly improves over CLC at matched flip
    budget, with k <~ 16 sufficing.

CLC (rung 1) tracks a SCALAR channel state  b = mu^T r  and zeroes |b| by a
budgeted prefix scan of restricted 1-opt flips (anti-residual direction).

Rung 2 replaces b with the (k+1)-vector

    z = ( mu^T r ,  Lambda_k^{1/2} U_k^T r )  in R^{k+1}

where  Sigma_II ~= U_k Lambda_k U_k^T  is the top-k eig of the block covariance.
A flip of coord i changes the residual by  dr_i = -flip_dir_i * s_i  (residual
convention r = W - W_dq, matching the *corrected* CLC in the Direction-1 code),
so the state increment is

    dz_i = ( mu_i * dr_i ,  Lambda_k^{1/2} U_k^T[:, i] * dr_i )   in R^{k+1}.

The prefix objective  min_k |b^{(k)}|  generalizes to  min_k ||z^{(k)}||_2^2 ,
which upper-bounds BOTH coupled terms of Eq.(4) restricted to the identified
subspace. k=0 recovers CLC exactly (z collapses to b).

This module is OFFLINE / encoder-side only. It depends on the Direction-1 file
for: groupwise_asym_rtn, decompose_block_distortion, exact_cvp_block,
clc_block, select_block_indices, SecondMomentCollector, calibrate_and_collect,
James-Stein mean, and the calibration driver. Import it as `d1`.

NOTE on objective faithfulness vs. CLC's single-scan prefix
-----------------------------------------------------------
CLC works because the scalar state has ONE anti-residual direction: every valid
flip moves b toward 0 by a known signed amount, so a single argsort + cumulative
sum + argmin-over-prefixes finds the budgeted optimum of |b|. With a vector z
this is no longer true -- a flip that shrinks one z-component can grow another,
and there is no global sign that monotonically helps. So the honest rung-2
encoder is a GREEDY ||z||^2-descent: repeatedly take the single not-yet-used,
feasible flip that most reduces ||z||^2, stop when no flip helps or the budget
is hit. The no-flip option is feasible by construction (we stop on no-improve),
so the aggregate-descent guarantee (||z|| never increases) carries over. This is
still a single light pass: at most `cap` rounds, each O(d' * k). k=0 reduces to
choosing flips that reduce |b| -- i.e. CLC's set, possibly in a different order;
we therefore ALSO keep a `clc_compat` path that reuses d1.clc_block verbatim for
the k=0 column so the sweep's k=0 == CLC bitwise.
"""

import torch
import numpy as np
import argparse
import json
import random
import gc

# Direction-1 module is expected alongside this file (same conventions/pipeline).
import importlib.util as _ilu
import os as _os


def _load_d1(path):
    spec = _ilu.spec_from_file_location("d1", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Block eigendecomposition of Sigma_II (top-k), shared across rows of a layer
# =============================================================================

@torch.no_grad()
def block_topk_eig(Sigma_II, k):
    """
    Return (U_k [d', k], sqrtLam_k [k]) for the top-k eigenpairs of the
    symmetric block covariance Sigma_II. Eigenvalues clamped at 0 (Sigma_II is
    PSD in the population but the sample estimate can have tiny negatives).
    k is clamped to d'. For k=0 returns (None, None).
    """
    dprime = Sigma_II.shape[0]
    k = min(k, dprime)
    if k <= 0:
        return None, None
    # symmetric eig; ascending eigenvalues
    evals, evecs = torch.linalg.eigh(Sigma_II)
    evals = evals.clamp(min=0.0)
    # take the top-k (largest)
    idx = torch.argsort(evals, descending=True)[:k]
    lam = evals[idx]                      # [k]
    U = evecs[:, idx]                     # [d', k]
    sqrt_lam = torch.sqrt(lam)            # [k]
    return U, sqrt_lam


# =============================================================================
# Rung-2 encoder: greedy ||z||^2 descent with budget, no-flip feasible
# =============================================================================

@torch.no_grad()
def eigflip_block(w_I, w_int_I, s_I, zp_I, max_int, mu_I, Sigma_II, k,
                  max_flip_percent=1.0, knee_tolerance=-10.0,
                  U=None, sqrt_lam=None):
    """
    Rung-2 eigenspace-aware encoder on one block.

    State:  z = ( mu_I^T r , sqrt_lam .* (U^T r) )   in R^{1+k}
            with residual r = W - W_dq on the block.

    Flip of coord i (code += flip_dir_i): dr_i = -flip_dir_i * s_i.
    A is the [1+k, d'] matrix mapping a per-coord residual delta to a z delta:
        A[0, i]   = mu_I[i]
        A[1:, i]  = sqrt_lam * (U[:, i])           (elementwise per eigen-comp)
    so dz for flipping i is  A[:, i] * dr_i.

    Greedy: while budget remains, pick the single feasible unused flip that most
    reduces ||z||^2; apply it; stop when none reduces ||z||^2.

    Returns (q [d'], int_codes [d'], n_flips, info dict).
    """
    device = w_I.device
    dprime = w_I.numel()

    # ---- precompute U, sqrt_lam if not supplied (shared per layer normally) ----
    if k > 0 and (U is None or sqrt_lam is None):
        U, sqrt_lam = block_topk_eig(Sigma_II, k)
    keff = 0 if (U is None) else U.shape[1]

    # ---- current RTN residual r = W - W_dq ----
    w_dq = (w_int_I - zp_I) * s_I
    r = (w_I - w_dq).clone()                          # [d']

    # ---- z-mapping matrix A: [1+keff, d'] ----
    rows = [mu_I.unsqueeze(0)]                         # [1, d']
    if keff > 0:
        # (U^T)[k, d'] scaled rowwise by sqrt_lam
        UT = U.t()                                     # [keff, d']
        rows.append(UT * sqrt_lam.unsqueeze(1))        # [keff, d']
    A = torch.cat(rows, dim=0)                          # [1+keff, d']

    z = A @ r                                           # [1+keff]

    # ---- flip direction toward the un-rounded value (matches CLC) ----
    c = w_I / s_I + zp_I
    flip_dir = torch.sign(c - w_int_I)
    flip_dir = torch.where(flip_dir == 0, torch.ones_like(flip_dir), flip_dir)
    # residual delta per coord if flipped:  dr_i = -flip_dir_i * s_i
    dr = -(flip_dir * s_I)                              # [d']

    # per-coord z-delta columns: dz_i = A[:, i] * dr_i  -> [1+keff, d']
    dZ = A * dr.unsqueeze(0)

    # ---- feasibility: must stay in [0, max_int]; outlier mask (knee=-10 -> none) ----
    proposed = w_int_I + flip_dir
    feas = (proposed >= 0) & (proposed <= max_int)
    feas = feas & (~d1_kneedle(mu_I, knee_tolerance))

    cap = max(1, int(max_flip_percent * dprime))

    int_codes = w_int_I.clone()
    used = torch.zeros(dprime, dtype=torch.bool, device=device)
    n_flips = 0

    # ||z + dz_i||^2 - ||z||^2 = 2 z . dz_i + ||dz_i||^2
    dz_sqnorm = (dZ * dZ).sum(dim=0)                    # [d'] (constant per coord)

    while n_flips < cap:
        # gain_i = current improvement in ||z||^2 from flipping i (negative = good)
        delta = 2.0 * (z @ dZ) + dz_sqnorm             # [d']
        mask = feas & (~used)
        if not bool(mask.any()):
            break
        big = torch.full_like(delta, float("inf"))
        cand = torch.where(mask, delta, big)
        i = int(torch.argmin(cand).item())
        if not (cand[i].item() < -1e-18):              # no flip strictly helps
            break
        # apply flip i
        z = z + dZ[:, i]
        int_codes[i] = (int_codes[i] + flip_dir[i]).clamp(0, max_int)
        used[i] = True
        n_flips += 1

    q = (int_codes - zp_I) * s_I
    info = {"keff": keff, "z_final_sqnorm": float((z * z).sum().item())}
    return q, int_codes, n_flips, info


# small shim so we don't import the private name; mirrors d1._kneedle_outlier_mask
def d1_kneedle(mu_I, tol):
    n = mu_I.numel()
    a = mu_I.abs()
    if n < 3:
        return torch.zeros_like(a, dtype=torch.bool)
    sorted_desc, _ = torch.sort(a, descending=True)
    y = sorted_desc.cpu().float().numpy()
    y_min, y_max = y.min(), y.max()
    if y_max - y_min < 1e-10:
        return torch.zeros_like(a, dtype=torch.bool)
    y_norm = (y - y_min) / (y_max - y_min)
    x_norm = np.linspace(0, 1, n)
    y_line = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
    knee = int(np.argmax(np.abs(y_norm - y_line)))
    knee = max(0, min(knee + int(tol * n), n - 1))
    if int(tol * n) <= -knee:
        return torch.zeros_like(a, dtype=torch.bool)
    threshold = sorted_desc[knee].item()
    return a > threshold


# =============================================================================
# Per-layer rung-2 sweep over k, with exact CVP and CLC references
# =============================================================================

@torch.no_grad()
def measure_layer_eigflip(d1, name, module, H, mu, bits, group_size,
                          dprimes, n_rows, block_method, k_list,
                          enum_threshold, device, seed=0):
    """
    For sampled rows and each (d', k): RTN, exact CVP, CLC (k=0 reference via
    d1.clc_block), and rung-2 eigflip block distortions, plus the off-eigenspace
    diagnostic (how much off-diagonal mass lives inside the top-k subspace).
    """
    W = module.weight.data.to(device).float()
    out_f, in_f = W.shape
    H = H.to(device).float()
    mu = mu.to(device).float()
    Sigma = H - torch.outer(mu, mu)

    _, W_int, scale_flat, zp_flat = d1.groupwise_asym_rtn(W, bits, group_size)
    max_int = 2 ** bits - 1
    rng = random.Random(seed)
    rows = rng.sample(range(out_f), min(n_rows, out_f))

    records = []
    for dprime in dprimes:
        block = d1.select_block_indices(mu, H, dprime, method=block_method).to(device)
        H_II = H[block][:, block]
        Sigma_II = Sigma[block][:, block]
        mu_I = mu[block]

        # precompute eig once per (layer, d'); reuse across rows and k via slicing
        kmax = min(max(k_list), dprime)
        U_full, sqrt_full = block_topk_eig(Sigma_II, kmax) if kmax > 0 else (None, None)

        # off-eigenspace residual row-sum rho(Sigma - U_k Lam_k U_k^T) per k
        # (the doc's tightening quantity); compute once per (d', k)
        offeig_rowsum = {}
        absSig_offdiag = (Sigma_II - torch.diag(Sigma_II.diagonal())).abs()
        full_offdiag_rowsum = float(absSig_offdiag.sum(dim=1).max().item())
        for k in k_list:
            kk = min(k, dprime)
            if kk <= 0:
                offeig_rowsum[k] = full_offdiag_rowsum
                continue
            Uk = U_full[:, :kk]
            lamk = (sqrt_full[:kk] ** 2)
            Sig_k = (Uk * lamk.unsqueeze(0)) @ Uk.t()
            resid = Sigma_II - Sig_k
            resid_off = (resid - torch.diag(resid.diagonal())).abs()
            offeig_rowsum[k] = float(resid_off.sum(dim=1).max().item())

        for r in rows:
            w_I = W[r, block]
            wint_I = W_int[r, block]
            s_I = scale_flat[r, block]
            zp_I = zp_flat[r, block]

            w_dq_rtn = (wint_I - zp_I) * s_I
            e_rtn = w_dq_rtn - w_I
            dist_rtn = float(e_rtn @ (H_II @ e_rtn))

            q_ex, _, dist_ex = d1.exact_cvp_block(
                w_I, s_I, zp_I, max_int, H_II, enum_threshold=enum_threshold)

            # CLC reference (rung 1) -- verbatim Direction-1 encoder
            q_clc, _, n_clc = d1.clc_block(
                w_I, wint_I, s_I, zp_I, max_int, mu_I,
                knee_tolerance=-10.0, max_flip_percent=1.0)
            e_clc = q_clc - w_I
            dist_clc = float(e_clc @ (H_II @ e_clc))

            gap = dist_rtn - dist_ex
            for k in k_list:
                kk = min(k, dprime)
                if kk > 0:
                    U = U_full[:, :kk]
                    sqrt_lam = sqrt_full[:kk]
                else:
                    U = sqrt_lam = None
                q_e2, _, n_e2, info = eigflip_block(
                    w_I, wint_I, s_I, zp_I, max_int, mu_I, Sigma_II, kk,
                    max_flip_percent=1.0, knee_tolerance=-10.0,
                    U=U, sqrt_lam=sqrt_lam)
                e_e2 = q_e2 - w_I
                dist_e2 = float(e_e2 @ (H_II @ e_e2))

                clc_rec = dist_rtn - dist_clc
                e2_rec = dist_rtn - dist_e2
                records.append({
                    "layer": name, "row": r, "dprime": dprime, "k": k,
                    "dist_rtn": dist_rtn, "dist_exact": dist_ex,
                    "dist_clc": dist_clc, "dist_eigflip": dist_e2,
                    "gap": gap,
                    "clc_recovered": clc_rec,
                    "eigflip_recovered": e2_rec,
                    "eigflip_minus_clc": e2_rec - clc_rec,   # >0 => rung2 beats CLC
                    "n_flips_clc": n_clc, "n_flips_eigflip": n_e2,
                    "offeig_rowsum": offeig_rowsum[k],
                    "full_offdiag_rowsum": full_offdiag_rowsum,
                })
    del W, H, mu, Sigma
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


# =============================================================================
# Aggregation: rung-2 vs CLC headroom curve over k
# =============================================================================

def summarize_eigflip(records):
    by = {}
    for rec in records:
        by.setdefault((rec["dprime"], rec["k"]), []).append(rec)

    curve = []
    for (dprime, k) in sorted(by):
        rs = by[(dprime, k)]
        rtn = np.array([r["dist_rtn"] for r in rs])
        gap = np.array([r["gap"] for r in rs])
        clc = np.array([r["clc_recovered"] for r in rs])
        e2 = np.array([r["eigflip_recovered"] for r in rs])
        delta = np.array([r["eigflip_minus_clc"] for r in rs])
        offeig = np.array([r["offeig_rowsum"] for r in rs])

        sgap = float(gap.sum())
        wins = int((delta > 1e-15).sum())
        ties = int((np.abs(delta) <= 1e-15).sum())
        losses = int((delta < -1e-15).sum())

        curve.append({
            "dprime": dprime, "k": k, "n": len(rs),
            # pooled fraction of the recoverable gap captured
            "clc_frac_of_gap": float(clc.sum()) / sgap if sgap > 0 else 0.0,
            "eigflip_frac_of_gap": float(e2.sum()) / sgap if sgap > 0 else 0.0,
            "eigflip_minus_clc_frac_of_gap": (float(delta.sum()) / sgap
                                              if sgap > 0 else 0.0),
            # per-row improvement distribution
            "rows_eigflip_beats_clc": wins,
            "rows_tie": ties,
            "rows_eigflip_worse": losses,
            "delta_median_abs": float(np.median(delta)),
            "delta_mean_abs": float(delta.mean()),
            # the doc's tightening quantity rho(Sigma - U_k Lam U_k^T)
            "offeig_rowsum_mean": float(offeig.mean()),
        })
    return {"eigflip_curve": curve}


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Direction 2: rung-2 eigenspace-aware flip selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--d1-path", type=str, default="./headroom_d1.py",
                   help="Path to the Direction-1 file (provides shared machinery).")
    p.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    p.add_argument("--output-json", type=str, default="./eigflip_d2_results.json")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--calib-dataset", type=str, default="c4",
                   choices=["c4", "wikitext2", "wikitext2-simple"])
    p.add_argument("--cache-dir", type=str, default="./calibration_cache")
    p.add_argument("--max-tokens-per-sample", type=int, default=2048)
    p.add_argument("--layer-batch-size", type=int, default=8)

    p.add_argument("--dprimes", type=int, nargs="+", default=[16, 24])
    p.add_argument("--k-list", type=int, nargs="+",
                   default=[0, 1, 2, 4, 8, 16, 32],
                   help="Eigenspace dims to sweep; 0 == CLC.")
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
        raise FileNotFoundError(
            f"Direction-1 file not found at {args.d1_path}. Pass --d1-path.")
    d1 = _load_d1(args.d1_path)

    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Direction 2: Rung-2 Eigenspace-Aware Flip Selection")
    print(f"Model: {args.model_path} | bits={args.bits} | g={args.group_size}")
    print(f"d' grid: {args.dprimes} | k grid: {args.k_list} | "
          f"rows/layer: {args.n_rows} | block: {args.block_method}")
    print(f"k=0 == CLC (rung 1) reference")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    print(f"\nLoading calibration: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib = d1.get_c4_calibration_data(tokenizer, n_samples=args.n_calib,
                                           seqlen=2048, seed=args.seed,
                                           cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        calib = d1.load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib = d1.get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib,
                                                  seqlen=2048, seed=args.seed,
                                                  cache_dir=args.cache_dir)

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

    collector = d1.SecondMomentCollector(
        max_tokens_per_sample=args.max_tokens_per_sample)
    all_records = []

    bs = args.layer_batch_size
    n_batches = (len(all_linear) + bs - 1) // bs
    for b in range(n_batches):
        batch = all_linear[b * bs:(b + 1) * bs]
        print(f"\n[Batch {b+1}/{n_batches}] {len(batch)} layers")
        d1.calibrate_and_collect(model, tokenizer, batch, calib,
                                 collector, device, args.n_calib, max_len=512)
        for name, module in batch:
            H, mu = collector.finalize(name, use_james_stein=True)
            if H is None:
                print(f"    no activations for {name}, skipping")
                continue
            recs = measure_layer_eigflip(
                d1, name, module, H, mu, args.bits, args.group_size,
                args.dprimes, args.n_rows, args.block_method, args.k_list,
                args.enum_threshold, device, seed=args.seed)
            all_records.extend(recs)
            collector.clear(name)
            del H, mu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        collector.clear()
        gc.collect()

    summary = summarize_eigflip(all_records)
    summary["config"] = vars(args)
    with open(args.output_json, "w") as f:
        json.dump({"records": all_records, "summary": summary}, f, indent=2)

    print("\n" + "=" * 80)
    print("EIGFLIP CURVE  (pooled frac of recoverable gap; k=0 == CLC)")
    print("=" * 80)
    print(f"{'d':>4} {'k':>4} {'n':>6} {'CLC/gap':>8} {'E2/gap':>8} "
          f"{'E2-CLC':>8} {'win':>5} {'tie':>5} {'lose':>5} {'offeigRS':>9}")
    cur = summary["eigflip_curve"]
    for c in cur:
        print(f"{c['dprime']:>4} {c['k']:>4} {c['n']:>6} "
              f"{c['clc_frac_of_gap']*100:>7.1f}% "
              f"{c['eigflip_frac_of_gap']*100:>7.1f}% "
              f"{c['eigflip_minus_clc_frac_of_gap']*100:>+7.2f}% "
              f"{c['rows_eigflip_beats_clc']:>5} "
              f"{c['rows_tie']:>5} "
              f"{c['rows_eigflip_worse']:>5} "
              f"{c['offeig_rowsum_mean']:>9.2e}")
    print(f"\nSaved {len(all_records)} records to {args.output_json}")


if __name__ == "__main__":
    main()