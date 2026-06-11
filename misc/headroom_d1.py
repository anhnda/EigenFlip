"""
Direction 1: Encoding Headroom Measurement
===========================================

How far is coordinate-wise RTN from the per-channel CVP optimum under H?

For each channel j and a small sub-block I of d' coordinates (the rest held at RTN),
solve EXACTLY:

        min_{q in s_j Z^{d'}}  (q - w_j)_I^T  H_II  (q - w_j)_I            (Eq. 3)

by enumerating the 2^{d'} up/down configurations (each block coordinate takes one of
the two lattice levels bracketing its real value). Then:

  - measure exact-vs-RTN distortion gap per block;
  - attribute the gap to the rank-one (mu^T e_j)^2 term vs the off-diagonal Sigma term
    via Eq. (4):

        e_j^T H e_j = (mu^T e_j)^2 + sum_i Sigma_ii e_ij^2 + sum_{i!=k} Sigma_ik e_ij e_kj
                      \_____________/  \________________/    \________________________/
                        rank-one         diagonal              off-diagonal

  - measure the fraction of the gap recovered by CLC's restricted 1-opt (rung 1).

Deliverable: a "headroom curve" -> recoverable distortion vs. rung, per bit-width.

Conventions match awq_js_xl.py / raw RTN:
  - group-wise asymmetric quantization [0, 2^bits - 1]
  - per-channel here means per OUTPUT row e_j := (Wq - W)[j, :], a vector over in_features.
    H is the in_features x in_features second moment of the layer INPUT activations.
  - activation collection via forward hooks, James-Stein-shrunk mu (rung 1 statistic).

CLC defaults (per request):
  - knee_tolerance = -10.0  -> "no knee": Kneedle offset pushed so far negative that
    NO channel is masked as an outlier (entire pool is flip-eligible).
  - max_flip_percent = 1.0  -> cap = in_features: take all beneficial flips if possible.

This is OFFLINE / encoder-side only. No model is modified or saved.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import json
import argparse
import random
import itertools
import numpy as np
import gc

try:
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data
except ImportError:
    print("⚠️  calibration_utils not found. Use --calib-dataset wikitext2-simple as fallback.")
    def get_c4_calibration_data(*a, **k):
        raise NotImplementedError("calibration_utils.py missing")
    def get_wikitext2_calibration_data(*a, **k):
        raise NotImplementedError("calibration_utils.py missing")


# =============================================================================
# James-Stein shrinkage (rung-1 statistic), copied from awq_js_xl.py for parity
# =============================================================================

def compute_james_stein_mean(raw_means, variance_estimate=None):
    p = raw_means.numel()
    if p < 3:
        return raw_means
    grand_mean = raw_means.mean()
    deviations = raw_means - grand_mean
    sum_sq_dev = (deviations ** 2).sum()
    if sum_sq_dev < 1e-10:
        return raw_means
    if variance_estimate is None:
        variance_estimate = ((raw_means - grand_mean).abs().mean()) ** 2
        variance_estimate = variance_estimate.clamp(min=1e-8)
    c = ((p - 2) * variance_estimate) / sum_sq_dev
    c = c.clamp(0.0, 1.0)
    return grand_mean + (1.0 - c) * deviations


# =============================================================================
# Second-moment estimator H = E[x x^T]  (full, on a chosen support)
# =============================================================================

class SecondMomentCollector:
    """
    Accumulates per-layer second moment H = (1/N) sum_t x_t x_t^T and
    mean mu = (1/N) sum_t x_t  for the INPUT activations of each Linear.

    H is in_features x in_features. For 4096-d layers that is 4096^2 * 4 bytes
    ~= 67 MB per layer in fp32 -- fine for a handful of layers, collected one
    layer-batch at a time exactly like awq_js_xl.py.
    """

    def __init__(self, max_tokens_per_sample=2048):
        self.max_tokens_per_sample = max_tokens_per_sample
        self.HtX = {}     # name -> [in, in] running sum of x x^T (fp64 on CPU)
        self.sumX = {}    # name -> [in]     running sum of x
        self.count = {}   # name -> int      token count

    def get_hook(self, name):
        def hook(_module, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 3 and x.shape[1] > self.max_tokens_per_sample:
                seq = x.shape[1]
                idx = torch.randperm(seq, device=x.device)[:self.max_tokens_per_sample]
                idx = idx.sort()[0]
                x = x[:, idx, :]
            # Keep the matmul ON DEVICE in fp32. x^T x is [in,in] but the GEMM
            # runs on GPU; only the running accumulator is touched per call.
            # (CPU fp64 outer-product every forward was the calibration bottleneck.)
            x = x.detach().reshape(-1, x.shape[-1]).float()        # [T, in] on device
            xtx = x.t() @ x                                        # [in, in] on device, fp32
            xs = x.sum(dim=0)                                      # [in] on device
            if name not in self.HtX:
                self.HtX[name] = xtx.clone()
                self.sumX[name] = xs.clone()
                self.count[name] = x.shape[0]
            else:
                self.HtX[name] += xtx
                self.sumX[name] += xs
                self.count[name] += x.shape[0]
        return hook

    def finalize(self, name, use_james_stein=True):
        """Return (H [in,in] fp32 CPU, mu [in] fp32 CPU) or (None, None)."""
        if name not in self.count or self.count[name] == 0:
            return None, None
        n = self.count[name]
        H = (self.HtX[name] / n).float().cpu()   # [in, in] -> CPU once, at the end
        mu = (self.sumX[name] / n).float().cpu()  # [in]
        if use_james_stein:
            mu = compute_james_stein_mean(mu)
        return H, mu

    def clear(self, name=None):
        if name is None:
            self.HtX.clear(); self.sumX.clear(); self.count.clear()
        else:
            self.HtX.pop(name, None); self.sumX.pop(name, None); self.count.pop(name, None)


# =============================================================================
# Group-wise asymmetric RTN -> returns dequant W AND the integer / scale / zp
# (so we can build sub-block lattices consistent with the real pipeline)
# =============================================================================

@torch.no_grad()
def groupwise_asym_rtn(W, bits, group_size):
    """
    Returns:
        W_dq      : [out, in] dequantized (fake-quant) weight
        W_int     : [out, in] integer codes (RTN), float
        scale_flat: [out, in] per-coordinate step s
        zp_flat   : [out, in] per-coordinate zero point
    """
    out_f, in_f = W.shape
    device, dtype = W.device, W.dtype
    n_groups = (in_f + group_size - 1) // group_size
    padded = n_groups * group_size

    if padded > in_f:
        Wp = torch.zeros(out_f, padded, device=device, dtype=dtype)
        Wp[:, :in_f] = W
    else:
        Wp = W

    Wg = Wp.reshape(out_f, n_groups, group_size)
    w_min = Wg.min(dim=2, keepdim=True)[0]
    w_max = Wg.max(dim=2, keepdim=True)[0]
    max_int = 2 ** bits - 1
    scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
    zp = torch.round(-w_min / scale).clamp(0, max_int)

    scale_flat = scale.repeat(1, 1, group_size).reshape(out_f, padded)
    zp_flat = zp.repeat(1, 1, group_size).reshape(out_f, padded)

    W_int = torch.round(Wp / scale_flat + zp_flat).clamp(0, max_int)
    W_dq = (W_int - zp_flat) * scale_flat

    if padded > in_f:
        W_dq = W_dq[:, :in_f]
        W_int = W_int[:, :in_f]
        scale_flat = scale_flat[:, :in_f]
        zp_flat = zp_flat[:, :in_f]
    return W_dq.to(dtype), W_int, scale_flat, zp_flat


# =============================================================================
# Eq. (4) term decomposition for a residual vector e_j over the block support
# =============================================================================

def decompose_block_distortion(e_I, mu_I, Sigma_II):
    """
    e_I       : [d'] residual (Wq - W) restricted to block I
    mu_I      : [d'] mean restricted to block I
    Sigma_II  : [d', d'] covariance restricted to block I  (Sigma = H - mu mu^T)
    Returns dict with rank_one, diagonal, offdiag, total (== e^T H_II e).
    """
    rank_one = float((mu_I @ e_I) ** 2)
    diag = float((Sigma_II.diagonal() * e_I * e_I).sum())
    full_sigma = float(e_I @ (Sigma_II @ e_I))
    offdiag = full_sigma - diag
    total = rank_one + full_sigma                      # = e^T H e on the block
    return {"rank_one": rank_one, "diagonal": diag,
            "offdiag": offdiag, "total": total}


# =============================================================================
# Exact CVP on a block by enumeration over the 2^{d'} up/down configs
# =============================================================================

@torch.no_grad()
def exact_cvp_block(w_I, s_I, zp_I, max_int, H_II, enum_threshold=24, max_int_codes=None):
    """
    Exact closest-vector solve on block I, restricted to the two lattice levels
    bracketing each real weight (floor and floor+1 in INTEGER code space).

    w_I    : [d'] real weights on the block
    s_I    : [d'] per-coordinate step
    zp_I   : [d'] per-coordinate zero point
    H_II   : [d', d'] second moment on the block
    Returns:
        q_best   : [d'] dequantized optimal weights
        int_best : [d'] integer codes of the optimum
        dist_best: scalar  (q_best - w_I)^T H (q_best - w_I)
    """
    dprime = w_I.numel()
    device = w_I.device

    # integer code in real domain: c = w/s + zp ; the two bracketing levels are
    # floor(c) and floor(c)+1, both clamped to [0, max_int].
    c = w_I / s_I + zp_I
    lo = torch.floor(c).clamp(0, max_int)
    hi = (lo + 1).clamp(0, max_int)
    levels = torch.stack([lo, hi], dim=1)             # [d', 2] integer codes

    # dequantized candidates per coordinate: (code - zp) * s
    deq = (levels - zp_I.unsqueeze(1)) * s_I.unsqueeze(1)   # [d', 2]

    if dprime <= enum_threshold:
        # full enumeration of 2^{d'} configs
        n_cfg = 1 << dprime
        # build all configs as a [n_cfg, d'] index tensor in {0,1}
        # (vectorized bit expansion)
        ar = torch.arange(n_cfg, device=device)
        bits = ((ar.unsqueeze(1) >> torch.arange(dprime, device=device)) & 1)  # [n_cfg, d']
        # gather dequant value per config
        q_all = torch.gather(deq.unsqueeze(0).expand(n_cfg, -1, -1), 2,
                             bits.unsqueeze(2)).squeeze(2)    # [n_cfg, d']
        diff = q_all - w_I.unsqueeze(0)                       # [n_cfg, d']
        # distortion = diff @ H @ diff  for each config
        Hd = diff @ H_II                                      # [n_cfg, d']
        dist = (Hd * diff).sum(dim=1)                         # [n_cfg]
        best = int(torch.argmin(dist).item())
        bit_best = bits[best]                                 # [d']
        q_best = torch.gather(deq, 1, bit_best.unsqueeze(1)).squeeze(1)
        int_best = torch.gather(levels, 1, bit_best.unsqueeze(1)).squeeze(1)
        return q_best, int_best, float(dist[best].item())
    else:
        # branch-and-bound fallback for d' up to ~32
        return _branch_and_bound_block(w_I, deq, levels, H_II)


@torch.no_grad()
def _branch_and_bound_block(w_I, deq, levels, H_II):
    """
    Depth-first B&B over the 2-level choice per coordinate.
    Lower bound: completed-prefix quadratic + diagonal-only relaxation of the
    remaining coordinates (each remaining coord picks its individually-best level
    against the diagonal of H), which underestimates the true coupled cost.
    For d' <= 32 this is tractable; ordering by |H diagonal * gap| helps pruning.
    """
    dprime = w_I.numel()
    device = w_I.device
    Hdiag = H_II.diagonal()

    # per-coordinate best/worst diagonal cost for the LB
    diff_lvl = deq - w_I.unsqueeze(1)                      # [d', 2]
    diag_cost = Hdiag.unsqueeze(1) * diff_lvl * diff_lvl   # [d', 2]
    min_diag = diag_cost.min(dim=1)[0]                     # [d']
    # ordering: descend coordinates with the largest diagonal spread first
    spread = (diag_cost[:, 0] - diag_cost[:, 1]).abs()
    order = torch.argsort(spread, descending=True).tolist()

    suffix_min = torch.zeros(dprime + 1, device=device)
    for t in range(dprime - 1, -1, -1):
        suffix_min[t] = suffix_min[t + 1] + min_diag[order[t]]

    best = {"dist": float("inf"), "bits": None}

    def recurse(depth, chosen_diff, partial_quad):
        # partial_quad = chosen^T H_subblock chosen for already-fixed coords (exact)
        lb = partial_quad + float(suffix_min[depth].item())
        if lb >= best["dist"]:
            return
        if depth == dprime:
            if partial_quad < best["dist"]:
                best["dist"] = partial_quad
                best["bits"] = list(chosen_diff)  # list of (coord, level)
            return
        coord = order[depth]
        for lvl in (0, 1):
            d_val = float(diff_lvl[coord, lvl].item())
            # incremental quadratic: 2 * d_val * sum_k H[coord,k]*prev_diff_k + H[coord,coord]*d_val^2
            inc = Hdiag[coord].item() * d_val * d_val
            for (c2, l2) in chosen_diff:
                inc += 2.0 * d_val * float(diff_lvl[c2, l2].item()) * float(H_II[coord, c2].item())
            recurse(depth + 1, chosen_diff + [(coord, lvl)], partial_quad + inc)

    recurse(0, [], 0.0)

    bit_best = torch.zeros(dprime, dtype=torch.long, device=device)
    for (coord, lvl) in best["bits"]:
        bit_best[coord] = lvl
    q_best = torch.gather(deq, 1, bit_best.unsqueeze(1)).squeeze(1)
    int_best = torch.gather(levels, 1, bit_best.unsqueeze(1)).squeeze(1)
    return q_best, int_best, best["dist"]


# =============================================================================
# CLC restricted 1-opt on the same block (rung 1)
# knee=-10 -> no outlier mask ; max_flip=1.0 -> cap = d' (all beneficial flips)
# =============================================================================

@torch.no_grad()
def clc_block(w_I, w_int_I, s_I, zp_I, max_int, mu_I,
              knee_tolerance=-10.0, max_flip_percent=1.0):
    """
    Single budgeted pass of restricted 1-opt against the rank-one term only,
    matching CLC at rung 1. Moves are limited to the adjacent level in the
    anti-residual direction, Delta = -sign(e) * s, scored by closeness-to-0.5
    rounding regret, with the no-move (prefix) option kept feasible.

    With knee_tolerance very negative -> threshold pushed so no channel is an
    outlier (whole block is flip-eligible). With max_flip_percent = 1.0 -> the
    cap equals d', so every beneficial flip in the prefix may be taken.

    Returns:
        q_clc   : [d'] dequantized weights after CLC flips
        int_clc : [d'] integer codes after flips
        n_flips : int
    """
    dprime = w_I.numel()
    device = w_I.device

    # RTN residual on the block (Wq - W)
    w_dq = (w_int_I - zp_I) * s_I
    # Use the SAME sign convention as awq_js_xl.py: residual r = W - W_dq, and
    # current rank-one state b = mu^T r. (clc_block previously used W_dq - W with
    # a negated flip_dir but did NOT flip the prefix target / validity test, so
    # the two conventions were inconsistent and CLC anti-optimized |b| ~67% of
    # the time. Verified: this convention never worsens |b|.)
    r = w_I - w_dq                                   # [d']  residual (W - W_dq)
    b = float((mu_I * r).sum().item())               # rank-one channel state

    # flip_dir = sign(c - W_int): the direction that moves the code toward the
    # un-rounded value (un-negated, matching the original).
    c = w_I / s_I + zp_I
    flip_dir = torch.sign(c - w_int_I)
    flip_dir = torch.where(flip_dir == 0, torch.ones_like(flip_dir), flip_dir)

    # impact of flipping coord i on b: delta_b = mu_i * flip_dir_i * s_i
    flip_impacts = mu_I * flip_dir * s_I             # [d']

    # validity: flip must reduce |b| -> sign(impact) == sign(b); stay in [0,max_int]
    target_sign = np.sign(b) if b != 0 else 1.0
    valid = (torch.sign(flip_impacts) == target_sign)
    proposed = w_int_I + flip_dir
    valid = valid & (proposed >= 0) & (proposed <= max_int)

    # outlier mask via Kneedle on |mu| -- with knee=-10 effectively masks nothing
    valid = valid & (~_kneedle_outlier_mask(mu_I, knee_tolerance))

    # rounding regret = |c - round(c)| in [0, 0.5], closeness to 0.5 -> cheaper flip
    regret = (c - w_int_I).abs()
    regret_masked = regret.clone()
    regret_masked[~valid] = -1.0

    order = torch.argsort(regret_masked, descending=True)
    imp_sorted = flip_impacts[order] * valid[order].float()

    # prefix that best zeroes b ; no-flip option (k=0) kept feasible
    cumsum = torch.cumsum(imp_sorted, dim=0)
    resid = (b - cumsum).abs()
    all_resid = torch.cat([torch.tensor([abs(b)], device=device), resid])
    best_k = int(torch.argmin(all_resid).item())

    # cap = max_flip_percent * d'  (=> d' when percent=1.0)
    cap = max(1, int(max_flip_percent * dprime))
    best_k = min(best_k, cap)

    take = order[:best_k][valid[order[:best_k]]]
    int_clc = w_int_I.clone()
    int_clc[take] = (int_clc[take] + flip_dir[take]).clamp(0, max_int)
    q_clc = (int_clc - zp_I) * s_I
    return q_clc, int_clc, int(take.numel())


def _kneedle_outlier_mask(mu_I, tolerance_offset):
    """
    Returns a boolean mask of 'outlier' coordinates (to be excluded from flips).
    Mirrors find_knee_point logic on sorted |mu| descending. A very negative
    tolerance_offset shifts the knee index below 0 -> threshold = max(|mu|) is
    never exceeded -> NO outliers masked.
    """
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
    knee = max(0, min(knee + int(tolerance_offset * n), n - 1))
    if int(tolerance_offset * n) <= -knee:           # pushed below 0 -> no mask
        return torch.zeros_like(a, dtype=torch.bool)
    threshold = sorted_desc[knee].item()
    return a > threshold


# =============================================================================
# Block selection per output row
# =============================================================================

def select_block_indices(mu, H, dprime, method="abs_mu"):
    """
    Choose d' input-coordinate indices to form the sub-block.
      abs_mu  : top-d' by |mu_i|                    (rank-one-relevant support)
      eigvec  : top-d' by |top-eigenvector loading| (off-diagonal-relevant support)
    Returns LongTensor [d'].
    """
    in_f = mu.numel()
    dprime = min(dprime, in_f)
    if method == "abs_mu":
        return torch.topk(mu.abs(), dprime).indices.sort()[0]
    elif method == "eigvec":
        Sigma = H - torch.outer(mu, mu)
        # power iteration for the top eigenvector (cheap, avoids full eigdecomp)
        v = torch.randn(in_f, device=H.device)
        v /= v.norm()
        for _ in range(50):
            v = Sigma @ v
            v /= (v.norm() + 1e-12)
        return torch.topk(v.abs(), dprime).indices.sort()[0]
    else:
        raise ValueError(f"unknown block method {method}")


# =============================================================================
# Per-layer headroom measurement
# =============================================================================

@torch.no_grad()
def measure_layer_headroom(name, module, H, mu, bits, group_size,
                           dprimes, n_rows, block_method,
                           enum_threshold, device, seed=0):
    """
    For a sample of output rows and each d', compute RTN / exact / CLC block
    distortions and the Eq.(4) decomposition of the RTN->exact gap.
    Returns a list of per-(row, d') record dicts.
    """
    W = module.weight.data.to(device).float()   # bf16 -> fp32 to match H/mu
    out_f, in_f = W.shape
    H = H.to(device).float()
    mu = mu.to(device).float()
    Sigma = H - torch.outer(mu, mu)

    # full-layer RTN once (consistent scale/zp/codes with the real pipeline)
    _, W_int, scale_flat, zp_flat = groupwise_asym_rtn(W, bits, group_size)

    max_int = 2 ** bits - 1
    rng = random.Random(seed)
    rows = rng.sample(range(out_f), min(n_rows, out_f))

    records = []
    for dprime in dprimes:
        # block selection is input-side (shared across rows): pick once per d'
        block = select_block_indices(mu, H, dprime, method=block_method).to(device)
        H_II = H[block][:, block]
        Sigma_II = Sigma[block][:, block]
        mu_I = mu[block]

        for r in rows:
            w_I = W[r, block]
            wint_I = W_int[r, block]
            s_I = scale_flat[r, block]
            zp_I = zp_flat[r, block]

            # RTN residual & distortion on the block
            w_dq_rtn = (wint_I - zp_I) * s_I
            e_rtn = w_dq_rtn - w_I
            dist_rtn = float(e_rtn @ (H_II @ e_rtn))
            dec_rtn = decompose_block_distortion(e_rtn, mu_I, Sigma_II)

            # EXACT CVP on the block
            q_ex, int_ex, dist_ex = exact_cvp_block(
                w_I, s_I, zp_I, max_int, H_II, enum_threshold=enum_threshold)
            e_ex = q_ex - w_I
            dec_ex = decompose_block_distortion(e_ex, mu_I, Sigma_II)

            # CLC restricted 1-opt on the block (rung 1)
            q_clc, int_clc, n_flips = clc_block(
                w_I, wint_I, s_I, zp_I, max_int, mu_I,
                knee_tolerance=-10.0, max_flip_percent=1.0)
            e_clc = q_clc - w_I
            dist_clc = float(e_clc @ (H_II @ e_clc))

            gap = dist_rtn - dist_ex                       # total recoverable headroom
            clc_recov = dist_rtn - dist_clc                # recovered by rung 1
            # attribution of the gap to each term (RTN term - exact term)
            gap_rank_one = dec_rtn["rank_one"] - dec_ex["rank_one"]
            gap_offdiag = dec_rtn["offdiag"] - dec_ex["offdiag"]
            gap_diag = dec_rtn["diagonal"] - dec_ex["diagonal"]

            records.append({
                "layer": name, "row": r, "dprime": dprime,
                "dist_rtn": dist_rtn, "dist_exact": dist_ex, "dist_clc": dist_clc,
                "gap": gap,
                "gap_frac": (gap / dist_rtn) if dist_rtn > 0 else 0.0,
                "clc_recovered": clc_recov,
                "clc_recovered_frac_of_gap": (clc_recov / gap) if gap > 1e-12 else 0.0,
                "gap_rank_one": gap_rank_one,
                "gap_offdiag": gap_offdiag,
                "gap_diagonal": gap_diag,
                "gap_rank_one_frac": (gap_rank_one / gap) if gap > 1e-12 else 0.0,
                "gap_offdiag_frac": (gap_offdiag / gap) if gap > 1e-12 else 0.0,
                "n_flips": n_flips,
            })
    del W, H, mu, Sigma
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


# =============================================================================
# Calibration driver (layer-batched, mirrors awq_js_xl.py)
# =============================================================================

def calibrate_and_collect(model, tokenizer, layer_batch, calib_texts,
                          collector, device, n_samples, max_len=512):
    collector.clear()
    handles = [m.register_forward_hook(collector.get_hook(n)) for n, m in layer_batch]
    with torch.no_grad():
        for i, text in enumerate(tqdm(calib_texts[:n_samples],
                                      desc="  Calibration", leave=False)):
            try:
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=max_len)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs, use_cache=False, return_dict=True)
                if (i + 1) % 32 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                continue
    for h in handles:
        h.remove()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Aggregation -> headroom curve
# =============================================================================

def summarize(records):
    """
    Build the headroom curve per d'.

    Aggregation note: the gap fraction and the term-attribution fractions are
    reported as RATIO-OF-SUMS (pool numerators and denominators across rows),
    not mean-of-per-row-ratios. Per-row ratios blow up on rows where the
    denominator (gap, or RTN distortion) is ~0 -- a single near-zero-gap row
    makes mean(clc_recovered/gap) explode to +/-1e6. Pooling is the right
    estimator for "what fraction of total recoverable distortion does X recover".
    Per-row medians are kept alongside as a robust cross-check.
    """
    out = {}
    by_dprime = {}
    for rec in records:
        by_dprime.setdefault(rec["dprime"], []).append(rec)

    curve = []
    for dprime in sorted(by_dprime):
        rs = by_dprime[dprime]
        dist_rtn = np.array([r["dist_rtn"] for r in rs])
        gap = np.array([r["gap"] for r in rs])
        clc_rec = np.array([r["clc_recovered"] for r in rs])
        g_ro = np.array([r["gap_rank_one"] for r in rs])
        g_od = np.array([r["gap_offdiag"] for r in rs])

        sum_rtn = float(dist_rtn.sum())
        sum_gap = float(gap.sum())

        # per-row fraction, but only over rows with a non-trivial gap, for medians
        eps = 1e-12
        nontrivial = gap > (1e-4 * np.maximum(dist_rtn, eps))
        per_row_gap_frac = gap[nontrivial] / np.maximum(dist_rtn[nontrivial], eps)
        per_row_clc_of_gap = (clc_rec[nontrivial] /
                              np.maximum(gap[nontrivial], eps))

        curve.append({
            "dprime": dprime,
            "n": len(rs),
            "n_nontrivial_gap": int(nontrivial.sum()),
            # pooled (ratio-of-sums) -- the headline numbers
            "gap_frac_pooled": sum_gap / sum_rtn if sum_rtn > 0 else 0.0,
            "clc_recovered_frac_of_gap_pooled": (float(clc_rec.sum()) / sum_gap
                                                 if sum_gap > 0 else 0.0),
            "gap_rank_one_frac_pooled": (float(g_ro.sum()) / sum_gap
                                         if sum_gap > 0 else 0.0),
            "gap_offdiag_frac_pooled": (float(g_od.sum()) / sum_gap
                                        if sum_gap > 0 else 0.0),
            # robust per-row medians (nontrivial-gap rows only)
            "gap_frac_median": (float(np.median(per_row_gap_frac))
                                if per_row_gap_frac.size else 0.0),
            "gap_frac_p95": (float(np.percentile(per_row_gap_frac, 95))
                             if per_row_gap_frac.size else 0.0),
            "clc_of_gap_median": (float(np.median(per_row_clc_of_gap))
                                  if per_row_clc_of_gap.size else 0.0),
            # ABSOLUTE scale -- needed to tell "real bit-width-robust headroom"
            # from "ratio of two tiny numbers" at higher bit-width. Mean per-block
            # distortion in raw units; compare across bits directly.
            "dist_rtn_mean_abs": float(dist_rtn.mean()),
            "dist_exact_mean_abs": float(np.array([r["dist_exact"] for r in rs]).mean()),
            "gap_mean_abs": float(gap.mean()),
            "dist_rtn_median_abs": float(np.median(dist_rtn)),
            "gap_median_abs": float(np.median(gap)),
        })
    out["headroom_curve"] = curve
    return out


# =============================================================================
# CLI
# =============================================================================

def load_wikitext2_simple(n_samples=128):
    print("Loading WikiText-2 (simple)...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [item['text'] for item in ds if len(item['text'].strip()) > 100]
    return texts[:n_samples]


def main():
    p = argparse.ArgumentParser(
        description="Direction 1: encoding headroom (exact CVP vs RTN vs CLC).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-path", type=str, default="./models/Mistral-7B-v0.3")
    p.add_argument("--output-json", type=str, default="./headroom_d1_results.json")
    p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--calib-dataset", type=str, default="c4",
                   choices=["c4", "wikitext2", "wikitext2-simple"])
    p.add_argument("--cache-dir", type=str, default="./calibration_cache")
    p.add_argument("--max-tokens-per-sample", type=int, default=2048)
    p.add_argument("--layer-batch-size", type=int, default=8)

    # headroom-specific
    p.add_argument("--dprimes", type=int, nargs="+", default=[16, 24, 32],
                   help="Sub-block sizes d' to evaluate.")
    p.add_argument("--n-rows", type=int, default=64,
                   help="Output rows sampled per layer.")
    p.add_argument("--block-method", type=str, default="abs_mu",
                   choices=["abs_mu", "eigvec"])
    p.add_argument("--enum-threshold", type=int, default=24,
                   help="d' <= this -> exhaustive 2^{d'}; above -> branch & bound.")
    p.add_argument("--max-layers", type=int, default=0,
                   help="0 = all Linear layers; else cap (for a quick run).")
    p.add_argument("--sample-layers", type=int, default=0,
                   help="0 = use all/capped layers; else randomly sample this many "
                        "layers across the model (Direction-1 spot check).")
    p.add_argument("--layer-filter", type=str, default="",
                   help="Substring filter on layer name (e.g. 'mlp' or 'q_proj').")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Direction 1: Encoding Headroom Measurement")
    print(f"Model: {args.model_path} | bits={args.bits} | g={args.group_size}")
    print(f"d' grid: {args.dprimes} | rows/layer: {args.n_rows} | "
          f"block: {args.block_method}")
    print(f"CLC knee=-10 (no outlier mask), max_flip=1.0 (cap=d')")
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
        calib = get_c4_calibration_data(tokenizer, n_samples=args.n_calib,
                                        seqlen=2048, seed=args.seed,
                                        cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        calib = load_wikitext2_simple(n_samples=args.n_calib)
    else:
        calib = get_wikitext2_calibration_data(tokenizer, n_samples=args.n_calib,
                                               seqlen=2048, seed=args.seed,
                                               cache_dir=args.cache_dir)

    # gather target layers
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

    collector = SecondMomentCollector(max_tokens_per_sample=args.max_tokens_per_sample)
    all_records = []

    bs = args.layer_batch_size
    n_batches = (len(all_linear) + bs - 1) // bs
    for b in range(n_batches):
        batch = all_linear[b * bs:(b + 1) * bs]
        print(f"\n[Batch {b+1}/{n_batches}] {len(batch)} layers")
        calibrate_and_collect(model, tokenizer, batch, calib,
                              collector, device, args.n_calib,
                              max_len=512)
        for name, module in tqdm(batch, desc="  Headroom", leave=False):
            H, mu = collector.finalize(name, use_james_stein=True)
            if H is None:
                print(f"    ⚠️  no activations for {name}, skipping")
                continue
            recs = measure_layer_headroom(
                name, module, H, mu, args.bits, args.group_size,
                args.dprimes, args.n_rows, args.block_method,
                args.enum_threshold, device, seed=args.seed)
            all_records.extend(recs)
            collector.clear(name)
            del H, mu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        collector.clear()
        gc.collect()

    summary = summarize(all_records)
    summary["config"] = vars(args)

    with open(args.output_json, "w") as f:
        json.dump({"records": all_records, "summary": summary}, f, indent=2)

    print("\n" + "=" * 80)
    print("HEADROOM CURVE   (pooled = ratio-of-sums; med = per-row median, "
          "nontrivial-gap rows)")
    print("gap = RTN - exact, as fraction of RTN distortion")
    print("=" * 80)
    print(f"{'d_prime':>7} {'n':>5} {'ntv':>5} {'gap%':>7} {'gap%_med':>9} "
          f"{'CLC/gap':>8} {'CLC_med':>8} {'rank1%':>8} {'offdiag%':>9}")
    for c in summary["headroom_curve"]:
        print(f"{c['dprime']:>7} {c['n']:>5} {c['n_nontrivial_gap']:>5} "
              f"{c['gap_frac_pooled']*100:>6.2f}% "
              f"{c['gap_frac_median']*100:>8.2f}% "
              f"{c['clc_recovered_frac_of_gap_pooled']*100:>7.1f}% "
              f"{c['clc_of_gap_median']*100:>7.1f}% "
              f"{c['gap_rank_one_frac_pooled']*100:>7.1f}% "
              f"{c['gap_offdiag_frac_pooled']*100:>8.1f}%")
        print(f"        {'':>5} {'':>5}  abs: dist_rtn(mean)={c['dist_rtn_mean_abs']:.3e} "
              f"dist_exact(mean)={c['dist_exact_mean_abs']:.3e} "
              f"gap(mean)={c['gap_mean_abs']:.3e} "
              f"gap(med)={c['gap_median_abs']:.3e}")
    print(f"\n✅ Saved {len(all_records)} records to {args.output_json}")


if __name__ == "__main__":
    main()