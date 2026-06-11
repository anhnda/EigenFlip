# EigenFlip / EigenFlip Solve — encoder-stage PTQ scaffold

A `base × encoder` quantization framework realizing the EigenFlip Solve paper.
The design separates the four PTQ stages so any **encoder** (corrector) runs on
any **base** (transform + scales), giving the paper's Table 6 comparison for free.

## Layout

```
eigenflip/
  statistics/
    accumulators.py   StreamingMoments / StreamingGram / StreamingSketch
                      + plan_gram (per-layer VRAM budgeting, heavy-layer safe)
    james_stein.py    one canonical JS mean (retires the 2–3 scattered copies)
    trust_region.py   LayerStats: builds D (floored residual diag) and
                      V=[mu | U_k Lam_k^{1/2}] — NEVER materializes H~ = D+VV^T
    collect.py        AWQ-style batched calibration; hook FOLDS into a streaming
                      accumulator and drops activations (never stores X)
  quantization/
    state.py          IntegerQuantizedTensorState with from_rtn / from_awq
                      constructors (GPTQ path lives in your gptq.py)
  encoders/
    base_encoder.py   Encoder protocol; IdentityEncoder ('base only')
    flip.py           CLC (rung 1) and EigenFlip (rung 2) budgeted-flip encoders
    eigenflip_solve.py  Algorithm 1: Woodbury sequential conditioning on H~_k
    dense_reference.py  DenseSurrogateGPTQ (§6.5 exactness ref) + DenseGPTQ
                        (the 'gptq' encoder row, on full H)
  pipeline/
    runner.py         base × encoder driver; default_registry()
```

## Memory model (your constraints)

- **Never form the activation matrix.** `(128·2048)×4096 ≈ 8.6 GB`, ×10000 ≈ 21 GB.
  The hook folds each fire into a streaming accumulator and drops it.
- **Stream the Gram, not X.** `StreamingGram` keeps one `d×d` buffer
  (`67 MB` fp32 / `134 MB` fp64 at d=4096; `400/800 MB` at d=10000) and never the
  activations. For heavy layers `plan_gram` picks fp64→fp32→sketch against a
  VRAM budget. **Exact if it fits, fp32 otherwise** — exactly your rule.
- **Sketch fallback (`O(dk)`).** When even a Gram won't fit the budget,
  `StreamingSketch` recovers top-k via a randomized range sketch, `~4 MB` at
  d=10000,k=16. Validate it against the Gram on a few layers (§6.5).
- **EigenFlip Solve never allocates `d×d`** — it asserts `stats.Sigma is None`.
  Only `V [d,k+1]`, `M,M⁻¹ [k+1,k+1]`, `G [C,k+1]`.
- **lm_head skipped by default** (`is_lm_head` filter in collect).
- **GPU-first, free aggressively**: accumulators `.free()` right after a layer
  is encoded; `empty_cache()` on every drop.

## Correctness (validated here, numpy ports)

- **EigenFlip Solve ≡ dense GPTQ-on-H~: 100% bitwise code agreement** across
  trials (`validate_solve.py`). This is the §6.5 claim — Solve is an *exact*
  structured implementation of the sequential rule, not an approximation. The
  fix that mattered: both the Woodbury capacitance **and** the dense reference
  must condition on the **shrinking remaining set R** (principal submatrix
  `H_RR`), not a fixed full inverse — otherwise they disagree by ±1 codes.
- **Flip descent + budget** (`validate_flip.py`): full budget makes ‖z‖²
  non-increasing on every row (Prop. 1); budgeted runs respect the per-row cap.

## GPTQ template — double-check findings

Reviewing your `gptq.py` against the `base × encoder` design:

1. **Scale ownership / AWQ confound.** `fasterquant` calls
   `quantizer.find_params(W)` — GPTQ fits its *own* min/max scales. For an honest
   `AWQ+gptq` cell the GPTQ encoder must use **AWQ's** scales. Add a
   `configure_from_state(scale, zero, maxq)` path that bypasses `find_params`,
   or the base comparison is confounded (every `X+gptq` silently uses gptq scales).
2. **Per-element vs per-group scale rep.** GPTQ stores `Q_scale/Q_zero` as full
   `[rows, cols]`; RTN/AWQ store `[C, n_groups, 1]`. `state.py` normalizes to the
   expanded `[C, pin]` form so encoders never need to know the grouping — adopt
   this everywhere.
3. **`_run_post_correction` is the right pattern** — it already builds a
   `quant_state` and hands it to a pluggable corrector. Generalize it: that
   corrector slot is exactly the `Encoder` interface. SmartFlip becomes
   `encoders/flip.py`, BiasCorrection stays as an optional post-step.
4. **act_order invariant.** GPTQ undoes `invperm` before `_build_quant_state`
   (correct) — the stored state is always natural-coordinate order. EigenFlip
   Solve has its *own* internal order (leverage); keep encoder order internal,
   never leak it into the state.
5. **JS / outlier prep belongs in `statistics/`.** `SmartFlip.prepare_activation_means`
   duplicates JS shrinkage; route it through `james_stein.py` so CLC/EigenFlip/
   Solve share one implementation.
6. **`from_rtn` recompute.** Solve-on-RTN-base has no `Q_pre` from GPTQ;
   `state.from_rtn` recomputes `pre_round = W/scale + zp`, so any encoder runs on
   an RTN base with no GPTQ machinery present.

## The comparison matrix

`runner.default_registry()` gives the encoders; bases are `{rtn, awq}` (and the
`gptq`-as-base case is just the `gptq` *encoder* on rtn/awq scales). Cells:

```
base ∈ {rtn, awq}        (awq needs awq_scales[name] from your AWQ grid search)
encoder ∈ {none, clc, eigenflip, eigenflip_solve, gptq}
```

`base + none` = base only · `base + clc` · `base + gptq` · `base + eigenflip` ·
`base + eigenflip_solve`. `DenseSurrogateGPTQ` is added only for §6.5 validation.

## Open decisions (flagged, not chosen)

- **Full-layer vs group-local** EigenFlip Solve (Remark 1): scaffold is
  full-layer (`d'≈d`, cross-group coupling retained). Group-local is a separate
  method/ablation.
- **eig backend default**: build/validate with `gram`, expose `sketch` via
  `force_backend`. Report gram↔sketch agreement as part of §6.5.
- **k=0 Solve** reduces to the one-accumulator Σ∆ error-feedback anchor (§4.5);
  `EigenFlipSolve` already handles `kp1=1` (V = [mu]).
```

## Commands

Put `calibration_utils.py` (the C4/WikiText-2 loader) on `PYTHONPATH` next to
`run_eigenflip.py`. Run from the dir that contains the `eigenflip/` package.

### 0. Sanity (no GPU needed) — Woodbury equivalence in numpy
```bash
python eigenflip/validate_solve.py     # expect 100% code agreement
python eigenflip/validate_flip.py      # expect descent + budget respected
```

### 1. Section 6.5 correctness gate (GPU, 1–2 layers)
```bash
PYTHONPATH=. python eigenflip/validate_65.py \
  --model-path ./models/Mistral-7B-v0.3 \
  --bits 3 --k 16 --n-calib 64 --n-layers 2
# (i) Solve vs dense surrogate: expect ~100% weight-equal
# (ii) gram vs sketch principal angles: expect small (a few degrees) on leading k
```

### 2. Build the table — ONE CELL PER INVOCATION (recommended, low memory)
RTN base, each encoder separately:
```bash
for ENC in none clc eigenflip eigenflip_solve gptq; do
  PYTHONPATH=. python eigenflip/run_eigenflip.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/eigenflip_3bit \
    --bits 3 --group-size 128 --k 16 \
    --bases rtn --encoders $ENC \
    --calib-dataset c4 --n-calib 128 --seqlen 2048 \
    --eig-backend auto --vram-fraction 0.4 \
    --layer-batch-size 16
done
```

AWQ base (needs AWQ scales from your awq run, saved as a .pt of layer_scales):
```bash
for ENC in none clc eigenflip eigenflip_solve gptq; do
  PYTHONPATH=. python eigenflip/run_eigenflip.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --output-dir ./quantized_models/eigenflip_3bit \
    --bits 3 --group-size 128 --k 16 \
    --bases awq --encoders $ENC \
    --awq-scales-pt ./awq_run/layer_scales.pt \
    --calib-dataset c4 --n-calib 128 --seqlen 2048
done
```

### 3. Multi-cell in one process (small selections / big-RAM machines only)
```bash
PYTHONPATH=. python eigenflip/run_eigenflip.py \
  --bases rtn --encoders none clc eigenflip_solve \
  --bits 3 --k 16 --model-path ./models/Mistral-7B-v0.3 \
  --output-dir ./quantized_models/eigenflip_3bit
# WARNING: holds (bases*encoders) CPU state_dicts. 2x5 ~ 10 model copies.
```

### 4. Heavy layers / tight VRAM — force the sketch backend
```bash
PYTHONPATH=. python eigenflip/run_eigenflip.py \
  ... --eig-backend sketch          # O(dk) memory, never forms dxd
# or let auto fall back: --vram-fraction 0.3 --no-prefer-exact
```

### 5. 4-bit variants — same commands, swap --bits 4

### 6. Perplexity for every saved cell (fills Table 6)
```bash
for D in ./quantized_models/eigenflip_3bit/*/; do
  echo "=== $D ==="
  PYTHONPATH=. python eigenflip/eval_ppl.py --model-path "$D" --dataset wikitext2
done
# calibration-shift column: --dataset c4  (calibrate on one, eval on the other)
```

### k-sweep (Section 6.3 operating point)
```bash
for K in 0 1 2 4 8 16 32 64; do
  PYTHONPATH=. python eigenflip/run_eigenflip.py \
    --bases rtn --encoders eigenflip_solve --k $K \
    --output-dir ./quantized_models/ksweep_k$K \
    --bits 3 --model-path ./models/Mistral-7B-v0.3
done
```

## Run order on xaibk1

1. `validate_solve.py` / `validate_flip.py` (instant, no GPU) — confirm math.
2. `validate_65.py` on 2 layers — confirm Solve==dense and gram==sketch on the
   real model in bf16->fp64 before committing to a full run.
3. One RTN cell end-to-end (`--encoders eigenflip_solve`) + `eval_ppl.py` — confirm
   the pipeline writes a loadable model with sane PPL.
4. Full table via the per-cell loops (step 2 above), then PPL sweep.

## Section 6.6 — the decisive shrinkage comparison

The gating experiment. EigenFlip Solve vs tuned covariance shrinkage run
through GPTQ, on a fixed base, with held-out lambda tuning.

Baselines (encoders/shrinkage.py):
- **shr_gptq (cov)** — mean-preserving `mu mu^T + (1-l)Sigma + l diag(Sigma)`
  (Eq. 13). PRIMARY falsification baseline: concedes the mean exactly as Solve.
- **shr_gptq (2m)** — second-moment blend `(1-l)H + l diag(H)`. SECONDARY.

Each in three lambda instantiations:
- **global** (i): one lambda, argmin of summed HELD-OUT distortion over probe
  layers (disjoint calib split B; never eval). The deployable competitor.
- **analytic** (ii): per-layer Ledoit-Wolf diagonal-target. No tuning cost.
  (NB: uses a documented O(1/n_eff) variance proxy, not full per-token 4th
  moments — see `ledoit_wolf_lambda_diag_target` docstring.)
- **perlayer** (iii): per-layer grid-tuned. Oracle-flavored upper bound.

Validated (numpy, `validate_66.py`): both H-builders preserve the diagonal to
machine epsilon at every lambda; lambda=1 cov collapses to `mu mu^T + diag(Sigma)`;
the distortion metric `((E@H)*E).sum()` is bit-exact vs explicit `sum_j e_j^T H e_j`.

### Run (one base per invocation; gram+Sigma required, so heavier)
```bash
PYTHONPATH=. python eigenflip/pipeline/section66.py \
  --model-path ./models/Mistral-7B-v0.3 \
  --output-dir ./quantized_models/section66_3bit \
  --base awq --awq-scales-pt ./awq_run/layer_scales.pt \
  --bits 3 --group-size 128 --k 16 \
  --calib-dataset c4 --n-calib 128 --seqlen 2048 \
  --layer-batch-size 8 \
  --families cov 2m --lambda-modes global analytic perlayer
```
The script prints the FALSIFICATION CHECK (Solve vs cov i/ii/iii on held-out
distortion) and writes `<base>_section66.json` with chosen lambdas + the
held-out distortion table. Then PPL each saved variant:
```bash
for D in ./quantized_models/section66_3bit/*/; do
  PYTHONPATH=. python eigenflip/eval_ppl.py --model-path "$D" --dataset wikitext2
done
```

### Faster tuning on a probe subset
`--probe-frac 0.25` tunes the GLOBAL lambda on the first quarter of layers
(per-layer lambda is always all layers). Use for quick iteration; report the
full-probe number for the paper.

### rtn base (controlled k=d / interpolation block)
```bash
PYTHONPATH=. python eigenflip/pipeline/section66.py --base rtn \
  --bits 3 --k 16 --model-path ./models/Mistral-7B-v0.3 \
  --output-dir ./quantized_models/section66_3bit_rtn
```

### Memory note
Section 6.6 forces the gram backend with keep_sigma on EVERY layer (shrinkage
and gptq need the dense Sigma). At d=4096 that's ~134 MB fp64/layer in the
active batch; at d~10k it's ~800 MB. Use `--layer-batch-size 8` (or lower) for
wide layers. Sigma is freed immediately after each layer's encoders run.

### Two-pass design (in section66.py)
PASS 1 collects split-A and split-B stats per batch, tunes per-layer lambda,
and caches probe-layer encode inputs (Sigma_A + H'(B)) for global tuning.
PASS 2 re-walks layers and encodes every method into its own state_dict.
Non-probe layers are re-collected (split A only) in PASS 2 to avoid holding all
Sigmas at once. This trades one extra calibration pass for bounded memory; if
RAM is ample, raise --probe-frac to 1.0 so all layers are cached and PASS 2
needs no re-collection.

## SPEED FIX (replaces slow per-batch collection)

The old per-batch collector re-ran the WHOLE model for every group of 16 layers
(n_batches full-model passes). The new `statistics/sequential.py` is GPTQ-style
block-paged: ONE pass total, block-by-block, with `inps,outs=outs,inps`.

Collect only what the encoder needs (conditional streaming):
  rtn -> nothing | clc -> mean E[X] only (O(d)) | eigenflip/solve/gptq/shr -> H

Run ONE cell per process:
```bash
PYTHONPATH=. python -m eigenflip.run_fast \
  --model-path /path/to/Meta-Llama-3.1-8B \
  --base rtn --encoder eigenflip_solve --k 16 \
  --bits 3 --group-size 128 \
  --calib-dataset c4 --nsamples 128 --seqlen 2048 \
  --output-dir ./quantized_models/ll31_3bit
```
Encoders: none clc eigenflip eigenflip_solve gptq shr_gptq_cov shr_gptq_2m
Heavy layers: add --eig-on-cpu to move eigh off GPU.
