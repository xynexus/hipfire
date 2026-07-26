# DSpark drafter training on the WMMA matrix cores (forward done, backward correct)

Status: **forward landed and validated; backward is now proven CORRECT and wired
behind `HIPFIRE_TRAIN_BWD=bf16x2` — the earlier "divergence bug" (§3.3) was a
misdiagnosis (chaos + atomic nondeterminism, not a defect).** One blocker remains:
the backward is not yet a net *speed* win at the body's small-M shapes (§3.4),
which needs Phase A window-batching. This doc explains the backward in detail and
outlines the path to a full bf16/fp16 WMMA training step.

Target hardware: gfx1151 (Strix Halo, RDNA3.5, wave32, `v_wmma_*_16x16x16_*`).
All measurements below are on halo, gemma3-4b DSpark drafter (639M params),
`window_batch=8`, unless noted.

---

## 1. Background: why WMMA at all

The DSpark train step is dominated by dense GEMMs that run on the scalar f32
kernel `kernels/src/gemm_f32_train.hip` (register-tiled, **no matrix cores, no
LDS** — the LDS variant is disabled for a gfx1103 HIP-719 fault). Measured on the
scalar path it sustains ~30 GFLOP/s; the RDNA3 WMMA units do bf16/f16 at
~16–33× that. Profiling one `wb=8` minibatch (`HIPFIRE_DSPARK_PROFILE=1`):

| phase        | f32 scalar | share |
|--------------|-----------:|------:|
| body_fwd     | ~5990 ms   | 66%   |
| heads_fwd    | ~1550 ms   | 17%   |
| body_bwd     | ~1280 ms   | 14%   |
| heads_bwd    | ~260 ms    | 3%    |
| loss/opt/free| ~270 ms    | 3%    |

So the whole step is this one scalar GEMM at various shapes. The plan is to move
it onto WMMA.

---

## 2. What already works (forward) — landed

Three F32-in / F32-out **NT** WMMA GEMMs were added. They read f32 operands, cast
in-register, accumulate in f32, and write f32 — so **master weights, the AdamW
optimizer, and the (f32) backward are unchanged; only the matmul compute is
low-precision** (mixed-precision *compute*, not mixed-precision *storage*).

| kernel (`kernels/src/`)       | math            | mantissa | use |
|-------------------------------|-----------------|---------:|-----|
| `gemm_bf16c_train_nt.hip`     | bf16 cast       | 8 bits   | body forward (`HIPFIRE_TRAIN_LOWP=bf16`) |
| `gemm_f16c_train_nt.hip`      | fp16 cast       | 10 bits  | (available; NaNs on real logits, see §4) |
| `gemm_bf16x2_train_nt.hip`    | split "2×bf16"  | ~16 bits | vocab heads (`HIPFIRE_TRAIN_HEADS=bf16x2`) + a gfx1151 LDS m128 variant |

All compute `Y[B,M] = X[B,K] · A[M,K]ᵀ` (NT — both operands stored `[rows, K]`,
contract the contiguous last axis). Mapping onto the linear forward
`Y[Mtok,Nout] = X[Mtok,K] · W[Nout,K]ᵀ`: `A=W` (kernel M=Nout), `X=X` (kernel
B=Mtok). `Gpu::gemm_bf16x2_train_nt` routes to the LDS m128 variant when
`n>=128 && m>=16 && k%16==0` on gfx1151 (the M-heavy lm-head regime), else the
LDS-free kernel.

Wiring (`crates/hipfire-train/src/ops/linear.rs`):

* `linear_forward` — `HIPFIRE_TRAIN_LOWP=f16|bf16` (unset = f32). Body only.
* `linear_forward_heads` — `HIPFIRE_TRAIN_HEADS=bf16x2` (unset = f32). Heads only.

**Precision defaults to f32 so the finite-difference gradchecks keep an exact
path.** Parity vs f32 (`examples/gemm_bf16c_parity`), cos and max-rel-err:

```
bf16   rel ~3e-4   f16 rel ~4e-5   bf16x2 rel ~2e-5   (all cos = 1.0000000)
```

Measured results:

* body_fwd 750 ms/window → **83 ms (~9×)** with bf16.
* heads_fwd **943 ms → ~320 ms (~2.9×)** with bf16x2 — the 3 split passes are
  nearly free because the 262k-vocab lm-head GEMM is bandwidth/launch-bound.
* **Quality unchanged**: overfit best_eval 1.79 (bf16 body + bf16x2 heads) vs f32
  1.63; all 5 `gradcheck_dspark_*` pass on the default f32 path.
* Net **~2.9×/epoch** (~38 → ~13 min).

Commits: `2a2fcbcee` (bf16 body forward), `f5013f5ab` (bf16x2 split heads).

---

## 3. The backward problem

### 3.1 Why the backward is harder than the forward

The training linear op has three matmuls that contract *different* axes:

```
forward  Y[M,N] = X[M,K] · Wᵀ         contract K   (NT — native)
dX[M,K]         = dY[M,N] · W[N,K]     contract N   (NN)
dW[N,K]         = dYᵀ[N,M] · X[M,K]    contract M   (TN)
```

WMMA fragment loads want 16 **contiguous** elements along the contracted axis.
The forward contracts K (the last, contiguous axis of both operands) → native.
The backward contracts N or M, which is *not* the last axis of at least one
operand → you must either transpose the operand or stage it transposed in LDS.

### 3.2 The approach that was tried (transpose + reuse NT)

Reformulate both backward matmuls as NT by pre-transposing operands
(`Gpu::transpose_f32`), then reuse the validated `gemm_bf16x2_train_nt`:

```
dX = NT(dY, Wᵀ)          Wᵀ  = transpose(W[N,K]) → [K,N]        (1 transpose)
dW = NT(dYᵀ, Xᵀ)         dYᵀ = transpose(dY[M,N]) → [N,M]
                         Xᵀ  = transpose(X[M,K]) → [K,M]        (2 transposes)
```

Accumulate (`dwk`/`dwv`/`d_xn1` fan-in) handled at the Rust level: WMMA into a
scratch buffer, then `add_inplace_f32` into the destination. Gated by
`HIPFIRE_TRAIN_BWD=bf16x2` (reverted; see the NOTE in `linear.rs`).

The math is right (`dX[m,k']=Σ_n dY[m,n]·W[n,k']`, `dW[n,k']=Σ_m dY[m,n]·X[m,k']`
both verified by hand and by the GEMM parity at backward shapes).

### 3.3 Blocker 1 — RESOLVED: never a bug; the overfit end-loss is chaos-dominated

The original claim was that f32-fwd + **bf16x2-backward** diverges (best_eval
3.60 @ ep60 → 8.5) vs an f32 run's 1.63, and that this was a "scale-only bug" in
`transpose_f32` / the scratch+add accumulate / the m128 LDS variant. **That was a
misdiagnosis.** Re-investigated 2026-07-08 (halo, `gemma3-4b-2k.dslb`, 2-window
overfit); three independent results show the backward is correct and the
divergence was chaotic amplification, not a defect:

1. **Per-op gradient parity at training dims** (`examples/gemm_bf16x2_backward_
   parity`): each backward piece vs the f32 reference — dX via the m128 variant
   (M≥16) and the LDS-free variant (M<16), dW via the two transposes, the
   scratch+add accumulate, and `sub_offset` views of a padded parent — all match
   to **≤1.2e-3, cos 1.0**. The wrappers the NOTE feared are individually exact.
   (dX contracting the 262k vocab tops out at ~1.2e-3, not ~1e-5, purely from the
   long bf16x2 reduction; cos still 1.0.)
2. **The loss curve tracks to the precision floor, then chaos.** f32-vs-bf16x2
   overfit train loss is **bit-identical for the first ~5 epochs**, then differs
   only in the **last decimal (~1e-4, bf16x2's rounding floor)** through ~ep11,
   before the two curves wander apart. Meanwhile the f32 loss itself **spikes
   16.8→23.9 at ep5 and bounces** (8→14→19→14) — the regime is unstable, so any
   ~1e-4 perturbation is amplified exponentially into a different trajectory.
3. **f32 is nondeterministic; it diverges from itself just as much.** Two
   *identical* f32 runs (same binary/config) go bit-identical for 5 epochs then
   diverge, ending best **5.70 vs 4.25** — the same spread as f32-vs-bf16x2 (3.65
   vs 4.71 at lr 3e-4; 7.54 vs 5.11 at lr 1e-4; the winner flips with LR). Source:
   `rmsnorm_train.hip` accumulates `dw` with `atomicAdd` (order-dependent), plus
   `pflash_score_f32_train.hip`. So the overfit end-loss **cannot discriminate** a
   correct low-precision backward from f32 — it is noise at this magnitude.

**Takeaway:** validate a low-precision backward by (a) per-op gradient parity and
(b) curve-tracking to the precision floor over the deterministic prefix — NOT the
overfit end-loss, which is chaos+atomic-nondeterminism dominated here. The WMMA
backward is now wired behind `HIPFIRE_TRAIN_BWD=bf16x2` in the single
`linear_backward_{x,w}` seam and passes both. What remains is **Blocker 2 only**
(perf), below.

**f32 is now a deterministic oracle (2026-07-08).** The nondeterminism was the
`rmsnorm_train_bwd` `dw` reduction: one block per row `atomicAdd`-ing into every
weight element, whose float order is run-dependent. Replaced with a dedicated
`rmsnorm_train_dw` kernel — one thread owns weight column `i` and sums the rows in
fixed order, then a single non-atomic `+=` (no race; preserves the zero-then-
accumulate contract). Result: **two identical f32 runs are now bit-identical for
all 40 epochs** (finals match to every digit), and `gradcheck_rmsnorm` (dW err
1.98e-5) + `gradcheck_dspark_body` still PASS — deterministic *and* correct. This
gives a stable reference to measure any low-precision format against directly
(per-step gradient/loss deltas), instead of comparing noise. (A stable *overfit*
gate would still want LR warmup / grad-clip so the f32 curve stops spiking — but
that is a training-recipe concern, separate from the now-deterministic engine.)

### 3.4 Blocker 2 — it is ~5× SLOWER even when correct

body_bwd went **~300 ms → ~1400 ms** (2-window profile). The body backward is
dominated by **small-M GEMMs** (contract dim = block = 7). At M=7 the transpose
passes + 3 split passes + per-call scratch allocation dwarf the tiny scalar-f32
cost. Unlike the forward (compute-bound, large K/N), the small-M backward has no
compute to accelerate — it is overhead-bound, and WMMA adds overhead.

---

## 4. Precision reference (for choosing the math per matmul)

| representation | mantissa | exponent | notes |
|----------------|---------:|---------:|-------|
| f32            | 24 bits  | 8 bits   | reference |
| bf16           | 8 bits   | 8 bits   | full f32 range; cheapest (1 pass) |
| fp16           | 10 bits  | 5 bits   | **overflows** on large logits/activations → NaN; do not use un-scoped |
| 2×bf16 (split) | ~16 bits | 8 bits   | 3 WMMA passes; ~f32 for our purposes |
| 3×bf16 (split) | ~24 bits | 8 bits   | 6 WMMA passes; = full f32; the ceiling |

Split-precision: `v = v_hi + v_lo` (`v_hi = round_bf16(v)`, `v_lo =
round_bf16(v − v_hi)`); product ≈ `hi·hi + hi·lo + lo·hi` (drop `lo·lo`). Each
term adds ~8 mantissa bits; passes grow ~quadratically with split levels. Even
3×bf16 (6 passes) beats the scalar core (~16–33× per pass ÷ 6 ≈ 3–5× net).

**fp16 is a trap for un-scoped use**: the *hi* term still overflows fp16's 5-bit
exponent — `HIPFIRE_TRAIN_LOWP=f16` NaN'd on real training data even though the
GEMM parity looked clean on small random inputs. fp16 is only safe where the
values are known-bounded (e.g. the rmsnorm'd body, if measured). **bf16 is the
default base everywhere.**

Empirical precision-vs-convergence (overfit best_eval, lower better):

| config                                   | best_eval | note |
|------------------------------------------|----------:|------|
| f32 (all)                                | 1.63      | reference |
| bf16 forward everywhere                  | 3.63      | 8-bit logits degrade the softmax |
| fp16 forward everywhere                  | NaN       | range overflow |
| bf16 body + **bf16x2 heads** (forward)   | 1.79      | shipped — heads need ≥16-bit |
| + bf16x2 backward                        | 3.60→8.5  | the §3.3 bug |

Takeaway on scoping: **the vocab-head logits (feed softmax/CE) need ≥16-bit
(bf16x2); the rmsnorm-bounded body tolerates 8-bit (bf16).** The backward's
precision requirement is untested pending the §3.3 bug fix, but by the same
logic bf16x2 (≈f32) should be safe for gradients.

---

## 4.1 Backward low-precision formats — MEASURED (2026-07-09)

All backward matmuls behind `HIPFIRE_TRAIN_BWD` (default f32), reformulated NT in
the single `linear_backward_{x,w}` seam. Measured against the now-deterministic
f32 oracle (§3.3): per-op rel-err (`examples/gemm_bf16x2_backward_parity`) and the
2-window overfit best_eval (chaotic tail — read as "oracle-class vs not", f32-vs-
f32 spread is ~4.25–5.70).

| `HIPFIRE_TRAIN_BWD` | passes | dW rel-err | overfit best | verdict |
|---------------------|-------:|-----------:|-------------:|---------|
| f32 (oracle)        | —      | 0          | 4.12         | reference |
| `bf16`              | 1      | 2–6e-3     | 6.28         | 8-bit mantissa too coarse for dW |
| `f16`               | 1      | 3e-4       | **NaN@ep2**  | 10-bit mantissa great, 5-bit exponent overflows real operands |
| `f16s` (scaled)     | 1      | 3e-4       | **4.08**     | **oracle-class at ONE pass** ✅ |
| `bf16x2`            | 3      | 5e-6       | 4.31         | oracle-class, 3× the passes |

**Key finding: `f16s` (per-tensor-scaled f16) matches bf16x2's convergence at 1/3
the passes.** dW (weight gradient) is the precision-critical matmul — it contracts
M and feeds the update — and 8-bit bf16 is too coarse there (2–6e-3), while f16's
10-bit mantissa (3e-4) is plenty. Plain f16 dies on *range* (activations/gradients
> 65504 → NaN on the first step), not precision. `f16s` fixes range with a
per-tensor power-of-two scale applied around the f16 WMMA (`gemm_f16s_train_nt`):
`max|op|·scale ≈ 2^14`, unscale the f32 accumulator by `1/(sx·sw)`. Per-tensor was
sufficient for this drafter (no per-channel outliers beyond f16's 30-binade window
after centering). Scale from a deterministic `abs_max_f32` (atomicMax on bit
patterns — order-independent, preserves the oracle).

Open perf work (correctness/quality proven; these are speedups, not blockers):
* **Fuse the amax into the transpose — DONE.** `transpose_f32_amax` computes
  `max|in|` (block-reduce + `atomicMax` on bit patterns; order-independent, so
  deterministic) in the same pass that already streams the operand, since
  `max|transpose(T)| == max|T|`. The f16s backward now gets `wt`/`dyt`/`xt`'s
  abs-max for free; only `dY` in `linear_backward_x` (not transposed, and small)
  keeps a standalone `abs_max_f32`. Saves re-streaming the operand — biggest on
  the lm-head weight (~2.7 GB). Validated: parity unchanged, overfit best 4.08
  (bit-equal to the pre-fuse f16s). A DPP/`permlane16` wave-max would shave the
  block-reduce further but is not the bottleneck (avoid `__shfl` → `ds_bpermute`,
  ~6–8× slower on gfx1151 per the iu4 tuning history).
* **On-device scale (drop the readback)**: compute the pow2 scale on-device and
  pass it to the GEMM by pointer, removing the last small host sync per call.
* **Delayed/carried scale**: reuse the previous step's amax (operand ranges drift
  slowly) with an amax-history window + safety margin to survive the loss spikes
  of this (unstable) overfit regime. Drops the per-step reduction to ~free.
* Per-channel scale only if a wider-spread operand ever needs it (transpose layout
  makes per-column amax natural).
* **Phase A landed and DISPROVED the "raise M" framing (2026-07-09).** Batching
  the body across windows (M = block = 7 → wb·block = 56) is done, correct
  (bit-exact batched-vs-looped equivalence at wb=1; forward+d_main_hidden
  byte-identical at wb=2/4, grads differ only by FP re-association ~1e-13; both
  gradchecks pass; overfit converges) — but the low-precision backward is STILL
  ~5–6× slower than f32 at M=56:

  | wb=2 overfit | body_bwd |
  |--------------|---------:|
  | f32          | ~162 ms  |
  | bf16x2       | ~756 ms  |
  | f16s         | ~1014 ms |

  So the body backward was never *compute*-bound — the bottleneck is the
  **wrapper**: per-matmul `transpose_f32` passes + amax reduction + host scale
  readback, while the actual GEMM (large K/N) is bandwidth-bound on gfx1151.
  Raising M amortizes f32 per-op overhead (body_bwd scales sub-linearly, 4×
  windows → 2.3× time) but does nothing for the wrapper cost that dominates the
  low-precision path. ⇒ The real backward-speedup lever is **Phase C2**, NOT
  further M-batching. Phase A is a correct prerequisite and an f32 amortization
  win, not the speedup itself.

* **Phase C2 landed and DELIVERS the speedup (2026-07-09).** Dedicated NN/TN
  scaled-f16 WMMA kernels (`gemm_f16s_backward.hip`, `HIPFIRE_TRAIN_BWD=f16sc2`)
  read the strided operand DIRECTLY — the transpose is folded into the WMMA
  fragment load as a strided global read, so W/dY/X are streamed once instead of
  the transpose path's 3× (read + write Wt + read Wt), and there's no Wt scratch.
  dX = dY·W is NN (W read strided along contract N); dW = dYᵀ·X is TN (both
  operands strided along contract M). Parity: bit-for-bit the same rel-err as the
  transpose-based f16s on every shape (my WMMA layout is correct). Profile (40ep
  overfit, same harness as above):

  | backward | body_bwd | vs f16s | vs f32 |
  |----------|---------:|--------:|-------:|
  | f32      | 161 ms   | —       | 1.0×   |
  | f16s     | 1059 ms  | 1.0×    | 6.6× slower |
  | **f16sc2** | **156 ms** | **6.8× faster** | **~parity** |

  So f16sc2 is an **oracle-class (bit-equal to f16s) 1-pass backward at
  f32-competitive speed** — the transpose passes WERE the entire overhead. But
  force-C2 is NOT uniformly best: for the lm-head (N=vocab~262k) the strided
  reads are massively UNCOALESCED and lose to the transpose (heads_bwd: transpose
  173 < C2 240). C2 wins the body (many small matmuls, overhead-bound); transpose
  wins the lm-head (few huge matmuls, coalescing amortizes).

* **Hybrid dispatch — BELOW f32 (2026-07-09).** `HIPFIRE_TRAIN_BWD=f16s` now
  dispatches per matmul on N (`F16S_STRIDED_MAX = 65536`): strided C2 kernels for
  small N (body), transpose+NT for huge N (lm-head). `f16sc2` forces all-strided
  (bench/tuning). Measured total backward (body+heads):

  | mode           | body_bwd | heads_bwd | total |
  |----------------|---------:|----------:|------:|
  | f32            | 161 ms   | 254 ms    | 415   |
  | **f16s hybrid**| 155 ms   | 173 ms    | **328** (1.27× vs f32) |
  | f16sc2 (all-C2)| 156 ms   | 240 ms    | 396   |

  Hybrid f16s is the fast default: below f32, best of both regimes, bit-equal to
  the transpose f16s (parity unchanged), converges identically. Further headroom:
  C2 still does 2 standalone `abs_max` per matmul (on-device/delayed/carried scale
  would drop them); an LDS-coalesced strided variant could push the lm-head C2
  below transpose. No-LDS keeps it off the gfx1103 HIP-719 fault. TN/NN for bf16x2
  is a mechanical copy if ≥16-bit gradients are ever needed.

## 5. Plan: full forward + backward on WMMA

Ordered; each phase is independently shippable and gated behind an env toggle
(default f32) until validated.

### Phase A — batch the drafter body across windows — DONE (see §4.1)

Landed 2026-07-09: correct + validated (bit-exact equivalence gate
`examples/dspark_batch_equiv`, gradchecks at n_win>1, overfit converges). It
amortizes f32 per-op overhead but does NOT speed up the low-precision backward —
that needs Phase C2 (§4.1). Original plan below.

Today `forward_loss_batch` runs the body **per window** (M = ctx_len=128 for the
ingest/ctx-K·V ops, M = block=7 for the block/MLP ops). Batching `wb` windows
raises the block-op M from 7 → `wb·block` (56) and, crucially, raises the
**backward** contract dim (dW contracts M) from 7 → 56, which is what makes WMMA
worth it there.

* Row-wise ops (rmsnorm, rope, q/k/v/o/gate/up/down projections, ingest fc) stack
  trivially along the token axis across windows.
* **Attention must stay per-window** (window *i*'s block queries attend only to
  window *i*'s `[ctx ++ block]` keys). Loop the attention over windows (assemble
  each window's keys from the stacked `k_ctx`/`k_blk`), or write a batched
  block-diagonal GQA. RoPE positions repeat per window.
* Threads `n_win` through `dspark_block_forward/backward`, `dspark_ingest_*`,
  `dspark_drafter_forward_train/backward`, and the activation structs.

**Expected forward gain is modest** — the dominant forward GEMMs (ingest, ctx
K·V) are already at M=128 with ample grid parallelism; only the M=7 block ops
improve. The real payoff is enabling the backward WMMA and shrinking per-op
launch/alloc overhead. Validate: `gradcheck_dspark_*` with `n_win>1`, then the
overfit (must still reach ~0.1).

### Phase B — root-cause the backward "bug" (§3.3) — DONE (no bug found)

Completed 2026-07-08. The backward WMMA is correct; there was no wrapper defect
to fix (see §3.3). What was actually built/learned:

1. **`examples/gemm_bf16x2_backward_parity`** — the backward-shape parity both the
   gradcheck (toy dims) and forward parity (clean output) were missing. Exercises
   dX (m128 + LDS-free), dW (transposes), the scratch+add accumulate, and
   `sub_offset` views at training dims. All ≤1.2e-3, cos 1.0. **This is the
   regression gate for the backward — run it, not the overfit.**
2. **`transpose_f32` on `sub_offset` views is fine**: `sub_offset` bakes the byte
   offset into the pointer, so views read correctly; the lm-head `[262208,2560]`
   (671M elts) is under i32 max and the GEMMs index in 64-bit — no overflow.
3. **The overfit end-loss is NOT a valid gate here** — it is chaos- +
   atomic-nondeterminism-dominated (f32 diverges from itself by the same spread).
   A stable gate needs LR warmup / grad-clip first (orthogonal).

The reformulation is wired in the single `linear_backward_{x,w}` seam behind
`HIPFIRE_TRAIN_BWD=bf16x2` (default f32).

### Phase C — backward kernels

Two options once the shapes are large (Phase A) and the bug is understood
(Phase B):

* **C1 — transpose + reuse NT (what was tried).** Simplest; reuses the validated
  NT split kernel. Costs 1–2 `transpose_f32` passes per matmul + a scratch buffer
  for accumulate. Viable once M is large (transposes amortize) and the wrapper bug
  is fixed. Reduce allocation churn by reusing per-shape scratch/transpose buffers.
* **C2 — dedicated NN + TN split-WMMA kernels.** Stage the strided operand tile
  transposed in LDS, then WMMA directly — no separate transpose pass, no
  transpose-of-sliced-tensor bug surface. More kernel-authoring, but the efficient
  end state. Recommended if C1's overhead is still too high after batching.

Precision per matmul (start conservative, relax by measurement):

* dX, dW: **bf16x2** (≈f32; gradients into the shared body + AdamW moments).
* Consider bf16 for dW of the body once convergence is confirmed (cheaper).

### Phase D — consolidate toggles + validate

* Collapse `HIPFIRE_TRAIN_LOWP` / `HIPFIRE_TRAIN_HEADS` / `HIPFIRE_TRAIN_BWD` into
  one policy (e.g. `HIPFIRE_TRAIN_WMMA=1` selecting the validated per-matmul mix:
  body-fwd bf16, heads-fwd bf16x2, backward bf16x2), keeping f32 the default so
  gradchecks stay exact.
* Validation matrix (all must hold before flipping any default):
  1. `gemm_*_parity` — near-f32 rel-err at **forward *and* backward** shapes,
     both GEMM variants.
  2. `gradcheck_dspark_*` — at **training dims** with the WMMA path on.
  3. Isolation overfits — f32-fwd+wmma-bwd and wmma-fwd+f32-bwd each converge to
     ~f32 before combining.
  4. Full-stack overfit — best_eval ≈ f32.
  5. `HIPFIRE_DSPARK_PROFILE=1` — per-phase speedup, no phase regressed.

---

## 6. Expected end state

With forward (done) + backward on WMMA, the f32 backward (~17% at f32, but the
largest phase *after* the forward WMMA) moves onto the matrix cores. If the
backward matches the forward's ~5–9× on its now-large GEMMs, the step drops from
the current ~13 min/epoch toward ~6–8 min/epoch, at f32-equivalent quality.

## 7. Risks / open questions

* The §3.3 bug may be in `transpose_f32` (shared infra) — fixing it could touch
  code beyond hipfire-train.
* Batched attention (Phase A) is the most delicate new code (per-window
  isolation); gradcheck it hard.
* Whether bf16x2 gradients converge is *unproven* until the bug is fixed — Phase B
  must confirm convergence before investing in Phase C's efficient kernels.
* Portability: the LDS m128 variant is gfx1151-specific; the LDS-free kernels
  cover other RDNA3/RDNA4 archs but slower. Keep the f32 fallback.

## References

* Kernels: `kernels/src/gemm_{bf16c,f16c,bf16x2}_train_nt.hip`,
  `kernels/src/gemm_f32_train.hip`, `kernels/src/transpose.hip`.
* Wiring: `crates/hipfire-train/src/ops/linear.rs` (see the WMMA-backward NOTE).
* Dispatch: `crates/hipfire-rdna/src/dispatch/gemm_base.rs`.
* Parity: `crates/hipfire-train/examples/gemm_bf16c_parity.rs`.
* Profiling / logging: `crates/hipfire-train/src/dspark_train.rs`
  (`HIPFIRE_DSPARK_PROFILE`, `--progress-updates`).
* Commits: `2a2fcbcee`, `f5013f5ab` (forward, landed); `a816ed3e9`, `6455ea283`
  (backward dead-end, documented).
