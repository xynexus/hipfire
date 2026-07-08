# DSpark drafter training on the WMMA matrix cores (forward done, backward blocked)

Status: **forward landed and validated; backward is a documented dead-end with two
concrete blockers.** This doc explains the backward problem in detail and outlines
the path to running the full training step (forward *and* backward) in bf16/fp16
WMMA math.

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

### 3.3 Blocker 1 — it DIVERGES training (a scale-only bug, NOT precision)

Isolated (f32 forward + **bf16x2 backward only**), the overfit reaches best_eval
**3.60 @ epoch 60 then climbs to 8.5** (f32 baseline: 1.63). Key evidence that
this is a *bug*, not lossy gradients:

* **The GEMM is numerically ~f32.** `gemm_bf16c_parity` extended to the backward
  m/k/n mappings (incl. the LDS m128 variant): bf16x2 rel-err **~1e-5** on every
  shape. bf16's own 8-bit gradients are used in production training and converge,
  so 16-bit gradients diverging is implausible → not precision. **More split
  terms would not help** (3×bf16 = full f32; the split ceiling — bf16 stays the
  base since fp16's exponent overflows regardless of split count).
* **The gradchecks are blind to it.** `gradcheck_dspark_body/block` use *small
  toy dims*, so their GEMMs fall to the **simple** (LDS-free) variant — they never
  exercise the LDS m128 variant, sub-16 batch tiling, sub_offset slices at scale,
  or the accumulate wrapper on large tensors. They "pass" while the training path
  breaks.
* **The GEMM parity is also blind to it** — it validates the GEMM *output* on
  clean contiguous tensors, not the *wrapper* (transpose_f32 on large / sliced
  tensors, the scratch+add accumulate, the LDS variant driven with the backward's
  large `m`).

So the bug lives in the **backward wrapper at training scale**: prime suspects
are (a) `transpose_f32` on the real training tensors — several backward operands
are `sub_offset` views of a parent buffer (e.g. `d_k_ctx`/`d_k_blk` split from
`d_kcat`); (b) the LDS m128 GEMM variant driven with the backward mapping (large
kernel-`B`, small contract); (c) the scratch+add accumulate. gradcheck passes
because none of these are in its regime.

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

## 5. Plan: full forward + backward on WMMA

Ordered; each phase is independently shippable and gated behind an env toggle
(default f32) until validated.

### Phase A — batch the drafter body across windows *(prerequisite for the backward)*

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

### Phase B — root-cause the backward wrapper bug (§3.3)

Before re-enabling the backward WMMA, close the validation blind spots so the bug
is reproducible in a test:

1. **Extend the gradchecks to TRAINING dims** (or add a large-dim gradcheck) so
   the LDS m128 variant + real slice/accumulate patterns are exercised. A
   gradcheck that fails at large dims but passes at small dims localizes the bug.
2. **Parity-test `transpose_f32` on `sub_offset` views** and on the exact large
   shapes (e.g. lm-head `[262208, 2560]`, ctx `[128, 2560]`). Check for i32 index
   overflow and offset handling.
3. **Bisect the wrapper**: force the *simple* GEMM variant for the backward
   (disable the LDS gate) and re-run the isolation overfit — if it converges, the
   bug is in the LDS-variant-for-backward mapping; if it still diverges, it is the
   transpose or the accumulate.

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
