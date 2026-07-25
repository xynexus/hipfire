# Two-stage lm_head: coarse shortlist → exact fine rescoring

A method for making the output projection (lm_head) cheap on bandwidth-bound GPUs
**without losing greedy exactness**. Developed on ZAYA1-8B (gfx1151/halo), where the
tied bf16 lm_head (vocab 262 272 × hidden 2048 ≈ 1.07 GB) dominated decode at ~6.3 ms
of a ~21 ms step. The technique generalises to any decoder whose lm_head is a large
dense `[V, H]` matmul — this doc is the recipe for porting it to other families.

## Why it works

After the final RMSNorm the logits are just `logit_i = W_i · h` for each vocab row
`W_i`. Writing `W_i = ‖W_i‖ · û_i` (norm × unit direction):

```
logit_i = ‖W_i‖ · (h · û_i)
```

The **argmax** (and any top-K sample) is a *ranking* problem. You do not need every
logit at full precision — you only need exact logits for the handful of rows that can
actually win. So:

- **Stage 1 (coarse):** score all `V` rows with a cheap, low-precision approximation
  of `W`. Cost ∝ coarse bytes read. Its only job is to produce a **shortlist** (top-K)
  that contains the true winner.
- **Stage 2 (fine):** rescore *only* the K shortlisted rows at exact bf16, and write
  those into a `-inf`-masked logit vector. Cost ∝ K rows (negligible for K ≪ V).

The approximation is confined to **selection**, never to the reported logits. Quality
is therefore a single measurable quantity: **recall@K** — does the shortlist contain
the true argmax? When recall@1 = 100 % at your chosen K, decode is **greedy-exact**
(byte-identical output to the full lm_head). For temperature sampling, K sets the
captured probability mass (this is exactly top-K sampling).

## The coarse scorer

Naive per-row-max Q4 fails: a few outlier channels set the row's scale and crush the
rest of the direction to a couple of levels. Three ingredients fix it.

### 1. Row-wise L2 normalisation (essential)

Store the **exact** per-row L2 norm `‖W_i‖` as an f32 scalar, and quantise only the
**unit direction** `û_i = W_i / ‖W_i‖`. The norm — which carries most of the row's
dynamic range — is lossless; only the direction is approximated. This is what makes
2-bit viable at all.

### 2. Global 3σ-clip symmetric quant

The unit-vector components have RMS `1/√H`. Map **3σ** of that to the max quant level
so outliers clip instead of dominating the step size:

```
unit_scale = 3 / (max_level · √H)      # Q4: max_level = 7 ; Q2: max_level = 2
q[d]       = clamp(round(û[d] / unit_scale), lo, hi)
scale_i    = ‖W_i‖ · unit_scale        # per-row f32, folded into the gemv
```

Q4 packs 2 signed nibbles/byte (levels −7..7); Q2 packs 4 signed 2-bit/byte
(two's-complement −2..1). Both are pure symmetric — no zero-point.

### 3. Optional Stage-3 low-rank residual correction

The quant residual `D = W − Q_recon` has exploitable low-rank structure. Take its
top-`r` right singular basis (randomised range finder — one small Jacobi, no full
`H×H` eigendecomp) as `B[r, H]`, and project `A[V, r] = D · Bᵀ`. At runtime add

```
coarse_score += A · (B · h)
```

a rank-`r` correction (two tiny gemvs) that recovers the low-rank part of the error
and sharply lifts tail recall. It lets an aggressive Q2 coarse reach greedy-exact.

> **Isotropy caveat (a dead end worth recording).** Row-normalisation makes the
> *direction* space near-isotropic/full-rank. SVD-based **dimensionality reduction**
> of the coarse input (`h → h_r`, `H → r`) therefore *destroys* ranking — the singular
> spectrum is flat, so no low-dim subspace preserves the dot-product order. SVD used
> as a **residual correction** (above) works; SVD used as **dim-reduction** does not.
> Do not spend time on the latter.

## The fine pass

1. Coarse-score → download the `V` scores → host `select_nth` for the top-K indices.
2. Upload the K indices; cast `h → bf16` (to match the full bf16 lm_head's arithmetic
   bit-for-bit).
3. `fill(logits, -inf)`, then a **gather-gemv** that computes the exact bf16 dot for
   each shortlisted row and **scatters** it to `logits[idx[k]]` (kernel
   `gemv_bf16_gather_f32`). Unselected vocab stays `-inf` and drops out of
   softmax/argmax.

The scatter is fused into the gather kernel (it writes `out[idx[k]]`, not `out[k]`), so
the only host round-trip is the coarse-score download for top-K.

## Measured results — ZAYA1-8B, gfx1151/halo

Recall@1 of the coarse shortlist (32-token probe), row-norm coarse:

| coarse | K=32 | K=256 | K=1024 | K=2048 | greedy-exact at |
|--------|------|-------|--------|--------|-----------------|
| Q4 (full-H)          | **100 %** | — | — | — | K=32 |
| Q2 (full-H)          | — | 93 % | 96 % | 96 % | never (96 % ceiling) |
| Q2 + corr r=64       | — | 93 % | 96 % | **100 %** | K=2048 |
| Q2 + corr r=128      | — | 96 % | **100 %** | 100 % | K=1024 |

lm_head latency (median over decode steps) and end-to-end greedy check:

| mode | env | coarse read | lm_head | vs bf16 | greedy-exact |
|------|-----|-------------|---------|---------|--------------|
| bf16 (full)  | *(unset)*                | 1074 MB | 6336 µs | 1×   | — (baseline) |
| **q4** (K=32) | `HIPFIRE_ZAYA_LMHEAD=q4`  | 268 MB  | **1622 µs** | **3.9×** | **yes** (byte-identical) |
| q2c (K=2048) | `HIPFIRE_ZAYA_LMHEAD=q2c` | 134+A   | 1703 µs | 3.7× | **yes** (byte-identical) |
| q4c (K=32)   | `HIPFIRE_ZAYA_LMHEAD=q4c` | 268+A   | 2236 µs | 2.8× | **yes** (corr wasted — q4 already exact@32) |
| q2  (K=2048) | `HIPFIRE_ZAYA_LMHEAD=q2`  | 134 MB  | 1115 µs | 5.7× | no (1 token drift) |

lm_head times are with the GPU top-K (default). The host packed-key select stays behind
`HIPFIRE_ZAYA_LMHEAD_HOSTSELECT=1` for A/B (q4: 1958 µs / 3.2× — ~340 µs slower).

`q4/q4c/q2c` reproduce the full bf16 output exactly; `q2`-alone drifts by one token
(a newline) — precisely its 96 % recall miss, which the correction removes. Overrides:
`HIPFIRE_ZAYA_LMHEAD_K` (shortlist size), `HIPFIRE_ZAYA_LMHEAD_CORR` (correction rank);
`HIPFIRE_ZAYA_LMHEAD_TIMING=1` prints the per-token coarse-gemv / host-select split.

**Recommendation: `q4` is the best exact mode.** It needs no correction and, once the
coarse gemv is vectorised (below), it reads more coarse bytes than q2 but avoids the
correction read + the larger K=2048 fine gather, so it wins overall. Use `q2` only if a
rare single-token drift is acceptable (fastest, 4.4×); `q4c` is strictly dominated by
`q4` here (the correction only pays off on an *aggressive* coarse like q2).

### Two tuning lessons from the measured breakdown

1. **Vectorise the coarse gemv — it is the dominant cost, not the top-K.** Instrumenting
   the split showed the coarse gemv was ~2.4 ms (q4) at only **~110 GB/s**, while the
   host `select_nth` was ~0.47 ms. The coarse kernel loaded weights a **byte at a time**;
   the nibble/2-bit unpack ALU starved the load-issue rate. Switching each lane to consume
   a **uint32 (8 nibbles / 16 two-bit) per load** lifted it to **~181 GB/s** (q4 coarse
   2418→1474 µs) — now *faster* than the bf16 gemv. Same 1-wave-per-block structure; the
   win was purely fewer, wider loads. (The bf16 gemv was already fast because its unpack
   is a trivial `h<<16`.) One block per row (grid=M), never a capped grid + row-stride —
   capping serialises rows and drops MLP.
2. **Pack the top-K key; skip the comparator closure.** `select_nth_unstable_by` with a
   closure that indexes back into the score array (`cv[a]` vs `cv[b]`) cost ~468 µs over
   V=262 k. Packing each score into an order-preserving `u64 = (monotone_f32_bits<<32)|idx`
   and selecting on the raw u64 (no closure, no indirection) cut it to ~303 µs.
   *Do not* reach for rayon here — two variants both lost to serial. Fine-grained
   (`into_par_iter` over 262 k elements) was 2–3× **slower** (per-element work-steal
   overhead). Coarse chunking (one chunk per thread, per-chunk partial top-K + merge —
   the textbook fix) helped only at small K and added large run-to-run **jitter**
   (310–553 µs vs a stable ~303 µs), and *regressed* at large K (K=2048: each chunk keeps
   most of itself, the merge grows back to ~V). Post-vectorisation this select is
   memory-bound and small relative to the coarse gemv, so more threads add variance, not
   throughput. The real way to remove it is on the GPU (below), not more host threads.

### GPU top-K (built — removes the host select)

The ~300 µs host select was the last non-bandwidth cost. It's now done on the device, so
the score never leaves the GPU: only three tiny scalars (min/max, a 16 KB histogram, and
the final count) cross to the host.

- **coarse score stays GPU-resident** — the low-rank correction is added on-device
  (`add_inplace_f32`, `coarse += A·(B·h)`) instead of downloaded and host-added.
- **device top-K = min/max → histogram → compact** (`lmhead_coarse_{minmax,hist,compact}`),
  sharing one **folded stats buffer** `[nbins bins | min | max]` so the whole top-K needs a
  *single* host download:
  1. min/max of the coarse scores as **order-preserving u32 keys** (`(u&sign)?~u:u|sign`),
     via integer `atomicMin/Max` into the buffer tail. **Not fused into the coarse gemv** —
     that would mean 262 k atomics on the *same two* addresses (crippling contention); the
     separate pass LDS-block-reduces to *one atomic per block* (~1000 total).
  2. a 4096-bin **linear histogram over `[min,max]`**, reading lo/hi from the buffer tail
     *on the device* (no round-trip). Binning in the *actual* key range (not the full u32
     range) keeps selectivity high however the scores cluster.
  3. one host download of the buffer → scan bins top-down to the bin where the cumulative
     count first reaches K → threshold τ; then a **compact** kernel stream-appends every
     row with key ≥ τ into a sentinel-filled idx buffer. The fine gather runs over all
     `cap` slots and skips sentinels, so **the count never crosses back to the host**.

The compacted set is a **superset** of the exact top-K (the boundary bin is taken whole),
which is harmless — the fine bf16 pass rescores every selected row exactly, so the argmax
is still correct. The idx buffer is sized from the histogram tail count + slack, so it
never truncates. Result: **q4 3.2× → 3.9×** (~1607 µs), q2c → 3.8×, all still byte-identical.

**What the last mile taught us.** Collapsing the three per-pass host round-trips to one
(folded buffer + sentinel gather) shaved only ~15 µs of latency but *tightened variance*
markedly (q4 now ±5 µs run-to-run). So the residual ~140 µs above the pure coarse-gemv
floor is **kernel-launch overhead across the ~6 small kernels**, not sync latency — and the
coarse gemv is already **~87 % of achievable memory bandwidth (~181 GB/s)**. There is no
cheap latency left: q4 ≈ 1.61 ms is the practical floor for this design. Deeper cuts would
need a single-kernel radix-select (fewer launches) or simply *fewer coarse bytes*
(e.g. storing the correction `A` at bf16/Q8 to trim `q2c`/`q4c`) — but plain `q4` already
wins outright, so neither is worth it here. A distribution with *massive* score ties could
make the boundary bin huge; a second-level histogram (radix-style) would bound it, but real
f32 dot-product scores don't tie, so the single 4096-bin pass suffices.

## Porting checklist (other families)

1. **Confirm the lm_head is bf16/high-precision on disk.** The fine pass is only
   "exact" relative to whatever precision the weight actually is. If it was widened to
   f32 in the loader, fix that first (the fine pass reads the source bf16).
2. **Build the coarse copy** once at load: row-norm L2 + 3σ-clip Q{2,4}. Two copies
   live in RAM (coarse + fine bf16) — bandwidth is the constraint, not capacity.
3. **Measure recall@K** with the diagnostic harness (in hipfire-arch-zaya:
   `HIPFIRE_ZAYA_LMHEAD_SHORTLIST=1`, tune `_BITS`, `_CORRECT`) to pick the
   `(bits, corr_r, K)` that gives recall@1 = 100 % on a representative prompt.
4. **Wire the serving path:** coarse gemv (+ correction) → top-K → `gemv_bf16_gather_f32`
   fine/scatter into a `-inf`-masked logit buffer. Reuse the zaya `build_lmhead_coarse`
   / `coarse_scores_host` / `lmhead_twostage_serve` structure.
5. **Validate greedy-exactness** by diffing generated text against the full lm_head at
   `temperature=0` — it must be byte-identical for the exact modes.

## Portability

Kernels (`gemv_q4sym_f32`, `gemv_q2sym_f32`, `gemv_bf16_gather_f32`) are plain wave32
f32-accumulate — no WMMA, no arch-specific intrinsics — so they run unchanged on
RDNA2/3/4. The coarse build is host-side rayon. The only per-family work is the loader
seam that exposes the bf16 lm_head weight and the arch's decode wiring.

## Code map (reference implementation)

- `kernels/src/gemv_q4sym_f32.hip`, `gemv_q2sym_f32.hip` — coarse per-row symmetric scorers
  (uint32-vectorized loads).
- `kernels/src/gemv_bf16_gather_f32.hip` — fused fine gather + scatter.
- `kernels/src/lmhead_coarse_{minmax,hist,compact}.hip` — device top-K passes.
- `crates/hipfire-arch-zaya/src/gpu.rs`:
  - `build_lmhead_coarse` — row-norm 3σ-clip Q-build + optional low-rank correction.
  - `coarse_scores_host` — coarse gemv (+ correction) → host score vector.
  - `lmhead_twostage_serve` — top-K → fine gather/scatter serving path.
  - `coarse_score_gpu` — GPU-resident coarse score (correction added on-device).
  - `gpu_topk` — device min/max → histogram → threshold-scan → compact shortlist.
  - `parse_lmhead_twostage` — `HIPFIRE_ZAYA_LMHEAD` preset parsing.
  - `lmhead_shortlist_measure` — the recall@K diagnostic.
- Perf log: `docs/perf/zaya-decode-optimization.md` (EXP-17…EXP-34).
