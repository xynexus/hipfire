# DFlash Phase 0 — working brief

Self-contained execution brief for Phase 0 of
`docs/plans/2026-07-19-hybrid-gpu-npu-cpu-spec-decode.md`. Carries the measured
state so a fresh session does not re-derive it, and the traps that cost the most
on the previous pass.

Full measurement detail: `docs/npu/dflash-native-driver-plan.md`.

**Goal.** Get the DFlash NPU draft block inside the GPU verify budget, measured
end-to-end — not projected. Then re-run Phase F to decide the weight format.

Repo `/home/sadara/hipfire`, branch `chaingun`, machine nix1 (gfx1103 GPU +
npu1/aie2 NPU, 4 columns).

## Measured state — do not re-derive

NPU DFlash block wall: **726 ms** (native driver,
`crates/hipfire-xdna/examples/dflash_body_native.rs`). Verify budgets: 9B 57 ms,
27B 155 ms, 31B 345 ms. Attribution, each term measured with the kernel pinned:

| term | original | NOW (measured, warm) |
|---|---|---|
| GEMM (weight-bandwidth-bound) | 317 ms | **102.3 ms** (98bbce9b6, r14 W4A8) |
| attention | 236 ms | **7.0 ms** (flash kernel wired into the body, task #28) |
| host glue (quant/bf16/packing) | 143 ms | **54.0 ms** (8da5aa5b3) |
| primitives (norm/rope/swiglu) | 24 ms | 23.7 ms |
| **warm block wall** | **726 ms** | **185.4 ms** |

**Attention is now the SMALLEST term (3.7% of the wall).** GEMM is 55%, host
glue 29%, primitives 13%. The `M_TILE=16` double-stream in the GEMM (~40 ms
above the ~60 ms bandwidth floor) is now the only large single lever left.

**⚠ MEASURE WARM-ONLY.** An earlier revision of this table was wrong twice from
one mistake: per-op means were averaged over ALL blocks including the cold first
block (~1.7× warm) and then subtracted from the WARM wall. That understated glue
(claimed 67.9, actually 105.7 before optimisation) and manufactured a
"~62 ms of hardware-context contention" term that **does not exist**.

**⚠ 236.0 ms is the 9B DRAFTER's block time. Do NOT compare it against a 27B or
31B budget — that is an invalid pairing.** A DFlash drafter is target-specific
(it consumes `target_hidden` from that target's layers), so each target needs its
own drafter and the draft cost scales with target size. Measured: Qwen3.5-9B
DFlash = 1.049 B params, Qwen3.6-27B DFlash = **1.730 B (×1.65)**.

Against its OWN budget the 9B pair is **185.4 vs 57 ms = 3.25× over** (was
4.14× before flash attention). Re-derived for the 27B pair from the CURRENT
per-term numbers — attention constant at 7.0 (both are 32q/8kv/128); GEMM
weight-bound ×1.65; glue/primitives track hidden ×1.25:

| term | 9B (measured) | 27B (derived) |
|---|---|---|
| GEMM | 102.3 | 168.8 |
| attention | 7.0 | 7.0 |
| host glue | 54.0 | 67.5 |
| primitives | 23.7 | 29.6 |
| **block** | **185.4** | **~273** |
| budget | 57 | 155 |
| **over by** | **3.25×** | **~1.76×** |

Moving to 27B helps by ~1.85×, not the ~2.7× that holding draft cost fixed
implied. See the corrected section in
`docs/plans/2026-07-19-hybrid-gpu-npu-cpu-spec-decode.md`.

**Items 1, 2 and attention are DONE.** Neither pair fits yet. The remaining gap
is **~118 ms for 27B**, and GEMM (168.8 derived, 55% of the wall) is now the
only large lever — specifically the `M_TILE=16` double-stream, ~40 ms above the
~60 ms bandwidth floor on 9B. Host glue (29%) is second.

**⚠ 185.4 ms IS A NO-CACHE, FULL-RECOMPUTE-EVERY-BLOCK WALL. In the actual loop
the per-cycle cost is much lower — the L axis is defeated (task #25, `9df9ae357`,
`--ctx-cache`).** The context cache stores `thp` and each layer's post-headnorm/
post-rope context K and raw V across cycles. Correctness is **bit-identical**
(cache-vs-recompute max|Δ| K=0 V=0 across all 5 layers; final block_hidden
cos = 1.000000000). It removes from the per-cycle path exactly the L-scaling
terms: `fc` (38.4 ms), the 5 per-layer `kv` GEMMs (21.0 ms), `rms-b32` (1.2 ms)
— 60.6 ms of direct compute at L=32, and these are the weight-bandwidth-bound
GEMMs that re-stream `ceil(L/16)` times, so without the cache they would be
~950 ms/block at L=512. **Cached, the only per-cycle term that still grows with
L is flash attention** (6.9 ms at tot=48 → ~33 ms at tot=528), which the cache
structurally cannot help (query rows are new every cycle) but which is cheap.

Projected cached per-cycle wall at L=512 ≈ **155–160 ms** (normalised to this
brief's machine state) — *below* the L=32 no-cache wall and ~10× under the
no-cache L=512 cost. **MEASURED: the bit-identical correctness, the per-op
removals, and a same-run 470→388 ms warm delta. PROJECTED: the L=512 number** —
in-body L-scaling could not be measured because the weights/manifest are built
at fixed N=l_ctx=32; a larger L needs a Python regen + new-shape kernel builds.
The k-side headnorm/rope exist only at b48, so the cached path still dispatches
them over all `tot` rows (output discarded); a b16 build trims those to noise
rows (projected, not measured).

Absolute walls in that run ran ~2.5× this brief's 185.4/132 (context-miss weight
re-faulting, a machine/thermal state, not the cache) — which is exactly why the
result is reported as a same-run delta.

**⚠ 236.0 ms IS A COLD-CONTEXT NUMBER, measured at L=32 with NO context cache.**
The harness recomputes the entire context projection every block. In a real
spec-decode loop the context grows every cycle, so the naive reading is that the
block wall scales with L (≈1700 ms at L=512, derived not measured). **That
scaling is a harness artifact, not a property of DFlash.** Confirmed by reading
`dflash_body_native.rs:994–1033`:

- `thp = rmsnorm(fc(target_hidden))` is computed **once per block** over all
  `l_ctx` rows (the code comment already calls it "one-time context projection").
- Each layer's context K/V comes from `thp` directly —
  `gemm!(.., gm_kv, .., &thp, l_ctx, ..)` — NOT from an evolving per-layer
  hidden state. Context rows contribute K/V only; they never run
  attention/FFN.

So every L-scaling term depends **only on that token's target hidden**, which is
frozen once the token is committed. `thp`, and each layer's post-headnorm,
post-rope `k_ctx` and `v_ctx`, are all position-invariant and therefore
**cacheable across cycles** (rope uses absolute position `pos0=0`, so row `r`
keeps position `r` as the context grows). Steady-state per-cycle work becomes
O(τ≈5 new rows), not O(L).

Cache size: k+v per token per layer = 2×8×128 f32 = 8 KiB, ×5 layers =
**40 KiB/token** (20 MiB at L=512, halved in bf16).

**What this does NOT fix:** attention is over `tot = L + B` and genuinely scales
with L. **MEASURED (task #26, `tools/npu/bench_dflash_attention_mc.py`,
`dflash_attn_mc` 4-core, block=16, 32q/8kv/128d, n=10 reps each):**

| tot (= L+16) | ms / dispatch (= ms/layer) |
|---|---|
| 16 | 4.262 |
| 32 | 8.465 |
| 48 | 12.938 (×5 layers = 64.7 ms; body measures 61.5 — instruments agree) |
| 55 | 14.327 |
| 56 | **BUILD FAILS** |

**Scaling is exactly linear, and the fixed overhead is ~zero.** Least-squares fit
over the four points: **0.262 ms per KV row + 0.096 ms fixed**. R² is essentially
1. Consequences:

- **Fusing the 5 per-layer attention dispatches into one buys ~0.4 ms of 61.5.**
  Not worth building. The 0.096 ms intercept is an *upper* bound — it includes
  the Python `iron` host call.
- **Bucketed builds vs shape-generic is the wrong question.** Neither helps: the
  cost is work, not shape-change. A bucket that pads tot up to the next power of
  two pays the padding linearly.

**HARD CEILING: tot ≤ 55.** tot=56 fails aiecc with `'aie.tile' op allocated
buffers exceeded available memory`. The design keeps one kv-head's whole KV
resident in core-tile L1 (64 KiB): stack 4096 + `memQ` 16384 + `outO` 16384 +
`memKV` = 512·tot. 512·tot ≤ 28660 → tot ≤ 55. This is **data memory, not the
16 KB program store and not the 1023 BD cap.** The current tot=48 build is
already at 87% of the ceiling — **this kernel cannot reach L=64, let alone 512,
by any amount of rebuilding.** It needs a tiled/streaming (flash-style) KV loop
before context length is even a discussable axis.

**The kernel is core-compute-bound, ~1000× off peak.** Core scaling at tot=48:
1 core 47.40 ms, 2 cores 24.84 (1.91×), 4 cores 12.94 (3.66×) — near-linear, so
it is not DMA- or dispatch-bound. But the arithmetic is only 12.6 MFLOP/dispatch,
i.e. **0.97 GFLOP/s across 4 cores (~0.1% of bf16 core peak)**, and the traffic is
~458 KB/dispatch = 35 MB/s against a ~10 GB/s path. `dflash_attention_sc_bf16.cc`
is vectorised at LANES=16 but computes each score with 8 `aie::mul` + 8
`aie::add` + a full `aie::reduce_add` per (q,k) pair — ~2400 core cycles per
128-length dot product. It uses no `aie::mmul`.

**⚠ THE NEXT SENTENCE WAS WRONG — kept because the mistake is instructive.**
This brief asserted: *"The lever is the inner loop (accumulator-tiled `mmul`,
hoisting the reduce)."* The rewrite (task #27, `75dafc5bc`) measured it:
**putting BOTH GEMMs on `aie::mmul<4,8,4>` bought only 1.36×** (0.262 → 0.192
ms/row), leaving ~1900 cycles/pair against the sc kernel's ~2400. The dot
product was never the limiter.

The actual limiter was the **one SCALAR `exp` per (q,k) pair**, because AIE2's
scalar unit has no fast float datapath. Vectorising it took 9.65 → 1.37 ms in a
single step — **7× of the total 10×.** The conclusion ("rewrite the inner loop")
was right for a reason that had nothing to do with the argument given for it.
**Generalizable rule: on AIE2, count scalar float ops before counting MACs.**

**Does the phase survive this? Yes — because the FLOPs are trivial.** Run the
arithmetic forward rather than extrapolating the measured ms:

- At the *current* 0.262 ms/row, tot=528 (L=512) costs 138 ms/layer × 5 =
  **691 ms of attention alone** against a 57 ms budget. Extrapolating the
  measured kernel says the architecture is dead.
- But attention at tot=528 is only 16×528×128×2×2 FLOP × 32 heads = 138 MFLOP
  per layer, **692 MFLOP per block**. At even a modest 50 GFLOP/s that is
  **~14 ms/block**; at 0.97 GFLOP/s it is 691 ms.

So the 691 ms is not a work bound, it is a 1000×-off-peak bound. Unlike the
GEMMs — which are genuinely weight-bandwidth-bound at a ~60 ms floor and cannot
be argued down — **attention has essentially unlimited headroom on paper.**
Do not extrapolate the current kernel's ms/row into any budget decision; it
measures the implementation, not the problem.

Note attention is the one term the context cache CANNOT help: the block's query
rows are new every cycle, so its O(L) growth is structural. That makes the inner
loop the load-bearing fix rather than an optimisation.

Attention, not the GEMMs, remains the real L-axis risk — but it is an
implementation risk, not an architectural one.

Re-measuring the *body* needs the golden set regenerated: `/tmp/dflash_w/index.json`
survives but `target_hidden.npy` / `noise_embedding.npy` are gone (tmpfs). The
attention kernel alone can be measured without it via
`tools/npu/bench_dflash_attention_mc.py`, which cross-validates against the body
figure (64.7 vs 61.5 ms at tot=48).

**CONTEXT CONTENTION IS NOT A LEVER — disproved, do not retry.** Sweeping cache
capacity 1→4 × {LRU, MRU} moves misses 31 → 16 while `npu_busy` stays FLAT at
186.9–187.7 ms. Misses cost only their host-side `load_peer` (~0.28 ms each), not
dispatch time; true alternation cost is ~0.2 ms/dispatch (~20%), not 5×. Fusing
the 8 primitives would recover ~5 ms, not 62.

**Remaining GEMM lever:** 103.5 ms against a ~60 ms bandwidth floor
(600 MiB packed W ÷ 10.4 GB/s). The gap is `M_TILE=16` forcing the rows=32 GEMMs
(`fc`, `kv`) to stream weights TWICE. That is the real one.

8-core attention is BLOCKED by the shim DMA budget (~2 MM2S/column; each worker
needs its own Q and KV stream) and would buy only ~30 ms — not the next move.

**The "~32–42 ms" projection this brief originally carried was WRONG** — it came
from an aggregate bandwidth figure (15.14 GB/s) mistaken for the weight path
(actually 10.0–10.7). Of the measured 123.7 ms, ~60 ms is the genuine bandwidth
floor (600 MiB packed W ÷ 10.4 GB/s — M_TILE=16 makes the rows=32 GEMMs stream
weights twice) and ~62 ms is **hardware-context contention**: npu1 admits 6 hw
contexts, the r14 array pins one, leaving 4 LRU slots for 8 primitive kernels →
36 context misses/block. Isolated probes hit 0.80–0.84 ms/dispatch; in-body the
same dispatches cost 1.24–3.30 ms. That contention is a NEW actionable term.

**Target 27B-class, NOT 9B** — but for a weaker reason than originally stated.
The budget scales with target size AND SO DOES THE DRAFTER (×2.72 vs ×1.65 for
9B→27B), so the net gain is ~1.6×. A 9B prototype is still permanently negative
(4.14× over its own budget); 27B is ~2.1× over its own. The integration plumbing
is architecture-independent, so build it on 9B and swap the target.

## CPU-offload levers (step-time, GEMM PINNED on the NPU)

The NPU dispatch loop is **strictly serial and single-threaded on the host**
(`dispatch_synced` + `sync_output` both block; `quantize_row` is AVX2 but
one core, no rayon in the body harness). So CPU and NPU never run at once, and
the CPU uses one of ~8 Zen4 cores. On Phoenix UMA there is **no transfer cost** —
an "upload" is a cache flush — which is what makes host offload cheap. Levers,
by expected payoff, with the GEMM assumed to stay on the NPU:

1. **Move the 8 primitives (rmsnorm ×2/layer, headnorm q+k, rope q+k, swiglu)
   to the CPU.** Compounds three ways: (a) removes 23.7 ms of NPU dispatch
   directly; (b) removes the bf16 pack/unpack glue that exists ONLY to feed them
   across the NPU boundary (part of the 54 ms), and keeps intermediates f32 on
   host; (c) **shrinks the NPU working set from ~10 resident kernels to ~2**,
   which is the real prize — the thrash diagnosis is "10 warm kernels ≫ ≤5 LRU
   slots" driving 40–46 context-misses/block and up to 2.5× wall inflation.
   MEASURED part = the 23.7 ms + glue; the thrash recovery is the larger but
   UNMEASURED part and is the number this work exists to get.

2. **Fuse / co-resident the r14 GEMM array + attention as the pinned NPU set.**
   The endpoint of lever 1: once the primitives are gone, the GEMM array and the
   flash attention kernel are the ONLY NPU consumers. Pin both resident (one
   persistent hw-context allocation, no LRU eviction between them) so a block
   never re-faults weight/attention BOs. This is what converts lever 1's
   working-set shrink into an actual zero-miss steady state — do it together with
   1, since 1 without pinning still lets the two evict each other. Watch the
   6-hwctx hard limit (`--ctx-budget 6` EINVALs; the r14 array already pins one).

3. **Overlap remaining glue behind NPU weight-streaming (double-buffer).** ~60 ms
   of the 102 ms GEMM is pure weight streaming — NPU busy, CPU idle. Pipeline so
   the CPU quantizes the next activation and rescales the previous int32 output
   during that window. Hides the glue that is not on the immediate dependency
   chain.

4. **Thread the remaining glue across cores.** `quantize_row` and the int32→f32
   rescale are per-row / per-element independent and currently one core. rayon is
   already a dep (`crates/hipfire-xdna/src/opus.rs`). Near-linear on 8 cores.

Not moving: attention (7 ms, wants NPU parallelism) and the GEMM (by assumption).
Structural/later: draft block N+1 on CPU while the GPU verifies block N (the
"tokens after next" axis) — Phase 2 DDTree/DSpark, not this step's breakdown.

**Gate for all of these:** cached-vs-recomputed / CPU-vs-NPU-primitive
intermediates bit-identical or cos > 0.999999, AND int8 full-body cos > 0.99
(the W4A8 path is ~0.898 by construction — gate it on acceptance rate, see
Gates). Keep every offload flag-gated so the all-NPU path stays reproducible.

## Tasks, in order

1. ~~**Wire the multi-core W4A8 GEMM into the body.**~~ **DONE** (98bbce9b6). Needs (a) an **oq4 DFlash
   sidecar** — only OQ8 exists. `dflash_convert` has `--oq4.<bits>`; use **pure W4
   (qt=33/34)**, since qt=36 mixed expands to dense int8 at upload and buys no
   bandwidth. (b) a **host-side stripe packer** matching r14's layout. Kernel
   artifacts already built and validated: `~/.hipfire/npu/r14_1x2x128_nb128`
   (recommended) and `r14_1x4x64_nb128`.
   **Trap:** `NpuGemmMp::load_cached` REJECTS an `r14_…` basename — its `_r{N}`
   guard matches the `r14` token itself. Use `load_with_tile`.
2. ~~**Multi-core the attention kernel.**~~ **DONE** (326971468, 4-core).
3. ~~**Attack host glue.**~~ **DONE** (8da5aa5b3, 105.7 → 53.0 ms). Remaining levers: the `M_TILE=16` double-stream in the GEMM (~40 ms above floor), and 8-core attention (~30 ms, blocked on shim DMA budget).
4. **Re-measure the block wall after each.** Report cold and warm separately.
5. ~~**Then re-run Phase F**~~ **DONE** — `benchmarks/results/dflash-phasef-acceptance-20260719.md`. Ship **oq4.25+** (τ 5.668, 0.9875 of f16).
6. **Phase 1 — context cache, then the seam.** The seam itself is clean:
   `spec_step_dflash` (`crates/hipfire-arch-qwen35/src/speculative.rs:6979`)
   leaves block hidden in `draft_scratch.x` and the next step applies
   `target.lm_head` to rows `1..B`, so the NPU swap is one substitution and
   losslessness is structurally safe (the target verifies everything).
   **Blocked on shape:** prebuilt kernels bake row counts into their names
   (`qwen35-rmsnorm-4096-b32` → L=32; `dflash-rope-k-8h128d-b48` → tot=48; GEMMs
   keyed by CompileTime `(M, K, N)` with N = rows). `ctx_len` is a bring-up
   choice, not a model property — the drafter sidecar has no `ctx_len` field
   (`max_position_embeddings: 262144`).
   Order of work:
   (a) **Host-side context cache** for `thp`, `k_ctx`, `v_ctx` — removes the L
       axis from `fc`, `kv`, `rmsnorm-b{L}`, `headnorm-k`, `rope-k`. These then
       run at a **fixed small row count** (16, matching `M_TILE`) over newly
       committed tokens, so no shape-generic build is needed for them.
   (b) ~~**Attention is the only remaining L-shaped kernel — and it is the
       blocker.**~~ **DONE.** `tools/npu/dflash_attention_flash_bf16.cc` +
       `build_dflash_attention_flash.py` (the sc pair is kept as the fallback).
       Streaming KV tiles + online softmax + `aie::mmul<4,8,4>`. MEASURED on
       nix1, 4 cores, block=16, 32q/8kv/128d, q_len=16 / kv_tile=48:

       | | sc kernel | flash kernel |
       |---|---|---|
       | tot=48 | 12.246 ms | **1.214 ms** (10.1×) |
       | tot=528 | would not build | **6.607 ms** |
       | tot=4080 | would not build | **48.342 ms** |
       | ms / KV row | 0.262 | **0.0125** (21×) |
       | GFLOP/s | 0.97 | **10.4 (tot=48) → 22.1 (tot=4080)** |
       | cos vs f32 / bf16 ref | 0.999997 / 0.999999 | **0.999996 / 0.999997** |

       **The tot ≤ 55 cap is gone** — core L1 now holds one KV tile, not the
       whole KV, so it is independent of tot. tot=4080 builds and runs with
       parity intact and ms/row still flat; no new limit was reached.
       Attention at the in-body shape drops **61.5 ms → ~6.1 ms**; at L=512
       (tot=528) 5 layers cost **33 ms**, versus the 691 ms the sc kernel's
       ms/row extrapolated to.

       **⚠ The brief's attribution of the sc kernel's cost to the dot product
       was WRONG, and it matters.** Putting both GEMMs on `aie::mmul` alone
       bought only 1.36× (0.262 → 0.192 ms/row) — the kernel still cost ~1900
       core cycles per (q,k) pair, essentially unchanged. The real limiter is
       that **AIE2's scalar unit has no fast float datapath**, and the sc kernel
       ran one SCALAR exp per (q,k) pair. Vectorising the exp (`exp_neg_v`, a
       magic-constant round-to-nearest 2^n split feeding a degree-5 series)
       took 9.65 ms → 1.37 ms in one step, 7× of the total 10×. **On AIE2,
       count scalar float ops before counting MACs.** `aie::exp2` is
       `arch::XDNA_2`-only (aie2p), so the software vector exp is required on
       npu1.

       Measured null: `q_len=32` (no change), `kv_depth=2` (no change — the
       kernel is not DMA-bound). `kv_tile=48` beats 16 by ~1.35× at long
       context. Bench: `tools/npu/bench_dflash_attention_flash.py`.

       **WIRED INTO THE BODY (task #28).** Not an xclbin swap — the host ABI
       differs on every axis (iterations are q-heads not kv-heads, Q/Kᵀ/V are
       mmul-tiled, O comes back C-tiled), so `dflash_body_native.rs`'s packing
       and unpacking were rewritten. Selected with `--attn flash`; the sc path
       stays the default fallback and is bit-identical to before.
       Registered by `tools/npu/swap_attn_flash_manifest.py`.

       | | sc (in body) | flash (in body) |
       |---|---|---|
       | attention / dispatch | 12.30 ms | **1.40 ms** (8.8×) |
       | attention / block (×5) | 61.5 ms | **7.0 ms** |
       | warm block wall | 238.8 ms | **185.4 ms** (185.0/181.9/189.4) |
       | cos vs f32 golden / bf16 ref (int8 GEMM path) | 0.998114 / 0.998170 | **0.998083 / 0.998140** |

       In-body 1.40 ms vs the standalone kernel's 1.21–1.35 ms: the ~0.18 ms
       gap per dispatch is the already-documented context-alternation cost,
       not a wiring loss. **Host glue moved 53.0 → 54.0 ms** despite the flash
       packing being tiled and replicating KV `groups`× — the extra packing is
       ~1 ms, not a tax worth designing around. Context budget unchanged: the
       flash kernel pins one hw context like the sc one, `--ctx-budget 4` still
       correct, dispatches/block still 117.

       **TAIL MASKING (v2), not padding-with-zeros.** v1 required `kv_tile` to
       divide `tot`, which the spec-decode loop cannot honour (`tot = L + B`
       grows every cycle). `tot` is now padded up to a multiple of `kv_tile`
       and the pad rows are removed by an **additive f32 score mask carried as
       runtime data inside each KV tile** (trailing `2*kv_tile` bf16 slots,
       read through a `float*` reinterpret). Zeroing the padded K/V does NOT
       work: with online softmax a zeroed K row scores 0, and `exp(0 - m)` is
       an ordinary weight that inflates the running sum and scales the output
       down. Masked lanes land at −3e30, which `exp_neg_v`'s −126 exponent
       clamp floors to 2^−126 — the same mechanism the `NEG_BIG` running-max
       sentinel already used, so no new numerical path.
       Cost: one vector add per (q, tile-row). Measured null — tot=48 and
       tot=528 timings and cosines are unchanged from the pre-mask kernel, and
       tot=50 (46 masked rows, which v1 could not build at all) holds
       cos 0.999996. In-body with `kv_tile=32` (tot=48 → 2 tiles, 16 masked
       rows) parity is 0.998143 / 0.998172.

       **Constraint this imposes on the context cache (task #25):** only
       `n_tiles` remains compile-time, so the attention build changes once per
       `kv_tile` committed tokens rather than every cycle. The cache must
       either tolerate a rebuild every `kv_tile` tokens or hold a prebuilt
       ladder of `n_tiles`. It does NOT need to advance `tot` in whole tiles —
       any `tot` is legal, it is just rounded up and masked.
   (c) Then the seam, with the gates below.
   `--ctx-slice 32` (`dflash_spec_demo.rs:618`) would run the seam today and
   prove plumbing + losslessness, but at 32 context rows the resulting τ and
   tokens/s say nothing about the architecture. It also needs `position >= 32`,
   which the ~10-token gate prompt does not satisfy on the first cycle. Use it
   as a plumbing smoke test only — do not quote numbers from it.

## Gates

**⚠ THE 0.99 PARITY GATE APPLIES TO THE int8 PATH ONLY.** As previously written
("Full-body cosine > 0.99 … Do NOT loosen") it was **unachievable on the
shipping W4 path and contradicted two other records in this repo**:

- `--gemm multicore` (W4A8) measures **cos 0.898/0.897** full-body. That is
  pre-existing 4-bit codebook loss, not a regression — `98bbce9b6`'s own commit
  message already states no W4 format can pass 0.99.
- Phase F **proved SNR is the wrong gate for a drafter** (see Traps): pure int4
  fails cosine badly yet costs only 7.2% of τ, and oq4.25+ costs 1.25%.

So use the right gate for the path under test:

| path | gate |
|---|---|
| int8 (`--gemm` default) | **full-body cos > 0.99** vs f16 golden AND int4/bf16 reference. Do NOT loosen — attention/primitive changes are isolated here, which is what makes it a usable regression gate. |
| W4A8 (`--gemm multicore`) | **acceptance rate (τ)**, per `benchmarks/results/dflash-phasef-acceptance-20260719.md`. Cosine is ~0.898 by construction; treat a *change* from 0.898 as the signal, not the absolute value. |

Reference: int8 path is **0.998083/0.998140** with flash attention wired
(sc was 0.998114/0.998170, Δ 3e-5). W4A8 is **0.898399/0.897333** (sc:
0.898395/0.897311).

The F32 sidecar is a bug repro only — `gemm_f32_batched` has a batch>1 transpose
bug and a pure-F32 drafter scores τ=0.

**Losslessness (must not regress).** At temp 0, all drafters commit
BYTE-IDENTICAL tokens to `--ar-baseline` while differing in accepted counts:

```
T=~/.hipfire/models/qwen3.5-9b-mq4.hfq
./target/release/examples/dflash_spec_demo --target $T --draft <D> \
  --prompt "Explain how a four-stroke engine works." --max 96 \
  2>/dev/null | md5sum        # -> 02e621bd56b5 for AR and every drafter
```

**`--ar-baseline` ALSO REQUIRES `--draft`** or it panics at
`dflash_spec_demo.rs:822`. Run it without, and stdout is empty — `md5sum`
returns `d41d8cd98f00`, the digest of the empty string, which reads as a total
mismatch against every drafter. Same false-regression trap as below, one layer
down. **Assert the digest is not `d41d8cd98f00` before comparing anything.**
Corrected, the gate passes: AR baseline and f16 drafter both `02e621bd56b5`,
3/3 repeats each.

**`2>/dev/null` IS LOAD-BEARING — do not drop it.** stdout is the generated text
(the substantive invariant); stderr carries a `BENCH METRICS` block with
wall-clock timings that changes every run. An earlier revision of this brief
wrote the gate as `md5 over | tail -20` WITHOUT specifying stderr handling, which
made the digest unstable and read as "every drafter diverges" — a false
correctness regression that was nearly filed. Digest **all of stdout**, not a
tail window.

## Traps that cost the most last time

- **The verify forward WAS nondeterministic**; single-run md5 comparison was
  measuring noise and produced 4+ wrong eliminations. Deterministic now
  (`6ca303af8`), but ALWAYS use ≥3 repeats and assert cross-run identity first.
- **`./tests/coherence-gate-dflash.sh` compares single runs** and structurally
  CANNOT catch that bug class. Not sufficient on its own.
- **A hypothesis must explain the PRIMARY SYMPTOM.** "Baked scalar" was proposed
  for a variance bug; a baked scalar is stale but DETERMINISTIC. Check first.
- **Check the claimed-CORRECT side of a comparison**, not just the claimed-broken
  side. One filed bug asserted the serial path "honors" a value it does not.
- **SNR is the WRONG gate for a drafter weight format — now PROVEN.** Spec decode
  is lossless, so quality costs ACCEPTANCE RATE, not correctness. Phase F
  (`benchmarks/results/dflash-phasef-acceptance-20260719.md`) measured it: pure
  int4 fails SNR badly (cos 0.898, ~22 dB down) yet costs only **7.2% of τ**.
  The SNR gate would have rejected a perfectly usable format.
- **Residual verify nondeterminism ~1.5% (1 in 68 runs)** — one token flip
  observed in a Phase F sweep cell, NOT a format effect (18/18 repeats of the
  same command reproduced the reference). Below what motivated `6ca303af8` but
  not zero. Single-run md5 remains an unsafe gate.

## Dead ends — do not repeat

- NPU weight path saturates **~10 GB/s per routing topology**, ~13 GB/s across two
  orthogonal routes. **EIGHT knobs measured null**: channel count, consumer count,
  buffer depth, compute load, activation traffic, shape, burst length,
  buffer-region layout. **Weight BYTES are the only lever left.**
- `opus`/`NpuOpusExecutor` is aie2p/npu2-only — unusable on npu1.
- Mixed-precision overlays are second-order for quality: n_out 3→63
  (4.25→8.00 b/w) buys **1 dB**; FWHT rotation buys **0.2 dB**. int4 loses ~22 dB
  and that is textbook (5.5 dB/bit), not a bug.
- **DeltaNet state must NEVER be Q8** (policy, `51e1ac078`). It is FP32 now.

## Guardrails

GPU/NPU work holds the lock: `./target/release/hipfire lock acquire <name>` /
`release`. rustup cargo (`export PATH="$HOME/.cargo/bin:$PATH"`). `graphify
query` before grepping repo source (hook-enforced) — include this in subagent
prompts. Do NOT touch `.agents/scheduled_tasks.lock`, `third_party/`, or
`benchmarks/npu_gemm_tuning/` except to add a new round dir. Commit validated
work with evidence in the message; report failures plainly with the numbers.
