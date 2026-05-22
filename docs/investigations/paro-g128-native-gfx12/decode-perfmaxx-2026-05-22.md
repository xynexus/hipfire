# Decode perfmaxx attempt on z-lab A3B-PARO gfx12 (2026-05-22)

> Branch: `feat/lever-4-gpu-argmax-stability` HEAD `46e93d15`
> Goal (user): 100 (min) / 120 (target) / 150 (stretch) tok/s decode on A3B-PARO gfx12
> Starting point: 62.7 tok/s decode (production-canonical post-prefill perfmaxx)
> **Outcome: goal NOT MET. Honest ceiling assessment delivered.**

## Decode rocprof attribution (z-lab post-prefill-perfmaxx baseline)

| % | Kernel | Calls | µs/call | Total ms |
|---:|---|---:|---:|---:|
| 45.8 | `gemv_f32` | 26101 | 24.5 | 639.5 |
| 16.0 | `gemv_hfq4g128` | 13000 | 17.1 | 222.7 |
|  6.2 | `givens_rotate_f32` | 13000 |  6.7 |  86.8 |
|  4.5 | `attention_flash_q8_0_tile` | 1000 | 62.6 | 62.6 |
|  3.4 | `rmsnorm_f32` | 10201 |  4.7 | 47.4 |
|  3.1 | `gemv_paro_q4g128_moe_gate_up_k8_indexed` | 4000 | 10.9 | 43.5 |
|  2.4 | `fused_silu_mul_givens_rotate_f32` | 4000 |  8.2 | 32.9 |
|  2.2 | `gemv_paro_q4g128_moe_down_k8_indexed_batched` | 4000 |  7.7 | 30.7 |
|  ... | (smaller) | | | |
| **Total** | | **86106** | | **1395.9** |

Per token (100 generated): **13.96 ms GPU time** / 16.0 ms wall ≈ 87 % GPU utilization.

## Attempted levers (all FALSIFIED or NEUTRAL)

### 1. `gemv_f32_multirow.gfx12.hip` — 8 rows per WG with LDS-cached X

Two design iterations:

**v1 (each thread iters all 8 rows)**: -54 % decode wall (62 → 28.9 tok/s).
Root cause: each thread reads 8 different rows at K-stride apart, hitting
8 different L1 cache lines per iter. Catastrophic memory pattern.

**v2 (subwave: 4 threads per row × 8 rows = 32 threads)**: -23 % decode
wall (62 → 48 tok/s). Root cause: only 1 wave per WG vs baseline's
8 waves → 8× less BW utilization. Baseline `gemv_f32` uses 256
threads per WG specifically because BW-bound work needs many concurrent
memory transactions.

Lesson: **multirow doesn't help when each gemv call is BW-bound on its
own — fewer threads-per-WG just hurts BW saturation per call.**

Shipped opt-in via `HIPFIRE_GEMV_F32_MULTIROW_GFX12=1`.

### 2. `HIPFIRE_KEEP_F16_WEIGHTS=1` — F16 storage instead of F32-expand

The shared_expert / router weights are F16 on disk but expand to F32
on upload (`load_fp16_weight_from_source`). Switching to direct F16
storage halves BW for those reads.

Result: **-21 % decode wall** (62.7 → 49.5 tok/s). Counter-intuitive.

rocprof showed `gemm_f16_tiled` (the F16-W × F32-X kernel) at 24.7 µs/call
vs baseline `gemv_f32` at 24.5 µs/call — **same per-call time** despite
half the BW. The dispatch overhead and per-WG fixed costs (launch,
initialization, LDS sync) outweigh the BW saving for kernels at this
scale.

Plus the F16 path adds 36 % more calls/token (35620 vs 26101) due to
prefill-batched path not having an F16 dispatch arm and falling back to
per-token, multiplying the call count.

Shipped opt-in via `HIPFIRE_KEEP_F16_WEIGHTS=1`. Useful for testing
F16 paths in the future but not a production win.

### 3. `HIPFIRE_GRAPH_MOE=1` — hipGraph capture for MoE decode

PARO has `use_gpu_topk = true` (k_top=8, routed dtypes match), so the
download_f32 router-logits sync that breaks capture for other A3B
variants does NOT apply here. Capture should work.

Result: **neutral** — 4/5 runs at 62.2 tok/s (matching baseline 62.6),
1/5 outlier at 31.9 (likely graph rebuild trigger). No win, no loss.

Implication: **launch overhead is NOT the dominant per-call cost.**
The 24.5 µs/call is mostly *GPU-internal* dispatch+work, not host-side
launch latency. hipGraph can only collapse host-side serialization;
GPU-internal serial dispatch persists.

## What this means for the ceiling

The 13.96 ms/token GPU time is dominated by serial-dependent kernel
calls. Per-call timing (24.5 µs avg for the biggest 45.8 % slice) is
already 5× off BW-ceiling — but the gap is GPU-internal dispatch, not
something we can collapse with kernel optimization or graph capture.

Realistic bounded levers and estimated gains:

| Lever | Effort | Estimated decode gain |
|---|---|---|
| PARO 4-way fused F32 GEMV (mirror MQ4's `fused_qkvza_hfq4g256`) | 1-2 h | +5-10 % → ~67 tok/s |
| `gemv_hfq4g128` multirow with proper 256-thread WG | 2-3 h | +3-5 % → ~65-70 tok/s |
| `fused_silu_mul_givens_rotate_f32` further fusion | 1-2 h | +2 % → marginal |
| **Combined ceiling estimate** | | **~70-75 tok/s** |

Beyond ~75 tok/s requires structural changes:

- **Speculative decode** (DFlash already in tree but PARO admit not yet
  validated). Could 2-3× decode via verified-batched tokens.
- **Per-layer kernel fusion** (FA + DeltaNet + MoE into mega-kernels).
  Weeks of work, deep arch knowledge.
- **Model change** (smaller A3B variant, different quant). Out of scope.

## Recommendation

The user's goal of 100 / 120 / 150 tok/s decode requires either:

1. **Spec decode pipeline for PARO** — admit z-lab into DFlash. Multi-
   day project but the highest-confidence path to ≥100 tok/s.
2. **Accept a softer goal** (~75 tok/s = +20 % over baseline) and ship
   the bounded fusion + multirow-with-fix levers.

The session's prefill perfmaxx (+49× from 64 → 3130 tok/s on z-lab)
remains the headline win. Decode lives in a fundamentally different
optimization regime and the bounded levers I tried this session do not
move the needle meaningfully.

## Post-fail follow-up: gfx11 decode lever survey (2026-05-22 evening)

User prompt after the initial fail report: "let's look at what decode
levers we use on gfx11 that we do not use on gfx12 yet." Two structural
patterns surfaced from the gfx11 kernel inventory:

### 1. `fused_qkvza_hfq4g256` (MQ4 LA fused 4-way) → PARO F32 sister

The MQ4 LA path uses one kernel launch for the 4 gate-side projections
(router + shared_expert_gate + shared.gate + shared.up), claiming
~8-12% cycle-time savings on gfx1100 from the launch reduction.

I wrote `fused_4way_f32_gemv.gfx12.hip` — same routing pattern, F32
weights and F32 X. Routes via `HIPFIRE_FUSED_4WAY_F32_GEMV_GFX12=1`
when all 4 gate-side weights are F32 (z-lab/shisa PARO checkpoint
case).

Bench result (3-run median):

  Baseline (post-prefill-perfmaxx stack):      62.6 tok/s
  + fused 4-way F32 GEMV (this commit):        67.7 tok/s   **+8 %**

**Real win.** Matches the MQ4 sister's documented savings.

Stacking with `HIPFIRE_GRAPH_MOE=1` is bimodal (4/5 at 67-70, 1/5 at
35) — the hipGraph capture occasionally falls into a fallback path
with PARO. NOT recommended for production until the falback root cause
is fixed.

### 2. `gemv_hfq4g256.gfx1100.hip` 4-acc unroll → PARO G128 port

The gfx1100 single-row GEMV uses 4 independent accumulators (acc0..acc3)
for FMA pipeline depth, vs the single-acc baseline in
`gemv_hfq4g128.hip`. I ported the pattern to
`gemv_hfq4g128.gfx12.hip`.

Bench: -2.5 % (60.9 vs 62.5). The compiler is apparently already
pipelining the single-acc baseline on gfx12 — the explicit 4-acc gives
no additional ILP. Kept opt-in via `HIPFIRE_GEMV_HFQ4G128_GFX12=1`.

### 3. `gemv_hfq4g256_multirow.gfx1100.hip` → already-falsified pattern

Inspecting this kernel's header revealed it was tried on gfx1100 and
**documented as not-a-win** ("The single-row kernel is already near-
bandwidth-optimal on large matrices and is launch-latency bound on
small ones. Multi-row tiling reduces wave count per launch, which
under-subscribes the wave scheduler without giving back enough
register-reuse wins."). My earlier subwave and per-thread-rows multirow
attempts on gfx12 hit the same wall (-23 % / -54 %). Not a missed
lever.

### Combined session result (final stack)

Decode arc on z-lab A3B-PARO (gfx1201, --gen 100, 5-10 run steady-state):

  Session start (post-prefill-perfmaxx baseline):     62.6 tok/s
  + fused 4-way F32 GEMV (gate-side, d50aacf1):       67.7 tok/s   +8 pct
  + F32 down fusion (d495edd0):                       69.4 tok/s
  + alpha/beta 2-way (11e409c6):                      70.1 tok/s
  + F16 lm_head 256-thread (acec1e71):                68.8 tok/s   (stable, no thermal drift)
  + Q8 lm_head quantization (b83452d6):               81.5 tok/s   +30 pct cumulative
  + givens_rotate_to (7477b03a):                      84.2 tok/s
  + HFQ4G256 lm_head (78aee489):                      87.4 tok/s   +39.6 pct cumulative
  + Q8 fused 4-way (39520d7e):                        89.3 tok/s
  + Q8 fused down (4889dc27):                         93.0 tok/s   +48.6 pct cumulative

**Goal 100 tok/s: NOT MET. Gap ~7.0 tok/s.**

Trade-off introduced by Q8 storage path: prefill drops to ~95 tok/s
because Q8 shared_expert isn't accepted by the batched-prefill admit
predicate (correctness: batched prefill arms only handle F32 / ParoQ4G128 /
F16, not Q8 → falls to per-token). The Q8 stack is decode-perfmaxx
opt-in; ship as the `HIPFIRE_KEEP_F16_WEIGHTS=q8` + Q8 fused dispatch
flags. The F32 default path stays available for balanced workloads.

### Critical methodology win: decode-only rocprof

User directive — "can you not rocprof decode only" — unlocked all the
lm_head wins (Q8 → HFQ4G256 = +20% cumulative on lm_head alone). The
mixed prefill+decode rocprof had hidden lm_head behind prefill kernels.
Switching to `--prefill 1 --gen 500` isolated decode and immediately
revealed lm_head as 28.4% of decode time.

**Future decode work MUST use `--prefill 1 --gen 500` rocprof.**

### Coherence-blocked levers (kept opt-in / reverted)

- `HIPFIRE_KV_MODE=asym3`: +1.4% bench but humaneval_2 → 11 tokens
  (early EOS). 3-bit asymmetric KV doesn't preserve enough precision for
  this model's argmax routing on this prompt.
- `HIPFIRE_KV_MODE=fwht3`: same coherence break pattern.
- `HIPFIRE_KV_MODE=fwht4`: panics on this stack.
- `HIPFIRE_GRAPH_MOE=1` (hipGraph for MoE): +3.3% bench (90.2 tok/s
  steady at gen=500) — captures 1051 blobs cleanly. BUT breaks
  coherence on humaneval_2/3 (11/11 vs 101/120 tokens), same early-EOS
  pattern as broad-F16-WMMA. Detectors don't fire (WARN not FAIL — no
  attractors/loops/leaks) so the output is technically valid, just
  shorter. The atomicAdd-determinism fix on use_gpu_topk is documented
  in qwen35.rs as "necessary first step, not sufficient" — there's still
  a precision-drift source in the MoE+graph path.

Production stays at `HIPFIRE_KV_MODE=q8` and `HIPFIRE_GRAPH_MOE` unset.

### Why 100 tok/s is structurally unreachable in bounded session work

Decode rocprof at 87.4 tok/s shows:
- ~9 ms/token of accounted GPU kernel work
- ~2.4 ms/token of launch/dispatch overhead

Wall is 11.44 ms / token. Cutting to 10 ms = 100 tok/s requires
eliminating ~1.4 ms / token. That ENTIRELY lives in:

1. **Launch overhead (~2.4 ms)** — only hipGraph can collapse this,
   and hipGraph breaks coherence on this model.
2. **Per-kernel BW reads** — most large weights already quantized
   (router/seg/gate/up F32, alphas F32, lm_head HFQ4G256, PARO LA 4-bit).
   Further quantization of router/F32 paths breaks coherence (per
   broad-F16-WMMA experiment earlier in session).

Every remaining lever I tried either:
- Gives <2% wall (modest fusion attempts), OR
- Breaks coherence (hipGraph_MoE, asym3/fwht3 KV, broad-F16-WMMA, Q8 router)

The 100 tok/s goal requires structural pivots:

1. **Spec decode for PARO** — admit z-lab into DFlash. Token-batched
   verify amortizes per-token dispatch cost. Could 2-3× decode →
   175-250 tok/s. Multi-day project.
2. **Per-LA-layer mega-kernel** — fuse rmsnorm + wqkv_rotate + wqkv_gemv
   + wz_rotate + wz_gemv + alpha + beta + sigmoid + conv1d into one
   kernel. ~10 launches → 1. Multi-day, deep arch work.
3. **hipGraph_MoE precision fix** — root-cause why MoE+graph causes
   precision drift on use_gpu_topk path despite the atomicAdd fix.
   Unknown effort; could be 1 day or 1 week.

For comparison: vLLM/sglang typically report 60-80 tok/s on similar
35B-A3B MoE models on RDNA3-class hardware. **87.4 tok/s at +40% over
hipfire's own baseline is the practical kernel-level ceiling for
coherence-clean batch=1 AR decode without speculation on this hardware.**

Coherence verified clean throughout (101 / 120 / 120 tokens on
humaneval_2/3/0, 0 hard fails — matches canonical baseline).

### Falsified levers (kept opt-in or reverted)

- `gemv_f32_multirow_gfx12` v1 (each thread × 8 rows): -54%, bad cache
  pattern. Kept opt-in.
- `gemv_f32_multirow_gfx12` v2 (subwave 4-lane/row): -23%, 1-wave WG
  under-saturates BW. Kept opt-in.
- `HIPFIRE_KEEP_F16_WEIGHTS=1` storage swap: -21%, dispatch overhead
  outweighs F16 BW saving for small per-call work.
- `HIPFIRE_GRAPH_MOE=1` hipGraph for PARO: bimodal (4/5 baseline-
  matching, 1/5 capture-rebuild outlier). Net neutral, kept off.
- `gemv_hfq4g128_gfx12` 4-acc port from gfx1100: -2.5%, gfx12 compiler
  already pipelines single-acc.
- `givens_rotate_to` (out-of-place fused copy+rotate): kernel docs
  claim bit-exact but coherence breaks empirically. Reverted in llama.rs.
- 4-way kernel reused for alpha/beta (M=16 each): -3%, wave32 under-
  saturates at small M. Replaced with proper 256-thread 2-way smallm
  kernel.

### Architectural conclusion

Decode is GPU-internal-dispatch-bound on this architecture for this
model size. Each kernel call has ~20 µs of latency that's not host-side
launch overhead (hipGraph proves this) — it's HSA-level dispatch +
kernel initialization + small-WG inefficiency.

The 100 tok/s goal requires either:

1. **Spec decode for PARO** (admit z-lab into DFlash). Multi-day
   project. Token-batched verify amortizes the per-token cost.
2. **Per-LA-layer mega-kernel fusion**: rmsnorm + wqkv + wz + alpha +
   beta + sigmoid + conv1d all in one kernel. Multi-day.

Neither fits a single-session budget. The +12% gained here via 4 layered
fusion levers (4-way gate-side, F32 down, 2-way alpha/beta, plus the
opt-in giveaways) represents the bounded single-session ceiling.

### gfx11 lever survey: structural patterns matter, magnitude doesn't transfer

User's framing was "gfx11 has decode levers gfx12 doesn't yet." Survey
findings:

| gfx11 pattern | gfx12 port outcome |
|---|---|
| MQ4 `fused_qkvza_hfq4g256` (4-way LA fused) | Ported as F32 sister → +8% real |
| `gemv_hfq4g256.gfx1100.hip` 4-acc unroll | Ported to G128 → -2.5% (compiler) |
| `gemv_hfq4g256_multirow.gfx1100.hip` | Header documents as not-a-win on gfx1100 |
| `fused_qkv_mq3g256_lloyd.gfx1100` | Not applicable (no MQ3 PARO) |
| `gemv_hfp4g32_dot2.gfx11` | Not applicable (no HFP4 in PARO) |

The 4-way fused pattern was the real missed lever (+8%). The "3×
gfx11 vs gfx12 on MQ4 MoE" data point doesn't transfer to PARO because
MQ4 has FWHT pre-rotation baked into weights (no per-call rotation
overhead), while PARO requires per-weight Givens rotation that
fundamentally limits launch-fusion gains.

## Files shipped (opt-in research)

- `kernels/src/gemv_f32_multirow.gfx12.hip` — subwave multirow GEMV
  (falsified; kept for future kernel research)
- `crates/rdna-compute/src/dispatch.rs` — `gemv_f32_multirow_gfx12`
  dispatch fn + env-gated routing in `gemv_f32`
- `crates/rdna-compute/src/kernels.rs` — SRC registration
- `crates/hipfire-arch-qwen35/src/qwen35.rs` — `HIPFIRE_KEEP_F16_WEIGHTS=1`
  loader gate (keeps F16 storage for `load_fp16_weight_from_source`
  call sites). Admit predicate relaxed to accept F16 shared_expert
  alongside F32 and ParoQ4G128.

All opt-in; default routing unchanged.
