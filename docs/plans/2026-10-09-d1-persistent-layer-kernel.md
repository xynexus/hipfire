# D1: one Qwen3.6-35B-A3B DeltaNetMoe layer as a counter-scheduled persistent kernel

Phase D1 of `2026-10-08-qwen35-fused-forward-scope.md`. Its single question:
**with dependency counters and idle-workgroup prefetch, does one A3B layer sustain
>= 70% of its byte floor at B = 1, 8, 32, 64?** Measured as bytes moved / layer
time, not tok/s. Kill: < 1.3x over today's op chain at every B.

## What runs today (multi-session batched decode, n rows, S sessions)

`forward_grouped_moe_session_batch_layers` (prefill_batch.rs), DeltaNetMoe arm +
`prefill_moe_ffn_body_batched` (prefill_chunk.rs). ~41 launches per layer at
n <= 32 (the attention half rotates and quantizes the normed activation once per
projection, each with that weight's AWQ scale):

| | op | kernel(s) | weights / state |
|---|---|---|---|
| A1 | attn rmsnorm | `rmsnorm_f32_gfx1151` | attn_norm |
| A2 | x4: rotate (per-weight AWQ) + int8 quantize + GEMV | `rotate_x_mq_awq`, `quantize_act_oq8`, `gemv_oq_compact_multicol_w{n}` | wqkv [8192x2048], wz [4096x2048], w_beta, w_alpha [32x2048] |
| A3 | gates | `fused_sigmoid_alpha_gate_f32` | a_log, dt_bias |
| A4 | conv1d + silu + split (routed) | `conv1d_silu_split_routed_f32` | conv_weight [8192x4], conv state ring (per session) |
| A5 | q/k l2-norm, repeat-interleave 16->32 heads | `fused_qk_l2_norm_scale_f32`, `repeat_interleave_qk_f32_batched` | |
| A6 | gated delta-net recurrence (routed) | `gated_delta_net_f16_routed_batch_seq` | S [32x128x128] f16 per session |
| A7 | gated norm | `gated_norm_f32` | norm_weight [128] |
| A8 | rotate + quantize + wo GEMV + residual | `rotate_x_mq_awq`, quant, multicol, `add_inplace_f32` | wo [2048x4096] |
| F1 | ffn rmsnorm, rotate (shared gate AWQ) | `rmsnorm_f32_gfx1151`, `rotate_x_mq_awq` | ffn_norm |
| F2 | router, shared-expert gate | `gemm_q8_0_batched(_rows)` x2 | router Q8 [256x2048], shared gate Q8 [1x2048] |
| F3 | shared gate/up | quant + multicol x2 | shared gate/up [512x2048] |
| F4 | softmax + top-8 renorm | `softmax_f32`, `moe_topk_renorm_k8_batched` | |
| F5 | shared silu*up, rotate, quant, down, sigmoid-scaled add | `fused_silu_mul_mq_rotate_awq`, quant, multicol (narrow), `scaled_add_inplace_gpu_sigmoid_rows_f32` | shared down [2048x512] |
| F6 | per-expert rotate, gate_up, silu*up rotate, down, combine | `rotate_x_mq_awq_indexed_batched`, `..._gate_up_k8_indexed_batched_splitk`, `fused_silu_mul_mq_rotate_awq_indexed`, `..._down_k8_indexed_batched_expanded`, `moe_down_combine_k8_batched` (n >= 16: the grouped W4A8 path) | 256 experts x [1024x2048] + [2048x512] |

## Byte floor per layer

dense ~20 MB (wqkv 8.9, wz 4.5, wo 4.5, shared 1.7, router 0.6) + the step's
expert union x 1.67 MB + DeltaNet state (S read + written, ~2 MB per session):

| B | experts (union) | bytes | floor at 248 GB/s | today (est.) | today / floor |
|---|---|---|---|---|---|
| 1 | 8 | ~35 MB | ~141 us | ~475 us | ~30% |
| 8 | ~46 | ~113 MB | ~455 us | ~1.3 ms | ~35% |
| 32 | ~111 | ~270 MB | ~1.09 ms | ~2.8 ms | ~39% |
| 64 | ~146 | ~392 MB | ~1.58 ms | ~6 ms | ~26% |

(today = decode step time / 40 layers at the current master rates; the harness
below measures it per layer directly.)

## Architecture

- **One cooperative launch per layer** (first cut), grid = full residency (WGPs x
  per-WGP fit), block 256. Cooperative launch is what guarantees residency, which
  the spin-waits need to be deadlock-free.
- **Work items.** Every op is split into items (GEMV row tiles, a session's
  conv/GDN heads, a row's norm, an expert's tile). Items are listed in a
  topological order on the host; a workgroup takes the next item with one atomic,
  waits until the item's op's prerequisites have completed (per-op completion
  counters in global memory, acquire loads), runs it, and bumps its op's counter
  (release). Taking items in topological order with all workgroups resident cannot
  deadlock: a waited-on item was taken earlier by a running workgroup.
- **Overlap comes from the graph**: wqkv / wz / w_beta / w_alpha items interleave;
  the shared-expert chain runs beside router -> top-k -> routed experts.
- **Idle prefetch (M2)**: a workgroup whose item is not ready touches the weight
  lines its item will read (and the next ops' dense weights) into L2/MALL instead
  of spinning.
- **Per-expert queues (M3)**: routed expert items are (expert, row tile) over the
  step's tokens routed to that expert, largest queue first, so a weight row is read
  once per expert per step at batch.

## Numerics

Each op body is the existing kernel's body, made a `__device__` function over a
"virtual block index" (several virtual blocks per 256-thread workgroup for the
32-thread kernels). Same arithmetic, same order: the layer's outputs (residual
stream, S, conv ring) must be BIT-IDENTICAL to the op chain. That is the parity
gate for every milestone.

## Milestones

- **M0 harness**: one real A3B DeltaNetMoe layer, its weights from the model file,
  random activations and states for B sessions; runs today's op chain for the layer
  and reports per-layer time and bytes/time at B = 1, 8, 32, 64.
- **M1**: persistent kernel, same op bodies, counters, no prefetch, no queue
  change. Bit-identical. Measures what removing launches + overlap buys.
- **M2**: idle prefetch.
- **M3**: per-expert queues at B >= 8.
- **Decision**: bytes/time vs the floor at each B against the 70% bar and the 1.3x
  kill line.

## Critical path at B = 1

~19 dependent steps (A1 -> A2 -> A3 -> A4 -> A5 -> A6 -> A7 -> A8 -> F1 -> F2 -> F4 -> F6 x4
...). Their bandwidth time is ~141 us; 70% of floor is ~200 us, so the counter
handoffs on that chain get ~3 us each. B = 1 is where the bar is hardest.

## Result (2026-10-09): killed at M1

Built on `exp/d1-persistent-layer` (parked, not merged): `kernels/src/d1_dn_moe_layer.hip`,
`dispatch/d1.rs`, and the `HIPFIRE_D1=1` arm in `forward_grouped_moe_session_batch_layers`.
It covers the whole DeltaNet **attention half** -- norm, the four in-projections (fused
rotate + int8 quantize + wide multicol), gates, routed conv1d, q/k l2-norm + head repeat,
routed f16 delta-net, gated norm, wo + residual -- as one cooperative launch per layer, 15
ops scheduled by per-op completion counters. The FFN half stays the chain.

**Numerics: bit-identical.** `HIPFIRE_D1_CHECK=1` runs D1 first on copies of the residual
stream and of each session's S and conv ring, runs the chain on the real ones, and compares
every buffer (17-29 incl. per-session state). 134 layer-steps at n = 2, 6, 8, 9, 14 with
1-8 sessions, multi-row sessions included: zero differing words. One trap on the way: the
copied conv body contracted `w3*x + w2*s0 + w1*s1 + w0*s2` differently than the chain
kernel under `-ffp-contract=fast` (~25% of words differ); it is pinned with explicit `fmaf`
in the chain kernel's order.

**Speed (serving, Qwen3.6-35B-A3B, aggregate tok/s, second round -- the first pays the
per-row-count kernel compiles):**

| n | chain | slice 1 only (norm + in-proj + gates) | attention half |
|---|---|---|---|
| 2 | 76.2-77.2 | 83.0 (+9%) | 67.0 (-13%) |
| 8 | 155.7-156.0 | 158.1 (+1%) | 133.7 (-14%) |
| 16 | 210.7-212.4 | 200.4 (-6%) | 183.8 (-13%) |
| 32 | 288.0-288.2 | 263.8 (-8%) | 247.9 (-14%) |

(n = 1 single-session short decode does not take this path.) **0.86x at every B against a
1.3x kill line.**

**Why.** A persistent kernel's register allocation is the maximum over every op body it
carries, so the light, latency-bound ops run at the occupancy of the heaviest one:

| | VGPRs | workgroups / WGP |
|---|---|---|
| chain multicol w2 / w8 / w16 / w32 | 95 / 135 / 191 / 223 | 8 / 5 / 4 / 3 |
| chain routed GDN (32-thread blocks) | 99 | -- (15 waves/SIMD) |
| chain conv1d | 19 | -- |
| D1, n = 1 / 2 / 8 / 32 | 138 / 145 / 188 / 256 (+14 spilled) | 5 / 5 / 4 / 3 |

At n = 2 that is 800 resident waves against the chain GEMV's 1280, and the delta-net
recurrence -- one wave per (head, 4-row tile, session) -- gets the same reduced slots. The
GEMVs and the recurrence are latency-bound, so fewer waves in flight is fewer bytes in
flight; that costs more than the ~30 launches and their gaps D1 removes. M2 (idle prefetch)
cannot buy back occupancy, and M3 (per-expert queues) is a property of the FFN's grouping,
which the chain already has at n >= 16 (#472). Neither was built.

**What the experiment found instead.** Measured cold (rotating > 96 MB of weight copies so
the 32 MB MALL cannot serve them), the wide multicol GEMV the attention half is made of
sustains only a fraction of 233 GB/s even alone:

| B | wqkv 8192x2048 | wz 4096x2048 | wo 2048x4096 |
|---|---|---|---|
| 1 | 52% | 44% | 32% |
| 8 | 63% | 48% | 30% |
| 16 | 43% | 34% | 23% |
| 32 | 23% | 21% | 18% |

So the layer's distance from its byte floor is mostly inside the kernels, not between them:
fixing scheduling around 30-60%-efficient GEMVs cannot reach 70% of the floor. The
follow-up is the GEMV's memory-level parallelism (issue a round's activation-tile and
weight loads together -- one exposed latency per round instead of two -- then pipeline the
next round's loads under the current round's math), which is bit-identical and helps the
chain directly.
