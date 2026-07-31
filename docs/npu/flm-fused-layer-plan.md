# Fused decoder layer — design plan (multi-agent, 2026-07-31)

Produced by an 11-agent workflow: four parallel surveys (layer math, existing
kernels, capacity budget, FLM constraints), three independent design proposals
optimised for different objectives, three adversarial feasibility judges, and a
synthesis. Recorded as the design of record; NOT built or validated beyond what
its own gates state.

**Tasks 0 and 1 are falsifiers — run them before building anything.** The design
rests on the cost of a phase barrier *inside* one dispatch being below the
measured 92.9 us per-dispatch cost, and that number does not exist yet.

The judges killed several proposals outright, and those rejections are worth as
much as the plan: a 24-arm gather tree is illegal (`objectfifo cannot be in more
than one ObjectFifoLinkOp`), a memtile join caps at 6 arms, a core has 2 inbound
DMA channels so no 24->1 gather to one core exists in IRON at all, and
`MemTile(c,r)` is not a real constructor (`Tile(c,r)` is).

---

# Fused llama‑3.2‑1B decode layer — buildable plan

**Base:** the max‑throughput design (16 cores, 8 pairs, one shared 20 KB operand fifo, one broadcast fifo, one result fifo — exactly `gemv_bench.py`'s proven wiring). **Grafted:** K‑chunked `down_proj` (judge 1+3), alternating‑acquire gate/up with a 64 B in‑core stash (judge 2). **Discarded:** the 24‑arm gather tree and single hub core (illegal — a fifo cannot be in two `ObjectFifoLink`s, and a memtile join caps at 6), `MemTile(c,r)` (doesn't exist — `Tile(c,r)`), the "14 direct shim streams" experiment (the 12 % split tax was retracted in `aee461b3b`), 32 cores / neighbour memory (unproven, buys 0 rows at every shape this model has), RoPE in a 32‑core QKV epilogue (straddles head boundaries), monolithic K=8192 `down_proj` in a fused layer (over 64 KB at any core count).

**Target: 1 dispatch per layer + 1 for lm_head = 17/token.** The 16‑layer unroll to 2 commands is a follow‑on, not a prerequisite.

---

## 0. What already exists (do not rewrite)

| file | status |
|---|---|
| `kernels/npu/flm_q4_1_tile.h` | tile body, `inline g_asum`, 5.00 bpw |
| `kernels/npu/flm_gemv_q4_1.cc` | 48.1 GB/s @16 cores, 1.9e‑07 |
| `kernels/npu/flm_norm_prepare.cc` | **RMSNorm fused into the GEMV prologue**, in place, `[act K][nw K]` in one buffer. 3.48e‑07 |
| `kernels/npu/flm_gemv_residual.cc` | residual in the GEMV epilogue, riding the weight tile *(trailer redesigned in Task 2)* |
| `kernels/npu/flm_rope.cc` | full rotary half‑split, `aie::msc`, 0.0e+00 |
| `kernels/npu/flm_attn_{decode,begin,finish}.cc` | GQA=4/core, TSEQ=32, PASS @ S=512/2048 |
| `kernels/npu/flm_asum_prepare.cc` | block sums (used where no norm precedes) |
| `tools/npu/flm/{gemv,normgemv,ffn,attn,rope}_verify.py`, `gemv_bench.py`, `q4nx.py` | references + wiring |

`flm_ffn_gate_up.cc` and `flm_swiglu.cc` are retired by Task 4.

---

## 1. The design

### 1.1 Topology (identical in every phase — this is what makes it legal)

```
shim in  ch0..7 : 8 operand streams, one per PAIR  -> memtile split([0, 20544]) -> 2 cores
shim in  ch8    : 1 broadcast stream               -> 16 core consumers (gemv_bench pattern)
shim out ch0..7 : 8 result streams, one per PAIR   <- memtile join([0, 64])
```

Per core: **2 in / 1 out DMA channels**, at the hard limit of 2 in with one out spare. Every operand a core needs is a DMA input and it gets two — so operands are *packed*, never given their own fifo (this is why `flm_norm_prepare` already carries the norm weight inside the activation buffer).

### 1.2 The universal weight tile — 64‑byte trailer

```
[NROWS*NB bf16 d][NROWS*NB bf16 m][NROWS*K/2 codes][f32 row_base][f32 flags][56 B pad]
= 20480 + 64 = 20544 B   at K=2048, NROWS=16
```

`row_base` is the global output‑row index of the tile's first row. It replaces every per‑core index the kernels would otherwise need — residual indexing, RoPE head identity (`head = row_base/64`, rotate iff `head < 40`), down‑chunk accumulator slot. One mechanism, **+0.31 % weight traffic**, no runtime scalar args, no static cursors for indexing. It supersedes `flm_gemv_residual.cc`'s bf16 residual trailer, which cannot work in a fused layer (the residual is computed on‑device, not at pack time).

### 1.3 The broadcast object — 8448 B, one shape for all phases

```
[ act 2048 bf16 = 4096 ][ aux 2048 bf16 = 4096 ][ cs_q 64 bf16 ][ cs_k 64 bf16 ]  = 8448 B
```

| phase | act half | aux half |
|---|---|---|
| P1 norm+qkv+rope | `x` | `input_layernorm.weight` |
| P2 attention | `q'` (32 heads) | unused |
| P3 o_proj+residual | `attn_out` | `x` |
| P4 norm2+gate+up+swiglu | `h` | `post_attention_layernorm.weight` |
| P5 down chunk c (×4) | `swiglu[2048c : 2048c+2048]` | `h` |

`cs_q` carries cos/sin **pre-multiplied by `head_dim^-0.5 · log2(e)` = 0.125·1.4427**, so attention's `exp2` needs no pre-scale and the host never touches `q'` mid-dispatch. `cs_k` is the plain table. Both from `rope_freqs.weight` (the stored bf16 llama3 divisor), same table for Q and K.

### 1.4 Phase schedule, 16 cores, one dispatch per layer

| # | phase | cores | prologue | rows/core | tiles/core | epilogue |
|---|---|---|---|---|---|---|
| P1 | qkv, K=2048, N=3072 | 16 | `flm_norm_prepare` | 192 | 12 | RoPE per 64-row head |
| P2 | attention, GQA=4, TSEQ=32 | **8** (pairs 0–3) | `flm_attn_begin` | 2 kv groups/pair | KV tiles | `flm_attn_finish` |
| P3 | o_proj, K=2048, N=2048 | 16 | `flm_asum_prepare` | 128 | 8 | `+ aux[row_base+r]` → `h` |
| P4 | gate+up+SwiGLU, N=8192 | 16 | `flm_norm_prepare` | 512 | 64 (32 pairs) | — |
| P5 | down, 4×(K=2048, N=2048) | 16 | `flm_asum_prepare` ×4 | 128 | 32 | `+ aux[row_base+r]` on chunk 3 |
| | | | | | **116** | |

`16 × 116 × 20544 = 38,130,944 B` = one layer's weights + trailers. **5 barriers/layer.**

Two structural notes:
- **48 heads (32 q + 8 k + 8 v) over 16 cores = exactly 3 whole heads per core**, so RoPE in the P1 epilogue never straddles a core. This is why qkv stays *fused* into one N=3072 projection (FLM does the same — its 6144 B split core) and why 16 cores, not 32, is the right count.
- **Attention runs on 8 cores** (pairs 0–3, two kv groups per pair, one per core), using `flm_attn_decode.cc` **unmodified at GQA=4**. Pairs 4–7 sit out one phase. This avoids the cross-core softmax merge entirely and needs no KV duplication — the alternative (16 cores, 2 heads each) forces either a merge or 2× KV DDR reads.

### 1.5 L1 per core — 56,512 of 65,536 B

| buffer | bytes |
|---|---|
| operand fifo, 20544 × depth 2 | 41,088 |
| broadcast fifo, 8448 × depth 1 (acquired once per phase, held — `886a1b6f7`) | 8,448 |
| result fifo, 128 B × depth 2 | 256 |
| `g_asum` bf16[64] | 128 |
| `g_stage` bf16[256] (P1 rope staging / P4 gate stash) | 512 |
| `g_gate` float[16] | 64 |
| `g_acc_down` float[128] | 512 |
| attention state `g_m/g_l/g_acc` (GQA=4) | 1,152 |
| stack (`stack_size=4096`; the IRON 1024 default overflows silently) | 4,096 |
| linker slack | 256 |
| **total** | **56,512 — 9,024 B free** |

Live sets never exceed the static union. Note what makes this fit: **K-chunked `down_proj`**, which keeps the broadcast at 4096 B of activation instead of 16384 B, and **alternating acquires** for gate/up, which keeps the operand object at one tile instead of two. Neither is an optimisation — both are load-bearing.

### 1.6 Shim / memtile

| | used | of |
|---|---|---|
| shim in | 9 (8 operand + 1 broadcast) | 16 |
| shim out | 8 (results; k′/v′ ride P1's result stream) | 16 |
| memtile per pair | S2MM 3, MM2S 3 | 6 / 6 |

7 in / 8 out spare. For contrast, 16 unpaired weight streams + 1 activation = 17 in and is the measured `no ShimNOCTile has sufficient DMA capacity`.

---

## 2. Task list

Each task ends with a numpy-referenced PASS before the next begins. Tasks 0 and 1 are the falsifiers — **do them first, they are ~2 hours total and either kills the design.**

### Task 0 — barrier probe *(falsifier #1, no kernels)*
**New:** `tools/npu/flm/barrier_probe.py`.
Trivial design: 16 cores, one broadcast fifo (8448 B), one result fifo, a no-op kernel. Runtime sequence: `bh.fill(ws); oh.drain(ws2, wait=True)` repeated N ∈ {1, 5, 20, 80} times on a 4 KB workspace BO. Fit `time = a + b·N`.
**Gate:** `b < 93 µs`. Below that, fusing phases beats separate dispatches; the design's whole thesis is `b`.
**Kills:** if `b ≥ 93 µs`, stop — keep one dispatch per *phase* and pursue the 16-layer unroll instead. Report `b`; it is the single number this plan is missing.

### Task 1 — program-memory probe *(falsifier #2)*
**New:** `tools/npu/flm/pmem_probe.py`.
One Worker whose body calls every entry point the fused layer needs (`flm_gemv_q4_1`, `flm_norm_prepare`, `flm_asum_prepare`, `flm_gemv_resid`, `flm_gemv_gate`, `flm_gemv_up_swiglu`, `flm_gemv_acc`, `flm_rope`, `flm_attn_{begin,tile,finish}`) inside a 5-phase `range_` structure. Build only; read `.text`:
```
llvm-readelf -S ~/.npu/cache/<hash>/elfs_main_core_0_2/elfs_main_core_0_2.elf
```
**Gate:** ≤ 14 KB of the 16 KB program module. Known inputs: gemv+asum 2544 B, ffn+asum 6640 B, attn 5712 B.
**Fallback if it fails:** 2 dispatches/layer (attention split out) — 33/token, still ~1.05× FLM.

### Task 2 — the 64-byte tile trailer, and residual from the broadcast
**Modify:** `tools/npu/flm/q4nx.py` (`pack_tile(..., row_base=0)` appends the 64 B trailer; delete `pack_tile_residual`), `kernels/npu/flm_q4_1_tile.h` (`TILE_TRAILER = 64`, `tile_row_base(wtile)` inline accessor).
**Replace:** `kernels/npu/flm_gemv_residual.cc` → `flm_gemv_resid(act_aux, wtile, out)`, epilogue `out[r] += float(aux[int(row_base) + r])` where `aux = act_aux + K`.
**Modify:** `tools/npu/flm/normgemv_verify.py` — `--residual` now packs the residual into the broadcast aux half, not the tile.
**Shapes:** K=2048, N=512, NROWS=16, broadcast 8448 B.
**Verify:** existing `normgemv_verify.py --residual` reference (`norm → GEMV → + resid`), tolerance 1e-2·mean|ref|. Also re-run `gemv_verify.py`, `ffn_verify.py`, `attn_verify.py` — the header changed.
**Also:** a distinct `row_base` per tile is the test that the trailer is read correctly; use non-tiled per-tile data, not `gemv_bench`'s tiled repeat.

### Task 3 — `down_proj` K-chunked ×4
**New:** `kernels/npu/flm_gemv_acc.cc` — `flm_gemv_acc(act_aux, wtile)`: `flm_q4_1_tile` into `g_acc_down[row_base % 128 ... +NROWS] +=`. `kernels/npu/flm_gemv_flush.cc` — `flm_gemv_flush(act_aux, wtile, out)`: last chunk, `out[r] = g_acc_down[...] + aux[row_base+r]`, then zero.
**New:** `tools/npu/flm/repack.py` — chunk-major slice of the planar 5120 B container row: chunk c is `d[64c:64c+64]` (128 B) + `m[64c:64c+64]` (128 B) + `codes[1024c:1024c+1024]` = 1280 B = exactly `TILE_BYTES(K=2048, NROWS=1)`. Arithmetically exact (the GEMV identity is linear in blocks); the container format is unchanged.
**New:** `tools/npu/flm/down_verify.py` — 16 cores, one dispatch, 4 broadcast acquires × 8 tiles.
**Shapes:** 4 × (K=2048, N=2048), 16 rows/tile, 32 tiles/core.
**Verify:** `ffn_verify.py`'s existing `down_proj` numpy reference (1.00e-09 today; expect ~1e-6 from f32 chunk accumulation order).
**Perf gate — state it in marginal microseconds, not GB/s.** From the repo's fit `t = 92.9 + 17.547·MB`, a standalone 10.49 MB dispatch cannot exceed 37.9 GB/s no matter how good the kernel. Gate: **slope time ≤ 200 µs** (ideal 184 µs) against today's ~277 µs marginal. Worth ~0.86 ms/token.

### Task 4 — gate+up by alternating acquires, 16 rows
**New:** `kernels/npu/flm_gemv_gate.cc` — `flm_q4_1_tile` into `g_gate[16]`.
**New:** `kernels/npu/flm_gemv_up_swiglu.cc` — tile into `u[16]`, then `out[r] = silu(g_gate[r])·u[r]`. Use `flm_ffn_gate_up.cc`'s existing 32-lane `aie::exp2` body (3.40e-04, at the NLF floor) — the scalar polynomial in `tools/npu/silu_mul_bf16.cc` is 10× more accurate and is the upgrade path, not the first build.
**Delete:** `flm_ffn_gate_up.cc`, `flm_swiglu.cc`.
**New:** `tools/npu/flm/ffn_alt.py`; weight stream reordered offline to `[gate t0][up t0][gate t1][up t1]…` — same bytes, no extra channel, no extra dispatch, and the 8192-wide intermediate is still never materialised.
**Shapes:** K=2048, N=8192, 16 rows/tile, 64 acquires/core.
**Verify:** `ffn_fused.py`'s existing float64 reference; error at the exp2 floor (~3.4e-04).
**Perf gate:** slope time ≤ 400 µs on 21.0 MB (ideal 369 µs) vs today's ~414 µs. Worth ~0.74 ms/token. *This is the most falsifiable claim here: if it lands at ~414 µs the 41.4 GB/s figure was the fused kernel's arithmetic, not the row count, and Task 4 should be reverted.*

### Task 5 — qkv fused N=3072 + RoPE epilogue
**New:** `kernels/npu/flm_gemv_qkv.cc` — `flm_q4_1_tile` into `g_stage[(row_base % 192)…]` as bf16; when a 64-row head completes (`row_base % 64 == 48`), call `flm_rope` in place with `cs_q` if `row_base/64 < 32`, `cs_k` if `< 40`, skip if v.
**New:** `kernels/npu/flm_qkv_emit.cc` — writes one 64-bf16 head to the result fifo.
**New:** `tools/npu/flm/qkv_verify.py`.
**Shapes:** K=2048, N=3072, 192 rows/core, 12 tiles, result object 128 B.
**Verify:** numpy `norm → [W_q;W_k;W_v]·y → rope(q,k)` using `rope_verify.py`'s existing `rope_freqs`-driven table. Check q′, k′, v separately.
**Open, and it must stay a `-D` flag:** `-DROPE_INTERLEAVED` vs half-split. `rope_freqs.weight` is bit-exactly llama.cpp's `ROPE_FREQS`, which means the converter is llama.cpp-derived and the container's q/k rows *may* carry `LlamaModel.permute`. Guessing wrong gives silently plausible wrong output. Build both; neither can be settled until §1.3 is.

### Task 6 — attention as a phase, 8 cores
**Modify:** nothing in the attention kernels.
**New:** `tools/npu/flm/attn_phase.py` — pairs 0–3, KV on the operand fifo, `q'` on the broadcast (already scaled by `cs_q`), k′/v′ appended by draining P1's k/v output into the KV cache BO with a strided BD (K channel-major `[64][32]` is a stride-32 scatter of 64 bf16; V is contiguous).
**Shapes:** GQA=4, HEAD=64, TSEQ=32; KV tile 8192 B padded into the 20544 B operand object (25 % waste on KV, +2.8 % of layer bytes at S=2048, 0.7 % at S=512).
**Verify:** `attn_verify.py`'s float64 reference at S=512 and S=2048, plus one S that is not a multiple of 32 to exercise the pad correction. Tolerance at the exp2 NLF floor (3.54 % mean, 5.86 % max) — **not** below it.
**Pad correction** (`flm_attn_finish.cc`, 3 lines): padded positions get K=0, V=0, so they add nothing to `g_acc` and exactly `npad·exp2(−m)` to `g_l`:
```c
g_l[h] -= npad * exp2f_hw(-g_m[h]);
```
`npad` arrives as **f32** in the broadcast aux — bf16 represents integers exactly only to 256 and `npad` reaches 2047.

### Task 7 — one layer, one dispatch
**New:** `tools/npu/flm/layer_verify.py` — all 5 phases, 5 barriers, real layer-0 weights.
**Verify:** float64 numpy reference for `x_out = h + W_down·SwiGLU(...)`, error at the exp2 floor.
**Also measure:** per-phase duty cycle and the realised barrier count against Task 0's `b`.

### Task 8 — 16 layers
**New:** `tools/npu/flm/layer_bench.py`. First real tok/s, against `flm run`'s 59.86.
Start at 17 dispatches (one per layer + lm_head, ~5 BOs each — well under the 20-BO firmware wall). The 16× unroll to 2 commands needs BD reuse (`dma_free_task`/`dma_await_task`) + `DDR_PATCH` and `full_elf=True`; it is worth a further ~1.4 ms/token and is a separate piece of work.

---

## 3. Open questions that need a measurement, not a decision

1. **In-sequence `drain(wait=True)` → `fill` cost.** Task 0. The design's only real unknown. 80 barriers/token; the whole plan is a bet that one is cheaper than one dispatch.
2. **Core program memory at 11 entry points + 5-phase control flow.** Task 1.
3. **Does a strided drain into the KV cache work** (stride-32 scatter of one position's K, channel-major)? Task 6. Fallback: drain k′/v′ to a 32 KB staging BO and let the host `memcpy` between tokens — ~2 MB/s at 60 tok/s, but it puts the host back in the loop and blocks the 16-layer unroll.
4. **Variable S.** `fill(offset_parameter=…)` exists but is untried here. Default: bucket at S = 512/1024/2048/4096 and recompile (xclbins are JIT-cached); correct the padding in `flm_attn_finish`.
5. **Does gate/up's 41.4 GB/s come from the row count or the fused arithmetic?** Task 4 answers it in one run.
6. **Whether `strides=[…,0,…]` (a 0-stride DMA repeat) is legal** — would let a pair read the same KV tile twice and put all 16 cores on attention. Not needed; only relevant if attention becomes the bottleneck at long context.

---

## 4. What this plan does not solve

- **The container row mapping (`§1.3`) and the RoPE pairing convention.** FLM's codes are not a quantization of the published checkpoint, and which `(output row, k-block)` each of the 256 slots in a 5120 B row maps to is unknown. **Every verification in Tasks 2–8 is self-consistency against a numpy reference built from the same reading of `model.q4nx`.** A wrong row mapping passes all of them. Only end-to-end logits against `flm run` is a real check, and it is blocked.

  **Measured 2026-07-31 (`tools/npu/flm/ground_truth.py`), so this is no longer a suspicion.** This was settleable after all — Meta's actual checkpoint (`original/consolidated.00.pth`) is on this machine, no `flm run` needed. The container's *unquantized* tensors are **100% bit-exact** against Llama-3.2-1B-**Instruct**, confirming the reader, the planar split, the naming and the model. The *quantized* blocks are confirmed **not** the checkpoint's under any arrangement: matched against all 524,288 ground-truth blocks with a scale- and nibble-order-invariant fingerprint, the container scores **0.99487**, against a **0.99503** look-alike floor (the score a block earns when its true match is masked out of the pool) and **0.99713** when the true match is present. Separately, the Frobenius norm is inflated **1.12–1.19x, differing per tensor** — orthogonal would be exactly 1.0000 — so a real scaling is present; the value distribution is ~1.14x ground truth at *every* quantile (1.097–1.157), which is closer to one scalar per tensor than to a broad per-channel vector, and a scaled rotation is not excluded. Also measured: **FLM's q4_1 is a search fit on a grid ~6.6% wider than min/max**, not llama.cpp's, so never compare container `d` against `(max-min)/15`. Tasks 2–8 stay self-consistent-only and `-DROPE_INTERLEAVED` stays open; the way forward is to identify that transform, not to wait for `flm run`.
- **The `exp2` NLF accuracy floor** (5.86 % max / 3.54 % mean, ~18× worse than bf16). It caps attention and SwiGLU. The fix exists unused in `tools/npu/softmax_bf16.cc` (range reduction + degree-2 polynomial, ~5 ULP, no NLF, no LUT). It is an accuracy project, and it must precede any KV-quantization ablation, which would otherwise read its own noise floor.
- **Prefill.** Decode only. `attn.xclbin`'s 32-query-position template is a different design.
- **The 16-layer single command.** Task 8 stops at 17 dispatches. BD reuse + `DDR_PATCH` is the remaining ~1.4 ms.
- **lm_head.** Unchanged — already 55.1 GB/s / 164.2 MB / 4.99e-07 in one dispatch.
- **32 cores, neighbour memory, memtile 4-D reshape.** All three are levers FLM uses and this plan does not. None is needed to fit; none has been shown to buy a row at any shape in this model.

---

## 5. How far this gets

From the branch's own fitted model (`82756fb1e`, `t_µs = 92.9 + 17.547·MB`, R² = 0.99997) on 772.3 MB/token:

| configuration | ms | tok/s | vs FLM |
|---|---|---|---|
| today, 5 dispatches/layer | 21.07 | 47.5 | 0.79× |
| **this plan, 17 dispatches, barrier = 5 µs, S≈0** | **15.53** | **64.4** | **1.08×** |
| this plan, barrier = 20 µs, S≈0 | 16.73 | 59.8 | 1.00× |
| this plan, S=2048 (KV +67 MB, 25 % pad) | +1.5 ms | ~57 | 0.95× |
| follow-on: 2 dispatches | 13.74 | 72.8 | 1.22× |

So: **roughly 1.0–1.1× FLM at short context, contingent on Task 0's barrier number**, with the 16-layer unroll worth another ~0.15×. Tasks 3 and 4 are worth ~1.6 ms/token *on their own* and are independently valuable even if the fused layer is abandoned — build them regardless of what Task 0 says.
