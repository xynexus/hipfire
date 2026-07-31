# FLM `layer.xclbin` — dataflow and host-side dispatch

Phase 1 deliverable for `~/flm-re-fe-mutate-goal.md`. Covers FastFlowLM's
decode path on AIE2P (Strix Halo NPU). Running log with the working:
`flm-refe-log.md`.

**Every number here carries its source.** Three kinds appear:

- **measured** — from timing or instrumenting FLM on this machine, method named.
- **derived** — arithmetic on measured values or on published model config.
- **open** — stated because it matters, explicitly not established.

**Provenance.** This document records *facts* about observable behaviour —
buffer sizes, submission counts, register addresses, bandwidths. Facts are not
protected expression regardless of how they were learned. It deliberately does
not reproduce FLM's schedules, register allocation, or the particular
organisation of its computation. See the provenance section of the goal doc
before extending it; if a clean-room implementation track is wanted, this file
is intended to be safe input to it and the running log is not.

Model used throughout: **Llama-3.2-1B**, 16 layers, hidden 2048, intermediate
8192, 32 heads / 8 KV heads, head_dim 64, `max_position_embeddings` 131072.

---

## 1. There are two dispatch regimes, not one

**measured** — `LD_PRELOAD` interposer on `ioctl()` counting
`DRM_IOCTL_AMDXDNA_EXEC_CMD` (`tools/npu/flm/npu_ioctl_count.c`), logging each
submission's `(hwctx, type, cmd_count, arg_count)`.

### Prefill — per-linear

145 submissions for a one-chunk prompt, decomposing exactly:

| context | args | count | per layer |
|---|---|---|---|
| 2 | 3 | 112 | **7.0** — the 7 linears (q, k, v, o, gate, up, down) |
| 3 | 5 | 16 | 1.0 |
| 4 | 3 | 16 | 1.0 |
| 1 | 4 | 1 | once (lm_head) |

`9 per layer x 16 layers + 1 = 145`. The 7.0 lands on the architecture with no
remainder.

~~**open** — intra-layer *ordering*.~~ **CLOSED 2026-07-31 — the order and every
context's role are settled**, by dumping each submission's argument BO sizes
(`NPU_ARGS_OUT` / `NPU_ARGS_MATCH`, same interposer). The layer starts at ctx3:

```
ctx3(5)  ctx2(3) ctx2(3) ctx2(3)  ctx4(3)  ctx2(3) ctx2(3) ctx2(3) ctx2(3)
dequant   q       k       v        attn     o       gate    up      down
```

The buffer sizes identify each one with no remainder (Llama-3.2-1B, hidden 2048,
kv dim 512, intermediate 8192, bf16):

| context | kernel | arg0 | arg1 | arg2 |
|---|---|---|---|---|
| 3 | `dequant.xclbin` | **37 MiB q4_1 weights, a different handle each layer** | 20 MiB | 32 MiB (+ 32, 32 as arg3/4) |
| 2 | `mm.xclbin` | weight slice | **out** | **in** |
| 4 | `attn.xclbin` | **256 MiB KV cache, a different handle each layer** | Q (14 MiB) | O (14 MiB) |

- **ctx3 is dequant, and it begins the layer.** Its arg0 is the only per-layer
  weight buffer, 38,797,312 B — Llama-3.2-1B's 60.9 M per-layer parameters at
  q4_1's 5.00 bpw. Its four outputs are **20 + 32 + 32 + 32 = 116 MiB**, which is
  exactly those parameters at bf16 (121.8 MB), split the obvious way:
  20 MiB = q+k+v+o (8+2+2+8), and one 32 MiB buffer each for gate, up, down
  (2048 x 8192 x 2 B). **Prefill dequantizes the whole layer to bf16 up front and
  runs bf16 GEMMs**; the q4_1 dequant chain that dominates the decode kernel is
  not on the prefill path at all.
- **ctx2 is `(weights, out, in)`.** Confirmed on `down_proj`, whose 8192-wide
  input and 2048-wide output are unambiguous: arg1 is the 14 MiB (2048-dim)
  buffer, arg2 the 56 MiB (8192-dim) one. Chaining then reads cleanly —
  q/k/v all take the layer input, o takes attention's output, down takes the
  gate/up product.
- **ctx4 is attention, and takes exactly three buffers**: the layer's 256 MiB KV
  cache (same size and per-layer handle as the decode path's), q_proj's output,
  and o_proj's input.
- **k_proj's and v_proj's outputs (4 MiB each) are passed to nothing.** They are
  not arguments to the attention dispatch and not to any later one. So the KV
  cache fill — and, on this path, RoPE — happens **off the NPU**, between
  dispatches, which is consistent with `fill_kv_cache(buffer<bfloat16_t>&, ...)`
  and `_rope(buffer<bfloat16_t>&, int)` being host-side symbols in
  `libllama_npu.so`.

Activation buffer sizes pin the chunk: 14,680,064 B / 2 B / 2048 = **3584 tokens**,
and 58,720,256 / 2 / 8192 gives the same 3584. (The 4 MiB k/v buffers are that
same 3584 x 512 x 2 B = 3.5 MiB rounded to the 1 MiB allocation granularity.)

### Decode — one command for the whole model

Every decode submission is on **context 1**, strictly alternating:

```
ctx1 args=50   ctx1 args=4   ctx1 args=50   ctx1 args=4   ...
```

The 50-argument command executes the **entire 16-layer body**. The 4-argument
one is lm_head (same shape as the single ctx1 submission ending prefill).

### Submissions per decoded token

**measured**, four points on llama, increments of exactly 400 per 200 tokens:

| model | layers | EXEC_CMD / token |
|---|---|---|
| Llama-3.2-1B | 16 | **2.00** |
| Qwen3.6-35B-A3B | 40 | **3.00** (2 points) |

**derived** — dispatch count tracks distinct kernel *phases*, not depth. 2.5x
the layers buys one extra command. Per-layer dispatch would predict 16 and 40.

**This is a 72x difference in dispatch granularity between prefill (145) and
decode (2) for the same model.** Prefill is compute-bound, so per-linear
dispatch costs little and buys scheduling freedom; decode is latency- and
bandwidth-bound, so it collapses to one command.

**derived** — this identifies the xclbins: `mm.xclbin` and `attn.xclbin` serve
the per-linear prefill contexts 2/3/4; **`layer.xclbin` is the decode path**,
and it is one fused *model*, not one fused *layer*.

---

## 2. The decode command's buffer contract

**measured** — recorded `CREATE_BO` handle -> (size, type), then dumped the
`args` array of a 50-argument submission. `args` is a user pointer to
`arg_count` **u32** BO handles (driver: `amdxdna_ctx.c:589`).

```
50 args = 16 x (weights, workspace, KV) + 2
```

| buffer | size | count | notes |
|---|---|---|---|
| weights | 38,797,312 B = **37 MiB** | 16 | one per layer |
| workspace | 1,048,576 B = **1 MiB** | 16 | one per layer |
| KV cache | 268,435,456 B = **256 MiB** | 16 | one per layer |
| activations | 1 MiB | 2 | in, out |

**KV size is exact, not approximate:**

```
8 kv_heads x 64 head_dim x 2 (K and V) x 2 bytes x 131072 max_position_embeddings
  = 268,435,456
```

**derived** — each layer's KV is preallocated for the **full 131,072-token
context** regardless of actual sequence length.

**derived** — allocation granularity is **1 MiB**. Computed per-layer weights
are 38,010,880 B = 36.25 MiB against a 37 MiB buffer; every buffer in the list
is a whole number of MiB. The 2.07% gap is rounding, not a format difference,
and it corroborates rather than undermines the 5.00 bpw figure.

**Footprint and bandwidth are decoupled.** One command references
`16 x (37 + 1 + 256) + 2 = 4,706 MiB = 4.60 GiB`, cross-checked against ~5.3 GB
of `/dev/accel` BO mappings measured independently by scanning the live process.
But only **772.3 MB** streams per token, because only the used KV prefix is
touched.

---

## 3. Weight format and per-token traffic

**derived**, from the model's safetensors manifest:

| | value |
|---|---|
| container `model.q4nx` | 1297.8 MB |
| streamed set (113 I8 tensors) | **772.3 MB** |
| non-streamed | `model.embed_tokens.weight` BF16 [128256, 2048] = 525.3 MB — a per-token *gather* of one 4 KB row |
| bits/weight over 1.236 B non-embedding params | **5.00** |

Decomposition closes to 0.01%: `16 layers x 38.0 MB + 164.2 MB lm_head =
772.4 MB` against the manifest's 772.3 MB.

**Quoting the container size overstates decode bandwidth by 1.7x.**

Weight format is **q4_1** (asymmetric, scale + min) — from
`Dequant::generate_dequant_q4_1_seq` in the symbol table.

---

## 4. Measured throughput

**measured** — `benchmarks/flm_baseline/flm_bench.py`, medians of 3,
`--pmode performance`, AIE clock 1.8 GHz.

Re-taken 2026-07-31 on a **verified-quiet machine** (no `flm` processes, nothing
holding an `accel0` fd). These supersede the earlier contended run.

| model | workload | prompt tok | tok/s | achieved BW | % of ~55 GB/s fabric |
|---|---|---|---|---|---|
| Llama-3.2-1B | decode | 61 | **59.86** | **46.2 GB/s** | 84% |
| Qwen3.6-35B-A3B | decode | 34 | **13.41** | **36.4 GB/s** | 66% |

Bandwidth = tok/s x streamed bytes/token (section 3).

**The contention caveat resolved the opposite way to expectation.** The earlier
run (61.07 / 47.2, 13.54) was taken with a resident second NPU client. Quiet, both
models came out *slower* — llama -2.0%, MoE -1.0%. Contention was not inflating
the figures. Mechanism unknown; plausibly the resident `flm serve` kept llama's
weights in page cache, so that run streamed warm and this one cold. Untested.
The quiet figure is also the better-corroborated one: against the independently
recorded historical 60.1 tok/s / 46.4 GB/s it agrees to 0.4%, where the contended
run was off by 1.6-1.7%.

### Prefill — baseline taken 2026-07-31, at a stated length

Previously recorded here as "not quotable" because two runs used prompts of very
different length (llama 7075 vs 3577 tokens). **The diagnosis was wrong.**
`flm_bench.py` *does* pin its prompt: `make_prompt_file` is deterministic for a
given `--prefill-tokens`, and regenerating it reproduces the checked-in
`results/prefill_prompt.txt` byte for byte (md5 `9e73e8...` at the default 2048).
The two runs simply used **different `--prefill-tokens`**. The harness now also
records that setting in its CSV so a run is self-describing.

Prefill throughput remains strongly prompt-length dependent (~121 tok/s at 52
prompt tokens against 1623 at 1828 for the same model, because a short prompt is
dominated by the fixed ~430 ms TTFT), so **the length below is part of the
number**:

| model | workload | `--prefill-tokens` | prompt tok | tok/s |
|---|---|---|---|---|
| Llama-3.2-1B | prefill | 2048 | **3577** | **2064.8** |
| Qwen3.6-35B-A3B | prefill | 2048 | **4127** | **178.2** |

Medians of 3; spread was under 1.6% on both. Prompt-token counts differ between
the models because their tokenizers do, not because the input did.

**Conditions: `flm serve llama3.2:1b` was resident, so this is a contended run**,
not the verified-quiet condition of the decode table above. The same run's decode
figures — llama **61.31**, MoE **13.55** — reproduce the earlier *contended*
numbers (61.07 / 13.54) to 0.4% and 0.1% rather than the quiet ones, which both
confirms the condition and bounds its effect at ~2%.

**derived** — the MoE prefills **11.6x slower** than llama at a comparable prompt
length, where active-parameter scaling predicts ~2.4x. Sparsity is a decode-only
benefit: a long prompt activates essentially all 256 experts.

**derived** — at 84% of fabric, llama decode has little bandwidth headroom left;
a decode win there must come from moving fewer bytes. The MoE at 66% is *not*
bandwidth-bound, so a third of its achievable rate is going elsewhere.

---

## 5. Host/driver interface details

**measured**, same interposer:

- **`WAIT_CMD` is never called. `SYNC_BO` is never called.** Completion is not
  observed through the driver's wait ioctl; most plausibly by polling the
  command BO's status word. A reproduction should not assume `WAIT_CMD`.
- **~17 `CREATE_BO` per decoded token.** Surprising churn for a steady-state
  decode loop; looks like an avoidable cost rather than something to copy.
- ~27 arguments per submission averaged over a decode run.

---

## 6. The embedded transaction ladder

**measured** — `tools/npu/flm/txn_scan.py`, which locates transaction binaries
by structural validation (no magic number exists: the op walk must land exactly
on `txn_size` *and* produce exactly `num_ops`).

`liblm_head.so` and `libllama_npu.so` each embed the same eight, byte-identical:

| cols | ops | direction | DDR bytes |
|---|---|---|---|
| 1 | 31 / 35 | egress / ingress | 524,288 |
| 2 | 61 / 69 | egress / ingress | 1,048,576 |
| 4 | 121 / 137 | egress / ingress | 2,097,152 |
| 8 | 241 / 273 | egress / ingress | 4,194,304 |

Four direction-pairs, parameterised by column count. **Every one touches only
row 0 (shim) and row 1 (memtile); none touches a compute core.** They are a
generic DDR<->memtile staging ladder.

Per column, each programs 22 `write32`, 2 `maskwrite32`, 4 buffer descriptors
and 2 `DDR_PATCH`, with all 8 columns receiving an identical program. Register
targets, decoded against aie-rt's AIE2P definitions:

| tile | offsets | registers |
|---|---|---|
| shim (row 0) | `0x1d200`-`0x1d20c` | `DMA_S2MM_0/1_CTRL`, `_TASK_QUEUE` |
| shim | `0x3f008`/`0x3f010`/`0x3f014` | `STREAM_SWITCH_MASTER_CONFIG_SOUTH0/2/3` |
| shim | `0x3f100`/`0x3f138`/`0x3f13c` | `SLAVE_CONFIG_TILE_CTRL`/`NORTH_0`/`NORTH_1` |
| shim | `0x1f004` (maskwrite) | `DEMUX_CONFIG` |
| memtile (row 1) | `0xa0630`-`0xa063c` | `DMA_MM2S_0/1_CTRL`, `_START_QUEUE` |
| memtile | `0xb001c`/`0xb0020` | `STREAM_SWITCH_MASTER_CONFIG_SOUTH0/1` |
| memtile | `0xb0100`/`0xb0104` | `SLAVE_CONFIG_DMA_0/1` |

Buffer descriptors: shim BD#0 (`0x1d000`), BD#1 (`0x1d020`), memtile BD#0
(`0xa0000`), BD#24 (`0xa0300`). Each is a **flat linear 256 KB transfer** —
`BUFFER_LENGTH` = `0x10000` 32-bit words, iteration field zero, D0/D1/D2
stepsizes zero. Word 7 carries only the valid bit (`0x02000000` shim,
`0x80000000` memtile), *not* an iteration count.

`DDR_PATCH` ops rewrite word 1 (the address word) of the two shim BDs, at a
uniform 256 KB stride across 16 patches. The v1.0 `DDR_PATCH` is aie-rt's
24-byte `patch_op_opt_t`, not the 44-byte `patch_op_t`:

```
+0  u8 Op, u8 padding[3]      XAie_OpHdr_opt
+4  u32 Size (= 24)           XAie_CustomOpHdr_opt
+8  u32 regaddr           \
+12 u8 argidx, u8 pad[3]   >  patch_op_opt_t
+16 u64 argplus           /
```

`regaddr` is 32-bit here and there is no `action` field.

---

## 6b. Core inventory, roles, and intra-array dataflow

**measured** — `cdo.py` on `layer.xclbin` extracts 27 cores with program memory.
Program sizes group exactly onto the roles:

| size | count | tiles | role |
|---|---|---|---|
| 9236 B | 16 | cols 0,1,6,7 x rows 2-5 | GEMM (`vmac.f:264 vextbcst.16:256 vunpack:64`) |
| 4580 B | 4 | cols 3,4 rows 2,4 | shuffle (`vshuffle:91`) |
| 6852 B | 4 | cols 3,4 rows 3,5 | elementwise (`vmul.f:86 vadd.f:73`) |
| 6388 B | 1 | col 2 row 2 | **RMSNorm** (confirmed below) |
| 1812 B | 1 | col 2 row 3 | norm / residual |
| 2036 B | 1 | col 5 row 2 | **SwiGLU** (confirmed below) |

**col 5 row 2 evaluates SiLU by clamped table lookup**, as part of the SwiGLU it
computes (the 2:1 I/O ratio establishing `silu(gate)*up` is in the next section).
Its one hardware loop (`lc=0x20`) does: `vfloor.s32.bf16` with shift 6 ->
`vmax_lt.32` / `vmin_ge.32` clamp to `[-0x200, 0x1ff]` -> add table base ->
`vldb.4x64.lo/hi` gather -> two `vshuffle`, `vmac.f`, `vmul.f` to interpolate ->
`vst.conv.bf16.fp32`.

**derived** — step `1/64`, domain **[-8.0, +7.984375]**, **1024 entries**. A
table spanning exactly [-8,8] is the signature of a sigmoid-family activation;
`config.json` gives `hidden_act: silu`.

The *mechanism* is read directly off the instructions. That the function is
specifically SiLU is inference from the config plus the absence of any other
activation core.

FLM clamps the *index*, so inputs beyond +8 return the endpoint rather than
SiLU's asymptotic identity — an approximation it accepts, harmless for
post-RMSNorm activations, and one a reproduction may knowingly make or avoid.

### Core roles confirmed from I/O geometry

**measured** — input/output BD chain sizes per core:

| tile | role | in (ch0 \| ch1) | out | basis |
|---|---|---|---|---|
| 16 GEMM | **GEMM** | 512/512 \| 5120/5120 | 132/132 | broadcast activations + private weights |
| (2,2) | **RMSNorm** | 4096/4096 \| 4096 | **4100** | 4096 B = hidden_size exactly; **4100 B out** = 1024 words of vector + **one extra word** = the reciprocal norm |
| (5,2) | **SwiGLU** | **2048/2048** \| — | **1024/1024** | exactly **2:1** — two inputs, one output; with the SiLU table lookup in its code, this is `silu(gate)*up` |
| (2,3) | norm/residual (inferred) | 6144 \| 128 | 4096 | not decomposed |
| (3,2)(4,2)(3,4)(4,4) | shuffle | 1024/1024 \| 4096/4096 | **none** | feeds the core above |
| (3,3)(4,3)(3,5)(4,5) | elementwise | — \| 4096/4096 | 1024/1024 | fed from the core below |

### Vertical pairing: a third of the dataflow does not use DMA

**measured** — all four DMA channels (S2MM 0/1, MM2S 0/1) on every core:

| tile | role | MM2S0 | MM2S1 |
|---|---|---|---|
| GEMM **rows 2, 4** | GEMM | **on** | -- |
| GEMM **rows 3, 5** | GEMM | **--** | **--** |
| (3,2)(4,2)(3,4)(4,4) | shuffle | **--** | **--** |
| (3,3)(4,3)(3,5)(4,5) | elementwise | on | -- |
| (2,2) | RMSNorm | **on** | **on** |
| (5,2) | SwiGLU | on | -- |

**derived** — two independent vertical pairings, both handing off without DMA:

- **GEMM cores work in pairs.** Rows 3 and 5 have *both* MM2S channels disabled;
  only rows 2 and 4 emit. So row 3 feeds row 2 and row 5 feeds row 4 —
  **8 result streams, not 16**.

  **RESOLVED 2026-07-31 — it is an N-split: the pair CONCATENATES.** Two
  independent sources agree.

  *Code.* Diffing the pair's programs with addresses stripped leaves 64 differing
  lines, all of them register housekeeping — stack spill/reload under a different
  regalloc, pointer and index setup, nop padding. **Zero differences in `vmac`,
  `vadd`, `vsub`, `vmul`, `vconv` or any `vst`**; the vector store sequence is
  byte-identical. A K-split requires the upper core to read the lower's partials
  and add them, and that code exists in neither program.

  *Buffer descriptors.* The pair's operand pointers differ only in memory window
  — upper `p0=0x73c00`, `p3=0x75400`; lower `p0=0x43c00`, `p3=0x45400`. Per
  `AIETargetModel.h` (NPU model) internal = East = `0x70000` and South =
  `0x40000`, so the lower writes into its south neighbour = the upper. Those
  offsets are exactly the upper's two output BDs (module `0x3C1C` and `0x541C`,
  **132 B each**), and the lower core has **no output BD at all**. Two
  equal-sized buffers with no reduction is concatenation; a sum would need one.

  Consequence for reproduction: **two program variants per pair, differing only
  in the output pointer base** (own vs south). The compute body is identical.
  And "8 result streams" means 8 streams each carrying two concatenated
  N-groups — not 8 reduced results.
- **shuffle -> elementwise.** Rows 2,4 (shuffle) take input and emit nothing;
  rows 3,5 (elementwise) sit directly above and do emit.

**This matters for reproduction: a substantial part of this kernel's
communication is invisible in the DMA graph, because it does not use DMA.**
Anything reconstructed from DMA connectivity alone would silently omit it.

**measured — the mechanism is SHARED NEIGHBOUR MEMORY, not cascade.**

The 16 GEMM cores carry exactly **two** program binaries, 8 each, splitting along
the emit/silent line:

```
8 x md5 fa1b2dc9ed3d5a50   <- rows 2,4  (emit)
8 x md5 f227d3d030724331   <- rows 3,5  (silent)
```

They differ in **178 of 9236 bytes (1.93%)**, all within `0x200e8-0x201ef` — the
**prologue**. The compute loop is byte-identical, and neither contains any
cascade instruction. The whole difference is buffer addressing:

| row 2 (emit) | offset | row 3 (silent) | offset |
|---|---|---|---|
| `0x75400` | 0x5400 | `0x45400` | 0x5400 |
| `0x73c00` | 0x3c00 | `0x43c00` | 0x3c00 |

Row 2 addresses only `0x7xxxx` (its own data-memory module). Row 3 uses
`0x7xxxx` **plus two buffers in `0x4xxxx`** — a neighbour's module — at exactly
the offsets row 2 reads locally.

**Row 3 stores its results straight into the neighbouring tile's data memory;
row 2 reads them as ordinary local loads.** No DMA, no cascade, no
sender/receiver instructions — just a pointer into a different window, fixed at
compile time.

**Consequence for reproduction:** this needs **two core programs per pair**,
differing only in output addressing — not one program plus configuration. And
the transfer costs nothing in the instruction stream. Any reproduction routing
the equivalent traffic through DMA or objectfifos pays for something FLM gets
free.

(2,2) is the only core using **both** MM2S channels, consistent with a norm
feeding two consumers.

### The GEMM core's MAC operand chain

**measured** — read off the hardware loop (`lc=0x2`, `ls=0x260`, `le=0x1850`):

```
vlda  x8,  [p0], #0x40           packed int4 weights
vunpack x9, wl8, unpacksign0     unpack nibbles
vups.4x dm2, x9, s0, upssign0    widen into accumulator
vadd    dm1, dm2, dm0, r0        zero-point / min term (q4_1 is asymmetric)
vconv.bf16.fp32 x3, cml1         accumulator -> bf16 weights
vldb  x11, [p1], #0x40           activations
vextbcst.16 x10, x11, #0x1d      broadcast ONE activation element
vmac.f dm1, dm1, x3, x10, r4     accumulate
```

The multiply is **bf16 dequantized weight x broadcast activation scalar** — the
M=1 GEMV shape. This is the dequant chain phase 3a exists to remove.

Pointer cadence per outer iteration: `p0` advances (weights), `p1` 512 B rewound
(activations, reused across N), `p4` advances with N (scales), `p5` 512 B
rewound — 8 x `vldb ... #0x40` then `paddb [p5], #-0x200`.

**measured** — L1 addresses, read from the immediates in the prologue:

| pointer | L1 address | size | role |
|---|---|---|---|
| `p0` | function arg | 2048 B | packed int4 weights |
| `p1` | **0x72800** | 512 B | activations |
| `p4` | **0x74000** | 512 B | per-group scales |
| `p5` | **0x7c000** | 512 B | unresolved |
| — | 0x73ca3 / 0x73ca4 | 1 B | scalar constants (`lda.s8`) |

All operand buffers sit in the same 64 KB data-memory module (core stack is at
`0x70000`), at module offsets `0x2800`, `0x4000`, `0xc000`.

### Core-tile DMA buffer descriptors

**measured** — `tools/npu/flm/cdo_dma.py`, decoding the CDO's BD writes against
aie-rt's AIE2P field definitions. Tile (0,2), a GEMM core:

| BD | core view | length | locks |
|---|---|---|---|
| BD0 | 0x78000 | 512 B | acq 127, rel_id 1, next=BD1 |
| BD1 | **0x7c000** | 512 B | acq 127, rel_id 1, next=BD0 |
| BD2 | **0x72800** | 5120 B | acq_id 2, rel_id 3, next=BD3 |
| BD3 | **0x74000** | 5120 B | acq_id 2, rel_id 3, next=BD2 |
| BD4 | 0x73c1c | 132 B | acq_id 7, rel_id 4, next=BD5 |
| BD5 | 0x7541c | 132 B | acq_id 8, rel_id 5, next=BD4 |

`BASE_ADDRESS` and `BUFFER_LENGTH` are in **32-bit words** (both 14-bit fields,
spanning the 64 KB module at 4 B/unit); the core addresses its own module
through a `0x70000` window.

**derived** — the operand pointers map onto BDs exactly: `p1` -> BD2,
`p4` -> BD3, **`p5` -> BD1**. Three of three, which is what confirms the unit.

**`p5` is DMA-fed**, buffer 512 B — matching exactly the 8 x 64 B it reads before
rewinding — double-buffered with BD0 as a ping-pong pair on its own lock
(`rel_id 1`), distinct from the activation stream's lock 3. This rules out `p5`
being locally derived, a constant table, or an alias of the activation buffer.

### Stream-switch connectivity, and where `p5` comes from

**measured** — CDO stream-switch config, port indices resolved against aie-rt's
`Aie2PTileStrmSwSlavePortMap` (`xaie2pgbl_reginit.c:721`). For core (0,2):

```
MASTER DMA0 <- EAST0    (circuit)      DMA0 -> S2MM ch0 -> BD0/BD1 -> p5
MASTER DMA1 <- SOUTH2   (circuit)      DMA1 -> S2MM ch1 -> BD2/BD3 -> p1, p4
                                       MM2S ch0 -> BD4/BD5 -> output
```

**`p5` is fed from the EAST neighbour tile — a core-to-core stream, not memory.**

DMA0 source across rows 2-3: columns **0-1 pull from EAST**, columns **6-7 pull
from WEST**, middle columns pull from **SOUTH** (memtile). DMA1 is SOUTH almost
everywhere.

**derived** — data enters from the memtile in the middle columns and propagates
**horizontally outward** through the stream switches to the edge GEMM columns,
while each column separately pulls its own stream from the memtile below. One
memtile read serves several columns rather than every core pulling its own copy.
This is a significant part of how FLM sustains 86% of fabric bandwidth on decode
(section 4), and it is a structure a reproduction has to match.

### The broadcast path, hop by hop

**measured** — `cdo_dma.py --graph`, **294 enabled routes** array-wide. The full
path feeding `p5`:

```
memtile(1,1) MM2S ch4  ->  NORTH4
    -> core(1,2) SOUTH4 slave
         |-> core(1,2) DMA0         -> its own BD0/BD1 -> its p5
         |-> core(1,2) WEST0 master -> core(0,2) EAST0 -> DMA0 -> BD0/BD1 -> p5
         `-> core(1,2) EAST2 master -> core(2,2)
```

Every hop is a decoded register. **One memtile MM2S channel feeds at least three
columns** by circuit-switched forwarding through the intermediate tile's switch —
a genuine 1-to-N broadcast, not N separate memtile reads.

**The structural result:**

| | source | buffer | shared? |
|---|---|---|---|
| `p5` (DMA0) | memtile (1,1), horizontal chain | 512 B | **broadcast across columns** |
| `p1`, `p4` (DMA1) | the column's **own** memtile | 5120 B | **private per column** |

One operand stream is shared by every GEMM column; the other is per-column. For
a GEMV that is the expected split — every core needs the same activation vector
and its own slice of the weights.

Core-to-core forwarding is **circuit**-switched (cheap); the shims are mostly
packet-switched.

**derived — `p5` carries the activation vector.** Exactly **16 cores** have DMA0
enabled with a 512 B BD0 at module `0x08000`, and they are exactly the **16 GEMM
cores**, byte-identical on every one; non-GEMM cores use entirely different
geometry (4096 B, 1024 B, 2048 B, 6144 B at other addresses). The stream is
512 B = 256 bf16 = K=256, K-indexed, rewound across N. Weights and per-group
scales cannot be broadcast identically because each core computes a different
output slice, so the only operand such cores can share is the input activation
vector.

This corrects the goal doc, which labelled `p1` as the activation stream: `p1`'s
*access pattern* (512 B at a time, rewound) is right, but its buffer is private
to the column, and the activations are necessarily the shared one.

Established from connectivity plus architectural necessity, not from reading the
bytes — core L1 is not reachable from userspace, so a byte-level confirmation
would need the driver's AIE debug path.

**This also confirms the GEMM role structurally** rather than by op mix: those 16
cores are exactly the set receiving a broadcast K=256 activation vector plus a
private per-column weight/scale stream, and emitting a small result stream
(BD4/BD5, 132 B).

**open** — the graph is not uniform: (2,3) uses `S2MM1 -> BD6`, and (3,3)/(4,3)
leave DMA0 unconfigured. Row-3 tiles in columns 3-4 have a different shape.

---

## 6c. `attn.xclbin`

Covered in its own deliverable: **`flm-attn-dataflow.md`**. Headline results,
**rewritten 2026-07-31 — the head-mapping version that stood here is retracted**:

- The array is a **fixed 32-core template** that does not track head count. Same
  32 cores and same buffer sizes on models with 32, 24, 16 and 4 query heads, and
  on one with a single KV head. `32 cores = 32 query heads`, `8 columns = 8 KV
  heads`, `4 rows = GQA ratio` and the `k03`/`k47` column mapping were all read
  off Llama-3.2-1B, where those numbers coincide by accident.
- Cores are differentiated by two compile-time constants, `row - 2` and
  `col & 1` — **8 distinct binaries** — plus which DMA stream they are wired to.
- **BD0 is Q** (64 query positions per column pair, 32 per core), **BD1/BD2 are
  K/V** (array-wide broadcast from memtile (3,1)), BD3/BD4 are the output. This
  is the reverse of what that doc first concluded.
- Tile width **32 query positions per core**, cross-checked by the memtile's
  output-collection BD gathering exactly 32 x 128 B rows.
- The layout transform is performed by the **memtile DMA's 4-D strided
  addressing**, not by the cores — unchanged, and true of both operands.

Consistent with this document: `attn.xclbin` is the **prefill** kernel.

---

## 7. What is NOT established

- **The decode command's instruction stream.** It is not resident in userspace
  in mlir-aie transaction format: 0 hits scanning 6.0 GB of the live process
  (182 regions including 84 device BOs, verified readable at 931.8 MB non-zero)
  for TXN headers, `MERGE_SYNC` ops, or 24-byte `DDR_PATCH` ops. How the
  whole-model program reaches the device is open.
- **Which phase the MoE's third command is.** Plausibly its
  `mtp_num_hidden_layers: 1` head or its 30 linear-attention layers; not shown.
- **Intra-layer ordering during prefill** (section 1).
- ~~**What is split between a GEMM pair**~~ — **CLOSED 2026-07-31: N-split, the
  pair concatenates.** No combining code exists in either program, and the pair
  emits two equal 132 B buffers rather than one reduced result. See section 1.
- **One core role remains inferred**: col 2 row 3 (6144 in / 4096 out, not
  decomposed). The cols 3-4 pipeline is **RESOLVED: it applies RoPE** — the
  elementwise cores carry `vmsc.f` (the multiply-subtract `x*cos - y*sin`
  needs) found essentially nowhere else, and the pipeline's operands are
  exactly **Q (4096 B = 2048 bf16 = 32x64)** and **K (1024 B = 512 bf16 =
  8x64)** with **no V stream**, which is precisely what RoPE touches. Still
  inferred within that: which of the pair deinterleaves vs rotates, and whether
  these cores also carry a further elementwise stage (`vadd.f:73` exceeds what
  the rotation alone needs). GEMM, RMSNorm (2,2) and
  SwiGLU (5,2) **are** confirmed, and the `p5` operand stream **is** resolved
  (the broadcast activation vector).
- ~~**Where RoPE is applied.**~~ **CLOSED 2026-07-31: cols 3-4.** The symbols put
  it in `layer.xclbin` (present in `libllama_npu.so`, absent from `libmha.so`);
  the **operand widths** locate it within the kernel. Both cols 3-4 cores carry
  exactly two double-buffered widths — **4096 B = 2048 bf16 = Q (32x64)** and
  **1024 B = 512 bf16 = K (8x64)** — matching `config.json` exactly, with **no
  third stream for V**. V is also 512 bf16, so it would be visible if present.
  RoPE touches Q and K and never V, which is precisely the pattern observed.
  Supporting: `vmsc.f` (the multiply-subtract `x*cos - y*sin` needs) occurs in
  `layer.xclbin` only in these four cores (24 each) and RMSNorm (4).

  Note the earlier objection stands on its own terms and is why the widths, not
  the opcode, carry this: `attn` cores also have `vmsc.f` = 24, so the opcode
  alone is not a RoPE fingerprint across kernels.
- **Whether any of this transfers to the MoE.** It is a *hybrid*: 30 linear
  attention / SSM layers, 10 full attention, `head_dim` 256,
  `num_key_value_heads` 2, 256 experts with 8 active, plus multi-token
  prediction. The Llama-shaped attention analysis should not be assumed to port.

---

## 8. Implications for reproduction

1. **Two dispatch designs are needed, not one.** Matching decode means a single
   command carrying the whole model — a host-side and control-code problem.
   hipfire's 4-dispatches-per-layer decode is **64 submissions per token** on
   this model against FLM's 2; at the measured ~37 us per-dispatch floor that is
   ~2.4 ms/token of pure submit latency, which dominates everything else at
   these sizes. On current evidence this is worth more than any kernel-body
   change.
2. **Do not copy the `CREATE_BO` churn** (section 5) or the
   full-context KV preallocation unless long context is required.
3. **KV capacity, not just bandwidth, is a target.** 4 GiB of KV for a 1B model
   at max context. The goal doc argues KVarN on bandwidth; capacity may matter
   more.
4. **llama decode has ~14% bandwidth headroom.** Wins must come from fewer bytes
   (oq4++ at 4.125 bpw vs FLM's 5.00 = 17.5% fewer), not better feeding.
5. **The MoE is the more interesting target** — 67% of fabric means a third of
   the achievable rate is lost to something other than weight streaming.
6. **Three distinct mechanisms move operands without paying DRAM bandwidth**,
   and a reproduction that misses them will not reach 86% of fabric:
   - **Horizontal broadcast.** One memtile MM2S channel feeds several columns by
     circuit-switched forwarding through intermediate tiles' stream switches
     (section "The broadcast path"). One memtile read, many consumers.
   - **Shared neighbour memory.** Paired cores hand off by storing into the
     adjacent tile's data memory — no DMA, no cascade, no instructions at all
     beyond a pointer into a different window. Free in the instruction stream.
   - **Strided DMA layout transform.** The memtile's 4-D buffer descriptors
     reshape data while it moves; the cores always receive flat contiguous
     tiles and never reshape anything (see `flm-attn-dataflow.md` §3).
7. **The GEMM cores need two program variants**, differing only in output
   addressing (one writes locally, its partner writes into the neighbour) — not
   one program plus configuration.
8. **The whole-model single dispatch needs one policy change and one design
   decision.** `layer.xclbin` declares the ordinary DPU signature
   (`connectivity` m_count 6: args 1, 3-7) yet binds **50** buffers, because the
   `args` array of `amdxdna_drm_exec_cmd` is a driver-level **buffer table**
   indexed by `DDR_PATCH`'s `arg_idx`, not the kernel signature. The 5 is
   `kMinHostBOs` — a **floor** required by the firmware command-chain ABI
   (`tools/aiecc/SidecarFiles.h:85`), not a ceiling.

   Probed with `tools/npu/flm/manybuf_probe.py`. Three limits sit in the path,
   only one of which is real:

   | limit | value | nature |
   |---|---|---|
   | driver `MAX_ARG_COUNT` | 4095 | not binding |
   | `address_patch` `arg_idx` | I32Attr | not binding |
   | `aiecc` `kMaxHostBOs` | **16** | **policy** — "unvalidated", raised to 64 and works |
   | **active BDs per shim tile** | **16** | **structural** |

   So: (a) raise `kMaxHostBOs`, a one-line change with FLM as the existence proof
   that the silicon is fine with 50; and (b) **reuse BDs rather than allocating
   one per buffer** — `aiex.dma_free_task` / `dma_await_task`. (b) is the real
   design requirement, and it is exactly what FLM does: **4 BDs per column,
   reused, with `DDR_PATCH` rewriting the address each dispatch**. Its 50 buffers
   are 50 patchable addresses cycling through a handful of descriptors, not 50
   descriptors.
