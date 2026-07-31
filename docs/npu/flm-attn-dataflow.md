# FLM `attn.xclbin` — dataflow

Phase 1 deliverable for `~/flm-re-fe-mutate-goal.md`, covering FastFlowLM's
flash-attention kernel on AIE2P. Companion to `flm-layer-dataflow.md` (the decode
path and host-side dispatch). Working and failures: `flm-refe-log.md`.

**Every number carries its source**, tagged **measured** (method named),
**derived** (arithmetic shown), or **open** (stated because it matters,
explicitly not established).

**Provenance.** Facts about observable behaviour — buffer sizes, register
values, geometry. It deliberately does not reproduce FLM's schedules or the
particular organisation of its computation. See the goal doc's provenance
section before extending.

Model: **Llama-3.2-1B** — 32 query heads, 8 KV heads, head_dim 64, GQA ratio 4,
`max_position_embeddings` 131072.

---

## 1. The array geometry *is* the head decomposition

**measured** — `tools/npu/flm/cdo.py` extracts **32 cores** with program memory,
one per tile in columns 0-7 x rows 2-5. **All 32 are byte-identical in DMA
configuration** (five buffer descriptors, same sizes, same lock structure);
program sizes differ only slightly (10596 / 10612 / 10628 B).

**derived** — every count matches the model with no remainder:

```
32 cores          == 32 query heads   -> ONE CORE PER QUERY HEAD
8 columns         ==  8 KV heads      -> ONE COLUMN PER KV HEAD
4 rows per column == GQA ratio 4      -> the 4 query heads sharing that KV head
```

**This answers the `k03`/`k47` split.** FLM's symbol table carries
`get_k03_offset` / `k47` / `v03` / `v47`, i.e. KV heads in two groups of four.
With one KV head per column that is simply **columns 0-3 and columns 4-7** — the
left and right halves of the array.

Contrast with `layer.xclbin`, which is heterogeneous (16 GEMM cores + norm +
shuffle + SiLU). Attention is homogeneous because the work decomposes cleanly by
head.

---

## 2. Per-core buffers, and the flash tile width

**measured** — `tools/npu/flm/cdo_dma.py`. Five BDs, identical on all 32 cores:

| BD | bytes | as bf16 | pairing | locks |
|---|---|---|---|---|
| BD3 / BD4 | 128 | 64 | ping-pong (NEXT 3<->4) | acq 5, rel 4 |
| BD1 / BD2 | 4096 | 2048 | ping-pong (NEXT 1<->2) | acq 2, rel 3 |
| BD0 | 8192 | 4096 | single | rel 1 |

**derived**, at head_dim 64 in bf16:

| BD | interpretation |
|---|---|
| BD3/BD4, 128 B | **1 x head_dim** — the Q vector for one head (decode, M=1), double-buffered |
| BD1/BD2, 4096 B | **32 x 64** — a KV tile of **32 tokens**, double-buffered |
| BD0, 8192 B | 2 x 32 x 64 — K and V for 32 tokens |

**The flash tile width is 32 tokens.** The goal doc lists this as not pinned
("loop counts are `lc = 0x20 / 0x1f / 0x8`; 32 tokens/tile is the likely read").
BD geometry settles it, and reconciles all three loop counts:

- `lc = 0x20` = **32** — the tile width
- `lc = 0x08` = **8** — the KV-head / column count
- `lc = 0x1f` = **31** — the tile loop with the first iteration peeled, which is
  what an online softmax does (initialise from element 0, fold in the other 31)

BD1/BD2 being a double-buffered pair is textbook flash attention: the next KV
tile streams in while the current one is consumed.

**open** — BD0's exact role. 8192 B, single, not ping-ponged, its own lock.
Consistent with K+V for 32 tokens or an output accumulator; not distinguished.

---

## 3. The KV layout transform is done by the DMA, not the kernel

**measured** — the memtile (row 1) has **25 buffer descriptors** against 5 per
core, and the structural difference is the point:

> **Every core BD in both kernels is a flat linear transfer. The memtile's are
> 4-dimensional strided.**

| BD group | bytes | addressing | locks |
|---|---|---|---|
| BD0/BD1, BD24/BD25 | 16384 | **flat** | 64->65, 68->69 |
| BD2/BD3, BD4/BD5, BD26/BD27 | 8192 | **4-D strided** | 65->67, 69->70, 67->64 |
| BD6-BD9 | 4096 | **4-D strided** | 71->72, 73->74 |

Every strided BD carries the same shape:

```
D0_WRAP=4   D1_STEPSIZE=31 D1_WRAP=8   D2_STEPSIZE=3 D2_WRAP=8   D3_STEPSIZE=255
```

**derived** — aie-rt encodes stepsize as `StepSize - 1`
(`dma/xaie_dma_aieml.c:316-355`), so the real strides are:

| dim | stride | note |
|---|---|---|
| D1 | 32 words = **128 B** | **exactly one head_dim vector** (64 x bf16) |
| D2 | 4 words = 16 B | 8 bf16 |
| D3 | 256 words = 1024 B | |

`8192 B / (4 x 8 x 8 = 256 units) = 32 B` per unit = 16 bf16.

Flat 16 KB regions are written in; strided 8 KB tiles are read out; the lock IDs
form a cycle (`64->65->67->64`), i.e. a recycling pipeline. **The cores never
reshape anything.**

**Implication for phase 3b.** The goal doc motivates KVarN's channel-major
`[head_dim x GROUP]` K layout by the need to avoid a transpose for the
`mac_4x16_16x16` B operand. FLM solves the equivalent problem **in the memtile
DMA, for free, while data moves**. A KVarN implementation has the same lever, and
should use it rather than constraining its storage layout to dodge a transpose
the DMA engine could perform anyway.

---

## 4. Fan-out topology: pairwise, not array-wide

**measured** — stream-switch config, port indices resolved against aie-rt's
`Aie2PTileStrmSwSlavePortMap`. DMA0's source alternates by column parity:

```
(0,2) SOUTH0   (1,2) WEST2      (2,2) SOUTH3   (3,2) WEST1
(4,2) SOUTH3   (5,2) WEST0      (6,2) SOUTH0   (7,2) WEST1
```

Even columns pull **SOUTH** (their own memtile); odd columns pull **WEST** (from
the even column beside them).

**derived** — attention broadcasts in **column pairs**, unlike `layer.xclbin`'s
long horizontal chains that span the array. Consistent with the sharing pattern:
here a KV head is shared by 2 columns, whereas `layer`'s activation vector is
shared by all 16 GEMM cores.

---

## 5. Compute

**measured** — `tools/npu/flm/aiedis.py` on core (0,2), 1693 bundles:

| op | count |
|---|---|
| `vshuffle` | 104 |
| `vmul.f` | 70 |
| `vadd.f` | 51 |
| `vmac.f` | 44 |
| `vconv.bfp16ebs8.fp32` | **37** |
| `vmsc.f` | 24 |

`vconv.bfp16ebs8.fp32` x37 **confirms** the goal doc's "operands load bf16,
convert in-register to BFP16-ebs8, feed `mac_8x8_8x8T`" — the conversion is real
and in the inner path.

Recall from `flm-layer-dataflow.md` that `mac_8x8_8x8T` on `v64bfp16ebs8`
measures **1.0 cyc/call = 512 MACs/cycle**, the same rate as int8 and int4.

---

## 6. What is NOT established

- **Where RoPE is applied.** Narrowed, not proven. RoPE symbols
  (`_set_rope_weights`, `_send_rope_weights`, `_rope`) exist in **exactly one**
  library — `libllama_npu.so` — and **`libmha.so` has none**. Combined with
  `layer.xclbin`'s cols 3-4 cores carrying the required
  `vshuffle`+`vmul.f`+`vmsc.f`+`vadd.f` combination, and RoPE architecturally
  applying to Q/K right after projection, the leading hypothesis is that **RoPE
  runs in `layer.xclbin`, not here**. Note opcode mix does *not* discriminate:
  `attn` cores and `layer` cols 3,4 rows 3,5 both have `vmsc.f` = 24 exactly, so
  that signature is a shared idiom. Settle it with a call-graph trace from
  `_send_rope_weights` (the direct-call xref is empty — the call is indirect).
- **BD0's role** (section 2).
- **Whether any of this transfers to the MoE.** Qwen3.6-35B-A3B is a hybrid: 30
  linear-attention/SSM layers, 10 full attention, `head_dim` 256,
  `num_key_value_heads` 2, plus multi-token prediction. The head-per-core mapping
  above is specific to Llama-shaped GQA and should not be assumed to port.
- **KV cache capacity behaviour.** `flm-layer-dataflow.md` establishes the decode
  path preallocates **256 MiB per layer** (full 131072 context), 4 GiB total for
  a 1B model. Relevant to 3b: KVarN is argued on KV *bandwidth*, but capacity is
  the binding resource at long context.
