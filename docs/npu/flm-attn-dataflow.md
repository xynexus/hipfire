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

## 1. The array is a fixed 32-core tile grid — NOT a head decomposition

> **RETRACTED 2026-07-31.** This section previously read the array as
> `32 cores == 32 query heads`, `8 columns == 8 KV heads`,
> `4 rows == GQA ratio 4`, and concluded `k03`/`k47` maps to columns 0-3 / 4-7.
> **All of that is wrong.** It was derived from Llama-3.2-1B alone, where the
> model's head counts happen to coincide with the array's dimensions. They do
> not coincide anywhere else. See below.

**measured** — `tools/npu/flm/cdo.py` extracts **32 cores** with program memory,
one per tile in columns 0-7 x rows 2-5.

**measured** — the same extraction on three other models' `attn.xclbin`, chosen
because their head geometry differs:

| model | q heads | kv heads | head_dim | cores | core BD sizes (B) |
|---|---|---|---|---|---|
| Llama-3.2-1B | 32 | 8 | 64 | **32** | 8192 / 4096 / 4096 / 128 / 128 |
| Llama-3.2-3B | 24 | 8 | 128 | **32** | 8192 / 4096 / 4096 / 128 / 128 |
| Qwen3-0.6B | 16 | 8 | 128 | **32** | 8192 / 4096 / 4096 / 128 / 128 |
| Gemma3-1B | 4 | **1** | 256 | **32** | 16384 / 8192 / 8192 / 128 / 128 |

**Always 32 cores, and the buffer sizes do not track the head count.** A model
with 4 query heads and a single KV head gets the same 32-core array with the
same topology (section 4). The array is a fixed template; the decomposition is
by **tile**, not by head.

Two immediate consequences for anything built on the old reading:

- **The 128 B output buffer is not "1 x head_dim".** It is 128 B on models with
  head_dim 64, 128 *and* 256. It is a transfer granule of 64 bf16, nothing more.
- **`k03` / `k47` does not map onto array columns.** That split is real in the
  symbol table; this document had no evidence for locating it, and the
  coincidence that made it look located was Llama-3.2-1B's 8 KV heads matching
  the 8 columns.

**measured** — the 32 cores carry **8 distinct programs, not 32 and not 1.**
Cores are byte-identical (md5 on the extracted program memory) exactly when they
share `(row, column parity)`. The differentiating constant is visible in the
prologue as two immediates passed into the entry call:

```
core (0,2):  movx r0, #0x0 ; mov r1, #0x0        r0 = row - 2   (0..3)
core (1,2):  movx r1, #0x1 ; mov r0, #0x0        r1 = col & 1   (0..1)
core (7,5):  movx r1, #0x1 ; mov r0, #0x3
```

Across all 32 cores `r0` is exactly `row - 2` and `r1` exactly `col & 1`, with
no other per-core immediate anywhere in the program and no per-core base
address (every core's address immediates are the identical set
`0x70000, 0x72000, 0x73000, 0x74000, 0x78000, 0x7a000, 0x7b000, 0x7c000,
0x7e000, 0x7e800`). Per-core data-memory init is 36 B at `0x30a4`, and its one
varying word tracks program *size*, not position.

**So a core's identity is `(row, column parity)` plus which DMA0 stream it is
wired to — nothing else.** Contrast with `layer.xclbin`, which is heterogeneous
(16 GEMM cores + norm + RoPE + SiLU); attention is homogeneous because the work
is a uniform tiling.

---

## 2. Per-core buffers, and the flash tile width

**measured** — `tools/npu/flm/cdo_dma.py`. Five BDs, identical on all 32 cores:

**measured** — directions are read from the DMA **channel queue registers**
(`0x1DE04`/`0x1DE0C` for S2MM ch0/1, `0x1DE14`/`0x1DE1C` for MM2S ch0/1; the low
nibble is the channel's starting BD), not inferred from size:

| BD | channel | direction | bytes | as bf16 | pairing | source / sink |
|---|---|---|---|---|---|---|
| BD0 | S2MM0 | **input** | 8192 | 4096 | single | memtile |
| BD1 / BD2 | S2MM1 | input | 4096 | 2048 | ping-pong | east neighbour |
| BD3 / BD4 | MM2S0 | **output** | 128 | 64 | ping-pong | -> memtile |

MM2S1 has its control written but no queue, so it is never started.

**derived**, at head_dim 64 in bf16 — **revised 2026-07-31, and BD0 vs BD1/BD2
swapped roles relative to what this section said before**:

| BD | interpretation |
|---|---|
| BD0, 8192 B | **the Q tile — 64 query positions x 64 dims**, private to a column pair. Each of the two cores takes one 4096 B half (32 positions) selected by its `col & 1` constant. |
| BD1/BD2, 4096 B | **the K / V tiles — 32 tokens x 64 dims**, array-wide broadcast, the flash-attention inner-loop operand. Not a ping-pong of one operand and not "from the east neighbour": one source, all 32 cores (section 4). |
| BD3/BD4, 128 B | **the attention OUTPUT, one row of 64 bf16**, double-buffered; 32 of them per core per pass |

> **CORRECTION (2026-07-31), first pass.** BD3/BD4 were originally read as "the Q
> vector for one head (decode, M=1)" — an *input*. MM2S0 starts at BD3, so they
> are the core's **output**. Size alone could not separate a 128 B Q input from a
> 128 B O output; the channel register could.
>
> **CORRECTION (2026-07-31), second pass — this is the one that matters.** With
> BD3/BD4 out as the Q candidate, this section then assigned *both* input
> channels to the KV side: BD0 = "K and V for a 32-token tile" (on a size match
> against the memtile's strided output) and BD1/BD2 = "a KV tile from the
> adjacent column". **Both assignments are wrong, and they are wrong in a way
> size-matching could not catch** — 8192 B is *not* uniquely the memtile's
> strided output size, because memtile (3,1) emits strided too, at a different
> size. Section 4's route resolution and section 6's argument settle it the other
> way round: BD0 is Q, BD1/BD2 are K/V.

**The tile width is 32 positions per core.** The goal doc lists this as not
pinned ("loop counts are `lc = 0x20 / 0x1f / 0x8`; 32 tokens/tile is the likely
read"). BD geometry settles it, and reconciles all three loop counts:

- `lc = 0x20` = **32** — the tile width
- `lc = 0x08` = **8** — an 8-way inner count (previously glossed as "the KV-head
  / column count"; that gloss depended on the retracted head mapping and should
  not be carried forward as established)
- `lc = 0x1f` = **31** — the tile loop with the first iteration peeled, which is
  what an online softmax does (initialise from element 0, fold in the other 31)

BD1/BD2 being a double-buffered pair is textbook flash attention: the next K/V
tile streams in while the current one is consumed.

~~**open** — BD0's exact role.~~ ~~**CLOSED: K+V input tile from the memtile.**~~
**RE-OPENED and then closed the other way, 2026-07-31: BD0 is the Q tile.** See
section 6.

---

## 3. The layout transform is done by the DMA, not the kernel

> **Scope correction 2026-07-31.** The shape decoded below belongs to memtiles
> (0,1), (2,1), (4,1), (6,1) — which section 6 establishes carry **Q**, not KV.
> The KV broadcast comes from memtile (3,1), whose egress BDs are also 4-D
> strided but at a different shape: `D1_STEPSIZE=127` (512 B), `D2_WRAP=32`,
> D3 stride 4096 B, 32768 B per transfer, fed by a 65536 B flat ingress. **The
> structural claim — the memtile DMA reshapes, the cores never do — is unchanged
> and holds for both operands.** Only the attribution of the 128 B stride to KV
> was wrong.

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

## 4. Fan-out topology: one channel pairwise, one channel array-wide

**measured** — stream-switch config, port indices resolved against aie-rt's
`Aie2PTileStrmSwSlavePortMap`, then each core's DMA source walked back hop by
hop to its origin. (Mirroring rule: a tile's slave `EASTn` is the east
neighbour's master `WESTn`, `SOUTHn` the south neighbour's `NORTHn`, and so on —
the same rule `flm-layer-dataflow.md` uses, where it was checked against a
hand-traced broadcast path.)

The two input channels have completely different topologies, and this is the
central structural fact about the kernel:

| channel | distinct origins | consumers per origin |
|---|---|---|
| **DMA0** (BD0, 8192 B) | **16** — memtiles (0,1), (2,1), (4,1), (6,1), MM2S channels 0-3, one channel per row | **2** — an even column and the odd column beside it |
| **DMA1** (BD1/BD2, 4096 B) | **1** — memtile (3,1) MM2S0 | **32** — every core in the array |

Even columns pull DMA0 **SOUTH** (from their own memtile) and forward it EAST;
odd columns pull **WEST** from the even column beside them, so the pair receives
byte-identical data and splits the work by the `col & 1` constant of section 1.

DMA1 fans out from a single memtile through the whole array — the deepest chain
is 8 hops, `(3,1) -> ... -> (7,5)`. Every hop is circuit-switched, so all 32
cores see identical bytes.

**Confirmed on a model whose head geometry cannot support the old reading**:
Gemma3-1B has **1 KV head**, and its `attn.xclbin` has exactly the same
topology — 16 distinct DMA0 streams, one array-wide DMA1 broadcast. Sixteen
*distinct* streams cannot be KV for a one-KV-head model, and a single broadcast
is exactly what one KV head wants. This is the strongest single piece of
evidence for the BD assignment in section 6.

**derived** — the array-wide broadcast is the same mechanism `layer.xclbin` uses
for its activation vector, and for the same reason: one memtile read serves many
cores instead of each pulling its own copy.

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

## 6. Where Q arrives — and why this is unambiguously the prefill kernel

**Q arrives on BD0 / S2MM0**: the 8192 B buffer fed by memtiles (0,1), (2,1),
(4,1), (6,1), private to each even/odd column pair, each core taking the 4096 B
half its `col & 1` constant selects. The array-wide 4096 B broadcast on
BD1/BD2 from memtile (3,1) is **K/V**.

Four independent lines, none of them a size match:

1. **A model that breaks the alternative.** Gemma3-1B has one KV head and the
   identical topology (section 4). Under the alternative — DMA0 = KV — its one
   KV head would be duplicated across 16 distinct streams while its Q was
   broadcast. Under this reading, the single KV head is broadcast once and 16
   distinct Q tiles are delivered. Only one of those is a design.
2. **The runtime argument list.** The attention dispatch (prefill ctx4) takes
   exactly three buffers: the 256 MiB KV cache, q_proj's output, o_proj's input
   (`flm-layer-dataflow.md` section 1, measured by dumping argument BO sizes).
   The array has five DDR ingress columns: 0, 2, 4, 6 and 3. Four carry the
   per-pair streams and one carries the broadcast, so the operand on the
   broadcast column is the one that every core needs identically.
3. **The output arithmetic closes.** Each core emits 128 B rows through a
   ping-pong (section 2), and the memtile's output-collection BD is **4096 B
   strided with D1 stride 128 B** — it gathers exactly **32 rows of 128 B**. So a
   core produces 32 output rows per pass, which requires 32 query positions of
   private Q per core, which is exactly the 4096 B half of BD0 that `col & 1`
   selects. Under the alternative, a core's output count would not close.
4. **It is what flash attention wants.** The KV cache is the operand that must be
   re-read once per query tile; broadcasting it to 32 cores at once divides that
   traffic by 32. Q is read once either way. Putting the *broadcast* on KV is the
   whole point, and it is the same trick `layer.xclbin` uses on its activation
   vector.

**This resolves the cross-document inconsistency, in favour of prefill.** An
array of 32 cores that does not scale with head count, each holding a private
tile of 32 query positions against a broadcast KV stream, is query-tile-parallel
flash attention — the prefill shape. The "decode, M=1" gloss in the original
section 2 was an artifact of misreading the 128 B output as a Q input;
`flm-layer-dataflow.md` section 1 is correct that `attn.xclbin` serves the
per-linear prefill contexts and `layer.xclbin` is the decode path.

**Still open within this**: what the `row - 2` constant (0..3) selects. Column
parity is accounted for (which half of the Q tile); the row index is not.
Candidates are an output slot in the memtile collection, or a phase offset in
the KV stream. It is a narrow question and does not block a rebuild of the
dataflow, but it does block writing the *scheduling*.

## 7. What is NOT established

- ~~**Where RoPE is applied.**~~ **CLOSED 2026-07-31 — the leading hypothesis
  here was right: RoPE runs in `layer.xclbin` cols 3-4, not in this kernel.** It
  was settled by **operand widths**, not by the opcode mix (this document was
  correct that `vmsc.f` = 24 is a shared idiom and cannot discriminate). The
  cols 3-4 cores carry exactly two double-buffered widths — **Q, 4096 B = 2048
  bf16 = 32x64** and **K, 1024 B = 512 bf16 = 8x64** — and **no V stream**. V is
  also 512 bf16, so it would be visible if present; RoPE touches Q and K and
  never V. See `flm-layer-dataflow.md` section on the cols 3-4 pipeline.
- ~~**BD0's role**~~ **CLOSED 2026-07-31, twice** — first wrongly as the K+V
  tile, then correctly as **Q** (section 6).
- ~~**Where Q arrives.**~~ **CLOSED 2026-07-31** — BD0 / S2MM0 (section 6).
- ~~**A cross-document inconsistency.**~~ **RESOLVED 2026-07-31 in favour of
  prefill** (section 6). `flm-layer-dataflow.md` was right; the "decode, M=1"
  gloss here was wrong.
- **What the `row - 2` core constant selects** (section 6). Narrow, but it is the
  gap between describing the dataflow and scheduling it.
- **Whether any of this transfers to the MoE.** Qwen3.6-35B-A3B is a hybrid: 30
  linear-attention/SSM layers, 10 full attention, `head_dim` 256,
  `num_key_value_heads` 2, plus multi-token prediction. Less of a risk than it
  was — the array is now known to be a fixed 32-core template rather than a head
  mapping, so the *shape* probably does port — but the tile sizes will not, and
  the SSM layers have no counterpart here at all.
- **KV cache capacity behaviour.** `flm-layer-dataflow.md` establishes the decode
  path preallocates **256 MiB per layer** (full 131072 context), 4 GiB total for
  a 1B model. Relevant to 3b: KVarN is argued on KV *bandwidth*, but capacity is
  the binding resource at long context.
