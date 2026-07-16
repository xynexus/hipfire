# TODO: batched NPU encode — un-pin `rows()==256`, amortize the per-dispatch floor

Date: 2026-07-15. Follow-up from the route-A step-1 instrumentation of the
EmbeddingGemma resident NPU encode. This is the highest-throughput lever we found.

## Why (the measured motivation)

Route-A step-1 instrumentation (env `HIPFIRE_XDNA_TRACE` in `hipfire-xdna`
`kernel.rs`; extra `attn_*` fields under `HIPFIRE_EMBED_TRACE_RESIDENT` in
`npu_opus.rs`) of the M256 oq8 encode on halo:

- **submit (host ioctl + cache flush) = 11.9 ms (1.5%)**; **wait (NPU execution)
  = 803.9 ms (98.5%).** The encode is NPU-execution-bound, *not* launch-bound —
  so dispatch-*batching* to cut launch overhead (route A1) is worthless.
- **A ~4 ms fixed per-dispatch NPU-execution floor.** The `tail` and each `prep`
  kernel move ~0 weight bytes / ~400 KB activations and do µs of math, yet each
  takes ~4 ms. At ~4 dispatches/layer that is ~16 ms/layer — **~50% of the
  31.9 ms** — spent on a fixed cost doing almost nothing.
- W4 (oq4) did **not** speed up the big kernels: the `DenseW8` resident kernel
  consumes 8-bit-aligned weights, so oq4 expands to 8-bit at load and the
  on-device DMA is unchanged. A real W4 DMA win needs an **int4-consuming**
  resident kernel (the r11 `mmul<…,int8,int4>` path).

M-sweep of one bf16 GEMM (K=N=768) — clean points (M=256/2048/4096, the sub-floor
outliers at 512/1024 were XRT readback-barrier artifacts): wall time is
**~flat at 6.0 → 6.5 ms across a 16× range of M**. i.e. ~6 ms fixed floor,
negligible per-row work at these shapes → **batching to M=4096 yields ~15× more
rows for the same dispatch cost.** (Re-measure precisely via the Rust r6/r11
harness; the Python/XRT path is timing-noisy.)

**Conclusion:** batch=1 is the NPU's pathological case (fixed cost dominates).
Batch amortizes the ~4 ms floor **and** the per-layer weights (8 MB attn + 11 MB
FFN) over 128–256× the rows → GEMMs flip compute-bound, the array approaches the
~55-TOPS int8 peak, and int8 / true-W4 / STEEL / KVarN-4 finally pay. Embeddings
are naturally batched (corpora, not single strings). Estimated 60×+ throughput
(200K+ tok/s ceiling at B=128×S256 vs 336 tok/s).

## The blocker

The resident kernels are hard-pinned to a single sequence:
`NpuEmbeddingLayerAttentionDenseW8::rows() == 256` (checked in
`npu_opus.rs::project_layer`, `resident_embedding_layer.rs`), likewise FFN / tail
/ prep. Everything assumes M256.

## Work items

1. **Un-pin to a single large-M dispatch (the crux).** Support M = B·S in the
   resident attention/FFN/tail/prep kernels via M-tiling, in **one** dispatch over
   all B·S rows. A batch *loop* of B separate M256 dispatches does **not** help —
   it pays the ~4 ms floor B times. The amortization only comes from one big-M
   dispatch (or a batched runlist that overlaps floors). Start with FFN (simplest,
   weight-heavy, biggest win) as proof.
2. **Block-diagonal attention.** Each document attends only within itself; do not
   materialize (B·S)². Per-sequence attention blocks or a segment-masked kernel.
   (For encode this is bidirectional/dense within each block, no causal mask.)
3. **Activation staging that scales with B.** ~100 MB/layer at B=256×S256,
   streamed through Mem tiles; fine in halo's 128 GB UMA but must tile on-chip.
4. **GPU bridge** (`pack_opus_npu_activations`) batches trivially — more rows,
   same code; already 0 on interior layers.
5. **Precision at batch (where it finally pays):** int4-consuming resident kernel
   for true-W4 weight-DMA halving (oq4++), int8 compute (2–4×), KVarN-4 K. These
   move the batched (compute/bandwidth-bound) regime, not batch-1.

## Open questions

- Does one big-M dispatch actually amortize the ~4 ms floor, or does the floor
  scale with M-tiles internally? (The M-sweep says work is ~flat to M=4096, which
  is the encouraging signal — confirm on the Rust path.)
- Attention: single segmented kernel vs per-sequence blocks — which tiles better
  at head_dim 256 within 64 KB local memory. (Cross-ref the STEEL todo: batched
  prefill is where the STEEL pipeline fill also pays.)
- Interaction with `MAX_CONTEXT_COMMANDS` / `recreate_hwctx` resets.

## Progress (2026-07-15)

**Generator un-pinned + compile-proven.** `r26_gen.py` parametrized by `--batch=N`
(BATCH=1 = original M256): `M_MACROS = 3*BATCH`, `PAD_M = 96*M_MACROS`,
`T_ROWS = 96*M_MACROS+8`, `REAL_M = 256*BATCH` feeding the direct-x byte/offset
literals + inverse-table record count. Verified:
- **Value-preserving at BATCH=1**: MLIR byte-identical to the pre-edit generator
  across all 5 ABI modes (base, canon, canonx2, directx, directx_reuse).
- **Compiles at BATCH=2 (base mode) through aiecc**: xclbin (PDI/core program)
  **516,111 B at both M256 and M512 — byte-identical** (core is M-agnostic, no
  16 KiB risk); insts.bin **39,376 → 76,768 B (1.95×, ~2 KB fixed + ~37 KB/batch)**
  — BD schedule scales linearly, no BD-ID/toolchain ceiling at 2×. At B=128 insts
  ≈ 4.7 MB, fits the 64 MB device heap.

**r35 wired + canonical M512 compiles (2026-07-15).** r35 is NOT a separate
kernel — it's the same `r26_w8_resident_ffn.cc` compiled with
`-DR35_CANONICAL_BF16` (r35_cache.sh). Built canonical-bf16 at M256 and M512
(`--batch=2`): xclbin **524,847 B identical** at both (core PDI M-invariant),
insts **95,792 → 190,256 B (1.99×)**. And the rebuilt M256 canonical cache
**passes the hardware parity check** (`npu_resident_ffn_w8_canonical_verify`:
cosine 0.99985, max_abs 0.0049, dispatch 10.1 ms) — the generator refactor is
value-preserving on hardware, not just in the MLIR diff.

**Block-index formulas generalize for free.** `gate_block_index` /
`down_block_index` / `gate_param_block_index` (resident_ffn_w8.rs:657-698) are all
`mblock`-linear, so they extend to 6 macros with no formula change. And
`upload_weights_inner` (:331,374 `for mblock in 0..3`) writes the **same** weight
tile into every macro slot (weight data is `mblock`-independent) — so at M512 doc0
(macros 0-2) and doc1 (macros 3-5) get identical weights, making **doc0==doc1 the
correct, oracle-free batching-correctness check**.

Remaining for the on-NPU M512=2×M256 number — the **host batched-FFN refactor**:
- `resident_ffn_w8.rs` is const-based: ~50 sites use compile-time `M`/`PAD_M`/
  `GATE_BLOCKS`/`WEIGHT_BLOCKS`/`T_ROWS`/`*_BYTES`. Batching needs a runtime
  `batch` (m_macros = 3·batch) threaded through: the two `0..3` mblock loops →
  `0..3·batch`, the block-*count* consts → ×batch, buffer sizes → ×batch.
  Block-index *formulas* are unchanged. Must preserve M256 (regression gate =
  the parity example above) across all 4 ABI modes.
- Then adapt `npu_resident_ffn_w8_canonical_verify` to run M512 with input `[X;X]`
  and assert `out[0:256] == out[256:512] == M256(X)`.
- direct-x mode adds the inverse-table scaling (already
  `PRE_INVERSE_BASE//PRE_INVERSE_RECORD_BYTES` in the generator; host side too).
- `npu_opus.rs` `rows()==256` gate + M-sized GTT buffers (for the full encode).

## DONE — FFN un-pinned + verified bit-exact (2026-07-15)

Host `resident_ffn_w8.rs` batch-parametrized (canonical path): added a `batch`
field parsed from `shape.txt` `m=256*batch`; scaled buffers, weight-block counts,
the two `0..3` macro loops (→ `0..3*batch`), and `down_block_index`'s gate offset;
block-index formulas and per-macro weight packing were already batch-agnostic.
M256 preserved (const API + logic unit test unchanged; parity example still
cosine 0.99985). New example `npu_resident_ffn_w8_batched_verify` runs M512 with
`[X;X]`:
- **M512 doc0 == doc1 == M256(X), BIT-EXACT** (cosine 1.00000000, max_abs 0.0 on
  all three). The batched FFN is numerically correct.

## KEY FINDING — naive FFN batching does NOT amortize the floor

Timing: **M256 = 10.15 ms, M512 = 19.04 ms → 1.07× throughput** (2×-linear = 20.3
ms). The FFN is **row-linear, not floor-dominated** (fit: ~1.3 ms fixed +
~0.035 ms/row). Reason: `upload_weights` **replicates the weight tiles per macro**
(same data in every macro slot), so the FFN's bottleneck — weight DMA (~960 MB/s,
NPU-RESULTS profiling) — **scales with rows**: M512 streams 2× the weight bytes.
Batching alone therefore buys almost nothing for the FFN.

**The batching win is gated on WEIGHT REUSE across macros** — stream each weight
tile ONCE and broadcast it across the row-macro loop (the STEEL K/V-broadcast
pattern, applied to FFN weights), instead of re-streaming per macro. That makes
weight DMA fixed while compute grows with rows → real amortization. This is a
kernel/generator change (`r26_gen.py` weight objectfifo + `r26_w8_resident_ffn.cc`
+ `upload_weights` single-copy layout), not a host-const change.

## SPIKE — weight-broadcast via BD-repeat FAILED (2026-07-15)

Tried the low-risk version: keep the core loop untouched, read one macro's weight
copy from DRAM and replay it across macros with a stride-0 outer DMA dimension
(host `%W` left full so copies are identical → isolates the DMA change). Findings:
- A `dma_configure_task` allows **exactly one `aie.dma_bd`** → gate/down replay
  must be two separate tasks (fixed).
- The **highest DMA dim is the BD repeat count**; transfer length must exclude it
  (fixed → length = one macro's copy).
- After both fixes it **compiles** (xclbin identical, core untouched) but
  **produces all-zero output** (cosine NaN): the BD-repeat replays the DMA read
  but does **not** push the replayed WB blocks as separate objectfifo elements, so
  the depth-1 per-`WB` `@wbc` acquires the core does never receive the reused
  weights. The stride-0 BD-repeat does not map to the objectfifo's per-element
  acquire semantics. Reverted (env-gated spike removed; generator back to clean,
  batch path intact and byte-identical at B=1).

**Conclusion:** weight reuse across macros needs a real objectfifo/dataflow
restructure, not a DMA-pattern tweak. Two viable paths remain:
- **(a) stationary-weight core loop** — hoist weight acquire out of the macro
  loop; needs M_MACROS `gacc` accumulators (~27 KB) or an accumulation reorder.
- **(c) memtile-resident weight objectfifo** — a `@wbc` whose memtile buffer holds
  one macro's blocks and is *consumed M_MACROS times* (objectfifo with a
  produce-once/consume-N or `repeat` at the objectfifo level, not the BD level).
  Study mlir-aie objectfifo broadcast/repeat semantics (STEEL's K/V broadcast in
  upstream `amd/iron` is the reference pattern) before the next attempt.

## PATH (c) DE-RISKED — iter_count memtile replay CONFIRMED (2026-07-15)

Built a compute-free IRON probe (`scratchpad/iter_probe.py`): shim→memtile
(`set_iter_count(1)`) → forward (`set_iter_count(M)`) → drain, on NPU2Col1.
Generated with the from-source `build/mlir-aie/install/python`; `aiecc` compiles
it clean. The **lowered memtile DMA (`input_with_addresses.mlir`) is the ground
truth** and proves the mechanism:
- IRON `set_iter_count(N)` on the memtile→consumer fifo lowers to
  **`aie.dma_start(MM2S, ..., repeat_count = N-1)`** on the memtile DMA.
- The **S2MM (L3→memtile load) has NO repeat — runs once**; the MM2S replay reads
  the **memtile-resident buffers** (`s2m_cons_buff_0/1`). So **L3 is read once, the
  memtile replays on-chip** — the exact bandwidth win.
- The replay iterates the **BD chain** (buffer *sequence*), i.e. `[buf0,buf1]×N` —
  **sequence order**, matching the FFN core's `[gate-seq]×M_MACROS` consumption
  (NOT per-object order, which is what `repeat_count` on a per-block fifo gives).

Root cause of the failed spike confirmed: it put `repeat_count` on the **shim**
BD (illegal — "unavailable for shim tiles") instead of the memtile MM2S.

**Recipe for r26 (hand-written MLIR):** put the replay on the `@wbc` (memtile→core)
objectfifo, not the shim. In raw MLIR that is `repeat_count = M_MACROS - 1` (or the
objectfifo `{iter_count}`/`{repeat_count}` attr) on the memtile side of the
gate/down weight chains, with `@wsh` loading one macro's copy once and the memtile
holding it resident. Single-copy host `%W` (28 blocks/col). Two chains (gate 18,
down 10) each replayed M_MACROS times, in the core's consumption order.

## WEIGHT-BROADCAST: fully mapped, blocked by r26 memtile tiling (2026-07-15)

Complete obstacle tree from the implementation attempts (all reverted; generator
clean, `--batch` intact; the attempt is saved at
`scratchpad/r26_gen.broadcast_attempt.py`):
1. Host replicates weights per macro → need single-copy `%W`. (solvable)
2. Shim BD-repeat replay → **illegal** ("repeat_count unavailable for shim tiles").
3. Two shim→memtile loads (@wshg+@wshd) → **shim output channel limit (2)**: shim
   already has @wsh+@xsh.
4. One shim load + memtile split-link to @wbcg/@wbcd → **memtile INPUT channel
   limit (6)**: mt0–3 are SHARED between weights (col) and activations (row) and
   are already at 6 inputs (@wsh+@xsh+4×@oc); the split-link + iter_count needs one
   more.

**Root cause:** the r26 tiling packs weights AND activations onto the same
memtiles (mt0–3), which sit at the DMA-channel limit. The iter_count replay (proven
correct as a mechanism) needs a channel those shared memtiles don't have. So the
optimization can't slot into r26 as-is.

**What it would take (a real redesign, not a patch):** free a memtile channel on
mt0–3 — e.g. re-tile so weights and activations use *different* memtiles, or reduce
the 4×@oc core-output fan-in, or the [gate,down]×M_MACROS core interleave (one
weight fifo → no split, but needs the gate→T→down per-macro pipeline + on-chip T).
Each is a fundamental FFN-dataflow change. The ~56% weight-DMA reduction is real,
but the implementation cost is a dedicated kernel-redesign effort.

### (earlier note: two-fifo split — blocked on shim, superseded above)

Implemented path (c) in `r26_gen.py` (env `HIPFIRE_R26_WEIGHT_BROADCAST`, canonical
non-reuse): split the weight fifo into `@wshg`/`@wbcg` (gate, memtile-resident 18
blocks, `iter_count=M_MACROS`) + `@wshd`/`@wbcd` (down, 10 blocks), rewired the core
gate/down acquires (`@wbc`→`@wbcg`/`@wbcd`), single-copy runtime_sequence load, and
shrank `%W` to 28 blocks/col. The MLIR generates correctly (`@wshg` 294912 B
resident, `iter_count=3`, `%W` = COLS·28·WB) and OFF stays byte-identical.

**But aiecc rejects it: `aie.tile op number of output DMA channel exceeded`.** The
gate/down split needs TWO memtile output channels on top of `@osh`/`@xbc`, over the
shim/memtile DMA-channel budget. This is a hardware limit, not memory. The split is
forced because the core consumes `[gate×M_MACROS]` then `[down×M_MACROS]` — two
replay sub-sequences.

**The fix (next):** ONE weight fifo (one channel) + **interleave the core loop** to
`[gate,down]×M_MACROS` so a single 28-block sequence replays via one `iter_count`.
That means merging the gate/down `mblock` loops and holding the T-scratch per-macro
(down macro-m runs right after gate macro-m, reading its T). Bigger core restructure
but channel-neutral. Code left env-gated + a STATUS:BLOCKED comment; off == verified.

## Next

1. **Weight-reuse-across-macros** — one-fifo + gate/down core interleave (the
   two-fifo path is proven to exceed DMA channels). Mechanism (iter_count memtile
   replay) is de-risked; the remaining work is the core-loop interleave. Re-time M256 vs M512 — this
   is the actual throughput lever. Gate: `batched_verify` bit-exact + canonical
   `_verify` absolute reference (self-consistency alone is insufficient — the
   broadcast spike passed doc0==doc1 trivially by outputting all zeros... it did
   NOT; it failed cosine=NaN, but a subtler bug could pass self-consistency, so
   always gate on the absolute reference check too).
2. Extend the host batch path to the **direct-x** mode (inverse-table records
   scale with batch) for the real resident encode.
3. `npu_opus.rs` `rows()==256` gate + M-sized GTT buffers → full batched encode;
   benchmark B∈{1,32,128,256}.

## R123 correction — object-FIFO replay rejected; core-stationary path correct but slower

The earlier “iter_count memtile replay CONFIRMED” claim was compile/lowering
evidence, not a hardware correctness result. The full one-FIFO implementation
showed that the memtile MM2S repeat re-acquires the source consumer lock for
every sequence, while shim→memtile S2MM produces that lock only once. With the
normal prime dispatch the measured command returned all zeros. Skipping prime
as a discriminator delivered only the first M macro (parity cosine 0.61336109;
later rows zero). Seven 64 KiB source buffers behaved the same as one 448 KiB
buffer, ruling out bank span. The alternate object-FIFO `repeat_count` attribute
expanded BDs and exhausted the memtile's 24-BD limit.

R123 then implemented real core-stationary reuse: each 15,552-byte weight record
is acquired once while every M96 row macro passes through it, with one f32
accumulator per macro. M512 required two three-macro accumulator buffers and a
rolling three-output DMA-task window; queuing all six gate / twelve down outputs
returned zeros. The final kernel passes the absolute M256 oracle (cosine
0.99984681) and M512 `[X;X]` is bit-exact against M256 for both documents.

The performance gate rejects it: stationary M256/M512 measured **20.111 / 33.500
ms** (1.20x row throughput), versus **10.520 / 19.929 ms** for the existing
replicated-weight path. Weight-major accumulation and routing cost more than the
saved DMA. Do not admit R123 as the runtime default. Preserve it as the bounded
negative result and move to next-prep/tail context consolidation before further
FFN redesign. Durable rows:
`benchmarks/npu_gemm_tuning/results/r123-weight-stationary-m-scaling-20260715.csv`.

## Bounded component characterization (2026-07-15)

Before un-pinning more stages, the active M256 resident route was instrumented
at every decision-relevant boundary. `HIPFIRE_EMBED_TRACE_RESIDENT=1` now splits
attention into prepare / GPU pack+sync / NPU run; splits the post-layer boundary
into next-activation prep / residual prep / host materialization; and records
final norm/mean separately from Dense/L2. The trace summarizer converts repeated
encodes into cold/primed layer and sample CSVs:

- `benchmarks/npu_gemm_tuning/results/embeddinggemma-resident-components-m256-20260715.csv`
- `benchmarks/npu_gemm_tuning/results/embeddinggemma-resident-samples-m256-20260715.csv`

Protocol: one fresh process, `EmbeddingGemma-300M.npu.oq8.hfq`, one 256-token
document, three encodes, resident completed-layer route, both resident and XDNA
dispatch tracing enabled. The two primed encodes averaged:

| Component | Mean per encode | Share of traced layer + finalization time |
|---|---:|---:|
| FFN | 269.042 ms | 35.0% |
| Attention total | 212.178 ms | 27.6% |
| — attention prepare | 11.540 ms | 1.5% |
| — attention NPU run | 200.629 ms | 26.1% |
| Next-layer activation prep | 168.409 ms | 21.9% |
| Tail | 86.612 ms | 11.3% |
| Per-layer setup | 8.322 ms | 1.1% |
| Final Dense/L2 | 18.392 ms | 2.4% |
| Final norm/mean | 2.575 ms | 0.3% |
| Final-layer host materialization | 1.992 ms | 0.3% |

The active route has no separate unit-RMS or residual-prep dispatch, and no GPU
activation pack on interior layers; their zero trace values mean *bypassed*, not
zero-cost alternate kernels. Layer work averaged 746.822 ms and finalization
20.976 ms. The XDNA split reported 98 dispatches, **1.894 ms submit** and
**744.854 ms wait** per primed encode. This independently confirms that host
launch batching remains the wrong lever.

The M256/M512 FFN check was repeated under the same lock: both M512 documents
remain bit-exact against M256, while 10.520 ms → 19.929 ms gives only **1.06x
throughput**. That confirms the earlier result rather than a one-off timing.

### Decision

The useful characterization is complete enough to proceed, but it rejects a
blind “un-pin every stage” implementation order:

1. **FFN:** do not extend to larger M until weights are reused across row macros;
   current batching duplicates weight traffic.
2. **Next prep + tail:** characterize/un-pin together or fuse their context
   transition. They consume about one third of the active critical path and are
   the cleanest remaining fixed-work candidates.
3. **Attention:** pursue segmented/block-diagonal batching after the boundary
   work; almost all attention time is the NPU run, not host preparation.
4. **Final norm/Dense:** leave pinned until the main layer path scales; together
   they are under 3% of the traced primed encode.

Stop the characterization campaign here unless a candidate changes one of
these boundaries. For each candidate, require a fresh-process correctness run,
a primed timing run, and an M-scaling row in the durable CSV rather than trying
to reverse-engineer every inactive NPU path.

## R124 — direct-X resident FFN batch contract admitted (2026-07-15)

The real resident handoff is now unpinned through M512. Direct-X cannot pack
two documents as 512 contiguous physical rows: the inverse selector maps only
one M256 record set, so physical rows 256-287 would alias rows 32-63. Each
document instead occupies one M288 slot (256 logical rows plus 32 padding
rows), followed by its own 32-record inverse-RMS plane. The host now packs and
decodes that document-padded layout; canonical M512 input allocation was also
corrected from one M288 buffer to two.

Both r55 direct-X/gate-reuse images compile, and the hardware verifier gates
against both an absolute canonical BF16x2 reference and duplicated-document
self-consistency. M256 reaches cosine `0.99988810`, maximum absolute error
`0.0147171`, and mean absolute error `0.00238566` against canonical. M512
`[X;X]` is bit-exact against direct M256 for both documents and between
documents. Timing is `8.366 ms` at M256 and `16.489 ms` at M512, only `1.01x`
row-throughput scaling, so this admits the ABI/correctness seam rather than an
FFN batching performance win.

The next implementation boundary is the paired post-FFN tail and next-layer
activation preparation. Unpin or consolidate that transition before segmented
attention. Durable rows:
`benchmarks/npu_gemm_tuning/results/r124-direct-x-batched-ffn-20260715.csv`.

## R125 — post-FFN tail and next-prep unpinned through M512 (2026-07-15)

Both sides of the cross-layer boundary now accept document-padded batches. The
tail initially exhausted shim BD IDs when M512 doubled its phase tasks. The
admitted schedule retains the original four tasks and adds a document DMA
dimension plus `repeat_count`, so descriptor count stays fixed while each core
consumes eight phase objects. M512 preserves the absolute tail oracle (cosine
`0.99999861`, maximum absolute error `0.0039062`) and the two documents are
bit-exact. Timing improves from `0.381335 ms` to `0.600561 ms`, a `1.27x`
row-throughput gain.

R111 next-prep repeats its fixed eight-row local transaction per document in
the same runtime sequence and places each document's R34 prefixes in a separate
2,949,120-byte output region. M512 produces byte-identical document outputs;
the known rounding envelope doubles from five to ten one-code differences,
with maximum Q delta 1 and scale error `7e-9`. Timing is `5.0606/10.0303 ms`,
only `1.01x` row-throughput scaling. The prep math, not its dispatch floor,
dominates this standalone boundary.

This closes the M512 tail/prep ABI but does not yet wire a full batched layer;
the next dependency is segmented/block-diagonal attention. Durable rows:
`benchmarks/npu_gemm_tuning/results/r125-batched-tail-next-prep-20260715.csv`.

## R126 — block-diagonal BF16 attention seam admitted (2026-07-15)

The standalone R27 attention graph now processes multiple 256-token documents
in one dispatch while resetting online-softmax state for every four-query group
and replaying only that document's sixteen K/V blocks. The first M512 layout
concatenated already-packed Q buffers behind one row descriptor. That was
wrong: R27 physical Q is row-major outside the query-group dimension, so the
second six groups came from another row rather than the second document. Both
document cosines fell to about 0.99867 and the layout was rejected.

The admitted schedule uses explicit Q, K/V, and output descriptors for each
document. Two deliberately different documents independently match their CPU
bidirectional-attention oracles: doc0 cosine `0.99999410`, maximum error
`0.0002527`; doc1 cosine `0.99999464`, maximum error `0.0002543`. Aggregate
cosine is `0.99999435`, proving that no cross-document scores enter either
softmax. M256/M512 timing is `1.4303/2.8932 ms`, essentially row-linear
(`0.99x` row-throughput scaling).

This proves the segmentation rule and descriptor topology, not yet the full
resident layer. The next step is to apply the same per-document Q/K/V schedule
inside the fused QKV/projection/norm attention image, then expose runtime batch
rows and segment offsets. Durable rows:
`benchmarks/npu_gemm_tuning/results/r126-segmented-bf16-attention-20260715.csv`.

## R127/R128 — fused segmented attention and full M512 encode admitted (2026-07-15)

R108 now accepts `--batch=N` by replaying its complete B1 descriptor schedule
inside one runtime command. The core program stays M-agnostic, Q/K/V scratch is
private per document, and each online-softmax pass consumes only that
document's sixteen K/V blocks. B1 generated MLIR remains byte-identical.

The first fused M512 layout made each document own a complete 5.1 MB R34 scratch
image. Its attention output was correct in isolation, but the next direct-X FFN
expects a compact shared prefix: all padded X documents followed by one inverse
plane per document. Full-model parity exposed this ABI mismatch. The admitted
layout keeps per-document scratch after the handoff prefix and scatters only the
final X/inverse records into the established direct-X format.

Two deliberately different documents are bit-exact against separate fused M256
hardware runs for both X and inverse metadata. Fused attention measures
`6.405/11.615 ms` at M256/M512, a `1.10x` row-throughput gain.

The architecture runtime now selects matching M-sized attention, dense-W8 FFN,
tail, and next-prep caches with `HIPFIRE_EMBED_NPU_BATCH`; rejects mismatched
component geometry; accepts explicit `[0,256,...]` segment offsets; and pools
each segment independently. Unsupported projectors or segment shapes retain
the existing sequential-document path. Since final norm/mean is still M256,
batched final state is materialized once and final-normed/pooled per segment on
the GPU.

End-to-end oq8 results for two distinct 256-token documents:

- separate M256 hardware references versus one M512 command: mean cosine
  `0.99999845`, minimum cosine `0.99999797`, maximum absolute error `0.00025596`;
- ten reused M512 commands retain exactly the same final parity;
- three-run timing is `774.491 ms/document` at M256 versus
  `633.625 ms/document` at M512 (`1267.250 ms` per two-document command), a
  `1.22x` row-throughput gain, not the originally estimated 60x.

The implementation proves useful whole-encoder batching and a modest measured
win. It also rejects the original fixed-floor model: projection, FFN,
next-prep, and attention remain substantially row-linear because weights and
activation schedules are replayed per document. Larger-B promotion should be
driven by weight/dataflow reuse, not descriptor replication alone. Durable rows:
`benchmarks/npu_gemm_tuning/results/r127-fused-segmented-attention-20260715.csv`
and
`benchmarks/npu_gemm_tuning/results/r128-full-batched-encode-20260715.csv`.
