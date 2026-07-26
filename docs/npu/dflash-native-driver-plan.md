# DFlash native NPU driver — close the wall-clock gap

Run the DFlash block body's dispatch sequence from **native Rust** instead of Python,
and measure the real block wall. This is the last lever for the plan's latency goal.

## Why (measured, not assumed)

| term | measured |
|---|---|
| per-dispatch overhead, Python/XRT | ~5300 µs |
| **per-dispatch overhead, native DRM** | **169 µs** (31× less) |
| NPU compute for the whole body | 23 ms |
| body dispatches (current) | 42 |

Projection: 23 ms compute + 42 × 169 µs ≈ **30 ms/block vs the 57 ms verify budget**
(1.9× under). So the latency goal is reachable **without further fusion** — the
remaining ≤3-dispatch/layer target was a proxy for latency, and the driver addresses
latency directly. The 30 ms figure is a projection from two separately-measured terms
(it slightly double-counts compute inside the dispatch number); **this task replaces
it with a real end-to-end measurement.**

## Foundation (already exists — do NOT write a new runtime)

`crates/hipfire-xdna`:
- `submit.rs` — direct amdxdna DRM ioctls (`EXEC_CMD` + syncobj wait on
  `/dev/accel/accel0`). No XRT, no Python.
- `kernel.rs` — `NpuKernel::load(xclbin_bytes, insts_bytes)`, `alloc_arg(size)`,
  `dispatch(&[&buf, ...])`.
- `xclbin.rs`, plus `gemm*` / `segmented_attention` / `qwen3_projection` modules.
- Examples to copy from: `npu_cascade_time.rs` (prep-once + timed dispatch loop),
  `hwctx_smoke.rs`, `npu_busy.rs`.

`crates/hipfire-npu` — probe/admission/inventory.

## The artifact-plumbing problem (the non-obvious part)

The body uses kernels from TWO sources with different layouts:

1. **Primitive xclbins** — `target/npu/{name}.xclbin` + `{name}-instr.bin`
   (rmsnorm, hnrope q/k, swiglu, and the `-b<rows>` batched variants). Direct to load.
2. **`@iron.jit` kernels** (int8 projection `int_matmul`, attention
   `dflash_attn_head` / `dflash_attn_all`) — these live in the JIT cache at
   `~/.npu/cache/<hash>/{final.xclbin,insts.bin}`, keyed by a hash of the design +
   CompileTime args. The native driver cannot compute that hash.

**First step: make the Python harness emit an artifact manifest.** Add a dump mode to
`tools/npu/dflash_body_npu.py` that records, for every dispatch it issues: the op
name, the resolved xclbin+insts paths, the buffer sizes and their order, and the
CompileTime shape args. That manifest is the contract the native driver consumes —
it removes all guessing about which cache dir is which kernel and what arg order each
expects. (`aie.utils` exposes the resolved artifact path for a jitted design; if not,
snapshot `~/.npu/cache` before/after a run and diff.)

## Build order

1. **Manifest dump** from `dflash_body_npu.py` (above). Verify each listed xclbin
   loads via `NpuKernel::load`.
2. **One op native** — pick the int8 projection. Feed it the SAME quantized inputs
   the Python path used (dump them to `.npy`), dispatch natively, and compare the
   int32 output **bit-for-bit** against the Python result. This proves buffer layout
   and arg order before anything is chained.
3. **Full body sequence** — chain all 42 dispatches natively, weights uploaded ONCE
   and kept resident (`alloc_arg` + fill once, reuse across dispatches and across
   blocks). Host-side glue (per-row int8 quant, rescale, residual adds, softmax
   pieces that currently live in numpy) must be ported or staged — keep it in Rust,
   f32, mirroring `dflash_body_npu.py` exactly.
4. **Measure + validate** — report cold and warm block wall, per-dispatch mean, and
   NPU-busy; validate the final `block_hidden` against the Phase-A golden with the
   same cosine gate (> 0.99 vs golden AND vs the int8/bf16 precision reference).

## Validation (non-negotiable)

Same honest gate as Phases C–E: cosine vs the f16 golden AND vs the int8/bf16
precision reference. Bit-exactness is expected for the integer GEMM steps — if the
native path differs from the Python path on the same inputs, that is a bug, not
precision. Do not loosen tolerances.

## RESULTS (measured on nix1, 2026-07-18) — budget NOT met, premise was wrong

The driver is built and validated; the projection above is superseded.

| | Python/XRT | native |
|---|---|---|
| warm block wall | 1164 ms | **726 ms** (cold 712 ms) |
| dispatches | 68 | 68 |
| parity vs f16 golden | 0.998092 | **0.998114** |
| parity vs int8/bf16 ref | 0.998147 | **0.998170** |

The native driver removes ~440 ms of host/XRT overhead per block (1.6×) and is
numerically correct. It does **not** reach 57 ms — it misses by 12.7×.

**The 30 ms projection was wrong in its premise, not its arithmetic.** It
assumed per-dispatch overhead dominates. It does not. Attribution, each term
measured with a probe that pins the kernel so there is *no* context churn:

| term | per block | share | evidence |
|---|---|---|---|
| GEMM weight streaming | ~317 ms | 55% | linear in weight bytes: 101 MB→26.5 ms, 50→13.6, 25→7.2, 17→5.3 (~3.8 GB/s); identical pinned vs churning ⇒ device work |
| attention compute | ~236 ms | 33% | `dflash_attn_all` = 37.5 ms standalone on 131/197 KB buffers |
| host glue (quant, bf16, packing) | ~143 ms | 20% | wall − dispatch time |
| primitives (norm/rope/swiglu) | ~24 ms | 3% | 0.35–1.22 ms each |

The 169 µs per-dispatch figure is real and reproducible (163 µs for the 1-row
`qwen35-rmsnorm-4096`, 308 µs for the `-b16` batch), but it is the **overhead
floor for a tiny kernel**. The body's expensive dispatches are dominated by
actual device work, so that floor never governed the block wall.

`int_matmul` re-streams the entire int8 weight from DDR on **every** dispatch —
"resident" keeps the buffer allocated, it does not keep the weight on-chip. At
16 activation rows the GEMM is pure weight bandwidth, and ~1.09 GB/block at
~3.8 GB/s is ~290 ms no matter how the dispatch is issued.

**UPDATE 2026-07-18 — the better GEMM is built and measured (not yet wired in).**
A multi-core W4A8 GEMM at DFlash shapes on npu1 measures **9.4 GB/s on the weight
stream vs the single-core `int_matmul`'s 3.8 GB/s (2.45×)**; int4 halves the bytes for
the same logical weight (2×, licensed by the W8-vs-int4 runs measuring 9.81 vs 9.42
GB/s — byte rate is ~precision-independent). Composed: **≈4.9×**, projecting the GEMM
term to **~51–58 ms, down from ~317 ms**. Artifacts: `~/.hipfire/npu/r14_1x2x128_nb128`
and `~/.hipfire/npu/r14_1x4x64_nb128`.

Correctness, all exact: C[0] = 2048 / 1024 / 1024 / 17408 against expected
2048 / 1024 / 1024 / 17408 (last is the W8 run, K=1024 × byte 0x11 = 17).

**This is a projection, not an end-to-end result.** It assumes essentially all of the
measured 317 ms was weight streaming — defensible, since that term was measured linear
in weight bytes at ~3.8 GB/s, but any surviving fixed per-dispatch overhead lands the
real number higher. The full-body parity gate has NOT been run against this kernel.

### The kernel is done; the remaining headroom is ~10%, and it is not in this dataflow

An r132 feed-only probe (no compute, no activations — `benchmarks/npu_gemm_tuning/r132/`)
isolates the weight path:

| config | channels | W-path GB/s |
|---|---|---|
| STREAMS=1 | 4 MM2S (1/col) | 10.02 |
| STREAMS=2 | 8 MM2S (2/col) | 10.37 |

**Doubling weight-fetch channels bought +3.4%** — channel count is NOT the limiter, and
the 8-channel variant allocated cleanly in aiecc (retiring r12's "number of output DMA
channel exceeded" as a concern at this shape). Independently: feed-only 837.4 µs vs the
full GEMM 890.5 µs — stripping *all* compute and activation traffic recovers only **6%**.

So the GEMM runs at ~92% of the *weight-path* feed ceiling. The limiter sits **upstream
of the shim MM2S channels** — DDR/NOC path into the column, or memtile write bandwidth.
**Which of the two is NOT determined**; the distinguishing test (shim→core direct feed,
one core/column, bypassing the memtile) was not run.

**Aggregate vs weight path — do not conflate them** (this corrects an earlier claim in
this file that the 13–16 GB/s figure was "not reachable through this dataflow"):

- **Aggregate IS reachable**: the npu1 GEMM measures **14.1 GB/s** (12.71 MB / 899.3 µs).
- **The weight path alone is NOT**: capped at 10.0–10.4 GB/s across six measured knobs.

Provenance caveat on that 13–16 GB/s figure: both npu1 documents that cite it
(`r11/README.md:37`, `r12/README.md:19`) source it from **R6, which is aie2p-only**
(`r14/r14_cache.sh:2`). The value happens to be about right for npu1 aggregate, so this
is a sourcing problem rather than a wrong number. The highest directly-measured *npu1*
aggregate in the tree before this work was r12's 11.7 GB/s. Genuinely non-transferable:
`docs/npu/npu-memory-bandwidth-cache-characterization.md:4` states `Host: halo`, so its
14.4 GB/s per-stream and 56.5 GB/s aggregate figures are npu2 and must not be applied
to nix1.

### OPEN: a locality anomaly that could move the GEMM term ~1.5×

Reads are **faster split across buffers than drawn from one region** (reads only; the
0.13 MB of C writes is excluded, so this is not a direction artifact):

| | read bytes | time | read GB/s |
|---|---|---|---|
| `r14_1x2x128_nb128` GEMM (A + W, separate buffers) | 12.58 MB | 899.3 µs | **13.99** |
| `r132` W-only (8 channels, one contiguous 8 MB region) | 8.39 MB | 809.1 µs | **10.37** |

The GEMM pulls 50% more read bytes at 35% higher read bandwidth than the dedicated feed
probe. If the ~10.4 GB/s "weight ceiling" is DRAM bank/page locality in the probe rather
than a real wall, then laying weights across distinct regions could approach ~14 GB/s and
take the DFlash GEMM term to **~35–39 ms instead of ~51–58 ms**.

Settling measurement (NOT yet run): rebuild the r132 W-only probe with each of the 8
channels reading a **distinct buffer region**, total held at 8 MB. `> 10.4 GB/s` ⇒ weight
layout is a live lever and the projection improves materially; `≈ 10.4 GB/s` ⇒ the wall is
genuine and the GEMM's higher aggregate comes from A and W using independent paths.

**Until that runs, treat 9.4 GB/s and ~51–58 ms as provisional.**

### NOISE FLOOR — read this before trusting any delta above

**Run-to-run variance on an identical binary is 3.4%** (the same 8-channel/4-core/burst-0
build measured 809.1 µs and 836.6 µs). Single-shot deltas below ~5% in this document are
therefore **not meaningful**, including several reported earlier as findings:

| earlier claim | verdict |
|---|---|
| channel count 4→8 = "+3.4%" | **noise** — strengthens the null: extra weight channels buy *nothing* |
| halving MACs = "+4%" | at the noise boundary |
| burst 0→64 = "+3.5%" | at the noise boundary |
| halving activation traffic = "+1%" | noise |
| buffer depth 3 vs 2 = "+0.04%" | noise |

Clearly outside noise: burst=256 (**−8%**), and the int4-vs-int8 shape difference.
Repeats should have been run before reporting deltas at this granularity.

### Knobs measured and retired (seven, all null or noise-level)

channel count · consumer count · buffer depth · compute load · activation traffic ·
shape · burst length. **The weight path holds 10.0–10.6 GB/s throughout.**

`burst_length` was previously left at its default (0). Best of four is 64 (10.56 GB/s);
larger is actively worse (256 → 9.44 GB/s, −8%). The DMA preferring many small bursts is
consistent with a **fabric/arbitration limit rather than a DDR-page-efficiency limit**.

n-D DMA auto-advance *is* in use (3-D `dma_bd` with hardware stride/size advance, no host
re-issue per block), but degenerately: **two of three dimensions are consumed expressing
contiguity**, because `_split()` exists solely to work around npu1's 1023-per-dimension
cap (a 16 KB contiguous run encoded as 32×512 B). Only the block stride does real work.

### Activation-stationary: BUILDS, and the fanout control is decisive

`benchmarks/npu_gemm_tuning/r14b/r14c_gen.py` — W pulled by **both** MM2S channels per
column and reassembled by a memtile **join**; activation held in a core-resident buffer.
Verified in lowered IR: `wsh{0..3}_{0,1}_shim_alloc(MM2S, 0)` **and** `(MM2S, 1)` on all
four shims, zero `ash` allocations.

| config | W chans | A path | cores | C[0] | W GB/s |
|---|---|---|---|---|---|
| R14 control | 4 | 2 MB streamed | 16 | 1024 ✓ | 9.5 |
| R14C 1 ch/col | 4 | resident | 16 | 3072 ✓ | 9.8 |
| R14C 2 ch/col | 8 | resident | 16 | 3072 ✓ | 10.1 |
| R14C 2 ch/col | 8 | resident | **4** | 3072 ✓ | **10.1** |
| R14C 2 ch/col, N_BLK=256 | 8 | resident | 16 | 3072 ✓ | 9.9 |

**The decisive control: 4 cores and 16 cores take the same time** at identical weight
bytes (827.2 vs 818.8 µs). Time is set purely by the DDR→shim weight stream — not compute,
not memtile broadcast, not the core side — and stays linear in weight bytes (2× W → 2.05×
time). Correctness exact throughout; resident-A configs use `AVAL=3` (not 1) so
C[0] = 3 × KT × 16 = 3072 **proves the resident buffer is actually read**.

Second npu1 wall found here: `ObjectFifoLinkOp does not support 'join' and 'distribute' at
the same time` — you get 8 weight channels **or** a streamed activation, not both.

**Revised projection: ~48 ms** (not the ~37 ms that 13 GB/s implied). More DDR streams is
retired as a lever. **The next lever must cut weight BYTES (deeper quant) or get more
activation rows per weight fetch (larger M) — or overlap dispatches.**

> DFlash consequence: since time is set by weight bytes and is **independent of activation
> rows**, draft tokens are close to free on the GEMM. This is now a measured argument for
> materializing the full 16-token block rather than truncating to ~8.

### RESOLVED: the distinct-buffer anomaly is not a lever — but BO SIZE is

`benchmarks/npu_gemm_tuning/r133/` swept region count with read bytes held at exactly
8,388,608 in every row. **At constant BO size, region count is flat:**

| BO size | 1 region | 2 | 4 | 8 |
|---|---|---|---|---|
| 57 MB | 5.84 | 5.98 | 5.96 | 5.99 GB/s |
| 15 MB | 9.70 | — | 9.68 | 9.70 GB/s |

Regions 1→8 moves nothing. Splitting across two separate buffer *objects* (4 MB in `%A` +
4 MB in `%W`) gives 10.26 vs 10.35 single-BO — also nothing. So the GEMM's 13.99 GB/s does
**not** come from separate BOs or from layout. **Nothing measured exceeded 10.4 GB/s**;
the weight-path wall is genuine. The residual is best explained by A and W riding
**independent paths**, so aggregate exceeds either stream alone.

**The real finding is a penalty, not a gain: larger weight BOs cost bandwidth.** The same
dense 8 MB read region measured at 8 / 15 / 57 MB BO gives **10.35 / 9.70 / 5.84 GB/s — a
1.77× penalty**. Cause not diagnosed (TLB/page-table pressure vs XRT allocation path vs
physical fragmentation).

> **⚠ THIS THREATENS THE HEADLINE NUMBER — VALIDATE BEFORE TRUSTING ~48–58 ms.**
> The new kernel was measured on **8 MB** weight buffers. Real DFlash weights are far
> larger (the native driver's own probe used up to **101 MB**). If the 1.77× large-BO
> penalty applies at DFlash sizes, the projection is optimistic and the 2.45× improvement
> is partly an artifact of comparing an 8 MB microbenchmark against a ~100 MB workload.
>
> Countervailing evidence: the *old* `int_matmul` measured **flat 3.8 GB/s from 17 MB to
> 101 MB** (101→26.5 ms, 50→13.6, 25→7.2, 17→5.3), showing **no** BO-size effect over that
> range. The two observations conflict and are from different kernels/paths.
>
> **RESOLVED — see below. The concern was inverted: the weight path IMPROVES at scale.**

### RESOLVED (r134): the BO penalty was a host cache-flush artifact, not bandwidth

`benchmarks/npu_gemm_tuning/r134/` + `crates/hipfire-xdna/examples/npu_bo_probe.rs`.

**Root cause:** `NpuKernel::dispatch` → `submit_synced(args, None)` flushes **every argument
on every iteration** (`crates/hipfire-xdna/src/kernel.rs:298-303`) — a full-buffer host cache
op costing time linear in **BO size**, independent of bytes the NPU reads. Implied flush
rate ~61 GB/s.

**Reads scaled to BO (what a real projection does):**

| W BO | sync=1 GB/s | sync=0 GB/s |
|---|---|---|
| 8 MB | 10.19 | 12.55 |
| 32 MB | 9.43 | 14.36 |
| **100 MB** | **11.58** | **15.14** |

> ### ⚠ CORRECTION 2026-07-19 — THESE ARE AGGREGATE, NOT WEIGHT-PATH
>
> **The table above is aggregate bytes ÷ time. It must NOT be used as a
> weight-path projection, and it was.** The `r14_selftest` harness
> (`crates/hipfire-xdna/examples/r14_selftest.rs`) separates them directly:
> **W-path 10.0–10.7 GB/s, aggregate 15.5–16.8 GB/s** on the same runs.
>
> That reproduces the ~10.4 GB/s weight ceiling stated **150 lines above this
> table**, in the section titled *"Aggregate vs weight path — do not conflate
> them"*. The ceiling has now been measured four independent ways — r132, r133,
> r135, and r14_selftest — and 15.14 is the outlier, because it is a different
> quantity.
>
> **Cost of the error:** 15.14 was propagated into
> `docs/plans/2026-07-19-dflash-phase0-brief.md`, the Phase 0 plan, and the goal
> prompt, producing a "~32–42 ms" GEMM projection. The measured result is
> **123.7 ms** (commit 98bbce9b6). Roughly ~60 ms of that is the genuine
> bandwidth floor (600 MiB packed W ÷ 10.4 GB/s) and ~62 ms is hardware-context
> contention — neither is recoverable by the dataflow tuning the projection
> implicitly assumed.
>
> **Rule:** when quoting a bandwidth number for a projection, state which stream
> it measures and prove it by varying the others. Aggregate ÷ time is not a
> ceiling for any single stream.

**Reads held constant from an inflated BO (r133's setup):** sync=1 gives 10.12 / 9.53 /
5.87 at 8/15/57 MB — a dead-on replication of r133's 10.35 / 9.70 / 5.84. Under sync=0 the
same arm is **flat** (12.41 / 12.54 / 12.54), proving the penalty is 100% host flush cost
and 0% DMA bandwidth.

**The old int_matmul and r133 were never in conflict.** With reads scaled to BO, flush time
and DMA time are both linear in size so the ratio is flat; r133 broke that by growing the BO
while holding reads constant, so only the flush term grew. Both behaviors reproduce from the
same kernel and allocator. (Caveat: int_matmul itself was not re-run.)

Correctness: every row gated at C[0] = 3072 = AVAL(3) × KT(64) × 16, exact — real compute
runs, not feed-only. Cross-process repeatability ~1%.

**Revised projection — the earlier number was conservative, not optimistic:**

| basis | rate | 482 MB int4/block | 964 MB int8/block |
|---|---|---|---|
| old `int_matmul` | 3.8 GB/s | 127 ms | 254 ms |
| r134 @100 MB, sync=1 | 11.58 GB/s | 41.6 ms | 83.2 ms |
| **r134 @100 MB, sync=0** | **15.14 GB/s** | **31.8 ms** | 63.7 ms |

**3.05× like-for-like, 3.97× with the flush removed** — and sync=0 is the correct model for
DFlash: `dflash_body_native.rs:407,411,472` already dispatches the GEMM as
`dispatch_synced(&[&gm.w, &gemm_b, &gemm_c], &[false, true, false])`, never re-flushing
weights. So the ~25% tax lives in the benchmark harness, not the driver. The driver's
remaining plain `dispatch()` calls (attention, rmsnorm, headnorm, rope, swiglu) act on small
buffers — attention's largest is ~197 KB ≈ 3 µs — so there is no large free win there.

**Net: the GEMM term projects from the measured ~317 ms to ~32–42 ms.**

Caveat to confirm before leaning on it: 15.14 GB/s weight-only plus 13.1 MB of C traffic is
~16.9 GB/s aggregate, at or slightly **above** the previously documented 13–16 GB/s band.
Also `burst_length` was left at 0 throughout, so these figures are likely ~4% pessimistic.

### CONFIRMED: npu1 is TOPOLOGY-limited, not bandwidth-limited (~10.3 per route, ~13 aggregate)

`benchmarks/npu_gemm_tuning/r135/` — the deciding test for the r133 dual-topology lead.
All feed-only, WB=8192, burst 64, sync=1, 5 passes each.

| config | topology | link | DDR read | delivered | median (spread) | GB/s on DDR |
|---|---|---|---|---|---|---|
| vert ×4 | single route | bcast, join 2→1 | 8 MiB | 32 MiB | 820.8 µs (2.2%) | 10.22 |
| vert ×4 | single route | **distribute** | 8 MiB | 8 MiB | 805.5 µs (2.9%) | **10.41** |
| horiz ×4 | single route | bcast | 8 MiB | 32 MiB | 809.6 µs (5.6%) | 10.36 |
| horiz ×4 | single route | **distribute** | 8 MiB | 8 MiB | 808.6 µs (5.4%) | **10.37** |
| vert+horiz | **two routes** | bcast/bcast | 8 MiB | 32 MiB | 639.1 µs (2.6%) | **13.13** |
| vert+horiz | **two routes** | **bcast/distribute** | 8 MiB | 20 MiB | 664.6 µs (1.2%) | **12.62** |
| vert+horiz | **two routes** | **distribute/bcast** | 8 MiB | 20 MiB | 673.2 µs (1.4%) | **12.46** |
| vert+horiz | two routes | dist/dist | — | — | **BUILD FAIL** | — |

**The 13.22 GB/s figure was computed on DDR read bytes, not delivered bytes** — the
broadcast-inflation worry is dead. r133's *vertical* fifo is also a 1-producer/4-consumer
broadcast, and r133 holds DDR constant across topologies by halving `NBLK` per route
(`r133_gen.py:89-93`); both rows are 8 MiB DDR / 32 MiB delivered.

**Distribute carries the lever.** Matched broadcast-vs-distribute pairs are null on DDR
bytes: vertical +1.2%, horizontal +0.1%, dual balanced −2.3% — all inside the 3.4% floor.
Distribute pulls 4× the DDR bytes at the same rate.

**Mechanism is concurrency of orthogonal routes, NOT that horizontal is faster** — horizontal
alone (10.36) ≈ vertical alone (10.29). Per-route wall ~10.3 GB/s; two concurrent routes
reach ~12.6–13.1 GB/s aggregate. This reframes the eight prior nulls: every one of them
varied a parameter *within* a single routing topology, which is why they were all null.

**Hard limit:** both routes distribute is unbuildable —
`'aie.tile' op number of output DMA channel exceeded!` (memtile needs 4+4 MM2S, cap 6).
**A dual-route design must keep one route broadcast.**

**Trap:** a naive `b/d` pairing measures 11.49 GB/s — that is **load imbalance, not a
distribute penalty**. The routes couple 1:1 per core iteration, so a distribute route pulls
4× its partner's bytes and the broadcast route drains early. `VREP=4` rebalances; a
two-phase model (13.1 shared, then 10.4 solo) predicts 737 µs vs 727 µs measured.

Corrections to earlier text: r133 cannot have run at WB=16384 (TOPO=2 fails L1 allocation);
the real shape is WB=8192/NBLK=256. r133's join asymmetry was a real confound — tested,
null (10.43 vs 10.29). The 32 MiB points carry 12–46% spread and must not be read as a size
trend.

**NOT measured:** compute (all feed-only, C[0]=0, no correctness meaning); composition with
r134's sync=0 flush removal (every r135 run is sync=1); concurrent activation/C traffic;
real GEMM shapes; ~100 MB weight scale; whether >2 routes or packet-switched routing lifts
the ~13 GB/s aggregate.

> **Composition is the open question.** At 8 MB/sync=1 both levers measure ~+25–29%
> independently (flush removal 10.19→12.55; topology 10.22→13.13). Whether they stack is
> **unmeasured and must not be assumed** — if they do, the result would exceed the
> documented 13–16 GB/s aggregate band and needs independent confirmation before use.

### ⚠ THE CORRECTNESS GATE USED THROUGHOUT IS STRUCTURALLY WEAK

Every validation in this effort checked **`C[0]` — a single element** (2048 / 1024 / 3072 /
17408). That is blind to whole classes of error: wrong per-column slice offsets, transposed
strides, dropped tail blocks — anything affecting elements 1..N but not element 0. The
results are not believed wrong, but **the evidence is thinner than "correctness exact"
implies.**

This matters most for exactly the dual-route/distribute designs above, where each core reads
a different quarter of a shared buffer via compile-time offsets — an indexing bug there would
hide perfectly behind `C[0]`. **Minimum fix before building the next variant: check the last
C element plus a canary past the buffer end.** Strengthen the gate *before* the next kernel,
not after.

### The surviving lever has a hardware mechanism: the BD iteration dimension

Our fully-lowered BD leaves **two of four levels of hardware pointer auto-advance idle**:

```
"aiex.npu.writebd"() <{ buffer_length = 262144, d0_size = 128, d0_stride = 0,
   d1_size = 16, d1_stride = 127, d2_size = 0, d2_stride = 4095,
   iteration_current = 0, iteration_size = 0, iteration_stride = 0, ... }>
```

- `d2_size = 0` — third traversal dimension unused (the innermost contiguous run folds
  into `buffer_length`, so `_split()`'s two dims occupy d0/d1).
- `iteration_size = 0, iteration_stride = 0` — the **iteration dimension is entirely
  unused**. This is design-guide lever #10, *"fuller weight-replay (`iter_count`/
  `repeat_count`) — one weight object feeds many activation macros."*

That matters because replay is the hardware expression of **the one lever seven null knobs
left standing: more activation rows per weight fetch**. It does not raise the ~10 GB/s
weight-path wall — it amortizes that wall over more useful work, which is the only
direction still open besides cutting bytes.

Caveats: the BD dump above is quoted from compiler output and has **not been independently
re-derived here**. And for DFlash specifically, replay pays only where multiple activation
blocks share one weight stream — a single sequence's block is already M=16 against a
one-shot weight fetch, so the win would come from batching sequences/blocks, not from the
existing per-block path. Scope before building.

*(A parallel branch proposed re-probing the distinct-buffer anomaly as motivation for this;
that anomaly was already resolved as a non-lever by r133 above — region count is flat at
constant BO size. The iteration-dimension finding stands on its own.)*

### r14b (activation-fold) — dead as written, not dead in principle

`benchmarks/npu_gemm_tuning/r14b/` folds A into the column shim object to free a channel.
It fails at **resource/channel allocation**:

```
aie.mlir:5:13: error: 'aie.tile' op number of input DMA channel exceeded!
    %c0_0 = aie.tile(0, 2)
AIECC COMPILATION FAILED
Error: Resource allocation pipeline failed
```

Compute tile (0,2) is given 4 inbound objectfifos (`wbc_j0`, `wbc_j1`, `abc_j0`, `abc_j1`);
an **AIE2 compute tile has 2 inbound DMA channels**. Fixable by joining the paired streams
memtile-side so each core sees one W fifo and one A fifo — r132 proves that topology
compiles. Not worth building: r132 already measured that rebalanced condition at +3.4%.

Measured nulls, all non-binding: buffer depth 3 vs 2 = +0.04%; halving MACs = +4%;
halving activation traffic = +1%. The **activation-stationary restructure was therefore
NOT built** — the +3.4% channel result removes its premise.

Two findings worth carrying forward:
- **The binding constraint is the weight path (~9.3 GB/s), not the ~13–16 GB/s
  aggregate DDR ceiling.** The control variant cut 1 MB of activation traffic and the
  time did not move (899.3 → 905.5 µs), so A and C ride concurrently on other channels
  and are effectively free. An earlier "we are at the DDR ceiling" reading was
  aggregate-bytes ÷ time — an artifact of one variant's traffic mix. The remaining gap
  to the ceiling is real headroom; **cascade (`aie.cascade_flow`, measured 4–10×
  elsewhere, unused in every shipped kernel) is untested on this path.**
- **Cores are ~12% utilized** (0.60 TOPS over 16 cores ≈ 18.75 GMAC/s/core vs r9's 150
  GMAC/s/core resident) — firmly feed-bound at M=16, as expected.

Remaining to convert this into a block-wall measurement: a host-side blocked A/W packer
matching r14's stripe layout, an **oq4 DFlash sidecar** (only the OQ8 one exists today),
and wiring into `dflash_body_native.rs`.

**Where the remaining gap actually lives:** a better int8 GEMM (multi-core,
reusing the streamed weight tile across the 16 activation rows — the current
design is single-core with its own tiling) and a multi-core attention kernel
(`dflash_attn_all` loops all 8 kv-heads on ONE core). Both are kernel work, not
driver work. Further *fusion* also will not help: it reduces dispatch count,
and dispatch count is 3% of the wall.

### Secondary finding: hardware-context budget

npu1 (Phoenix) admits only **six** concurrent hardware contexts (`NpuKernel::load`
returns EINVAL on the 7th — the same limit the Python harness's LRU-of-6 was
built around), while the body uses 12 distinct kernels per layer. The driver
therefore runs a pinned-anchor LRU. This turned out to be cheap and off the
critical path:

- `NpuKernel::load` ≈ **19.5 ms** — re-opens the DRM file and a 64 MiB heap.
- `NpuKernel::load_peer` ≈ **205 µs** — shares the anchor's file + heap.

At 62 misses/block via `load_peer` that is ~30 ms, ~4% of the wall. Argument
buffers survive eviction because they belong to the shared device, not the
context — which is what makes ~1.09 GB of resident weights compatible with
kernel churn.

### Artifacts

- `tools/npu/dflash_body_npu.py --dump-manifest|--dump-weights|--dump-op|--dump-ref`
- `crates/hipfire-xdna/examples/dflash_manifest_load.rs` (step 1, `--hold` probes the ctx budget)
- `crates/hipfire-xdna/examples/dflash_op_parity.rs` (step 2, bit-exact)
- `crates/hipfire-xdna/examples/dflash_ctx_swap_time.rs`
- `crates/hipfire-xdna/examples/dflash_body_native.rs` (steps 3–4, `--probe-gemm`/`--probe-attn`)

## Guardrails

- Hold the GPU/NPU lock while measuring: `./target/release/hipfire lock acquire
  dflash-native` / `... lock release`.
- The markov head does NOT belong here — it runs on one CPU thread via the top-k
  shortlist (8.7 ms/block, exact). Do not port it to the NPU.
- Keep the Python harness working; the native driver is an addition, and Python
  stays the reference for parity.
- graphify before grepping repo source.
- Report cold vs warm separately, and state plainly if the measured wall misses the
  57 ms budget and why (per-dispatch floor vs compute vs host glue).
