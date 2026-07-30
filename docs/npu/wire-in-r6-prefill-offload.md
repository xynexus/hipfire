# Wiring R6 W4A8 into the runtime prefill-offload path

## Why now

R6 (2D-tiled W4A8 GEMM, `benchmarks/npu_gemm_tuning/r6/`) does **20.7 TOPS of real
prefill on the halo NPU** — 4× the SOTA NPU inference stack (FastFlowLM ~5) and past
the 15.7 int8 reference — through hipfire's own XRT-free amdxdna dispatch
(`crates/hipfire-xdna`). Against the gfx1151 GPU's ~50 TOPS real W4A8, running NPU
prefill **concurrently** with GPU work is a genuine ~+40% aggregate win, not a
rounding error. The kernel is proven; this doc scopes making the runtime use it.

## The seam

The runtime issues quantized linears through `crates/hipfire-rdna/src/dispatch/quant.rs`
(+ `gemm_misc.rs`, `gemv.rs`). A W4A8 prefill linear is a GEMM `C[M,N] = A[M,K]·W[K,N]`
with M = prompt length (large), W = the oq4/mq4 weight. That is exactly R6's shape.
The offload decision belongs **at this dispatch seam**: for an eligible prefill GEMM,
route to the NPU instead of (or concurrently with) the GPU iu4 kernel.

## Build steps (each independently testable, none touches the hot path until step 4)

1. **Offline xclbin build (tooling, not hot path).** R6 is shape-specialized
   (MT/NT/KCHUNK compiled). Extend the `r6_gen.py` + `r6_build.sh` flow into a small
   offline tool: given a model's distinct `(K, N)` linear shapes, emit + cache R6
   xclbins (`~/.hipfire/npu/<shape>.xclbin`). Python/aiecc stays offline — the
   inference binary only *loads* cached bytes (AGENTS.md: no Python in production
   tooling; compat/build tooling lives outside the daemon).

2. **`NpuGemm` primitive (`hipfire-xdna`, isolated + tested).** A general
   `npu_gemm(M, K, N, a_i8, w_oq4, c_i32)` that: tiles M/N/K into R6's
   (MT·4)×(NT·16)×KCHUNK blocks, marshals A/W into the tile-major SHMEM layout the
   kernel expects, K-accumulates across KCHUNK chunks (add into C), and drives the
   dispatches via `NpuKernel`. Validate **numerically vs a CPU int8×int4 reference**
   on real (non-all-ones) data — the current benches only prove the all-ones ceiling.
   This is the unit the runtime calls; it carries no hot-path risk.

3. **Marshaling splits by static vs dynamic — and the layouts do NOT match the GPU.**
   Measured (`prepack_weights`/`run_packed`): CPU marshaling dominates end-to-end
   (0.02 TOPS). Verified against the GPU iu4 kernel (`fused_qkvza_oq4_wmma.hip`): it
   stores W as `[N_out, K/2]` nibbles + per-group f32 scales, loaded through the WMMA
   lane-distributed fragment; R6 wants `[K, N]` 16×16 aie2p tile-major raw int4. So
   **the buffer cannot be shared** — different orientation (transposed), different
   tiling (WMMA fragment vs aie2p mmul tile), and R6 applies **no scales**. Transposing
   either kernel to match orientation doesn't help: it only moves the transpose onto
   the *dynamic* activation, and the tiling still differs. Split:

   - **3a. Weights → the loader (static, once).** Produce an NPU-specific weight buffer
     at load: read GPU `W[n][k]` → write NPU `[k][n]` 16×16 tile-major + int4 pack. The
     `[N][K]→[K][N]` transpose is *absorbed for free* into the re-tile index mapping (no
     separate pass). Store it alongside the GPU copy for offloaded layers (4-bit, small).
     The hot path then DMAs it linearly — zero per-inference weight work. `prepack_weights`
     is the reference impl; move it into the loader / NPU-quantize path.
   - **3b. R6 scale handling (a real correctness item).** R6 is a raw int4×int8 GEMM;
     the `0/256` validation used *unscaled* int4. oq4 carries per-group f32 scales, so
     the NPU path must accumulate int32 then apply the group scale (at the tail, per
     group) to match the GPU. Design this with 3a (scales travel with the arranged W).
   - **3c (REVISED — the real fix). Reshuffle in-core via tensor buffer streams.**
     aie2p (AIE-ML/XDNA) has **multidimensional addressing**: `aie::tensor_descriptor`
     + `aie::make_tensor_buffer_stream(ptr, desc)` let the *kernel* read/write **row-major**
     memory in tile order — the address generators walk the 4D pattern in parallel with
     the vector MACs (free, and exactly the scalar/AGU-alongside-fixed-point point). The
     reference bf16 GEMM (`aie_api/detail/mmul.hpp` `group_mmul_page_multidim_gemm`) feeds
     **row-major `matA`/`matB`/`matC`**: `a_desc`/`b_desc`/`c_desc` 4D descriptors,
     `tsA >> Xbuff` / `tsC << C.to_vector()`, no marshaling. My R6 kernel used plain
     `load_v`/`store_v` (which *demand* pre-tiled memory) — that self-inflicted the whole
     marshaling problem. **Rewrite R6 to use tensor buffer streams for A/W/C.** Then: DMA
     stays linear; `NpuGemm` passes row-major A/W/C with **zero** marshaling (no CPU
     reshuffle, no memtile hop, no `prepack`); the shim-`dma_bd`/memtile experiments (3c
     above) become moot; the loader only needs the **scales** (3b), not a layout repack.
     This is the concrete next step and should recover the kernel's real throughput
     end-to-end. Still stream the whole GEMM in one dispatch for the ~78 µs latency.

     **DONE + MEASURED (r6_gemm_ts.cc, NpuGemm converged, npu_gemm_e2e).** The R6-TS
     kernel reads row-major A via a `[mt,k,m]` tensor stream (concat 4 rows → mmul
     64-vec) and writes row-major C via a `[mt,nt,m]` stream (`vector::extract<16>` per
     row); W stays pre-packed. Validated 0-mismatch (`r6_ts_verify` MT=8/MT=24;
     `npu_gemm_verify` single-core + array groups=4). `NpuGemm` now copies A/C row-major
     (no reshuffle) — `pack_a`→`load_a` block copy, `unpack_c`→per-group block copy.
     **End-to-end M768·K512·N4096: 0.02 → 0.254 (64 dispatches, 4 K-chunks) → 0.535
     (24 dispatches, KCHUNK=32 = one K-chunk so C is copied once not 4×) → 1.00 TOPS
     (software-pipelined + redundant-W-copy skip).** ~50× over the CPU-marshaling floor.
     **The marshaling wall is cleared.**
     - **Pipelining DONE.** `NpuKernel` split into `submit()`/`wait()` + a multi-slot cmd
       cache; `NpuGemm` double-buffers `c_buf` and submits dispatch *i* before reading
       *i-1* back, so the host C read-back overlaps dispatch *i* on the NPU. The
       flattened loop also skips the per-dispatch W memcpy when the slab is unchanged
       (single-K-chunk + one N-block reuses one W across all M-blocks). 6.02 → 3.22 ms.
     - **Input-sync skip DONE; C-sync is load-bearing.** `submit_synced(args, mask)`
       flushes only changed inputs (A every dispatch, W only when the slab is re-copied).
       The output-C `sync_bo` canNOT be skipped — it doubles as the read-back cache
       reconcile; without it the result is total garbage.
     - **Pipeline coherence bug found + fixed.** The first pipelined commit was
       intermittently wrong (~1 run in 3): the host read-back of one `c_buf` overlaps a
       concurrent DMA write to the other, and the CPU prefetcher can cache stale lines of
       the in-flight buffer with no invalidate before its later read. Fix: re-`sync_bo`
       the slot after `wait`, before read-back. `FROM_DEVICE` EINVALs on data BOs, but
       `TO_DEVICE` clean+invalidates on this driver. Now 0/16 reliable.
     - **M-parallel W-broadcast DONE + wins (`r6_gen_mp.py`).** Each of COLS cores
       computes a distinct M-block over full N, sharing ONE broadcast W (shim → memtile →
       all cores). W is read from DRAM once and one dispatch covers COLS M-blocks, so
       M768·K512·N4096 is **3 dispatches, not 24**. End-to-end **~1.45 TOPS** (2.0–2.5 ms,
       0/5 fail) vs 1.12 N-parallel — and it's *blocking*, so no pipelined-readback
       coherence dance. Raw single dispatch = 3.07 TOPS: feed-bound on the memtile's 8-way
       broadcast sync over 64 N-slabs, not compute. Standalone array + benches
       (`r6_mp_verify`, `r6_mp_e2e`); wiring an M-parallel mode into `NpuGemm` is the
       follow-up.
     - **Whole-GEMM-in-one-dispatch DONE + best (`r6_gen_mp.py` ROUNDS).** Each core
       streams ROUNDS M-blocks, so the whole GEMM is a **single dispatch** (ROUNDS=3 ×
       COLS=8 = 24 M-blocks): the array runs continuously (no inter-dispatch host stall),
       one exec, one C read-back, no coherence dance. **~1.9 TOPS** (1.5–1.7 ms, 0/6 fail,
       all 24 M-blocks correct), reliable via a single blocking dispatch. Lesson: stream
       via **pure-linear** DMAs, not repeat BD dims — a repeat dim doesn't re-check the
       objectfifo semaphore, so rounds >0 overrun (round 0 correct, rest garbage); W is
       replicated ROUNDS× in DRAM since the broadcast fifo can't replay.

     **Full progression (M768·K512·N4096): 0.02 → 1.0 → 1.45 → ~1.9 TOPS (~95× over the
     CPU-marshaling floor).** Levers left: (1) reduce the broadcast-sync feed cost (raw
     single-dispatch ceiling is ~3 TOPS); (2) the real aggregate win — a **concurrent
     NPU ‖ GPU split** (the ~40% prefill win lives here, not in sync offload). At ~1.9
     TOPS this is still below GPU (~50), so the step-4 hot-path hook stays gated for
     *sync* offload; the concurrent split is its own effort.

   - **(superseded) 3c-old. Activations + output → the DMA (dynamic, per inference).** A and C are
     computed at runtime, so the loader can't pre-arrange them. Feed A row-major and
     let the DMA tile it; write C tiled and let the DMA de-tile — the reshuffle in
     hardware, not the CPU. **Attempted (measured):** a hand-rolled reshuffle on the
     *shim runtime `dma_bd`* (shim→core direct). It builds — learned the aie2p
     convention: `len` = product of the **lowest three** dims (the highest/outermost
     dim is the BD repeat count, excluded from `len`). And the A strides
     `[(16,1024),(16,16),(4,256),(16,1)]` match IRON's `my_matmul` A pattern exactly.
     **But it produces wrong data** (A-reshuffle-only: 4093/4096 mismatch). So the
     shim→core *direct* `dma_bd` reshuffle does not deliver tile-major to the core.
     **The proven path is IRON's: put the reshuffle on a MEMTILE hop** — shim→memtile
     (linear) then `objectfifo.cons().split(..., dims_to_stream=…, placement=Tile(col,1))`
     memtile→core does the tiling. Next: add the memtile hop with `dims_to_stream` for
     A (and the join for C) instead of the shim-direct `dma_bd`. Plus stream the whole
     GEMM in one dispatch to amortize the ~78 µs latency.

4. **Runtime offload hook (the hot-path change — smallest possible).** In
   `dispatch/quant.rs`, add an opt-in path: if a prefill W4A8 GEMM is large enough to
   amortize the ~78 µs dispatch latency AND an R6 xclbin is cached for its shape,
   dispatch it on the NPU. Gate behind a flag first (`HIPFIRE_NPU_PREFILL`), measure
   end-to-end, then consider default-on.

## CRITICAL measured finding — marshaling, not the kernel, is the bottleneck

Array `NpuGemm` is validated correct, but end-to-end it is **catastrophically slow**:
`NpuGemm::run` on M=768 K=512 N=4096 (peak config, 32 dispatches) = **351 ms/run =
0.01 TOPS** (result numerically correct). The kernel computes at 20.7 TOPS in ~µs; the
**CPU marshaling** (re-shuffling row-major A/W into the tile-major int4 SHMEM layout,
per-element bit-packing) takes ~348 ms and dwarfs everything. Dispatch latency is
~78 µs × 32 = 2.5 ms — also non-trivial but small next to marshaling.

**So the wire-in's hard problem is marshaling + dispatch overhead, not the kernel.**
The fixes, in impact order:
1. **Pre-marshal weights once at load** (weights are static): re-marshaling W every
   dispatch is most of the 348 ms. Marshal each layer's W into its tile-major SHMEM
   form at model load; per inference, only activations move. Projected: 0.01 → ~0.6
   TOPS (then latency-bound).
2. **Fewer dispatches**: the array can stream the whole GEMM in one dispatch (large
   NB) instead of one M-block/K-chunk per call — removes the ×32 latency and the
   per-dispatch re-marshal. The 20.7-TOPS bench used one huge-NB dispatch; realistic
   shapes must stream similarly.
3. **Fast A marshal / C un-marshal** (SIMD/memcpy-shaped), and keep A resident.

Until (1)+(2) land, the offload is a net loss vs the GPU (which needs no marshaling).
The 20.7 TOPS is a real *compute* rate; the deliverable end-to-end rate depends
entirely on beating this overhead — that, not the kernel, is now the open question.

### Measured: pre-packing W helps 2×, but CPU marshaling is still a dead end

`prepack_weights` (once, 23 ms) + `run_packed` (per inference, weight cost = memcpy):
**351 → 177 ms/run (0.01 → 0.02 TOPS)** on the same shape, still correct. So the W
re-pack was ~half — but the residual 177 ms is the **C tile→row-major reshuffle** (M·N
= 3.1 M scalar index ops per inference) plus per-dispatch A-pack. Even a perfect CPU
version floors around ~20 ms (0.16 TOPS) — still below the GPU.

**The real fix is DMA-side reshuffle, not CPU.** The shim DMA supports strided access
(`dims_to_stream` on the objectfifo `dma_bd`), so A/W/C can be fed **row-major** and
the DMA does the tile-major reshuffle *for free* during transfer — which is exactly
how IRON's whole_array gemm avoids CPU marshaling. My hand-written R6 MLIR uses plain
linear `dma_bd` (hence the CPU marshaling). Porting the tile-major stride pattern into
the `dma_bd` eliminates CPU marshaling entirely; that is the substantive next step and
the true gate on offload viability (alongside the ~78 µs/dispatch latency, which still
argues for streaming the whole GEMM in one dispatch). `prepack_weights`/`run_packed`
stay useful (weights still pre-arranged once), but the reshuffle must move to the DMA.

## Concurrency premise — PROVEN (`npu_concurrency_demo`)

The whole offload thesis rests on one assumption: an async NPU dispatch overlaps with
concurrent host/GPU work, so the NPU GEMM is *hidden* rather than added to the critical
path. Measured on the whole-GEMM xclbin with a CPU workload proxy for GPU-dispatch-issuing
host work (`submit` → host work → `wait`):

| schedule | ms |
|---|---|
| T_npu (dispatch alone) | 0.855 |
| T_host (host work alone) | 1.530 |
| T_serial (submit; wait; host) | 2.929 |
| **T_overlap (submit; host; wait)** | **1.678** |

T_overlap ≈ T_host — the **entire 0.855 ms NPU GEMM is hidden** behind concurrent host
work (saves ~1.25 ms/iter). So the `submit`/`wait` split does deliver true async
concurrency; when there is GPU work to run alongside, NPU prefill offload is a net win, not
a loss. This de-risks the concurrent split below: the mechanism is validated; what remains
is wiring a *real* GPU GEMM (hipfire-rdna) as the concurrent partner instead of the CPU
proxy, plus the work-splitting policy.

## The one architectural decision (needs a call)

**Concurrency model.** The win is *concurrent* NPU-prefill ‖ GPU-work. Options:
- **Sync offload** (simplest): block on the NPU GEMM. Only wins if the NPU GEMM is
  faster than the GPU for that shape — at 20.7 vs ~50 TOPS it usually is *not* alone,
  so sync offload mostly doesn't help. Good for a first correctness wiring only.
- **Concurrent split** (the real win): split each prefill GEMM (or alternate
  layers/experts) between NPU and GPU so both run in parallel, ~+40% aggregate.
  Needs a work-splitting policy + a join, and careful interaction with the
  HIP-direct scheduler.
- **Async pipeline**: NPU prefills layer L+1's projections while the GPU finishes
  layer L. Most throughput, most complexity.

Recommend: land steps 1–3 + a **sync**, flag-gated step 4 first (proves the whole
path end-to-end on a real model), then design the concurrent split as its own effort
— that is where the aggregate win actually lives and where the HIP/NPU scheduler
interaction needs deliberate design.

---

## Measured 2026-07-30 (halo, gfx1151/aie2p) — ROUNDS landed, numbers corrected

Re-measured this whole path on halo. Two claims above are stale; a third fix landed.

**The 20.7 TOPS figure does not reproduce.** Rebuilt via `r6_cache.sh` and timed with
`npu_gemm_bench`, the R6 kernel peaks at **3.17 TOPS W4A8** (`r6_gen_mp.py`, COLS=8
NB=1024) and **2.72 TOPS** N-parallel. The W8A8 variants are lower still: **1.15 TOPS**
(`R6_W8_M8=1`, COLS=8 NB=256) and 0.28 TOPS (`R6_W8=1`). `MT=32`/`MT=64` do not build at
KCHUNK=16 — `Resource allocation pipeline failed`, L1-capped — so 16x4x16 is the ceiling
for this kernel shape. Treat 3.2 TOPS, not 20.7, as the W4A8 kernel rate until someone
reproduces the original measurement.

**Marshaling is no longer the bottleneck; dispatch count was.** The "CRITICAL measured
finding" section above (351 ms/run, 0.01 TOPS) predates the R6-TS tensor-stream kernel.
With row-major A/C and prepacked W, the same M=768 K=512 N=4096 shape runs at **0.447
TOPS end-to-end** — 22x better. The residual was per-dispatch latency: 64 dispatches x
~78 us = 5.0 ms of the 7.2 ms.

**Fix: `NpuGemm::load_rounds` (ROUNDS support).** `r6_gen.py` has always supported
streaming ROUNDS M-blocks per dispatch; `NpuGemm` only ever used ROUNDS=1. Wiring it in
(plus fusing the K-accumulation pass into the C unpack) gives, on that shape:

| config | dispatches | ms/run | TOPS e2e |
|---|---|---|---|
| MT=24 KCHUNK=8 groups=32, ROUNDS=1 (prior) | 64 | 7.20 | 0.447 |
| MT=8 KCHUNK=32 groups=64, ROUNDS=1 | 24 | 3.03 | 1.062 |
| MT=8 KCHUNK=32 groups=64, **ROUNDS=4** | 6 | 1.88 | **1.713** |
| MT=8 KCHUNK=32 groups=64, ROUNDS=8 | 3 | 2.19 | 1.472 |

ROUNDS=8 regresses: too few dispatches to keep the readback pipeline full, and W is
replicated ROUNDS times in DRAM (`r6_gen.py` can't replay a broadcast fifo), so the
weight working set grows with ROUNDS. ROUNDS=4 is the current sweet spot; the raw-kernel
sweep peaks at ROUNDS=16 (2.80 TOPS) and regresses at 32 for the same reason.

On realistic llama-3.2-1B linear shapes it holds ~1.2 TOPS end-to-end (all
correctness-checked against a CPU W4A8 reference at rows 0/mid/last):

| shape (M x K x N) | ms/run | TOPS e2e |
|---|---|---|
| 512 x 2048 x 4096 (qkv/o) | 4.41 | 1.947 |
| 512 x 2048 x 8192 (gate/up) | 9.11 | 1.887 |
| 512 x 8192 x 4096 (down) | 17.85 | 1.924 |
| 2048 x 2048 x 8192 (prefill) | 31.84 | 2.158 |

**W copy traffic was the real residual, and a loop reorder fixed it.** `run_packed`
walked `mo` outermost with `ko` innermost, so the W slab — a function of `(ko, no)` only
— was refilled on *every* dispatch: 128 fills x `groups*rounds` slabs = ~537 MB of host
writes per 2048x2048x8192 GEMM, more than the C readback. Reordering to `no, ko, mo`
(M-blocks innermost) makes W constant across all M-blocks, cutting it to `nns*nks` = 8
fills. K-partials now accumulate directly into the output block (overwrite at `ko==0`),
so the scratch accumulator and its extra pass over M*N are gone as well. That took the
prefill shape from 56.39 ms to 31.84 ms (1.219 -> 2.158 TOPS) with the CPU W4A8
reference still matching at rows 0/mid/last.

End-to-end is now at ~2.0-2.2 TOPS against a raw-kernel ceiling of 2.18 TOPS at
ROUNDS=4, i.e. host overhead is largely gone and **further gains need a faster kernel,
not better marshaling**. Re-sweeping ROUNDS after the fix confirms 4 is still the sweet
spot (8: 2.00, 16: 1.56 TOPS) — with W fills no longer scaling with M-blocks, the
remaining ROUNDS penalty is purely the DRAM replication.

**Next bottleneck: the kernel itself.** For reference, `findings.md`'s tuned int8
`whole_array` reaches 15.7 TOPS with m128 k32 n128 on 8 columns, while R6-TS runs a
4x16x16 `mmul` at MT=8 — a far smaller output tile, which is exactly the lever
`findings.md` identifies (throughput scales with output-tile size because it amortizes
per-tile feed/sync). Closing that gap is the work that matters now.

**Superseded: C readback traffic.** At 2048x2048x8192 the dispatch latency is now
only ~18% of the run. The dominant residual is the host C unpack: every dispatch reads
back `block_m x block_n` int32 (128 dispatches x 2 MB = 268 MB per GEMM), because the
kernel overwrites C per dispatch and each K-chunk's partial must come back to be summed.
Making the kernel *accumulate* into C across K-chunk dispatches would cut this by `nks`
(4x on K=2048, 16x on K=8192) and is the single highest-leverage change left before the
dispatch seam in `dispatch/quant.rs` is worth wiring.

### Tile-shape sweep: K-depth beats output-tile height (2026-07-30)

`findings.md` says throughput scales with output-tile size, so the obvious next move was
a taller tile. Measured on 2048x2048x8192, ROUNDS=4, it is the opposite — trading KCHUNK
for MT is strongly negative:

| MT | KCHUNK | W fifo depth | L1 est | ms/run | TOPS e2e |
|---|---|---|---|---|---|
| 8 | 32 | 2 | ~56 KB | 31.62 | **2.173** |
| 8 | 32 | 1 | ~40 KB | 32.96 | 2.085 |
| 4 | 64 | 1 | ~52 KB | 37.83 | 1.817 |
| 16 | 16 | 2 | ~48 KB | 67.75 | 1.014 |
| 32 | 8 | 2 | ~56 KB | 111.18 | 0.618 |
| 16 | 32 | either | >64 KB | — | build fail |

The reason the earlier finding does not transfer: with C accumulated host-side across
K-chunks, shrinking KCHUNK multiplies `nks`, and `nks` drives BOTH the dispatch count and
the number of host passes over M*N. On this shape KCHUNK=8 means 16 K-chunks instead of
4. Feed amortization from a taller tile does not come close to paying for that. Dropping
the W fifo to single-buffered (freeing L1 for more K) also loses — the double-buffered
feed is worth its space. **MT=8, NT=4, KCHUNK=32, depth=2, ROUNDS=4 is the optimum** for
the buildable set; `R6_W_FIFO_DEPTH` was added to `r6_cache.sh` for this sweep.

### Structural gap: there is no NPU decode path

Everything above is GEMM, i.e. prefill-shaped (large M). The FastFlowLM numbers this work
is measured against are dominated by **decode** tok/s, which is GEMV (M=1) and
bandwidth-bound, not compute-bound. `hipfire-xdna` has no GEMV/decode kernel — R6's
minimum `block_m` is `ROUNDS*MT*4` (128 at the optimum), so a decode step would waste
127/128 of the M dimension. Prefill offload can win TTFT and the prefill rate; it cannot
move the tok/s figure at all. A separate W4A8 GEMV kernel is required, and its ceiling is
the ~55 GB/s fabric: ~88 tok/s for a 1.24 GB oq4 llama-3.2-1B pass, ~36 tok/s for the
35B-A3B's ~1.5 GB active set (vs FLM's measured 60.1 and 13.4).

## Decode (GEMV) on the NPU — 4.3 -> 32.5 GB/s (2026-07-30)

Decode is the figure of merit for the FastFlowLM comparison and had no path at all. It
turns out the R6-TS GEMM kernel can serve it, because decode is BANDWIDTH-bound: padding
M=1 up to `block_m` wastes compute but no weight traffic, and the wasted rows ride along
on bytes that had to move anyway. Two changes make it viable, both in `NpuGemm`:

1. **`upload_weights` / `run_resident` — device-resident weights.** `run_packed` refills a
   shared `w_buf` from host memory per dispatch, which for GEMV means copying the entire
   weight matrix every token (8.4 MB for one llama-1B projection). `upload_weights` puts
   each `(ko, no)` block in its own `DeviceBuffer` once; `run_resident` binds the right
   one per dispatch and marks W clean in `submit_synced`, so the steady state has ZERO
   host weight traffic.
2. **Overlapped dispatches.** With A and C ringed and each weight block its own buffer,
   dispatch `i` shares no buffer with `i-1`, so the submit no longer waits. That hides the
   ~78 µs submit behind the previous dispatch's streaming.

Measured on `r6ts_1x4x32_c8_nb16` (MT=1, KCHUNK=32, groups=128, ROUNDS=1), one token
through one linear, W-stream rate against the ~55 GB/s fabric, row 0 checked against a CPU
W4A8 reference every run:

| step | K=2048 N=8192 | rate |
|---|---|---|
| first working decode (MT=8, ROUNDS=4, groups=64) | 1.962 ms | 4.3 GB/s |
| ROUNDS=1 (W not replicated per round) | 0.957 ms | 8.8 GB/s |
| MT=1 + single N-slab (`groups=128`, 4 dispatches) | 0.455 ms | 18.4 GB/s |
| + overlapped dispatches (PIPE=2) | 0.305 ms | **27.5 GB/s** |
| same, K=8192 N=8192 | 1.031 ms | **32.5 GB/s** |

Config notes, all measured: ROUNDS>1 is actively harmful for decode (W is replicated per
round, so the kernel streams `rounds`x the bytes for the same weights). MT=1 beats MT=2/4
(17.7 / 16.4 GB/s) — every extra M row is pure waste at M=1. A single N-slab covering the
whole matrix minimises dispatches. PIPE=4 is slightly worse than 2 (27.4 / 28.9 GB/s), so
two in flight already saturates.

**Projection against FastFlowLM** (weights-only, from `/home/sadara/flm-benchmarks.md`):
at 32.5 GB/s, llama-3.2-1B's ~620 MB/token oq4 pass is ~19 ms = **~52 tok/s** vs FLM's
60.1 — close but short. The 35B-A3B's ~1.5 GB active set is ~46 ms = **~22 tok/s** vs
FLM's 13.4 — a 1.6x win. Combined with prefill (2.17 TOPS vs the ~1.74 FLM implies), the
MoE is the model this path beats first.

Caveat: these are single-linear microbenchmarks. A real decode step adds attention, norms,
sampling, routing, and the small projections (llama's k/v are N=512, far below the 8192
N-slab that makes these numbers) — none of which are measured here.

### Real projection shapes: fusion matters more than anything else

The 32.5 GB/s above is a best-case shape. Measured on llama-3.2-1B's actual per-layer
projections (MT=1, KCHUNK=32, ROUNDS=1, groups sized to the matrix):

| linear | shape | ms | GB/s |
|---|---|---|---|
| k / v | 2048 x 512 | 0.162 | 3.2 |
| q / o | 2048 x 2048 | 0.182 | 11.5 |
| down | 8192 x 2048 | 0.592 | 14.2 |
| gate / up | 2048 x 8192 | 0.296 | 28.4 |

Narrow-N matrices are dispatch-latency-bound, not bandwidth-bound: k/v move 0.5 MB and
still cost 0.162 ms. Fusing projections that share an activation and a K fixes it, which
is exactly why checkpoints ship `in_proj_qkv` / `gate_up_proj`:

| fused linear | shape | ms | GB/s | vs separate |
|---|---|---|---|---|
| qkv | 2048 x 3072 | 0.193 | 16.3 | 0.506 ms -> **2.6x** |
| gate_up | 2048 x 16384 | 0.463 | **36.3** | 0.592 ms -> 1.28x |

With fusion a llama-3.2-1B layer is qkv 0.193 + o 0.182 + gate_up 0.463 + down 0.592 =
**1.43 ms**, so 16 layers + lm_head is ~26.5 ms = **~38 tok/s** against FLM's 60.1. That
supersedes the ~52 tok/s projection above, which assumed every linear ran at the wide-shape
rate. **Any wire-in should fuse qkv and gate_up at upload time**, not dispatch them
separately.

**Negative result — pipeline depth does not fix latency-bound shapes.** `down` (16 narrow
dispatches) was the obvious candidate for a deeper queue, but PIPE=4 measures 13.9 GB/s vs
PIPE=2's 14.2, and it costs the wide shapes too (gate_up 35.6 vs 36.3, qkv 16.2 vs 16.3).
So the ~37 µs per-dispatch floor is not queue occupancy — two in flight already saturates
whatever serialises submission. Dispatch COUNT is the only lever, which means bigger
KCHUNK (L1-capped at 32) or wider N per dispatch. PIPE stays at 2.

### What FastFlowLM actually does (deconstructed 2026-07-30)

FLM ships per-model xclbin sets under `/opt/fastflowlm/share/flm/xclbins/<model>-NPU2/`.
For `Llama-3.2-1B-NPU2` there are exactly four: `layer.xclbin`, `attn.xclbin`,
`mm.xclbin`, `dequant.xclbin`. The decisive one is **`layer.xclbin` — a fused
whole-decoder-layer kernel** (MLIR-AIE, 5 buffer args). Bigger models add `mv.xclbin`
(matrix-vector, i.e. an explicit decode path) and `lm_head.xclbin`.

So FLM's decode is roughly ONE dispatch per layer, not one per linear. That is the
structural difference from `NpuGemm`: at the measured ~37 µs per-dispatch floor, our
4-dispatch-per-layer decode (qkv, o, gate_up, down — already fused) burns ~0.15 ms/layer
of pure submit latency before any weight moves, and the narrow linears never reach fabric
rate at all.

**CORRECTION (later the same day): FLM's memory throughput is 46.4 GB/s and is NOT
exceptional.** An earlier revision of this section claimed 43-78 GB/s and used it to argue
the ~55 GB/s fabric figure was not a ceiling. That was wrong: it counted the whole 1.24 GB
container. `model.q4nx` is a safetensors file whose manifest shows
`model.embed_tokens.weight` is **BF16 [128256, 2048] = 525.3 MB** — a per-token *gather*
of one 4 KB row, not streamed. The streamed set is the 113 `I8` tensors, 772.3 MB, which
over llama-3.2-1B's 1.236 B non-embedding weights is exactly **5.00 bits/weight**
(lm_head is its own I8 tensor, 164.2 MB — not tied to the embedding).

At 60.1 tok/s that is **46.4 GB/s**, comfortably *inside* the ~55 GB/s figure. For
comparison, hipfire's own GPU path already matches or beats it on the same model:

| | bytes/token | tok/s | weight-stream rate |
|---|---|---|---|
| FLM (NPU, 5.00 bpw) | 772.3 MB | 60.1 | 46.4 GB/s |
| hipfire oq4++ (4.125 bpw) | 637.2 MB | 75.94 | 48.4 GB/s |
| hipfire mq4 | 637.2 MB | 100.14 | 63.8 GB/s |

**So FLM has no bandwidth advantage** — it streams MORE bytes per token than an oq4
artifact (5.00 vs 4.125 bpw) and decodes SLOWER. Its sole advantage is prefill (~2750
t/s), which on 1.236 B weights is 2*1.236e9*2750 = **~6.8 TFLOP/s of compute**. The gap is
therefore compute/batching, not memory. Do not cite a bandwidth ceiling to explain it.

**Conclusion for the wire-in: fusing the layer is the design, not an optimisation.**
Per-linear `NpuGemm` dispatch tops out around ~38 tok/s on llama-3.2-1B against FLM's
60.1. Closing that needs the layer resident as one kernel — projections, norms, rope and
attention in a single dispatch with weights streamed continuously — which is what
`layer.xclbin` is. The R6 work stands as the GEMM primitive and the prefill path
(2.17 TOPS, which already beats what FLM's 35B-A3B prefill implies), but the decode
figure of merit needs the fused-layer kernel.
