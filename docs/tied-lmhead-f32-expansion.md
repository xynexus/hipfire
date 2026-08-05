# The tied lm_head is expanded to F32, and why lut3 can't fix it yet

Two findings, one cause.

## 1. Why the two-stage BF16 gate is false

`lmhead_project` (`runtime/src/llama.rs:66`) gates the two-stage shortlist on
`w.gpu_dtype == DType::BF16`. For a **tied-embedding** model it is never BF16.

Both tied-head load paths dequantise the embedding to **F32** and upload 4 bytes
per element:

| path | site |
|---|---|
| HFQ | `runtime/src/hfq.rs:2689-2727` — `quant_type 16` → `f32::from_bits(u16 << 16)`, `gpu_dtype: DType::F32` |
| PARO/source | `runtime/src/hfq.rs:3158-3199` — same, `gpu_dtype: DType::F32` |

So on Llama-3.2-1B-Instruct (`tie_word_embeddings: true`) the head is BF16 on
disk and **F32 in VRAM**. The gate cannot match, and the two-stage path is dead
code for every tied model — which is exactly the class where the head is the
largest single tensor.

That is the measured flat TTFT: 59.0 / 60.2 / 56.6 ms for unset / `q4` / `q2`.
The coarse build never runs because the branch is never taken.

### Verified, both halves

The claim needs two things to be true, and both are directly observed rather
than inferred:

* **The artifact takes the tied branch.** `hipfire inspect --tensors` on
  `Llama-3.2-1B-Instruct--oq4++.hfq` matches `lm_head` **zero** times. The only
  head-shaped tensors are `model.embed_tokens.weight` (BF16) and
  `model.embed_tokens.coarse.weight` (CoarseQ4Row). So
  `hfq.find_tensor("lm_head.weight").is_some()` is false and the `else` branch
  runs.
* **That branch uploads F32.** Both sites build a `Vec<f32>` and construct
  `WeightTensor { gpu_dtype: DType::F32 }` — there is no conditional inside them.

Two measurement routes that do NOT work on this box, recorded so they are not
retried:

* **VRAM via `/sys/class/drm/card1/device/mem_info_vram_used`** stays flat at
  ~156 MiB through a full load. gfx1151 is a UMA APU; weights do not show up as
  discrete VRAM.
* **`/usr/bin/time -v` on `hipfire chat`** reports 10.5 MB peak RSS. The CLI
  hands off to a daemon it spawns, so the weights are never in the measured
  process. Sampling the daemon's RSS instead is a race — load takes seconds and
  a shell poll loop finishes in milliseconds.

## 2. What it costs

`128256 x 2048` head, per token:

| head form | bytes | vs F32 |
|---|---|---|
| **F32 — today, tied models** | **1050.7 MB** | 1.00x |
| BF16 (as stored on disk) | 525.3 MB | 2.00x better |
| BF16L3 lossless (1.38x) | 380.7 MB | 2.76x better |
| coarse Q4 shortlist (lossy tail) | 131.3 MB | 8.00x better |

Whole-model per-token traffic against FLM's measured 772.3 MB for the same model:

| configuration | MB/token | vs FLM |
|---|---|---|
| **hipfire today (F32 tied head)** | **1545.5** | **2.00x** |
| head kept BF16 | 1020.1 | 1.32x |
| head as BF16L3 | 875.5 | 1.13x |
| two-stage coarse Q4 | 626.1 | 0.81x |

The earlier claim that hipfire streams 1020 MB/token was optimistic: it assumed
the head stayed BF16. It expands. **hipfire currently moves exactly twice FLM's
bytes per token on llama3.2:1b**, and the single largest cause is a dequant at
load, not a format choice.

## 3. Why bf16lut3 is the right idea and still blocked

BF16L3 (`primitives/src/bf16_lut3.rs`) is **exactly lossless** — every `u16`
reproduced bit-for-bit including NaN payloads — at ~11.6 bits/element, 1.38x
measured, and it is decoded *in the kernel*, so the ratio applies to bandwidth
rather than only to file size. For an exact lm_head that is strictly better than
BF16: same numerics, fewer bytes. There is no accuracy argument against it.

It cannot be used on a tied head today. From `HIPFIRE_BF16L3_RESIDENT`'s own
documentation:

> Requires kernels that decode the packed form natively (`gemv_bf16l3`); **a
> gather-read table (a tied embed/lm_head) has no such path and will fail to
> load.**

The tied tensor serves two consumers with different access patterns: the
embedding **gather** (one random row per token) and the logit **GEMV** (every
row). BF16L3 is block-addressable — which is why `hfq_out.rs:167` deliberately
steers gather-shaped tensors to LUT3 over the better-compressing Huffman — but
no gather kernel decodes it, so the tied case fails.

Note also that `HIPFIRE_BF16L3_RESIDENT` buys ~1.18x bandwidth **only** once the
working set exceeds last-level cache, and is a measured slowdown below that. On
a bandwidth-bound NPU that is a win; on a GPU with a 1B model it may not be.

## Why F32 may have been deliberate — read before "just keep it BF16"

Step 1 below looks free. It may not be, and the reason is a kernel-dispatch
detail rather than anything about numerics.

`weight_gemv` handles a BF16 weight with an explicit special case
(`weights.rs`, inside `weight_gemv`):

    // BF16 weights use WMMA GEMM directly (dispatch family has no BF16 GEMV entry).
    if w.gpu_dtype == DType::BF16 {
        return gpu.gemm_bf16_x_bf16_wmma(&w.buf, x, y, w.m, w.k, 1);
    }

So a BF16 head works, but it is served by a **batch-1 WMMA GEMM**, not by a
GEMV. F32 and F16 heads instead reach dedicated residual GEMV paths. Expanding
the tied head to F32 buys the better decode kernel at twice the bytes, and on a
GPU whose working set fits in cache that can be the right trade — which is the
same regime where `HIPFIRE_BF16L3_RESIDENT` is documented as "a measured
slowdown".

Note that BF16 GEMV kernels do exist in the tree — `gemv_bf16_f32.hip`,
`gemv_bf16_vec8.hip`, and a gather variant `gemv_bf16_gather_f32.hip`, plus
`gemv_bf16l3.hip` for the packed form. They are simply not wired into the
dispatch family that `run_auto` consults. That is what makes the comment true
today, and it is also the smallest change that would make step 1 unambiguously
a win rather than a trade.

### Measured — and the answer is neither of the two obvious options

`cargo run --release -p hipfire-runtime --example bench_lmhead_dtype`, at the
real 128256 x 2048 shape, 30 iterations on gfx1151:

| path | per call | achieved | tok/s if head-bound |
|---|---|---|---|
| **BF16 via `gemv_bf16_f32`** | **3.18 ms** | 165.5 GB/s | **315.0** |
| F32 via `gemv_f32` | 5.28 ms | 199.2 GB/s | 189.6 |
| BF16 via `gemm_bf16_x_bf16_wmma(batch=1)` | 14.97 ms | 35.1 GB/s | 66.8 |

Two conclusions, and the first corrects this document's own earlier advice.

**The F32 expansion is right, given the dispatch family as it stands.** Against
the BF16 path `weight_gemv` actually takes, F32 is **2.95x faster**. "Just keep
the head BF16" would have been a near-3x decode regression at the head, not a
free halving. A batch-1 WMMA GEMM sustains only 35.1 GB/s — it is built for
batched work and collapses at one row.

**But the best option is already in the tree and unused.** `gemv_bf16_f32` beats
the F32 GEMV by **1.66x** *while reading half the bytes*. Note it does so at a
LOWER achieved bandwidth (165.5 vs 199.2 GB/s): the F32 kernel saturates memory
better, and still loses on wall time because it has twice as much to move. Bytes
beat efficiency here.

The kernel exists (`kernels/src/gemv_bf16_f32.hip`), the binding exists
(`dispatch/gemv.rs:4843`), and neither is in the dispatch family `run_auto`
consults — which is exactly why `weight_gemv` needs its WMMA special case, and
why the loader expands to F32 to avoid it. Wiring it in is the fix: 1.66x faster
decode at half the per-token head traffic, no accuracy change, and it makes the
two-stage path reachable as a side effect.

The NPU case prefers fewer bytes independently — a 56.5 GB/s fabric does not
care that a GEMV kernel is nicer.

## Ordering

1. **Wire `gemv_bf16_f32` into the dispatch family, THEN stop expanding to F32.**
   Measured 1.66x faster than the F32 GEMV at half the bytes. Doing the second
   half without the first is a 2.95x regression — see the measurement above.
2. **Then BF16L3**, which needs a gather-read decode path so the tied tensor can
   stay packed for both consumers — 1.32x -> 1.13x, still lossless.
3. **Two-stage** is a decode/argmax fast path only; eval and scoring must keep
   the exact head, so 1 and 2 are what those paths get.
