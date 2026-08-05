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

| path | per call | achieved | tok/s if head-bound | argmax vs F32 | worst diff |
|---|---|---|---|---|---|
| **BF16 via `gemv_bf16_f32`** | **3.20 ms** | 164.1 GB/s | **312.3** | OK | 1.16e-3 |
| F32 via `gemv_f32` | 5.22 ms | 201.3 GB/s | 191.6 | reference | — |
| BF16 via `gemm_bf16_x_bf16_wmma(batch=1)` | 14.89 ms | 35.3 GB/s | 67.2 | OK | 1.16e-3 |

Both BF16 paths land at the same 1.16e-3 worst deviation against the F32
reference on a 0.284 magnitude — that is bf16 mantissa rounding, identical for
the two, and the decoded token is unchanged.

**The correctness column is not decoration.** The first run of this benchmark
recommended `gemv_bf16_f32` on speed alone, and it was producing garbage: worst
|diff| 5.8e8 and a different argmax. The cause was a caller error, not a kernel
bug — `gemm_bf16_x_bf16_wmma` asserts an **F32** activation and stages it to
bf16 internally, while `gemv_bf16_f32` wants a **BF16** activation already (its
`x` is `const unsigned short*`). Passing the F32 buffer reads float bytes as
bf16 pairs. The two contracts differ, and mixing them yields wrong numbers
rather than an error. Timing was unchanged between the broken and fixed runs
(3.199 vs 3.202 ms), so the bad run was not fast by skipping work — speed alone
would never have exposed it.

That contract is also the one real cost of wiring this in: runtime activations
are F32, so the path needs an x -> bf16 staging step. It is 2048 elements
against a 128256 x 2048 weight read, and the WMMA path already does exactly
this internally.

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

## Landed: `gemv_bf16_xf32`

`kernels/src/gemv_bf16_xf32.hip` + `weight_gemv` routing (commit
`feat(gemv): add gemv_bf16_xf32 and route BF16 weights through it`).

| path | time | worst diff vs f32 ref (same bf16 W) |
|---|---|---|
| **`gemv_bf16_xf32` (x=f32)** | **3.24 ms** | **6.7e-8** |
| `gemv_bf16_f32` (x=bf16) | 3.20 ms | 4.5e-4 |
| `gemm_wmma` (x staged to bf16) | 14.6 ms | 4.5e-4 |
| `gemv_f32` (W widened) | 5.22 ms | reference |

6.7e-8 is f32 accumulation noise, so this is numerically the widened-F32 path
at half the bytes and 1.61x its speed. The 1.1% it gives up against the
bf16-activation variant buys back a 6800x accuracy difference.

### Gate status — the 8 failures are NOT from this change

`tests/tiny-affected-gate.sh --require-coverage` reports 8 drifted cells. They
are pre-existing. Reverting only the `weight_gemv` routing, rebuilding, and
re-running the same cells reproduces them **to six decimals**:

| cell | with change | routing reverted |
|---|---|---|
| `qwen2/kld:hfq4` | 0.001790 | 0.001790 |
| `qwen3_5_moe/kld:q8f16` | 0.179210 | 0.179210 |
| `qwen3_5_moe/kld:mq6` | 0.215099 | 0.215099 |
| `qwen3_5_moe/kld:mq4` | 0.215099 | 0.215099 |

Bit-identical means the BF16 path is not exercised by these fixtures at all —
their linear weights are quantized, so nothing reaches the branch. This is the
stale-gfx1151-baseline issue already on the open-decisions list, and
`minimax/kld:mq4` drifting by exactly 0.000000 is the known vacuous cell.

The coherence battery reported no hard errors.

Re-run after the dispatch-family registration: the same 8 cells, same numbers.
So the registration is behaviourally inert for these fixtures too, as expected —
nothing in them dispatches BF16.

### Models that DO exercise the path

`zaya1-8b-parity.bf16` generates 1 token and stops. That predates the change:
the same prompt on the pre-change build (`gemm_bf16_x_bf16_wmma`) does the same,
22.48 vs 20.25 tok/s. It is a parity fixture, not a chat model. Worth recording
because "a full-bf16 model emits one token" is exactly the shape a regression
here would take, and the only way to tell was to run the old code.

`Llama-3.2-1B-Instruct--bf16` is the working end-to-end check: byte-identical
generation across all three arrangements (WMMA special case, gemv_bf16_xf32
special case, gemv_bf16_xf32 via the family) at 24.65 / 24.68 tok/s.

## Attempted and reverted: keeping the tied head BF16

The loader change — upload a BF16 embedding as bf16 instead of widening — was
written and then reverted, because it could not be shown to execute.

`hipfire bench` on `Llama-3.2-1B-Instruct--oq4++` gave **tg128 76.09 t/s against
76.07 before**: no change, where a 1050.7 -> 525.3 MB head at 1.61x the kernel
speed should have been plainly visible. Rather than accept a null result, a
diagnostic was added to both branches of the tied path. **Neither ever fired**,
while `eprintln!("  loading output...")` two lines above them did.

That is not a possible state for one binary, and chasing it burned the rest of
the session's budget:

* `cargo build -p hipfire-runtime` builds the LIBRARY. The `hipfire` binary
  lives in `hipfire-cli` and needs `--bin hipfire`, so the first several
  measurements ran against a stale binary.
* Rebuilding the binary did not fix it either, and `hipfire stop` reports
  `no /home/sadara/.hipfire/serve.pid` while `hipfire chat` had separately
  refused with `FATAL: hipfire daemon already running (PID ...)`. So daemon
  lifetime is not fully described by the pidfile, and which binary serves a
  given `chat` is not obvious from outside.

**Nothing here says the change is wrong** — only that this session could not
demonstrate it running, and an unproven change to a hot load path is worse than
no change. What it does establish is that the measurement loop for loader edits
is unreliable in a way that silently returns "no difference", which is exactly
the failure mode that makes a bad change look safe.

Anyone picking this up: verify a diagnostic in the edited branch actually
appears before trusting any number. The `gemv_bf16_xf32` work it depends on is
committed and separately verified.

## Ordering

1. **Wire `gemv_bf16_f32` into the dispatch family, THEN stop expanding to F32.**
   Measured 1.66x faster than the F32 GEMV at half the bytes. Doing the second
   half without the first is a 2.95x regression — see the measurement above.
2. **Then BF16L3**, which needs a gather-read decode path so the tied tensor can
   stay packed for both consumers — 1.32x -> 1.13x, still lossless.
3. **Two-stage** is a decode/argmax fast path only; eval and scoring must keep
   the exact head, so 1 and 2 are what those paths get.
