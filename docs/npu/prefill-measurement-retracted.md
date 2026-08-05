# RETRACTED: "prefill gets no batching benefit in the daemon"

Two commits — `bcf7f93e4` and `b890248a0`, previously
`prefill-not-batching-confirmed.md` — claimed to confirm in the daemon that
prefill runs per-position. **The claim is unsupported. The measurement never
reached a prefill path.**

## What was actually measured

`hipfire bench --pp-tokens N` issues `DaemonRequest::BenchPrefill`, handled by
`handlers::diag::bench_prefill`. For LLaMA/Qwen3 (arch 0/1) that handler is a
per-token loop:

```rust
} else if let Some(backend) = m.llama_backend.as_mut() {
    // ... per-token decode_step saturates the dense
    // attention/GEMV/RoPE kernel set before the first real request.
    for (i, &tok) in synthetic.iter().enumerate() {
        backend.decode_step(&mut daemon_state.gpu, tok, i)
    }
```

It is a **warm-pass**, described correctly in its own comment and reported to
the user as `pp512 t/s`. The deepseek4 arm a few lines above is explicit that
this is *"not the production prefill path (that's
`forward_prefill_batch_chunked` in `generate`)"*.

So every figure in the retracted document was a decode loop measured 512 times.

## Every "finding" it produced, explained by that alone

| observation | claimed meaning | actual cause |
|---|---|---|
| pp512 37.40 vs tg128 36.28 | prefill = decode cost | pp512 **is** tg128, run N times |
| ms/token flat 9.91 / 9.99 / 10.33 across 8x | no amortisation | nothing to amortise; it is decode |
| no `weight_gemm` fallback warning | batched kernels selected | `weight_gemm` never called |
| `SimpleAr::prefill` gets the whole prompt | batching not defeated upstream | true, and irrelevant — bench does not call it |

It also explains the "sharpest lead" the retracted document flagged: a recorded
gfx1103 A/B at **pp512 602 t/s** against **37.40** here. Both numbers were
right. The 602 measured the real batched path; the 37.40 measured a decode loop.

## What survives

- **The `examples/infer_hfq` profile in `decoder-layer-npu-scope.md`** — 85008
  `gemv_oq4_grouped` dispatches, no `gemm_*` kernel — is untouched by this. It
  used rocprofv3 on a real forward, not `bench_prefill`. Its own caveat stands
  unchanged: it is not the daemon, and confirming the daemon "requires profiling
  the daemon through its JSON-lines protocol". **`hipfire bench` is not that.**
- **The argument that NPU decode is the wrong target** — `decoder-layer-npu-scope.md:92`
  "Decode does NOT need the NPU", and the ~38 tok/s NPU ceiling in
  `wire-in-r6-prefill-offload.md:404` against GPU decode at 102.41 — rests on
  measurements taken elsewhere and is unaffected.
- **The lm_head work** (F32 expansion, `gemv_bf16_xf32`, BF16L3 packed head) is
  GPU-measured through `hipfire bench`'s decode path, which is what that path
  legitimately measures. Unaffected.

## ANSWERED: production prefill IS fully batched

Measured 2026-08-06 with `HIPFIRE_KERNEL_TRACE=1` (added for exactly this) on a
real `generate` request, so the path is
`SimpleAr::prefill` → `llama::prefill_forward`:

    [kernel-trace] llama prefill_forward (211 tokens): 693 dispatches across 11 kernels
            211   30.4%  embedding_bf16
            112   16.2%  gemm_oq4_residual_mmq
            112   16.2%  quantize_q8_1_mmq_ds4
            112   16.2%  rotate_x_mq_awq
             33    4.8%  rmsnorm_f32_gfx1151
             32    4.6%  add_inplace_f32
             32    4.6%  kv_cache_write_q8_0_batched
             16    2.3%  attention_causal_batched
             16    2.3%  rope_batched_f32
             16    2.3%  silu_mul_f32
              1    0.1%  gemv_bf16_xf32

**693 dispatches for 211 tokens — 3.3 per token, against the warm-pass's 371.**

Every count is the batched shape:

* `gemm_oq4_residual_mmq` **112** = 7 projections x 16 layers. One GEMM per
  projection for the WHOLE prompt, not per position. A GEMM kernel, not a GEMV.
* `attention_causal_batched` **16**, `rope_batched_f32` **16** — once per layer.
* `kv_cache_write_q8_0_batched` **32** — batched, two per layer.
* `gemv_bf16_xf32` **1** — the lm_head runs at the last position only. The
  "skip lm_head for non-final positions" win is already implemented.
* `embedding_bf16` **211** — one per token, which is correct: a gather reads one
  row per token by definition.

So the daemon's prefill does not have the problem. The `examples/infer_hfq`
profile in `decoder-layer-npu-scope.md` — 85008 `gemv_oq4_grouped` dispatches,
no `gemm_*` — describes that example, not this path, exactly as its own caveat
warned.

### MEASURED: the prefill gap is ~13%, not 34x

Timed inside the same trace, so the rate and the kernel list come from one call:

| prompt | prefill | tok/s |
|---|---|---|
| 403 tokens | 167.9 ms | **2400.0** |
| 1339 tokens | 1034.3 ms | 1294.6 |

`decoder-layer-npu-scope.md` records **hipfire prefill 80.8 t/s** against FLM's
**~2750**, and calls prefill "the only losing axis" — the justification for a
prefill-only NPU offload as "the smallest change that flips the last axis".

**Production prefill is ~2400 t/s at 403 tokens, 30x that 80.8.** The 80.8 has
the signature of `bench_prefill`: a per-token decode warm-pass reported as
`pp512 t/s`. Against FLM's ~2750 the real gap is roughly **13%**, not 34x.

Two honest caveats. FLM's ~2750 is not stated at a prompt length, and hipfire's
rate falls with length — 2400 at 403 tokens, 1294.6 at 1339 — because attention
is O(n^2), so a like-for-like comparison needs FLM's length. And 2400 t/s is one
host, one artifact (`Llama-3.2-1B-Instruct--oq4++`, gfx1151).

**Consequence for the NPU plan — narrower than first written.** An earlier
revision of this section said every row of the scope table was suspect
"including the 35B's 36.4 t/s". That was an overcorrection. `bench_prefill`
dispatches per arch, and only some arms are warm-passes:

| arch | `bench_prefill` arm | prefill figure |
|---|---|---|
| qwen3.5/3.6 family (incl. **35B-A3B**) | `qwen35::forward_prefill_batch` | **real prefill** |
| LLaMA / Qwen3 (arch 0/1) | per-token `decode_step` | warm-pass — not prefill |
| Qwen2 | per-token `forward_step` | warm-pass |
| DeepSeek V4, MiniMax-M2, LFM2.5-MoE | per-token `decode_step` | warm-pass |

So the two rows of the scope table land differently:

* **llama-1B prefill 80.8 t/s is not a prefill number.** Measured properly it is
  ~2400 t/s, and the gap to FLM's ~2750 is ~13%, not 34x.
* **35B-A3B prefill 36.4 t/s IS a prefill number** — the qwen35 arm runs the
  batched path. Against FLM's ~290 that is a genuine ~8x gap.

**The MoE case for a prefill NPU offload therefore survives, and it is the case
the scope doc actually makes** — "a prefill-only offload for the MoE is the
smallest change that flips the last axis", noting the MoE is `prefill = "full"`
in `model-support.toml` where llama is only `partial`. What does not survive is
the llama row, and any argument resting on it.

Still worth re-deriving the 35B figure with the trace, to confirm the batched
path is what runs and to get a rate at a stated prompt length. But the premise
is not refuted there.

## Instrument fix

`bench_prefill` should either call the production prefill path for arch 0/1, or
stop reporting `pp512 t/s` for arches where it runs a warm-pass. Tracked as the
motivating case in `~/measurement-integrity-goal.md`.

## Lead: the qwen35 batched prefill is GEMV-dominated

Traced `bench_prefill` on `Qwen3.5-0.8B--mq4` (qwen35 family, so the arm runs
the real `forward_prefill_batch`), 256 tokens — one chunk, since
`PREFILL_MAX_BATCH = 256`:

| kernel | dispatches |
|---|---|
| `gemv_hfq4g256_residual` | **4608** |
| `gemm_hfq4g256_residual_mmq_full_set` | 62 |
| `gemm_hfq4g256_residual_mmq` | 44 |

32736 dispatches total across 31 kernels for 256 tokens — **128 per token**.

GEMMs are dispatched, so the path is not wholly per-position. But GEMV outnumbers
GEMM **43:1** inside a call that batches the entire prompt in one chunk. Some
projections are taking a batched GEMM and most are not.

That is worth chasing before any NPU offload: the 35B's ~8x prefill gap against
FLM is measured on this same path, and a GEMV-dominated "batched" prefill is a
GPU-side inefficiency, not evidence that the work belongs on the NPU. If most
linears can be moved onto the existing GEMM kernels, the gap may close without
new hardware.

Next: identify which projections fall back and why — the LA (linear-attention)
QKVZA/gate_up/wo/w_down set in `forward_prefill_chunk` is where the counts point.
`HIPFIRE_KERNEL_TRACE=1` plus the per-arm dispatch counts is enough to attribute
them without a profiler.

## ROOT CAUSE: the KV cache format, not the NPU

The 43:1 GEMV dominance is a gated fallback, and the gate now says why.
`prefill_chunk.rs` splits the batched-FA condition and reports it under
`HIPFIRE_KERNEL_TRACE`:

    FA layers take the PER-TOKEN fallback: kv_ok=false (quantized=true,
    q8=false, asym4=false, asym3=false, asym2=false), first non-batchable
    FA weight dtype=<none — weights all batchable>

**The weights are fine. The KV cache format alone disqualifies batched FA
prefill.** The accepted set is `q8 | asym4 | asym3 | asym2`; the shipped default
in `~/.hipfire/config.json` is `kvarn4`, which is not in it.

Same model, same build, `kv_cache` the only variable — `Qwen3.5-0.8B--mq4`,
gfx1151:

| kv_cache | pp512 | tg64 |
|---|---|---|
| **q8** | **6307.60 t/s** | 103.40 |
| kvarn4 (default) | 598.30 t/s | 101.85 |

**10.5x prefill. Decode unchanged within noise.**

For scale: FLM does ~2750 t/s prefill on the 1B. hipfire with a q8 KV cache does
6307 — **2.3x faster than FLM**, on the axis the NPU work exists to fix.

### Why this is NOT simply "add kvarn to the gate"

The obvious patch faults. `prefill_chunk.rs` carries the reason at three sites:

> KVarN K = 4-bit block records + fp16 window (not a contiguous buffer), so the
> generic batched writes below would fault.

KVarN's K cache is not a contiguous buffer, and the batched KV write and fused
attention in the FA layer body assume contiguity. The gate is a correctness
precondition, correctly placed. The linear-attention path already carries KVarN
branches for exactly this; the FA prefill path does not.

**So the work is: give the FA prefill path a KVarN batched write + attention
branch, mirroring the LA path's.** That is real kernel work, but it is bounded,
it is on the GPU, and it is worth up to ~10x prefill for every model on the
default KV format.

### What this means for the NPU plan

Prefill was "the only losing axis". On the 1B it is not losing at all once the
KV format allows the batched path — it beats FLM by 2.3x. The 35B has not been
measured this way yet and is the model that matters, but the mechanism is
per-model dtype/KV checks, not anything model-specific.

Before any NPU prefill offload is scoped: measure the 35B with a q8 KV cache. If
it moves the way the 1B did, the offload is solving a problem that a KVarN
batched-FA branch solves on hardware already present.
