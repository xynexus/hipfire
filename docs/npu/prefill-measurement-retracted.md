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

**Consequence for the NPU plan.** The premise that prefill is 34x behind does not
survive. Whatever case exists for a prefill NPU offload has to be rebuilt on
measured numbers, at a matched prompt length, with the path shown. The
`decoder-layer-npu-scope.md` table should not be used until its hipfire column
is re-derived — every row in it is suspect for the same reason, including the
35B's 36.4 t/s.

## Instrument fix

`bench_prefill` should either call the production prefill path for arch 0/1, or
stop reporting `pp512 t/s` for arches where it runs a warm-pass. Tracked as the
motivating case in `~/measurement-integrity-goal.md`.
