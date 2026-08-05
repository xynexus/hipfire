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

## What is now open again

**Production prefill has never been measured on this host.** Whether the daemon
batches prefill is unknown, in either direction.

To measure it, drive a real `generate` request through the JSON-lines protocol
(see the `run-model` skill) so the path is
`SimpleAr::prefill` → `llama::prefill_forward`, and profile that. Do not use
`hipfire bench` for any prefill claim until its handler is fixed.

## Instrument fix

`bench_prefill` should either call the production prefill path for arch 0/1, or
stop reporting `pp512 t/s` for arches where it runs a warm-pass. Tracked as the
motivating case in `~/measurement-integrity-goal.md`.
