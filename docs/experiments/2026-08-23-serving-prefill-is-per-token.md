# ⚠️ Serving prefill never reaches the batched path — 15.4 vs 286 tok/s

State box: halo, gfx1151, Qwen3.8-27B oq4.25++ +CASK, kvarn KV, MAX_BATCH=512,
ROCm 7.14. Found 2026-08-23 while trying to validate W4A4 quality.

## The finding

`bench_qwen35_speed` and the **daemon** disagree by ~19x on the same model, same
machine, same prompt shape:

| harness | prefill |
|---------|---------|
| `bench_qwen35_speed --prefill 2059` | **286-329 tok/s** |
| `hipfire-daemon` (stdin protocol, 720-token prompt) | **15.4 tok/s** |

A `rocprofv3 --kernel-trace` of the daemon says why, unambiguously:

```
DAEMON kernel profile (45.8 s GPU):
  gemv_oq_compact_grouped_v3      361,097 calls   41,917.8 ms   91.4%
  gated_delta_net_f32              34,944 calls    1,691.0 ms    3.7%
  attention_flash_kvarn_tile_batched 11,648 calls    531.8 ms    1.2%
```

**361,097 GEMV calls to prefill 720 tokens**, and ZERO calls to
`gemm_oq_compact_iu4x2_w64`. The daemon prefills with the per-token DECODE
kernel — one full weight sweep per token.

## It is not "declined", it is not reached

Two independent eligibility diagnostics exist and BOTH stay silent under the
daemon:

- `[kernel-trace] pbs_eligible inputs: ...` (`qwen35::prefill_batch_pbs_eligible`,
  under `HIPFIRE_KERNEL_TRACE=1`)
- `[prefill-eligible] final=... base=... kv_f32=...`
  (`prefill_batch.rs`, under `HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1`)

Neither prints. So `forward_prefill_batch` is never entered and the gate is never
evaluated — this is not a gate returning false, it is a different code path.
Ruled out along the way: `--state fp32` (the `dn_state.quant ∈ {FP32,FP16}`
condition) changes nothing, and the prompt is far above `MIN_BATCH`.

## Why this matters more than any kernel work

Every prefill optimization in this session and the ones before it was measured
through `bench_qwen35_speed`:

  216.3 -> 256.3 tok/s from the goal round (overlay, swizzle, zero-seed, hoist)
  +14.9% more from opt-in W4A4

**None of it is on the path serving uses.** Against a 15.4 tok/s serving
baseline, reaching the batched path at all is worth ~19x — an order of magnitude
more than the ~1.2x the kernel work bought, and it is a serving-integration
problem, not a kernel problem.

It also blocks quality work: there is currently no route from a text-producing
harness (`coherence_probe`, `hipfire eval`, the daemon) to the batched GEMM, so
W4A4 activations cannot be validated end to end. That is why W4A4 ships opt-in
and off.

## Open questions for whoever picks this up

1. Which prefill entry does the daemon actually call for arch qwen35? It is not
   `forward_prefill_batch`. Start by tracing the call path from
   `DaemonRequest::Generate` down, rather than from the gate up — the gate is a
   dead end by construction here.
2. Is `bench_qwen35_speed` representative of what serving WOULD do once it
   reaches the batched path, or does it bypass state/KV bookkeeping serving
   needs? If the latter, the 286 tok/s figure is optimistic and needs restating.
3. Does DFlash/spec-decode force the per-token route (`tape_in_play`), and is it
   on by default for this artifact?
