# Session: give qwen4_exp a batched multi-token forward

**Blocked on:** nothing technical. **Est:** multi-session.
**Value:** the largest user-visible win available on this box.

## Objective

`qwen4_exp` (arch 26, Qwen3.8-Flash-Next 180B) prefills **one token at a time**.
Give it a batched forward so a real prompt is usable.

Done when prefill scales sub-linearly with prompt length and a 512-token prompt
completes in seconds rather than minutes, with decode output unchanged.

## Why now — measured, not estimated

`serve_real <model> <steps> <prompt_len>`, 8 GiB expert budget, warm:

| prompt tokens | prefill | tok/s |
|---|---|---|
| 8 | 2.35 s | 3.4 |
| 16 | 4.64 s | 3.4 |
| 32 | 6.49 s | 4.9 |
| 64 | 10.95 s | **5.8** |

Linear, as a per-token forward must be. **Extrapolated: a 2048-token prompt costs
~6 minutes before the first output token**; 512 tokens costs ~88 s.

`Qwen3.6-35B-A3B--oq4.25++`, which HAS batched prefill, does **224.6 tok/s at
pp32 and 360.1 at pp128** on the same box. A **40-60x gap**, and it is entirely
the batched/per-token distinction — the 180B's decode is 0.18 s/tok, so prefill
and decode run at comparable rates, which is the signature of a prefill that
never batches.

**This is a usability bug, not a spec-decode enabler.** An earlier framing called
it "gates everything speculative", which undersells it — speculation is a
second-order benefit of the same work.

## The blocker, and the reference implementation

`decode_step_into` advances exactly one position, and both recurrent halves are
sequential by construction:

- **Gated DeltaNet** state
- the **PLE conv ring**

The qwen35 side already solved the same shape: `gated_delta_net_f16` has a
chunked-scan treatment. That is the reference to copy, and it is why this is a
session rather than a patch.

## First moves

1. Read how qwen35 batches its DeltaNet (`gated_delta_net_f16`, and the chunked
   SSD prefill work) — the state recurrence is the hard half.
2. Decide the PLE conv ring separately; a conv over a known window may batch more
   easily than the SSM.
3. Land the two independently if possible, measuring after each.

## The verification bar

- `./tests/qwen4exp-gate.sh` PASS, including the paged arm asserting bit-identical
  argmax against the resident path.
- `serve_real` prefill scaling re-measured at 8/16/32/64 — the slope must bend.
- **Decode argmax unchanged**: 1892 (13.9764) on the standard prompt.

## Traps

- **Discard the first run after any rebuild** — kernels JIT-compile inside the
  timed window (3.45x, with tau bit-identical).
- **Rebuild the example, not just the workspace.** `cargo build --workspace` did
  not relink `serve_real`, and a stale binary produced a wrong "no effect"
  reading that only the pager counters exposed.
- The paged-expert path resolves both projections per expert in one pager
  round-trip (`ExpertStack::expert_pair`); keep residency ensured immediately
  before use, because a tight budget can evict expert 1 while ensuring expert k.
