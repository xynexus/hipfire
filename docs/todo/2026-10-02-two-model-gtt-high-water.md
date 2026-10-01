# TODO: ~53 GiB of GTT above the weights with two models serving a swarm

Status: TODO (bounded, not explained). Date: 2026-10-02.
Models: Qwen3.8-27B--oq4.25++ + Qwen3.6-35B-A3B--oq4.25++ co-resident, gfx1151
(APU: GTT is host RAM, 116 GiB cap of 125 GiB).

## What is fixed (same night)

Unbounded growth over a Corrode CAE swarm turn (24 -> 94..107 GiB, host down to
11-24 GB free) had four causes, all fixed and verified by replaying captured turns:

| cause | fix |
|---|---|
| cold prefill checkpointed every turn boundary (~45 at once) | `46a302380` cap 4 |
| GPU pool cached every freed buffer until device memory ran short | `313b65fb1` 4 GiB cap, >= 256 MiB never cached |
| finished sessions released only at cycle end; with mid-cycle admission a cycle never ends | `313b65fb1` release every step |
| prefix-index eviction released other models' checkpoints to the wrong worker (40 resident vs cap 16) | `b8507d5ac` per-worker cap |

Now: resident sessions = 16 checkpoints per model + live requests, flat across
the run (`qwen35 release` debug line, `hipfire_serving_core::session=debug`).

## What is not explained

Both models loaded, nothing resident: **40.8 GiB** (27B 17.7 + 35B 23.1).
Replaying a captured two-model turn (178 requests, 3 concurrent, outputs capped at
256 tokens), twice back to back on one instance: idle **94.9 GiB, then 93.8 GiB**
-- a plateau, not a leak, but ~53 GiB of state.

Accounted for: 32 checkpoints (16 per model). A cold 13K-token 27B request measured
+1.6 GiB for 1 session + 4 checkpoints, so ~0.4 GiB per checkpoint at that length:
~13 GiB. Pool cache <= 4 GiB (logged ~0.4). Live sessions ~1-2 GiB. That leaves
~35-40 GiB.

Leads, in order:
1. Grow-only scratch: something sized by the largest prompt/batch seen and kept
   per model (prefill activations, GDN chunk state, attention partials -- the old
   tile path's `partials` is n_heads x max_tiles x (2+head_dim) x rows x 4 B, up to
   ~1.6 GiB at 32K context and 256 rows). Not in the pool stats if hipMalloc'd
   directly.
2. A checkpoint is a fork of its source session and keeps the SOURCE'S capacity
   (prompt + that request's max_tokens; `qwen35_fork_session_state`), not its own
   prefix length. With Corrode's 8192-token output cap that is up to 8K tokens of
   empty KV per checkpoint (~200 MiB on the 27B). A checkpoint is never decoded
   into; it could be compacted to its prefix at mint time.
3. The fresh-start run that loaded the 35B mid-way plateaued lower (75.6 GiB) than
   both-loaded-first (94 GiB), so placement/fragmentation may matter.

How to measure: `rocprofv3 --memory-allocation-trace` on `hipfire serve` during a
`replay.py` of a captured turn (see the Corrode memory note on the request-capture
proxy), then sum live allocations by call site at idle.

## Safety meanwhile

The memory watchdog kills the daemon below 16 GB MemAvailable. At the 94 GiB
plateau the host keeps ~28 GB. Lower `HIPFIRE_SERVER_PREFIX_CACHE_MAX` (per model
now) if running heavier than this.
