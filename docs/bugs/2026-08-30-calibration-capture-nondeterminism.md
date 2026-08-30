# Calibration capture is nondeterministic; inference is not

Status: **ROOT-CAUSED AND FIXED** — 2026-08-30, nix2 (gfx1103). A data race in
`zaya_value_compose_f32` when `num_key_value_heads == 1`. Two earlier suspects in
this document (the gather skip, then attention itself) are both DISPROVEN; the
trail is kept because the eliminations are what made the last step obvious.

Chasing the flaky `zaya/kld:oq4.25++` tiny-quant cell led to something larger
than a flaky cell. `tiny_quant_probe collect` — the instrumented forward that
captures Hessians and imatrices — **does not produce the same artifact twice**.

This matters beyond the gate: that same capture path is what produces the
`.calib.hfq` behind every `oq4+` / `oq4++` / `oq8+` / `oq8++` artifact. Two
calibration runs over identical inputs can yield different quantized weights.

## What was measured

Each stage was isolated on a fixed zaya fixture (seed 42), GPU lock held:

| stage | runs | result |
|---|---|---|
| `--emit-fixture zaya --seed 42` | 5 | **deterministic** (content-identical) |
| `--format q8f16` anchor from one fixture | 5 | **deterministic** |
| `--format oq4.25++` from one calib | 5 | **deterministic** (also with `HIPFIRE_OQ_RAGGED_Q8=1`) |
| `ar-hash` (plain greedy decode, exact logit hash) | 10 | **deterministic**, `0x136dd20f13e71b86` ×10 |
| `collect` (calibration capture) | 10 | **9 identical, 1 different** |
| `collect` with `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3` | 10 | **6 distinct** |

Inference is bit-exact. Only the capture path moves.

## What the divergence looks like

When `collect` diverges, **all 54 captured tensors differ** — Hessians and
imatrices, every layer, including `model.embed_tokens.hessian`, which precedes
routing. The differences are not last-bit: on the embed Hessian, 60% of stored
bytes differ, mean byte delta ~50, only 15% of those are ±1.

The metadata narrows it to a single decision. Both runs record
`n_calib_tokens: 24`, `n_hessian: 19`, `n_imatrix: 35` — identical. But
`per_tensor_tokens` differs in exactly two entries:

    model.layers.0.mlp.experts.2.down_proj:     10 vs 9
    model.layers.0.mlp.experts.2.gate_up_proj:  10 vs 9

One token's routed-expert assignment moves. Everything else follows.

## Ruled out

- **Not an async race.** `AMD_SERIALIZE_KERNEL=3` + `AMD_SERIALIZE_COPY=3` made
  it strictly WORSE (6 distinct results instead of 2). A missing stream sync
  would have been suppressed by serialization, not amplified. Amplification
  under changed allocation/scheduling timing is the signature of reading memory
  whose contents depend on what was there before.
- **Not the router tie-break.** `zaya_router_select_f32` is single-threaded
  (`blockIdx.x != 0 || threadIdx.x != 0` returns) and uses strict `>`, so equal
  scores deterministically pick the lowest expert index.
- **Not unzeroed accumulators.** `allocate_acc` takes every buffer from
  `gpu.zeros`, which is a real `memset`/`memset_async`, not a bare alloc.
- **Not the corpus.** `collect` defaults to `--seed 42` and the battery never
  overrides it; 9 of 10 runs agree exactly, which a varying corpus would not do.
- **Not the crest atomics.** `calib_group_crest_reduce_f32` does use a float
  `atomicAdd`, but it feeds crest telemetry, not the Hessian or imatrix, and the
  Hessian kernel (`calib_hessian_outer_f32`) is a fixed-schedule tiled GEMM with
  one thread per output and no atomics.

## Where to look next

The staging tile in `calibration.rs` is zeroed **once at allocation** and reused
across flushes (`flush_result` sets `buf_rows = 0` without re-zeroing). The
gather that fills it, `calib_gather_rows_f32`, *skips* writing any row whose
slot index is negative:

    int flat = sorted_slot_index[sorted_start + row];
    if (flat < 0) return;

A skipped row inside `0..rows` is still inside `buf_rows` and is still reduced —
so it would contribute the PREVIOUS tile's values rather than nothing. That is
the one place found where the capture can read a row nobody wrote this tile, and
it fits "depends on what was there before". It has not been confirmed as the
live cause: the admission planner in `calibration/expert_capture.rs` is supposed
to exclude padding, so proving it needs instrumentation of whether a negative
slot is ever admitted, not just inspection.

## Impact today

The tiny-quant gate is green (`5f6bb9dd6`) because the affected baselines were
re-recorded and the ±0.25 relative tolerance absorbs the variation on every cell
except the smallest. `zaya/kld:oq4.25++` at ~2e-5 KLD was the only cell where the
spread crossed its budget, which is why it presented as a "bimodal" flake. It is
inside tolerance now, not fixed.

Production calibration is affected in principle — the same code path, without the
tolerance. Whether the resulting weight differences matter at real model scale is
unmeasured; the fixture is random-init and tiny, where a single re-routed token
is a large fraction of the evidence.


## Instrumented follow-up — the gather hypothesis is dead

`HIPFIRE_CALIB_TRACE=1` (added in `calibration.rs`) prints, per reduction, the
names, row count, and an FNV-1a hash of the exact staged rows that reduction is
about to consume — plus a line whenever the gather is handed a negative slot
index.

**12 traced `collect` runs, 3 of which diverged. `NEGATIVE SLOTS` never printed
once.** The `calib_gather_rows_f32` skip path named above is NOT the cause; the
staging tile is fully written every tile. That hypothesis is closed.

### Reading the trace correctly

Flush *order* varies between runs even when the resulting artifact is
byte-identical, because independent accumulators interleave differently. Order
is therefore not signal. Comparing each tensor's OWN ordered flush sequence is:
two runs that produced identical calibs show **0** tensors differing, which is
what makes the method trustworthy.

### Where it actually diverges

On a divergent run, 31 of 35 captured tensors differ. The four that AGREE are
the decisive ones:

    model.layers.0.self_attn.qkv_proj.q_proj          IDENTICAL
    model.layers.0.self_attn.qkv_proj.k_proj          IDENTICAL
    model.layers.0.self_attn.qkv_proj.v_proj_current  IDENTICAL
    model.layers.0.self_attn.qkv_proj.v_proj_delayed  IDENTICAL

    model.layers.0.self_attn.o_proj    24 rows, DIFFERENT hash   <-- first divergence

Those four capture the *input* to the QKV projections; `o_proj` captures the
*output* of attention. Identical input, divergent output, same row count.

**Layer-0 attention turns identical inputs into different outputs.** Everything
downstream — the router MLP, expert routing (which is what shifts expert 2 from
10 tokens to 9), every later layer, and the tied-embedding capture — inherits it.
`model.embed_tokens` is the lm_head input, i.e. the final hidden state, so its
divergence is a consequence and not, as first assumed, an early cause.

### Also ruled out this round

- **Not the token stream.** `synthetic_tokens` is a pure SplitMix64 over
  `(len, seed)`, seed defaults to 42, and the battery never overrides it.
- **Not the KVarN scratch.** `gpu.rs` allocates `tiles`, `flash_partials`, and
  `positions` with `alloc_tensor` (uninitialized) rather than `zeros`, which
  looked like the answer. It is not: `HIPFIRE_ZAYA_KVARN` is **opt-in** and
  unset in these runs, so that whole branch is dead code here — the f32 `k_cache`
  / `v_cache` rings actually used are allocated through the zeroing helper.
  Switching those three to `zeros` was tried and changed nothing (3 distinct
  results in 16 runs), consistent with the branch never executing. Reverted.

### Next step

The question is now narrow and does not involve calibration at all: **why does
zaya layer-0 attention produce a different output for a bit-identical Q/K/V on
gfx1103?** Worth noting that `ar-hash` (plain greedy decode, no capture armed)
is bit-exact across 10 runs, so whatever it is, arming the capture is part of
reproducing it — the capture hooks add allocations and downloads around the same
kernels, which is the kind of timing/occupancy change that surfaces a latent
multi-wave or LDS hazard. gfx1103 has a documented one (README "gfx1103 /
Phoenix LDS status", HIP-719 / CWSR), which makes the attention reduction on this
arch the first place to look.


## Root cause: `zaya_value_compose_f32` races when `nkv == 1`

Bisected by hashing tensors along the calibration forward. Per layer, in order:

    L0.vcur_postgemv    identical
    L0.vdel_postgemv    identical
    L0.vcur_precompose  identical
    L0.vdel_precompose  identical
    L0.q_in             identical
    L0.k_in             identical
    L0.v_in             DIFFERENT   <-- first divergence
    L0.attn_out         DIFFERENT   (consequence)

So attention was innocent — as its structure already implied: one thread per
(token, head), private output row, no atomics, no LDS, no shuffles. Its `value`
input was already wrong. And `v_cur`/`v_del` are bit-identical immediately before
the compose, so the compose kernel turns identical inputs into different output.

The kernel writes the composed value as two KV heads:

    value[(t*nkv + 0)*hd + d] = v_cur[t*hd + d];
    value[(t*nkv + 1)*hd + d] = (t == 0) ? 0.0f : v_del[(t-1)*hd + d];

The zaya toy fixture is `heads: 2, kv_heads: 1, head_dim: 128`. With `nkv == 1`,
`(t*1 + 1)*hd` is **token t+1's head 0**. Thread `t` writes `v_del[t-1]` to the
exact address thread `t+1` writes `v_cur[t+1]` to — same location, different
values, winner decided by scheduling. A data race, from bit-identical inputs.

Two more consequences of the same off-by-one:

- The `t = s-1` thread writes one full head *past the end* of the
  `s*nkv*hd` buffer.
- Attention computes `groups = nq/nkv`, so with `nq=2, nkv=1` every query head
  reads value head 0. The delayed half is never consumed, while its write
  corrupts the half that is.

This explains every earlier observation: identical inputs with divergent output,
rarity (one order usually wins), amplification under `AMD_SERIALIZE_*` (changed
scheduling), invisibility to plain decode (`gpu_forward_calib` prefills all 24
tokens at once, so `s > 1` and adjacent-token threads coexist; decode runs s=1
and cannot race), and the Heisenbug where adding a `download_f32` probe
serialized layer 0 and pushed the divergence to layer 1.

`nkv >= 2` is structural for a compositional value, not a tuning choice.

## Fix

- `zaya_value_compose_f32` (dispatch) now **refuses** `nkv < 2` with a message
  naming the aliasing, rather than computing a silently wrong answer — the house
  rule from the 2026-08-29 hunt.
- The zaya toy fixture takes `kv_heads: 2`, which is what the design requires and
  what makes the tiny gate exercise the real path. All 7 zaya baselines were
  re-recorded (the geometry changed, so every one is stale by construction);
  `oq8` fell from 3.85e-6 to 2e-8, i.e. near-lossless once the value tensor stops
  being corrupted.

## Verification

- `collect` × 16 on the corrected fixture: **1 distinct artifact** (was 2 of 10,
  and 6 of 10 under `AMD_SERIALIZE_KERNEL=3`).
- `tiny-quant-gate.sh`: **187 pass / 0 fail, twice consecutively.**

## Scope

The refusal is the part that matters beyond this fixture. Any zaya artifact with
`num_key_value_heads == 1` was taking the same racing, out-of-bounds path in
prefill — including production calibration, where nothing would have flagged it.
It now fails loudly at dispatch instead.
