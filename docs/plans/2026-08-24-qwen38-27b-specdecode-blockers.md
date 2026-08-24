# Qwen3.8-27B toward 55 tok/s: two blockers, one of them free

**Date:** 2026-08-24 · **Box:** halo gfx1151 · **Model:** `Qwen3.8-27B--oq4.25++.hfq`
(14.4 GiB, 64 layers: 16 FullAttention + 48 LinearAttention, dim 5120)

Measured baseline, daemon stdin protocol, greedy:

| config | decode tok/s | prefill tok/s |
|---|---|---|
| plain decode, **fp32 KV (the default)** | 14.5 | **15.5** |
| plain decode, kvarn8 | 14.5 | 270.1 |
| plain decode, kvarn | 14.5 | **274.6** |
| dflash2 drafter + kvarn | **5.7** | 64.5 |
| dflash drafter + kvarn | 4.0 | 64.4 |

## Blocker 1 — fp32 KV silently disables batched prefill (17.7x, free)

`prefill_batch.rs`:

```rust
let kv_f32 = !kv_cache.quantized && !kv_cache.quant_q8 && !kv_cache.quant_hfq4;
let eligible = eligible && !kv_f32 && !kv_asym2_tree;
```

F32 KV is disqualified because "F32 KV has only BatchEq(1) -> MissingImpl at
resolve". This model loads `KV cache: fp32` by DEFAULT, so every prompt token
takes a full weight sweep: 389 tokens = 25 SECONDS, flat 15.5 tok/s from 29 to
389 tokens. Switching to kvarn: **274.6 tok/s, 17.7x**, decode unchanged.

Note what this list contains. The modes that make a model eligible are `q8` and
`hfq4` — both now REFUSED at load as deprecated ("hipfire is retiring KV storage
down to two families: kvarn and unquantized"). `quant_kvarn`, the surviving
quantized family, is NOT in the predicate. It passes only incidentally, because
kvarn also sets `quantized = true`. Ten lines away in `prefill_chunk.rs`,
`fa_kv_ok` lists `quant_kvarn` explicitly, with a comment recording that omitting
it cost this same model 54 tok/s against 301. The two predicates disagree about
what KVarN is.

So: the surviving supported KV modes are fp32 (ineligible) and kvarn (eligible by
accident). That is worth making explicit before it regresses.

## CORRECTION to blocker 2, and the measured budget

The 7th argument to `prefill_batch_pbs_eligible` is **`tape_in_play`**, not
`force_fallback` (which is computed inside from env). I read it wrong first time.
It sets `allow_compact = !tape_in_play`, and only `tree_verify || gdn_tape` sets
it -- `hidden_rb` does not, by default.

And for a compact target the tape is ALREADY dropped before verify:

```rust
let verify_populates_tape = pbs_eligible(.., /* tape_in_play */ true) && kv_batched_capable;
let use_tape_replay = gdn_tape.is_some() && verify_populates_tape;
let mut gdn_tape_opt = if use_tape_replay { gdn_tape } else { None };
```

`pbs_eligible(.., true)` is false for compact, so `gdn_tape_opt` is None and
verify's own `tape_in_play` is false. Verify is therefore NOT simply per-token,
and "write a compact tape writer" is not the fix I claimed it was.

**Measured budget** (`HIPFIRE_SPEC_PHASES=1`, B=8, dflash2, kvarn):

```
draft=52ms  verify=322ms  replay=67-336ms  total=376-712ms   (tau 2.28)
```

Plain decode is 69 ms/token. A fully batched verify of 8 tokens should cost ~1
sweep (~69ms); it costs 322ms, about 4.7 sweeps. So verify is partially batched
and there is ~4.7x still on the table there -- but that is not where the target
lives.

### Why tau, not verify, is the lever

Even with a PERFECT one-sweep verify: 52 + 69 + 67 = 188ms per cycle for ~2.3
accepted tokens = 82 ms/token = **12 tok/s, still below plain decode's 14.5**.
The fixed draft (52ms) and replay (67ms) terms dominate at low tau.

55 tok/s = 18 ms/token. At B=8 with near-full acceptance: (52 draft + 69 verify)
/ 8 = 15 ms/token = **66 tok/s**. That is the only shape of the budget that
reaches the target: acceptance close to B, so the fixed costs amortise.

And the daemon says acceptance is being left on the table, unprompted:

> DFlash2 candidate_selector (rank=256, top_k=16) is carried by this drafter but
> NOT applied — the draft path still takes a per-position argmax. Output stays
> correct (the target verifies every token); acceptance rate is below what this
> checkpoint can do.

So the order is: **apply the DFlash2 candidate selector** (tau 2.28 -> ?), then
the 4.7x verify, then the replay. Not the tape.

## Blocker 2 (as first written, superseded above) — the GDN tape

The eligibility call passes

```rust
tree_verify.is_some() || gdn_tape.is_some() || (...)
```

as **`force_fallback`**. Spec decode always carries a GDN tape, so verify always
falls back to per-token. Visible in the table: prefill drops 274.6 -> 64.5 the
moment a drafter loads, on the same KV mode.

That inverts the economics. With tau = 2.282 a batched verify should be ~2.3x
FASTER than plain decode; instead each verify costs K full weight sweeps, so
spec decode measures 5.7 against plain 14.5 — 2.5x slower. The
`.parked-slower-than-plain-decode` suffixes on both 27B drafters are accurate,
and they are accurate for a fixable reason.

The stated cause is that the tape has no compact writer, and that is real: an
earlier attempt to batch the compact verify took accept_rate 0.468 -> 0.000 with
the draft emitting random vocab ids. The fix is a batched compact GDN-tape
export, not removing the guard.

## What 55 tok/s requires

Plain decode is bandwidth-bound at ~14.5 tok/s: 14.4 GiB per token against a
measured ~248 GB/s gives ~17 tok/s as the hard ceiling, and decode is already at
~90% of it (see the phase-1 bandwidth work). **55 tok/s is unreachable without
spec decode** — it is 3.8x the single-sweep ceiling.

With verify batched, the arithmetic is tau x plain: 2.28 x 14.5 = ~33 tok/s, and
that is with today's drafter. 55 needs BOTH a batched verify AND tau ~3.8, or a
verify cheap enough that a larger K pays. So the order is:

1. batched compact GDN-tape export (unblocks everything; ~33 tok/s at current tau)
2. drafter quality / tau (33 -> 55 needs tau ~3.8)

Also relevant: a drafter is currently REFUSED on this target unless
`HIPFIRE_DFLASH_ALLOW_OPUS=1`, because the target's lm_head is qt=36 and Opus was
"measured slower than plain decode on this family". That measurement was taken
under per-token verify, so it should be re-taken once blocker 2 lifts.

## Reproduce

```sh
# blocker 1 — same model, only kv_mode differs
printf '%s\n' '{"type":"load","model":".../Qwen3.8-27B--oq4.25++.hfq","params":{"max_seq":4096,"dflash_mode":"off","kv_mode":"kvarn"}}' \
  '{"type":"generate","id":"p","prompt":"<389 tokens>","max_tokens":64,"temperature":0.0}' \
  '{"type":"unload"}' | hipfire-daemon
# HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1 prints the verdict and kv_f32 directly.
```
