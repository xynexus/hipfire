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


## The ceiling, computed — 55 tok/s is not reachable with this drafter

`HIPFIRE_DFLASH_BLOCK` (added here) sweeps the draft block size. Decode is nearly
FLAT in B, which is itself the finding:

| B | tau | accept_rate | decode tok/s |
|---|---|---|---|
| 2 | 0.853 | 0.426 | 5.59 |
| 3 | 1.37 | 0.457 | 6.18 |
| 4 | 1.75 | 0.438 | **6.23** |
| 6 | 2.25 | 0.375 | 6.19 |
| 8 | 2.421 | 0.303 | 6.13 |

Flat in B means a fixed per-cycle cost dominates, and the phase timings say which:

```
B=2   draft= 13.8ms   verify=310ms   replay=67ms
B=8   draft= 51.5ms   verify=322ms   replay=67ms
```

**Verify is constant in B.** It is not per-token (that would be 8x69=552ms at
B=8); it is a fixed ~4.5 weight sweeps per cycle where a batched forward should
cost ~1. Draft scales cleanly at ~6.5 ms/token.

### What a perfect verify would buy, and what it would not

At 1 sweep (69ms) and ignoring replay:

```
B=2  10.4    B=3  15.5    B=4  18.4    B=6  20.8    B=8  20.0   tok/s
```

Best ~20.8 at B=6 — a 43% win over plain decode's 14.5, and enough to unpark
both drafters. But 55 tok/s needs:

```
B=4: tau 5.22 (131% acceptance — impossible)
B=6: tau 5.94  (99%)
B=8: tau 6.65  (83%)
```

against a measured 30%. **55 tok/s is a drafter-quality bound, not an
implementation one.** No amount of verify or block-size work reaches it; it needs
a drafter that accepts ~5-6x more, or a target cheap enough per sweep that the
whole budget shrinks.

### The verify defect is still worth fixing

Verify IS batch-eligible (`final=true n=8`, confirmed 19/19 cycles), so it takes
the batched path and is still 4.5x off. The likely cause is kernel SHAPE:
`gemm_oq_compact_iu4x2_w64` is tuned for large-n prefill (N-heavy 2x8 tiling,
~50% of peak at prefill widths) and at n=8 it wastes most of its lanes while
still reading every weight byte. A multi-column GEMV in exactly this shape
already exists and appears in the profile —`gemv_oq_compact_multicol_b8`— so the
fix is likely routing small-n verify to it rather than writing a new kernel.

Expected: verify 310 -> ~70ms, spec decode 6.2 -> ~20 tok/s, beating plain decode
for the first time on this model.


## CORRECTION: I measured tau on one prompt, and it was the worst one

Everything above computes the ceiling from tau = 2.42. That number came from a
single prose prompt, and tau is strongly prompt-dependent:

| prompt | tau | accept_rate | decode tok/s |
|---|---|---|---|
| prose (all analysis above) | 2.20 | 0.275 | 5.75 |
| **code** | **5.333** | 0.667 | **13.29** |
| list | 5.125 | 0.641 | 13.23 |
| repeat | 4.765 | 0.596 | 11.73 |
| json | 3.174 | 0.397 | 7.42 |

So the drafter is not weak — it is 2.4x better on code-like text than on the one
prompt I benchmarked, and 5.333 matches the tau 4.875 the multicol routing
comment in `quant.rs` recorded. The "55 tok/s is a drafter-quality bound"
conclusion was drawn from the worst case and is WITHDRAWN.

Spec decode already beats plain decode (14.5) on nothing yet — 13.29 is close —
but the projection changes completely with a representative tau. Fixing verify
from 322ms to one sweep (69ms):

| prompt | now | with verify fixed |
|---|---|---|
| prose | 5.75 | 17.0 |
| code | 13.29 | **36.0** |
| list | 13.23 | **38.1** |
| repeat | 11.73 | 31.1 |
| json | 7.42 | 18.2 |

That is 2.5-2.9x, and every prompt then beats plain decode.

### What 55 actually needs now

On a code-like prompt (tau 5.333) the cycle must fall to 97ms. Draft at B=8 is
52ms, leaving 45ms for verify+replay — under one weight sweep (69ms). At B=6
draft is 39ms, leaving 58ms. So 55 tok/s needs BOTH:

1. verify at roughly one sweep (the 4.5x defect), and
2. the draft phase cheaper than 6.5 ms/token, or a B/tau point that beats
   B=8's 5.333.

It is not out of reach the way the prose-only analysis implied — it is one
kernel-shape fix plus a draft-cost win away, on the prompts that matter.

### Method note

Benchmark spec decode on a PROMPT MIX, never one prompt. Acceptance is a property
of the text, and a single prose prompt understates this drafter by 2.4x — enough
to have closed the goal as impossible.


## Where 55 tok/s comes from: it is the roofline

With the multicol routing and batched rollback in, the code prompt measures
18.87 tok/s at tau 5.333, and the budget model now predicts it to within 1%:

```
now:             draft 52 + verify 229 = 281ms ->  18.9 tok/s   (measured 18.87)
perfect verify:  draft 52 + verify  58 = 110ms ->  48.5 tok/s
+ draft floor:   draft 38 + verify  58 =  96ms ->  55.3 tok/s
```

58ms is ONE weight sweep of the 14.4 GiB target at the measured ~250 GB/s.
38ms is the drafter's own floor: 1.18 GiB is 4.8ms a sweep and drafting is
sequential, so B=8 cannot go below 8 x 4.8.

**55 tok/s is what perfect execution yields at tau 5.333.** Not an arbitrary
target — it is the roofline, and hitting it requires verify at ~100% of memory
bandwidth AND the drafter at its floor simultaneously. That is almost certainly
where the number came from.

Current standing against it:

| term | now | floor | gap |
|---|---|---|---|
| verify | 229ms (63 GB/s) | 58ms (250 GB/s) | **4.0x** |
| draft | 52ms | 38ms | 1.4x |

### The verify gap is instruction issue, not bandwidth

`gemv_oq_compact_multicol`'s own comment already diagnosed it: "bound by memory-
instruction ISSUE on activations that are already cache hits, not by weight
bandwidth". Per group-round only ~4 of ~32 memory instructions move weights, so
~12% of issue slots carry weight traffic — which lands almost exactly on the
measured 63 GB/s of a 250 GB/s part.

RW (rows per wave) is the knob the kernel already uses to amortize that, and it
is ALREADY at its optimum. Swept:

    RW=4  code 18.84   RW=6  code 16.90   RW=8  code 15.11

Higher RW regresses — `facc[RW][BC]` is RW*BC floats of accumulator, so RW=8 at
BC=8 is 64 VGPRs of accumulator alone and spills. Recorded so nobody re-runs it.

What is left is a shape change, not a knob: either widen the weight load to
`dwordx4` so each lane covers 32 weights (what `gemv_oq_compact_grouped_v3` does
to reach ~239 GB/s, against multicol's 4-byte dword), or stage the activation
tile in LDS so the 8 waves of a workgroup stop each re-loading it from global.
Both are real kernel rewrites.

## The binding constraint is the DRAFTER'S SIZE, not the kernels

Context: 55 tok/s is not a roofline guess — a closed-source engine reaches 60
tok/s on this exact GPU. So the question is what they do differently, and the
byte budget answers it.

Per spec-decode cycle we stream the drafter B times and the target once:

```
B=8, tau 5.333
  draft    9.4 GiB   39.6% of cycle bytes
  verify  14.4 GiB   60.4%
  total   23.8 GiB / 5.333 tokens = 4.47 GiB per token
  at the 233 GB/s ceiling -> 50.9 tok/s MAXIMUM
```

**50.9 is the ceiling with this drafter at 100% memory efficiency.** Not 55.
Every kernel in the path could hit peak bandwidth and the target would still be
out of reach, because the drafter is 1.18 GiB — 8.2% of the 14.4 GiB target —
and it is swept B times for every one target sweep.

Shrink the drafter and the ceiling moves immediately:

```
0.60 GiB drafter, B=8, tau 5.333 -> 63.2 tok/s max
0.30 GiB                          -> 72.2
0.12 GiB                          -> 79.0
```

A DFlash2 head of 5 layers at hidden 5120 is very large for a speculative
drafter; EAGLE-class heads are typically well under 1% of the target. At 8.2%
and B sweeps per cycle, ours spends 40% of the memory budget guessing.

### What this means for the remaining GPU work

The kernel work is close to done, and the isolated bench says so —
`gemv_oq_compact_multicol` (wide) at real 27B shapes:

```
gate/up B=8   224.4 GB/s   96.3% of the 233 ceiling
down    B=8   206.1        88.4%
gate/up B=1   231.4        99.3%
wo      B=8   173.4        74.4%
```

Against 64.2 GB/s (27.6%) for the narrow kernel it replaced. The verify GEMVs
are at peak; what is left in verify's 80ms is the non-weight tail (transposes,
DeltaNet state, overlay correction, norms, quantize, copies) which the profile
puts at ~11% of GPU time combined.

So the remaining GPU-side headroom is roughly 36.9 -> 45-51 tok/s. Reaching
55-60 needs a smaller drafter, and that is a training/checkpoint decision rather
than a kernel one.


## CORRECTION: the drafter size is NOT the blocker — tree verify is

The section above concluded that 55 tok/s needs a smaller drafter. That is
**wrong**, and the fact that kills it is simple: the closed-source engine hits
60 tok/s on this GPU **with this exact 1.18 GiB drafter**.

The byte arithmetic was right; the reading of it was not. At 233 GB/s, 60 tok/s
means ~3.62 GiB per token. Our cycle streams 23.8 GiB (drafter B times + target
once). 23.8 / 3.62 = **tau ~6.6** — and a *chain* drafter at B=8 gives 5.333.
So the missing factor is not fewer bytes per cycle, it is more accepted tokens
per cycle. That is exactly what tree-structured verification buys: the tree
keeps depth B (same drafter cost) and adds top-k branching, so one target sweep
scores many candidate continuations instead of one chain.

### We were never running it

hipfire has a full DDTree implementation (`hipfire-runtime/src/ddtree.rs`,
Ringel & Ro Algorithm 1/2, with `spec_step_ddtree`, `spec_step_ddtree_batched`
and `spec_step_ddtree_path_c`). Every number in this document was measured on
the **chain** path, `spec_step_dflash`. Tree mode was not evaluated and rejected
on this family — it could not run at all. Two independent blockers:

**1. No Opus arm in the DDTree lm_head ladder — FIXED here.**
`run_dflash_draft_for_logits` and `run_dflash_draft_for_topk_gpu` each carry
their own `w_out.gpu_dtype` match for the batched draft lm_head, and it stopped
at the MQ/HFQ families:

```
ddtree: unsupported target.output dtype (need Q8/HFQ4G256/MQ4G256/MQ3G256/HFQ6G256/MQ6G256)
```

Our target's lm_head is `OqCompactG256` (qt=36), so DDTree died on the first
cycle. `dflash_enqueue_verify_lm_head` has had the Oq8 and compact arms all
along; the draft ladders simply never gained them. Added
`dflash_gemm_opus_lmhead`, whose math is the verify arm's term for term.

**2. Compact has no GDN-tape writer — this is the real remaining blocker.**
Both tree paths depend on the tape: the DFS `spec_step_ddtree` calls
`gdn_tape.replay_gdn` / `gather_accepted`, and `spec_step_ddtree_batched` passes
`Some(gdn_tape)` into the tree forward as its "key optimization". But
`prefill_batch_pbs_eligible` sets `allow_compact = !tape_in_play`, so a compact
model is refused the batched forward the moment a tape is requested — and the
caller then asserts `tree_verify.is_none()`:

```
tree-verify mode requires the batched-FA-eligible prefill path;
kv quant + FA weight dtypes do not match on this model
```

The assertion message misattributes it: KV tier and FA weight dtypes are both
fine here (`kv_f32=false kv_asym2_tree=false`, asym3 carries the masked batched
variants). The declining term is `gdn_tape.is_some()`. Confirmed by dropping
tree-verify out of `tape_in_play` behind a flag — the verdict stayed false at
n=17 while n=9 passed.

Running the DFS path anyway shows the failure mode exactly as documented for a
zero tape — `replay_gdn_tape=2`, target emits `<think>\n<|im_end|>` and stops:

```
chain      48 tok  cycles 13  accepted 37  tau 2.846
dfs b16k4   3 tok  cycles  2  accepted  0  tau 0.000   <- zero-tape corruption
dfs b24k8   4 tok  cycles  2  accepted  1  tau 0.500
```

So it is ONE root blocker, not two: **compact-resident Opus cannot capture a
GDN tape**, and every tree path needs one.

### What to build

A compact GDN-tape writer. That single gap blocks tree verification on every
`oq4.25++` target, and tree verification is the term the byte budget says is
missing. It is also already named as the outstanding lever in
`prefill_batch_pbs_eligible`'s own comment: "a compact tape writer would be the
other one."

Also still unused on this model, worth checking after the tape lands:
- **DSpark** (`hipfire-specdecode-dspark`) — never wired for this target.
- **DFlash2's carried candidate selector** — measured worse in chain mode
  (tau 2.421 -> 2.25), but that was a chain measurement; a tree consumes top-K
  per position, which is what the selector actually produces.


## RESULT: the compact GDN tape is worth +28-34%, and it was a rollback cost

The "no compact GDN-tape writer" story was wrong twice over. There is no missing
writer — the tape write is a dtype-independent `memcpy_dtod` out of
`x_rot_batch` / `dn_alpha_batch` / `dn_beta_batch` — and the cost it was hiding
was not in the forward at all. It was in the ROLLBACK.

Phase breakdown of the chain path, compact locked out of tape capture:

```
draft=52.9ms  verify=83.1ms  replay=79.3ms  total=218.8ms
replay_gdn_tape=0  replay_batched_prefill=0  replay_full_prefill=11
```

**Replay was 36% of the cycle**, and every rejection replayed a whole prefill
because the cheap tape replay was unreachable. Earlier notes on this path quoted
`draft 39 + verify 80` and did not count replay at all, which is why the budget
looked like it closed and did not.

With the ladder fixed and the gate open, `replay_gdn_tape=17`,
`replay_full_prefill=0`, replay ~40ms:

| prompt | tape off | tape on | delta |
|---|---|---|---|
| merge two sorted lists | 11.67 / 11.72 | 15.68 / 15.69 | **+34%** |
| B-tree index | 9.41 | 12.05 | **+28%** |
| `def quicksort(arr):` | 8.87 | 11.46 | **+29%** |

Repeats are tight (11.67/11.72, 15.68/15.69). Default ON, opt out with
`HIPFIRE_COMPACT_GDN_TAPE=0`.

### Tree verify LOSES here, and the reason is a kernel width

Now that DDTree runs, it can be measured, and on this target it does not pay:

| config | tau | decode tok/s |
|---|---|---|
| chain (tape on) | 3.571 | **15.69** |
| ddtree budget 8, k2 | 3.571 | 13.70 |
| ddtree budget 12, k4 | 3.800 | 11.10 |
| ddtree budget 16, k4 | 3.800 | 6.95 |
| ddtree budget 24, k4 | 3.800 | 5.63 |

tau moves only 3.571 -> 3.800 (+6%), nowhere near the ~6.6 the byte budget wants,
and per-cycle cost swamps it. Note the cliff between budget 12 (n=13) and budget
16 (n=17): that is `gemv_oq_compact_multicol` running out of columns. It is
instantiated `_w1.._w16`, so a tree wider than 16 nodes falls off the
multi-column GEMV — which reads each weight row once for all columns — onto a
path that does not. **Compact tree verify is capped at 16 nodes by kernel width.**

So tree verify is not dead, but it needs two things it does not have: a
multicol GEMV wider than 16 columns, and a drafter whose per-position top-K is
worth branching on (+6% tau at k4 says this drafter's second choice is rarely
right). The selector this drafter carries is the obvious thing to try there,
since a tree consumes exactly the per-position top-K it produces.

### On the 38.97 figure

It does not reproduce. Same harness, same model and drafter, best measured here
is 15.69 tok/s. The gap is mostly tau and prompt: 38.97 was recorded at tau
5.333 and B=6, while these prompts give tau 2.3-3.6 at B=8. Treat 38.97 as a
best-prompt number, not a baseline — and note it omitted the 79ms replay term
that dominated the cycle.


## Where the cycle actually goes (kernel trace, all fixes in)

Phase timers are NOT usable for attribution on this path — they insert syncs, and
they report cycle TOTAL rising 218 -> 306 ms across a change that raised
end-to-end throughput 11.67 -> 15.68 tok/s. Use `rocprofv3 --kernel-trace`.

Trace of a 64-token run on a high-tau prompt, tape + drafter multicol on:

```
gemv_oq_compact_grouped_v3        9934 calls   51.73%   SINGLE-column GEMV
gemv_oq_compact_multicol_w8       4235 calls   20.79%   batched verify
gemm_oq_compact_iu4x2_w64          496 calls    8.55%
gated_delta_net_f32               1200 calls    3.47%
attention_flash_q8_0_tile         1408 calls    2.61%
gemm_dflash_oq4_plain_multicol_w8  451 calls    1.84%   drafter (was ~14%)
```

**The single-column GEMV is 52% of GPU time.** A full model sweep is ~500 GEMV
calls (64 layers x 7-8 projections), and this run is ~11 cycles, so 9934 calls is
roughly **2 extra single-token full-model sweeps per cycle** — each re-reading
all 14.4 GiB — on top of the batched verify that costs one. That is the largest
remaining term, worth more than verify itself.

Finding their source is the next lever, and it is worth more than anything else
on the list: if a cycle currently moves ~3 target sweeps where it needs 1, the
available win is close to 2x, which is the difference between 25.65 and the
target.

The drafter fix worked as intended and is now down at 1.84%, so the draft side
is done for the moment.

## Honest position against 55 tok/s

Best measured, all fixes in, across a prompt mix:

| prompt | tau | decode tok/s |
|---|---|---|
| numbers 1..30 as a list | 5.733 | **25.65** |
| first 20 primes | 5.400 | 25.55 |
| count 1..40 | 5.125 | 23.88 |
| JSON months | 4.500 | 21.09 |
| merge two sorted lists | 3.571 | 16.98 |
| MIT license header | 2.000 | 11.28 |

Cycle time is ~220 ms almost independently of prompt, so tau is what moves
throughput. 55 tok/s at tau 5.733 needs a 104 ms cycle; the bandwidth floor is
~67 ms (verify 14.4 GiB + one drafter sweep at 233 GB/s), so 55 is NOT ruled out
by bandwidth — it is ruled out by the ~2 redundant target sweeps above plus
per-cycle overhead. On a hard prompt (tau 3.571) the ceiling is ~53 tok/s, so 55
is a best-prompt target, not an every-prompt one.
