# M4 — unfusing MoE decode costs ~0.5% at batch 1 on gfx1103

Status: measured 2026-08-26 on nix1 (`gfx1103`, Phoenix APU, RDNA3, UMA).
Artifact: `~/.hipfire/models/Qwen3.6-35B-A3B--oq4.hfq` (19.1 GB, local NVMe),
resident experts, `k_top = 8`, 256 experts, `Oq4G256` routed.

## The question

§M4 has two options for arches whose MoE decode is a single fused kernel
(deepseek4, and per the M2a4b scoping, qwen35 too):

- **A** — emit a coarse `Escape` op carrying a capability predicate that names
  the reason, and keep calling the hand-written fused arm.
- **B** — unfuse MoE decode into per-expert ops so `MoeExpert(e)` is real.

The objection to B was performance: the fusion exists because it is fast, and
per-expert dispatch round-trips the gather and combine through memory. **This
measures that objection.**

## B is mostly already built

`moe_decode.rs` already contains a per-expert `weight_gemv` loop — it is the
correctness fallback for layers that are not all-MQ4 or have `k != 8`. So B did
not need a new kernel to measure; it needed a way to drive the *same* artifact
down both arms. `HIPFIRE_QWEN35_MOE_FORCE_PER_EXPERT=1` does that by clearing
`use_gpu_topk` (and `use_kernarg_fused` with it).

**The arm change is probed, not assumed.** `HIPFIRE_QWEN35_MOE_DTYPE_DEBUG=1`
reports 760 layer-dispatches at `use_gpu_topk=true` with the flag off and 760 at
`false` with it on. Greedy 4-token output is identical across the two.

## Result

`hipfire bench --pp-tokens 512 --tg-tokens 128 -r 5`, decode t/s per sample:

| run | flag | per-sample tg t/s | avg |
|---|---|---|---|
| `fused3`  | off | 11.6 11.5 11.5 11.5 11.5 | **11.52** |
| `perexp3` | on  | 11.5 11.5 11.5 11.5 11.3 | **11.46** |

**Delta: −0.06 t/s, −0.52%.** Unfusing MoE decode is approximately free at
batch-1 decode on this arch and artifact.

And it ties while carrying a handicap: the forced arm also performs a **CPU
top-K D2H sync** per token, which the fused indexed path does not. So B's cost
is `per-expert GEMV + round trip` and it still matches. A per-expert loop fed by
the existing device-side top-K should be no worse, plausibly better.

**Why it is free:** batch-1 decode is bandwidth-bound on expert weights. The
fused kernel's advantage is launch amortization, which does not matter when the
GPU is waiting on weight traffic. This is an explanation, not just an
observation — and it predicts where the result should NOT hold.

## Scope — where this does not generalize

- **Batch 1 only.** The fused advantage should grow with batch size as launch
  cost stops being hidden by memory traffic. Untested above 1.
- **gfx1103 only.** gfx1151's WMMA grouped path is a different kernel family.
- **One dtype/shape**: `Oq4G256` routed, resident (not paged), `k=8`, 256
  experts.
- Prefill is untouched — this is the decode arm.

## The measurement hazard that nearly produced the opposite answer

The first A/B looked decisive and was wrong:

```
fused (r=3):      20.8 20.8 20.8      -> 20.8 avg
per-expert (r=3): 20.3 11.5 11.5      -> 14.4 avg
```

Read naively that is "unfusing costs 31%". It is not. Running the **flag off**
with 5 reps gives:

```
fused (r=5):      20.8 20.8 11.6 11.5 11.4
```

The fused path decays from 20.8 to 11.4 *within a single run with no flag set*.
The 14.4 was fused's boosted early samples averaged against a mixed run.

**Cause: the memory clock, not heat.** After sustained load `rocm-smi` reports
`mclk level 0 (1000 MHz)` at only **46 °C** — the DPM governor has dropped
memory clock to its lowest state, and it is not thermal. Batch-1 decode is
bandwidth-bound, so tok/s tracks mclk: `20.8 / 11.5 = 1.81`, which is the clock
ratio.

**Consequences for anyone benchmarking on this box:**

1. A short run on a fresh GPU reports ~1.8× the sustained number. Neither is
   wrong; they are different clock states. Say which one you measured.
2. Never compare a first run against a later run. Any A/B must be A/B/A, or run
   long enough that both arms sit in the same DPM state. The numbers above are
   valid *because* both runs sat at level 0.
3. This is much larger than the ~8.6% "first-run position effect" recorded
   previously for gfx1103 — that is a separate, smaller effect and this does not
   supersede it.

## What this means for M4

The performance objection to B does not hold at batch-1 decode on gfx1103, so
**B is viable here and the coarse `Escape` is not forced by throughput** in this
regime. That inverts the earlier recommendation of A-on-cost-grounds.

It does **not** settle M4 on its own. Still open:

- the same measurement at batch > 1, where the fused advantage should appear;
- gfx1151, where the grouped WMMA path differs;
- deepseek4, whose fused MoE decode is a different kernel and has no existing
  per-expert fallback to borrow — B there is real kernel work, not a flag.

A defensible middle: take B where a per-expert arm already exists and measures
free (qwen35 decode), and keep the coarse `Escape` for deepseek4 until someone
writes and measures its per-expert path. That gives `MoeExpert(e)` real meaning
on one arch instead of a hole on both.

## Reproducing

```sh
M=~/.hipfire/models/Qwen3.6-35B-A3B--oq4.hfq
# arm probe (expect 760 true, then 760 false)
HIPFIRE_QWEN35_MOE_DTYPE_DEBUG=1 hipfire chat -m "$M" --max-tokens 4 "hi" 2>&1 \
  | grep -oE 'use_gpu_topk=(true|false)' | sort | uniq -c
HIPFIRE_QWEN35_MOE_FORCE_PER_EXPERT=1 HIPFIRE_QWEN35_MOE_DTYPE_DEBUG=1 \
  hipfire chat -m "$M" --max-tokens 4 "hi" 2>&1 \
  | grep -oE 'use_gpu_topk=(true|false)' | sort | uniq -c

# A/B/A, 5 reps each, killing the daemon between runs by recorded pid
hipfire bench "$M" --json -r 5
HIPFIRE_QWEN35_MOE_FORCE_PER_EXPERT=1 hipfire bench "$M" --json -r 5
hipfire bench "$M" --json -r 5
```

Do not wrap `hipfire bench` in `hipfire lock` — it drives the daemon, which
takes the lock itself. Clear a lingering daemon by the pid you recorded; never
`pkill -f`, which matches the calling shell's own command line.
