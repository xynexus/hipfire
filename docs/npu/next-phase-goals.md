# Next phase: from reference kernel to production NPU decode

Written 2026-08-02, closing the FLM reverse-engineering phase.

The reference kernel is done: it reproduces FLM's tokens exactly and beats its
throughput. `docs/npu/flm-refe-log.md` is the full record. This document is what
carries forward and what to do next.

## The one number that decides the next phase

**Decode on this NPU is DMA-bound. Bits per weight IS speed.**

With every one of the 32 cores' compute stubbed out, the fused design still runs
731.5 of its 791.5 us/layer — 92.4% — at 52.29 GB/s against lm_head's measured
54.7 GB/s ceiling. All the arithmetic in the model is worth 60.0 +- 3.5 us/layer.

The two-pass lm_head measured the mechanism directly, which is why this is not an
inference from a roofline sketch:

    coarse tier  4.02 bits/weight   131.8 MB   54.7 GB/s
    exact tier   5.00 bits/weight   164.7 MB   55.2 GB/s
    bytes ratio 0.8006 against time ratio 0.8033 — matching to 0.3%

Same GB/s both ways. The kernel is not the lever and never was.

### What that implies, in tok/s

Llama-3.2-1B decode moves 60.8M params/layer x 16 layers plus a 262M-param
lm_head every token. At the measured 54.7 GB/s ceiling:

    format        bits   bytes/token   tok/s       note
    q4nx (FLM)    5.00     775.7 MB     63.4        MEASURED, our fused design
    oq4/oq4++     4.0625   630   MB    ~78          +27%; the GEMV is MEASURED at
                                                    55.5 GB/s, the token is projected
    oq3/oq3++     3.0625   475   MB   ~104          +70%; wholly projected
    oq8++         8.0625  1244   MB    ~44          -28% vs FLM; ruled out

Bits are the real block sizes from `hipfire-quant-format`, not estimates: 130 B,
98 B and 258 B per 256-group respectively. The `++` suffixes share a container
with their plain forms — calibration changes which codes are stored, never how
many bytes — so they are one row each.

**This CONFIRMS the standing `oq4++`/W4A8 target rather than revising it** — the
project's quant target was set to `oq4++` on 2026-07-30, and the measurement above
is independent evidence for that choice. The row is kept because ruling `oq8`
out is worth having in writing: it moves 59% more bytes than the format FLM
already ships and lands near 44 tok/s. W8A8's activation half is free here —
activations are negligible against weights — but the weight half is the entire
cost. On a compute-rich, bandwidth-poor NPU the quant format is the performance
story, which is the opposite of where GPU intuition points.

**So the next phase's headline lever is a TIGHTER format, not a faster kernel.**
`oq4++` (symmetric Opus Quant, clip-search/AWQ, Hessian error feedback) is both
faster AND more accurate than the q4_1-shaped container FLM uses. That is the
rare case where the two goals do not trade against each other.

**`oq3` is the bigger lever and is scoped nowhere.** `Oq3G256` is W3A4,
`[f16 scale][8 x 3 u32 bit-planes]` = 98 B per 256-group = 3.0625 b/w, and
hipfire's own format docs call it "the memory-ceiling lever (25% less weight
traffic than Oq4)". On a decode path that is 92.4% DMA that is ~104 tok/s against
oq4's ~78. Its bit-plane storage IS the kernel layout, so unlike oq4 it needs no
repack. Codes are `[-3, 3]`, 8 blocks of 3 u32 planes per 256-group.

**AIE2P has the intrinsics for the unpack; do not assume it is expensive.**
`aie_api` provides `aie::mask<32>::from_uint32` (a bit-plane IS a 32-lane mask),
`aie::select`, `aie::bit_and/or/xor`, `aie::interleave_zip/unzip`, and
`aie::unpack` / `unpack_sign` — the last being the same `vldb.unpack` int4->int8
widening the oq4 kernel already gets for free. So the promising shape is NOT
mask-select per value: spread the 3 planes into packed int4 NIBBLES with shifts
and bit ops, then hand them to the existing unpack path, which already consumes
exactly that and costs nothing.

The real question is ops-per-byte against the DMA budget, not whether an unpack
exists. oq3 moves 75% of oq4's bytes, so it gets 75% of the time per 256 weights
and must fit its extra unpack inside that. That is measurable, and the method is
already in the tree: build it, and put a control tile beside it that streams the
identical bytes with trivial arithmetic. This file's oq4 numbers went
16.3 -> 31.4 -> 35.5 -> 55.5 GB/s on formulation alone with the bytes never
changing, so measure the formulation you actually wrote rather than the format.

Sanity-check the projections before building on them: the oq4 GEMV is now measured,
but the TOKEN figures still assume the same dispatch structure across sixteen
layers, and no layer has been built in either format.

**MEASURED 2026-08-02 — the assumption holds, and the format is tighter than assumed.**
`kernels/npu/flm_gemv_oq4g256.cc` + `lmhead_twostage.py --vs-oq4`, three designs
interleaved in one process at lm_head size:

    coarse  2384.6 us  131.8 MB  55.3 GB/s   16448 B/tile
    oq4     2403.8 us  133.4 MB  55.5 GB/s   16640 B/tile
    oq4-1s  2406.1 us  133.4 MB  55.4 GB/s   SAME TILE, coarse arithmetic (control)

oq4 streams at the ceiling: +0.4% on the coarse tier, +0.1% on the control. Device
argmax 16309 matches an independent host reference, so this is a checked kernel and
not just a timed one.

`Oq4G256` is **4.0625 b/w**, not the ~4.25 the table above assumed — 130 B per
256-group. So the oq4 row is 630 MB/token and **~78 tok/s, +27% vs FLM**, better
than the +18% projected.

**Getting there took four kernels, and the first one looked like a verdict:**

    one reduce_add per GROUP (128 a tile)                    16.3 GB/s
    group accum folded into a row accum by broadcast FMA     31.4
    ... with the 4-iteration inner loop fully unrolled       35.5
    scale folded into the WEIGHTS, ONE row accumulator       55.5

3.4x slower for 1.2% more bytes is a convincing-looking number and it was entirely
my arithmetic. The control tile — identical bytes and loads, coarse arithmetic — is
what separated "this shape cannot stream" from "this code is slow" in one
measurement. Keep that control if the layout is ever changed again.

**The 130 B on-disk block is a STORAGE form, never a kernel form.** Its nibbles
start at byte 130g+2, a stride that is not a multiple of 4 — vector-loading it is
the silent misaligned-load failure this tree has already paid for. That is what
`Oq4G256ArchPacked` and the loader's per-arch repack exist for; the NPU repack is
planar, `[NROWS*NG bf16 scales][NROWS*K/2 nibbles]`, same byte count.

**WHAT WAS ACTUALLY MEASURED: the oq4 CONTAINER, not the oq4 CODEC.** The probe packs
weights with a naive absmax quantizer (`s = max|w|/7`). `quantize_oq4g256`
(`hipfire-quantize/src/codecs.rs:701`) additionally applies

    cpu_fwht_256(&mut group, signs1, signs2);       // randomized Hadamard rotation
    let scale = symmetric_clipsearch(&group, 7.0);  // clip-search = the first '+'

so the probe is weaker than `oq4+` and well short of `oq4++`. That is deliberate and it
does not weaken the throughput result: FWHT, clip-search and LDLQ change WHICH codes and
scales are stored, never HOW MANY BYTES, and the block is 130 B per 256-group whatever
chose the numbers. **The 55.5 GB/s and the ~78 tok/s are the real format's.** What the
probe says nothing about is the format's ACCURACY, and no accuracy claim rests on it.

**The port still owes the FWHT.** Stored codes decode as `scale * sext4` under an inverse
FWHT. The rotation is orthogonal, so `x . W^T = (Rx) . (RW)^T` — the GEMV body is
unchanged and what is missing is a pre-pass that rotates the ACTIVATION in 256-element
groups. It is per-GEMV, not per-row: a 256-point transform over 8 groups is ~16K ops
amortised across 128256 rows. Free in time, absent in code.

**Two things this did NOT settle**, both real:

  - **Scale precision.** The shipped kernel folds the scale into the weights, which
    rounds `scale * code` to bf16 BEFORE the MAC and moves relative error
    1.651e-03 -> 3.666e-03 against the accumulator-scaling variant. Irrelevant for
    a coarse shortlist the host rescores; a live question for SIXTEEN CHAINED
    LAYERS, where it compounds and has not been measured. The accurate variant
    costs 36%.
  - **f16 vs bf16 scales.** On-disk is f16; AIE2P's native 2-byte float is bf16,
    so the repack converts. bf16 gives ~0.4% scale accuracy against f16's ~0.05%.
    f32 scales would fix it at 4.125 b/w instead of 4.0625 — 1.5% of bandwidth for
    8x the scale precision. Packing f16 and reading bf16 passes every size and
    alignment check and produces garbage (rel 1.0); only a value check finds it.

## Context depth changes which lever matters (measured 2026-08-02)

The `flm-benchmarks.md` sweep stops at 3135 tokens — **2.4% of a 131072 window** — so
its "~13% decay" describes the shallow end of a much steeper curve. Swept properly
against the running server:

    ctx tokens   decode tok/s   prefill tok/s   TTFT
         1010        58.36          1258         0.8 s
         2035        55.74          1842         1.1 s
         4110        50.22          1592         2.6 s
         8210        43.37          1598         5.1 s
        16435        33.89          1331        12.4 s
        32885        23.21           942        34.9 s
        65785        14.51           587       112.0 s
        98685        10.56           426       231.8 s

**FLM loses 82% of its decode rate between 1K and 96K.** The advertised window works
to the top — 122880 was accepted; 131072 refuses with "Max length reached" only
because prompt plus generation must fit inside it, which is the window being full
rather than a lower runtime cap.

This is the same bandwidth story, one level up. KV traffic per token is
16 layers x 8 KV heads x 64 dim x 2 (K,V) x 2 B = **32 KB per position**, so at
context L a token moves 775.7 MB of weights PLUS L x 32 KB of cache:

    ctx      KV bytes   +weights   bandwidth model   measured   ratio
      1010      33 MB     809 MB      67.6 tok/s       58.4     0.86
      8210     269 MB    1045 MB      52.4             43.4     0.83
     32885    1078 MB    1854 MB      29.5             23.2     0.79
     98685    3234 MB    4010 MB      13.6             10.6     0.78

The ratio is roughly CONSTANT at ~0.8. FLM is bandwidth-bound at ~80% efficiency
across the whole range, and the decay is inherent to KV growth — not an attention
kernel with headroom left in it. Do not go looking for that headroom; it is not there.

**So the lever inverts with depth.** At 98K, KV is 3234 MB against 776 MB of weights
— the cache is 80% of the traffic and the weight format is nearly irrelevant. The
`oq4++` port wins ~18% at short context and progressively less as context grows;
past ~16K the dominant lever is **KV cache quantization**, which is a different piece
of work and is not currently scoped anywhere.

**And our design cannot reach any of this.** The KV tile is a fixed 40 columns. We do
not decay, because we cannot go deep enough to decay. Any claim of beating FLM
"across the context range" requires the cache to stop being a tile and become a
streamed structure — a much larger change than a quant-format port. Until then the
honest claim is bounded: 63.4 vs 61.18 at short context.

## Sizing the two levers — they barely overlap

Our fused design runs 777 MB/token at 63.4 tok/s = **49.3 GB/s effective**, against FLM's
781 MB at 60.1 = 46.9 GB/s. We are 5.0% more bandwidth-efficient on identical weights.
Holding that efficiency and varying only the FORMATS (projection, not measurement):

        ctx     FLM |  us q4nx  us oq4++  oq4+kv8  oq4+kv4
       1010   58.36 |     61.0      71.3     73.0     73.8
       8210   43.37 |     47.7      53.8     62.5     68.1
      32885   23.21 |     27.3      29.2     42.0     53.8
      65785   14.51 |     17.4      18.1     29.2     42.0
      98685   10.56 |     12.8      13.2     22.4     34.4

**The two levers are nearly disjoint.** At 1K, `oq4++` IS the win (+22% over FLM) and KV
quantization adds 3%. At 98K, `oq4++` adds almost nothing over the format we already have
(13.2 against 12.8) while KV quantization is worth 2.6x — because KV is 82% of traffic at
bf16 and 54% even at 4-bit.

Which to build first is therefore a question about which context regime matters, not about
which is the better idea:

  - **short-context serving** -> `oq4++`, ~+22%, well-understood path, no cache redesign
  - **long-context serving**  -> streamed KV + KV quantization, ~2-3x, but it needs the
    cache to stop being a fixed 40-column tile first, which is the larger engineering item

Both are real; neither subsumes the other. Doing `oq4++` first is defensible because it is
bounded and reuses everything here — but note it wins in exactly the region where FLM is
strongest, and leaves untouched the region where FLM falls to 10 tok/s.

Projections assume 49.3 GB/s holds at other formats and depths, which is exactly the
assumption the first task below is meant to test. KV quantization also costs accuracy in a
way weight quantization at 4 bits largely does not — 4-bit KV is aggressive and unvalidated
here.

## What is established and should not be re-derived

Hard limits, all measured on this hardware:

    dispatch floor        92.9 us       7-point fit, R^2 0.99997
    in-dispatch barrier    6.00 us      fit R^2 0.99904; 80/token = 3.8%
    shim budget           16 in / 16 out    8 shim tiles x 2 each way, HARD
    memtile budget        48 in / 48 out    8 memtiles, ~6 each way
    program memory        16 KB/core
    data memory           64 KB/core
    effective bandwidth   ~54.7 GB/s
    same-build spread     2.6% run to run

Structural findings:

  - FLM issues ~2.5 dispatches per TOKEN. A design issuing one per layer pair
    loses to the dispatch floor alone. Fusion into ONE dispatch is what made the
    reference kernel competitive; interleaved as separate dispatches the same
    kernels run 55.0 tok/s, a 10% loss.
  - Latency-bound slopes DO NOT transfer. `groups_ab` standing alone measured
    123.6 us/layer at 31.9 GB/s; fused, its bytes queue at the DDR ceiling and
    cost 75.2. Bytes add. Slopes measured on a latency-bound design do not.
  - Two-pass output projection works and is worth ~20% of the head: a coarse
    low-bit shortlist on the NPU, an exact rescore of K=32 rows on the host.
    Device recall verified 48/48, worst rank 3.

## Traps that cost real time here

Each of these produced a confident wrong result that survived at least one check.

  - **Position 0 is structurally blind to attention.** Its softmax is over a
    single entry, so no score reaches the output. A misaligned `aie::load_v` left
    pos 0 BIT-EXACT while every later position collapsed to cosine 0.05-0.28.
    The attention scale error hid the same way. Nothing about attention is
    verified until it is verified at pos > 0.
  - **`aie::load_v` on a misaligned pointer does not fault** — it reads the wrong
    bytes, silently.
  - **`iron.jit` caches on the AST.** Comments and compile FLAGS do not bust it.
    Stamp cache-busting values into fifo NAMES. A design whose generator cannot
    even run will be served from cache indefinitely (this happened: `groups_ab`
    at 16 cores has never been able to generate its own design).
  - **A held dispatch that rebinds nothing silently ignores new inputs.** Binding
    once and dispatching many times is only correct if each dispatch reads the
    buffer's CURRENT contents — perturb, dispatch, restore, dispatch, and demand
    the output moved and came back bit for bit. This shipped broken and read as
    "the coarse tier has no recall".
  - **Benchmark by INTERLEAVING, never sequentially.** A sequential A/B called
    chunk=16 a 40% win twice; interleaved, it was a loss. Whichever ran first
    kept the cache.
  - **Verify against something that shares no code.** Five correctness faults all
    passed every check because each check compared a stage against a reference
    computed from the same wrong input. The fp32 forward from
    `consolidated.00.pth` is what eventually caught them.
  - **Stale constants in scripts that re-measure everything else.** A fresh
    lm_head number composed with a layers constant from a different build config
    overstated throughput by 1.1 tok/s. If a constant tracks a build knob, say so
    where it is defined.

## Goals for the next phase

1. **Port the decode path to `oq4++`.** The headline. Target ~72 tok/s on
   Llama-3.2-1B, +18% over the current reference kernel and +17% over FLM.
   Success is measured on the WALL CLOCK, not device time.

   **Which FLM number you are beating matters.** Two exist and they are not
   interchangeable: 61.18 tok/s is FLM's own server figure at short context and
   is what every comparison in `flm-refe-log.md` uses; `flm-benchmarks.md`
   records 60.1 tok/s at a 159-token prompt, DECAYING to 52.6 across the context
   range — a ~13% decay with depth. Our design does not decay that way because
   its cache is a fixed 40 columns, but it also cannot reach the depths where
   FLM slows down. So 63.4 vs 61.18 is like-for-like at short context and is the
   honest claim; any comparison at depth is unsupported in BOTH directions until
   the cache stops being a fixed tile.
2. **Keep the correctness apparatus.** Token-for-token agreement with FLM on its
   own `context` array is the acceptance test; the fp32 oracle is the reference.
   Both exist and work — `decode.py`, `decode_oracle.py`, `sweep.sh`.
3. **Then MoE: `qwen3.6-moe:35b-a3b`.** This is a genuinely different problem,
   not a port. Expert routing, per-expert scales, and a working set that does not
   fit the assumptions above. Per project memory, no MoE model has taken the
   batched-prefill path and it needs per-expert AWQ scales.

## Open questions

  - Does an `oq4++` tile shape hit the same 54.7 GB/s? The projections assume it
    does. Measure a single GEMV at the new format before porting 16 layers.
  - Prefill is UNEXAMINED. Everything here is decode. Prefill is compute-bound
    and batched — a different regime where the format argument above does not
    apply, and where FLM may still have something to teach.
  - The 16.2 us/layer surcharge over projection is attributed to interaction
    (12.2 +- 4.3 us/layer at 2.9 sigma) but not eliminated.
  - Whether the two-pass head's coarse tier should be `oq4++` too, or stay a
    purpose-built 4-bit direction encoding.

## Kickoff prompt for the next session

```
Read docs/npu/next-phase-goals.md first — it closes the FLM reverse-engineering
phase and states the findings that drive this one.

CONTEXT. tools/npu/flm/ reproduces FLM's tokens exactly (4/4 on FLM's own
/api/generate context array) and runs 63.4 tok/s against FLM's 61.18 at short
context. Full record in docs/npu/flm-refe-log.md — long, and its CURRENT STATE
section supersedes everything above it.

THE SHAPE OF THE PROBLEM. NPU decode here is bandwidth-bound end to end: 92.4% of
layer time survives with all 32 cores' compute stubbed. Bits per weight IS speed,
measured not inferred (coarse 4.02 vs exact 5.00 bits gave a bytes ratio 0.8006
against a time ratio 0.8033). Two levers follow, and they are nearly disjoint:

  oq4++ weights        +22% at 1K context, ~nothing past 32K
  streamed+quantized KV  ~nothing at 1K, 2.6x at 98K

Pick per the scoping decision in the goal doc. If unspecified, default to oq4++.

FIRST TASK EITHER WAY, before touching sixteen layers: build ONE GEMV in the target
format at lm_head size and measure whether it reaches the same ~54.7 GB/s. Every
projection in the goal doc assumes it does, and that assumption is untested. If it
does not, say so and stop — the plan rests on it.

CONSTRAINTS THAT ARE NOT NEGOTIABLE:
- Verify at pos > 0. Position 0's softmax is over one entry, so it is structurally
  blind to attention; it stayed bit-exact through a bug that collapsed every later
  position to cosine 0.05.
- Benchmark by INTERLEAVING, never sequential A-then-B. Same-build spread is 2.6%
  and a sequential A/B has twice called a loss a 40% win here.
- Verify against something that shares no code — the fp32 oracle from
  consolidated.00.pth. Five correctness faults once passed every check because
  each check's reference came from the same wrong input.
- iron.jit caches on the AST. Comments and compile flags do NOT bust it; stamp
  values into fifo names.
- ./tools/npu/flm/sweep.sh is the regression gate. Run it after touching anything
  shared: kernels/npu/flm_kv_pair.h, flm_q4_1_tile.h, pyxrt_design.py.
- A user-owned `flm serve llama3.2:1b` runs on port 52625 (NOT 11434) and is the
  token oracle. Query it, never restart it. Requesting a different model would
  evict llama from it — do not, without asking.
- Quote the right baseline: 61.18 tok/s is FLM at short context. Its decode falls
  to 10.56 at 98K, so a deep-context comparison is a different claim. Read the
  baseline caveat before quoting any tok/s number.

Work on a new branch off feat/npu-flm-reverse-engineering.
```
