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
    oq3/oq3++     3.0625   475   MB    see below   DMA win MEASURED and real;
                                                    the UNPACK is 2x over budget
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

**REVISED 2026-08-02 — the lever is CALIBRATION, not the container.** The
sentence above is right about throughput and wrong about what makes the quality.
Measured end to end against q4nx on the same forward:

    q4nx (FLM)                  5.0000 b/w   KLD 0.03066   THE BAR
    my q4_1 at group 32         5.0000 b/w   KLD 0.13898   4.53x worse
    my oq4 sym g256             4.0625 b/w   KLD 0.23732   7.74x
    + FWHT rotation             4.0625 b/w   KLD 0.34528  11.26x
    + clip-search               4.0625 b/w   KLD 0.34296  11.18x

The second row is q4_1 AT GROUP 32 — q4nx's own format, its own bit rate, 4-bit
codes with a scale and a min per 32 weights. Same container, same bits, **4.53x
worse**, while reconstructing only 5% worse (0.0814 vs 0.0775 relRMS). Confirmed
by inspection of FLM's container: its stored min differs from the naive block
minimum by nearly a full standard deviation of the weights, so q4nx is a
CALIBRATED q4_1 and naive min/max over identical blocks is not close.

A 5% reconstruction difference is worth 4.5x in KLD. The error MAGNITUDE is
nearly identical; the error STRUCTURE is not, and only the structure reaches the
output.

**So picking a tighter container is not the work.** oq4's 4.0625 b/w is measured
at the DMA ceiling and is worth +27% — that half is done and idle. The open
problem is matching or beating FLM's CALIBRATION at that container, and the
throughput does not arrive until the quality does.

**RECONSTRUCTION ERROR IS NOT A USABLE OBJECTIVE HERE.** It failed to predict
output quality six times in one session, in both directions: clip-search 13%
better RMSE and 43% worse KLD; FWHT 2.1-2.5x better ACTIVATION RMSE and 4%
better KLD; asymmetry 15% better RMSE and worse KLD; LDLQ deliberately worse
RMSE by design; and 5% better RMSE for 4.5x better KLD above. It is the metric a
quantiser is naturally tuned against — hipfire's clip-search optimises exactly it
— and on this model it is not a proxy for anything. Any calibration work needs
the KLD harness in the loop, not an RMSE target.

**3-BIT MEASURED 2026-08-02: oq3++ IS 4.29x WORSE THAN oq4++ AND DOES NOT REACH
THE BAR.** Four artifacts, one 503-position wikitext2 slice, one fp16 reference:

                        b/w      PPL      KLD      vs fp16   vs oq4++
    fp16 reference     16.0000  16.4027  0.000000     —          —
    oq4++ (bf16 src)    4.0625  17.2238  0.045964   +5.01%     1.00x
    oq4++ (f16 src)     4.0625  17.2091  0.046167   +4.92%     1.00x   CONTROL
    oq3++               3.0625  19.9025  0.197836  +21.34%     4.29x
    qtip3 (no cond.)    3.1250  25.2538  0.368288  +53.96%     7.98x

**q4nx costs +4.89%. oq3++ costs +21.34%.** 3-bit does not reach the bar; parity
was an oq4++ result and does not survive the drop to 3 bits. The 4.29x is worth
recording precisely because it was PREDICTED as "about 4-5x" before measurement —
the intuition about this codec family is calibrated, the absolute standard was
not.

DO NOT read the qtip3 row as "trellis loses". That artifact has NO conditioning:
`format_needs_calibration("qtip3")` is false, so it took neither AWQ nor LDLQ
while oq3++ took both. The row measures conditioned 3-bit against UNconditioned
3-bit trellis, which is the same lesson as everything else in this file —
calibration dominates the codec.

**RESOLVED SAME DAY — CONDITION THE TRELLIS AND IT WINS.** AWQ was wired into
`pack_qtip_real_tensors` (hipfire `ad547ff9d`) and re-measured on the same slice:

    oq3++  (AWQ + LDLQ)  3.0625 b/w  PPL 19.9025  KLD 0.197836     —
    qtip3  (no cond.)    3.1250 b/w  PPL 25.2538  KLD 0.368288  1.86x worse
    qtip3+ (AWQ only)    3.1250 b/w  PPL 20.0937  KLD 0.152725  1.30x BETTER

**AWQ alone cuts qtip3's KLD 2.41x, and that beats oq3++ WHICH ALSO HAS LDLQ** —
half the conditioning, better distribution match, 2% more bits. PPL disagrees
slightly (20.09 vs 19.90), so KLD and PPL are not ranking these identically;
do not quote one as if it settled the other.

**ERROR FEEDBACK MEASURED 2026-08-03 — BEST 3-BIT YET.** Three formulations were
implemented and compared at a FIXED encoder and beam (conditioning the only
variable), then the winner was paired with the strong encoder:

    arm (beam 4, CPU)   PPL      KLD        vs plain   encode cost
    plain             33.1883  0.590375       —           1x
    weighted          29.0761  0.509756    1.16x          8x
    greedy            23.1624  0.300033    1.97x         23x
    beamldlq          22.5309  0.260646    2.27x         35x

They rank in cost order. But the number that decides the design is NOT in that
table: plain qtip3+ with GPU exact Viterbi scores 0.152725 — better than the best
conditioned CPU arm. **Encoder quality dominates conditioning; you cannot pay for
feedback by narrowing the beam.** So the two must be combined, and only `greedy`
can be: it needs the CHOSEN path only, so each block delegates to the Viterbi
encoder and still feeds error forward. `beamldlq` keeps a residual per beam
candidate and cannot delegate.

    qtip3+        (AWQ, Viterbi)      3.1250 b/w  PPL 20.0937  KLD 0.152725
    qtip3+greedy  (AWQ+OBS, Viterbi)  3.1250 b/w  PPL 18.8525  KLD 0.093290
    oq3++         (AWQ+LDLQ)          3.0625 b/w  PPL 19.9025  KLD 0.197836
    oq4++         (AWQ+LDLQ)          4.0625 b/w  PPL 17.2091  KLD 0.046167

**qtip3+greedy is 2.12x better than oq3++ at the same 3-bit tier** and closes the
gap to oq4++ from 4.29x to 2.02x, at 23% fewer bits. Still short of the q4nx bar
(+14.9% PPL vs +4.89%), so oq4++ remains the only quant at parity.

Predicted 0.078 from the beam-4 ratio and got 0.093290: conditioning buys LESS on
top of a stronger encoder. Assume that direction rather than extrapolating a
weak-encoder ratio. Cost is 3584s vs ~100s (36x) — the tensor is encoded
block-by-block instead of once. Offline only.

`weighted` was expected to be a no-op (the FWHT exists to flatten the rotated
diagonal it weights by). It is the weakest at 1.16x, but not nothing.

`qtip3++` still REFUSES to run: the trellis has no in-beam Hessian weighting, so
the second `+` would promise error feedback the pack does not implement. The
shipped combination is spelled `qtip3+` with `HIPFIRE_QTIP_COND=greedy`.

**CODEBOOK MEASURED 2026-08-03, AND IT IS A REAL LEVER.** The trellis codebook is
not stored — it is RECOMPUTED from the state at decode — so changing it means
changing kernels, not data. 3INST (excess kurtosis -0.111) sits closer to the
Gaussian the rotated weights follow than 1MAD (-0.312), and existed encoder-side
only until hipfire `ce4544da9` added `QuantType::Qtip3G256I3` (51) with its own
GEMV and Viterbi kernels.

    qtip3+ 1MAD   (AWQ, Viterbi)         3.1250 b/w  PPL 20.0937  KLD 0.152725
    qtip3+ 3INST  (AWQ, Viterbi)         3.1250 b/w  PPL 19.8332  KLD 0.142660
    greedy + 1MAD (AWQ+OBS, Viterbi)     3.1250 b/w  PPL 18.8438  KLD 0.092899
    greedy + 3INST(AWQ+OBS, Viterbi)     3.1250 b/w  PPL 18.0755  KLD 0.088172
    oq4++ (for scale)                    4.0625 b/w  PPL 17.2091  KLD 0.046167

**Best 3-bit is greedy + 3INST at KLD 0.088172** — 2.24x better than oq3++'s
0.197836 where this phase started, and 1.91x from oq4++ at 23% fewer bits.

THE GAIN SHRINKS AS THE SURROUNDING MACHINERY STRENGTHENS, measured four ways:
16.4% (beam 4, no AWQ), 8.3% (Viterbi, no AWQ), 6.6% (Viterbi + AWQ), 5.1% (on
top of greedy). Same shape as the conditioning result. **Treat any ratio measured
at a weak operating point as an upper bound, not an estimate** — it over-predicts
by roughly 2x in this codec family.

A SEPARATE QUANT TYPE, NOT A FLAG, because the codebook is part of the wire
contract: nothing in a block distinguishes 1MAD from 3INST, so a cross-decoded
artifact yields noise while every length, checksum and shape check passes. That
choice paid off immediately — the first end-to-end run gave PPL 2.7e6, which is
unmistakable, where a shared code would have produced a plausible-looking
regression. Adding the type needed FIVE separate registrations (size, dispatch
tables, supports_awq_sidecar, three rotation predicates, loader arm) and none of
the omissions failed loudly: they produced 2.7e6, 1.07e38, and 8.27.

QUANTIZE COST, once the encoder stopped being sabotaged: 524s for the model
(hipfire `9e80f3d19`). `gpu.take()` had left the outer Option None from the
second tensor on, silently demoting every later encode to the CPU beam — 3435s of
a 3482s phase. Three FLOP-count predictions about that bottleneck were wrong
before instrumenting seven stages found it in one run. Measured breakdown after
the fix: cholesky 193.8s, rotate_H 119.0s (since parallelized), encode 95.7s,
propagate 89.1s.

LQER IS A DEAD END ON THIS PATH, measured, not assumed. `HIPFIRE_LOWRANK_R`
emits 224 `lr_u`/`lr_v` tensors and llama scores BIT-IDENTICAL (25.2538 /
0.368288 at r=16 AND r=32, +55 MB for nothing) because `lr_u` has exactly one
consumer in the tree: `hipfire-arch-minimax`. The llama path never applies the
correction it paid for. Fixing that means a runtime consumer — `y += (x·U)·V`,
two rank-r GEMVs — not a quantizer change.

The CONTROL is why these numbers can be trusted: oq4++ rebuilt from the fp16
source landed at 17.2091 against 17.2238 from bf16, a 0.09% difference. All four
therefore share a parent, and that parent is the exact model the .pkld reference
was generated from.

Three hipfire gaps had to be closed to measure any of this, none of them the
measurement: `--format oq3++` did not resolve (the recipe existed, the flag parse
accepted only the bare form); hfq -> hfq requant silently dropped config and
tokenizer, producing artifacts that load far enough to look healthy then panic
with no cause; and `load_weights_hfq` had no qt 38 arm, so the quantizer could
write W3 but nothing could read it. The last is now served by a LOSSLESS int8
upcast — int3 sign-extends into int8 exactly and the f16 group scale carries
over — so 3-bit shares the iu8 W8A8 kernels with oq4/oq8 rather than needing a
W3 GEMV. That also retires the earlier "oq3 unpack is unaffordable" worry: that
number came from a kernel decoding bit-planes to bf16, not to a shared int8 grid.

**GOAL MET 2026-08-03: oq4.25++ BEATS q4nx ON BOTH METRICS AT 15% FEWER BITS.**

                              b/w      PPL       KLD
    q4nx (FLM, the bar)     5.0000  17.1949  0.034954
    oq4.25++ @ alpha 0.45   4.2500  17.1547  0.034862   BETTER ON BOTH
    oq4.25++ @ alpha 0.55   4.2500  17.1767  0.036631
    oq4++    @ alpha 0.55   4.0625  17.2091  0.046167

Margins are thin — 0.23% PPL, 0.26% KLD — so this is "clears the bar", not
"clears it comfortably". They are meaningful because the harness is
DETERMINISTIC: the same artifact reproduced KLD 0.088172 and 0.152725 exactly
across repeated runs all session, so a 0.26% difference between artifacts is real.

**THE BAR ITSELF WAS FINALLY MEASURED ON OUR HARNESS** (q4nx dequantised to dense
f16 and scored through perplexity.rs against the same .pkld). Until then it was
inferred from relative PPL degradation across two different harnesses, and that
inference silently did NOT transfer to KLD: measured directly, plain oq4++ was
1.32x WORSE than q4nx on KLD while matching it on PPL. The 2026-08-02 "parity"
entry below is a PPL statement only.

**WHAT MOVED IT, and what did not:**

    alpha sweep 0.35-0.75    KLD floor 0.046167 at the 0.55 default — no gain
    per-layer no-clip        KLD 0.048006 — worse; better PPL (17.0054)
    oq4.125++ / .25 / .5     KLD 0.038748 / 0.036631 / 0.037291 — 21% at .25
    then alpha on top of .25 KLD 0.034862 at alpha 0.45 — clears the bar

Everything that adjusted the CENTRE of the weight distribution (smoothing,
clipping) traded PPL against KLD and never moved KLD's floor. KLD is a top-128
DISTRIBUTION match and the tail is what symmetric int4 at group 256 discards;
q4nx keeps it with a zero-point and 8x more scales (asymmetric q4_1 at group 32).
Sparse int8 outliers put the tail back — that is why mixed precision was the
lever and parameter tuning was not.

**The alpha optimum SHIFTED once outliers handled the tail** (0.55 -> 0.45), so
sweeping alpha first and stopping there would have missed this entirely. Tune the
structural knob, then re-tune the parameter.

Cost: 4.25 b/w vs oq4++'s 4.0625 is ~5% of the bandwidth advantage, leaving 15%
against q4nx rather than 19% — still ~15% throughput on a 92.4% DMA-bound decode.

**ANSWERED 2026-08-02: oq4++ REACHES PARITY WITH q4nx AT 19% FEWER BITS.** The
measurement this section called the most valuable open one in the phase, on a
common corpus slice — the same 512 wikitext2 tokens, same 8-position warmup, the
same 503 scored positions:

                              reference    quantised    degradation   b/w
    my harness  fp32/q4nx       16.9183      17.7460       +4.89%    5.0000
    hipfire     fp16/oq4++      16.4027      17.2238       +5.01%    4.0625

**Equivalent quality at 4.0625 vs 5.00 b/w.** The 0.12-point spread is inside
noise at 503 positions and the two references differ by 3.1% themselves (fp32 vs
fp16, different harnesses), so this reads as PARITY — not better, not worse. What
is unambiguous is the bit rate: 630 MB/token against 775.7. Against the +27%
throughput already banked at the DMA ceiling, **the port is justified end to
end.**

    hipfire's own KLD, oq4++ vs its fp16 reference:  0.045964 (top-128, 503 pos)
    sanity floor, fp16 vs its OWN reference:         0.000000  <- exact

DO NOT compare that 0.046 against q4nx's 0.031 in the table above. Different
harnesses, different references, different corpora (that 0.031 came from a
synthetic paragraph at 64 tokens against fp32) and different lengths. Relative
PPL degradation on one shared slice is the only quantity both sides can express;
the KLD pair would manufacture a "48% worse" conclusion out of two incomparable
numbers.

This also RESOLVES the section above. "The lever is calibration, not the
container" stays true — naive q4_1 at q4nx's own bit rate really is 4.53x worse —
but the conclusion drawn from it, that matching FLM's calibration was the open
problem, is now closed: hipfire's existing oq4++ calibration already matches it.
The remaining work is the port, not the quant.

Both routes named here as blocked were cleared by going through hipfire rather
than around it: `read_oq4.py`'s AWQ folding is still unsolved (5 of 7 tensor
types correct — q/k/v and gate/up share a norm but carry separate scales, so no
per-tensor division inverts it), so the artifact was scored by
`examples/perplexity.rs` instead, which needed a llama calibration collector
(hipfire `2be6f2e70`, validated at 0.9999 against the shipped package) and arch
dispatch (hipfire `125aa0993`).

**`oq3` MEASURED 2026-08-02: the DMA win is real, the unpack is not affordable yet.**
At lm_head size, three designs interleaved in one process:

    coarse   2396.8 us  131.8 MB  55.0 GB/s
    oq3      4781.1 us  100.6 MB  21.0 GB/s
    oq3-1s   1866.8 us  100.6 MB  53.9 GB/s   SAME TILE, vector loads (the DMA floor)

**The 3-bit tile streams beautifully — 100.6 MB in 1866.8 us against coarse's 131.8 MB in
2396.8. It moves 76% of the bytes in 78% of the time, a real ~22% win.** The entire deficit
is the UNPACK: 4781 against an 1866 floor. To beat coarse, oq3 must land under 2397 us, so
the unpack overhead has to fall from 2914 us to under 530 — a further 5.5x.

Four formulations were measured, and the op count is not the lever it looks like:

    32-lane spread, bf16 MAC                     7187.9 us
    64-lane spread (native int8 width), bf16 MAC 4781.1 us   1.50x

**The remaining gap is a design error, not a tuning problem.** The format is W3A4 —
FOUR-BIT ACTIVATIONS — and both kernels above convert each code to bf16 and MAC in bf16
(`to_float` + `mul` + bf16 MAC). That is the expensive part and it is not what W3A4 means.
The intended path stays integer: int8 codes against int4/int8 activations, int32
accumulate, group scale applied ONCE to the integer sum. That deletes two ops per iteration
and moves the MAC to the integer pipeline. Plausibly the 3-5x required; more tuning of the
bf16 form is worth ~1.2x. It is a pipeline change, not a kernel one, because it needs
quantised activations and it moves accuracy.

**This also settles QTIP-3**, which is the better 3-bit quant and cannot help here: its
trellis decode needs MORE arithmetic per weight than oq3's spread, so it cannot clear a bar
oq3 misses by 2x. Rate-distortion does not rescue a compute-bound decode. If the integer
W3A4 path lands, revisit — the bandwidth is identical (100 B vs 98 B per group) and the
quality is much better.

**`#pragma clang loop unroll(full)` MISCOMPILES the oq3 decode loop**: rel 2.8e-1 unrolled
against 2.8e-3 rolled, identical source. The same pragma on oq4's loop is correct, so it is
not universally broken — but it cannot be used without a correctness check beside it.

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
