# R14B/R14C — activation-stationary + 8 weight streams: **the shim channel count is not the wall**

Negative result, logged per the design-guide rule ("a feature is under-utilized only until
measured to help — log the ones that don't").

## Hypothesis under test

At DFlash shapes (M=16, K- and N-blocked, W4A8) the R14 whole-array GEMM is pinned at
**9.3–9.8 GB/s on the weight stream** in every ablation: halving MACs bought 4%, halving A
bought 1%, extra buffering bought 0%. The suspected root cause was that only **4 of the 8
shim MM2S channels carry weights** — R14 spends the other MM2S of each column shim on the
A-stripe, and that A channel moves only 4096 B/block against the W channel's 16384 B/block.

Prediction: hoist A off the shim, split each column's W stripe across **both** MM2S
channels, get 8 balanced weight streams, and land at 13–16 GB/s.

## What was built

`r14c_gen.py` — W stripe pulled by both of column j's MM2S channels and reassembled with an
objectfifo **join** in the memtile (`link [@wsh_j0, @wsh_j1] -> [@wbc_j]`), activation held
in a core-resident `aie.buffer` with an initial value. 8192 B/block on all 8 channels.
Args add `NCH` (1 or 2 shim channels per column) and `ROWS` (active core rows) so channel
count and broadcast fanout can be varied independently. `r14b_run.sh` builds the R14 control
with identical kernel flags and runs both through `npu_gemm_bench`.

Verified in the lowered IR (`input_with_addresses.mlir`): the variant emits
`wsh{0..3}_{0,1}_shim_alloc(MM2S, 0)` **and** `(MM2S, 1)` on all four shim tiles — 8 weight
channels, zero activation channels. The control emits 4 `wsh` + 4 `ash`.

## Measured (nix1, gfx1103 / npu1, LM=1 LN=4 KT=64 MT=1 NT=4, N_BLK=128, W = 8 MB, 200 iters)

| config | W chans | A path | cores | C[0] (expect) | time | **W GB/s** |
|---|---|---|---|---|---|---|
| R14 control | 4 | 2 MB streamed | 16 | 1024 (1024) ✓ | 872.0 / 883.4 / 891.7 µs | **9.5** |
| R14C, 1 ch/col | 4 | resident | 16 | 3072 (3072) ✓ | 834.0 / 871.1 µs | **9.8** |
| R14C, 2 ch/col | 8 | resident | 16 | 3072 (3072) ✓ | 818.8 / 825.6 / 838.2 µs | **10.1** |
| R14C, 2 ch/col | 8 | resident | 8 | 3072 (3072) ✓ | 825.5 µs | 10.2 |
| R14C, 2 ch/col | 8 | resident | 4 | 3072 (3072) ✓ | 827.2 µs | 10.1 |
| R14C, 2 ch/col, N_BLK=256 | 8 | resident | 16 | 3072 (3072) ✓ | 1695.9 µs (W = 16 MB) | 9.9 |
| R14C, 1 ch/col, N_BLK=256 | 4 | resident | 16 | 3072 (3072) ✓ | 1766.7 µs (W = 16 MB) | 9.5 |

Correctness is exact in every row. The resident-A configs use `AVAL = 3` (not 1) precisely
so the gate proves the resident buffer is really read: C[0] = 3 × KT × 16 = 3 × 1024 = 3072.

**Doubling the weight channels 4 → 8 bought ~3%; removing the entire 2 MB activation stream
bought another ~4%.** Run-to-run spread is 2–4%, so the whole effect is ~6% and the wall
moved 9.5 → 10.1 GB/s. Nowhere near the predicted 13–16.

The fanout sweep is the decisive control: **4 cores and 16 cores take the same time** (827.2
vs 818.8 µs) at identical weight bytes. Time is set entirely by the DDR→shim weight stream —
not compute, not the memtile broadcast, not the core side. Time also stays linear in W bytes
(2× W_BLK → 2.05× time), so the variant is still weight-byte-bound.

## Hard walls found on the way (npu1-specific)

1. **A core tile has only 2 S2MM channels.** R14 already uses both (W broadcast + A
   broadcast). Any design giving a core a third inbound objectfifo — e.g. splitting blocks
   even/odd across two (W,A) fifo pairs — is rejected:
   `'aie.tile' op number of input DMA channel exceeded!`
2. **`ObjectFifoLinkOp` cannot be a join and a distribute at once.** The natural
   activation-folded design — fuse `[W_j | A_j]` into one shim object, pull it with two MM2S
   channels (join), then split it in the memtile into the W column-broadcast and the A
   row-broadcast (distribute) — fails at parse:
   `ObjectFifoLinkOp does not support 'join' and 'distribute' at the same time`.

Consequence: on npu1 you can have 8 weight channels **or** a dynamically-streamed activation,
not both, unless the memtile's 4 C-join S2MM channels are freed first (e.g. by relaying C
core-to-core through shared memory before it reaches the memtile). Given the measurement
above, that work is not worth doing for bandwidth — the channel count is not what binds.

## Implication for the DFlash GEMM term

The projection scales as measured, not as hoped: ~52 ms at 9.3 GB/s becomes **~48 ms**, not
the ~37 ms that 13 GB/s would have given. The weight path saturates near **~10 GB/s per
dispatch** regardless of how many shim channels carry it, which is below the ~13–16 GB/s
aggregate DDR ceiling and above what any single-stream config reaches. The next lever has to
reduce weight **bytes** (deeper quantization, weight replay across more activation rows —
i.e. larger M per weight fetch) or overlap dispatches; it is not more DDR streams.
