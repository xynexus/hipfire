# The prefill hidden-state gate reported five comparisons that never happened

Status: **FIXED 2026-09-02** (the gate now says so). The underlying #377
divergence is unchanged and still open.

## What was wrong

`tests/tiny-prefill-gate.sh` compares a BATCHED prefill against a per-token one
and asserts, for unquantised KV:

    hidden fp32: 0.00e0 (invariant: must be exactly 0)

On several configurations the batched path **declines**, both arms run per-token,
and the probe compares a run against itself. The invariant then reads as
satisfied because nothing was measured.

Measured on the tiny fixtures, checking whether the batched arm actually ran:

| fixture | fp32 | q8 | kvarn |
|---|---|---|---|
| qwen3_5 | **not measured** | 5.75e-3 real | 3.35e-2 real |
| qwen3_5_moe | **not measured** | **not measured** | **not measured** |
| qwen3_5_moe_indexed | **not measured** | 7.16e-2 real (FAIL) | 7.40e-2 real (FAIL) |

Five of the eleven cells were comparing nothing. `qwen3_5_moe` was reporting
`0.00e0` across all three modes — a perfect score for a fixture whose batched
path never executes.

## The fix

The batched arm announces itself with a `[features] ... prefill batched` line.
The gate now requires that line before believing a number, and reports a missing
one as **NOT-MEASURED** rather than as a pass.

NOT-MEASURED is deliberately NOT counted as a failure. A batched path declining
for a fixture is a coverage gap, not a divergence, and conflating the two would
make the gate cry wolf on five cells that have no defect. It is counted and
printed separately:

    tiny-prefill-gate: ran=4 fail=2 skip=2 not-measured=5

The failure count is unchanged — the two real `qwen3_5_moe_indexed` divergences —
while the five gaps are now visible instead of green.

## Why this keeps happening

This is the THIRD vacuous-pass defect found in this probe in one session:

1. `--n 32` — below the 256-token prefill chunk, so attention never reads the KV
   cache and every `--kv-mode` returns identical numbers. The probe warns about
   it; the gate ran that way for its whole life
   (`2026-09-02-kvarn-write-path-is-batch-invariant.md`).
2. fp32 KV in the standalone probe — the same declined-batched-arm problem, which
   made an "fp32 KV is exact" reading unsupportable.
3. This.

The pattern is a comparison tool that returns a number whether or not it compared
anything, and callers that read the number without checking. The durable fix is
the one applied here and in `gdn_chunk_seq_parity`: make the check assert that it
ran, and prove it can fail before trusting it.

## #377 status

The two genuine failures remain: `qwen3_5_moe_indexed` batched prefill diverges
from per-token at **layer 1**, 7.16e-2 (q8) and 7.40e-2 (kvarn), against a 5e-2
ceiling. The FP16 GDN chunk-invariance fix landed earlier today did NOT change
them, so the cause is not the DeltaNet recurrence. Layer 1 on this fixture is an
indexed-MoE layer, so the routed-expert path is the remaining suspect — the
probe's own `MoE PATH-2 GATE` (grouped vs indexed, both batched) passes at
0.000e0, which narrows it to batched-vs-per-token within the indexed path rather
than a grouped/indexed disagreement.
