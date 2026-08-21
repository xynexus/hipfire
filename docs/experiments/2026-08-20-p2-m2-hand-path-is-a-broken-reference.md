# P2/M2: the hand path is a broken reference — parity is the wrong exit

> **RESOLVED 2026-08-20 — the cause found here is fixed on master (`1f7c2eeba`).**
> The dense DeltaNet arm never applied `ffn_norm`. Root cause, fix, and evidence:
> `2026-08-20-qwen35-hand-dense-ffn-norm-fix.md`. Post-fix, hand and lowered agree
> at every strength measured (5/5, same md5), and both pass the `strength = 0.0`
> identity anchor.
>
> This document is kept as the record of how the breakage was found and why M2's
> exit was rewritten. **The rewritten exit stands** — parity is now available but
> is still the weaker assertion, and the high-strength "agreement on garbage" trap
> below is a property of steering, untouched by the fix. Everything below
> describes the pre-fix state.

**Status:** superseded — was "M2 blocked on a plan decision, not on code".
**Measured on:** nix1 / gfx1103, `Qwen3.5-0.8B-Base--oq8.hfq` and
`qwen3.5-0.8b--oq4++.hfq` (arch `qwen3_5`, 24L, dim 1024), release daemon at
`edc13da5c` (P2/M1).

## Summary

M2's exit is "byte-identical token streams between hand-path steering and
lowered-path steering". That criterion cannot be satisfied here, and **should
not be**: on both artifacts tested, the qwen35 hand decode path is broken
independently of steering. Matching it would mean making the lowered path
reproduce a broken forward.

The escape this plan is retiring has therefore been routing every steer session
onto a forward that miscomputes.

## The measurement that settles it

Steering at `strength = 0.0` is `x += 0·v` — an identity. Both paths must
reproduce the unsteered baseline. Only one does:

| strength | hand path | lowered path | match |
|---|---|---|---|
| 0.0 | 69 tok `  0\n;\n    0\n;` | 126 tok `\nHmm, the user is…` | NO |
| 0.1 | 1 tok `<\|endoftext\|>` | 126 tok `\nHmm, the user is…` | NO |
| 0.25 | 1 tok `<\|endoftext\|>` | 128 tok `The capital of Fran…` | NO |
| 0.5 | 18 tok `:\n\n:\n\n` | 15 tok `\n\n\n\n` | NO |
| 0.75 | 11 tok `00000000000` | 11 tok `\n\n\n\n` | NO |
| 1.0 | 18 tok `>\n\n>\n\n` | 18 tok `>\n\n>\n\n` | YES |
| 2.0 | 11 tok `用电用电…` | 11 tok `用电用电…` | YES |

At `0.0` the lowered path reproduces the unsteered baseline exactly. The hand
path does not — it emits the same garbage it emits with **no steer session at
all**:

```
# no steer session, no apply — pure forward comparison
HIPFIRE_FORWARD_LOWERED=1 → '\nHmm, the user is asking for the capital of France. This is a straight'
HIPFIRE_FORWARD_LOWERED=0 → '  0\n;\n    0\n; ( 0)s0;0\n    0;\n< 0;\n;\n< 0;\n;\n 0; 1 1 1 '
```

Byte-identical to the `strength = 0.0` hand output. The steer hook is not
involved; the hand forward is simply wrong. This corroborates the existing
in-tree comment at `qwen35/decode_layers.rs` ("the hand path is currently
broken — bf16 self-KLD 13.89 vs lowered 0.000",
`docs/roughquant/phase3-real-format-scope.md`).

`qwen3.5-0.8b--oq4++` degrades less violently but is incoherent the same way
(`' How to the answer is the 1.\n\n with a 格式 10 10: 10'`), so this is not one
bad artifact.

## The trap in the table

**Parity is achieved at 1.0 and 2.0 — on garbage.** At those strengths steering
has destroyed the model on both paths and they converge on the same degenerate
attractor (`>\n\n>\n\n`, `用电用电`). An M2 run that picked a single "obviously
steering hard enough to see an effect" strength would have reported parity and
been wrong.

This is exactly the accept-and-miscompute class the plan names. The defence is
not a bigger token budget — 128 tokens of identical garbage is still identical.
It is anchoring on `strength = 0.0`, where the correct answer is known
independently (the unsteered baseline) rather than defined as "whatever the
other path did".

## What this means for the plan

M2's exit needs rewording before it can be run. "Lowered matches hand" is
unsatisfiable while the hand path miscomputes, and satisfying it would be a
regression. Proposed replacement — lowered steering is *correct*, not *equal*:

1. `strength = 0.0` reproduces the unsteered baseline byte-identically (this
   passes today; the hand path fails it);
2. the apply math matches a host-side oracle — `hipfire-steer`'s
   `apply_stack_host` already exists and `examples/gpu_validate.rs` already
   cross-checks the on-GPU decode apply against it;
3. sweep low strengths and require graceful degradation, not agreement with the
   hand path.

Fixing the hand path first is the other option, but it is a larger, unrelated
piece of work, and M4 deletes that path anyway.

## Secondary finding: `CaptureMeans` cannot see the decode boundary

M2 also asks `Capturing` sessions to produce matching `CaptureMeans`, on the
reasoning that a boundary placed one op early or late "shows up there and
nowhere else". Over the daemon protocol it cannot show up there at all:

- `CaptureAcc::observe` only overwrites `current[layer]`;
- only `commit()` folds `current` into `sums` and increments `count`;
- `means()` divides by `count.max(1)`, so `count == 0` yields all zeros;
- `commit()` is reached only from the `steer_capture` op, which is
  **prefill-only** and runs `maybe_steer_block_batched` in `prefill_chunk.rs` —
  a call site M1 does not touch and the escape does not gate.

A capture session driven through `generate` therefore returns all-zero means.
Comparing them across the two paths yields a **vacuous pass** — both sides are
zero. Verified: `steer_begin_capture` → `generate` → `steer_finish_capture`
returns 24576/24576 zero elements on both paths.

To test the decode boundary under `Capturing`, M2 needs either a commit reachable
from decode, or the in-process route (`hipfire-steer/examples/gpu_validate.rs`
calls `maybe_steer_block` directly).

Related: because decode-side `observe` is never committed, an active `Capturing`
session makes every decode token run a `download_f32` per layer whose result is
discarded — 24 wasted device→host copies per token on this model.

## Reproduction

Artifacts under the session scratchpad; the shape is:

```jsonl
{"type":"load","model":"…/Qwen3.5-0.8B-Base--oq8.hfq","params":{"max_seq":1024}}
{"type":"steer_begin_apply","directions":[[…24×1024…]],"mode":"steer","strength":0.0,"layer_start":0,"layer_end":24}
{"type":"generate","id":"p","prompt":"The capital of France is","temperature":0.0,"max_tokens":128}
{"type":"steer_clear"}
{"type":"unload"}
```

run twice, once with `HIPFIRE_STEER_LOWERED=1`. Directions were derived from
real contrastive prefill captures (two arbitrary but distinct prompt sets,
`normalize(mean_A − mean_B)`, verified unit-norm) — not a synthetic constant
vector, which is degenerate and destroys output even at low strength.

Deterministic: the `strength = 0.25` split (hand 1 token, lowered 128) reproduced
3/3 runs.
