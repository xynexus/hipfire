# TODO: record the 10 missing gfx1103 tiny-state baselines — AFTER verification

**Status:** OPEN. Blocking `tiny-affected-gate --require-coverage` from ever
reaching a verdict on this box.

## Symptom

`./tests/tiny-affected-gate.sh --base origin/master --require-coverage` on nix1
(gfx1103, `hip=7.14.60850-d34cbb6409`) reports:

    tiny-state-gate: INCONCLUSIVE (10 skipped/no-baseline cell(s), 8 matched)
    tiny-affected-gate: INCONCLUSIVE -> some gate reached no verdict

Zero cells FAIL. Every family that has a baseline matches it. The verdict is
withheld purely because ten families have no gfx1103 baseline recorded at this
HIP version:

| family/format | observed state hashes |
|---|---|
| gemma3/fp16 | `0xd1dd59a3807131d8` `0xfb591816f3e3b34d` |
| gemma3_vl/fp16 | `0x72e13898ce117e12` `0xfaf10e68e8ce3735` |
| gemma4_dense/fp16 | `0x184dd5465e461e1d` `0x6f30179301ebd645` |
| gemma4_moe/fp16 | `0x20426fedd1d90bb3` `0x86b6caa04468b010` |
| gemma4_ple/fp16 | `0x6611ccb2a600853a` `0xf0be579c9a58c48d` |
| mamba2/fp16 | `0x41484800e3bd1e1f` `0xea870c172ccfeef5` |
| qwen3_5/fp16 | `0xccf17f154e164290` `0xbccf1d6a241a4482` |
| qwen3_5_moe/fp16 | `0x0a181c7e2756317a` `0x25975b98e79e8b12` |
| qwen3_5_moe_indexed/fp16 | `0x88bee0d762b4413d` `0x896886a88a4039f3` |
| qwen3_5_vl/fp16 | `0x0f69e5d73fa4f254` `0x7d4becc77df5201e` |

Observed 2026-08-25 at `d6a51bb15`.

## Why this is NOT just "run --record"

**Recording a baseline blesses whatever the code does today as correct.** Six of
these ten families are qwen3.5, and the qwen3.5 prefill path was substantially
rewritten immediately before these numbers were taken (`0bbbfd08f`, which folded
the duplicated `DeltaNetMoe` / `FullAttnMoe` attention bodies onto the shared
lowered super-ops and deleted 1834 lines). Recording now would freeze that
rewrite as the reference with no independent oracle behind it — and if any of it
is wrong, the baseline makes the error permanent and invisible, which is the
exact false-coverage failure mode the tiny gates exist to prevent.

That is why this is a TODO and not a one-liner. **Verify first, record second.**

## What is already verified, and what is not

Already covered, so do NOT re-litigate it:

- **Prefill** — `tiny-prefill-gate` runs a live in-tree differential (batched vs
  per-token) with no baseline at all. `ran=4 fail=0`: `qwen3_5` 2.04e-6 / 9.6e-7
  and `qwen3_5_moe_indexed` 5.4e-6 / 2.8e-6 against a 1e-4 tolerance, both KV
  modes, with `--corrupt-kv-prefix` still caught at 0.34 so the check can fail.
  Execution is proved positively via `BATCHED_PREFILL_ROWS`, not inferred.
- **Decode is untouched by that rewrite** — and this matters most here, because
  tiny-state exercises DECODE. `decode_layers.rs` is untouched; `moe_decode.rs`
  and `mod.rs` differ from the merge base only by rustfmt line-wrapping plus one
  `pub use` (`git diff -w` confirms no semantic change). So the ten hashes above
  should be identical to what pre-rewrite code produces.
- **Real model** — `Qwen3.6-35B-A3B--oq4` is admitted to the batched path
  (`force_fallback=false n=57 K=8/E=256`) and generates correctly.

NOT covered: the four non-qwen3.5 families (gemma3*, gemma4*, mamba2) have no
baseline for reasons unrelated to any of this work. They were already missing.

## Suggested procedure

1. **Prove the hashes predate the rewrite.** Cheapest real check: a scratch
   `git worktree` at the merge base (`git merge-base origin/master <this>`), run
   `tests/tiny-state-gate.sh` there, and diff the ten observed hashes against
   the table above. Identical => the rewrite provably did not perturb decode and
   the numbers are safe to record. Use a worktree, not `git stash` — see
   AGENTS.md on why.
2. Only then record, scoped by family rather than wholesale:
   `HIPFIRE_TINYQUANT_FAMILIES=<family> ./tests/tiny-state-gate.sh --record`
   (scoping `--record` with FAMILIES is the documented habit; a bare `--record`
   has re-recorded unrelated drift before).
3. Re-run `./tests/tiny-affected-gate.sh --base origin/master --require-coverage`
   and confirm it reaches a real verdict instead of INCONCLUSIVE.

If step 1 shows any hash MOVED, stop and treat it as a decode regression in the
rewrite — do not record.

## Related

- `docs/plans/2026-08-24-raw-f16-moe-prefill-divergence.md` — the defect the
  rewrite fixed, the measurement chain, and two traps it set.
- `tests/tiny-state-gate.sh` header — the baseline traps this gate already
  documents.
