# TODO: add mixed-precision Opus Quant to the tiny-model tests

**Status:** open (proposed 2026-07-21).

## Motivation

The `tiny_quant` battery (`crates/hipfire-eval/src/executor_tinyquant.rs`,
`families()` / `tiny_quant_rows()`) exercises each arch's quantizer→loader→dequant
path on a tiny random-init fixture and KLD-scores it vs a per-family anchor. Its
Opus-Quant coverage today is **pure-width only**:

- `gemma3`: `candidates = [q8f16, hfq4, oq4, oq8]`, `calibrated = [oq4++]`.
- no family exercises **mixed-precision** OQ.

A single mixed-precision OQ cell would cover more than a pure-width one: the
emitted model contains **both** oq4 and oq8 tensors, so it tests the oq4 codec,
the oq8 codec, **and** the per-tensor mixed-precision assignment machinery in one
cell — exactly the gap. (Confirming the framing that prompted this: yes, one
mixed cell = oq4 + oq8 + the assignment system.)

## What "mixed-precision Opus Quant" is (two distinct systems)

Both live in `crates/hipfire-quantize/src/mixed_precision.rs` and are selected in
`main.rs`:

1. **Decimal-bitwidth OpusMixedSpec** — formats like `oq4.25++` (also the
   `DEFAULT_QUANT_FORMAT`), parsed by `parse_opus_mixed_format`. Per-tensor blend
   of oq4/oq8 to hit a target bpw. Emitted layout is `OqPlusCompact`.
2. **Tiered assignment** — `assign_tiers` / `TierPlan`, driven by
   `--mix-target-bpw` (+ `--mix-floor`). 3-tier oq2/oq4/oq8 by `oq2_sensitivity`;
   `--mix-floor` excludes oq2 to explore the oq4/oq8-only regime (commit
   983abc133). NOTE per `main.rs:268` the mixed `TierPlan` is currently consulted
   at the **LFM2 dense-linear** call site — verify whether it applies to other
   archs before assuming a gemma3 cell exercises the tiering (see Open questions).

## Proposed change

Add mixed-precision candidate(s) to a `FamilyPlan`. `gemma3` is the natural host
— its loader already handles `oq4` and `oq8`, so a per-tensor mix should load if
it dispatches quant type per tensor.

- Add `"oq4.25++"` to `gemma3.candidates` (OpusMixedSpec / OqPlusCompact path),
  and/or a `calibrated` `oq4.25++` cell (also exercises the Hessian/LDLQ path on
  the mixed format).
- Optionally a tiered cell via `quant_flags` (`--mix-target-bpw <x> --mix-floor`)
  if the tiering is generic to gemma3 (Open question #2).

Then record baselines and commit them:

```bash
HIPFIRE_TINYQUANT_RECORD=1 ./tests/tiny-quant-gate.sh   # on gfx1151
# repeat on gfx1103; update tests/tiny-quant-baselines.txt
# (and tests/fixture-golden-baselines.txt if a golden cell is added)
```

## What it would validate

- Mixed emission: quantizer produces a per-tensor oq4/oq8 blend (`OqPlusCompact`)
  and the assignment logic runs on a real (tiny) model, not just unit tests.
- Loader: the arch loader accepts a model whose tensors carry **different** OQ
  quant types — the mixed-serving path, distinct from a uniform oq4 or oq8 model.
- Regression tripwire: any future change to the assignment heuristics, the
  OqPlusCompact codec, or the mixed loader is caught by KLD drift.

## Open questions / risks

1. **Per-tensor-mixed loader support.** gemma3 loads uniform oq4 and uniform oq8;
   does it dispatch a *mixed* model (some tensors oq4, some oq8) correctly? Verify
   before recording a baseline — this is the main unknown.
2. **Is `TierPlan` generic or LFM2-gated?** `main.rs:268` implies the tiered plan
   is consumed at the LFM2 dense-linear site. If LFM2-only, a gemma3 cell won't
   exercise the tiering and either the plan must be generalized or an `lfm2`
   fixture family added to `families()`. The OpusMixedSpec (`oq4.25++`) path is
   the safer first cell if so.
3. **oq2 not generally serveable.** Full 3-tier (oq2/oq4/oq8) can't tiny-test
   until oq2 serving lands (codec+DType+gemv+dispatch — see
   `project_opus_quant_family_coverage`). Use `--mix-floor` (oq4/oq8 only) so the
   cell stays within the serveable regime.
4. **Anchor.** gemma3's anchor is `fp16`; the mixed cell scores vs that, same as
   the existing oq4/oq8 cells — no new anchor needed.
