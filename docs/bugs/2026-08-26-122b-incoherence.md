# Qwen3.5-122B-A10B serves incoherent text

Status: **OPEN**. Opened 2026-08-26. Summary entry in `BUGS.md`.
Full measurement log: `docs/plans/2026-08-26-122b-perf-findings.md`.

## Symptom

`122b-lmbf16.hfq` and `Qwen3.5-122B-A10B--oq4.25++fix.hfq` both emit
BYTE-IDENTICAL garbage:

    <think>\n\nHere's a thinking'skeyider'\n<think>\nHere's a thinking, [\n</think>\n\n theur\n\n the.焄

Loading and memory are healthy — 68.99 GB resident against a 64.56 GiB payload,
so the old 3.5x GTT blowup (expanded experts + 2 MiB rounding, ~137 GiB for a
63.9 GiB artifact) is gone. Consistent with `a51be9b78` ("the 122B itself is NOT
fixed").

Note the garbage is a CORRUPTED version of the control's correct opening
("Here's a thinking process:" → "Here's a thinking'skeyider'"), which reads as
slightly-wrong numerics rather than a broken code path.

## Control

`Qwen3.6-35B-A3B--oq4.25++` — same arch (`qwen3_5_moe`), same kernels, same
quant family — is coherent at 61.5 tok/s decode.

| | expert dtypes | gate_up AWQ missing | down AWQ missing |
|---|---|---|---|
| 35B (coherent) | **uniform** `OqPlusCompact` | 0 (0%) | 0 (0%) |
| 122B (incoherent) | **mixed** compact + Oq8 | 644 (5.2%) | 2018 (16.4%) |

The control never exercises a mixed layer nor an expert without an AWQ sidecar.
37 of the 122B's 48 layers are mixed.

## Ruled out, each by measurement

- **lm_head.** `--fix` and `122b-lmbf16` differ ONLY there (2 `Bf16Lut3` tensors
  vs 1) and the garbage is byte-identical, so the earlier lm_head→BF16 fix is not
  the current cause.
- **OQ8 router path.** `HIPFIRE_OQ8_ROUTER=1` changes nothing.
- **Per-expert missing AWQ.** The design is documented ("a 0 entry means that
  expert has no sidecar", `qwen35/layout.rs`) and the kernel honours it:
  `rotate_x_mq_awq_indexed_batched.hip:64` resolves the pointer per expert,
  `:74` skips the rotation when null.
- **Mixed compact+Oq8 expert layers — the leading suspect, and it is WRONG.**
  `tests/tiny-moe-mixed-gate.sh` builds the same layout at 13 MB. Mixing moves
  KLD by under 1%:

      HFQ -> oq4.25++, 14 promoted, NONE an expert ....... 0.1960
      HFQ -> oq4.25++, 14 promoted, 3 Oq8 + 61 compact ... 0.1946

- **Compact residency / quantization of unrotated data.** Exonerated upstream by
  `f3d3a5efd` (106 real-weight checks) and `ce7a3d25f` (626 unrotated tensors at
  cosine 1.000000 against source).

## A near-miss worth recording

The sharpest circumstantial evidence pointed at the mixed path and was still
wrong. Of the three 122B artifacts, only the Oq8-fallback one LOADS:
`servable_by_stride_table` accepts `OqCompactG256 | Oq8G256` only, so the
BF16-fallback variant (`122b-passA`) cannot be compact-resident and needs
148.7 GiB. So "the only loadable 122B is the one riding the newest, least-covered
path" — true, and not the cause. Building the reproducer is what settled it.

## Remaining candidates

1. **Scale or real weights.** The toy has 16 experts of [1536, 256]; the 122B has
   256 of [~5120, 3072] with top-8 routing. The stride-table arm
   (`gemv_oq_compact_moe_indexed.hip:76`) could be correct at toy shapes and
   wrong at production ones — `n_ov` bounds and 256-expert pointer tables are
   untested there.
2. **Something outside the MoE FFN.** The bisect only isolated expert precision;
   the two models also differ in depth, E (256 vs 128) and dim.
