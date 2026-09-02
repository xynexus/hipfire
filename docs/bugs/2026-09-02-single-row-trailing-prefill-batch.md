# A prefill batch with exactly ONE trailing row computes that row's attention wrong

Status: **OPEN, newly found 2026-09-02** while investigating #377. Distinct from
#377's reported symptom; see "Relationship to #377".

## The defect

`PREFILL_MAX_BATCH = 256`. A prompt of length `n` is processed as
`floor(n/256)` full batches plus a trailing batch of `n mod 256` rows. When that
trailing batch holds **exactly one row**, that row's hidden state diverges
catastrophically from the per-token reference.

`compare_prefill_hidden_paths --n 257 --kv-mode q8`, worst rows by relative error:

    dense qwen3_5        256:4.2e-1   102:5.7e-3  233:5.6e-3  62:4.7e-3 ...
    qwen3_5_moe_indexed  256:8.7e-1   205:7.2e-2   66:2.9e-2  56:1.2e-2 ...

Row 256 — the lone row of the second batch — is 74x worse than the next row on
the dense fixture and 12x worse on the MoE one. Everything else in the ranking is
ordinary.

It is the batch SIZE, not the position. Widen the tail and it disappears:

    n=257  (tail = 1 row)   row 256 at 8.7e-1   <- broken
    n=258  (tail = 2 rows)  row 256 not in the top 8
    n=260  (tail = 4 rows)  row 256 not in the top 8

## Why no gate sees it

`tests/tiny-prefill-gate.sh` runs the probe at `--n 300`, a 44-row tail, which is
clean. Dense `qwen3_5` PASSES that gate at 5.75e-3 while carrying a 4.2e-1 row at
`n=257`. Any prompt whose length is `1 (mod 256)` hits this in production and
nothing would report it.

## Scope

- Reproduces on TWO structurally different fixtures — dense `qwen3_5` and
  `qwen3_5_moe_indexed` — so it is not MoE-specific.
- Reproduces under `q8` KV. On `qwen3_5_moe_indexed` with `kvarn` the row-256
  blowup does NOT appear (worst is an ordinary 160:7.4e-2), so it is at least
  partly KV-path dependent.
- That KV-mode dependence also argues against a probe bookkeeping artifact: a
  ring-buffer or comparison bug in the harness would not care which KV format is
  in use.

## Relationship to #377

#377 reports `qwen3_5_moe_indexed` exceeding the 5e-2 hidden-state ceiling at
7.16e-2 (q8) and 7.40e-2 (kvarn). That is a DIFFERENT effect:

- it is **row 205**, and the offending set is identical at n=257, 258, 260, 300
  and 384 — content-dependent, not batch-dependent;
- it sits on a broad base of ~1e-3 divergence across 298 of 300 rows;
- the FP16 GDN chunk-invariance fix landed the same day did not move it.

Layer 0 of that fixture (`linear_attn`, no KV) is clean at 4.76e-4 with zero rows
over 1e-3, while layer 1 (`self_attn`, reads the quantised KV) carries all of it.
Both layers are MoE with 10 experts, so the router is not implicated — the
divergence tracks the KV-reading attention layer.

So #377 is plausibly "quantised-KV attention error on outlier tokens exceeds a
tight ceiling", which is a different question from this document's batch-size
defect, and may be a ceiling-calibration decision rather than a code fix.

## Next

1. Read the batched prefill's handling of a 1-row batch — the degenerate case is
   where a tile/wave-shaped kernel is most likely to fall back or mis-stride.
2. Add `--n 257` to `tests/tiny-prefill-gate.sh` so the 1-row tail is covered.
   Deliberately not done in the same change as the discovery: the gate is
   currently reporting NOT-MEASURED cells and adding a new red cell on top would
   muddy which failure is which.
