# A prefill batch with exactly ONE trailing row computes that row's attention wrong

Status: **FIXED 2026-09-02**, found while investigating #377 and distinct from it;
see "Relationship to #377".

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

## Root cause

`forward_prefill_batch_with_pbs_opts` decides `eligible` ONCE for the whole
prompt — `prefill_batch_pbs_eligible(..., n, ...)`, which requires
`n >= MIN_BATCH` (2) — and then runs every chunk of the loop batched. A prompt of
257 tokens is eligible on its total length, so the trailing chunk of 1 row goes
through the batched path even though, judged on its own, it would have been
refused.

The multi-GPU sibling in `ep.rs` has the same `n_total >= 2` guard and the same
structure, which is why the guard reads as sufficient until you notice it is
evaluated on the prompt rather than on the chunk.

## Fix

The chunker no longer CREATES a one-row trailing chunk:

    let mut chunk_end = (chunk_start + chunk_batch).min(n);
    if n - chunk_end == 1 && chunk_end - chunk_start > MIN_BATCH {
        chunk_end -= 1;
    }

A 257-token prompt now runs 255 + 2 instead of 256 + 1. Chunk boundaries already
move with `n`, so nothing a caller can observe changes except the defect, and the
guard cannot shrink a chunk below `MIN_BATCH` and recreate the problem.

Fixing the boundary rather than the kernel was deliberate. `forward_prefill_batch`
is a 13k-line path whose own comments record that widening its eligibility gates
produced garbage (KLD 10.69 vs 0.0389 on a MoE model); not producing the
degenerate batch is a change whose blast radius can be reasoned about completely.

**Verified**, `--n 257 --kv-mode q8`:

| fixture | before | after |
|---|---|---|
| dense qwen3_5 | row 256 at **4.18e-1** | no row over 1e-2; tail cell 5.75e-3 |
| qwen3_5_moe_indexed | row 256 at **8.69e-1** | row 256 absent; worst is the pre-existing 205 |

Through the gate, dense `qwen3_5`'s one-row-tail cell went from FAIL 4.18e-1 to
5.75e-3 — identical to its n=300 value, which is what "the tail is no longer
special" should look like. `qwen3_5_moe_indexed`'s tail cells now read 7.16e-2 /
7.40e-2, exactly matching its n=300 cells: the blowup is gone and only #377's
content-dependent divergence remains.

No regressions: `tiny-state-gate` PASS 18/18 (token output unchanged), 171
qwen35 unit tests pass.

## Next## Next

1. ~~Read the batched prefill's handling of a 1-row batch~~ — DONE; see "Root
   cause" and "Fix" above. What remains unknown is WHY a 1-row batch computes
   attention wrongly, as opposed to merely differently: the fix removes the
   degenerate input rather than correcting the kernel that mishandles it. If a
   caller ever constructs a 1-row batch by another route, it will still be
   wrong. Worth a targeted kernel-level look. `forward_prefill_batch` is a 13k-line path whose own
   comments record that widening its eligibility gates produced garbage (KLD
   10.69 vs 0.0389 on a MoE model), and a speculative fix there without the
   ability to validate it broadly would be worse than the documented bug. The
   gate cell above is the safe half: the defect is now covered and cannot
   regress further or be forgotten.

   One lead for whoever takes it: the multi-GPU sibling in `ep.rs` gates batched
   prefill on `n_total >= 2`, so a 1-row batch there falls back to per-token. If
   the in-play path lacks that guard, a batch of 1 may be entering a kernel that
   assumes at least two rows.
2. ~~Add `--n 257` to the gate~~ — DONE. `tests/tiny-prefill-gate.sh` now runs a
   second probe size with a one-row trailing batch, per quantised KV mode,
   reported separately from the n=300 cells so a failure in one is never
   confused with the other:

       qwen3_5              FAIL one-row trailing batch (n=257) q8 = 4.18e-1
       qwen3_5              tail n=257 kvarn: 3.35e-2 (clean)
       qwen3_5_moe_indexed  FAIL one-row tail q8 = 8.69e-1, kvarn = 7.40e-2

   Dense `qwen3_5` passed this gate completely before; it now fails on a defect
   it was always carrying. Overridable via `HIPFIRE_TINYPREFILL_HID_N_TAIL`.
