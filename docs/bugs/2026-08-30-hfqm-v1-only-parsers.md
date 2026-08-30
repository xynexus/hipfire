# Three HFQM parsers read the version field and threw it away

Status: found and **FIXED** 2026-08-30 on master `e4025250f`, nix2. Confirmed
against real artifacts in `/srv/hipfire/models` before and after. No GPU needed.

Found by running one of the mechanical sweeps
`docs/bugs/2026-08-29-hunt-coverage-gaps.md` proposed but never executed —
"20 files parse `b\"HFQM\"`; 12 have no `version >= 2` branch". Three of the
twelve turned out to be live; the rest are writers, magic-only checks, or
delegate to the canonical reader.

## The defect

HFQM v2 adds a `u64 data_offset / 32` to each index entry, taking the per-entry
tail from 12 bytes to 20, and drops v1's promise that payloads are stored back
to back. `hipfire-quantize` has emitted v2 since `hfq_out::HFQ_VERSION = 2`, and
most artifacts in the store are v2:

    BLS-Mini-Code-1.0--bf16.hfq       ver=2
    gemma-4-31B-it--bf16.hfq          ver=2
    gemma-4-E2B-it--bf16.hfq          ver=2
    EmbeddingGemma-300M.bf16.hfq      ver=1

Three parsers read the version into a discarded binding and then walked a v1
index unconditionally, deriving each offset from a running sum:

| file | binding | what it does with the result |
|---|---|---|
| `crates/hipfire-train/src/hfq_patch.rs` | version never read | 6 consumers, incl. `patch_norms_inplace` — writes at those offsets |
| `crates/hipfire-runtime/examples/hfq_split.rs` | `version` parsed, unused in the walk | copies payload ranges into split outputs |
| `crates/hipfire-quantize/src/tools/draft_to_mq4.rs` | `let _version` | copies payload ranges into a new artifact |

On a v2 file the 8 offset bytes are read as the NEXT entry's `name_len`, so
every entry after the first is garbage. Observed:

    $ parse_hfq(gemma-4-E2B-it--bf16.hfq)
    ERR: invalid utf-8 sequence of 1 bytes from index 84

That error is luck. `draft_to_mq4` uses `String::from_utf8_lossy`, so a garbage
name does not error there at all — it lands in the output artifact along with
payload sliced from the wrong offsets.

## Second defect in the same function

`hfq_patch::parse_hfq` returns a `Result` but sliced raw. A truncated container
panicked instead of erroring — `/srv/hipfire/models/gemma-4-31b.bf16.hfq` is a
4096-byte stub whose header claims `data_offset = 37228544`:

    thread 'main' panicked at hfq_patch.rs:41:
    range end index 37228544 out of range for slice of length 4096

## Fix

All three now branch on the version and take v2's explicit offset; all three
refuse a version above 2 rather than guessing a layout. `hfq_patch`'s index walk
goes through a bounds-checked `take`, so a truncated file is an `Err`.

Verified on real artifacts: `gemma-4-E2B-it--bf16.hfq` (v2) parses to 2011
entries whose names, order and stored sizes match `hipfire inspect --tensors`
exactly; `EmbeddingGemma-300M.bf16.hfq` (v1) parses byte-identically to before;
the 4096-byte stub returns an error.

`hfq_patch` gains three unit tests, the load-bearing one being that a v1 and a
v2 encoding of the same logical content parse to the same entries — the
property that was broken.

## Not defects, checked

- `mtp_extract.rs` reads back a file it just wrote as v1, so its v1 walk is
  correct by construction. It now asserts the version it reads back equals its
  own `HFQ_VERSION` instead of discarding it, so bumping the writer without
  teaching the walk fails loudly.
- `mq4_merge_mtp.rs` concatenates whole containers and checks magic only.
- `dspark_convert.rs`, `dspark_export.rs` are writers.
- `hfq_cli.rs`, `hipfire-evidence`, `induction/orchestrate.rs`,
  `serving-core/load.rs`, `hipfire-lora-hfq` do not walk an HFQM tensor index.

## The other half of the sweep, refuted

The same coverage doc flags "`HFQ_MAGIC`/`HFQ_VERSION` is redeclared in 8 files
with the value disagreeing". The disagreement is real — `hfq_out.rs` says 2,
five other files say 1 — but it is **not a defect**. All five are writers that
emit a self-consistent v1 container (12-byte tails, contiguous payloads), which
every reader still accepts. Importing the canonical `HFQ_VERSION` into them
would be the actual bug: they would claim v2 while writing a v1 index. Do not
re-file this as constant drift; if the writers should move to v2, that is a
format change, not a de-duplication.
