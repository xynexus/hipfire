# Promote the remaining coexistence groups to real subcommands

Follow-on to step 6 of `docs/plans/2026-08-27-single-binary-merge.md`, whose
status is corrected there. Five groups still reach the user only as
`hipfire convert <group> …`, forwarded verbatim through
`#[command(external_subcommand)]`.

## Why this is not cosmetic

An external subcommand hands clap a bare `Vec<String>`. Clap therefore cannot
enumerate the group, which means **none** of the following can see it:

- `hipfire convert --help`
- `gen-docs` → `docs/CLI.md` and `man/hipfire-convert.1`
- the CLI-docs freshness gate in `tests/no-gpu-ci.sh`, which compares generated
  output against the clap definition and so can only ever confirm that the
  omission is faithfully reproduced

`download` shipped on 2026-08-26 and was undiscoverable from `--help` for four
days for exactly this reason. Promotion is what fixes it, and the test that a
promotion is real is that the man page appears without anyone writing it.

## Done (2026-08-30)

| group | now | note |
|---|---|---|
| `download` | `hipfire download <org/name>` | logic extracted to `hipfire_coexistence::download`; one implementation, two front doors |
| `induct` | `hipfire induct <org/name>` | promoted to top level to mirror `download` |
| `import` | `hipfire import {gguf,safetensors}` | |
| `export` | `hipfire export safetensors` | |
| `repack` | `hipfire repack` | `optimize` lost its colliding `repack` alias |
| `hub` | **retired** | was one spelling of `download` + `repack --check` |
| `lora` | `hipfire lora {export,merge,convert}` | |

## Remaining

| group | ops | shape |
|---|---|---|
| `artifact` | `audit-calibration`, `compare-calibration`, `compare-calibration-stability`, `compare-residuals`, `moe-router-profile` | five read-only reporters; `inspect` is folded into `hipfire inspect` (see below) |
| `calibrate` | one op, ~30 flags | the largest bag by far |
| `two-pass` | one op | shares `induction/` with `induct` |
| `npu` | `pair-hfp` | linux-only (`#[cfg(target_os = "linux")]`) |

Suggested order: `artifact`'s five non-inspect ops, `two-pass`, `npu`, then
`calibrate` last — its flag bag is big enough that a mechanical transcription is
where an argument would silently drift.

## Things to decide, not just transcribe

**`import gguf` spells its paths `--in`/`--out`; everything else uses
`--input`/`--output`.** The promoted commands preserve both spellings exactly,
because scripts already use them. Unifying is a breaking change and wants its
own decision — a clap `alias` could accept both, at the cost of two documented
names for one flag.

**`artifact inspect` overlap — RESOLVED 2026-08-30, and it was not what it
looked like.** `hipfire inspect --json` was already a near-superset, so the
obvious move was to delete the duplicate. Comparing the outputs on the same
artifact showed why that would have been wrong: `artifact inspect` reports an
`artifact_fingerprint` that is **not** the same number as `hipfire inspect`'s
`fingerprint`, and `calibration_audit` records THAT one as an artifact's
provenance identity.

Both answer "identify this artifact's metadata and tensor layout without reading
payloads". `hipfire` computes FNV-1a inline over `metadata_json + arch_id +
per-tensor fields`; coexistence hashes a JSON serialization of
`{version, arch_id, metadata, tensors}` via `hipfire_hash::stable_hash_bytes`.
Same purpose, different algorithms, different values.

`hipfire inspect` now carries both, each under its own name, plus `file_bytes`
(on-disk size, distinct from `totals.bytes` which is tensor payload). Verified
byte-identical to what the coexistence tool reports, so existing provenance
records stay valid.

**The remaining question is the duplication itself, and it is a DATA decision.**
Two fingerprint algorithms for one purpose is one too many, but unifying them
invalidates every recorded `artifact_fingerprint` — calibration provenance and
resume checks. That needs a migration story (record both during a transition,
or version the scope string), not a rename. Until then `fingerprint_scope`
exists precisely so a reader can tell which is which.

`artifact`'s other five ops (`audit-calibration`, `compare-calibration`,
`compare-calibration-stability`, `compare-residuals`, `moe-router-profile`) have
no such overlap and are ordinary promotions.

**The `.hfa` inspect gap.** `hipfire inspect` refuses an `.hfa` with a pointer to
`hipfire-coexistence repack`. That pointer now names a command that no longer
needs the standalone binary — `hipfire repack` — and the message should be
updated when `artifact` is settled.

**Every promotion is a bridge, not a rewrite.** The current commands build an
argv vector and call the existing `run_cli`. That keeps behaviour identical and
the diff small, but it means the flag bag remains the real parser and the clap
definition is a description of it that can drift. Extracting a typed entry point
(as `download` got) is the version that cannot drift; it is more work per group
and worth it where the flags are load-bearing.

## Not in scope

The standalone `hipfire-coexistence` binary stays. All 17 bin targets were kept
deliberately so nothing invoking them by name broke, and that is still true.
