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
| `artifact` | `hipfire artifact {audit-calibration,compare-calibration,compare-calibration-stability,compare-residuals,moe-router-profile}` | `inspect` folded into `hipfire inspect` |
| `calibrate` | `hipfire calibrate` | bridged to `CalibrateCommand::parse`, which keeps sole ownership of defaults and cross-flag rules |
| `two-pass` | `hipfire two-pass` | `--` tail forwarded to the quantizer |
| `npu` | `hipfire npu pair-hfp` | linux only |

## Remaining

**None.** All eleven groups are promoted or retired. `hipfire convert` is back to
its five drafter tools, and the `external_subcommand` catch-all is gone — so
every conversion command is now enumerable, `--help`-visible, and rendered by
`gen-docs` without anyone writing prose for it.

One consequence worth stating: `ArgBag` silently ignored unknown flags, so
`two-pass --typo 1` used to be accepted and dropped. Under clap it is an error.
That is a behaviour change, and a good one, but it will surface any script that
was passing a flag the tool never read.

## Things to decide, not just transcribe

**~~`import gguf` spells its paths `--in`/`--out`~~ — DECIDED AND DONE
2026-09-01: unified on `--input`/`--output`, as a deliberate break.**

The alias option was rejected: two documented names for one flag is a cost
carried forever, and the inconsistency is the thing being removed.

The break names itself rather than failing blankly, which is the half that
matters. On the standalone binary, `--in` returns "`--in` was renamed to
`--input` ... Use `--input <path>`" — rejected BY NAME, because a bare
"--input is required" against a command line that plainly says `--in` is the
error that costs someone ten minutes. That mirrors `RENAMED_KEYS` on the config
side, which honours the old key and says what to write instead.

The promoted `hipfire import gguf` path needed no such shim: clap already
answers with *"unexpected argument '--in' found — tip: a similar argument
exists: '--input'"*. Verified on both entry points.

Blast radius was documentation only — no script in the tree passed the old
spellings. `docs/CLI.md` regenerated; the reference in
`docs/bugs/2026-08-29-gguf-dequant-q4_0-q5_k.md` updated.

**`artifact inspect` overlap — RESOLVED 2026-08-30, and the duplicate is gone.**
`hipfire inspect --json` was already a near-superset. The one thing
`artifact inspect` had that it did not was an `artifact_fingerprint` computed by
a SECOND algorithm — coexistence hashed a JSON serialization of
`{version, arch_id, metadata, tensors}` while the runtime FNV-1a's
`metadata_json + arch_id + per-tensor fields` inline. Same question, two answers,
different values, and `calibration_audit` recorded the coexistence one as
provenance.

Both now use `HfqFile::index_fingerprint`. `crate::artifact::index_identity` and
`index_fingerprint` are deleted, `fingerprint_scope` is deleted (a scope string
distinguishes algorithms, and there is only one now), and `hipfire inspect`
reports a single `fingerprint` plus `file_bytes` — the on-disk size, which is
genuinely distinct from `totals.bytes`, the tensor payload sum. Verified: the
audit report and `hipfire inspect` print the same number for the same artifact.

**This invalidated previously recorded fingerprints, deliberately.** Any
`artifact_fingerprint` written into an induction manifest or calibration audit
before this change carries the old algorithm's value and will not match a
freshly computed one. Accepted rather than migrated: the alternative was
carrying two hashes indefinitely so that stale records keep verifying.

**Every promotion is a bridge, not a rewrite.** The current commands build an
argv vector and call the existing `run_cli`. That keeps behaviour identical and
the diff small, but it means the flag bag remains the real parser and the clap
definition is a description of it that can drift. Extracting a typed entry point
(as `download` got) is the version that cannot drift; it is more work per group
and worth it where the flags are load-bearing.

## Not in scope

The standalone `hipfire-coexistence` binary stays. All 17 bin targets were kept
deliberately so nothing invoking them by name broke, and that is still true.
