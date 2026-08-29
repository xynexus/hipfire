# `hipfire model compose` is broken for every default bf16-codec artifact

Status: found and **FIXED** 2026-08-29, master `0c9e3d252`, nix1. Confirmed 3/3,
reproduced end-to-end, then fixed via a new `HfqFile::physical_extent` accessor
that all four arithmetic sites now use. Verified by composing a real artifact
with 110 `Bf16Huff` tensors: the bundle carries 137.19 MB packed (not 207.14 MB
expanded) and re-inspects at the same 1.510x ratio. Reverting only the coverage
loop reproduces the original error.

## Symptom

```
$ hipfire model compose --check /srv/hipfire/models/LFM2.5-350M--bf16.hfq aux.hfq
compatible: 2 component(s), bundle arch 11, manifest hipfire.hfqm.compose.v2

$ hipfire model compose -o bundle.hfq /srv/hipfire/models/LFM2.5-350M--bf16.hfq aux.hfq
Error: /srv/hipfire/models/LFM2.5-350M--bf16.hfq contains overlapping tensor ranges
```

`--check` passes and the compose aborts, **naming a corruption that does not
exist** — the artifact is well-formed. `hipfire induct` fails the same way at its
final fold-in step, which shells `hipfire model compose`
(`crates/hipfire-cli/src/commands/induct.rs:553-562`).

## Cause

`expand_bf16_index` (`crates/hipfire-runtime/src/hfq.rs:1017-1078`) presents a
**logical** view of a losslessly recoded tensor: it rewrites
`t.data_size = stored.logical_byte_len(n)` and `t.quant_type = stored.logical()`
while leaving `t.data_offset` at the **packed** offset. The physical extent
survives only in a private table (`hfq.rs:967`, `physical_range` `hfq.rs:1804`,
`stored_recoding` `hfq.rs:1907`).

That expansion runs inside `open_index_only_at_offset` (`hfq.rs:1211`), which is
what compose uses — `HfqFile::open_index_only` at
`crates/hipfire-hfq-tooling/src/lib.rs:1400`.

`compose_hfq` then does raw file-range arithmetic on that logical view:

| site | what it uses |
|---|---|
| `lib.rs:1509` | `data_len: entry.data_size as u64` — expanded |
| `lib.rs:1522` | `original_offset: entry.data_offset as u64` — packed |
| `lib.rs:1573` | `cursor = offset.checked_add(entry.data_size as u64)` |
| `lib.rs:1753` | `io::copy(&mut file.take(info.data_size as u64), w)` |

Mixing a packed offset with an expanded length makes each tensor appear to run
past the next one's start, so the coverage loop reports overlap. `hipfire-hfq-tooling`
has **zero** references to `stored_recoding` / `physical_range` / `has_bf16_expanded`
— there is no guard anywhere in the crate.

## Arithmetic, from the real index

Parsing the HFQM v2 index of `LFM2.5-350M--bf16.hfq` and replaying the
`lib.rs:1573` loop:

| view | overlaps | final cursor |
|---|---|---|
| logical (what compose sees) | **135** | — |
| stored (physical truth) | **0** | 471,791,370 ≤ file_len 477,247,156 |

The first overlap is the second tensor: `model.embedding_norm.weight` at offset
89,272,320 against a cursor of 134,230,016, because `embed_tokens` is stored in
89,260,030 bytes but reports 134,217,728.

## Blast radius

`crates/hipfire-quantize/src/cli.rs:7787`:

```rust
arg_value(&args, "--bf16-codec").unwrap_or("huff")
```

The codec is the **default**, and it applies to BF16-typed tensors in *any*
output format — not just `--format bf16`. A `--format mq4` build of the tiny
fixture carries 17 `Bf16Huff` tensors and compose aborts on it identically. So
essentially every artifact `hipfire-quantize` has produced since that default
landed cannot be composed.

Controlled A/B on one source, only the codec differing:

| build | compose |
|---|---|
| `--format bf16 --bf16-codec none` | `composed 2 inputs -> ./bundle-none.hfq` |
| `--format bf16` (default `huff`) | `Error: … contains overlapping tensor ranges` |

## What is NOT true

The finder also claimed a silent-corruption case — a recoded tensor last, its
expansion delta smaller than a trailing blob, so `lib.rs:1753` copies wrong bytes
into the bundle. **That is unreachable.** With the recoded tensor last, `cursor`
overshoots EOF and a second guard fires first (`lib.rs:1608-1612`, *"tensor data
exceeds file length"*), and `write_hfqm_package_streaming`
(`crates/hipfire-runtime/src/hfq.rs:580-648`) writes nothing after the final
payload, so there is no tail blob to absorb the delta. The real-world outcome is
always a hard abort. Recorded here so nobody re-derives it.

## Fix

Add a physical-extent accessor aware of `stored_recoding(idx)` and use it for the
offset/length arithmetic at `lib.rs:1509`, `:1522`, `:1573` and `:1753`, carrying
the **stored** quant_type into the stream entry so the bundle stays byte-identical.
Failing that, fail closed early with a message naming the recoding rather than
accusing the artifact of overlapping ranges.
