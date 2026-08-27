# HFQ compose/decompose for DFLASH and TRIA

Status: implemented

Owner: offline tooling / HFQ format

Related: [bundled model induction program](2026-07-22-bundled-model-induction-program.md)

## Problem

`hipfire model compose` currently opens every input as `HfqPackage`, requires a
sidecar arch ID equal to the base or zero, and inserts every tensor into one flat
index under its original name. That is sufficient for already-partitioned HFQM
role sidecars, but not for the components needed by induction:

- CASK emits a raw `TRIA` v1 file, not HFQM.
- DFLASH emits HFQM with draft arch ID 20, while primary Qwen and Gemma models
  use arch IDs 5/6 and 24.
- DFLASH uses ordinary names such as `layers.0...`, so relaxing only the arch
  check would still collide with primary-model names.

The existing `hipfire.hfqm.compose.v1` manifest cannot describe a renamed entry
or an opaque non-HFQM component. The fix must remain lossless: decompose must
reproduce the input DFLASH HFQM and TRIA files byte-for-byte.

## Scope

Upgrade composition to a component-aware v2 format while retaining v1 read and
decompose support. Keep package writing in offline tooling. The runtime may
share the read-only manifest types but must not gain conversion logic.

Primary seams:

- `crates/hipfire-hfq-tooling/src/lib.rs`
- `crates/hipfire-runtime/src/hfq_compose.rs` (read-only schema/views)
- `crates/hipfire-runtime/src/hfq.rs`
- `crates/hipfire-cli/src/commands/model.rs`
- `crates/hipfire-cli/src/commands/inspect.rs`

## Implementation evidence: 2026-07-22

The current worktree implements `hipfire.hfqm.compose.v2`, reserved component
namespaces, original-name maps, opaque reconstruction spans, whole-source
length/SHA-256 checks, raw TRIA ingestion, arch-20 DFLASH ingestion, exact
decompose, `--check`, JSON output, and fail-closed overwrite behavior. Focused
tooling tests cover byte-exact standard HFQM, colliding DFLASH tensors, raw
TRIA, a three-way Gemma arch-24 bundle, serving-file component views, digest
corruption, geometry mismatch, duplicate roles, reserved namespaces, and v1
compatibility.

All write-side compatibility logic now lives in the dedicated offline
`hipfire-hfq-tooling` crate. `hipfire-runtime` retains only the stable manifest
schema, bounded read views, and digest verification required by inference.
Composition opens only the HFQM header/metadata/index, streams exact tensor
ranges into the output, and releases each completed range from the page cache;
it neither mmaps nor materializes a model-sized component. This also avoids the
AMD-HMM slowdown observed when a 42 GiB calibration artifact was finalized
through a whole-file mapping. The focused 16-test tooling suite and full
no-GPU gate pass. Real Qwen3.5-9B composition streamed a 5.9 GiB target+DFLASH bundle,
validated both strong digests, and loaded its embedded component. The affected
quant and DFLASH spec tiers pass. The affected state tier reports two
pre-existing Mamba2/Qwen3.5-MoE baseline drifts that reproduce byte-for-byte on
an isolated clean HEAD and are not compose regressions.

Composition also rebuilds the base HFQM module table after repacking and drops
the source file's absolute tail-metadata locator. A stale locator/module table
initially produced a structurally written bundle whose tail hash failed at
inspect time; fixed-point metadata sizing plus rebased module offsets now make
the bundle inspectable. A live arch-25 OQ4+CASK compose/decompose rerun through
the index-only path reproduced both source SHA-256 values byte-for-byte.
The product Qwen3.5-2B row also composed a 1,762,802,069-byte, 506-entry
`oq4.25++` base with its 77,824-byte, six-entry HFQM CASK into the canonical
1,794,181,536-byte staging bundle. The manifest records base SHA-256
`12861c5eda4e077b52d47894387a188121d37d91d1fd33dcf84ec22f576fb632`
and CASK SHA-256
`de1191e500d270abba747bd68d227f1e1f5678d09da61e1f471389c8d7a633b7`;
post-compose inspect and embedded runtime load both pass.

## Format contract

Introduce `hipfire.hfqm.compose.v2`. Each component records:

- stable role (`base`, `dflash`, `triattn`, and existing roles);
- source format (`hfqm` or `tria-v1`);
- original filename and byte length;
- original arch ID and verbatim metadata when applicable;
- SHA-256 or the repository-standard strong artifact fingerprint;
- stored entries;
- an explicit `stored_name -> original_name` map; and
- optional opaque payload entry for raw component bytes.

Reserve a namespace that normal model tensors may not use:

```text
__hipfire_component/<role>/<component-index>/<original-name>
```

The base component keeps its original names. Every non-base HFQM component is
stored under the reserved namespace, even when no current collision exists.
This makes role ownership deterministic and prevents a later producer from
silently changing bundle behavior by adding a colliding name.

Raw TRIA input is stored as one opaque byte entry under the same namespace. Add
a named blob encoding for opaque bytes; do not pretend raw bytes are a weight
quant type. The manifest binds the entry length and digest. Decompose writes the
original bytes directly rather than reconstructing TRIA fields.

## Role and architecture policy

Composition validates a role-specific source contract rather than comparing
every component to the base arch:

- `base`: exactly one HFQM component; its arch becomes the bundle arch.
- `dflash`: HFQM, arch ID 20, top-level DFLASH metadata present, and target
  geometry compatible with the base metadata.
- `triattn`: raw TRIA v1 or the future canonical TriAttention HFQM component;
  geometry must match the target's eligible attention layers.
- ordinary same-architecture sidecars: retain the existing base-or-zero rule.

Do not permit an arbitrary mismatched arch merely because its filename contains
`.dflash.`. Role identification must be content-backed.

## CLI behavior

Preserve the requested user surface:

```bash
hipfire model compose BASE.hfq DRAFT.dflash.oq4+.hfq CENTERS.triattn.hfq \
  --output MODEL.dflash.triattn.oq4.25++.hfq

hipfire model decompose MODEL.dflash.triattn.oq4.25++.hfq OUT_DIR
```

The command may delegate to a dedicated tooling crate/binary. Add `--check`
for a read-only compatibility report and `--json` for automation. Never replace
an existing output unless an explicit overwrite flag is supplied.

Default output naming inserts sorted role groups before the primary quant token.
It must not derive the DFLASH component's quant token as the bundle's primary
quant.

## Split behavior

- v2 decompose uses the manifest name map and original component metadata.
- Raw TRIA is copied back verbatim from its opaque payload.
- DFLASH is reconstructed with arch 20 and original tensor names.
- v1 composed models retain their current byte-identical path.
- Manifestless inference remains lossy and must never be used automatically for
  a v2 bundle with a corrupt manifest.
- A digest mismatch is a hard error.

## Tests and gates

1. Synthetic base arch 5 + DFLASH arch 20 with deliberately colliding names.
2. Synthetic base arch 24 + DFLASH arch 20.
3. Raw TRIA v1 input with nontrivial center bytes.
4. Three-way base + DFLASH + TRIA composition.
5. Byte-identical decompose for every component.
6. Digest, length, role, geometry, duplicate-role, reserved-namespace, and
   malformed-manifest rejection tests.
7. v1 backward-compatible decompose tests.
8. Streaming peak-memory test: composition must not materialize a model-sized
   component in RAM.
9. `hipfire inspect` reports component role, source format, original arch,
   encoding, size, and digest without reading tensor payloads.

Run `./tests/no-gpu-ci.sh`. Runtime consumption is deliberately outside this
scope and is covered by the embedded-runtime document.

## Definition of done

The CLI composes Qwen/Gemma base HFQM + arch-20 DFLASH HFQM + raw TRIA without
name collisions, and decompose reproduces all three inputs byte-for-byte. No
runtime loadability claim is made until the embedded component reader lands.
