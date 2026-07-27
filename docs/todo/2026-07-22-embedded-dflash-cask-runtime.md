# Embedded DFLASH and CASK runtime loading

Status: standalone/embedded readers, heterogeneous runtime, and CASK eval implemented; first two product recall gates rejected

Owner: model/runtime/serving

Depends on: [HFQ compose/decompose for DFLASH and TRIA](2026-07-22-hfq-compose-dflash-tria.md)

## Problem

Composition alone does not make a feature usable. The current serving path
still receives a DFLASH draft path and a CASK/TriAttention sidecar path. DFLASH
loads through a standalone `HfqFile`; `TriAttnCenters::load` opens a filesystem
path and parses raw `TRIA`. A bundle that merely carries those bytes would be
advertised as complete while the runtime ignores them.

## Scope

Add read-only role-component views and make existing DFLASH/CASK consumers load
either a standalone artifact or an embedded component through the same parser.
Do not extract embedded components to temporary files.

Primary seams:

- `crates/hipfire-model/src/lib.rs`
- `crates/hipfire-runtime/src/hfq.rs`
- `crates/hipfire-runtime/src/hfq_compose.rs`
- `crates/hipfire-runtime/src/dflash.rs`
- `crates/hipfire-runtime/src/triattn.rs`
- `crates/hipfire-serving-core/src/load.rs`
- daemon/CLI shared load-parameter construction

## Implementation evidence: 2026-07-22

The current worktree provides bounded component views over serving `HfqFile`
objects, per-component digest/config verification, DFLASH tensor access through
a shared standalone-or-embedded source trait, strict TRIA parsing from bytes,
and common source resolution with the required precedence:

```text
off > explicit > embedded > sibling
```

Explicit invalid paths remain selected and fail rather than falling back.
Embedded DFLASH and TRIA are verified before GPU weight upload, and server-side
implicit sibling injection was removed so the shared resolver owns discovery.
Unit and no-GPU gates cover resolver precedence and parser/manifest failures.

Real-model validation now also covers the known-good Qwen3.5-9B MQ4 + DFLASH
pair. A 5.9 GiB compose.v2 bundle selected and digest-verified its embedded
arch-20 component, loaded through the daemon with no draft path, and generated
the exact same 16-token stream and speculative counters as the loose explicit
draft (`proposed=48`, `accepted=12`, `committed=18`, `tau=4.0`) under Q8 KV.
This exposed and fixed an invalid target/draft head-geometry equality check:
DFLASH is an independent transformer and only target-bound interface geometry
may be compared. The daemon's FP32-KV speculative route still lacks
`KvWriteF32`; Q8 KV is the admitted path used by the coherence diagnostic.

The same real model now also covers the combined parser/lifecycle seam. The
Qwen3.5-9B production tap generated a 64-token, 8-full-attention-layer TRIA v1
sidecar (16 query heads, head dimension 256). Compose v2 accepted the MQ4 base,
arch-20 `oq4+` DFLASH, and raw TRIA input, recording independent SHA-256
digests. The resulting 5.4 GiB
`Qwen3.5-9B.dflash.triattn.mq4.hfq` loaded through a freshly built daemon with
neither loose path supplied. The daemon selected and digest-verified both
embedded roles, built the TriAttention eviction context (`budget=512`,
`beta=128`, `physical_cap=896`), uploaded the packed DFLASH component, and
generated a finite `OK` response. No component extraction was used.

The runtime also accepts canonical heterogeneous TriAttention HFQM components.
Registered backends receive the decoded per-layer package without forcing it
through the legacy uniform view; Gemma 4 validates and executes owned full-
context F32 layers while retaining local layers in bounded rings. A synthetic
gfx1103 smoke passed two compaction cycles with different full-layer geometries
and both RoPE conventions. Legacy TRIA remains accepted only by uniform
consumers; a heterogeneous registered backend fails it explicitly.

Cohere2/BLS now uses the same layered arena and eviction context rather than a
family-specific sidecar reader. Its uniform head geometry still carries
per-layer full/sliding and interleaved/unrotated policy: sliding layers stage
and update bounded rings, full layers use the CASK-capped cache, and only full
groups are compacted. The registered factory validates arch 25 and the complete
physical-layer roster before allocating the eviction context. A live two-layer
arch-25 OQ4 fixture loaded with `max_seq=64`, `physical_cap=16`, and the same
HFQM CASK through both supported forms: an explicit loose sidecar and an
embedded compose.v2 role. Both generated four deterministic tokens and the
same `length` completion event, then unloaded cleanly. The synthetic tokenizer
decoded those token IDs to empty text, so this is lifecycle/token-count parity,
not human-readable output or long-context admission evidence.

The first product-format embedded-CASK candidate is Qwen3.5-2B. Its compose.v2
bundle contains a 506-entry `oq4.25++` base and a six-entry arch-5 HFQM CASK;
the daemon selected `TriAttention component source: embedded`, configured CASK
eviction, loaded all 24 layers in 4.03 seconds, generated eight finite tokens,
and unloaded cleanly. A separate formal smoke battery passed metadata load,
64-token finite greedy decode (19.7 tok/s), and multi-turn reset recall with
matching outputs.

The formal CASK battery now drives the daemon with explicit CASK budget/beta
controls and the committed 12.6k-token multidocument needle fixture. A row can
pass only if it recovers the exact needle and the observed prefill exceeds the
derived physical KV cap. With the current release daemon, the product target
plus explicit loose HFQM CASK and the compose.v2 embedded form both processed
8,159 tokenizer tokens through `budget=512`, `beta=128`, `physical_cap=896`.
They produced identical 20-token, 92-byte output hash
`fnv64:c399f73c03a184b6`, proving loose/embedded storage parity. Both failed to
recover `twenty-one`, so the product candidate is rejected for long-context
recall rather than promoted. An earlier run accidentally selected the stale
installed daemon under `~/.hipfire/bin`; it is not admission evidence. Current
worktree measurements pin `HIPFIRE_DAEMON_BIN=target/release/hipfire-daemon`.

Qwen3.5-4B independently confirms the same runtime result at the next product
size. Its compose.v2 bundle contains a 674-entry `oq4.25++` base and an
eight-entry arch-5 HFQM CASK. The current daemon selected the embedded role and
passed all three formal smoke rows, including finite 64-token decode at 8.8
tok/s and multi-turn reset recall. The frozen loose and embedded CASK runs each
prefilled 8,159 tokens through the same 896-token physical cap, generated 128
tokens and 537 text bytes, and matched exactly at output hash
`fnv64:58f77d8c1455931e`. Both failed committed-needle recovery. This proves
storage-form parity for a second product row but rejects its long-context
recall; the bundle remains local and was not transferred.

This is not yet a combined-feature admission result. Loose-versus-embedded
long-context execution parity now passes for Qwen3.5-2B and Qwen3.5-4B, but
both recall gates are rejected. Full Gemma 4 center/recall evidence, combined
DFLASH+CASK recall, and concurrent unload/reload tests remain open.

## Component view contract

Introduce a bounded, borrowed component view over the bundle mmap. It exposes:

- role and source format;
- original arch ID and metadata;
- original tensor names mapped to stored names;
- blob lookup returning borrowed byte ranges; and
- component fingerprint/length validation.

Refactor DFLASH parsing/loading to depend on this narrow blob-source contract,
not a concrete top-level `HfqFile`. Standalone DFLASH creates a view with an
identity name map; embedded DFLASH creates a namespaced view.

Add `TriAttnCenters::from_reader`/`from_bytes` with strict version, geometry,
length, finite-value, and trailing-byte checks. A standalone path and an
embedded opaque payload must call the same parser.

## Resolution precedence

Use one shared resolver for CLI, daemon, server, and eval:

1. explicit user-supplied DFLASH/CASK path;
2. embedded component;
3. canonical loose sibling discovered by the model registry;
4. absent/off.

An explicit `off` remains off. If a higher-precedence source exists but is
invalid or incompatible, fail the load; do not silently fall through to a
different component. Log the selected source, role, encoding, and fingerprint.

## Compatibility checks

DFLASH validation binds:

- target architecture/family;
- target-hidden and vocabulary sizes;
- target layer count and extraction layer IDs;
- block size;
- tokenizer identity where available; and
- DFLASH encoding/rotation contract.

Draft attention heads, KV heads, and head dimension are internal drafter
geometry and are deliberately not required to equal the target's attention
geometry.

CASK validation binds every eligible attention layer's query-head count,
head dimension, RoPE basis/theta, and layer identity. Uniform TRIA v1 is accepted
only when the target geometry is actually uniform. Canonical HFQM v2 is required
for heterogeneous Gemma 4 geometry.

## Model inventory and protocol

- Extend model cards/`hipfire inspect` to distinguish `embedded`, `explicit`,
  and `sibling` sources.
- Treat manifest roles as authoritative only after digest validation.
- Keep existing optional protocol fields for explicit override; embedded auto
  selection does not require sending a synthetic path.
- Do not add family-specific fields to the central loaded-model struct. Reuse
  the serving backend and shared component resolver seams.
- Ensure unloading releases borrowed component state before the base mmap.

## Verification

1. Standalone versus embedded DFLASH config and tensor-byte parity.
2. Standalone versus embedded CASK center-bit parity.
3. Explicit > embedded > sibling precedence tests, including explicit `off`.
4. Corrupt embedded component fails closed even when a valid sibling exists.
5. No temporary files or component-sized allocations during embedded load.
6. Model list/inspect surfaces both roles and their fingerprints.
7. Same-engine separate-versus-bundled logits and decoded output are identical
   for storage formats whose arithmetic path is unchanged.
8. Combined bundled DFLASH+CASK multi-turn and long-context recall.
9. Daemon unload/reload and concurrent-session lifetime tests.

Run `./tests/no-gpu-ci.sh` and
`./tests/tiny-affected-gate.sh --require-coverage`. Use the manual DFLASH
coherence gate as a diagnostic, not as a substitute for `hipfire-eval` evidence.

## Non-goals

- This scope does not implement packed DFLASH OQ kernels.
- It does not claim that a family supports DFLASH verification merely because a
  component can be parsed.
- It does not auto-promote a bundle based on successful loading.

## Definition of done

A bundle containing DFLASH and CASK loads without either loose artifact, uses
the same parsers as standalone inputs, reports deterministic source selection,
and matches the separate-artifact path under the appropriate correctness gates.
