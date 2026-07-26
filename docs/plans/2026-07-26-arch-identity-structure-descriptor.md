# Arch Identity: Replacing `arch_id` With a Structure Descriptor

Date: 2026-07-26

## Goal

Replace the flat numeric `arch_id` with a declared **structure descriptor**, so
that "which architecture is this artifact, and can this binary serve it?" is a
checkable property of the container rather than a lookup through a small-integer
id that carries four orthogonal meanings at once.

## Status

Phase 0 (survey) is **complete** and its result **changed the plan**. On the
strength of it, **Option A is chosen**: identity becomes a canonical family name
plus a declared variant and role, resolved through the arch registry. The
structure-descriptor approach (Option B) is **not** being built; its design is
retained as an appendix because phase 0 was run for it and it remains the
fallback if nominal identity proves insufficient.

Implementation is **not started**. See [Option A phases](#option-a-phases).

## Why `arch_id` Is Being Replaced

`arch_id` is one flat `u32` encoding a *product* of four orthogonal axes, so
every combination burns an id:

| axis | evidence |
|---|---|
| family | `ARCH_ID_LLAMA_MISTRAL` 0, `ARCH_ID_GEMMA3_TEXT` 12 |
| variant | qwen35 **dense=5 / moe=6** |
| modality | gemma3 **text=12 / vl=13** |
| topology | `ARCH_ID_NEMOTRON_H` 14 / `ARCH_ID_MAMBA2` 15 |

Consequences visible in-tree today:

- Permanent holes: 2, 3, 4, 20, 21 are retired and must not be reused.
- One member is not a small integer at all:
  `ARCH_ID_DIFFUSION_LEGACY = 0x3046_4944` (ASCII `DIF0`).
- `0` means llama **and** "unknown-arch default" (`docs/architecture-ids.md`),
  **and** the HFQM non-weight-package sentinel.
- 1073 occurrences across 164 files; 33 files branch on it. `load_model`
  dispatches with an `if`-chain, not a `match`, so there is no exhaustiveness
  check anywhere in the load path.

## Phase 0 Result (measured, 2026-07-26)

### Method

Extract a structure descriptor from every model under `/srv/huggingface`,
canonicalize it, and count distinct structures. Run twice: once deriving the
descriptor from `config.json` alone, once from `config.json` **plus the
safetensors tensor manifest**. Reproducible via
`scripts/arch_structure_survey.py` (diagnostics only; reads, writes nothing).

Corpus: 118 repos → **106 decoder LMs** → **102 with a readable manifest**,
spanning 22 `model_type`s. Deliberately includes families hipfire does not
serve (`nanbeige`, `llama4`, `cohere2_moe`, `hrm_text`, `neo_chat`,
`diffusion_gemma`, `NemotronH_Nano_Omni_Reasoning_V3`).

### Headline: a config-derived descriptor is insufficient

| Descriptor derived from | Distinct structures | Collapse groups |
|---|---|---|
| `config.json` only | 30 | **4** |
| `config.json` + tensor manifest | 34 | **1** |

Three of the four apparent collapses were artifacts of not reading the weights.
The one that survives is `gemma3 == gemma4` (4 repos). Six config-identical
groups split once the manifest was consulted.

The largest — 29 repos that `config.json` cannot tell apart:

| family | distinguishing tensors | repos |
|---|---|---|
| `qwen3` | `q_norm` / `k_norm` | 15 |
| `qwen2` | `q_proj.bias` (QKV bias) | 11 |
| `llama` | neither — plain | 3 |

Qwen2 declares **no** `attention_bias` key; its QKV bias is a property of the
HF *implementation*, not of the config. Qwen3 declares `attention_bias: false`
and carries `q_norm` tensors that the config never mentions.

**This refutes the premise that motivated Option B.** llama / qwen2 / qwen3 are
not one model wearing several ids — they are three distinct structures.
Structure-based identity buys **one** family merge across 102 models, not the
wholesale backend reduction that was assumed.

### What does survive: ambiguity

8 of 19 manifest-verified `model_type`s map to **more than one** real structure:

```
qwen3_5_moe → 4        llama      → 3        gemma3      → 2
qwen3_5     → 4        gemma4     → 3        gemma3_text → 2
qwen3       → 3        nemotron_h → 3
```

So `arch_id` is not merely coarse — it is **ambiguous**. Every loader already
re-derives the real structure from config at load time (hence the per-family
`config.rs` files). A structure descriptor makes that derivation *the identity*
instead of a side effect, and makes serve-ability checkable before load.

### Structural facts confirmed

- **33% of the corpus is hybrid-mixer** (34/102): `gqa+linear` 26 repos
  (qwen3.5), `gqa+short_conv` 4 (lfm2), `gqa+mamba2` 4 (nemotron_h). A
  descriptor modelling a
  layer as `attn+ffn` fails for a third of real models — not a nemotron edge
  case. The stack must be a sequence of **single-role blocks**; `attn+ffn` is
  sugar that canonicalizes to two blocks.
- **Four layer-pattern encodings in the wild**: `uniform` 52, `explicit_list`
  49, `explicit_blocks` 4 (nemotron `hybrid_override_pattern`), `period` 1
  (gemma3 `sliding_window_pattern`). `gemma3_text` appears under two different
  encodings. **Canonicalization is mandatory**, not a nicety: two spellings of
  one structure break exact-match resolution.

### Vocabulary corrections forced by the survey

The proposed vocabulary broke in five places on first contact with real data:

1. **`linear_attention` was missing.** qwen3.5's hybrid mixer — **26 repos** —
   was silently classified as plain `gqa`.
2. **`rope_parameters`** is the current HF spelling of `rope_theta` /
   `rope_scaling`. 35 repos read as `rope=none` before this was handled.
3. **`sliding_window` present ≠ sliding.** 11 Qwen2.5 repos set
   `use_sliding_window: false`; presence of a field is not activation of a
   feature.
4. **`dflash_config` / draft heads** are towers. 9 repos, undetected.
5. **The manifest is authoritative**, per the headline.
6. **Probes must be decoder-scoped** — see below; this one produced a wrong
   answer (zero collapses) before it was caught.

### Tensor probes must be decoder-scoped

The first manifest run reported **zero** collapses. That was wrong: gemma3's 81
`qkv_bias` tensors are **100% vision-tower (siglip), 0% decoder**, and counting
them invented a structural difference that does not exist. Scoping the probes to
decoder tensors (`decoder_keys` in the survey script) restored exactly one real
collapse, `gemma3 == gemma4`.

Generalized: **a structural probe must name the sub-model it describes.** A
descriptor for a multimodal container has to carry per-tower structure
separately, or tower tensors will leak into the decoder's identity. This is a
requirement on the phase 1 schema, not just on the survey.

### Other limitations

- **4 repos could not be manifest-verified** — `minimax_m2`, `diffusion_gemma`,
  `llama4`, `nanbeige` — because their snapshots have no readable index. (The
  MiniMax checkpoint under `/srv` is partially fetched.) Their config-only
  collapses are **unconfirmed, not refuted**; in particular
  `minimax_m2 == qwen3_moe` remains open.
- The survey covers `/srv/huggingface` only. Large and varied, but not the
  universe.
- Quantized `.hfq` artifacts are not surveyed — only upstream HF checkpoints.

## What Phase 0 Changes

| Claim before phase 0 | After |
|---|---|
| llama/mistral/qwen2/qwen3-legacy are one structure | **False.** Three structures, split on weights. |
| Option B collapses backends | **Barely.** One pair (`gemma3`/`gemma4`) across 102 models. |
| A config-derived descriptor suffices | **No.** Must read the tensor manifest. |
| `layers: [{repeat, block: "attn+ffn"}]` | Insufficient for 33% of models. |
| Vocabulary is roughly right | Six corrections in one pass; expect more. |

The cost/benefit that selected Option B over Option A no longer holds as stated.
Option B's remaining advantage is ambiguity-resolution and load-time
checkability, not backend reduction.

## Decision: Option A

**Identity is a canonical family name plus a declared variant and role,
resolved through the arch registry.** Not a structure descriptor.

Rationale, all of it from phase 0:

- The backend-collapse payoff that justified Option B is **one pair** across
  102 models. Option A gives that up and loses almost nothing.
- Option B's descriptor is a model IR. Its vocabulary needed **six corrections
  in a single survey pass**, one of which produced a confidently wrong answer.
  That is an unbounded design surface for a bounded gain.
- Option A fixes what actually hurts: the id holes, the `DIF0` magic value, the
  triple meaning of `0`, the dual `u32`/`String` representation, and — with a
  variant tag — the ambiguity result.
- Option A's container work (HFQ v3, the JSON identity span, the call-site
  sweep) is Option B's phase 2 **regardless**, so this is not a detour. If
  nominal identity later proves insufficient, Option B resumes from here.

**What Option A does not fix:** identity remains *nominal*. A mislabelled
artifact still loads, because nothing compares the declared name against the
actual weights. Option B would have caught that. This is the accepted cost —
mitigated, not solved, by the fixture gate in phase 5.

## Option A: Identity

```jsonc
// HFQ v3 metadata JSON
"identity": {
  "family":  "qwen3_5",   // canonical; resolved via ArchRegistry
  "variant": "partial",   // small in-family discriminator, or null
  "role":    "mtp"        // tower/sidecar, or null
}
```

Three fields, three distinct jobs:

- **`family`** — which arch crate serves it. Resolved through
  `ArchRegistry::resolve` (`model_types` then `family`, separator- and
  case-insensitive), which already exists.
- **`role`** — the tower/sidecar: `vl`, `mtp`, `dflash`, `audio`. This
  vocabulary **already exists** in hipfire, both in `Arch::sidecar_config_keys`
  and in the canonical artifact naming convention (`.mtp.hfq`, `.dflash.hfq`).
- **`variant`** — an opaque label for a structural difference *within* one
  family. Assigned by the extractor, understood only by the registry. It is a
  name, not a description; that is precisely what makes Option A cheap.

### The variant table, derived from phase 0

Of 19 manifest-verified families, **11 need no variant at all**. Factoring the
towers out into `role` — where hipfire's own artifact convention already puts
them — leaves a small, enumerable set:

| family | repos | structures | discriminator after `role` is factored out |
|---|---|---|---|
| `qwen3_5` | 18 | 4 | `rope` linear/partial → **2 variants** (`mtp` is a role) |
| `qwen3_5_moe` | 8 | 4 | `rope` linear/partial → **2 variants** (`mtp` is a role) |
| `qwen3` | 24 | 3 | sliding vs not → **2 variants** (`draft` is a role) |
| `llama` | 9 | 3 | `rope` linear/llama3/none × gqa/mha → **3 variants** |
| `gemma4` | 6 | 3 | dense vs moe → **2 variants** (`audio` is a role) |
| `nemotron_h` | 3 | 3 | dense vs moe → **2 variants** (`mtp` is a role) |
| `gemma3` | 4 | 2 | sliding vs sliding_periodic → **2 variants** |
| `gemma3_text` | 2 | 2 | `rope` dual/linear → **2 variants** |
| 11 others | 27 | 1 each | **none** |

Maximum three variants per family, and most discriminators (`rope` scheme,
dense-vs-moe, sliding) are things the loaders **already branch on**. Regenerate
this table with `scripts/arch_structure_survey.py --with-manifest`.

**Rule:** a new variant is added only when two artifacts of one family need
different loading or a different forward pass. Variants are not a place to
record trivia.

## Option A: Phases

| Phase | Work | Notes |
|---|---|---|
| **0** | Survey and collapse matrix | **DONE** — above |
| **A1** | Freeze the `family`/`variant`/`role` vocabulary; write the variant table into `docs/architecture-ids.md` (which becomes the identity table, not an id table) | small; the enum already exists as data |
| **A2** | HFQ v3: write `identity` into the JSON metadata span; keep writing the legacy `arch_id` as a hint. On read, `version >= 3` → JSON authoritative; `< 3` → frozen legacy `u32` → identity map | low risk; v1→v2 precedent exists. See [Migration](#migration) |
| **A3** | Quantizer emits `identity` for all 20 families | mechanical; the extractor only has to pick a label, not describe a shape |
| **A4** | `ArchId` → `ArchRef { family, variant, role }`; resolve once at load. Sweep the 33 branching files; `load_model`'s four `if`-chains become one `match` over a real enum | the bulk of the work |
| **A5** | Kill the dual representation: `ModelWorkerKey.arch_id: String` becomes the family tag it was always pretending to be; delete `ModelArchFamily` in favour of the registry | partly unblocked already — see below |
| **A6** | One golden fixture per `(family, variant)`; CI gate | the only defence against a mislabelled artifact |

**Already landed toward A5:** `model_arch_family_from_str` now resolves arch
tags through `ArchRegistry::resolve` rather than parsing only numeric ids, and
the server threads the daemon-reported tag into `ModelWorkerKey` instead of
hardcoding `"unknown"` (PR #191). That was a scheduler bug fix, but it is the
first real consumer of name-based identity and the pattern A4/A5 generalize.

## Migration

Common to either option; required by **A2**.

44 `.hfq` artifacts exist (13 local, 31 `/srv`), all regenerable. HFQ has a
version field with a shipped v1→v2 precedent and a JSON metadata span, so
**no binary header change is required**. Header layout, for reference
(`crates/hipfire-model/src/lib.rs`, `read` path):

```
[0..4]   b"HFQM"
[4..8]   version   u32
[8..12]  arch_id   u32     <- becomes a legacy hint
[16..24] metadata_offset u64  -> JSON span, where `identity` lands
[24..32] data_offset     u64
```

Artifacts written before v3 keep their `arch_id` and are mapped through a frozen
`u32 → (family, variant, role)` table at load. That table is append-only and
never renumbered — it is the compatibility contract for every artifact already
on disk.

## Appendix: The Structure Descriptor (Option B, not chosen)

Retained because phase 0 was run to evaluate it, and because it is the fallback
if nominal identity proves insufficient. **Not being built.**

### Two sources of truth, in priority order

1. **Tensor manifest** — authoritative for anything a weight reveals: QKV bias,
   QK-norm, expert layout (stacked vs per-expert), shared experts, SSM state
   (`A_log`, `dt_bias`), `conv1d`, vision/audio towers, MTP heads.
2. **`config.json`** — authoritative only for what weights cannot show:
   counts, window sizes, rope parameters, layer ordering, activation choice.

Where they disagree, the manifest wins. This is a larger extractor than a config
parser, but hipfire already reads manifests — the `Ingest` capability and
`TensorLoadPlan` are the natural home.

### Axes (post-correction)

**Mixer** `kind`: `gqa` · `mha` · `mla` · `linear` · `mamba2` · `short_conv`
— `type`: `generic` · `sliding` · `sliding_periodic`

**FFN** `kind`: `dense` · `moe`
— `type`: `swiglu` · `gelu` · `relu2` · `routed` · `shared+routed`
— `layout` (manifest-derived): `stacked` · `per_expert`

**Rope** `kind`: `linear` · `partial` · `llama3` · `yarn` · `dual` · `none`

**Norm**: `rmsnorm-pre` · `rmsnorm-post` · `rmsnorm-sandwich`

**Towers**: `vision` · `audio` · `mtp` · `draft`

**Weight facts** (manifest only): `qkv_bias` · `qk_norm` · `o_bias` ·
`mlp_bias` · `shared_expert` · `ssm` · `conv1d`

### `kind` / `type` / `version`

| level | question | changing it means |
|---|---|---|
| `kind` | which mathematical operator | **different kernels** |
| `type` | which variant within the kind | same kernels, **different orchestration** |
| `version` | which revision of that variant | same shape, **different semantics** |

`version` is the escape hatch for structurally-identical-but-semantically-
different. It is worthless unless something forces a bump, so:

- The **extractor derives it**, never a human — it is a function of the source
  config/manifest signals.
- **One golden fixture per `(kind, type, version)` triple**, riding the existing
  `caps.toy_model` / `hipfire-quantize/src/fixture.rs` infrastructure. If a
  family's numerics drift without a bump, its fixture diverges and CI fails.
  This is the enforcement and it is not optional.

### Canonical form

Two descriptors are equal iff their canonical forms are byte-identical.
Canonicalization must:

- Expand every **period** into an explicit block list, then run-length compress.
- Split paired layers into **single-role blocks** (`attn+ffn` → two blocks).
- Fold the three `layer_types` value spaces (`full_attention`, `conv`,
  nemotron's `*`/`M`) into the mixer `kind` enum.
- Sort keys; **omit defaults** rather than writing them — absent `bias` and
  `bias: false` must not be two descriptors.

### Resolution

**Exact-or-refuse.** Never nearest-match. Backends declare a set of servable
canonical descriptors; wildcards permitted only on *numerics* (head counts,
expert counts), never on `kind` / `type` / `version`.

### Phases Option B would have needed

Recorded for completeness; **not being built**. B's phase 2 (HFQ v3 + the JSON
identity span) is Option A's A2, so resuming B later would restart at its
phase 1, not from scratch.

| Phase | Work |
|---|---|
| 1 | Freeze enum vocabulary + canonicalization rules + schema doc — where the design risk lives |
| 2 | HFQ v3: canonical `structure` into the JSON metadata span (**= A2**) |
| 3 | Manifest-derived extractor per family, probes decoder-scoped, per-tower structure carried separately |
| 4 | Backends declare servable descriptors; `resolve_by_structure` exact-match |
| 5 | Golden fixture per `(kind,type,version)`; CI gate (**≈ A6**) |

## Why Not Option B

Phase 0 was run to decide whether B's phases are worth it. It says: **not for
the reason Option B was chosen.**

- **Option A** — canonical family name + declared `role`/`variant`. Cheaper, and
  much of it exists: `Arch::model_types()`, `ArchRegistry::find_by_model_type`,
  the duplicate-`model_type` panic at registry build, the `role` sidecar
  vocabulary. Fixes holes, the `DIF0` magic value, the `0`-collision, and the
  dual `u32`/`String` representation. Does **not** fix ambiguity by itself, but
  a variant tag gets most of the way.
- **Option B** — structure descriptor. Fixes ambiguity properly and makes
  serve-ability checkable, but the measured backend-collapse payoff is
  **one pair** (`gemma3`/`gemma4`), and the descriptor is a model IR whose
  vocabulary already needed six corrections.

Option A's container work (HFQ v3, JSON identity span, the call-site sweep) is
Option B's phase 2 regardless, so starting with A is not a detour.

**Resolved: Option A.** The deciding number is that B's distinguishing benefit —
backend collapse — is one pair across 102 models, against a model IR whose
vocabulary needed six corrections in one pass. B's remaining benefit, structural
verification of a declared identity, is real but is bought at the price of the
whole IR; A6's fixture gate covers the same failure mode far more cheaply.

**Reopen B only on evidence**, specifically: a mislabelled artifact reaching
serving and producing wrong output that a `(family, variant)` fixture gate would
not have caught. Absent that, nominal identity is sufficient.

## Reproducing

```sh
python3 scripts/arch_structure_survey.py                 # config-derived
python3 scripts/arch_structure_survey.py --with-manifest # + tensor manifest
```

Reads `/srv/huggingface` (see `AGENTS.local.md`); writes nothing. Diagnostics
only — not production tooling.

## Related

- `docs/architecture-ids.md` — the current id table
- `crates/hipfire-arch-api/src/lib.rs` — `ArchId`, `Caps`, `register_arch!`
- `crates/hipfire-serving-core/src/load.rs` — the four `arch_id` if-chains
