# Architecture Identity

How hipfire says *which architecture* an artifact is.

Identity is a triple — **family**, **variant**, **role** — declared in the arch
registry (`crates/hipfire-arch-api`) and, from HFQ v3, in the container's
metadata. The numeric `arch_id` is the legacy form: still written to the header
and still authoritative for artifacts written before v3, but no longer the thing
new code should branch on.

Plan and rationale: `docs/plans/2026-07-26-arch-identity-structure-descriptor.md`.

## The triple

```
family[/variant][+role]     e.g.  qwen3.5     nemotron-h/moe     gemma4/dense+vl
```

| leg | question | type |
|---|---|---|
| **family** | which arch crate serves it | `Arch::family()` — the registry key |
| **variant** | which in-family shape, when one family loads two ways | `Arch::variants()` — opaque label, or none |
| **role** | which sidecar tower/head rides with it | `Role` — a closed enum |

`ArchRegistry::resolve` maps a tag (a canonical HF `model_type`, or a family
name) onto the registered arch, separator- and case-insensitively, so
`nemotron-h`, `nemotron_h` and `Nemotron-H` all land on the same family.

## Families

Every `register_arch!`-registered family. Serving support lives in
`docs/model-support.toml`, not here — this table is identity only.

| family | legacy id(s) | serving crate |
|---|---|---|
| `llama` | 0, 1 | `hipfire-arch-llama` |
| `qwen2` | 7 | `hipfire-arch-qwen2` |
| `qwen3.5` | 5 | `hipfire-arch-qwen35` |
| `qwen3.5-moe` | 6 | `hipfire-arch-qwen35` |
| `dots-ocr` | 8 | `hipfire-arch-dots-ocr` |
| `deepseek4` | 9 | `hipfire-arch-deepseek4` |
| `minimax` | 10 | `hipfire-arch-minimax` |
| `lfm2` | 11 | `hipfire-arch-lfm2moe` |
| `gemma3` | 12 | `hipfire-arch-gemma3` |
| `gemma3-vl` | 13 | `hipfire-arch-gemma3-vl` |
| `nemotron-h` | 14 | `hipfire-arch-nemotron` |
| `mamba2` | 15 | `hipfire-arch-nemotron` |
| `zaya` | 16 | `hipfire-arch-zaya` |
| `embeddinggemma` | 19 | `hipfire-arch-embeddinggemma` |
| `gemma4` | 24 | `hipfire-arch-gemma4` |
| `cohere2-moe` | 25 | not yet served |
| `krea2` | 17 | `hipfire-diffusion` |
| `qwen-image` | 18 | `hipfire-diffusion` |
| `flux2` | 23 | `hipfire-diffusion` |

## Variants

A variant exists **only** where two artifacts of one family need different
loading or a different forward pass. It is not a place to record trivia. The
label is opaque: the registry knows two identities differ; only the family's own
loader knows what the difference means.

| identity | what differs | where the loader already branches |
|---|---|---|
| `nemotron-h/dense` | dense FFN blocks | `has_moe`, `hipfire-arch-nemotron/src/lib.rs` |
| `nemotron-h/moe` | routed-MoE blocks (`E` in `hybrid_override_pattern`) | same |
| `gemma4/dense` | dense FFN | `FfnPlan`, `hipfire-arch-gemma4/src/config.rs` |
| `gemma4/moe` | dense + routed MoE | `FfnPlan::DensePlusMoe` |

Every other family currently declares no variant — one shape, one way to load.

A survey of 106 checkpoints found that 11 of 19 families genuinely need none,
and that no family needs more than three. Regenerate the evidence with:

```sh
python3 scripts/arch_structure_survey.py --with-manifest
```

Families the survey shows *will* need variants once their extractors land (A3):
`llama` (rope scheme), `qwen3.5` / `qwen3.5-moe` (rope linear vs partial),
`gemma3` (sliding vs periodic-sliding). They stay undeclared until the extractor
can actually distinguish them — an undeclared variant is honest, a wrong one is
not.

## Roles

The closed sidecar vocabulary — `hipfire_arch_api::Role`. These change what the
artifact *is*, and match the dot-groups in the canonical artifact name
(`AGENTS.md`), so a filename and a declared identity cannot disagree.

| role | artifact tag | meaning |
|---|---|---|
| `vl` | `.vl.hfq` | vision tower spliced into the decoder input |
| `audio` | `.audio.hfq` | audio tower |
| `mtp` | `.mtp.hfq` | multi-token-prediction head |
| `dflash` | `.dflash.hfq` | DFlash / DDTree speculative-decode draft head |
| `triattn` | `.triattn.hfq` | tri-attention sidecar |

Roles are orthogonal to variants: any role may ride any base.

**Not roles:** `calib`, `hessian` and `jinja` are data sidecars — they ride
beside a model without changing its architecture, so they are not identity. They
remain in `KNOWN_ROLES` (`hipfire-hfq-tooling`), which is the broader
artifact-*name* vocabulary, and are deliberately absent from `Role`.

## Legacy `arch_id`

The numeric id is a `u32` at bytes `[8..12]` of the HFQM header. It is retained
as the compatibility contract for every artifact already on disk, and is
**append-only**: ids are never renumbered and retired ids are never reused.

Known holes: **2, 3, 4** (historically retired), **20, 21** (tooling-only, see
below). `ARCH_ID_DIFFUSION_LEGACY = 0x3046_4944` (ASCII `DIF0`) is a legacy
magic value; the `Diffusion` capability replaced it for routing.

Constants live once, in `crates/hipfire-arch-api/src/lib.rs`, and are re-exported
by `hipfire-model` so existing `hipfire_model::ARCH_ID_*` callers keep working.
`hipfire-arch-api` is a leaf, CPU-only crate, so the quantizer can reference ids
without pulling in runtime deps.

**Rule:** reference the named constant, never a bare `arch_id == N` literal or a
raw match arm (`"gemma3" => 12`).

### Detection

| id | constant | detection `arch_str` |
|----|----------|----------------------|
| 0  | `ARCH_ID_LLAMA_MISTRAL`      | `llama`, `mistral` (+ unknown-arch default) |
| 1  | `ARCH_ID_QWEN3_QWEN2_LEGACY` | `qwen3`, `qwen2` |
| 5  | `ARCH_ID_QWEN35_DENSE`       | `qwen3_5`, `qwen3_5_text` |
| 6  | `ARCH_ID_QWEN35_MOE`         | `qwen3_5_moe`, `qwen3_5_moe_text`, GGUF `qwen3moe` |
| 7  | `ARCH_ID_QWEN2`              | via `--arch-id 7` (auto-detect maps plain Qwen2 to 1) |
| 8  | `ARCH_ID_DOTS_OCR`           | `dots_ocr` |
| 9  | `ARCH_ID_DEEPSEEK4_FLASH`    | `deepseek_v4` |
| 10 | `ARCH_ID_MINIMAX_M2`         | `minimax_m2` |
| 11 | `ARCH_ID_LFM2_MOE`           | `lfm2_moe`, `lfm2` |
| 12 | `ARCH_ID_GEMMA3_TEXT`        | `gemma3_text`, `gemma3` |
| 13 | `ARCH_ID_GEMMA3_VL`          | `gemma3` + `vision_config` (auto-promoted from 12) |
| 14 | `ARCH_ID_NEMOTRON_H`         | `nemotron_h` |
| 15 | `ARCH_ID_MAMBA2`             | `mamba2` (+ `is_mamba2_config`) |
| 16 | `ARCH_ID_ZAYA`               | `zaya` |
| 19 | `ARCH_ID_EMBEDDINGGEMMA`     | `gemma3_text`/`gemma3` bidirectional encoder w/ ST pooling+Dense modules |
| 23 | `ARCH_ID_FLUX2`              | `Flux2Transformer2DModel` / `Flux2KleinPipeline` / `SEFIInferencePipeline` |
| 24 | `ARCH_ID_GEMMA4`             | `gemma4`, `gemma4_text`, `gemma4_unified`, `gemma4_unified_text` |
| 25 | `ARCH_ID_COHERE2_MOE`        | `cohere2_moe` (BLS Mini Code; offline identity only) |

The per-arch capability matrix (prefill / dflash / mtp / kv / vision support)
lives in `docs/model-support.toml`, keyed by these same ids.

### Tooling-only ids

Defined locally in the quantizer binaries, not in the capability layer, because
they never reach runtime dispatch:

| id | constant | where | purpose |
|----|----------|-------|---------|
| 20 | `ARCH_ID_DFLASH_DRAFT`     | `hipfire-quantize` `bin/dflash_convert.rs` | DFlash draft-head sidecar |
| 21 | `ARCH_ID_QWEN35_MTP_HEAD`  | `hipfire-quantize` `bin/mtp_extract.rs`    | Qwen3.5 MTP-head sidecar |

## Adding an architecture

The previous version of this file listed four steps. That was wrong by roughly
an order of magnitude: landing `gemma4` touched **133 files, 70 of them outside
the new crate**. The steps below are the identity-layer requirements only —
necessary, nowhere near sufficient, and shrinking as the A-phases land.

1. Add `pub const ARCH_ID_<FAMILY>: u32 = <next free id>;` to
   `crates/hipfire-arch-api/src/lib.rs` (append only — never reuse a hole) and
   to the `pub use` re-export in `crates/hipfire-model/src/lib.rs`.
2. Create a lean `hipfire-arch-<family>-spec` crate: implement `Arch`
   (`id`, `family`, `model_types`, and `variants` if the family loads two ways),
   plus `Ingest`. Register it with `register_arch!`.
3. Add the crate to `hipfire-arch-specs` **and** a `use … as _;` line in its
   `lib.rs` — a Cargo dependency alone is pruned by the linker and the
   registration vanishes with it.
4. Add the family, and any variants, to the tables above. A test in
   `hipfire-arch-specs` fails if you skip this.
5. Add the detection `arch_str` arm(s) in `hipfire-quantize/src/main.rs` and, if
   GGUF-importable, `gguf_import.rs`, using the constant — not a literal.
6. Add an `[[arch]]` entry to `docs/model-support.toml` and regenerate
   (`cargo run -p hipfire-cli -- gen-model-support`).

Serving is a separate, larger job: the `ServingBackend` / `SimpleAr` seam, the
loader arm, and the quantizer's per-tensor policy.
