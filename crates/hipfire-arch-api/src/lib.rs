// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! # hipfire-arch-api — the capability layer
//!
//! Every model architecture is a small `static` singleton implementing the base
//! [`Arch`] trait plus whichever *capability* traits it actually supports
//! ([`BatchedPrefill`], [`SpecDecodeChain`], [`ToyModel`], …). The arch crate then
//! calls [`register_arch!`] to publish itself and declare its capabilities.
//!
//! The daemon, scheduler and quantizer never branch on `arch_id == N`. They build
//! an [`ArchRegistry`] once and ask, per capability, "does this arch support it?":
//!
//! ```ignore
//! match reg.get(id).and_then(|a| a.caps.batched_prefill) {
//!     Some(bp) => schedule_batched(bp, ...),   // supported → dispatch
//!     None     => schedule_one_at_a_time(...), // unsupported → safe fallback
//! }
//! ```
//!
//! so the daemon can never call `arch_batch_prefill` on an arch that does not
//! implement it — the type system makes that unrepresentable.
//!
//! ## Why declarative (not autoref specialization)
//!
//! A spike (2026-07-03) showed autoref-based specialization — deriving the `Caps`
//! automatically from which traits are `impl`'d — is toolchain-fragile (two call
//! forms gave two different wrong resolutions). So capabilities are *listed* in
//! `register_arch!`. The `as &dyn Cap` cast means you **cannot over-claim** (listing
//! a capability you don't `impl` fails to compile); the completeness gate catches
//! **under-claiming** (an `impl` you forgot to list). See
//! `docs/plans/2026-07-03-arch-capability-layer.md`.
//!
//! ## Leaf-crate invariant
//!
//! This crate depends on nothing but `std` + `inventory`. Capability trait method
//! signatures use only plain data, so the offline quantizer can link the trait
//! definitions without pulling in the serving/kernel stack. Serving-heavy method
//! bodies live in the arch crates that `impl` them, not here.

pub mod ingest;
pub use ingest::{
    allocate, default_importance, default_precision_class, default_requires, mmdit_role,
    target_bits, transformer_role, CapReq, CodecCaps, ExpertLayout, Ingest, PrecisionClass,
    TensorRole,
};

pub mod toy;
pub use toy::{Dt, Init, TensorSpec, ToyFixture};

/// Stable numeric identity of an architecture family (the on-disk/header id).
///
/// Named constants for the concrete families live alongside the registry (see the
/// `ARCH_ID_*` block below); the registry keys on this value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArchId(pub u16);

// Canonical numeric arch-family ids (the on-disk/header value). This is the single
// source of truth; `hipfire-model` re-exports these so existing
// `hipfire_model::ARCH_ID_*` callers keep working. Ids are stable and gap-tolerant
// (2..=4 are historically retired). Consumers on the capability layer (quantizer,
// daemon dispatch) reference these instead of bare `arch_id == N` literals.
// See `docs/architecture-ids.md` for the full id table and the add-an-arch checklist.
pub const ARCH_ID_LLAMA_MISTRAL: u32 = 0;
pub const ARCH_ID_QWEN3_QWEN2_LEGACY: u32 = 1;
pub const ARCH_ID_QWEN35_DENSE: u32 = 5;
pub const ARCH_ID_QWEN35_MOE: u32 = 6;
pub const ARCH_ID_QWEN2: u32 = 7;
pub const ARCH_ID_DOTS_OCR: u32 = 8;
pub const ARCH_ID_DEEPSEEK4_FLASH: u32 = 9;
pub const ARCH_ID_MINIMAX_M2: u32 = 10;
pub const ARCH_ID_LFM2_MOE: u32 = 11;
pub const ARCH_ID_GEMMA3_TEXT: u32 = 12;
pub const ARCH_ID_GEMMA3_VL: u32 = 13;
pub const ARCH_ID_NEMOTRON_H: u32 = 14;
pub const ARCH_ID_MAMBA2: u32 = 15;
pub const ARCH_ID_ZAYA: u32 = 16;
// Diffusion denoiser families (image/video MMDiT). First-class arch ids: the
// container header carries these instead of the legacy generic-diffusion marker.
pub const ARCH_ID_KREA2: u32 = 17;
pub const ARCH_ID_QWEN_IMAGE: u32 = 18;
/// Legacy generic-diffusion container marker (ASCII-ish "DIF0"), pre-A2. Still
/// recognized as diffusion for backward compat; never written for new containers.
pub const ARCH_ID_DIFFUSION_LEGACY: u32 = 0x3046_4944;
/// embeddinggemma-300m and siblings: a **bidirectional** Gemma3 encoder (non-causal
/// attention, mean-pooling, Matryoshka dense projection heads). Serves text
/// embeddings, not autoregressive logits — see `docs/architecture-ids.md`.
/// (17/18 are the diffusion denoisers KREA2/QWEN_IMAGE upstream; 19 is next free.)
pub const ARCH_ID_EMBEDDINGGEMMA: u32 = 19;
// Speculative-decode drafter sidecar ids (NOT loadable base architectures — a
// `.hfq` header carries one of these only when the file is a draft sidecar
// discovered next to a base target). DFlash draft = 20 and the Qwen3.5 MTP head
// = 21 already exist as local consts in the quantize bins (`dflash_convert.rs`,
// `mtp_extract.rs`); the DSpark drafter sidecar takes the next free id.
pub const ARCH_ID_DSPARK_DRAFT: u32 = 22;
/// FLUX.2 MMDiT image denoisers, including Klein and the SeFi semantic-first
/// variant. Pipeline metadata distinguishes the vanilla and dual-time drivers.
pub const ARCH_ID_FLUX2: u32 = 23;
/// Gemma 4 text core. Standard, text-only, and unified wrappers share one base
/// identity; modality and MTP artifacts are roles/capabilities, not base ids.
pub const ARCH_ID_GEMMA4: u32 = 24;
/// Cohere2-MoE text models, including CohereLabs BLS Mini Code 1.0.
pub const ARCH_ID_COHERE2_MOE: u32 = 25;

impl core::fmt::Display for ArchId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "arch#{}", self.0)
    }
}

/// A sidecar role: a tower or head that ships beside a base model and changes
/// what the artifact *is*, not merely what it contains.
///
/// This is the `role` leg of the `{family, variant, role}` identity triple (see
/// `docs/architecture-ids.md`). The vocabulary matches the canonical artifact
/// naming convention in `AGENTS.md` — `.vl.hfq`, `.mtp.hfq`, `.dflash.hfq`,
/// `.triattn.hfq` — so an artifact name and its declared identity cannot
/// disagree.
///
/// Data-only sidecars (`.jinja.`, `.hessian`) are deliberately absent: they ride
/// alongside a model without changing its architecture, so they are not identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// Vision tower spliced into the decoder input.
    Vl,
    /// Audio tower.
    Audio,
    /// Multi-token-prediction head.
    Mtp,
    /// DFlash / DDTree speculative-decode draft head.
    Dflash,
    /// Tri-attention sidecar.
    Triattn,
}

impl Role {
    /// Every role, in declaration order. The frozen vocabulary.
    pub const ALL: &'static [Role] = &[
        Role::Vl,
        Role::Audio,
        Role::Mtp,
        Role::Dflash,
        Role::Triattn,
    ];

    /// The canonical tag, as it appears in an artifact name and on disk.
    pub const fn as_str(self) -> &'static str {
        match self {
            Role::Vl => "vl",
            Role::Audio => "audio",
            Role::Mtp => "mtp",
            Role::Dflash => "dflash",
            Role::Triattn => "triattn",
        }
    }

    /// Parse a canonical tag. Returns `None` for anything outside the frozen
    /// vocabulary — callers must treat an unknown role as an error, never as
    /// "no role", or a typo silently becomes a base model.
    pub fn parse(tag: &str) -> Option<Role> {
        Role::ALL.iter().copied().find(|r| r.as_str() == tag)
    }
}

impl core::fmt::Display for Role {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A resolved architecture identity: which family, which in-family variant, and
/// which sidecar role — the replacement for a bare numeric `arch_id`.
///
/// `variant` is an opaque label, not a structural description. It exists only
/// where one family needs different loading or a different forward pass for two
/// artifacts; see the variant table in `docs/architecture-ids.md`. 11 of 19
/// surveyed families need none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArchRef {
    pub family: &'static str,
    pub variant: Option<&'static str>,
    pub role: Option<Role>,
}

impl ArchRef {
    /// A base model of `family` with no variant and no role.
    pub const fn base(family: &'static str) -> Self {
        ArchRef {
            family,
            variant: None,
            role: None,
        }
    }

    pub const fn with_variant(mut self, variant: &'static str) -> Self {
        self.variant = Some(variant);
        self
    }

    pub const fn with_role(mut self, role: Role) -> Self {
        self.role = Some(role);
        self
    }
}

impl core::fmt::Display for ArchRef {
    /// `family[/variant][+role]` — stable enough to log and to key a map on.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.family)?;
        if let Some(v) = self.variant {
            write!(f, "/{v}")?;
        }
        if let Some(r) = self.role {
            write!(f, "+{r}")?;
        }
        Ok(())
    }
}

/// The frozen legacy `arch_id` → family map.
///
/// Every artifact written before identity landed carries only a numeric id, so
/// this is the compatibility contract for everything already on disk. It is
/// **append-only**: entries are never removed, never renumbered, and retired
/// ids are never reused.
///
/// Kept as a plain table rather than derived from the registry on purpose — the
/// registry only contains archs linked into the current binary, and a legacy id
/// must map the same way in every binary, including ones that cannot serve it.
const LEGACY_ARCH_ID_FAMILY: &[(u32, &str)] = &[
    (ARCH_ID_LLAMA_MISTRAL, "llama"),
    (ARCH_ID_QWEN3_QWEN2_LEGACY, "llama"),
    (ARCH_ID_QWEN35_DENSE, "qwen3.5"),
    (ARCH_ID_QWEN35_MOE, "qwen3.5-moe"),
    (ARCH_ID_QWEN2, "qwen2"),
    (ARCH_ID_DOTS_OCR, "dots-ocr"),
    (ARCH_ID_DEEPSEEK4_FLASH, "deepseek4"),
    (ARCH_ID_MINIMAX_M2, "minimax"),
    (ARCH_ID_LFM2_MOE, "lfm2"),
    (ARCH_ID_GEMMA3_TEXT, "gemma3"),
    (ARCH_ID_GEMMA3_VL, "gemma3-vl"),
    (ARCH_ID_NEMOTRON_H, "nemotron-h"),
    (ARCH_ID_MAMBA2, "mamba2"),
    (ARCH_ID_ZAYA, "zaya"),
    (ARCH_ID_KREA2, "krea2"),
    (ARCH_ID_QWEN_IMAGE, "qwen-image"),
    (ARCH_ID_EMBEDDINGGEMMA, "embeddinggemma"),
    (ARCH_ID_FLUX2, "flux2"),
    (ARCH_ID_GEMMA4, "gemma4"),
    (ARCH_ID_COHERE2_MOE, "cohere2-moe"),
];

/// Identity for an artifact that predates the `identity` metadata key.
///
/// Returns the family only. A legacy header cannot express a variant or a role,
/// so both come back `None` — and for a family that *does* declare variants
/// (`nemotron-h`, `gemma4`) `variant: None` means **"unspecified, derive it"**,
/// which is exactly what the loaders already do from config. It does not mean
/// "no variant".
///
/// `None` for an unknown id: an id absent from the frozen table is not a legacy
/// artifact, it is a corrupt or future one, and callers must not guess.
pub fn identity_for_legacy_arch_id(arch_id: u32) -> Option<ArchRef> {
    LEGACY_ARCH_ID_FAMILY
        .iter()
        .find(|(id, _)| *id == arch_id)
        .map(|(_, family)| ArchRef::base(family))
}

/// The base every architecture implements. Identity only — behaviour lives in the
/// capability traits so unsupported behaviour is `None`, not a panic.
pub trait Arch: Sync + 'static {
    /// Stable numeric id (registry key, on-disk header value).
    fn id(&self) -> ArchId;
    /// Human family name for logs/errors, e.g. `"llama"`, `"nemotron-h"`.
    fn family(&self) -> &'static str;

    /// In-family variant labels, or `&[]` when every artifact of this family
    /// loads the same way.
    ///
    /// A variant is added **only** when two artifacts of one family need
    /// different loading or a different forward pass — it is not a place to
    /// record trivia. The labels are opaque: the registry knows they differ,
    /// and only the family's own loader knows what they mean.
    ///
    /// Derived from a survey of 106 models; regenerate the table with
    /// `scripts/arch_structure_survey.py --with-manifest`.
    fn variants(&self) -> &'static [&'static str] {
        &[]
    }

    /// Canonical Hugging Face `model_type` aliases that resolve to this base id.
    /// Wrapper distinctions remain in config; this list only owns offline
    /// identity. Empty by default for families not yet migrated to name-based
    /// detection.
    fn model_types(&self) -> &'static [&'static str] {
        &[]
    }

    /// Config JSON keys that belong to a sidecar [`Role`] rather than the base
    /// model. When a bundle is split (`hipfire model decompose`), these keys
    /// move OUT of the base config and INTO the role sidecar; compose moves
    /// them back. This is what keeps a decomposed base from advertising a
    /// feature whose tensors were carved into a sidecar (e.g. a `vision_config`
    /// with no vision tensors). Arches that carry no sidecar-specific config
    /// return `&[]` (the default).
    fn sidecar_config_keys(&self, _role: Role) -> &'static [&'static str] {
        &[]
    }
}

// ---------------------------------------------------------------------------
// Capability traits. Add one here + a `Caps` field + a `__set_cap!` arm.
// Names describe BEHAVIOUR + AXIS — never a codename (no `*Flash`). See the
// naming table in the plan.
// ---------------------------------------------------------------------------

/// The arch can prefill a whole prompt in one batched forward pass (rather than
/// one token at a time). Presence of the capability is the "yes"; the method
/// carries the batch-shaping knobs migrated in later.
pub trait BatchedPrefill: Sync + 'static {
    /// Largest prompt length accepted in a single batched prefill call.
    fn max_prefill_batch(&self) -> usize;
}

/// The arch can be served through the server-side continuous-batching runner:
/// concurrent same-model requests fused into one `generate_batch_prefill` plus a
/// batched decode-step lifecycle over N co-resident sessions. Distinct from
/// [`BatchedPrefill`] (single-request prompt-in-one-forward). Presence is the
/// "yes" that lets the serving layer route requests through the batch runner
/// rather than the per-request path; the method carries the batch-size ceiling.
pub trait ContinuousBatching: Sync + 'static {
    /// Default upper bound on sessions fused into one batch (a starting value;
    /// `HIPFIRE_SERVER_PREFILL_BATCH_MAX` still overrides at runtime).
    fn max_batch_sessions(&self) -> usize;
}

/// The arch supports a speculative-decode *chain* drafter (linear draft of N
/// tokens verified in one step). Distinct from a tree drafter.
pub trait SpecDecodeChain: Sync + 'static {
    /// Max draft length proposed per verification step.
    fn max_draft_len(&self) -> usize;
}

/// The arch can DESCRIBE a tiny deterministic fixture checkpoint for tests/CI — its
/// config + tensor manifest, co-located with the arch instead of scattered in a
/// quantizer `match arch`. The offline tooling owns byte generation + writing (seeded
/// RNG, safetensors/tokenizer), mirroring the [`Ingest`] declare-vs-do split.
pub trait ToyModel: Sync + 'static {
    /// Describe a minimal self-consistent fixture, seeded deterministically. The
    /// quantizer renders it into a loadable HF model dir (safetensors + config +
    /// shared tokenizer) and then quantizes it on the normal `--input` path.
    fn fixture(&self, seed: u64) -> ToyFixture;

    /// Named fixture variants. Most architectures expose only `default`; a
    /// family may keep multiple structurally distinct tinies local to its spec.
    fn fixture_names(&self) -> &'static [&'static str] {
        &["default"]
    }

    /// Resolve a named variant without growing a quantizer family match arm.
    fn fixture_named(&self, name: &str, seed: u64) -> Option<ToyFixture> {
        (name == "default").then(|| self.fixture(seed))
    }
}

/// The arch is a diffusion (image/video) denoiser rather than a text LM. Presence
/// of this capability is how routing tells diffusion containers apart from language
/// models — a data-driven replacement for the legacy magic `arch_id` constant, so
/// diffusion families can share the small-integer id space with LLMs.
pub trait Diffusion: Sync + 'static {
    /// Stable denoiser family tag, e.g. `"krea2-mmdit"`, `"qwen-image-mmdit"`.
    fn denoiser_family(&self) -> &'static str;
}

/// Per-arch capability table: `Some(&dyn Cap)` iff the arch declared it. Built at
/// registry construction from the arch's `register_arch!` list.
pub struct Caps {
    pub batched_prefill: Option<&'static dyn BatchedPrefill>,
    pub continuous_batching: Option<&'static dyn ContinuousBatching>,
    pub spec_decode_chain: Option<&'static dyn SpecDecodeChain>,
    pub toy_model: Option<&'static dyn ToyModel>,
    pub ingest: Option<&'static dyn Ingest>,
    pub diffusion: Option<&'static dyn Diffusion>,
}

impl Caps {
    /// All capabilities absent — the base every `register_arch!` starts from.
    pub const fn none() -> Self {
        Caps {
            batched_prefill: None,
            continuous_batching: None,
            spec_decode_chain: None,
            toy_model: None,
            ingest: None,
            diffusion: None,
        }
    }

    /// Fill any capability this table is missing from `other` (used when an arch
    /// registers from more than one crate). Panics if both tables declare the same
    /// capability — that's two crates claiming one capability for one arch, a bug.
    /// Keep one line per `Caps` field.
    pub(crate) fn merge_from(&mut self, other: Caps, id: ArchId) {
        macro_rules! merge_field {
            ($field:ident) => {
                match (self.$field.is_some(), other.$field.is_some()) {
                    (false, true) => self.$field = other.$field,
                    (true, true) => panic!(
                        "{id}: capability `{}` registered by two crates — one arch, one owner per capability",
                        stringify!($field)
                    ),
                    _ => {}
                }
            };
        }
        merge_field!(batched_prefill);
        merge_field!(continuous_batching);
        merge_field!(spec_decode_chain);
        merge_field!(toy_model);
        merge_field!(ingest);
        merge_field!(diffusion);
    }
}

/// One link-time registration record. Constructed by [`register_arch!`] and
/// collected by `inventory`; the registry turns each into a [`RegisteredArch`].
pub struct ArchEntry {
    /// The arch singleton, as its base trait object.
    pub base: &'static dyn Arch,
    /// Builds the capability table (runs the `as &dyn Cap` coercions at startup,
    /// so `ArchEntry` itself stays const-constructible for `inventory::submit!`).
    pub make_caps: fn() -> Caps,
}

inventory::collect!(ArchEntry);

/// Re-exported so `register_arch!` can reference `$crate::inventory` regardless of
/// where the macro is invoked.
pub use inventory;

/// A fully-resolved arch: identity + capability table. Yielded by [`ArchRegistry`].
pub struct RegisteredArch {
    pub id: ArchId,
    pub family: &'static str,
    pub model_types: Vec<&'static str>,
    pub base: &'static dyn Arch,
    pub caps: Caps,
}

/// Fold an arch tag to a comparison key: lowercase, separators removed. Lets
/// `nemotron-h`, `nemotron_h` and `Nemotron.H` compare equal without each
/// caller carrying its own spelling table.
fn normalize_arch_tag(tag: &str) -> String {
    tag.chars()
        .filter(|c| !matches!(c, '-' | '_' | '.' | ' '))
        .flat_map(char::to_lowercase)
        .collect()
}

/// The process-wide arch registry, built once from whatever `register_arch!`
/// registrations are linked into this binary.
///
/// A binary that does not link an arch's spec crate will not see it here —
/// which is the correct answer, since it cannot serve that family either.
pub fn registry() -> &'static ArchRegistry {
    static REG: std::sync::OnceLock<ArchRegistry> = std::sync::OnceLock::new();
    REG.get_or_init(ArchRegistry::build)
}

/// Runtime index over every `register_arch!`-published architecture linked into the
/// binary. Build once (e.g. into a `OnceLock`), then look up by [`ArchId`].
pub struct ArchRegistry {
    archs: Vec<RegisteredArch>,
}

impl ArchRegistry {
    /// Collect all link-time registrations. Cheap; call once and cache.
    pub fn build() -> Self {
        let mut archs: Vec<RegisteredArch> = Vec::new();
        for entry in inventory::iter::<ArchEntry> {
            let id = entry.base.id();
            let caps = (entry.make_caps)();
            // An arch may register from more than one crate (its lean offline
            // `-spec` crate declares Ingest; its serving crate declares
            // BatchedPrefill, …). Merge entries sharing an ArchId into one arch,
            // unioning their capability tables. Conflicting claims panic.
            if let Some(existing) = archs.iter_mut().find(|a| a.id == id) {
                existing.caps.merge_from(caps, id);
                for &model_type in entry.base.model_types() {
                    if !existing.model_types.contains(&model_type) {
                        existing.model_types.push(model_type);
                    }
                }
            } else {
                archs.push(RegisteredArch {
                    id,
                    family: entry.base.family(),
                    model_types: entry.base.model_types().to_vec(),
                    base: entry.base,
                    caps,
                });
            }
        }
        for arch in &archs {
            for &model_type in &arch.model_types {
                if let Some(other) = archs
                    .iter()
                    .find(|other| other.id != arch.id && other.model_types.contains(&model_type))
                {
                    panic!(
                        "model_type `{model_type}` registered for both {} and {}",
                        arch.id, other.id
                    );
                }
            }
            // Identity vocabulary invariants. These are cheap and they run at
            // registry build, so a malformed declaration fails at startup
            // rather than becoming an unresolvable artifact later.
            let family = arch.base.family();
            assert!(
                !family.is_empty()
                    && family.bytes().all(|b| b.is_ascii_lowercase()
                        || b.is_ascii_digit()
                        || b"-_.".contains(&b)),
                "{}: family `{family}` must be lowercase [a-z0-9._-]",
                arch.id,
            );
            let variants = arch.base.variants();
            for (i, &v) in variants.iter().enumerate() {
                assert!(
                    !v.is_empty()
                        && v.bytes().all(|b| b.is_ascii_lowercase()
                            || b.is_ascii_digit()
                            || b"-_.".contains(&b)),
                    "{}: variant `{v}` must be lowercase [a-z0-9._-]",
                    arch.id,
                );
                assert!(
                    !variants[..i].contains(&v),
                    "{}: variant `{v}` declared twice",
                    arch.id,
                );
                assert!(
                    Role::parse(v).is_none(),
                    "{}: `{v}` is a role, not a variant — declare towers via Role, \
                     so `family/{v}` and `family+{v}` cannot both mean the same thing",
                    arch.id,
                );
            }
        }
        ArchRegistry { archs }
    }

    /// The registered arch for `id`, if any is linked in.
    pub fn get(&self, id: ArchId) -> Option<&RegisteredArch> {
        self.archs.iter().find(|a| a.id == id)
    }

    /// Resolve a canonical Hugging Face `model_type` through linked arch specs.
    pub fn find_by_model_type(&self, model_type: &str) -> Option<&RegisteredArch> {
        self.archs
            .iter()
            .find(|arch| arch.model_types.contains(&model_type))
    }

    /// Every valid base identity across linked archs: one [`ArchRef`] per
    /// family, plus one per declared variant. Roles are orthogonal and are not
    /// enumerated here — any role may ride any base.
    ///
    /// This is the frozen vocabulary, rendered. `docs/architecture-ids.md`
    /// documents exactly this set, and a test asserts the two agree.
    pub fn identities(&self) -> Vec<ArchRef> {
        let mut out = Vec::new();
        for arch in &self.archs {
            let family = arch.base.family();
            let variants = arch.base.variants();
            if variants.is_empty() {
                out.push(ArchRef::base(family));
            } else {
                out.extend(
                    variants
                        .iter()
                        .map(|&v| ArchRef::base(family).with_variant(v)),
                );
            }
        }
        out.sort();
        out
    }

    /// Resolve an arch *tag* — either a canonical HF `model_type` or a family
    /// name — through linked arch specs.
    ///
    /// Callers hold a string whose provenance they cannot always pin: the
    /// daemon reports a `model_type` for families that declare one, and a
    /// [`Arch::family`] name for those that don't (e.g. `nemotron-h`,
    /// `mamba2`, whose specs leave `model_types` empty). Matching is
    /// separator- and case-insensitive so `nemotron-h`, `nemotron_h` and
    /// `Nemotron-H` all land on the same arch.
    ///
    /// Returns `None` for a tag no linked arch claims — which is also the
    /// honest answer when the binary genuinely cannot serve that family,
    /// because an arch absent from the link is absent from the registry.
    pub fn resolve(&self, tag: &str) -> Option<&RegisteredArch> {
        let needle = normalize_arch_tag(tag);
        if needle.is_empty() {
            return None;
        }
        self.archs
            .iter()
            .find(|arch| {
                arch.model_types
                    .iter()
                    .any(|mt| normalize_arch_tag(mt) == needle)
            })
            .or_else(|| {
                self.archs
                    .iter()
                    .find(|arch| normalize_arch_tag(arch.family) == needle)
            })
    }

    /// True if `id` is a registered diffusion denoiser arch (declares the
    /// [`Diffusion`] capability). Routing uses this instead of a magic id.
    pub fn is_diffusion(&self, id: ArchId) -> bool {
        self.get(id).is_some_and(|a| a.caps.diffusion.is_some())
    }

    /// The denoiser family tag for `id`, if it is a registered diffusion arch.
    pub fn diffusion_family(&self, id: ArchId) -> Option<&'static str> {
        self.get(id)
            .and_then(|a| a.caps.diffusion)
            .map(|d| d.denoiser_family())
    }

    /// Iterate every registered arch (used by the completeness gate).
    pub fn iter(&self) -> impl Iterator<Item = &RegisteredArch> {
        self.archs.iter()
    }

    pub fn len(&self) -> usize {
        self.archs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.archs.is_empty()
    }
}

/// Internal: map a capability ident to its `Caps` field with the `as &dyn Cap`
/// coercion. The coercion is what makes over-claiming a compile error. One arm per
/// capability trait — keep in sync with the `Caps` fields.
#[doc(hidden)]
#[macro_export]
macro_rules! __set_cap {
    ($caps:ident, $inst:expr, BatchedPrefill) => {
        $caps.batched_prefill =
            ::core::option::Option::Some($inst as &'static dyn $crate::BatchedPrefill);
    };
    ($caps:ident, $inst:expr, ContinuousBatching) => {
        $caps.continuous_batching =
            ::core::option::Option::Some($inst as &'static dyn $crate::ContinuousBatching);
    };
    ($caps:ident, $inst:expr, SpecDecodeChain) => {
        $caps.spec_decode_chain =
            ::core::option::Option::Some($inst as &'static dyn $crate::SpecDecodeChain);
    };
    ($caps:ident, $inst:expr, ToyModel) => {
        $caps.toy_model = ::core::option::Option::Some($inst as &'static dyn $crate::ToyModel);
    };
    ($caps:ident, $inst:expr, Ingest) => {
        $caps.ingest = ::core::option::Option::Some($inst as &'static dyn $crate::Ingest);
    };
    ($caps:ident, $inst:expr, Diffusion) => {
        $caps.diffusion = ::core::option::Option::Some($inst as &'static dyn $crate::Diffusion);
    };
    ($caps:ident, $inst:expr, $other:ident) => {
        ::core::compile_error!(::core::concat!(
            "unknown capability `",
            ::core::stringify!($other),
            "` in register_arch! — add a `__set_cap!` arm + a `Caps` field for it"
        ));
    };
}

/// Publish an architecture and declare its capabilities.
///
/// `$inst` is a `static` singleton implementing [`Arch`]; each listed capability
/// must be `impl`'d on it (else the `as &dyn Cap` cast fails to compile — you
/// cannot over-claim). Capabilities you `impl` but forget to list are caught by the
/// completeness gate.
///
/// ```ignore
/// static INSTANCE: LlamaArch = LlamaArch;
/// hipfire_arch_api::register_arch!(INSTANCE, BatchedPrefill, ToyModel);
/// ```
#[macro_export]
macro_rules! register_arch {
    ($inst:path $(, $cap:ident)* $(,)?) => {
        $crate::inventory::submit! {
            $crate::ArchEntry {
                base: &$inst,
                make_caps: {
                    // A named fn (const fn item) keeps `ArchEntry` const-constructible
                    // for `inventory::submit!`; the `as &dyn Cap` casts run when the
                    // registry calls it, not in the static initializer.
                    fn __make_caps() -> $crate::Caps {
                        let mut caps = $crate::Caps::none();
                        $( $crate::__set_cap!(caps, &$inst, $cap); )*
                        caps
                    }
                    __make_caps
                },
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestArch;
    impl Arch for TestArch {
        fn id(&self) -> ArchId {
            ArchId(0x7E57)
        }
        fn family(&self) -> &'static str {
            "test"
        }
    }
    impl ToyModel for TestArch {
        fn fixture(&self, _seed: u64) -> ToyFixture {
            ToyFixture {
                config_json: "{}".to_string(),
                tensors: Vec::new(),
            }
        }
    }
    impl SpecDecodeChain for TestArch {
        fn max_draft_len(&self) -> usize {
            4
        }
    }
    // Deliberately does NOT impl BatchedPrefill.

    static TEST_INSTANCE: TestArch = TestArch;
    register_arch!(TEST_INSTANCE, ToyModel, SpecDecodeChain);

    #[test]
    fn registry_discovers_declared_caps() {
        let reg = ArchRegistry::build();
        let a = reg
            .get(ArchId(0x7E57))
            .expect("TestArch should be link-time registered");
        assert_eq!(a.family, "test");

        // Declared → Some.
        assert!(a.caps.toy_model.is_some(), "ToyModel was listed");
        assert!(
            a.caps.spec_decode_chain.is_some(),
            "SpecDecodeChain was listed"
        );
        // Not declared / not impl'd → None (the daemon's safe-fallback branch).
        assert!(
            a.caps.batched_prefill.is_none(),
            "BatchedPrefill was neither impl'd nor listed"
        );

        // Dispatch actually works through the capability object.
        assert_eq!(a.caps.spec_decode_chain.unwrap().max_draft_len(), 4);
        // ToyModel now DESCRIBES a fixture (config + manifest); the stub returns empty.
        assert!(a.caps.toy_model.unwrap().fixture(7).tensors.is_empty());
    }

    // Two crates registering the SAME arch id with DISJOINT capabilities — the
    // offline `-spec` / serving split. The registry unions them into one arch.
    struct MergeOffline;
    impl Arch for MergeOffline {
        fn id(&self) -> ArchId {
            ArchId(0x7E58)
        }
        fn family(&self) -> &'static str {
            "merge"
        }
    }
    impl ToyModel for MergeOffline {
        fn fixture(&self, _s: u64) -> ToyFixture {
            ToyFixture {
                config_json: "{}".into(),
                tensors: Vec::new(),
            }
        }
    }
    static MERGE_OFFLINE: MergeOffline = MergeOffline;
    register_arch!(MERGE_OFFLINE, ToyModel);

    struct MergeServing;
    impl Arch for MergeServing {
        fn id(&self) -> ArchId {
            ArchId(0x7E58) // same id as MergeOffline
        }
        fn family(&self) -> &'static str {
            "merge"
        }
    }
    impl BatchedPrefill for MergeServing {
        fn max_prefill_batch(&self) -> usize {
            512
        }
    }
    static MERGE_SERVING: MergeServing = MergeServing;
    register_arch!(MERGE_SERVING, BatchedPrefill);

    #[test]
    fn registry_merges_caps_across_crates_for_one_id() {
        let reg = ArchRegistry::build();
        // Entries merged, not duplicated: exactly one arch for the shared id.
        assert_eq!(
            reg.iter().filter(|a| a.id == ArchId(0x7E58)).count(),
            1,
            "the two registrations must collapse to one arch"
        );
        let a = reg.get(ArchId(0x7E58)).unwrap();
        // Union of both registrations' capabilities.
        assert!(a.caps.toy_model.is_some(), "offline cap merged in");
        assert!(a.caps.batched_prefill.is_some(), "serving cap merged in");
        assert_eq!(a.caps.batched_prefill.unwrap().max_prefill_batch(), 512);
    }

    #[test]
    #[should_panic(expected = "registered by two crates")]
    fn merge_conflict_panics() {
        let mut a = Caps {
            toy_model: Some(&TEST_INSTANCE),
            ..Caps::none()
        };
        let b = Caps {
            toy_model: Some(&TEST_INSTANCE),
            ..Caps::none()
        };
        a.merge_from(b, ArchId(0x7E57));
    }
}
