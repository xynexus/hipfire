// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! The single declaration point for every hipfire environment variable.
//!
//! This is the Rust equivalent of a C header that declares `NAME` + a real
//! description and is consumed twice: once by the code that reads the variable,
//! once by `hipfire help env`. [`env_vars!`] expands each entry into both a
//! named constant (used at the read site) and an element of [`ALL`] (walked by
//! the CLI), so the two can never disagree.
//!
//! # Why a central table and not `inventory`
//!
//! The workspace already uses `inventory` for architecture registration, where
//! collecting only what is linked into the binary is exactly right. Env-var
//! documentation wants the opposite: `hipfire help env` must list every
//! variable regardless of which feature-gated arch crates a given build pulled
//! in. A distributed slice would silently shorten the list per build config, so
//! the table is central.
//!
//! # Adding a variable
//!
//! Add a line to [`env_vars!`] with a description a user can act on — what it
//! changes, accepted values, and the default — then read it through the
//! constant. `std::env::var` is denied by `clippy.toml` outside this file; the
//! lint message points here.
//!
//! # Migration status
//!
//! Crates opt into enforcement by adding `[lints] workspace = true` to their
//! `Cargo.toml`, which is only valid once all of that crate's reads go through
//! this table. So the set of opted-in crates is the migration progress, and
//! `rg -L 'workspace = true' crates/*/Cargo.toml` shows what is left.

use std::str::FromStr;

/// Who a variable is for. `hipfire help env` shows [`Tier::User`] by default
/// and requires `--all` for [`Tier::Developer`], which keeps bench/probe knobs
/// out of the user-facing listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Documented, supported, safe for an operator to set.
    User,
    /// Diagnostics, benchmarks, and bring-up switches. Unsupported.
    Developer,
}

/// One declared environment variable.
#[derive(Debug, Clone, Copy)]
pub struct EnvVar {
    pub name: &'static str,
    pub description: &'static str,
    pub tier: Tier,
}

impl EnvVar {
    /// Raw value, or `None` when unset.
    //
    // The one sanctioned `std::env::var` call in the workspace. Anything else
    // adding this `allow` is visible to `rg 'allow\(clippy::disallowed_methods\)'`.
    #[allow(clippy::disallowed_methods)]
    pub fn get(&self) -> Option<String> {
        std::env::var(self.name).ok()
    }

    /// Whether the variable is present at all, regardless of value. Use for
    /// presence-only switches where `FOO=0` should still count as set.
    #[allow(clippy::disallowed_methods)]
    pub fn is_set(&self) -> bool {
        std::env::var_os(self.name).is_some()
    }

    /// Common on/off spelling: `1` / `true` / `on` / `yes`, case-insensitive.
    /// Anything else — including unset — is false.
    pub fn flag(&self) -> bool {
        self.get().is_some_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "on" | "yes"
            )
        })
    }

    /// Inverse of [`Self::flag`]'s spelling: `0` / `false` / `off` / `no`.
    /// Distinct from `!flag()` because unset must mean "not explicitly off".
    pub fn is_off(&self) -> bool {
        self.get().is_some_and(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            )
        })
    }

    /// Parse the value, falling back to `default` when unset or unparseable.
    /// Centralises the `.and_then(|v| v.parse().ok()).unwrap_or(..)` chain that
    /// was previously copy-pasted at every read site.
    pub fn parse_or<T: FromStr>(&self, default: T) -> T {
        self.get()
            .and_then(|v| v.trim().parse::<T>().ok())
            .unwrap_or(default)
    }

    /// Parse the value, or `None` when unset or unparseable.
    pub fn parse<T: FromStr>(&self) -> Option<T> {
        self.get()?.trim().parse::<T>().ok()
    }
}

/// The user's home directory, or `None` when `HOME` is unset or empty.
///
/// `HOME` is not a hipfire variable, so it is deliberately NOT in [`env_vars!`]
/// — that table is `HIPFIRE_`-prefixed by invariant (see the test below), and a
/// standard POSIX variable has no business in `hipfire help env`. But
/// `clippy.toml` denies `std::env::var` workspace-wide, so a crate cannot opt
/// into enforcement while reading `HOME` directly. This is the sanctioned
/// reader: one more `allow` in the one file that is allowed to have them,
/// rather than an exception scattered across call sites.
#[allow(clippy::disallowed_methods)]
pub fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
}

/// `HF_HUB_CACHE` — HuggingFace's hub cache directory, or `None` when unset or
/// empty.
///
/// Same reasoning as [`home_dir`]: a HuggingFace variable is not a hipfire
/// variable, so it does not belong in [`env_vars!`] (`HIPFIRE_`-prefixed by
/// invariant) or in `hipfire help env`. But `clippy.toml` denies `std::env::var`
/// workspace-wide, so a crate that resolves an HF cache path cannot opt into
/// enforcement without a sanctioned reader. Precedence between this and
/// [`hf_home`] is the caller's policy, not this crate's.
#[allow(clippy::disallowed_methods)]
pub fn hf_hub_cache() -> Option<std::path::PathBuf> {
    std::env::var_os("HF_HUB_CACHE")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
}

/// `HF_HOME` — HuggingFace's home directory, or `None` when unset or empty.
/// The hub cache lives in its `hub/` subdirectory. See [`hf_hub_cache`].
#[allow(clippy::disallowed_methods)]
pub fn hf_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HF_HOME")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
}

/// Declare env vars once; expand into constants **and** the [`ALL`] table.
///
/// The two expansions are the point: a read site that names the constant and a
/// CLI that walks `ALL` cannot drift apart, and a variable cannot be documented
/// without being declared or vice versa.
macro_rules! env_vars {
    ($(
        $(#[$meta:meta])*
        $konst:ident = $name:literal, $tier:ident, $desc:literal;
    )*) => {
        $(
            $(#[$meta])*
            #[doc = concat!("`", $name, "` — ", $desc)]
            pub const $konst: EnvVar = EnvVar {
                name: $name,
                description: $desc,
                tier: Tier::$tier,
            };
        )*

        /// Every declared variable, in declaration order.
        pub const ALL: &[EnvVar] = &[$($konst),*];
    };
}

env_vars! {
    // ── Tokenizer / prompt (hipfire-model) ──────────────────────────────────
    NORMALIZE_PROMPT = "HIPFIRE_NORMALIZE_PROMPT", User,
        "Normalise prompts before tokenizing: CRLF to LF, NBSP to space, strip \
         trailing line whitespace, collapse 3+ blank lines. Set 0/false/off/no \
         to disable. Default on.";

    PROMPT_HEAT_JSON = "HIPFIRE_PROMPT_HEAT_JSON", Developer,
        "Emit the prompt heat-class dump as JSON instead of a table. Set 1. \
         Diagnostic for BPE merge-rank distribution of a prompt.";

    PROMPT_HEAT_LIMIT = "HIPFIRE_PROMPT_HEAT_LIMIT", Developer,
        "Maximum tokens listed in the prompt heat dump. Default 64.";

    // ── DeltaNet state (hipfire-arch-qwen35) ────────────────────────────────
    DN_STATE_FP32_BELOW = "HIPFIRE_DN_STATE_FP32_BELOW", Developer,
        "DeltaNet state stays FP32 when head_dim x n_value_heads is below this \
         threshold. Defaults to usize::MAX, so FP32 always. Quantized (Q8/Q4) \
         DeltaNet state is refused by policy — lowering this errors out rather \
         than degrading quality.";

    // ── lm_head (hipfire-runtime, hipfire-quantize) ─────────────────────────
    LMHEAD_TWOSTAGE = "HIPFIRE_LMHEAD_TWOSTAGE", Developer,
        "Two-stage lm_head decode: coarse Q4 shortlist then bf16 rescore. \
         Presets `q4` or `1`; or `<bits>,<topk>` explicitly. Unset disables it \
         and the exact full-precision gemv runs.";

    NO_COARSE_LMHEAD = "HIPFIRE_NO_COARSE_LMHEAD", Developer,
        "Set to any value to stop the quantizer emitting the `<embed>.coarse.weight` \
         shortlist tier. Equivalent to `--no-coarse-lmhead`. Default is to emit it.";

    // ── Dispatch diagnostics (hipfire-dispatch) ─────────────────────────────
    DUMP_HIDDEN = "HIPFIRE_DUMP_HIDDEN", Developer,
        "Path prefix for hidden-state and router-logit dumps written during \
         decode. Unset disables dumping. Diagnostic only — it forces device \
         synchronisation and is slow.";

    // ── Quantizer: threading and diagnostics (hipfire-quantize) ─────────────
    QUANT_THREADS = "HIPFIRE_QUANT_THREADS", User,
        "Rayon worker threads for the quantizer. `--threads` wins over this; \
         unset uses the default pool size. Lower it when a quantize run has to \
         share the machine with serving traffic.";

    QUANT_DIAG_PATH = "HIPFIRE_QUANT_DIAG_PATH", Developer,
        "`.hfq` artifact the `#[ignore]`d quantizer diagnostic tests open \
         (metadata dump, weight-distribution sample, tensor inventory, block \
         range). Defaults to a local DeepSeek V4 path that will not exist on \
         another machine, so set it when running those tests.";

    // ── Quantizer: per-layer treatment (hipfire-quantize) ───────────────────
    NO_CLIP_LAYERS = "HIPFIRE_NO_CLIP_LAYERS", Developer,
        "Comma-separated tensor-name suffixes to EXCLUDE from clip search, e.g. \
         `down_proj,o_proj`. Unset keeps the uniform behaviour of clipping every \
         tensor. Only meaningful where clipping is lossy — on a bf16 or oq8++ \
         tier it matters, on an oq4++ tier it is close to a no-op.";

    OUTLIERS_BY_LAYER = "HIPFIRE_OUTLIERS_BY_LAYER", Developer,
        "Per-layer outlier budget for `oq<BITS>++`, as comma-separated \
         `<suffix>=<frac>` entries with an optional bare `<frac>` fallback. \
         `OqPlusCompact` stores the count implicitly in the block length \
         (`130 + 2*N_out`), so varying it per tensor needs no format change.";

    OQ8_ROUTER = "HIPFIRE_OQ8_ROUTER", Developer,
        "Set 1 to promote the MoE router (`mlp.gate`, `mlp.shared_expert_gate`) \
         to OQ8 W8A16 under any Opus format. The router is precision-sensitive \
         and tiny, so the bit cost is negligible.";

    NO_EXPERT_AWQ = "HIPFIRE_NO_EXPERT_AWQ", Developer,
        "Set 1 to suppress per-expert AWQ smoothing so routed experts fall back \
         to plain MQ4/MQ8. An A/B knob for measuring the expert-AWQ quality \
         delta; dense tensors are unaffected.";

    AWQ_F1_ONLY = "HIPFIRE_AWQ_F1_ONLY", Developer,
        "Set 1 to restrict AWQ to the F1 whitelist (attention input projections \
         and gate/up only), excluding the F2 additions (o_proj / wo / out_proj / \
         down_proj / w_down). Produces an F1-equivalent quant for A/B comparison \
         against the same binary's F2 output. Unset applies the full F2 list.";

    MINIMAX_DOWN_FORMAT = "HIPFIRE_MINIMAX_DOWN_FORMAT", Developer,
        "Promote ONLY the MiniMax expert `w2` (down) tensors to this format \
         (`mq6`, `mq4`, `mq3-lloyd`), leaving `w1`/`w3` at the base format. The \
         forward dispatches each on its own dtype, so they may differ.";

    GPTQ_DAMPING = "HIPFIRE_GPTQ_DAMPING", Developer,
        "Sequential GPTQ error-feedback damping for the MQ2 Lloyd path. At 0 \
         (the default) the pass is a no-op and output is byte-identical to \
         plain `quantize_mq2g256_lloyd_weighted`.";

    // ── Opus compact residency (hipfire-runtime) ────────────────────────────
    OQ_COMPACT_RESIDENT = "HIPFIRE_OQ_COMPACT_RESIDENT", Developer,
        "Keep OqPlusCompact (qt=36) weights compact in VRAM instead of expanding \
         them to one int8 per weight at load. oq4.25++ is ~4.25 bits/weight on \
         disk but 8 bits resident without this, so the format's VRAM win is lost. \
         Set 1 to opt in while the compact-resident path is validated; the \
         default stays on the expanded path.";

    LLOYD_K3 = "HIPFIRE_LLOYD_K3", Developer,
        "Set 1 to encode MQ2-Lloyd with a ternary 3-level codebook (\"MQ1.58\") \
         instead of 4 levels. Reuses the same kernel and block layout.";

    TIER_RATIO = "HIPFIRE_TIER_RATIO", Developer,
        "Fraction of tensors placed in the high tier by the tiered MQ router. \
         `--tier-ratio` wins over this. Default 0.30.";

    // ── Quantizer: research-format opt-in gates (hipfire-quantize) ──────────
    ALLOW_MQ2 = "HIPFIRE_ALLOW_MQ2", Developer,
        "Set 1 to allow `--format mq2`, equivalent to `--allow-mq2`. Gated \
         because the uniform 4-level codebook collapses at every model size \
         validated locally (0.8B/4B/9B Qwen 3.5 produced mojibake on all four \
         coherence-gate prompts).";

    ALLOW_MQ2_LLOYD = "HIPFIRE_ALLOW_MQ2_LLOYD", Developer,
        "Set 1 to allow the Lloyd-Max MQ2 formats, equivalent to \
         `--allow-mq2-lloyd`. Research-only: better than uniform MQ2 but still \
         text-collapse.";

    ALLOW_MQ3_LLOYD = "HIPFIRE_ALLOW_MQ3_LLOYD", Developer,
        "Set 1 to allow the Lloyd-Max MQ3 formats, equivalent to \
         `--allow-mq3-lloyd`. Research-only: 8-entry codebook + 3-bit indices \
         (112 B/group, +7.7% over uniform MQ3).";

    ALLOW_MQ4_LLOYD = "HIPFIRE_ALLOW_MQ4_LLOYD", Developer,
        "Set 1 to allow the Lloyd-Max MQ4 format, equivalent to \
         `--allow-mq4-lloyd`. Research-only: 16-entry codebook + 4-bit indices \
         (160 B/group, +17.6% over uniform MQ4), quality not yet validated.";

    ALLOW_UNIT_IMATRIX = "HIPFIRE_ALLOW_UNIT_IMATRIX", Developer,
        "Set 1 to let the GPTQ-all formats run WITHOUT `--imatrix`, using unit \
         column weights. Intended for DeepSeek V4, where no imatrix exists yet; \
         quality is strictly worse than a real imatrix.";

    // ── Quantizer: QTIP trellis (hipfire-quantize) ──────────────────────────
    QTIP_BEAM = "HIPFIRE_QTIP_BEAM", Developer,
        "Beam width for the QTIP trellis encoder. Default 128. Lower trades \
         quality for a large encode speedup on big models — the beam search is \
         the offline bottleneck. `--beam` sets this, so the `.hfq`-requantize \
         and direct-source paths agree.";

    QTIP_COND = "HIPFIRE_QTIP_COND", Developer,
        "Hessian conditioning for the QTIP trellis: `weighted`, `greedy`, or \
         `beamldlq`. All three need `--hessian`; without one the tensor falls \
         back to the plain unweighted beam, which is also the unset default.";

    QTIP_CODEBOOK = "HIPFIRE_QTIP_CODEBOOK", Developer,
        "Set `3inst` to use the 3INST computed codebook instead of 1MAD. The \
         encoder codebook, artifact quant type and decode kernel all derive from \
         this one read, so they cannot drift apart — a 3INST block under a 1MAD \
         kernel would dequantize to noise with every structural check passing.";

    QTIP_HESSIAN = "HIPFIRE_QTIP_HESSIAN", Developer,
        "Path to a Hessian sidecar enabling LDLQ error feedback for the QTIP \
         and roughquant simulation paths. Unset disables LDLQ.";

    QTIP_LM_HEAD = "HIPFIRE_QTIP_LM_HEAD", Developer,
        "Set 1 to trellis-quantize the lm_head as well. Off by default because \
         the head is output-sensitive and usually wants a higher tier.";

    QTIP_CPU_ENCODE = "HIPFIRE_QTIP_CPU_ENCODE", Developer,
        "Set 1 to force the CPU beam encoder even when the GPU encoder is \
         available. Conditioning arms otherwise swap the ENCODER and the \
         conditioning at once, so a conditioned arm could lose on the encoder \
         change alone — this holds the encoder fixed across arms.";

    QTIP_BBT_ALPHA = "HIPFIRE_QTIP_BBT_ALPHA", Developer,
        "BBT spectral influence exponent (SpectralLLM): per-channel scaling by \
         spectral activation energy from the calibration Hessian before the \
         trellis, math-invariant. 0.5 is the paper default; unset disables BBT.";

    QTIP_EVAL_ST = "HIPFIRE_QTIP_EVAL_ST", Developer,
        "Path to a `model.safetensors` for the QTIP real-weight quality-gate \
         tests. Unset skips them rather than failing.";

    GPU_CHOLESKY = "HIPFIRE_GPU_CHOLESKY", Developer,
        "Set 1 to run the LDLQ Cholesky factorization on the GPU. Measured \
         quality-neutral but 34x SLOWER than the CPU path end to end: the \
         trailing update is faster on device, but ~4500 block iterations each \
         pay two device syncs and a ~4 MB panel round-trip. Off by default; \
         needs the `gpu` feature.";

    LOWRANK_R = "HIPFIRE_LOWRANK_R", Developer,
        "Rank of a low-rank correction of the quantization error added back into \
         the emitted weight (LQER / CALDERA residual probe). Simulates W4 plus a \
         2-WMMA UV correction. 0 (the default) disables it.";

    // ── Quantizer: roughquant simulation sweeps (hipfire-quantize) ──────────
    RQ_PROTECT_FRAC = "HIPFIRE_RQ_PROTECT_FRAC", Developer,
        "roughquant-sim: fraction of columns kept at full precision, the rest \
         crushed to a uniform grid and baked back to bf16. Default 0.015. \
         Saliency is diag(H) when a Hessian exists, else the column L2 norm.";

    RQ_BULK_BITS = "HIPFIRE_RQ_BULK_BITS", Developer,
        "roughquant-sim: uniform bit-width for the unprotected bulk. Default 2.";

    RQ_GROUP = "HIPFIRE_RQ_GROUP", Developer,
        "roughquant-sim: quantization group size for the bulk. Default 256.";

    RQ2_PROTECT_FRAC = "HIPFIRE_RQ2_PROTECT_FRAC", Developer,
        "roughquant2-sim (PCA rotation): fraction of highest-energy columns kept \
         at full precision. Default 0.015. Tensors without a Hessian, or whose \
         eigensolve fails, stay at the staged bf16.";

    RQ2_BULK_BITS = "HIPFIRE_RQ2_BULK_BITS", Developer,
        "roughquant2-sim: trellis bit-width for the rotated bulk. Default 3.";

    RQ2_DAMP = "HIPFIRE_RQ2_DAMP", Developer,
        "roughquant2-sim: Hessian ridge before the eigensolve, as a fraction of \
         the diagonal mean. Default 0.01.";

    RQ2_SHARE_RESID = "HIPFIRE_RQ2_SHARE_RESID", Developer,
        "roughquant2-sim: set 1 to force one SHARED residual-stream rotation \
         across true residual readers (the foldable ResQ-U_A design) instead of \
         a per-weight rotation. Weights reading internal activations keep their \
         own. Tests whether foldability preserves the win.";

    RQ2_Q8_EMBED = "HIPFIRE_RQ2_Q8_EMBED", Developer,
        "roughquant2-sim: set 1 to simulate Q8 on embed/lm_head instead of \
         leaving them bf16. Without it the mq4 comparison is confounded — mq4 \
         uses Q8 there, worth ~20% of params on a tied-embedding 0.8B.";

    RQ3_PROTECT_FRAC = "HIPFIRE_RQ3_PROTECT_FRAC", Developer,
        "roughquant3-sim (permutation + protection): protected fraction. Default \
         0.03. A permutation folds for free, so this is the foldable analog of \
         roughquant2 minus the channel-mixing decorrelation.";

    RQ3_BULK_BITS = "HIPFIRE_RQ3_BULK_BITS", Developer,
        "roughquant3-sim: trellis bit-width for the bulk. Default 3.";

    RQ3_Q8_EMBED = "HIPFIRE_RQ3_Q8_EMBED", Developer,
        "roughquant3-sim: set 1 to simulate Q8 on embed/lm_head for an honest \
         mq4 comparison. Same purpose as the roughquant2 switch.";

    RQ4_PROTECT_FRAC = "HIPFIRE_RQ4_PROTECT_FRAC", Developer,
        "roughquant4-sim / roughquant-real / roughquant5: fraction of residual \
         channels kept exact, in reader COLUMNS and writer ROWS alike. Defaults \
         0.03 for roughquant4 and roughquant-real, 0.05 for roughquant5.";

    RQ4_BULK_BITS = "HIPFIRE_RQ4_BULK_BITS", Developer,
        "roughquant4-sim: trellis bit-width for the bulk. Default 3.";

    RQ4_BULK = "HIPFIRE_RQ4_BULK", Developer,
        "roughquant4-sim bulk codec: `mq4` for the real mq4 format (with a \
         protected fraction of 0 this is the plain-mq4 baseline), `void` to zero \
         the bulk, anything else for QTIP at the configured bulk bit-width.";

    RQ4_MQ_BITS = "HIPFIRE_RQ4_MQ_BITS", Developer,
        "roughquant4-sim: uniform bit-width for the mq bulk (4 = mq4, 5, 6 = \
         mq6). Default 4. With protect_frac 0 this is a fair FWHT uniform-N-bit \
         anchor on the same machinery.";

    RQ4_OBS_DAMP = "HIPFIRE_RQ4_OBS_DAMP", Developer,
        "roughquant4-sim: Hessian ridge for OBS saliency, as a fraction of the \
         diagonal mean. Default 0.01. Only read when saliency is `obs`.";

    RQ4_PROTECT_Q8 = "HIPFIRE_RQ4_PROTECT_Q8", Developer,
        "roughquant4-sim: set 1 to protect at 8-bit per-channel Q8 rather than \
         bf16, so the reported bit cost is honest.";

    RQ4_SALIENCY = "HIPFIRE_RQ4_SALIENCY", Developer,
        "roughquant4-sim channel-importance metric: `diag` (E[x^2], activation \
         energy, the default), `wnorm` (||W[:,c]||^2), `product` \
         (||W[:,c]||^2 * E[x^2], output-error contribution), `obs`, or `random` \
         for the chance-level control.";

    RQ4_RANDOM_SEED = "HIPFIRE_RQ4_RANDOM_SEED", Developer,
        "Seed for the `random` saliency control, so the chance-level baseline is \
         reproducible. Default 1234567.";

    RQ4_DUMP_RANK = "HIPFIRE_RQ4_DUMP_RANK", Developer,
        "roughquant4-sim: set 1 to print the residual-channel saliency ranking \
         as `RANK<tab>channel<tab>energy` and exit without quantizing. Used to \
         pick ablation-oracle targets sampled across the diag spectrum.";

    RQ4_INVERT = "HIPFIRE_RQ4_INVERT", Developer,
        "roughquant4-sim: set 1 to invert the selection so the VOIDED set is the \
         top-ranked channels instead of the bottom. With a high protect_frac \
         this separates \"the metric finds outliers\" from \"the metric ranks the \
         tail correctly\".";

    RQ4_VOID_ONLY = "HIPFIRE_RQ4_VOID_ONLY", Developer,
        "roughquant4-sim: comma-separated residual channel indices to ablate, \
         overriding protect_frac and saliency selection. Isolates the marginal \
         KLD damage of specific channels — the gold-standard per-channel \
         importance signal to validate diag(H) against.";

    RQ4_Q8_EMBED = "HIPFIRE_RQ4_Q8_EMBED", Developer,
        "roughquant4-sim: set 1 to simulate Q8 on embed/lm_head for an honest \
         mq4 comparison. Same purpose as the roughquant2 switch.";

    MIXED_BPW_RANK = "HIPFIRE_MIXED_BPW_RANK", Developer,
        "Set 1 to dump the `--mixed-bpw` sensitivity ranking and exit before \
         quantizing. Separates a bad RANKING from a bad SEARCH when the \
         allocator loses to hand-picked promotions — the two need different \
         fixes and the dump costs only the sensitivity pass.";

    MIXED_BPW_FULL_H = "HIPFIRE_MIXED_BPW_FULL_H", Developer,
        "Set 1 to rank `--mixed-bpw` candidates by the FULL Hessian output error \
         `tr(dW H dW^T)` instead of its diagonal (imatrix) approximation. Off by \
         default because it was measured not to matter: it reorders 38 of 113 \
         tensors by at most 3 places and does not move o_proj out of the bottom \
         third, for several minutes of extra work. Any tr(dW H dW^T) is \
         layer-local, and o_proj matters through error propagated downstream.";

    MIXED_BPW_GAMMA = "HIPFIRE_MIXED_BPW_GAMMA", Developer,
        "Path to a `calib_gamma` JSON table of per-tensor output-gradient energy. \
         Multiplies the `--mixed-bpw` sensitivity, supplying the K-FAC output \
         factor the H-side objective omits. A tensor absent from the table \
         scores 0 rather than keeping an unscaled score, which would let it \
         dominate. Unset leaves the objective input-covariance only.";

    // ── Quantizer: MiniMax expert precision (hipfire-quantize) ──────────────
    // Presence-only switches: these are read with `is_set`, so `=0` still counts
    // as set. That is the pre-existing `var_os` behaviour, kept deliberately.
    MINIMAX_EXPERT_MQ6 = "HIPFIRE_MINIMAX_EXPERT_MQ6", Developer,
        "Set (to any value) to emit MiniMax routed experts as MQ6 — the oracle \
         check against the MQ4 baseline. Equivalent to `--format mq6` for the \
         experts alone.";

    MINIMAX_EXPERT_MQ2L = "HIPFIRE_MINIMAX_EXPERT_MQ2L", Developer,
        "Set (to any value) to emit MiniMax routed experts as MQ2-Lloyd, the \
         sub-4-bit hipx target. Equivalent to `--format mq2-lloyd` for the \
         experts alone.";

    MINIMAX_EXPERT_MQ3L = "HIPFIRE_MINIMAX_EXPERT_MQ3L", Developer,
        "Set (to any value) to emit MiniMax routed experts as MQ3-Lloyd. \
         Equivalent to `--format mq3-lloyd` for the experts alone.";

    MINIMAX_PROMOTE_MQ4 = "HIPFIRE_MINIMAX_PROMOTE_MQ4", Developer,
        "Comma-separated MiniMax layer ranges (`12-45,50`, inclusive) whose \
         experts are forced UP to MQ4 regardless of the base `--format`. The \
         forward dispatches expert dtype per layer, so a model can carry an \
         MQ2-Lloyd base with MQ4 on the quant-sensitive middle layers.";

    MINIMAX_PROMOTE_MQ6 = "HIPFIRE_MINIMAX_PROMOTE_MQ6", Developer,
        "Comma-separated MiniMax layer ranges forced UP to MQ6. Same form and \
         mechanism as `HIPFIRE_MINIMAX_PROMOTE_MQ4`.";

    // ── Diagnostics fixtures (hipfire-quantize) ─────────────────────────────
    HFHS_REAL = "HIPFIRE_HFHS_REAL", Developer,
        "Path to a real `.hessian.bin` for the `hfhs_diag` opt-in smoke test. \
         Unset skips that test rather than failing it. Renamed from the bare \
         `HFHS_REAL` so it fits the HIPFIRE_ prefix the table requires.";

    // ── Lossless BF16 recodings (hipfire-runtime) ───────────────────────────
    BF16L3_RESIDENT = "HIPFIRE_BF16L3_RESIDENT", Developer,
        "Extend LUT3 residency to EVERY Bf16Lut3 tensor. A LUT3 lm_head is already resident by DEFAULT — it is the only large pure-GEMV consumer, served by `gemv_bf16l3_xf32`, worth tg128 90.05 -> 101.45 with byte-identical output. Set to `0` to opt out entirely, head included. Setting it to anything else also packs layer weights, which is rarely wanted: there is no BF16L3 GEMM, so they are decoded at load anyway. Huffman is never resident (bit-serial). Gather reads decode explicitly — a lookup takes one arbitrary row and BF16L3's escape plane needs a block walk.";

    BF16_WEIGHTS = "HIPFIRE_BF16_WEIGHTS", Developer,
        "Set `f32` to force F16 weights to upcast to F32 at load instead of \
         staying native. Native F16 lets `weight_gemm` take the batched \
         `gemm_f16_x_f32_wmma` path; the upcast falls back to a per-token GEMV. \
         Rollback switch for that change.";

    // ── Quantizer: Opus container layout (hipfire-quantize) ─────────────────
    OQ_RAGGED_Q8 = "HIPFIRE_OQ_RAGGED_Q8", Developer,
        "Set (to any value) to emit Opus tensors whose K is not a multiple of \
         256 as Q8 rather than zero-padding them to a 256 group. The GPU serving \
         loaders assert `K % 256 == 0`, so padded ragged Opus tensors load only \
         on the NPU-native path; this keeps such artifacts GPU-loadable. Default \
         stays padded-Opus.";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_prefixed() {
        let mut seen = std::collections::BTreeSet::new();
        for v in ALL {
            assert!(
                v.name.starts_with("HIPFIRE_"),
                "{} is not HIPFIRE_-prefixed",
                v.name
            );
            assert!(seen.insert(v.name), "duplicate declaration for {}", v.name);
        }
    }

    #[test]
    fn descriptions_say_more_than_the_name() {
        for v in ALL {
            // Guards against the generated-boilerplate problem the scanner had:
            // 704 of 705 entries restated the variable name as a sentence.
            assert!(
                v.description.len() > 40,
                "{} needs a description a user can act on",
                v.name
            );
            assert!(
                !v.description.contains(v.name),
                "{} description just restates the name",
                v.name
            );
        }
    }

    #[test]
    fn flag_and_is_off_spellings() {
        // Exercised through a variable that is certainly unset.
        const ABSENT: EnvVar = EnvVar {
            name: "HIPFIRE_DEFINITELY_NOT_SET_IN_TESTS",
            description: "test-only",
            tier: Tier::Developer,
        };
        assert!(!ABSENT.flag());
        assert!(!ABSENT.is_off(), "unset must not read as explicitly off");
        assert!(!ABSENT.is_set());
        assert_eq!(ABSENT.parse_or(7usize), 7);
        assert_eq!(ABSENT.parse::<usize>(), None);
    }
}
