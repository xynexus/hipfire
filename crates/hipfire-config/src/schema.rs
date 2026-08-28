// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Config schema metadata shared by CLI, daemon operator APIs, TUI/WebUI, and
//! generated docs.

use serde::Serialize;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigScope {
    Global,
    Host,
    Node,
    Pool,
    Model,
    Runtime,
    Eval,
    Training,
    Request,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigMutability {
    Static,
    LoadTime,
    RuntimeReloadable,
    RequestOnly,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RestartImpact {
    None,
    ReloadModel,
    RestartDaemon,
    RestartService,
    ReconnectClients,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigType {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I32,
    F64,
    String,
    Path,
    Enum { values: &'static [&'static str] },
    Json,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "condition", rename_all = "snake_case")]
pub enum Requirement {
    Optional,
    Required,
    RequiredWhen(&'static str),
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ConfigField {
    pub key: &'static str,
    #[serde(rename = "type")]
    pub ty: ConfigType,
    pub requirement: Requirement,
    pub default: Option<&'static str>,
    pub scopes: &'static [ConfigScope],
    pub mutability: ConfigMutability,
    pub owner: &'static str,
    pub description: &'static str,
    pub validation: Option<&'static str>,
    pub secret: bool,
    pub restart_impact: RestartImpact,
    pub env: &'static [&'static str],
}

const GLOBAL_RUNTIME: &[ConfigScope] = &[ConfigScope::Global, ConfigScope::Runtime];
const GLOBAL_MODEL_RUNTIME: &[ConfigScope] = &[
    ConfigScope::Global,
    ConfigScope::Model,
    ConfigScope::Runtime,
];
const GLOBAL_MODEL_REQUEST: &[ConfigScope] = &[
    ConfigScope::Global,
    ConfigScope::Model,
    ConfigScope::Request,
];

macro_rules! field {
    (
        $key:literal,
        $ty:expr,
        $requirement:expr,
        $default:expr,
        $scopes:expr,
        $mutability:expr,
        $description:literal
        $(, validation: $validation:literal)?
        $(, env: [$($env:literal),* $(,)?])?
    ) => {
        ConfigField {
            key: $key,
            ty: $ty,
            requirement: $requirement,
            default: $default,
            scopes: $scopes,
            mutability: $mutability,
            owner: "hipfire-config",
            description: $description,
            validation: field!(@validation $($validation)?),
            secret: false,
            restart_impact: RestartImpact::None,
            env: field!(@env $($($env),*)?),
        }
    };
    (@validation $validation:literal) => {
        Some($validation)
    };
    (@validation) => {
        None
    };
    (@env $($env:literal),*) => {
        &[$($env),*]
    };
    (@env) => {
        &[]
    };
}

pub static CONFIG_FIELDS: &[ConfigField] = &[
    field!(
        "host",
        ConfigType::String,
        Requirement::Optional,
        Some("127.0.0.1"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Bind host for the OpenAI-compatible HTTP server. Defaults to loopback; set to 0.0.0.0 to expose on all interfaces.",
        validation: "valid IP address or hostname"
    ),
    field!(
        "port",
        ConfigType::U16,
        Requirement::Optional,
        Some("11435"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Bind port for the OpenAI-compatible HTTP server."
    ),
    field!(
        "cors_allowed_origins",
        ConfigType::Json,
        Requirement::Optional,
        Some("[]"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Browser origins allowed to call the HTTP API cross-origin. Empty disables CORS (same-origin only); [\"*\"] allows any origin; otherwise an explicit allowlist such as [\"http://localhost:8080\"].",
        validation: "JSON array of origin strings, or [\"*\"]"
    ),
    field!(
        "admin_user",
        ConfigType::String,
        Requirement::Optional,
        Some("admin"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Username for the /admin console login. The password is set separately with `hipfire admin set-password` (argon2id hash stored in ~/.hipfire/admin.passwd, never in config)."
    ),
    field!(
        "api_auth_mode",
        ConfigType::Enum {
            values: &["auto", "off", "optional", "required"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "API credential policy. auto allows anonymous API calls only on loopback and requires credentials on non-loopback binds; off, optional, and required are explicit overrides.",
        validation: "one of: auto, off, optional, required"
    ),
    field!(
        "unsafe_allow_unauthenticated_remote",
        ConfigType::Bool,
        Requirement::Optional,
        Some("false"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Explicit acknowledgement required before off or optional API authentication may bind to a non-loopback address."
    ),
    field!(
        "sdapi_output_root",
        ConfigType::String,
        Requirement::Optional,
        Some("/tmp/hipfire-sdapi"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Root directory for images saved by the SD API compatibility routes (save_images: true). Client-supplied outdir_* override_settings are ignored; every SD API image write stays under this root."
    ),
    field!(
        "sdapi_max_dimension",
        ConfigType::U32,
        Requirement::Optional,
        Some("4096"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Upper bound on any single SD API dimension (width/height and their highres/firstphase variants). Requests above it get a 400. The admin's DoS ceiling; clients may request smaller, never larger."
    ),
    field!(
        "sdapi_max_steps",
        ConfigType::U32,
        Requirement::Optional,
        Some("200"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Upper bound on SD API step counts (steps and hr_second_pass_steps)."
    ),
    field!(
        "sdapi_max_batch_size",
        ConfigType::U32,
        Requirement::Optional,
        Some("8"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Upper bound on SD API batch_size."
    ),
    field!(
        "sdapi_max_n_iter",
        ConfigType::U32,
        Requirement::Optional,
        Some("16"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Upper bound on SD API n_iter."
    ),
    field!(
        "sdapi_max_total_batches",
        ConfigType::U32,
        Requirement::Optional,
        Some("32"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Upper bound on batch_size × n_iter (total images generated per request)."
    ),
    field!(
        "models_dir",
        ConfigType::Path,
        Requirement::Optional,
        None,
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Primary local model root. When unset, Hipfire uses ~/.hipfire/models."
    ),
    field!(
        "models_network_dir",
        ConfigType::Path,
        Requirement::Optional,
        None,
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Optional extra read-only model root (e.g. an NFS share such as /srv/hipfire). When set, the network-facing server routes resolve model identifiers within this root in addition to models_dir. Unset by default; local CLI/eval callers are unaffected."
    ),
    field!(
        "default_model",
        ConfigType::String,
        Requirement::Optional,
        None,
        GLOBAL_RUNTIME,
        ConfigMutability::LoadTime,
        "Model tag, alias, or path to use when a request omits the model."
    ),
    field!(
        "prewarm_priority",
        ConfigType::U32,
        Requirement::Optional,
        Some("0"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Startup background prewarm priority for a model. Set per model under model_overrides; 0 disables prewarm, higher values load earlier."
    ),
    field!(
        "max_seq",
        ConfigType::U32,
        Requirement::Optional,
        Some("8192"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Maximum context/KV-cache capacity allocated at model load."
    ),
    field!(
        "max_tokens",
        ConfigType::U32,
        Requirement::Optional,
        Some("512"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Default maximum number of generated tokens per request."
    ),
    field!(
        "temperature",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.3"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Default sampling temperature.",
        validation: "0.0.."
    ),
    field!(
        "top_p",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.8"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Default nucleus sampling probability.",
        validation: "0.0..=1.0"
    ),
    field!(
        "repeat_penalty",
        ConfigType::F64,
        Requirement::Optional,
        Some("1.05"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Default repeat penalty for generated text.",
        validation: "0.0.."
    ),
    field!(
        "resource_lock_enabled",
        ConfigType::Bool,
        Requirement::Optional,
        Some("true"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Whether hipfire serve asks the daemon to acquire physical accelerator resource locks at startup."
    ),
    field!(
        "resource_lock_gpus",
        ConfigType::Json,
        Requirement::Optional,
        Some("[\"auto\"]"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "GPU resources to lease before HIP initialization. [\"auto\"] maps to the daemon's detected/visible HIP device."
    ),
    field!(
        "resource_lock_npus",
        ConfigType::Json,
        Requirement::Optional,
        Some("[]"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "NPU resources to lease before accelerator initialization. [] disables NPU leases; [\"auto\"] leases every detected NPU."
    ),
    field!(
        "resource_lock_wait_ms",
        ConfigType::U32,
        Requirement::Optional,
        Some("0"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "Milliseconds to wait for busy resource leases during daemon startup; 0 fails fast."
    ),
    field!(
        "scheduler_system_memory_budget_bytes",
        ConfigType::U64,
        Requirement::Optional,
        Some("0"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "System-memory budget claimed by the residency scheduler. 0 disables the budget guard."
    ),
    field!(
        "scheduler_system_memory_headroom_bytes",
        ConfigType::U64,
        Requirement::Optional,
        Some("0"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "System-memory headroom preserved by residency admission. 0 disables the headroom guard."
    ),
    field!(
        "scheduler_vram_budget_bytes",
        ConfigType::U64,
        Requirement::Optional,
        Some("0"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "VRAM budget claimed by the residency scheduler. 0 disables the budget guard."
    ),
    field!(
        "scheduler_vram_headroom_bytes",
        ConfigType::U64,
        Requirement::Optional,
        Some("0"),
        GLOBAL_RUNTIME,
        ConfigMutability::Static,
        "VRAM headroom preserved by residency admission. 0 disables the headroom guard."
    ),
    field!(
        "sampler_rng",
        ConfigType::Enum {
            values: &["fixed", "random"]
        },
        Requirement::Optional,
        Some("fixed"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Sampler RNG seeding. 'fixed' reproduces today's behaviour (every temperature>0 request starts from the same constant, so identical prompts give identical output); 'random' seeds each stream from entropy so concurrent requests are independent. Greedy decode never consults the RNG and is unaffected either way."
    ),
    field!(
        "model_residency_mode",
        ConfigType::Enum {
            values: &["auto", "full", "qwen_moe_modules"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Model residency strategy selected by the scheduler."
    ),
    field!(
        "kv_cache",
        ConfigType::Enum {
            values: &["auto", "q8", "asym2", "asym3", "asym4", "kvarn2", "kvarn", "kvarn4", "kvarn8"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "KV-cache precision and memory policy. NOTE: asym2/asym3/asym4 are DEPRECATED — \
         single-tier KVarN strictly dominates them (better PPL+KLD at iso-memory, both \
         short and long ctx; see docs/plans/2026-07-12-hot-cold-hierarchical-kv-implementation.md \
         and NEXT-STEPS Phase D). Prefer kvarn. asym is retained only for back-compat and \
         because TriAttention/CASK eviction scoring reads the asym format."
    ),
    field!(
        "lmhead_twostage",
        ConfigType::String,
        Requirement::Optional,
        Some("\"\""),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Two-stage lm_head decode: a coarse shortlist pass then a full-precision \
         rescore. `q2` (2-bit tier) or `q4`; append `:<K>` to set the shortlist \
         width. Empty disables it and the exact full-vocab GEMV runs. \
         \
         On Qwen3.8-27B it is a NET LOSS, for two reasons worth knowing before \
         enabling it. (1) Under spec decode it never runs at all: the two-stage \
         site is the batch-1 lowered decode forward, and verify scores B tokens \
         through the batched path. (2) Under plain decode it does run and decode \
         gets 2.7% faster (14.7 -> 15.1 tok/s) — exactly the predicted saving, \
         since the coarse tier cuts a 675 MB head to 318 MB and that head is only \
         4.4% of a 15.46 GB model — but the tier is BUILT AT RUNTIME because the \
         artifact carries none, costing ~4.2 s charged to TTFT. Breakeven is \
         ~2350 tokens in one generation; at 160 tokens it is 25% worse \
         end-to-end. Quantize with a coarse tier (or cache the built one) and \
         this becomes a free 2.7%."
    ),
    field!(
        "oq_compact_multicol_wide",
        ConfigType::Bool,
        Requirement::Optional,
        Some("false"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Wide (8-lane, dwordx4) compact multicol decode GEMV. Needs K % 1024 == 0; \
         the narrow kernel is the fallback where that does not hold. Measured \
         24.3 -> 55.5 tok/s on Qwen3.8-27B (gfx1151), which makes it the single \
         largest decode lever on that model — and it defaults OFF while it is \
         proven against the narrow kernel on more shapes. Was reachable only as \
         HIPFIRE_OQ_COMPACT_MULTICOL_WIDE, so the engine's headline throughput \
         depended on knowing an env var."
    ),
    field!(
        "qwen35_paged_experts",
        ConfigType::Bool,
        Requirement::Optional,
        Some("false"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Stream qwen3.5-MoE routed experts from host memory instead of keeping \
         every expert resident. Lets a routed-expert artifact larger than the \
         host's headroom load at all; costs a host-to-GPU fetch on an expert \
         cache miss. Defaults OFF so existing deployments keep full residency.",
        env: ["HIPFIRE_QWEN35_PAGED_EXPERTS"]
    ),
    field!(
        "qwen35_expert_cache_mb",
        ConfigType::U32,
        Requirement::Optional,
        Some("8192"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Resident budget in MiB for the paged routed-expert cache. Only \
         meaningful with qwen35_paged_experts. Larger trades host memory for \
         fewer expert fetches.",
        env: ["HIPFIRE_QWEN35_EXPERT_CACHE_MB"]
    ),
    field!(
        "load_mem_check",
        ConfigType::Bool,
        Requirement::Optional,
        Some("true"),
        GLOBAL_RUNTIME,
        ConfigMutability::LoadTime,
        "Refuse a load whose estimated resident size will not fit in \
         MemAvailable. On a unified-memory host an over-large load does not \
         fail the loader — it invokes the OOM killer on whatever else is \
         running, so turn this off only when the estimate is known to \
         over-count (e.g. paged experts, which it does not model).",
        env: ["HIPFIRE_LOAD_MEM_CHECK"]
    ),
    field!(
        "load_mem_reserve_gib",
        ConfigType::U32,
        Requirement::Optional,
        Some("4"),
        GLOBAL_RUNTIME,
        ConfigMutability::LoadTime,
        "GiB the load check leaves free for the rest of the system. Enough to \
         keep the session's supervisor processes alive so a too-large load \
         fails as a refusal rather than a reaping.",
        env: ["HIPFIRE_LOAD_MEM_RESERVE_GIB"]
    ),
    field!(
        "deltanet_state_precision",
        ConfigType::Enum {
            values: &["fp16", "fp32"]
        },
        Requirement::Optional,
        Some("fp16"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Storage dtype for the Gated DeltaNet recurrent state (the S matrices). \
         Defaults to fp16 as of 2026-08-27. Measured on Qwen3.8-27B--oq4.25++: \
         generation is BYTE-IDENTICAL to fp32 across code/prose/numbers/JSON \
         prompts, and spec-decode acceptance IMPROVES — tau 3.682 -> 4.103 \
         (+11.4%), decode 14.5-16.8 -> 18.9-19.1 tok/s. tau is deterministic and \
         reproduced exactly across repeats, so that is signal, not run noise. \
         Plain AR decode is flat, so the win is specific to speculative decode: \
         the drafter sees the verify path's hidden states and agrees with the \
         fp16 ones more often, while the verifier's committed tokens are \
         unchanged. fp32 remains available as the diffing oracle — losing the \
         ability to compare against it is how quantised state hid for months. \
         NOTE fp16 narrows once per LAUNCH, so a batched call is not identical \
         to the same tokens issued one at a time; fp32 has no narrowing and is \
         identical either way. That asymmetry is real but was MEASURED not to \
         drive the spec-decode/AR divergence (docs/bugs/2026-08-27-spec-decode-\
         ar-divergence.md)."
    ),
    field!(
        "kv_window_precision",
        ConfigType::Enum {
            values: &["auto", "f16", "f32"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Storage dtype for the KVarN recent window (the trailing partial block held \
         unquantised before flush). `auto` picks the narrowest dtype every consumer \
         supports. Each value in the window is read by exactly ONE dot product per \
         decode step before being quantised to 4-bit on flush, so f16 is ~900x tighter \
         than the fate of the data it holds (measured Q.K rel err: f16 2.07e-4, bf16 \
         1.67e-3, the 4-bit it becomes 1.88e-1); f16 beats bf16 because K is bounded, so \
         mantissa outweighs exponent range. `auto` still resolves to f32 while \
         Gpu::kvarn_attend stages the window with a 4-bytes/element dtod blit and gathers \
         tiles from it as f32 — the fallback is announced at load. Worth ~4 MiB and ~0.1% \
         of bandwidth on a 16-KV-layer 27B, more on a full-attention model."
    ),
    field!(
        "kv_adaptive",
        ConfigType::Enum {
            values: &["off", "auto"]
        },
        Requirement::Optional,
        Some("off"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Adaptive KV-cache policy."
    ),
    field!(
        "flash_mode",
        ConfigType::Enum {
            values: &["auto", "always", "never"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Flash-attention selection policy."
    ),
    field!(
        "dflash_mode",
        ConfigType::Enum {
            values: &["off", "auto", "on"]
        },
        Requirement::Optional,
        Some("off"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "DFlash speculative decode mode."
    ),
    field!(
        "dflash_adaptive_b",
        ConfigType::Bool,
        Requirement::Optional,
        Some("true"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Whether DFlash may adapt draft batch size."
    ),
    field!(
        "dflash_ngram_block",
        ConfigType::Json,
        Requirement::Optional,
        Some("\"auto\""),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "DFlash n-gram blocking policy; accepts boolean or auto."
    ),
    field!(
        "mtp_mode",
        ConfigType::Enum {
            values: &["auto", "off", "on"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Multi-token prediction sidecar mode."
    ),
    field!(
        "mtp_k",
        ConfigType::U32,
        Requirement::Optional,
        Some("3"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Number of MTP candidate tokens to consider."
    ),
    field!(
        "thinking",
        ConfigType::Enum {
            values: &["off", "on"]
        },
        Requirement::Optional,
        Some("off"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Reasoning/thinking display policy for compatible models."
    ),
    field!(
        "gpu_slab_load",
        ConfigType::Enum {
            values: &["auto", "off", "on"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "GPU slab loading policy for model weights."
    ),
    field!(
        "prompt_normalize",
        ConfigType::Bool,
        Requirement::Optional,
        Some("true"),
        GLOBAL_MODEL_REQUEST,
        ConfigMutability::RequestOnly,
        "Whether prompts are normalized before tokenization."
    ),
    field!(
        "cask_auto_attach",
        ConfigType::Bool,
        Requirement::Optional,
        Some("true"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Whether compatible CASK/TriAttention sidecars may auto-attach."
    ),
    field!(
        "cask_sidecar",
        ConfigType::Path,
        Requirement::RequiredWhen("cask == true && cask_auto_attach == false"),
        None,
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Explicit CASK/TriAttention sidecar path."
    ),
    field!(
        "cask",
        ConfigType::Bool,
        Requirement::Optional,
        Some("false"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Enable CASK/TriAttention behavior where supported."
    ),
    field!(
        "cask_budget",
        ConfigType::U32,
        Requirement::Optional,
        Some("512"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "CASK token or block budget."
    ),
    field!(
        "cask_beta",
        ConfigType::U32,
        Requirement::Optional,
        Some("128"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "CASK beta control value."
    ),
    field!(
        "cask_core_frac",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.5"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Fraction of CASK core candidates to keep.",
        validation: "0.0..=1.0"
    ),
    field!(
        "cask_fold_m",
        ConfigType::U32,
        Requirement::Optional,
        Some("2"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "CASK fold factor."
    ),
    field!(
        "mmq_screen",
        ConfigType::Enum {
            values: &["auto", "off", "on"]
        },
        Requirement::Optional,
        Some("auto"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "MMQ safety screening mode."
    ),
    field!(
        "mmq_screen_threshold",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.10"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "MMQ screening rejection threshold.",
        validation: "0.0..=1.0"
    ),
    field!(
        "prefill_compression",
        ConfigType::Enum {
            values: &["off", "auto", "on"]
        },
        Requirement::Optional,
        Some("off"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Long-context prefill compression mode."
    ),
    field!(
        "prefill_threshold",
        ConfigType::U32,
        Requirement::Optional,
        Some("32768"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Context length threshold for prefill compression."
    ),
    field!(
        "prefill_keep_ratio",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.05"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Fraction of prefill blocks to keep under compression.",
        validation: "0.0..=1.0"
    ),
    field!(
        "prefill_alpha",
        ConfigType::F64,
        Requirement::Optional,
        Some("0.85"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Prefill compression scoring alpha.",
        validation: "0.0..=1.0"
    ),
    field!(
        "prefill_min_keep",
        ConfigType::U32,
        Requirement::Optional,
        Some("2048"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Minimum tokens or blocks retained during prefill compression."
    ),
    field!(
        "prefill_sink",
        ConfigType::U32,
        Requirement::Optional,
        Some("256"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Prefix sink size retained during prefill compression."
    ),
    field!(
        "prefill_recent",
        ConfigType::U32,
        Requirement::Optional,
        Some("1024"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Recent context size retained during prefill compression."
    ),
    field!(
        "prefill_block",
        ConfigType::U32,
        Requirement::Optional,
        Some("128"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Block size used by prefill compression."
    ),
    field!(
        "prefill_drafter",
        ConfigType::Path,
        Requirement::RequiredWhen("prefill_compression != 'off' && prefill_drafter_device >= 0"),
        None,
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Optional drafter artifact for prefill compression."
    ),
    field!(
        "prefill_drafter_device",
        ConfigType::I32,
        Requirement::Optional,
        Some("-1"),
        &[
            ConfigScope::Global,
            ConfigScope::Host,
            ConfigScope::Node,
            ConfigScope::Model
        ],
        ConfigMutability::LoadTime,
        "Preferred accelerator device for the prefill drafter."
    ),
    field!(
        "prefill_profile",
        ConfigType::Bool,
        Requirement::Optional,
        Some("false"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Emit prefill compression profiling details."
    ),
    field!(
        "prefill_sparse_threshold",
        ConfigType::U32,
        Requirement::Optional,
        Some("32768"),
        GLOBAL_MODEL_RUNTIME,
        ConfigMutability::LoadTime,
        "Context threshold for sparse prefill behavior."
    ),
    ConfigField {
        key: "model_overrides",
        ty: ConfigType::Json,
        requirement: Requirement::Optional,
        default: Some("{}"),
        scopes: &[ConfigScope::Global, ConfigScope::Model],
        mutability: ConfigMutability::LoadTime,
        owner: "hipfire-config",
        description: "Sparse per-model override map layered on top of global config.",
        validation: Some("object keyed by model tag"),
        secret: false,
        restart_impact: RestartImpact::ReloadModel,
        env: &[],
    },
];

pub fn config_schema() -> &'static [ConfigField] {
    CONFIG_FIELDS
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{config_schema, Requirement};

    #[test]
    fn config_schema_keys_are_unique() {
        let fields = config_schema();
        let mut seen = BTreeSet::new();
        for field in fields {
            assert!(seen.insert(field.key), "duplicate config key {}", field.key);
        }
    }

    #[test]
    fn config_schema_has_conditional_required_fields() {
        assert!(config_schema()
            .iter()
            .any(|field| matches!(field.requirement, Requirement::RequiredWhen(_))));
    }
}
