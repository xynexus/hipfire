// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Shared CLI/server configuration and local filesystem paths.

pub mod editor;
pub mod resolve;
pub mod schema;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

pub use editor::{
    apply_config_edit, build_config_editor_snapshot, build_config_editor_snapshot_from_paths,
    cycle_editor_value, encode_editor_value, ConfigEditOperation, ConfigEditRequest,
    ConfigEditTarget, ConfigEditorPaths, ConfigEditorRow, ConfigEditorSnapshot,
};
pub use resolve::{
    config_layer_from_env, config_layer_from_env_with, config_layers_from_document,
    config_layers_from_documents, env_var_name_for_key, resolve_config_layers,
    validate_resolved_value, ConfigLayer, ConfigLayerKind, ConfigResolution, ConfigValueSource,
    ResolvedConfigValue, UnknownConfigKey,
};
pub use schema::{
    config_schema, dflash_draft_setting, ConfigField, ConfigMutability, ConfigScope, ConfigType,
    PathExistence, Requirement, RestartImpact, NGRAM_STORE_ROOT_RAM, RENAMED_KEYS,
};

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    11435
}
fn default_cors_allowed_origins() -> Vec<String> {
    Vec::new()
}
fn default_admin_user() -> String {
    "admin".to_string()
}
fn default_api_auth_mode() -> ApiAuthMode {
    ApiAuthMode::Auto
}
fn default_sdapi_output_root() -> String {
    "/tmp/hipfire-sdapi".to_string()
}
// SD API request-geometry caps. These bound `batch × channels × height ×
// width` allocations on the network-facing routes, so they are the admin's
// DoS ceiling. Defaults are portability-safe for the smallest supported GPU
// class (UMA APUs); clients may request smaller, never larger. `pub` so the
// server can reuse them as the canonical default for its in-memory limits.
pub fn default_sdapi_max_dimension() -> u32 {
    4096
}
pub fn default_sdapi_max_steps() -> u32 {
    200
}
pub fn default_sdapi_max_batch_size() -> u32 {
    8
}
pub fn default_sdapi_max_n_iter() -> u32 {
    16
}
pub fn default_sdapi_max_total_batches() -> u32 {
    32
}
fn default_max_seq() -> u32 {
    8192
}
fn default_max_tokens() -> u32 {
    512
}
fn default_prewarm_priority() -> u32 {
    0
}
fn default_temperature() -> f64 {
    0.3
}
fn default_top_p() -> f64 {
    0.8
}
fn default_repeat_penalty() -> f64 {
    1.05
}
fn default_resource_lock_enabled() -> bool {
    true
}
fn default_resource_lock_gpus() -> Vec<String> {
    vec!["auto".to_string()]
}
fn default_resource_lock_npus() -> Vec<String> {
    Vec::new()
}
fn default_resource_lock_wait_ms() -> u32 {
    0
}
fn default_scheduler_memory_budget_bytes() -> u64 {
    0
}
/// `"fixed"` — today's behaviour, kept as the default deliberately. Every
/// `temperature>0` request has always started from the same constant, so
/// flipping this to `"random"` by default would silently change sampled output
/// for every existing deployment. Opt in instead.
fn default_sampler_rng() -> String {
    "fixed".to_string()
}

fn default_model_residency_mode() -> String {
    "auto".to_string()
}
fn default_kv_cache() -> String {
    "auto".to_string()
}
fn default_kv_adaptive() -> String {
    "off".to_string()
}
fn default_lmhead_twostage() -> String {
    String::new()
}
fn default_oq_compact_multicol_wide() -> bool {
    false
}
/// Off: a routed-expert model keeps every expert resident unless asked
/// otherwise, which is the behaviour every existing deployment already has.
fn default_qwen35_paged_experts() -> bool {
    false
}
/// Off. Prefill mints a Final checkpoint — a deep clone of the whole session
/// state — for every request, and nothing can attach to it: the reuse path is
/// gated on a `runtime_state_handle` no production caller ever sets. Releasing
/// the request session does not free the clone, so each request retained a
/// second session's worth of KV until the host ran out. Boundary checkpoints
/// were already opt-in; this makes Final match them.
fn default_qwen35_final_checkpoints() -> bool {
    false
}
/// Matches the historical `HIPFIRE_QWEN35_EXPERT_CACHE_MB` fallback.
fn default_qwen35_expert_cache_mb() -> u64 {
    8192
}
/// On. The check is what turns an over-large load into a refusal instead of an
/// OOM reaping unrelated processes on a UMA host.
fn default_load_mem_check() -> bool {
    true
}
/// Slack left for the rest of the system after a load. Not a KV estimate: the
/// KV cache is sized after the model config is parsed, well past the check. 4
/// GiB is enough to keep the session's supervisor processes alive so a
/// too-large load fails as a refusal instead of as a reaping.
fn default_load_mem_reserve_gib() -> u32 {
    4
}
fn default_kv_window_precision() -> String {
    "auto".to_string()
}
fn default_deltanet_state_precision() -> String {
    "fp16".to_string()
}
fn default_flash_mode() -> String {
    "auto".to_string()
}
fn default_max_resident_workers() -> u32 {
    2
}
fn default_dflash_draft() -> String {
    "off".to_string()
}
fn default_spec_adaptive_block() -> bool {
    false
}
fn default_spec_block() -> u32 {
    0
}
fn default_ngram_spec() -> bool {
    false
}
fn default_ngram_spec_store_root() -> String {
    String::new()
}
fn default_ngram_spec_scope() -> String {
    String::new()
}
fn default_ngram_spec_store_mb() -> u32 {
    256
}
fn default_ngram_spec_orders() -> String {
    "8,7,6,5,4,3,2".to_string()
}
fn default_ngram_spec_min_acceptance() -> f64 {
    0.0
}
fn default_ngram_spec_min_acceptance_proposals() -> u32 {
    8
}
fn default_ngram_spec_chain_floor() -> u8 {
    8
}
fn default_ngram_spec_max_spine() -> u32 {
    16
}
fn default_ngram_spec_promote_count() -> u16 {
    3
}
fn default_ngram_spec_write_target() -> String {
    "user".to_string()
}
fn default_dflash_no_repeat_ngram() -> serde_json::Value {
    serde_json::Value::String("auto".to_string())
}
fn default_mtp_mode() -> String {
    "auto".to_string()
}
fn default_mtp_k() -> u32 {
    3
}
fn default_thinking() -> String {
    "off".to_string()
}
fn default_gpu_slab_load() -> String {
    "auto".to_string()
}
fn default_jinja_chat() -> String {
    "auto".to_string()
}

fn default_prompt_normalize() -> bool {
    true
}
fn default_cask_auto_attach() -> bool {
    true
}
fn default_cask_budget() -> u32 {
    512
}
fn default_cask_beta() -> u32 {
    128
}
fn default_cask_core_frac() -> f64 {
    0.5
}
fn default_cask_fold_m() -> u32 {
    2
}
fn default_mmq_screen() -> String {
    "auto".to_string()
}
fn default_mmq_screen_threshold() -> f64 {
    0.10
}
fn default_prefill_compression() -> String {
    "off".to_string()
}
fn default_prefill_threshold() -> u32 {
    32768
}
fn default_prefill_keep_ratio() -> f64 {
    0.05
}
fn default_prefill_alpha() -> f64 {
    0.85
}
fn default_prefill_min_keep() -> u32 {
    2048
}
fn default_prefill_sink() -> u32 {
    256
}
fn default_prefill_recent() -> u32 {
    1024
}
fn default_prefill_block() -> u32 {
    128
}
fn default_prefill_drafter_device() -> i32 {
    -1
}
fn default_prefill_sparse_threshold() -> u32 {
    32768
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiAuthMode {
    Auto,
    Off,
    Optional,
    Required,
}

impl std::fmt::Display for ApiAuthMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Optional => "optional",
            Self::Required => "required",
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipfireConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Cross-origin origins allowed to call the HTTP API from a browser.
    /// Empty (default) disables CORS entirely (same-origin only); `["*"]`
    /// allows any origin; otherwise an explicit allowlist of origins.
    #[serde(default = "default_cors_allowed_origins")]
    pub cors_allowed_origins: Vec<String>,
    /// Username required to log into the `/admin` console. The password is
    /// not stored here — set it with `hipfire admin set-password` (hash
    /// lands in `~/.hipfire/admin.passwd`).
    #[serde(default = "default_admin_user")]
    pub admin_user: String,
    /// API credential rollout policy. `auto` preserves anonymous loopback
    /// compatibility and requires credentials for non-loopback binds.
    #[serde(default = "default_api_auth_mode")]
    pub api_auth_mode: ApiAuthMode,
    /// Explicitly acknowledge an unauthenticated non-loopback bind when
    /// `api_auth_mode` is `off` or `optional`.
    #[serde(default)]
    pub unsafe_allow_unauthenticated_remote: bool,
    /// Rate limits applied when `host` is a loopback address, where the only
    /// possible clients are processes on this machine. Consulted ONLY for a
    /// loopback bind — a network-reachable bind always uses the standard
    /// policy, so nothing here can loosen limits for a remote client.
    ///
    /// Unset fields keep `RatePolicy::loopback_default()`, which is
    /// effectively unlimited; set any field to narrow it. Same shape as a
    /// user/token override, e.g.:
    ///
    /// ```json
    /// "local_rate_policy": { "max_in_flight_text": 2 }
    /// ```
    #[serde(default)]
    pub local_rate_policy: hipfire_auth::RatePolicyOverride,
    /// Root directory for images saved by the SD API compatibility routes
    /// (`save_images: true`). Client-supplied `outdir_*` override_settings
    /// are ignored; every SD API image write stays under this root.
    #[serde(default = "default_sdapi_output_root")]
    pub sdapi_output_root: String,
    /// Upper bound on any single SD API dimension (width/height and their
    /// highres/firstphase variants). Client requests above it get a 400.
    #[serde(default = "default_sdapi_max_dimension")]
    pub sdapi_max_dimension: u32,
    /// Upper bound on SD API step counts (steps and hr_second_pass_steps).
    #[serde(default = "default_sdapi_max_steps")]
    pub sdapi_max_steps: u32,
    /// Upper bound on SD API `batch_size`.
    #[serde(default = "default_sdapi_max_batch_size")]
    pub sdapi_max_batch_size: u32,
    /// Upper bound on SD API `n_iter`.
    #[serde(default = "default_sdapi_max_n_iter")]
    pub sdapi_max_n_iter: u32,
    /// Upper bound on `batch_size × n_iter` (total images per request).
    #[serde(default = "default_sdapi_max_total_batches")]
    pub sdapi_max_total_batches: u32,
    /// Primary local model root. When unset, Hipfire uses
    /// `~/.hipfire/models`.
    #[serde(default)]
    pub models_dir: Option<String>,
    /// Optional extra read-only model root (e.g. an NFS share such as
    /// `/srv/hipfire`). When set, the network-facing server routes resolve
    /// model identifiers within this root in addition to `models_dir`.
    /// Unset by default; local CLI/eval callers are unaffected.
    #[serde(default)]
    pub models_network_dir: Option<String>,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default = "default_prewarm_priority")]
    pub prewarm_priority: u32,
    #[serde(default = "default_max_seq")]
    pub max_seq: u32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_repeat_penalty")]
    pub repeat_penalty: f64,
    #[serde(default = "default_resource_lock_enabled")]
    pub resource_lock_enabled: bool,
    #[serde(default = "default_resource_lock_gpus")]
    pub resource_lock_gpus: Vec<String>,
    #[serde(default = "default_resource_lock_npus")]
    pub resource_lock_npus: Vec<String>,
    #[serde(default = "default_resource_lock_wait_ms")]
    pub resource_lock_wait_ms: u32,
    #[serde(default = "default_scheduler_memory_budget_bytes")]
    pub scheduler_system_memory_budget_bytes: u64,
    #[serde(default = "default_scheduler_memory_budget_bytes")]
    pub scheduler_system_memory_headroom_bytes: u64,
    #[serde(default = "default_scheduler_memory_budget_bytes")]
    pub scheduler_vram_budget_bytes: u64,
    #[serde(default = "default_scheduler_memory_budget_bytes")]
    pub scheduler_vram_headroom_bytes: u64,
    #[serde(default = "default_model_residency_mode")]
    pub model_residency_mode: String,
    /// Sampler RNG seeding: `"fixed"` (reproducible) or `"random"` (per stream).
    #[serde(default = "default_sampler_rng")]
    pub sampler_rng: String,
    #[serde(default = "default_kv_cache")]
    pub kv_cache: String,
    #[serde(default = "default_kv_adaptive")]
    pub kv_adaptive: String,
    #[serde(default = "default_kv_window_precision")]
    pub kv_window_precision: String,
    #[serde(default = "default_deltanet_state_precision")]
    pub deltanet_state_precision: String,
    #[serde(default = "default_oq_compact_multicol_wide")]
    pub oq_compact_multicol_wide: bool,
    /// Stream routed experts from host memory instead of keeping every expert
    /// resident. The env var `HIPFIRE_QWEN35_PAGED_EXPERTS` still overrides.
    #[serde(default = "default_qwen35_paged_experts")]
    pub qwen35_paged_experts: bool,
    /// Mint a Final checkpoint per prefilled session. Off by default — the
    /// attach path that would consume it is unreachable, and the clone is never
    /// freed. The env var `HIPFIRE_QWEN35_FINAL_CHECKPOINTS` still overrides.
    #[serde(default = "default_qwen35_final_checkpoints")]
    pub qwen35_final_checkpoints: bool,
    /// Resident budget for the paged-expert cache, in MiB. Only meaningful with
    /// `qwen35_paged_experts`.
    #[serde(default = "default_qwen35_expert_cache_mb")]
    pub qwen35_expert_cache_mb: u64,
    /// Refuse a load that would not fit in `MemAvailable`. Turning this off on a
    /// unified-memory host lets an over-large load invoke the OOM killer.
    #[serde(default = "default_load_mem_check")]
    pub load_mem_check: bool,
    /// GiB of headroom the load check leaves for the rest of the system.
    #[serde(default = "default_load_mem_reserve_gib")]
    pub load_mem_reserve_gib: u32,
    #[serde(default = "default_lmhead_twostage")]
    pub lmhead_twostage: String,
    #[serde(default = "default_flash_mode")]
    pub flash_mode: String,
    /// Resident model workers allowed at once, counting the active one.
    #[serde(default = "default_max_resident_workers")]
    pub max_resident_workers: u32,
    /// `off` / `auto` / `on`, or an absolute drafter path (which implies `on`).
    /// Split with [`dflash_draft_setting`].
    #[serde(default = "default_dflash_draft")]
    pub dflash_draft: String,
    #[serde(default = "default_spec_adaptive_block")]
    pub spec_adaptive_block: bool,
    #[serde(default = "default_spec_block")]
    pub spec_block: u32,
    #[serde(default = "default_dflash_no_repeat_ngram")]
    pub dflash_no_repeat_ngram: serde_json::Value,
    /// Opt-in drafter-free n-gram speculative decode. Drafts from token
    /// statistics; on a miss the DFlash drafter runs unchanged.
    #[serde(default = "default_ngram_spec")]
    pub ngram_spec: bool,
    /// Root directory for the persistent n-gram tables, or a RAM-only sentinel.
    ///
    /// Empty, `ram`, `none` or `off` all mean hot tier only — session-local
    /// RAM, nothing written to disk. The word forms exist so the RAM case is
    /// expressible as a value: in a settings UI an empty string is
    /// indistinguishable from a field nobody has touched.
    #[serde(default = "default_ngram_spec_store_root")]
    pub ngram_spec_store_root: String,
    /// Scope name identifying the *tokenizer* these tables belong to. Empty =
    /// derive from the model filename, which never wrongly shares a table.
    /// Set two models to the same scope only when they share a tokenizer —
    /// records are token ids and mean nothing across tokenizers.
    #[serde(default = "default_ngram_spec_scope")]
    pub ngram_spec_scope: String,
    /// Size of a newly created per-scope table, in MiB. This *is* the budget:
    /// the file is allocated in full and never grows, so a full block evicts
    /// rather than expanding. 256 MiB = 65536 blocks of 4 KiB.
    #[serde(default = "default_ngram_spec_store_mb")]
    pub ngram_spec_store_mb: u32,
    /// Probe orders, longest first. Measured on 1M tokens of Rust: going past
    /// quad keeps paying on code (2..5 -> 1.80 accepted/step, 2..8 -> 2.11),
    /// and is flat on prose.
    #[serde(default = "default_ngram_spec_orders")]
    pub ngram_spec_orders: String,
    /// After the first drafted token, only extend the chain while the winning
    /// order is at least this. The load-bearing knob: without it the chain pads
    /// to `max_spine` and burns verify width (floor 0 -> 16.0 drafted/step at
    /// 20.6% efficiency; floor 8 -> 6.94 at 37.9%). 0 disables the gate.
    #[serde(default = "default_ngram_spec_chain_floor")]
    pub ngram_spec_chain_floor: u8,
    #[serde(default = "default_ngram_spec_min_acceptance")]
    pub ngram_spec_min_acceptance: f64,
    #[serde(default = "default_ngram_spec_min_acceptance_proposals")]
    pub ngram_spec_min_acceptance_proposals: u32,
    #[serde(default = "default_ngram_spec_max_spine")]
    pub ngram_spec_max_spine: u32,
    /// Observations before a gram is worth a disk write. Gates persistence
    /// only, never drafting — precision is flat across counts, and requiring a
    /// count to draft measurably hurts (1.80 accepted/step at 1, 0.75 at 9).
    #[serde(default = "default_ngram_spec_promote_count")]
    pub ngram_spec_promote_count: u16,
    /// Which store the write path feeds: `user`, `topic`, or `none`. Only a
    /// store private to its scope may be written; a shared one is read-only.
    #[serde(default = "default_ngram_spec_write_target")]
    pub ngram_spec_write_target: String,
    #[serde(default = "default_mtp_mode")]
    pub mtp_mode: String,
    #[serde(default = "default_mtp_k")]
    pub mtp_k: u32,
    #[serde(default = "default_thinking")]
    pub thinking: String,
    #[serde(default = "default_gpu_slab_load")]
    pub gpu_slab_load: String,
    /// Render prompts through the model's Jinja chat template rather than the
    /// hand-rolled ChatFrame: `auto` | `on` | `off`. `auto` (the default) defers to
    /// the model's architecture, which does not answer the same way for every
    /// family — see `hipfire_model::chat_prompt_policy`. The env var
    /// `HIPFIRE_JINJA_CHAT` still overrides it.
    #[serde(default = "default_jinja_chat")]
    pub jinja_chat: String,
    /// Chat template to use instead of the artifact's embedded one. Only meaningful
    /// while `jinja_chat` is on, since that is the only path where a template renders
    /// the prompt.
    #[serde(default)]
    pub chat_template_file: Option<String>,
    #[serde(default = "default_prompt_normalize")]
    pub prompt_normalize: bool,
    #[serde(default = "default_cask_auto_attach")]
    pub cask_auto_attach: bool,
    #[serde(default)]
    pub cask_sidecar: Option<String>,
    #[serde(default)]
    pub cask: bool,
    #[serde(default = "default_cask_budget")]
    pub cask_budget: u32,
    #[serde(default = "default_cask_beta")]
    pub cask_beta: u32,
    #[serde(default = "default_cask_core_frac")]
    pub cask_core_frac: f64,
    #[serde(default = "default_cask_fold_m")]
    pub cask_fold_m: u32,
    #[serde(default = "default_mmq_screen")]
    pub mmq_screen: String,
    #[serde(default = "default_mmq_screen_threshold")]
    pub mmq_screen_threshold: f64,
    #[serde(default = "default_prefill_compression")]
    pub prefill_compression: String,
    #[serde(default = "default_prefill_threshold")]
    pub prefill_threshold: u32,
    #[serde(default = "default_prefill_keep_ratio")]
    pub prefill_keep_ratio: f64,
    #[serde(default = "default_prefill_alpha")]
    pub prefill_alpha: f64,
    #[serde(default = "default_prefill_min_keep")]
    pub prefill_min_keep: u32,
    #[serde(default = "default_prefill_sink")]
    pub prefill_sink: u32,
    #[serde(default = "default_prefill_recent")]
    pub prefill_recent: u32,
    #[serde(default = "default_prefill_block")]
    pub prefill_block: u32,
    #[serde(default)]
    pub prefill_drafter: Option<String>,
    #[serde(default = "default_prefill_drafter_device")]
    pub prefill_drafter_device: i32,
    #[serde(default)]
    pub prefill_profile: bool,
    #[serde(default = "default_prefill_sparse_threshold")]
    pub prefill_sparse_threshold: u32,
    #[serde(default)]
    pub model_overrides: HashMap<String, serde_json::Value>,
}

impl HipfireConfig {
    /// Merge per-model overrides for `tag` on top of global config.
    pub fn resolve_for_model(&self, tag: &str) -> Self {
        let raw = serde_json::to_value(self).unwrap_or_else(|_| Value::Object(Map::new()));
        resolve_typed_config_document(&raw, Some(tag)).config
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigDiagnosticSeverity {
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigDiagnostic {
    pub severity: ConfigDiagnosticSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedTypedConfig {
    pub config: HipfireConfig,
    pub layers: Vec<ConfigLayer>,
    pub resolution: ConfigResolution,
    pub diagnostics: Vec<ConfigDiagnostic>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadedConfig {
    pub config_path: PathBuf,
    pub raw_document: Value,
    pub read_error: Option<String>,
    pub host_config_path: PathBuf,
    pub host_raw_document: Value,
    pub host_read_error: Option<String>,
    pub additional_layers: Vec<ConfigLayer>,
    pub layers: Vec<ConfigLayer>,
    pub config: HipfireConfig,
    pub resolution: ConfigResolution,
    pub diagnostics: Vec<ConfigDiagnostic>,
}

impl LoadedConfig {
    pub fn from_config(config: HipfireConfig) -> Self {
        let raw_document =
            serde_json::to_value(&config).unwrap_or_else(|_| Value::Object(Map::new()));
        loaded_config_from_document(config_path(), raw_document, None, Vec::new())
    }

    pub fn with_additional_layer(mut self, layer: ConfigLayer) -> Self {
        if !layer.values.is_empty() {
            self.additional_layers.push(layer);
            self.refresh();
        }
        self
    }

    pub fn resolve_for_model(&self, model_tag: &str) -> ResolvedTypedConfig {
        resolve_typed_config_documents_with_layers(
            &self.raw_document,
            &self.host_raw_document,
            Some(model_tag),
            &self.additional_layers,
        )
    }

    fn refresh(&mut self) {
        let resolved = resolve_typed_config_documents_with_layers(
            &self.raw_document,
            &self.host_raw_document,
            None,
            &self.additional_layers,
        );
        self.config = resolved.config;
        self.layers = resolved.layers;
        self.resolution = resolved.resolution;
        self.diagnostics = resolved.diagnostics;
    }
}

impl Default for HipfireConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            cors_allowed_origins: default_cors_allowed_origins(),
            admin_user: default_admin_user(),
            api_auth_mode: default_api_auth_mode(),
            unsafe_allow_unauthenticated_remote: false,
            // Empty = keep `RatePolicy::loopback_default()` unnarrowed.
            local_rate_policy: hipfire_auth::RatePolicyOverride::default(),
            sdapi_output_root: default_sdapi_output_root(),
            sdapi_max_dimension: default_sdapi_max_dimension(),
            sdapi_max_steps: default_sdapi_max_steps(),
            sdapi_max_batch_size: default_sdapi_max_batch_size(),
            sdapi_max_n_iter: default_sdapi_max_n_iter(),
            sdapi_max_total_batches: default_sdapi_max_total_batches(),
            models_dir: None,
            models_network_dir: None,
            default_model: None,
            sampler_rng: default_sampler_rng(),
            prewarm_priority: default_prewarm_priority(),
            max_seq: default_max_seq(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            repeat_penalty: default_repeat_penalty(),
            resource_lock_enabled: default_resource_lock_enabled(),
            resource_lock_gpus: default_resource_lock_gpus(),
            resource_lock_npus: default_resource_lock_npus(),
            resource_lock_wait_ms: default_resource_lock_wait_ms(),
            scheduler_system_memory_budget_bytes: default_scheduler_memory_budget_bytes(),
            scheduler_system_memory_headroom_bytes: default_scheduler_memory_budget_bytes(),
            scheduler_vram_budget_bytes: default_scheduler_memory_budget_bytes(),
            scheduler_vram_headroom_bytes: default_scheduler_memory_budget_bytes(),
            model_residency_mode: default_model_residency_mode(),
            kv_cache: default_kv_cache(),
            kv_adaptive: default_kv_adaptive(),
            kv_window_precision: default_kv_window_precision(),
            deltanet_state_precision: default_deltanet_state_precision(),
            oq_compact_multicol_wide: default_oq_compact_multicol_wide(),
            qwen35_paged_experts: default_qwen35_paged_experts(),
            qwen35_final_checkpoints: default_qwen35_final_checkpoints(),
            qwen35_expert_cache_mb: default_qwen35_expert_cache_mb(),
            load_mem_check: default_load_mem_check(),
            load_mem_reserve_gib: default_load_mem_reserve_gib(),
            lmhead_twostage: default_lmhead_twostage(),
            flash_mode: default_flash_mode(),
            max_resident_workers: default_max_resident_workers(),
            dflash_draft: default_dflash_draft(),
            spec_adaptive_block: default_spec_adaptive_block(),
            spec_block: default_spec_block(),
            ngram_spec: default_ngram_spec(),
            ngram_spec_store_root: default_ngram_spec_store_root(),
            ngram_spec_scope: default_ngram_spec_scope(),
            ngram_spec_store_mb: default_ngram_spec_store_mb(),
            ngram_spec_orders: default_ngram_spec_orders(),
            ngram_spec_chain_floor: default_ngram_spec_chain_floor(),
            ngram_spec_min_acceptance: default_ngram_spec_min_acceptance(),
            ngram_spec_min_acceptance_proposals: default_ngram_spec_min_acceptance_proposals(),
            ngram_spec_max_spine: default_ngram_spec_max_spine(),
            ngram_spec_promote_count: default_ngram_spec_promote_count(),
            ngram_spec_write_target: default_ngram_spec_write_target(),
            dflash_no_repeat_ngram: default_dflash_no_repeat_ngram(),
            mtp_mode: default_mtp_mode(),
            mtp_k: default_mtp_k(),
            thinking: default_thinking(),
            gpu_slab_load: default_gpu_slab_load(),
            jinja_chat: default_jinja_chat(),
            chat_template_file: None,
            prompt_normalize: default_prompt_normalize(),
            cask_auto_attach: default_cask_auto_attach(),
            cask_sidecar: None,
            cask: false,
            cask_budget: default_cask_budget(),
            cask_beta: default_cask_beta(),
            cask_core_frac: default_cask_core_frac(),
            cask_fold_m: default_cask_fold_m(),
            mmq_screen: default_mmq_screen(),
            mmq_screen_threshold: default_mmq_screen_threshold(),
            prefill_compression: default_prefill_compression(),
            prefill_threshold: default_prefill_threshold(),
            prefill_keep_ratio: default_prefill_keep_ratio(),
            prefill_alpha: default_prefill_alpha(),
            prefill_min_keep: default_prefill_min_keep(),
            prefill_sink: default_prefill_sink(),
            prefill_recent: default_prefill_recent(),
            prefill_block: default_prefill_block(),
            prefill_drafter: None,
            prefill_drafter_device: default_prefill_drafter_device(),
            prefill_profile: false,
            prefill_sparse_threshold: default_prefill_sparse_threshold(),
            model_overrides: HashMap::new(),
        }
    }
}

pub fn hipfire_dir() -> PathBuf {
    dirs::home_dir()
        .expect("no home directory")
        .join(".hipfire")
}

pub fn config_path() -> PathBuf {
    hipfire_dir().join("config.json")
}

pub fn host_config_path() -> PathBuf {
    hipfire_dir().join("config.local.json")
}

pub fn models_dir() -> PathBuf {
    hipfire_dir().join("models")
}

pub fn configured_models_dir(config: &HipfireConfig) -> PathBuf {
    config
        .models_dir
        .as_deref()
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(models_dir)
}

/// Path to the local admin bearer secret. Same-box clients (CLI/TUI) read
/// this file and present it as `Authorization: Bearer <secret>` to skip the
/// `/admin` login flow — "can read the file ⇒ you're the admin".
pub fn admin_secret_path() -> PathBuf {
    hipfire_dir().join("admin.secret")
}

/// Path to the argon2id hash of the `/admin` console password, written by
/// `hipfire admin set-password`.
pub fn admin_password_path() -> PathBuf {
    hipfire_dir().join("admin.passwd")
}

/// Read the local admin bearer secret if it exists. Read-only: never creates
/// the file (only the daemon does that, via [`ensure_admin_secret`]).
pub fn read_admin_secret() -> Option<String> {
    let secret = std::fs::read_to_string(admin_secret_path()).ok()?;
    let secret = secret.trim().to_string();
    (!secret.is_empty()).then_some(secret)
}

/// Read-or-create the local admin bearer secret (0600). Called by the daemon
/// at startup so same-box CLI/TUI clients can authenticate without a password.
pub fn ensure_admin_secret() -> std::io::Result<String> {
    if let Some(existing) = read_admin_secret() {
        return Ok(existing);
    }
    let path = admin_secret_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let secret = random_token();
    write_private(&path, &secret)?;
    Ok(secret)
}

/// Hash a password with argon2id (PHC string) and persist it to
/// `admin.passwd` (0600). Used by `hipfire admin set-password`.
pub fn set_admin_password(password: &str) -> std::io::Result<()> {
    let hash = hash_admin_password(password).map_err(std::io::Error::other)?;
    let path = admin_password_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_private(&path, &hash)
}

/// Load the stored argon2id password hash, if a password has been set.
pub fn read_admin_password_hash() -> Option<String> {
    let hash = std::fs::read_to_string(admin_password_path()).ok()?;
    let hash = hash.trim().to_string();
    (!hash.is_empty()).then_some(hash)
}

/// Compute an argon2id PHC hash string for `password`.
pub fn hash_admin_password(password: &str) -> Result<String, String> {
    use argon2::password_hash::{PasswordHasher, SaltString};
    use argon2::Argon2;
    let salt = SaltString::encode_b64(&random_bytes::<16>()).map_err(|err| err.to_string())?;
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|hash| hash.to_string())
        .map_err(|err| err.to_string())
}

/// Verify `password` against a stored argon2id PHC hash.
pub fn verify_admin_password(password: &str, phc_hash: &str) -> bool {
    use argon2::password_hash::{PasswordHash, PasswordVerifier};
    use argon2::Argon2;
    let Ok(parsed) = PasswordHash::new(phc_hash) else {
        return false;
    };
    Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok()
}

/// Constant-time comparison of a presented bearer secret against the stored
/// one. An empty stored secret never authorizes.
pub fn verify_admin_secret(presented: &str, stored: &str) -> bool {
    if stored.is_empty() {
        return false;
    }
    let a = presented.as_bytes();
    let b = stored.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn random_token() -> String {
    random_bytes::<32>()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    getrandom::getrandom(&mut bytes).expect("getrandom failed");
    bytes
}

fn write_private(path: &Path, contents: &str) -> std::io::Result<()> {
    std::fs::write(path, contents)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

pub fn load_config_bundle() -> LoadedConfig {
    let (path, document, read_error) = read_config_document(config_path());
    let (host_path, host_document, host_read_error) = read_config_document(host_config_path());
    loaded_config_from_documents(
        path,
        document,
        read_error,
        host_path,
        host_document,
        host_read_error,
        Vec::new(),
    )
}

pub fn load_config() -> HipfireConfig {
    load_config_bundle().config
}

pub fn resolve_typed_config_document(raw: &Value, model_tag: Option<&str>) -> ResolvedTypedConfig {
    resolve_typed_config_document_with_layers(raw, model_tag, &[])
}

pub fn resolve_typed_config_document_with_layers(
    raw: &Value,
    model_tag: Option<&str>,
    additional_layers: &[ConfigLayer],
) -> ResolvedTypedConfig {
    resolve_typed_config_documents_with_layers(
        raw,
        &Value::Object(Map::new()),
        model_tag,
        additional_layers,
    )
}

pub fn resolve_typed_config_documents_with_layers(
    raw: &Value,
    host_local: &Value,
    model_tag: Option<&str>,
    additional_layers: &[ConfigLayer],
) -> ResolvedTypedConfig {
    let mut layers = config_layers_from_documents(raw, Some(host_local), model_tag);
    // Environment sits between the files and the caller's CLI/request layers:
    // an env override beats what is written on disk, and an explicit request
    // still beats the environment.
    let (env_layer, env_rejected) = config_layer_from_env(config_schema());
    if let Some(env_layer) = env_layer {
        layers.push(env_layer);
    }
    layers.extend(additional_layers.iter().cloned());
    let mut resolved =
        resolve_typed_config_layers(&layers, model_overrides_from_documents(raw, host_local));
    resolved
        .diagnostics
        .extend(env_rejected.into_iter().map(|message| ConfigDiagnostic {
            severity: ConfigDiagnosticSeverity::Warning,
            message,
        }));
    resolved
}

pub fn resolve_typed_config_layers(
    layers: &[ConfigLayer],
    model_overrides: HashMap<String, Value>,
) -> ResolvedTypedConfig {
    let (layers, mut diagnostics) = apply_renamed_keys(layers);
    let layers = layers.as_slice();
    let mut resolution = resolve_config_layers(config_schema(), layers);
    unknown_keys_in_unapplied_overrides(layers, &model_overrides, &mut resolution);
    diagnostics.extend(domain_diagnostics(&resolution));
    diagnostics.extend(deprecated_kv_diagnostics(&resolution));
    let config = materialize_config(&resolution, model_overrides, &mut diagnostics);
    ResolvedTypedConfig {
        config,
        layers: layers.to_vec(),
        resolution,
        diagnostics,
    }
}

pub fn loaded_config_from_document(
    config_path: PathBuf,
    raw_document: Value,
    read_error: Option<String>,
    additional_layers: Vec<ConfigLayer>,
) -> LoadedConfig {
    loaded_config_from_documents(
        config_path,
        raw_document,
        read_error,
        host_config_path(),
        Value::Object(Map::new()),
        None,
        additional_layers,
    )
}

pub fn loaded_config_from_documents(
    config_path: PathBuf,
    raw_document: Value,
    read_error: Option<String>,
    host_config_path: PathBuf,
    host_raw_document: Value,
    host_read_error: Option<String>,
    additional_layers: Vec<ConfigLayer>,
) -> LoadedConfig {
    let resolved = resolve_typed_config_documents_with_layers(
        &raw_document,
        &host_raw_document,
        None,
        &additional_layers,
    );
    LoadedConfig {
        config_path,
        raw_document,
        read_error,
        host_config_path,
        host_raw_document,
        host_read_error,
        additional_layers,
        layers: resolved.layers,
        config: resolved.config,
        resolution: resolved.resolution,
        diagnostics: resolved.diagnostics,
    }
}

fn read_config_document(path: PathBuf) -> (PathBuf, Value, Option<String>) {
    match std::fs::read_to_string(&path) {
        Ok(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(document) => (path, document, None),
            Err(err) => (
                path,
                Value::Object(Map::new()),
                Some(format!("parse error: {err}")),
            ),
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            (path, Value::Object(Map::new()), None)
        }
        Err(err) => (
            path,
            Value::Object(Map::new()),
            Some(format!("read error: {err}")),
        ),
    }
}

fn materialize_config(
    resolution: &ConfigResolution,
    model_overrides: HashMap<String, Value>,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) -> HipfireConfig {
    let mut object = Map::new();
    for resolved in &resolution.values {
        if let Some(value) = &resolved.value {
            object.insert(resolved.key.clone(), value.clone());
        }
    }
    if !model_overrides.is_empty() {
        object.insert(
            "model_overrides".to_string(),
            Value::Object(model_overrides.into_iter().collect()),
        );
    }

    match serde_json::from_value::<HipfireConfig>(Value::Object(object.clone())) {
        Ok(config) => config,
        // ONE unparseable field must not discard the other ninety.
        //
        // This arm used to return `HipfireConfig::default()`, which meant a
        // single type mismatch anywhere silently reset EVERY setting —
        // `models_dir`, `kv_cache`, scheduler budgets, `model_overrides`, all of
        // it. Silently, because the diagnostic below is dropped on the floor by
        // `HipfireConfig::resolve_for_model`, which returns only `.config`.
        //
        // That was not hypothetical. `hipfire-server`'s `apply_resource_list_env`
        // writes `HIPFIRE_RESOURCE_LOCK_NPUS=1` for `resource_lock_npus:
        // ["auto"]` — correct for its consumer, which documents `1` as "lease
        // every detected NPU" — but this crate's env layer claims the same name
        // for a `Vec<String>` field and reads `1` as a JSON number. A server with
        // NPU auto-leasing therefore ran on default config for everything, with
        // no error and no log. Two `hipfire-server` prewarm tests were the only
        // thing that ever noticed.
        //
        // Recovery is field-by-field: start from the defaults and accept each
        // resolved key only if the document still deserializes with it. The bad
        // key alone falls back, and is NAMED in a diagnostic and on stderr so it
        // stops being invisible. O(keys) deserializations, on the error path
        // only — the happy path above is untouched.
        Err(err) => {
            diagnostics.push(ConfigDiagnostic {
                severity: ConfigDiagnosticSeverity::Error,
                message: format!("failed to materialize typed config: {err}"),
            });
            let mut accepted = match serde_json::to_value(HipfireConfig::default()) {
                Ok(Value::Object(map)) => map,
                // Defaults themselves do not serialize: nothing to recover onto.
                _ => return HipfireConfig::default(),
            };
            let mut rejected: Vec<String> = Vec::new();
            for (key, value) in object {
                let mut trial = accepted.clone();
                trial.insert(key.clone(), value);
                if serde_json::from_value::<HipfireConfig>(Value::Object(trial.clone())).is_ok() {
                    accepted = trial;
                } else {
                    rejected.push(key);
                }
            }
            for key in &rejected {
                diagnostics.push(ConfigDiagnostic {
                    severity: ConfigDiagnosticSeverity::Error,
                    message: format!(
                        "config key `{key}` has a value of the wrong type; it fell back to its \
                         default. Every OTHER key was kept."
                    ),
                });
            }
            if !rejected.is_empty() {
                // Deliberately stderr, not just a diagnostic: the caller that
                // hits this most (`resolve_for_model`) discards diagnostics, and
                // a config that silently reverts to defaults is the failure this
                // whole arm exists to stop being silent.
                eprintln!(
                    "hipfire-config: ignoring {} malformed config key(s): {}. \
                     They fell back to defaults; all other keys were kept.",
                    rejected.len(),
                    rejected.join(", ")
                );
            }
            serde_json::from_value::<HipfireConfig>(Value::Object(accepted))
                .unwrap_or_else(|_| HipfireConfig::default())
        }
    }
}

/// Move values written under a renamed key onto the key that replaced it.
///
/// The old key is HONOURED, not dropped, and the operator is told what to write
/// instead. Dropping it would be the failure mode this whole area keeps
/// producing: a setting that parses, applies to nothing, and says nothing.
///
/// If both names are present the new one wins and the old is reported as
/// ignored, because guessing which the operator meant is worse than saying they
/// disagree.
fn apply_renamed_keys(layers: &[ConfigLayer]) -> (Vec<ConfigLayer>, Vec<ConfigDiagnostic>) {
    let mut diagnostics = Vec::new();
    let mut out = layers.to_vec();
    for layer in &mut out {
        for (old, new) in RENAMED_KEYS {
            let Some(value) = layer.values.remove(*old) else {
                continue;
            };
            if layer.values.contains_key(*new) {
                diagnostics.push(ConfigDiagnostic {
                    severity: ConfigDiagnosticSeverity::Warning,
                    message: format!(
                        "config key `{old}` was renamed to `{new}`, and both are set — the \
                         value of `{old}` is ignored. Remove it."
                    ),
                });
                continue;
            }
            diagnostics.push(ConfigDiagnostic {
                severity: ConfigDiagnosticSeverity::Warning,
                message: format!(
                    "config key `{old}` was renamed to `{new}`; the value was applied, but \
                     rename it — the old name will stop working."
                ),
            });
            layer.values.insert((*new).to_string(), value);
        }
    }
    (out, diagnostics)
}

/// Which [`PathExistence`] a value is actually subject to, if any.
///
/// A union resolves to whichever arm ACCEPTS the value, so
/// `ngram_spec_store_root: "ram"` takes the sentinel arm and is not a path at all,
/// while `"/var/lib/hipfire/ngram"` takes the Path arm and is. Asking the type
/// alone would get both wrong.
fn path_existence_for(value: &Value, ty: &ConfigType) -> Option<PathExistence> {
    match ty {
        ConfigType::Path { existence } => Some(*existence),
        ConfigType::OneOf { arms } => arms
            .iter()
            .find(|arm| validate_resolved_value(value, arm).is_ok())
            .and_then(|arm| path_existence_for(value, arm)),
        _ => None,
    }
}

/// Warn about configured paths that are not on disk.
///
/// Separate from [`domain_diagnostics`] because this one does I/O. The resolver
/// is pure and runs in tests and the settings editor; a filesystem probe there
/// would be wrong and slow. Callers run this once, at startup or from `doctor`.
///
/// Always a warning. A network models dir may be mounted after the daemon
/// starts, and existence is TOCTOU regardless — it says the path was there at
/// boot, not that it will be there at use. Refusing to start would trade a
/// visible warning for an outage.
pub fn path_existence_diagnostics(config: &HipfireConfig) -> Vec<ConfigDiagnostic> {
    let values = config_value_map(config);
    let mut out = Vec::new();
    for field in config_schema() {
        let Some(value) = values.get(field.key) else {
            continue;
        };
        let Some(text) = value.as_str() else { continue };
        let trimmed = text.trim();
        // An empty path is "unset", not "missing"; Requirement covers that.
        if trimmed.is_empty() {
            continue;
        }
        let Some(existence) = path_existence_for(value, &field.ty) else {
            continue;
        };
        // A malformed path is domain_diagnostics' finding, not ours — do not
        // report the same value twice under two headings.
        if resolve::validate_path(trimmed).is_err() {
            continue;
        }
        let path = std::path::Path::new(trimmed);
        let message = match existence {
            PathExistence::Exists if !path.exists() => {
                format!("config key `{}` = {trimmed} does not exist", field.key)
            }
            PathExistence::ParentExists => match path.parent() {
                Some(parent) if !parent.as_os_str().is_empty() && !parent.is_dir() => format!(
                    "config key `{}` = {trimmed} will be created on first use, but its parent \
                     directory {} does not exist and nothing here creates one",
                    field.key,
                    parent.display()
                ),
                _ => continue,
            },
            PathExistence::Exists => continue,
        };
        out.push(ConfigDiagnostic {
            severity: ConfigDiagnosticSeverity::Warning,
            message,
        });
    }
    out
}

/// KV storage modes hipfire is retiring, in the one place both the config layer
/// and the loader read.
///
/// `hipfire-serving-core`'s `reject_deprecated_kv_mode` refuses these at model
/// load. That refusal used to be the FIRST time an operator heard about it: the
/// `kv_cache` enum accepted `q8`/`asym2`/`asym3`/`asym4`, config validation
/// passed, the server reported healthy, and only the first request that needed a
/// load failed (issue #386). The list lives here so the warning fires when the
/// config resolves and the loader's refusal stays the same decision, not a
/// second opinion.
pub const DEPRECATED_KV_MODES: &[&str] = &[
    "q4", "q8", "int8", "int8c", "hfq4kv", "hfq4", "hfq8", "asym2", "asym3", "asym4", "fwht2",
    "fwht3", "fwht4", "turbo4",
];

/// Warn, at config-resolution time, about a `kv_cache` the loader will refuse.
///
/// Warning rather than error for the same reason as [`domain_diagnostics`]: these
/// are configs already running, and `HIPFIRE_KV_ALLOW_DEPRECATED=1` still admits
/// them. Naming the value early is the fix; refusing to resolve is not.
fn deprecated_kv_diagnostics(resolution: &ConfigResolution) -> Vec<ConfigDiagnostic> {
    resolution
        .values
        .iter()
        .filter(|resolved| resolved.key == "kv_cache")
        // Only what someone actually WROTE, matching domain_diagnostics.
        .filter(|resolved| {
            !matches!(
                resolved.source.as_ref().map(|source| source.kind),
                None | Some(ConfigLayerKind::CompiledDefault)
            )
        })
        .filter_map(|resolved| {
            let mode = resolved.value.as_ref()?.as_str()?;
            if !DEPRECATED_KV_MODES.contains(&mode) {
                return None;
            }
            Some(ConfigDiagnostic {
                severity: ConfigDiagnosticSeverity::Warning,
                message: format!(
                    "config key `kv_cache` = \"{mode}\" is DEPRECATED and the model loader \
                     will refuse it. hipfire is retiring KV storage down to two families: \
                     kvarn (kvarn2 / kvarn / kvarn4 / kvarn8) and unquantized (fp32). Set \
                     HIPFIRE_KV_ALLOW_DEPRECATED=1 to run it during migration, or pick a \
                     supported mode."
                ),
            })
        })
        .collect()
}

/// Warn about resolved values that violate their field's declared domain.
///
/// `resolve_field` takes a file value verbatim, so until now nothing checked
/// one: only the environment went through `parse_env_value`, and the typed
/// materialize step catches a Rust type mismatch but never a domain one —
/// `kv_cache: "kvarnn"` is a perfectly good String and reached the KV match as
/// an unrecognized mode.
///
/// Warning, not error, and the value still resolves as it always did. These are
/// configs already running; naming a bad value is the fix, refusing to boot on
/// it is not.
fn domain_diagnostics(resolution: &ConfigResolution) -> Vec<ConfigDiagnostic> {
    let by_key = config_schema()
        .iter()
        .map(|field| (field.key, field))
        .collect::<BTreeMap<_, _>>();
    resolution
        .values
        .iter()
        .filter(|resolved| {
            // Only what someone actually WROTE. A compiled default that fails
            // its own domain is a schema bug, and reporting it to the operator
            // would be noise they cannot act on.
            !matches!(
                resolved.source.as_ref().map(|source| source.kind),
                None | Some(ConfigLayerKind::CompiledDefault)
            )
        })
        .filter_map(|resolved| {
            let field = by_key.get(resolved.key.as_str())?;
            let value = resolved.value.as_ref()?;
            let want = validate_resolved_value(value, &field.ty).err()?;
            let source = resolved
                .source
                .as_ref()
                .map(|source| match source.id.as_deref() {
                    Some(id) => format!("{:?}:{id}", source.kind),
                    None => format!("{:?}", source.kind),
                })
                .unwrap_or_default();
            Some(ConfigDiagnostic {
                severity: ConfigDiagnosticSeverity::Warning,
                message: format!(
                    "config key `{}` = {value} (from {source}) is outside its declared \
                     domain: {want}. It was kept as written.",
                    resolved.key
                ),
            })
        })
        .collect()
}

/// Report unknown field names inside `model_overrides` entries that did NOT
/// become a layer.
///
/// `resolve_config_layers` can only inspect layers, and a `model_overrides`
/// entry becomes one only when its key matches the tag being resolved
/// (`config_layers_from_documents`). So a typo'd field in an entry for any
/// OTHER model was never schema-checked — and neither was the entry whose key
/// itself was misspelled, since that key matches no tag and so contributes no
/// layer at all. Such an entry was silent twice over: it applied nothing, and
/// nothing said its contents were unreadable.
///
/// Entries that DID become a layer are skipped; `resolve_config_layers` has
/// already reported those, and re-reporting would double every finding.
///
/// This only names fields the schema does not define. Whether an override KEY
/// matches a real model needs the model listing, which this crate does not
/// have.
fn unknown_keys_in_unapplied_overrides(
    layers: &[ConfigLayer],
    model_overrides: &HashMap<String, Value>,
    resolution: &mut ConfigResolution,
) {
    let applied = layers
        .iter()
        .filter(|layer| {
            matches!(
                layer.kind,
                ConfigLayerKind::Model | ConfigLayerKind::ModelHost
            )
        })
        .filter_map(|layer| layer.id.as_deref())
        .collect::<BTreeSet<_>>();

    let known = config_schema()
        .iter()
        .map(|field| field.key)
        .collect::<BTreeSet<_>>();

    let mut tags = model_overrides.keys().collect::<Vec<_>>();
    tags.sort();
    for tag in tags {
        if applied.contains(tag.as_str()) {
            continue;
        }
        let Some(object) = model_overrides[tag].as_object() else {
            continue;
        };
        for key in object.keys() {
            if !known.contains(key.as_str()) {
                resolution.unknown_keys.push(UnknownConfigKey {
                    key: key.clone(),
                    source: ConfigValueSource {
                        kind: ConfigLayerKind::Model,
                        id: Some(tag.clone()),
                    },
                });
            }
        }
    }
}

fn model_overrides_from_documents(raw: &Value, host_local: &Value) -> HashMap<String, Value> {
    let mut overrides = model_overrides_from_single_document(raw);
    for (key, value) in model_overrides_from_single_document(host_local) {
        overrides.insert(key, value);
    }
    overrides
}

fn model_overrides_from_single_document(raw: &Value) -> HashMap<String, Value> {
    raw.get("model_overrides")
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default()
}

pub fn config_value_map(config: &HipfireConfig) -> BTreeMap<String, Value> {
    serde_json::to_value(config)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .map(|object| object.into_iter().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_server_config_values() {
        let cfg = HipfireConfig::default();

        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 11435);
        assert!(cfg.cors_allowed_origins.is_empty());
        assert_eq!(cfg.admin_user, "admin");
        assert_eq!(cfg.api_auth_mode, ApiAuthMode::Auto);
        assert!(!cfg.unsafe_allow_unauthenticated_remote);
        assert_eq!(cfg.max_seq, 8192);
        assert_eq!(cfg.max_tokens, 512);
        assert_eq!(cfg.temperature, 0.3);
        assert_eq!(cfg.top_p, 0.8);
        assert_eq!(cfg.repeat_penalty, 1.05);
        assert!(cfg.resource_lock_enabled);
        assert_eq!(cfg.resource_lock_gpus, vec!["auto".to_string()]);
        assert!(cfg.resource_lock_npus.is_empty());
        assert_eq!(cfg.resource_lock_wait_ms, 0);
        assert_eq!(cfg.scheduler_system_memory_budget_bytes, 0);
        assert_eq!(cfg.scheduler_system_memory_headroom_bytes, 0);
        assert_eq!(cfg.scheduler_vram_budget_bytes, 0);
        assert_eq!(cfg.scheduler_vram_headroom_bytes, 0);
        assert_eq!(cfg.model_residency_mode, "auto");
        assert_eq!(cfg.kv_cache, "auto");
        assert_eq!(cfg.kv_adaptive, "off");
        assert_eq!(cfg.flash_mode, "auto");
        assert_eq!(cfg.dflash_draft, "off");
        assert!(!cfg.spec_adaptive_block);
        assert_eq!(cfg.dflash_no_repeat_ngram, serde_json::json!("auto"));
        assert_eq!(cfg.mtp_mode, "auto");
        assert_eq!(cfg.mtp_k, 3);
        assert_eq!(cfg.thinking, "off");
        assert_eq!(cfg.gpu_slab_load, "auto");
        assert!(cfg.prompt_normalize);
        assert!(cfg.cask_auto_attach);
        assert!(!cfg.cask);
        assert_eq!(cfg.cask_budget, 512);
        assert_eq!(cfg.cask_beta, 128);
        assert_eq!(cfg.cask_core_frac, 0.5);
        assert_eq!(cfg.cask_fold_m, 2);
        assert_eq!(cfg.mmq_screen, "auto");
        assert_eq!(cfg.mmq_screen_threshold, 0.10);
        assert_eq!(cfg.prefill_compression, "off");
        assert_eq!(cfg.prefill_threshold, 32768);
        assert_eq!(cfg.prefill_keep_ratio, 0.05);
        assert_eq!(cfg.prefill_alpha, 0.85);
        assert_eq!(cfg.prefill_min_keep, 2048);
        assert_eq!(cfg.prefill_sink, 256);
        assert_eq!(cfg.prefill_recent, 1024);
        assert_eq!(cfg.prefill_block, 128);
        assert_eq!(cfg.prefill_drafter_device, -1);
        assert!(!cfg.prefill_profile);
        assert_eq!(cfg.prefill_sparse_threshold, 32768);
        assert_eq!(cfg.sdapi_max_dimension, 4096);
        assert_eq!(cfg.sdapi_max_steps, 200);
        assert_eq!(cfg.sdapi_max_batch_size, 8);
        assert_eq!(cfg.sdapi_max_n_iter, 16);
        assert_eq!(cfg.sdapi_max_total_batches, 32);
        assert_eq!(cfg.models_dir, None);
        assert_eq!(cfg.models_network_dir, None);
        assert_eq!(configured_models_dir(&cfg), models_dir());
    }

    #[test]
    fn loaded_config_preserves_sdapi_caps_and_model_dirs() {
        // Regression: these fields must be registered in `config_schema()`, or
        // the schema-driven `from_config` round-trip silently drops any
        // admin-set value and the SD API caps / model roots never take
        // effect. Set non-default values and require they survive.
        let mut cfg = HipfireConfig::default();
        cfg.sdapi_max_dimension = 2048;
        cfg.sdapi_max_steps = 40;
        cfg.sdapi_max_batch_size = 4;
        cfg.sdapi_max_n_iter = 4;
        cfg.sdapi_max_total_batches = 8;
        cfg.api_auth_mode = ApiAuthMode::Required;
        cfg.unsafe_allow_unauthenticated_remote = true;
        cfg.models_dir = Some("/data/hipfire/models".to_string());
        cfg.models_network_dir = Some("/srv/hipfire".to_string());

        let loaded = LoadedConfig::from_config(cfg);
        assert_eq!(loaded.config.sdapi_max_dimension, 2048);
        assert_eq!(loaded.config.sdapi_max_steps, 40);
        assert_eq!(loaded.config.sdapi_max_batch_size, 4);
        assert_eq!(loaded.config.sdapi_max_n_iter, 4);
        assert_eq!(loaded.config.sdapi_max_total_batches, 8);
        assert_eq!(loaded.config.api_auth_mode, ApiAuthMode::Required);
        assert!(loaded.config.unsafe_allow_unauthenticated_remote);
        assert_eq!(
            loaded.config.models_dir.as_deref(),
            Some("/data/hipfire/models")
        );
        assert_eq!(
            loaded.config.models_network_dir.as_deref(),
            Some("/srv/hipfire")
        );
        assert_eq!(
            configured_models_dir(&loaded.config),
            PathBuf::from("/data/hipfire/models")
        );
    }

    #[test]
    fn admin_password_hash_round_trips() {
        let hash = hash_admin_password("hunter2").expect("hash");
        assert!(hash.starts_with("$argon2"));
        assert!(verify_admin_password("hunter2", &hash));
        assert!(!verify_admin_password("wrong", &hash));
        assert!(!verify_admin_password("hunter2", "not-a-phc-string"));
    }

    #[test]
    fn admin_secret_compare_is_exact_and_rejects_empty() {
        assert!(verify_admin_secret("abc123", "abc123"));
        assert!(!verify_admin_secret("abc123", "abc124"));
        assert!(!verify_admin_secret("abc", "abc123"));
        assert!(!verify_admin_secret("", ""));
        assert!(!verify_admin_secret("anything", ""));
    }

    #[test]
    fn schema_defaults_materialize_to_typed_defaults() {
        let resolved = resolve_typed_config_document(&serde_json::json!({}), None);

        assert!(resolved.diagnostics.is_empty());
        assert_eq!(
            config_value_map(&resolved.config),
            config_value_map(&HipfireConfig::default())
        );
    }

    #[test]
    fn loaded_config_preserves_raw_model_overrides() {
        let loaded = loaded_config_from_document(
            PathBuf::from("/tmp/config.json"),
            serde_json::json!({
                "temperature": 0.4,
                "model_overrides": {
                    "qwen": {
                        "temperature": 0.1,
                        "max_tokens": 64
                    }
                }
            }),
            None,
            Vec::new(),
        );

        assert_eq!(loaded.config.temperature, 0.4);
        assert!(loaded.config.model_overrides.contains_key("qwen"));

        let resolved = loaded.resolve_for_model("qwen");
        assert_eq!(resolved.config.temperature, 0.1);
        assert_eq!(resolved.config.max_tokens, 64);
        assert!(resolved.config.model_overrides.contains_key("qwen"));
    }

    #[test]
    fn host_local_config_overrides_global_config() {
        let loaded = loaded_config_from_documents(
            PathBuf::from("/tmp/config.json"),
            serde_json::json!({
                "temperature": 0.4,
                "max_tokens": 512
            }),
            None,
            PathBuf::from("/tmp/config.local.json"),
            serde_json::json!({
                "temperature": 0.2
            }),
            None,
            Vec::new(),
        );

        assert_eq!(loaded.config.temperature, 0.2);
        assert_eq!(loaded.config.max_tokens, 512);
        let temperature = loaded
            .resolution
            .values
            .iter()
            .find(|value| value.key == "temperature")
            .expect("temperature");
        assert_eq!(
            temperature.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::Host)
        );
    }

    #[test]
    fn host_local_model_overrides_win_over_global_model_overrides() {
        let loaded = loaded_config_from_documents(
            PathBuf::from("/tmp/config.json"),
            serde_json::json!({
                "temperature": 0.4,
                "model_overrides": {
                    "qwen": {
                        "temperature": 0.2,
                        "max_tokens": 128
                    }
                }
            }),
            None,
            PathBuf::from("/tmp/config.local.json"),
            serde_json::json!({
                "temperature": 0.3,
                "model_overrides": {
                    "qwen": {
                        "temperature": 0.1
                    }
                }
            }),
            None,
            Vec::new(),
        );

        let resolved = loaded.resolve_for_model("qwen");
        assert_eq!(resolved.config.temperature, 0.1);
        assert_eq!(resolved.config.max_tokens, 128);
        let temperature = resolved
            .resolution
            .values
            .iter()
            .find(|value| value.key == "temperature")
            .expect("temperature");
        assert_eq!(
            temperature.source.as_ref().map(|source| source.kind),
            Some(ConfigLayerKind::ModelHost)
        );
    }

    #[test]
    fn model_overrides_preserve_typed_merge_policy() {
        let mut cfg = HipfireConfig {
            temperature: 0.3,
            max_tokens: 512,
            ..Default::default()
        };
        cfg.model_overrides.insert(
            "qwen".to_string(),
            serde_json::json!({
                "temperature": 0.1,
                "top_p": 0.7,
                "max_tokens": 64,
                "kv_cache": "q8",
                "kv_adaptive": "balanced",
                "dflash_ngram_block": true,
                "cask": true,
                "cask_budget": 1024,
                "prefill_compression": "auto",
                "prefill_drafter_device": 1,
                "unknown": "ignored"
            }),
        );

        let resolved = cfg.resolve_for_model("qwen");
        assert_eq!(resolved.temperature, 0.1);
        assert_eq!(resolved.top_p, 0.7);
        assert_eq!(resolved.max_tokens, 64);
        assert_eq!(resolved.kv_cache, "q8");
        assert_eq!(resolved.kv_adaptive, "balanced");
        assert_eq!(resolved.dflash_no_repeat_ngram, serde_json::json!(true));
        assert!(resolved.cask);
        assert_eq!(resolved.cask_budget, 1024);
        assert_eq!(resolved.prefill_compression, "auto");
        assert_eq!(resolved.prefill_drafter_device, 1);
        assert_eq!(cfg.resolve_for_model("other").temperature, 0.3);
    }
    /// One malformed key must not discard the rest of the config.
    ///
    /// Before this, `materialize_config` returned `HipfireConfig::default()` on
    /// any deserialization failure, so a single wrong-typed value silently reset
    /// EVERY setting. The live trigger was `HIPFIRE_RESOURCE_LOCK_NPUS=1`, which
    /// `hipfire-server` sets on itself for `resource_lock_npus: ["auto"]` — valid
    /// for the daemon adapter that reads it, wrong type for the `Vec<String>`
    /// config field of the same name. A server with NPU auto-leasing ran on
    /// defaults for everything, silently.
    #[test]
    fn a_malformed_key_does_not_discard_the_others() {
        let raw = serde_json::json!({
            "prewarm_priority": 7,
            "kv_cache": "kvarn",
            // Wrong type on purpose: this field is a Vec<String>.
            "resource_lock_npus": 1,
        });

        let resolved = resolve_typed_config_document(&raw, None);

        assert_eq!(
            resolved.config.prewarm_priority, 7,
            "a good key was discarded because a different key was malformed"
        );
        assert_eq!(
            resolved.config.kv_cache, "kvarn",
            "a good key was discarded because a different key was malformed"
        );
        assert_eq!(
            resolved.config.resource_lock_npus,
            default_resource_lock_npus(),
            "the malformed key should fall back to its own default"
        );
        assert!(
            resolved
                .diagnostics
                .iter()
                .any(|d| d.message.contains("resource_lock_npus")),
            "the malformed key must be NAMED — being unnamed is what made this \
             invisible for so long"
        );
    }

    #[test]
    fn a_union_field_accepts_every_arm() {
        for raw in ["", "ram", "none", "off", "/var/lib/hipfire/ngram"] {
            let resolved = resolve_typed_config_document(
                &serde_json::json!({ "ngram_spec_store_root": raw }),
                None,
            );
            assert_eq!(resolved.config.ngram_spec_store_root, raw);
            assert!(
                !resolved
                    .diagnostics
                    .iter()
                    .any(|d| d.message.contains("ngram_spec_store_root")),
                "`{raw}` is a valid arm but was reported: {:?}",
                resolved.diagnostics
            );
        }
    }

    #[test]
    fn a_dflash_path_implies_on() {
        assert_eq!(dflash_draft_setting("off"), ("off", None));
        assert_eq!(dflash_draft_setting(""), ("off", None));
        assert_eq!(dflash_draft_setting("auto"), ("auto", None));
        assert_eq!(dflash_draft_setting("on"), ("on", None));
        // Naming a drafter is asking for it, so the mode follows the path.
        assert_eq!(
            dflash_draft_setting("/drafts/x.hfq"),
            ("on", Some("/drafts/x.hfq"))
        );
    }

    #[test]
    fn dflash_mode_migrates_onto_dflash_draft() {
        // The two were one question asked twice; the old key's values carry
        // over unchanged rather than being dropped.
        for value in ["off", "auto", "on"] {
            let raw = serde_json::json!({ "dflash_mode": value });
            let resolved = resolve_typed_config_document(&raw, None);
            assert_eq!(
                resolved.config.dflash_draft, value,
                "dflash_mode={value} must carry over"
            );
            assert!(
                resolved.diagnostics.iter().any(
                    |d| d.message.contains("dflash_mode") && d.message.contains("dflash_draft")
                ),
                "the migration must name both keys: {:?}",
                resolved.diagnostics
            );
        }
    }

    #[test]
    fn a_dflash_draft_union_accepts_both_arms() {
        for value in ["off", "auto", "on", "/etc/hostname"] {
            let raw = serde_json::json!({ "dflash_draft": value });
            let resolved = resolve_typed_config_document(&raw, None);
            assert!(
                !resolved
                    .diagnostics
                    .iter()
                    .any(|d| d.message.contains("dflash_draft")),
                "`{value}` is a valid arm but was reported: {:?}",
                resolved.diagnostics
            );
        }
        // A relative path is neither a keyword nor an absolute path.
        let raw = serde_json::json!({ "dflash_draft": "drafts/x.hfq" });
        let resolved = resolve_typed_config_document(&raw, None);
        let msg = resolved
            .diagnostics
            .iter()
            .find(|d| d.message.contains("dflash_draft"))
            .map(|d| d.message.clone())
            .expect("a value matching no arm must be reported");
        assert!(
            msg.contains("off") && msg.contains("absolute"),
            "the report must name both arms' expectations: {msg}"
        );
    }

    #[test]
    fn a_renamed_key_is_honoured_and_reported() {
        let raw = serde_json::json!({ "ngram_store_root": "/tmp", "ngram_orders": "4,3,2" });
        let resolved = resolve_typed_config_document(&raw, None);
        assert_eq!(
            resolved.config.ngram_spec_store_root, "/tmp",
            "the old key's value must still apply — dropping it silently is the bug"
        );
        assert_eq!(resolved.config.ngram_spec_orders, "4,3,2");
        for (old, new) in [
            ("ngram_store_root", "ngram_spec_store_root"),
            ("ngram_orders", "ngram_spec_orders"),
        ] {
            assert!(
                resolved
                    .diagnostics
                    .iter()
                    .any(|d| d.message.contains(old) && d.message.contains(new)),
                "the rename must name BOTH the old and new key: {:?}",
                resolved.diagnostics
            );
        }
        assert!(
            resolved.resolution.unknown_keys.is_empty(),
            "a renamed key is not an unknown key: {:?}",
            resolved.resolution.unknown_keys
        );
    }

    #[test]
    fn the_new_key_wins_when_both_are_set() {
        let raw = serde_json::json!({
            "ngram_store_root": "/old", "ngram_spec_store_root": "/new",
        });
        let resolved = resolve_typed_config_document(&raw, None);
        assert_eq!(resolved.config.ngram_spec_store_root, "/new");
        assert!(
            resolved
                .diagnostics
                .iter()
                .any(|d| d.message.contains("both are set")),
            "a disagreement between the two names must be reported, not guessed"
        );
    }

    #[test]
    fn a_missing_input_path_is_reported_and_a_present_one_is_not() {
        let mut config = HipfireConfig::default();
        config.cask_sidecar = Some("/definitely/not/here/x.hfq".to_string());
        assert!(
            path_existence_diagnostics(&config)
                .iter()
                .any(|d| d.message.contains("cask_sidecar") && d.message.contains("does not exist")),
            "a missing input file must be reported"
        );

        config.cask_sidecar = Some("/etc/hostname".to_string());
        assert!(
            !path_existence_diagnostics(&config)
                .iter()
                .any(|d| d.message.contains("cask_sidecar")),
            "a present input file must be silent"
        );
    }

    #[test]
    fn an_output_path_needs_only_its_parent() {
        let mut config = HipfireConfig::default();
        // Created on first use, parent exists: silent.
        config.ngram_spec_store_root = "/tmp/hipfire-ngram-does-not-exist-yet".to_string();
        assert!(
            path_existence_diagnostics(&config).is_empty(),
            "an output path whose parent exists must not be reported: {:?}",
            path_existence_diagnostics(&config)
        );

        // Parent missing: nothing here builds a directory tree, so warn.
        config.ngram_spec_store_root = "/definitely/not/here/ngram".to_string();
        assert!(
            path_existence_diagnostics(&config)
                .iter()
                .any(|d| d.message.contains("parent directory")),
            "an output path with no parent directory must be reported"
        );
    }

    #[test]
    fn a_ram_sentinel_is_never_treated_as_a_path() {
        // The union arm decides: "ram" took the sentinel arm, so no I/O check
        // applies to it however unlike a path it looks.
        for sentinel in NGRAM_STORE_ROOT_RAM {
            let mut config = HipfireConfig::default();
            config.ngram_spec_store_root = sentinel.to_string();
            assert!(
                path_existence_diagnostics(&config).is_empty(),
                "sentinel {sentinel:?} was checked as a path"
            );
        }
    }

    #[test]
    fn a_relative_path_is_reported() {
        // The point of absolute-only: a sentinel typo is a legal relative path,
        // so until this rule `rma` resolved silently as a directory.
        for raw in ["rma", "./tables", "../ngram", "~/ngram", "tables"] {
            let resolved = resolve_typed_config_document(
                &serde_json::json!({ "ngram_spec_store_root": raw }),
                None,
            );
            let message = resolved
                .diagnostics
                .iter()
                .find(|d| d.message.contains("ngram_spec_store_root"))
                .map(|d| d.message.clone())
                .unwrap_or_else(|| panic!("`{raw}` is relative and must be reported"));
            assert!(
                message.contains("absolute"),
                "the report must say what is wrong: {message}"
            );
            assert_eq!(
                resolved.config.ngram_spec_store_root, raw,
                "reporting must not change what resolves"
            );
        }
    }

    #[test]
    fn a_union_reports_every_arms_expectation_not_just_the_last() {
        // A NUL byte fails the sentinel arm AND the path arm, so neither
        // accepts and the operator should see both domains, not "want a path".
        let raw = serde_json::json!({ "ngram_spec_store_root": "bad\0path" });
        let resolved = resolve_typed_config_document(&raw, None);
        let message = resolved
            .diagnostics
            .iter()
            .find(|d| d.message.contains("ngram_spec_store_root"))
            .map(|d| d.message.clone())
            .expect("a value matching no arm must be reported");
        assert!(
            message.contains("ram") && message.contains("path"),
            "the report must name every arm's expectation: {message}"
        );
    }

    #[test]
    fn a_bad_enum_value_from_a_file_is_reported() {
        // The gap this closes: resolve_field takes file values verbatim, so
        // `kvarnn` used to reach the KV match as an unrecognized mode with
        // nothing said at config level.
        let raw = serde_json::json!({ "kv_cache": "kvarnn" });
        let resolved = resolve_typed_config_document(&raw, None);
        assert!(
            resolved
                .diagnostics
                .iter()
                .any(|d| d.message.contains("kv_cache")),
            "an out-of-domain enum value went unreported: {:?}",
            resolved.diagnostics
        );
        assert_eq!(
            resolved.config.kv_cache, "kvarnn",
            "reporting must not change what resolves — this warns, never rejects"
        );
    }

    #[test]
    fn a_deprecated_kv_mode_is_reported_before_the_loader_refuses_it() {
        // Issue #386: `q8` passed the enum, the server reported healthy, and the
        // loader's refusal was the first anyone heard of it.
        for mode in ["q8", "asym2", "asym3", "asym4"] {
            let raw = serde_json::json!({ "kv_cache": mode });
            let resolved = resolve_typed_config_document(&raw, None);
            let message = resolved
                .diagnostics
                .iter()
                .find(|d| d.message.contains("DEPRECATED"))
                .map(|d| d.message.clone())
                .unwrap_or_else(|| panic!("{mode} resolved silently: {:?}", resolved.diagnostics));
            assert!(
                message.contains("HIPFIRE_KV_ALLOW_DEPRECATED"),
                "the warning must carry the migration escape hatch: {message}"
            );
            assert_eq!(
                resolved.config.kv_cache, mode,
                "reporting must not change what resolves — this warns, never rejects"
            );
        }
        // A supported mode stays silent.
        let ok = resolve_typed_config_document(&serde_json::json!({ "kv_cache": "kvarn" }), None);
        assert!(
            ok.diagnostics.is_empty(),
            "a supported kv_cache must not warn: {:?}",
            ok.diagnostics
        );
    }

    #[test]
    fn a_compiled_default_is_never_reported() {
        let resolved = resolve_typed_config_document(&serde_json::json!({}), None);
        assert!(
            resolved.diagnostics.is_empty(),
            "an untouched config must be silent: {:?}",
            resolved.diagnostics
        );
    }

    #[test]
    fn a_typo_in_an_unmatched_model_override_is_still_reported() {
        // Two failures that used to be silent together: the entry's KEY names no
        // model being resolved, so it contributes no layer — and because it
        // contributes no layer, the misspelled FIELD inside it was never
        // schema-checked either.
        let raw = serde_json::json!({
            "model_overrides": {
                "MiniCPM5-1B.oq4.25++": { "kv_cach": "q8", "thinking": "on" },
            },
        });

        let resolved = resolve_typed_config_document(&raw, Some("MiniCPM5--1B.oq4.25++"));

        let unknown = &resolved.resolution.unknown_keys;
        assert!(
            unknown.iter().any(|k| k.key == "kv_cach"),
            "the misspelled field in an unapplied override went unreported: {unknown:?}"
        );
        assert!(
            unknown
                .iter()
                .any(|k| k.key == "kv_cach"
                    && k.source.id.as_deref() == Some("MiniCPM5-1B.oq4.25++")),
            "the report must name WHICH override entry it came from: {unknown:?}"
        );
        assert!(
            !unknown.iter().any(|k| k.key == "thinking"),
            "`thinking` is a real schema field and must not be reported: {unknown:?}"
        );
    }

    #[test]
    fn an_applied_model_override_is_not_reported_twice() {
        let raw = serde_json::json!({
            "model_overrides": {
                "some-model": { "kv_cach": "q8" },
            },
        });

        let resolved = resolve_typed_config_document(&raw, Some("some-model"));

        assert_eq!(
            resolved
                .resolution
                .unknown_keys
                .iter()
                .filter(|k| k.key == "kv_cach")
                .count(),
            1,
            "the matched override is checked as a layer; checking it again double-reports"
        );
    }
}
