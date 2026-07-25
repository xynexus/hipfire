// SPDX-License-Identifier: Apache-2.0
// hipfire — family-neutral calibration job contracts.

//! Pure-data contracts shared by resident and layer-streamed calibration.
//!
//! This module deliberately contains no model-family types and no GPU work. It
//! owns the deterministic sample geometry, logical capture identities, routed
//! expert accounting, coverage admission, and canonical KLDREF packing that
//! arch adapters feed into.

use crate::hfq::HfqMemTensor;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

pub const CALIBRATION_JOB_SCHEMA_VERSION: u32 = 2;
pub const DEFAULT_MIN_EXPERT_ACTIVATIONS: u64 = 2_048;
pub const DEFAULT_EXPERT_CAPTURE_TARGET: u64 = 4_096;
pub const DEFAULT_EXPERT_CAPTURE_TILE_ROWS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibError {
    InvalidOptions(String),
    InvalidSamples(String),
    DuplicateCaptureId(CaptureId),
    DuplicateOutputName(String),
    InvalidCapture(String),
    InvalidRouting(String),
    ExpertCoverage(String),
    InvalidKldRef(String),
    InvalidSourcePlan(String),
    ReadLedger(String),
    Boundary(String),
    Checkpoint(String),
    Runtime(String),
}

impl fmt::Display for CalibError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(message) => write!(f, "invalid calibration options: {message}"),
            Self::InvalidSamples(message) => write!(f, "invalid calibration samples: {message}"),
            Self::DuplicateCaptureId(id) => write!(f, "duplicate capture id {}", id.0),
            Self::DuplicateOutputName(name) => write!(f, "duplicate capture output name {name}"),
            Self::InvalidCapture(message) => write!(f, "invalid capture: {message}"),
            Self::InvalidRouting(message) => write!(f, "invalid expert routing: {message}"),
            Self::ExpertCoverage(message) => write!(f, "expert coverage failed: {message}"),
            Self::InvalidKldRef(message) => write!(f, "invalid KLDREF: {message}"),
            Self::InvalidSourcePlan(message) => write!(f, "invalid tensor source plan: {message}"),
            Self::ReadLedger(message) => write!(f, "tensor read ledger error: {message}"),
            Self::Boundary(message) => write!(f, "calibration boundary error: {message}"),
            Self::Checkpoint(message) => write!(f, "calibration checkpoint error: {message}"),
            Self::Runtime(message) => write!(f, "calibration runtime error: {message}"),
        }
    }
}

impl std::error::Error for CalibError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationSample {
    pub id: String,
    pub tokens: Vec<u32>,
    pub stratum: String,
}

impl CalibrationSample {
    pub fn new(id: impl Into<String>, tokens: Vec<u32>, stratum: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            tokens,
            stratum: stratum.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplePosition {
    pub sample_index: usize,
    pub position: usize,
}

impl SamplePosition {
    pub const fn new(sample_index: usize, position: usize) -> Self {
        Self {
            sample_index,
            position,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleRow {
    pub sample_index: usize,
    pub position: usize,
    pub token: u32,
    pub reset_state: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleSet {
    samples: Vec<CalibrationSample>,
    context_len: usize,
    sampling_seed: u64,
    fingerprint: String,
}

impl SampleSet {
    pub fn new(
        mut samples: Vec<CalibrationSample>,
        context_len: usize,
        sampling_seed: u64,
    ) -> Result<Self, CalibError> {
        if context_len == 0 {
            return Err(CalibError::InvalidSamples(
                "context length must be greater than zero".into(),
            ));
        }
        if samples.is_empty() {
            return Err(CalibError::InvalidSamples(
                "at least one independent sample is required".into(),
            ));
        }
        let mut ids = HashSet::with_capacity(samples.len());
        for sample in &samples {
            if sample.id.is_empty() {
                return Err(CalibError::InvalidSamples(
                    "sample id must not be empty".into(),
                ));
            }
            if !ids.insert(sample.id.clone()) {
                return Err(CalibError::InvalidSamples(format!(
                    "duplicate sample id {}",
                    sample.id
                )));
            }
            if sample.tokens.is_empty() {
                return Err(CalibError::InvalidSamples(format!(
                    "sample {} has no tokens",
                    sample.id
                )));
            }
            if sample.tokens.len() > context_len {
                return Err(CalibError::InvalidSamples(format!(
                    "sample {} length {} exceeds context length {context_len}",
                    sample.id,
                    sample.tokens.len()
                )));
            }
        }

        samples.sort_by(|a, b| {
            sample_order_key(sampling_seed, a)
                .cmp(&sample_order_key(sampling_seed, b))
                .then_with(|| a.id.cmp(&b.id))
        });
        let fingerprint = sample_set_fingerprint(&samples, context_len, sampling_seed);
        Ok(Self {
            samples,
            context_len,
            sampling_seed,
            fingerprint,
        })
    }

    pub fn samples(&self) -> &[CalibrationSample] {
        &self.samples
    }

    pub const fn context_len(&self) -> usize {
        self.context_len
    }

    pub const fn sampling_seed(&self) -> u64 {
        self.sampling_seed
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn total_rows(&self) -> usize {
        self.samples.iter().map(|sample| sample.tokens.len()).sum()
    }

    /// Time-major rows preserve monotonic position within every independent
    /// sample while making the state reset at position zero explicit.
    pub fn rows_time_major(&self) -> Vec<SampleRow> {
        let max_len = self
            .samples
            .iter()
            .map(|sample| sample.tokens.len())
            .max()
            .unwrap_or(0);
        let mut rows = Vec::with_capacity(self.total_rows());
        for position in 0..max_len {
            for (sample_index, sample) in self.samples.iter().enumerate() {
                if let Some(&token) = sample.tokens.get(position) {
                    rows.push(SampleRow {
                        sample_index,
                        position,
                        token,
                        reset_state: position == 0,
                    });
                }
            }
        }
        rows
    }
}

fn stable_hash(mut state: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        state ^= byte as u64;
        state = state.wrapping_mul(0x100_0000_01b3);
    }
    state
}

fn sample_order_key(seed: u64, sample: &CalibrationSample) -> u64 {
    let mut state = stable_hash(0xcbf2_9ce4_8422_2325 ^ seed, sample.id.as_bytes());
    state = stable_hash(state, sample.stratum.as_bytes());
    for token in &sample.tokens {
        state = stable_hash(state, &token.to_le_bytes());
    }
    state
}

fn sample_set_fingerprint(samples: &[CalibrationSample], context_len: usize, seed: u64) -> String {
    let mut state = stable_hash(0xcbf2_9ce4_8422_2325, &seed.to_le_bytes());
    state = stable_hash(state, &(context_len as u64).to_le_bytes());
    for sample in samples {
        state = stable_hash(state, sample.id.as_bytes());
        state = stable_hash(state, &[0]);
        state = stable_hash(state, sample.stratum.as_bytes());
        state = stable_hash(state, &[0]);
        for token in &sample.tokens {
            state = stable_hash(state, &token.to_le_bytes());
        }
    }
    format!("fnv1a64:{state:016x}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionRole {
    QueryInput,
    KeyInput,
    ValueInput,
    AttentionOutputInput,
    GateUpInput,
    DownInput,
    RouterInput,
    SharedExpertInput,
    DenseMlpInput,
    LmHeadInput,
    Other(u16),
}

impl ProjectionRole {
    const fn code(self) -> u16 {
        match self {
            Self::QueryInput => 1,
            Self::KeyInput => 2,
            Self::ValueInput => 3,
            Self::AttentionOutputInput => 4,
            Self::GateUpInput => 5,
            Self::DownInput => 6,
            Self::RouterInput => 7,
            Self::SharedExpertInput => 8,
            Self::DenseMlpInput => 9,
            Self::LmHeadInput => 10,
            Self::Other(code) => code,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CaptureId(pub u64);

impl CaptureId {
    pub fn new(layer: usize, role: ProjectionRole, expert: Option<usize>) -> Self {
        let expert_code = expert.map(|value| value as u64 + 1).unwrap_or(0);
        Self(((layer as u64) << 40) | (expert_code << 16) | role.code() as u64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapturePolicy {
    HessianAndImatrix,
    ImatrixOnly,
    Skip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExpertSamplingPolicy {
    DeterministicFirst { seed: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertCaptureQuota {
    pub min_rows: u64,
    pub target_rows: u64,
    pub tile_rows: usize,
    pub sampling: ExpertSamplingPolicy,
}

impl Default for ExpertCaptureQuota {
    fn default() -> Self {
        Self {
            min_rows: DEFAULT_MIN_EXPERT_ACTIVATIONS,
            target_rows: DEFAULT_EXPERT_CAPTURE_TARGET,
            tile_rows: DEFAULT_EXPERT_CAPTURE_TILE_ROWS,
            sampling: ExpertSamplingPolicy::DeterministicFirst { seed: 0 },
        }
    }
}

impl ExpertCaptureQuota {
    pub fn limit_rows(&self) -> Result<u64, CalibError> {
        let tile_rows = u64::try_from(self.tile_rows).map_err(|_| {
            CalibError::InvalidOptions("expert capture tile rows exceed u64".into())
        })?;
        if tile_rows == 0 {
            return Err(CalibError::InvalidOptions(
                "expert capture tile rows must be greater than zero".into(),
            ));
        }
        self.target_rows
            .div_ceil(tile_rows)
            .checked_mul(tile_rows)
            .ok_or_else(|| CalibError::InvalidOptions("expert capture limit overflow".into()))
    }

    pub fn validate(&self) -> Result<(), CalibError> {
        if self.min_rows == 0 {
            return Err(CalibError::InvalidOptions(
                "minimum expert activations must be greater than zero".into(),
            ));
        }
        if self.target_rows < self.min_rows {
            return Err(CalibError::InvalidOptions(format!(
                "expert capture target {} is below minimum {}",
                self.target_rows, self.min_rows
            )));
        }
        if self.tile_rows == 0 {
            return Err(CalibError::InvalidOptions(
                "expert capture tile rows must be greater than zero".into(),
            ));
        }
        self.limit_rows()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureDescriptor {
    pub id: CaptureId,
    pub output_names: Vec<String>,
    pub input_width: usize,
    pub policy: CapturePolicy,
    pub layer: usize,
    pub role: ProjectionRole,
    pub expert: Option<usize>,
    pub expert_quota: Option<ExpertCaptureQuota>,
}

impl CaptureDescriptor {
    fn validate(&self) -> Result<(), CalibError> {
        if self.id != CaptureId::new(self.layer, self.role, self.expert) {
            return Err(CalibError::InvalidCapture(format!(
                "capture id {} does not match layer/role/expert fields",
                self.id.0
            )));
        }
        if matches!(self.role, ProjectionRole::Other(code) if code <= 10) {
            return Err(CalibError::InvalidCapture(
                "custom projection role codes must be greater than 10".into(),
            ));
        }
        if self.output_names.is_empty() {
            return Err(CalibError::InvalidCapture(format!(
                "capture {} has no output names",
                self.id.0
            )));
        }
        if self.output_names.iter().any(String::is_empty) {
            return Err(CalibError::InvalidCapture(format!(
                "capture {} has an empty output name",
                self.id.0
            )));
        }
        if self.input_width == 0 {
            return Err(CalibError::InvalidCapture(format!(
                "capture {} has zero input width",
                self.id.0
            )));
        }
        match (self.expert, self.expert_quota) {
            (Some(_), Some(quota)) => quota.validate()?,
            (Some(_), None) => {
                return Err(CalibError::InvalidCapture(format!(
                    "expert capture {} has no quota",
                    self.id.0
                )))
            }
            (None, Some(_)) => {
                return Err(CalibError::InvalidCapture(format!(
                    "dense capture {} unexpectedly has an expert quota",
                    self.id.0
                )))
            }
            (None, None) => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureRegistry {
    descriptors: BTreeMap<CaptureId, CaptureDescriptor>,
    output_to_id: HashMap<String, CaptureId>,
}

impl CaptureRegistry {
    pub fn register(&mut self, descriptor: CaptureDescriptor) -> Result<(), CalibError> {
        descriptor.validate()?;
        if self.descriptors.contains_key(&descriptor.id) {
            return Err(CalibError::DuplicateCaptureId(descriptor.id));
        }
        let mut local_names = HashSet::with_capacity(descriptor.output_names.len());
        for name in &descriptor.output_names {
            if !local_names.insert(name) || self.output_to_id.contains_key(name) {
                return Err(CalibError::DuplicateOutputName(name.clone()));
            }
        }
        for name in &descriptor.output_names {
            self.output_to_id.insert(name.clone(), descriptor.id);
        }
        self.descriptors.insert(descriptor.id, descriptor);
        Ok(())
    }

    pub fn get(&self, id: CaptureId) -> Option<&CaptureDescriptor> {
        self.descriptors.get(&id)
    }

    pub fn resolve_output(&self, name: &str) -> Option<CaptureId> {
        self.output_to_id.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn descriptors(&self) -> impl Iterator<Item = &CaptureDescriptor> {
        self.descriptors.values()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryPrecision {
    F32,
    Bf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExpertCoveragePolicy {
    Strict,
    PreserveUndercovered,
}

/// Shared option surface for resident collectors while the streamed engine
/// uses the larger [`CalibrationOptions`] job contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KldRefOptions {
    pub kldref: bool,
    pub kldref_topk: usize,
}

impl Default for KldRefOptions {
    fn default() -> Self {
        Self {
            kldref: false,
            kldref_topk: 64,
        }
    }
}

impl KldRefOptions {
    pub fn validate(&self) -> Result<(), CalibError> {
        if self.kldref && self.kldref_topk == 0 {
            Err(CalibError::InvalidOptions(
                "KLDREF top-k must be greater than zero".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationOptions {
    pub sequence_batch: Option<usize>,
    pub time_tile: Option<usize>,
    pub max_rows: usize,
    pub boundary_precision: BoundaryPrecision,
    pub expert_quota: ExpertCaptureQuota,
    pub required_expert_fraction: f64,
    pub expert_coverage_policy: ExpertCoveragePolicy,
    pub kldref: bool,
    pub kldref_top_k: usize,
    /// Cap on retained KLDREF positions. `None` keeps every corpus row. A KLD
    /// *reference* is statistically ample over a few thousand positions while the
    /// Hessians need every row, and the reference costs a full-vocabulary lm-head
    /// projection per row — so this is the lever on the dominant KLDREF term.
    pub kldref_rows: Option<usize>,
}

impl Default for CalibrationOptions {
    fn default() -> Self {
        Self {
            sequence_batch: None,
            time_tile: None,
            max_rows: 256,
            boundary_precision: BoundaryPrecision::F32,
            expert_quota: ExpertCaptureQuota::default(),
            required_expert_fraction: 1.0,
            expert_coverage_policy: ExpertCoveragePolicy::Strict,
            kldref: true,
            kldref_top_k: 64,
            kldref_rows: None,
        }
    }
}

impl CalibrationOptions {
    pub fn validate(&self) -> Result<(), CalibError> {
        if self.max_rows == 0 {
            return Err(CalibError::InvalidOptions(
                "maximum microbatch rows must be greater than zero".into(),
            ));
        }
        if self.sequence_batch == Some(0) || self.time_tile == Some(0) {
            return Err(CalibError::InvalidOptions(
                "sequence batch and time tile must be greater than zero when specified".into(),
            ));
        }
        if !self.required_expert_fraction.is_finite()
            || !(0.0 < self.required_expert_fraction && self.required_expert_fraction <= 1.0)
        {
            return Err(CalibError::InvalidOptions(
                "required expert fraction must be finite and in (0, 1]".into(),
            ));
        }
        if self.kldref && self.kldref_rows == Some(0) {
            return Err(CalibError::InvalidOptions(
                "kldref row cap must be nonzero when set".into(),
            ));
        }
        if self.kldref && self.kldref_top_k == 0 {
            return Err(CalibError::InvalidOptions(
                "KLDREF top-k must be greater than zero".into(),
            ));
        }
        self.expert_quota.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationJob {
    pub schema_version: u32,
    pub source_fingerprint: String,
    pub tokenizer_fingerprint: String,
    pub corpus_fingerprint: String,
    pub samples: SampleSet,
    pub options: CalibrationOptions,
}

impl CalibrationJob {
    pub fn new(
        source_fingerprint: impl Into<String>,
        tokenizer_fingerprint: impl Into<String>,
        samples: SampleSet,
        options: CalibrationOptions,
    ) -> Result<Self, CalibError> {
        options.validate()?;
        let source_fingerprint = source_fingerprint.into();
        let tokenizer_fingerprint = tokenizer_fingerprint.into();
        if source_fingerprint.is_empty() || tokenizer_fingerprint.is_empty() {
            return Err(CalibError::InvalidOptions(
                "source and tokenizer fingerprints must not be empty".into(),
            ));
        }
        let corpus_fingerprint = samples.fingerprint().to_string();
        Ok(Self {
            schema_version: CALIBRATION_JOB_SCHEMA_VERSION,
            source_fingerprint,
            tokenizer_fingerprint,
            corpus_fingerprint,
            samples,
            options,
        })
    }

    pub fn with_corpus_fingerprint(
        mut self,
        corpus_fingerprint: impl Into<String>,
    ) -> Result<Self, CalibError> {
        let corpus_fingerprint = corpus_fingerprint.into();
        if corpus_fingerprint.is_empty() {
            return Err(CalibError::InvalidOptions(
                "corpus fingerprint must not be empty".into(),
            ));
        }
        self.corpus_fingerprint = corpus_fingerprint;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertCaptureRole {
    GateUpInput,
    DownInput,
}

impl fmt::Display for ExpertCaptureRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GateUpInput => f.write_str("gate_up_input"),
            Self::DownInput => f.write_str("down_input"),
        }
    }
}

const EXPERT_CAPTURE_ROLES: [ExpertCaptureRole; 2] =
    [ExpertCaptureRole::GateUpInput, ExpertCaptureRole::DownInput];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayerExpert {
    pub layer: usize,
    pub expert: usize,
}

impl LayerExpert {
    pub const fn new(layer: usize, expert: usize) -> Self {
        Self { layer, expert }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct WeightStats {
    pub count: u64,
    pub sum: f64,
    pub sum_squared: f64,
}

impl WeightStats {
    fn record(&mut self, weight: f32) {
        let weight = weight as f64;
        self.count += 1;
        self.sum += weight;
        self.sum_squared += weight * weight;
    }

    pub fn effective_sample_size(&self) -> f64 {
        if self.sum_squared == 0.0 {
            0.0
        } else {
            self.sum * self.sum / self.sum_squared
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ExpertCaptureStats {
    pub seen_rows: u64,
    pub admitted_rows: u64,
    pub batch_slack_rows: u64,
    pub quota_skipped_rows: u64,
    /// Number of row-gather launches used to fill persistent reduction tiles.
    pub capture_gather_launches: u64,
    /// Distinguishes legacy/resident telemetry from the streamed tile path.
    pub launch_telemetry_recorded: bool,
    /// Complete reduction tiles launched during normal microbatch processing.
    pub full_reduction_tiles: u64,
    /// Final nonempty partial tiles launched at corpus exhaustion.
    pub partial_reduction_tiles: u64,
    pub full_weight: WeightStats,
    pub admitted_weight: WeightStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureAdmission {
    Capture,
    TelemetryOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerRouterStats {
    pub routed_tokens: u64,
    pub routed_slots: u64,
    pub dropped_indices: u64,
    pub microbatches: u64,
    pub active_expert_sum: u64,
    pub max_active_experts: usize,
    pub padded_routed_rows: u64,
    pub saturated_after_routed_tokens: Option<u64>,
    pub top1_hits: Vec<u64>,
    pub topk_hits: Vec<u64>,
    pub route_weights: Vec<WeightStats>,
    pub cooccurrence: BTreeMap<u64, u64>,
    /// Per-expert routed-token histogram, `token id -> count`. Empty unless the
    /// family adapter supplies the token with each routing decision (a shape-only
    /// routing plan cannot). Answers *what* an expert specialises in rather than
    /// only how much load it takes — a starved expert whose tokens are all digits
    /// or CJK is a corpus gap, not a dead expert.
    ///
    /// Truncated to the top [`TOKEN_PROFILE_KEEP`] ids per expert when a layer is
    /// snapshotted; `token_profile_dropped` records how many distinct ids that
    /// discarded so a report never implies it saw the whole tail.
    #[serde(default)]
    pub token_counts: Vec<BTreeMap<u32, u64>>,
    #[serde(default)]
    pub token_profile_dropped: Vec<u64>,
    /// Per-expert routed-*sample-stratum* histogram, `stratum -> count`. Token
    /// identity is the right lens for lexically-driven routing but says little
    /// where routing is semantic: the middle third of a decoder can be
    /// language-universal while still specialising by domain. This counts the
    /// label of the sample each routed token came from, so an expert can be
    /// characterised by *what kind of document* reaches it.
    ///
    /// Empty unless the corpus supplies per-sample strata (see
    /// [`CalibrationSample::stratum`]) — a single-stratum corpus records one
    /// bucket, which the report treats as no signal rather than a finding.
    #[serde(default)]
    pub stratum_counts: Vec<BTreeMap<String, u64>>,
}

/// Distinct token ids kept per expert when persisting a token profile. The live
/// accumulator is exact; only the snapshot is bounded.
pub const TOKEN_PROFILE_KEEP: usize = 256;

/// Where a routed row came from, for the per-expert specialisation profiles.
/// Every field is optional because not every routing seam knows them: the
/// grouped-MoE dispatch callback sees only indices and weights, while a
/// per-token adapter knows both the corpus token and its sample's stratum.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoutedRowContext<'a> {
    pub token: Option<u32>,
    pub stratum: Option<&'a str>,
}

impl<'a> RoutedRowContext<'a> {
    /// No provenance — load and gate statistics only.
    pub const fn unknown() -> Self {
        Self {
            token: None,
            stratum: None,
        }
    }

    pub const fn new(token: u32, stratum: &'a str) -> Self {
        Self {
            token: Some(token),
            stratum: Some(stratum),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertLayerTelemetry {
    pub layer: usize,
    pub num_experts: usize,
    pub k_top: usize,
    pub quota: ExpertCaptureQuota,
    pub router: LayerRouterStats,
    pub gate_up: Vec<ExpertCaptureStats>,
    pub down: Vec<ExpertCaptureStats>,
}

impl ExpertLayerTelemetry {
    /// Validate a serialized per-layer telemetry snapshot without requiring
    /// the live [`ExpertTelemetry`] accumulator that produced it.
    ///
    /// This is intentionally family-neutral so offline artifact tooling can
    /// reject malformed routing/capture accounting before a quantizer trusts
    /// expert coverage or high-precision fallback metadata.
    pub fn reconcile(&self) -> Result<(), CalibError> {
        self.quota.validate()?;
        if self.num_experts == 0 || self.k_top == 0 || self.k_top > self.num_experts {
            return Err(CalibError::InvalidRouting(format!(
                "layer {} has invalid expert geometry: experts={}, K-top={}",
                self.layer, self.num_experts, self.k_top
            )));
        }
        for (name, actual) in [
            ("router top-1", self.router.top1_hits.len()),
            ("router top-k", self.router.topk_hits.len()),
            ("router weights", self.router.route_weights.len()),
            ("gate-up capture", self.gate_up.len()),
            ("down capture", self.down.len()),
        ] {
            if actual != self.num_experts {
                return Err(CalibError::InvalidRouting(format!(
                    "layer {} {name} length {actual} differs from expert count {}",
                    self.layer, self.num_experts
                )));
            }
        }

        let expected_slots = self
            .router
            .routed_tokens
            .checked_mul(self.k_top as u64)
            .ok_or_else(|| {
                CalibError::InvalidRouting(format!(
                    "layer {} routed slot count overflows u64",
                    self.layer
                ))
            })?;
        if self
            .router
            .routed_slots
            .saturating_add(self.router.dropped_indices)
            != expected_slots
        {
            return Err(CalibError::InvalidRouting(format!(
                "layer {} valid slots {} + dropped indices {} != routed tokens {} * K-top {}",
                self.layer,
                self.router.routed_slots,
                self.router.dropped_indices,
                self.router.routed_tokens,
                self.k_top
            )));
        }
        let recorded_slots = self.router.topk_hits.iter().try_fold(0u64, |sum, hits| {
            sum.checked_add(*hits).ok_or_else(|| {
                CalibError::InvalidRouting(format!(
                    "layer {} top-k hit count overflows u64",
                    self.layer
                ))
            })
        })?;
        if recorded_slots != self.router.routed_slots {
            return Err(CalibError::InvalidRouting(format!(
                "layer {} top-k hits sum to {recorded_slots}, expected {} routed slots",
                self.layer, self.router.routed_slots
            )));
        }
        let top1_hits = self.router.top1_hits.iter().try_fold(0u64, |sum, hits| {
            sum.checked_add(*hits).ok_or_else(|| {
                CalibError::InvalidRouting(format!(
                    "layer {} top-1 hit count overflows u64",
                    self.layer
                ))
            })
        })?;
        if top1_hits > self.router.routed_tokens {
            return Err(CalibError::InvalidRouting(format!(
                "layer {} top-1 hits {top1_hits} exceed routed tokens {}",
                self.layer, self.router.routed_tokens
            )));
        }
        if self.router.max_active_experts > self.num_experts
            || self.router.active_expert_sum
                > self
                    .router
                    .microbatches
                    .saturating_mul(self.num_experts as u64)
        {
            return Err(CalibError::InvalidRouting(format!(
                "layer {} grouped-batch active-expert accounting is invalid",
                self.layer
            )));
        }
        if self
            .router
            .saturated_after_routed_tokens
            .is_some_and(|tokens| tokens > self.router.routed_tokens)
        {
            return Err(CalibError::InvalidRouting(format!(
                "layer {} saturation point exceeds routed token count",
                self.layer
            )));
        }

        let validate_weights =
            |label: &str, expected_count: u64, weights: &WeightStats| -> Result<(), CalibError> {
                if weights.count != expected_count
                    || !weights.sum.is_finite()
                    || !weights.sum_squared.is_finite()
                    || weights.sum_squared < 0.0
                {
                    return Err(CalibError::InvalidRouting(format!(
                        "layer {} {label} weight stats count/values are invalid",
                        self.layer
                    )));
                }
                Ok(())
            };
        let limit_rows = self.quota.limit_rows()?;
        for expert in 0..self.num_experts {
            let expected = self.router.topk_hits[expert];
            validate_weights(
                &format!("expert {expert} router"),
                expected,
                &self.router.route_weights[expert],
            )?;
            for (role, stats) in [
                (ExpertCaptureRole::GateUpInput, &self.gate_up[expert]),
                (ExpertCaptureRole::DownInput, &self.down[expert]),
            ] {
                if stats.seen_rows != expected {
                    return Err(CalibError::InvalidRouting(format!(
                        "layer {} expert {expert} {role}: saw {} capture rows but router recorded {expected}",
                        self.layer, stats.seen_rows
                    )));
                }
                if stats.admitted_rows.saturating_add(stats.quota_skipped_rows) != stats.seen_rows {
                    return Err(CalibError::InvalidRouting(format!(
                        "layer {} expert {expert} {role}: admitted {} + skipped {} != seen {}",
                        self.layer, stats.admitted_rows, stats.quota_skipped_rows, stats.seen_rows
                    )));
                }
                if stats.admitted_rows != stats.seen_rows.min(limit_rows) {
                    return Err(CalibError::InvalidRouting(format!(
                        "layer {} expert {expert} {role}: admitted {} does not match quota-capped seen rows {}",
                        self.layer,
                        stats.admitted_rows,
                        stats.seen_rows.min(limit_rows)
                    )));
                }
                let expected_slack = stats.admitted_rows.saturating_sub(self.quota.target_rows);
                if stats.batch_slack_rows != expected_slack {
                    return Err(CalibError::InvalidRouting(format!(
                        "layer {} expert {expert} {role}: batch slack {} does not match admitted-above-target rows {expected_slack}",
                        self.layer, stats.batch_slack_rows
                    )));
                }
                validate_weights(
                    &format!("expert {expert} {role} full-stream"),
                    stats.seen_rows,
                    &stats.full_weight,
                )?;
                validate_weights(
                    &format!("expert {expert} {role} admitted-stream"),
                    stats.admitted_rows,
                    &stats.admitted_weight,
                )?;
                if stats.launch_telemetry_recorded {
                    let tile_rows = self.quota.tile_rows as u64;
                    let expected_full_tiles = stats.admitted_rows / tile_rows;
                    let expected_partial_tiles = u64::from(stats.admitted_rows % tile_rows != 0);
                    if stats.full_reduction_tiles != expected_full_tiles
                        || stats.partial_reduction_tiles != expected_partial_tiles
                    {
                        return Err(CalibError::InvalidCapture(format!(
                            "layer {} expert {expert} {role}: reduction tiles full={} partial={} do not match admitted rows {} at tile width {tile_rows}",
                            self.layer,
                            stats.full_reduction_tiles,
                            stats.partial_reduction_tiles,
                            stats.admitted_rows,
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn coverage_report(
        &self,
        policy: ExpertCoveragePolicy,
        required_fraction: f64,
    ) -> ExpertCoverageReport {
        let mut deficits = Vec::new();
        let mut covered = 0usize;
        for expert in 0..self.num_experts {
            for (role, stats) in [
                (ExpertCaptureRole::GateUpInput, &self.gate_up[expert]),
                (ExpertCaptureRole::DownInput, &self.down[expert]),
            ] {
                if stats.admitted_rows >= self.quota.min_rows {
                    covered += 1;
                } else {
                    deficits.push(ExpertCaptureDeficit {
                        layer: self.layer,
                        expert,
                        role,
                        admitted_rows: stats.admitted_rows,
                        required_rows: self.quota.min_rows,
                    });
                }
            }
        }
        let required = self.num_experts * EXPERT_CAPTURE_ROLES.len();
        let covered_fraction = covered as f64 / required as f64;
        ExpertCoverageReport {
            policy,
            required_fraction,
            covered_fraction,
            complete: covered_fraction >= required_fraction,
            deficits,
        }
    }
}

impl LayerRouterStats {
    fn new(num_experts: usize) -> Self {
        Self {
            routed_tokens: 0,
            routed_slots: 0,
            dropped_indices: 0,
            microbatches: 0,
            active_expert_sum: 0,
            max_active_experts: 0,
            padded_routed_rows: 0,
            saturated_after_routed_tokens: None,
            top1_hits: vec![0; num_experts],
            topk_hits: vec![0; num_experts],
            route_weights: vec![WeightStats::default(); num_experts],
            cooccurrence: BTreeMap::new(),
            token_counts: vec![BTreeMap::new(); num_experts],
            token_profile_dropped: vec![0; num_experts],
            stratum_counts: vec![BTreeMap::new(); num_experts],
        }
    }

    /// Bound the persisted token profile to the `keep` most-routed ids per expert,
    /// recording how many distinct ids were discarded.
    fn truncate_token_profile(&mut self, keep: usize) {
        for (expert, counts) in self.token_counts.iter_mut().enumerate() {
            if counts.len() <= keep {
                continue;
            }
            let mut ranked: Vec<(u32, u64)> = counts.iter().map(|(id, n)| (*id, *n)).collect();
            // Count descending, then token id ascending — deterministic on ties.
            ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
            let dropped = ranked.len() - keep;
            ranked.truncate(keep);
            *counts = ranked.into_iter().collect();
            if let Some(slot) = self.token_profile_dropped.get_mut(expert) {
                *slot = dropped as u64;
            }
        }
    }
}

impl Default for LayerRouterStats {
    fn default() -> Self {
        Self::new(0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertTelemetry {
    pub num_layers: usize,
    pub num_experts: usize,
    pub k_top: usize,
    pub quota: ExpertCaptureQuota,
    pub max_cooccurrence_pairs: usize,
    router: Vec<LayerRouterStats>,
    capture: Vec<ExpertCaptureStats>,
}

impl ExpertTelemetry {
    pub fn new(
        num_layers: usize,
        num_experts: usize,
        k_top: usize,
        quota: ExpertCaptureQuota,
        max_cooccurrence_pairs: usize,
    ) -> Result<Self, CalibError> {
        quota.validate()?;
        if num_layers == 0 || num_experts == 0 || k_top == 0 {
            return Err(CalibError::InvalidOptions(
                "expert telemetry requires non-zero layers, experts, and K-top".into(),
            ));
        }
        if k_top > num_experts {
            return Err(CalibError::InvalidOptions(format!(
                "K-top {k_top} exceeds expert count {num_experts}"
            )));
        }
        Ok(Self {
            num_layers,
            num_experts,
            k_top,
            quota,
            max_cooccurrence_pairs,
            router: (0..num_layers)
                .map(|_| LayerRouterStats::new(num_experts))
                .collect(),
            capture: vec![
                ExpertCaptureStats::default();
                num_layers * num_experts * EXPERT_CAPTURE_ROLES.len()
            ],
        })
    }

    fn validate_layer(&self, layer: usize) -> Result<(), CalibError> {
        if layer >= self.num_layers {
            Err(CalibError::InvalidRouting(format!(
                "layer {layer} is outside 0..{}",
                self.num_layers
            )))
        } else {
            Ok(())
        }
    }

    fn validate_expert(&self, expert: usize) -> Result<(), CalibError> {
        if expert >= self.num_experts {
            Err(CalibError::InvalidRouting(format!(
                "expert {expert} is outside 0..{}",
                self.num_experts
            )))
        } else {
            Ok(())
        }
    }

    fn capture_index(&self, layer: usize, expert: usize, role: ExpertCaptureRole) -> usize {
        let role_index = match role {
            ExpertCaptureRole::GateUpInput => 0,
            ExpertCaptureRole::DownInput => 1,
        };
        (layer * self.num_experts + expert) * EXPERT_CAPTURE_ROLES.len() + role_index
    }

    /// Record one token's routing decision. [`RoutedRowContext`] carries whatever
    /// the caller knows about where the row came from; a shape-only routing plan
    /// supplies neither field and only load/weight stats accrue.
    pub fn record_router_selection(
        &mut self,
        layer: usize,
        context: RoutedRowContext<'_>,
        indices: &[usize],
        weights: &[f32],
    ) -> Result<(), CalibError> {
        let RoutedRowContext { token, stratum } = context;
        self.validate_layer(layer)?;
        for (rank, &expert) in indices.iter().take(self.k_top).enumerate() {
            if expert < self.num_experts {
                let weight = weights.get(rank).copied().unwrap_or(0.0);
                if !weight.is_finite() {
                    return Err(CalibError::InvalidRouting(format!(
                        "non-finite router weight at layer {layer}, rank {rank}"
                    )));
                }
            }
        }
        let stats = &mut self.router[layer];
        stats.routed_tokens += 1;
        let mut valid_experts = Vec::with_capacity(self.k_top);
        for (rank, &expert) in indices.iter().take(self.k_top).enumerate() {
            if expert >= self.num_experts {
                stats.dropped_indices += 1;
                continue;
            }
            let weight = weights.get(rank).copied().unwrap_or(0.0);
            if rank == 0 {
                stats.top1_hits[expert] += 1;
            }
            stats.topk_hits[expert] += 1;
            stats.routed_slots += 1;
            stats.route_weights[expert].record(weight);
            if let (Some(token), Some(counts)) = (token, stats.token_counts.get_mut(expert)) {
                *counts.entry(token).or_insert(0) += 1;
            }
            if let (Some(stratum), Some(counts)) = (stratum, stats.stratum_counts.get_mut(expert)) {
                *counts.entry(stratum.to_string()).or_insert(0) += 1;
            }
            valid_experts.push(expert);
        }
        valid_experts.sort_unstable();
        valid_experts.dedup();
        for i in 0..valid_experts.len() {
            for j in (i + 1)..valid_experts.len() {
                let key =
                    valid_experts[i] as u64 * self.num_experts as u64 + valid_experts[j] as u64;
                if stats.cooccurrence.contains_key(&key)
                    || stats.cooccurrence.len() < self.max_cooccurrence_pairs
                {
                    *stats.cooccurrence.entry(key).or_insert(0) += 1;
                }
            }
        }
        Ok(())
    }

    pub fn record_grouped_batch_shape(
        &mut self,
        layer: usize,
        total_slots: usize,
        padded_rows: usize,
        active_experts: usize,
    ) -> Result<(), CalibError> {
        self.validate_layer(layer)?;
        if padded_rows < total_slots || active_experts > self.num_experts {
            return Err(CalibError::InvalidRouting(format!(
                "grouped batch shape slots={total_slots}, padded={padded_rows}, active={active_experts} is invalid for {} experts",
                self.num_experts
            )));
        }
        let stats = &mut self.router[layer];
        stats.microbatches = stats.microbatches.saturating_add(1);
        stats.active_expert_sum = stats
            .active_expert_sum
            .saturating_add(active_experts as u64);
        stats.max_active_experts = stats.max_active_experts.max(active_experts);
        stats.padded_routed_rows = stats
            .padded_routed_rows
            .saturating_add((padded_rows - total_slots) as u64);
        Ok(())
    }

    pub fn record_capture_launches(
        &mut self,
        layer: usize,
        expert: usize,
        role: ExpertCaptureRole,
        gather_launches: usize,
        full_reduction_tiles: usize,
    ) -> Result<(), CalibError> {
        self.validate_layer(layer)?;
        self.validate_expert(expert)?;
        let index = self.capture_index(layer, expert, role);
        let stats = &mut self.capture[index];
        stats.launch_telemetry_recorded = true;
        stats.capture_gather_launches = stats
            .capture_gather_launches
            .saturating_add(gather_launches as u64);
        stats.full_reduction_tiles = stats
            .full_reduction_tiles
            .saturating_add(full_reduction_tiles as u64);
        Ok(())
    }

    pub fn record_partial_reduction_tile(
        &mut self,
        layer: usize,
        expert: usize,
        role: ExpertCaptureRole,
    ) -> Result<(), CalibError> {
        self.validate_layer(layer)?;
        self.validate_expert(expert)?;
        let index = self.capture_index(layer, expert, role);
        self.capture[index].partial_reduction_tiles = self.capture[index]
            .partial_reduction_tiles
            .saturating_add(1);
        Ok(())
    }

    pub fn mark_layer_saturated_if_complete(&mut self, layer: usize) -> Result<(), CalibError> {
        self.validate_layer(layer)?;
        if self.layer_capture_saturated(layer)
            && self.router[layer].saturated_after_routed_tokens.is_none()
        {
            self.router[layer].saturated_after_routed_tokens =
                Some(self.router[layer].routed_tokens);
        }
        Ok(())
    }

    pub fn record_capture_route(
        &mut self,
        layer: usize,
        expert: usize,
        role: ExpertCaptureRole,
        weight: f32,
    ) -> Result<CaptureAdmission, CalibError> {
        self.validate_layer(layer)?;
        self.validate_expert(expert)?;
        if !weight.is_finite() {
            return Err(CalibError::InvalidRouting(format!(
                "non-finite capture weight at layer {layer}, expert {expert}, role {role}"
            )));
        }
        let index = self.capture_index(layer, expert, role);
        let stats = &mut self.capture[index];
        stats.seen_rows += 1;
        stats.full_weight.record(weight);
        let limit_rows = self.quota.limit_rows()?;
        if stats.admitted_rows < limit_rows {
            stats.admitted_rows += 1;
            if stats.admitted_rows > self.quota.target_rows {
                stats.batch_slack_rows += 1;
            }
            stats.admitted_weight.record(weight);
            Ok(CaptureAdmission::Capture)
        } else {
            stats.quota_skipped_rows += 1;
            Ok(CaptureAdmission::TelemetryOnly)
        }
    }

    pub fn record_capture_batch(
        &mut self,
        layer: usize,
        expert: usize,
        role: ExpertCaptureRole,
        weights: &[f32],
    ) -> Result<usize, CalibError> {
        let mut admitted = 0;
        for &weight in weights {
            if self.record_capture_route(layer, expert, role, weight)? == CaptureAdmission::Capture
            {
                admitted += 1;
            }
        }
        Ok(admitted)
    }

    pub fn capture_stats(
        &self,
        layer: usize,
        expert: usize,
        role: ExpertCaptureRole,
    ) -> &ExpertCaptureStats {
        &self.capture[self.capture_index(layer, expert, role)]
    }

    pub fn layer_snapshot(&self, layer: usize) -> Result<ExpertLayerTelemetry, CalibError> {
        self.validate_layer(layer)?;
        let mut router = self.router[layer].clone();
        router.truncate_token_profile(TOKEN_PROFILE_KEEP);
        Ok(ExpertLayerTelemetry {
            layer,
            num_experts: self.num_experts,
            k_top: self.k_top,
            quota: self.quota,
            router,
            gate_up: (0..self.num_experts)
                .map(|expert| {
                    self.capture_stats(layer, expert, ExpertCaptureRole::GateUpInput)
                        .clone()
                })
                .collect(),
            down: (0..self.num_experts)
                .map(|expert| {
                    self.capture_stats(layer, expert, ExpertCaptureRole::DownInput)
                        .clone()
                })
                .collect(),
        })
    }

    pub fn layer_coverage_report(
        &self,
        layer: usize,
        policy: ExpertCoveragePolicy,
        required_fraction: f64,
    ) -> Result<ExpertCoverageReport, CalibError> {
        self.validate_layer(layer)?;
        let mut deficits = Vec::new();
        let mut covered = 0usize;
        let required = self.num_experts * EXPERT_CAPTURE_ROLES.len();
        for expert in 0..self.num_experts {
            for role in EXPERT_CAPTURE_ROLES {
                let admitted = self.capture_stats(layer, expert, role).admitted_rows;
                if admitted >= self.quota.min_rows {
                    covered += 1;
                } else {
                    deficits.push(ExpertCaptureDeficit {
                        layer,
                        expert,
                        role,
                        admitted_rows: admitted,
                        required_rows: self.quota.min_rows,
                    });
                }
            }
        }
        let covered_fraction = covered as f64 / required as f64;
        Ok(ExpertCoverageReport {
            policy,
            required_fraction,
            covered_fraction,
            complete: covered_fraction >= required_fraction,
            deficits,
        })
    }

    pub fn layer_capture_saturated(&self, layer: usize) -> bool {
        let Ok(limit_rows) = self.quota.limit_rows() else {
            return false;
        };
        layer < self.num_layers
            && (0..self.num_experts).all(|expert| {
                EXPERT_CAPTURE_ROLES.iter().all(|&role| {
                    self.capture_stats(layer, expert, role).admitted_rows >= limit_rows
                })
            })
    }

    pub fn reconcile(&self) -> Result<(), CalibError> {
        for layer in 0..self.num_layers {
            self.layer_snapshot(layer)?.reconcile()?;
        }
        Ok(())
    }

    pub fn coverage_report(
        &self,
        policy: ExpertCoveragePolicy,
        required_fraction: f64,
    ) -> ExpertCoverageReport {
        let mut deficits = Vec::new();
        let mut covered = 0usize;
        let required = self.num_layers * self.num_experts * EXPERT_CAPTURE_ROLES.len();
        for layer in 0..self.num_layers {
            for expert in 0..self.num_experts {
                for role in EXPERT_CAPTURE_ROLES {
                    let admitted = self.capture_stats(layer, expert, role).admitted_rows;
                    if admitted >= self.quota.min_rows {
                        covered += 1;
                    } else {
                        deficits.push(ExpertCaptureDeficit {
                            layer,
                            expert,
                            role,
                            admitted_rows: admitted,
                            required_rows: self.quota.min_rows,
                        });
                    }
                }
            }
        }
        let covered_fraction = if required == 0 {
            0.0
        } else {
            covered as f64 / required as f64
        };
        ExpertCoverageReport {
            policy,
            required_fraction,
            covered_fraction,
            complete: covered_fraction >= required_fraction,
            deficits,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertCaptureDeficit {
    pub layer: usize,
    pub expert: usize,
    pub role: ExpertCaptureRole,
    pub admitted_rows: u64,
    pub required_rows: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertCoverageReport {
    pub policy: ExpertCoveragePolicy,
    pub required_fraction: f64,
    pub covered_fraction: f64,
    pub complete: bool,
    pub deficits: Vec<ExpertCaptureDeficit>,
}

impl ExpertCoverageReport {
    pub fn finalize(&self) -> Result<ExpertCoverageOutcome, CalibError> {
        match self.policy {
            ExpertCoveragePolicy::Strict if !self.complete => {
                Err(CalibError::ExpertCoverage(format!(
                    "{} capture points are undercovered; covered fraction {:.6} is below required {:.6}",
                    self.deficits.len(), self.covered_fraction, self.required_fraction
                )))
            }
            ExpertCoveragePolicy::Strict => Ok(ExpertCoverageOutcome {
                preserve_high_precision: Vec::new(),
            }),
            ExpertCoveragePolicy::PreserveUndercovered => {
                let preserve_high_precision = self
                    .deficits
                    .iter()
                    .map(|deficit| LayerExpert::new(deficit.layer, deficit.expert))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                Ok(ExpertCoverageOutcome {
                    preserve_high_precision,
                })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertCoverageOutcome {
    pub preserve_high_precision: Vec<LayerExpert>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KldRefRow {
    pub sample_index: usize,
    pub position: usize,
    pub indices: Vec<u32>,
    pub logits: Vec<f32>,
    pub log_z: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KldRefBuilder {
    top_k: usize,
    rows: Vec<KldRefRow>,
}

impl KldRefBuilder {
    pub fn new(top_k: usize) -> Result<Self, CalibError> {
        if top_k == 0 {
            return Err(CalibError::InvalidKldRef(
                "top-k must be greater than zero".into(),
            ));
        }
        Ok(Self {
            top_k,
            rows: Vec::new(),
        })
    }

    pub fn push(&mut self, row: KldRefRow) -> Result<(), CalibError> {
        if self.rows.iter().any(|existing| {
            existing.sample_index == row.sample_index && existing.position == row.position
        }) {
            return Err(CalibError::InvalidKldRef(format!(
                "duplicate sample/position row {}/{}",
                row.sample_index, row.position
            )));
        }
        if row.indices.len() != row.logits.len() {
            return Err(CalibError::InvalidKldRef(format!(
                "row has {} indices but {} logits",
                row.indices.len(),
                row.logits.len()
            )));
        }
        if row.indices.is_empty() || row.indices.len() > self.top_k {
            return Err(CalibError::InvalidKldRef(format!(
                "row top-k width {} is outside 1..={} ",
                row.indices.len(),
                self.top_k
            )));
        }
        if !row.log_z.is_finite() || row.logits.iter().any(|value| !value.is_finite()) {
            return Err(CalibError::InvalidKldRef(
                "row contains non-finite logits or log-normalizer".into(),
            ));
        }
        self.rows.push(row);
        Ok(())
    }

    pub fn finish(self) -> Result<KldRefPayload, CalibError> {
        if self.rows.is_empty() {
            return Err(CalibError::InvalidKldRef(
                "at least one reference row is required".into(),
            ));
        }
        Ok(KldRefPayload {
            top_k: self.top_k,
            rows: self.rows,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KldRefPayload {
    top_k: usize,
    rows: Vec<KldRefRow>,
}

impl KldRefPayload {
    pub fn n_positions(&self) -> usize {
        self.rows.len()
    }

    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn position_map(&self) -> Vec<SamplePosition> {
        self.rows
            .iter()
            .map(|row| SamplePosition::new(row.sample_index, row.position))
            .collect()
    }

    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "n_positions": self.n_positions(),
            "top_k": self.top_k,
            "position_map": self.position_map(),
        })
    }

    pub fn to_hfq_tensors(&self) -> Vec<HfqMemTensor> {
        let mut indices = Vec::with_capacity(self.rows.len() * self.top_k);
        let mut logits = Vec::with_capacity(self.rows.len() * self.top_k);
        let mut log_z = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            log_z.push(row.log_z);
            for column in 0..self.top_k {
                indices.push(row.indices.get(column).copied().unwrap_or(0) as f32);
                logits.push(row.logits.get(column).copied().unwrap_or(f32::NEG_INFINITY));
            }
        }
        let n_positions = self.rows.len() as u32;
        let top_k = self.top_k as u32;
        vec![
            HfqMemTensor {
                name: "lm_head.kldref_idx".into(),
                quant_type: 2,
                shape: vec![n_positions, top_k],
                group_size: 0,
                data: f32_bytes(&indices),
            },
            HfqMemTensor {
                name: "lm_head.kldref_logit".into(),
                quant_type: 2,
                shape: vec![n_positions, top_k],
                group_size: 0,
                data: f32_bytes(&logits),
            },
            HfqMemTensor {
                name: "lm_head.kldref_logz".into(),
                quant_type: 2,
                shape: vec![n_positions],
                group_size: 0,
                data: f32_bytes(&log_z),
            },
        ]
    }
}

/// Compatibility adapter for the resident arch collectors' historical
/// `(log_z, top-k pairs)` rows. New streamed code should build rows with their
/// real sample/position mapping through [`KldRefBuilder`] directly.
pub fn legacy_kldref_payload(
    rows: &[(f32, Vec<(u32, f32)>)],
) -> Result<Option<KldRefPayload>, CalibError> {
    let Some((_, first_topk)) = rows.first() else {
        return Ok(None);
    };
    let mut builder = KldRefBuilder::new(first_topk.len())?;
    for (position, (log_z, topk)) in rows.iter().enumerate() {
        builder.push(KldRefRow {
            sample_index: 0,
            position,
            indices: topk.iter().map(|(index, _)| *index).collect(),
            logits: topk.iter().map(|(_, logit)| *logit).collect(),
            log_z: *log_z,
        })?;
    }
    builder.finish().map(Some)
}

pub fn legacy_kldref_tensors(
    rows: &[(f32, Vec<(u32, f32)>)],
) -> Result<Vec<HfqMemTensor>, CalibError> {
    Ok(legacy_kldref_payload(rows)?
        .map(|payload| payload.to_hfq_tensors())
        .unwrap_or_default())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Necessary (but not sufficient under skewed routing) corpus capacity for
/// every expert to receive `min_rows` routed activations.
pub fn minimum_tokens_for_expert_coverage(
    num_experts: usize,
    k_top: usize,
    min_rows: u64,
) -> Result<u64, CalibError> {
    if num_experts == 0 || k_top == 0 || min_rows == 0 {
        return Err(CalibError::InvalidOptions(
            "coverage capacity requires non-zero experts, K-top, and minimum rows".into(),
        ));
    }
    if k_top > num_experts {
        return Err(CalibError::InvalidOptions(format!(
            "K-top {k_top} exceeds expert count {num_experts}"
        )));
    }
    let required_slots = (num_experts as u64)
        .checked_mul(min_rows)
        .ok_or_else(|| CalibError::InvalidOptions("coverage capacity overflow".into()))?;
    Ok(required_slots.div_ceil(k_top as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quota() -> ExpertCaptureQuota {
        ExpertCaptureQuota {
            min_rows: 2,
            target_rows: 4,
            tile_rows: 2,
            sampling: ExpertSamplingPolicy::DeterministicFirst { seed: 7 },
        }
    }

    #[test]
    fn sample_set_is_deterministic_and_preserves_reset_boundaries() {
        let samples = vec![
            CalibrationSample::new("beta", vec![20, 21], "code"),
            CalibrationSample::new("alpha", vec![10, 11, 12], "text"),
        ];
        let a = SampleSet::new(samples.clone(), 8, 99).unwrap();
        let b = SampleSet::new(samples, 8, 99).unwrap();
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.rows_time_major(), b.rows_time_major());

        let rows = a.rows_time_major();
        assert_eq!(rows.iter().filter(|row| row.reset_state).count(), 2);
        for sample_idx in 0..2 {
            let sample_rows: Vec<_> = rows
                .iter()
                .filter(|row| row.sample_index == sample_idx)
                .collect();
            assert!(sample_rows[0].reset_state);
            assert_eq!(sample_rows[0].position, 0);
            assert!(sample_rows
                .windows(2)
                .all(|pair| { pair[1].position == pair[0].position + 1 && !pair[1].reset_state }));
        }
    }

    #[test]
    fn sample_set_rejects_empty_duplicate_and_over_context_samples() {
        assert!(matches!(
            SampleSet::new(Vec::new(), 8, 1),
            Err(CalibError::InvalidSamples(_))
        ));
        assert!(SampleSet::new(
            vec![CalibrationSample::new("empty", Vec::new(), "text")],
            8,
            1
        )
        .is_err());
        assert!(SampleSet::new(
            vec![
                CalibrationSample::new("same", vec![1], "text"),
                CalibrationSample::new("same", vec![2], "text"),
            ],
            8,
            1
        )
        .is_err());
        assert!(SampleSet::new(
            vec![CalibrationSample::new("long", vec![1, 2, 3], "text")],
            2,
            1
        )
        .is_err());
    }

    #[test]
    fn capture_registry_aliases_identical_inputs_once() {
        let mut registry = CaptureRegistry::default();
        let id = CaptureId::new(0, ProjectionRole::GateUpInput, Some(3));
        registry
            .register(CaptureDescriptor {
                id,
                output_names: vec![
                    "model.layers.0.experts.3.gate_proj".into(),
                    "model.layers.0.experts.3.up_proj".into(),
                ],
                input_width: 4096,
                policy: CapturePolicy::ImatrixOnly,
                layer: 0,
                role: ProjectionRole::GateUpInput,
                expert: Some(3),
                expert_quota: Some(quota()),
            })
            .unwrap();

        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry.resolve_output("model.layers.0.experts.3.up_proj"),
            Some(id)
        );
        assert!(matches!(
            registry.register(CaptureDescriptor {
                id: CaptureId::new(0, ProjectionRole::DownInput, Some(3)),
                output_names: vec!["model.layers.0.experts.3.up_proj".into()],
                input_width: 4096,
                policy: CapturePolicy::ImatrixOnly,
                layer: 0,
                role: ProjectionRole::DownInput,
                expert: Some(3),
                expert_quota: Some(quota()),
            }),
            Err(CalibError::DuplicateOutputName(_))
        ));
    }

    #[test]
    fn expert_capture_quota_stops_capture_but_keeps_seen_telemetry() {
        let mut telemetry = ExpertTelemetry::new(1, 1, 1, quota(), 16).unwrap();
        for _ in 0..6 {
            telemetry
                .record_router_selection(0, RoutedRowContext::unknown(), &[0], &[0.5])
                .unwrap();
            assert!(telemetry
                .record_capture_route(0, 0, ExpertCaptureRole::GateUpInput, 0.5)
                .is_ok());
            assert!(telemetry
                .record_capture_route(0, 0, ExpertCaptureRole::DownInput, 0.5)
                .is_ok());
        }

        let stats = telemetry.capture_stats(0, 0, ExpertCaptureRole::GateUpInput);
        assert_eq!(stats.seen_rows, 6);
        assert_eq!(stats.admitted_rows, 4);
        assert_eq!(stats.quota_skipped_rows, 2);
        assert_eq!(stats.full_weight.sum, 3.0);
        assert_eq!(stats.admitted_weight.sum, 2.0);
        assert!(telemetry.layer_capture_saturated(0));
        telemetry.reconcile().unwrap();
    }

    #[test]
    fn unaligned_capture_target_uses_only_one_tile_of_free_slack() {
        let quota = ExpertCaptureQuota {
            min_rows: 2,
            target_rows: 3,
            tile_rows: 2,
            sampling: ExpertSamplingPolicy::DeterministicFirst { seed: 7 },
        };
        quota.validate().unwrap();
        assert_eq!(quota.limit_rows().unwrap(), 4);

        let mut telemetry = ExpertTelemetry::new(1, 1, 1, quota, 8).unwrap();
        telemetry
            .record_capture_batch(0, 0, ExpertCaptureRole::GateUpInput, &[1.0; 6])
            .unwrap();

        let stats = telemetry.capture_stats(0, 0, ExpertCaptureRole::GateUpInput);
        assert_eq!(stats.seen_rows, 6);
        assert_eq!(stats.admitted_rows, 4);
        assert_eq!(stats.batch_slack_rows, 1);
        assert_eq!(stats.quota_skipped_rows, 2);
        assert!(telemetry.layer_capture_saturated(0) == false);
    }

    #[test]
    fn quota_crossing_admits_only_remaining_rows() {
        let mut telemetry = ExpertTelemetry::new(1, 1, 1, quota(), 8).unwrap();
        let admitted = telemetry
            .record_capture_batch(0, 0, ExpertCaptureRole::GateUpInput, &[1.0; 6])
            .unwrap();
        assert_eq!(admitted, 4);
        let stats = telemetry.capture_stats(0, 0, ExpertCaptureRole::GateUpInput);
        assert_eq!(
            (
                stats.seen_rows,
                stats.admitted_rows,
                stats.quota_skipped_rows
            ),
            (6, 4, 2)
        );
    }

    #[test]
    fn strict_coverage_fails_and_preserve_reports_fallback_experts() {
        let mut telemetry = ExpertTelemetry::new(1, 2, 1, quota(), 8).unwrap();
        for expert in [0usize, 0, 1] {
            telemetry
                .record_router_selection(0, RoutedRowContext::unknown(), &[expert], &[1.0])
                .unwrap();
            telemetry
                .record_capture_route(0, expert, ExpertCaptureRole::GateUpInput, 1.0)
                .unwrap();
            telemetry
                .record_capture_route(0, expert, ExpertCaptureRole::DownInput, 1.0)
                .unwrap();
        }

        let strict = telemetry.coverage_report(ExpertCoveragePolicy::Strict, 1.0);
        assert!(!strict.complete);
        assert!(matches!(
            strict.finalize(),
            Err(CalibError::ExpertCoverage(_))
        ));

        let preserve = telemetry.coverage_report(ExpertCoveragePolicy::PreserveUndercovered, 1.0);
        let outcome = preserve.finalize().unwrap();
        assert_eq!(
            outcome.preserve_high_precision,
            vec![LayerExpert::new(0, 1)]
        );
    }

    #[test]
    fn router_and_capture_counts_must_reconcile() {
        let mut telemetry = ExpertTelemetry::new(1, 1, 1, quota(), 8).unwrap();
        telemetry
            .record_router_selection(0, RoutedRowContext::unknown(), &[0], &[1.0])
            .unwrap();
        telemetry
            .record_capture_route(0, 0, ExpertCaptureRole::GateUpInput, 1.0)
            .unwrap();
        let error = telemetry.reconcile().unwrap_err().to_string();
        assert!(error.contains("down_input"));
    }

    #[test]
    fn serialized_layer_telemetry_reconciles_without_live_accumulator() {
        let mut telemetry = ExpertTelemetry::new(1, 1, 1, quota(), 8).unwrap();
        for _ in 0..2 {
            telemetry
                .record_router_selection(0, RoutedRowContext::unknown(), &[0], &[1.0])
                .unwrap();
            telemetry
                .record_capture_route(0, 0, ExpertCaptureRole::GateUpInput, 1.0)
                .unwrap();
            telemetry
                .record_capture_route(0, 0, ExpertCaptureRole::DownInput, 1.0)
                .unwrap();
        }
        let snapshot = telemetry.layer_snapshot(0).unwrap();
        snapshot.reconcile().unwrap();

        let mut malformed = snapshot.clone();
        malformed.down.clear();
        assert!(malformed
            .reconcile()
            .unwrap_err()
            .to_string()
            .contains("down capture length"));

        let mut malformed = snapshot;
        malformed.gate_up[0].admitted_weight.count = 1;
        assert!(malformed
            .reconcile()
            .unwrap_err()
            .to_string()
            .contains("admitted-stream weight stats"));
    }

    #[test]
    fn legacy_telemetry_json_defaults_new_launch_and_batch_counters() {
        let capture: ExpertCaptureStats = serde_json::from_value(serde_json::json!({
            "seen_rows": 3,
            "admitted_rows": 3,
            "batch_slack_rows": 0,
            "quota_skipped_rows": 0,
            "full_weight": {"count": 3, "sum": 3.0, "sum_squared": 3.0},
            "admitted_weight": {"count": 3, "sum": 3.0, "sum_squared": 3.0}
        }))
        .unwrap();
        assert_eq!(capture.capture_gather_launches, 0);
        assert!(!capture.launch_telemetry_recorded);
        assert_eq!(capture.full_reduction_tiles, 0);
        assert_eq!(capture.partial_reduction_tiles, 0);

        let router: LayerRouterStats = serde_json::from_value(serde_json::json!({
            "routed_tokens": 3,
            "routed_slots": 3,
            "dropped_indices": 0,
            "top1_hits": [3],
            "topk_hits": [3],
            "route_weights": [{"count": 3, "sum": 3.0, "sum_squared": 3.0}],
            "cooccurrence": {}
        }))
        .unwrap();
        assert_eq!(router.microbatches, 0);
        assert_eq!(router.active_expert_sum, 0);
        assert_eq!(router.max_active_experts, 0);
        assert_eq!(router.padded_routed_rows, 0);
        assert_eq!(router.saturated_after_routed_tokens, None);
    }

    #[test]
    fn kldref_builder_packs_canonical_tensors_and_position_map() {
        let mut builder = KldRefBuilder::new(2).unwrap();
        builder
            .push(KldRefRow {
                sample_index: 3,
                position: 7,
                indices: vec![5, 9],
                logits: vec![2.5, 1.25],
                log_z: 3.0,
            })
            .unwrap();
        let payload = builder.finish().unwrap();
        assert_eq!(payload.n_positions(), 1);
        assert_eq!(payload.top_k(), 2);
        assert_eq!(payload.position_map()[0], SamplePosition::new(3, 7));
        let tensors = payload.to_hfq_tensors();
        assert_eq!(tensors.len(), 3);
        assert_eq!(tensors[0].name, "lm_head.kldref_idx");
        assert_eq!(tensors[0].shape, vec![1, 2]);
        assert_eq!(tensors[2].name, "lm_head.kldref_logz");
    }

    #[test]
    fn legacy_kldref_adapter_preserves_canonical_tensor_bytes() {
        let rows = vec![
            (3.0, vec![(5, 2.5), (9, 1.25)]),
            (4.0, vec![(2, 3.5), (7, 0.75)]),
        ];
        let tensors = legacy_kldref_tensors(&rows).unwrap();
        let decode = |data: &[u8]| {
            data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        assert_eq!(decode(&tensors[0].data), vec![5.0, 9.0, 2.0, 7.0]);
        assert_eq!(decode(&tensors[1].data), vec![2.5, 1.25, 3.5, 0.75]);
        assert_eq!(decode(&tensors[2].data), vec![3.0, 4.0]);
    }

    #[test]
    fn calibration_job_metadata_round_trips() {
        let samples = SampleSet::new(
            vec![CalibrationSample::new("s0", vec![1, 2], "text")],
            8,
            42,
        )
        .unwrap();
        let job = CalibrationJob::new(
            "source-fingerprint",
            "tokenizer-fingerprint",
            samples,
            CalibrationOptions::default(),
        )
        .unwrap();
        let json = serde_json::to_string(&job).unwrap();
        let decoded: CalibrationJob = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, job);
        assert_eq!(decoded.options.expert_quota.min_rows, 2048);
        assert_eq!(decoded.options.expert_quota.target_rows, 4096);
    }

    #[test]
    fn expert_coverage_capacity_matches_a17b_bound() {
        assert_eq!(
            minimum_tokens_for_expert_coverage(512, 10, 2048).unwrap(),
            104_858
        );
    }
}
