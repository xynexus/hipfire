// SPDX-License-Identifier: Apache-2.0
//! Family-neutral native layer-stream calibration orchestration.

use hipfire_hash::{file_hash, stable_hash_bytes};
use hipfire_model::tokenizer::Tokenizer;
use hipfire_model::ModelSource;
use hipfire_rdna::Gpu;
use crate::calibration::boundary::{BoundaryBackend, BoundaryCheckpoint, BoundaryStore};
use crate::calibration::contracts::{
    BoundaryPrecision, CalibError, CalibrationJob, CalibrationOptions, CalibrationSample,
    CaptureRegistry, ExpertCaptureQuota, ExpertCoveragePolicy, ExpertLayerTelemetry,
    ExpertSamplingPolicy, KldRefBuilder, KldRefPayload, KldRefRow, LayerExpert, SampleSet,
};
use crate::calibration::residual_probe::ResidualProbe;
use crate::calibration::schedule::{
    LayerMicrobatch, MicrobatchGeometry, MicrobatchPlanner,
};
use crate::calibration::source::{
    LayerPrefetch, LayerPrefetchReport, PlannedTensorReader, ReadLedger, ReadLedgerSnapshot,
    TensorLoadPlan, TensorOwner,
};
use crate::calibration::stream::{
    registered_calibration_adapter, CalibrationFamilyAdapter, CalibrationResourceEstimate,
    ModelInspection,
};
use crate::calibration::{
    build_calibration_metadata, combine_calib_parts, CalibSummary, CalibTensorDesc,
};
use crate::hfq::HfqFile;
use crate::safetensors_source::SafetensorsSource;
use std::error::Error;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Instant;

const CORPUS_TOKENIZE_WINDOW_BYTES: usize = 256 * 1024;
const CALIBRATION_PROGRESS_SCHEMA_VERSION: u32 = 2;
// The CLI treats this as an auto-tuning ceiling, not an unconditional
// allocation. Live resource estimates and allocation probes select a smaller
// geometry when the model/architecture cannot support the full row count.
const CALIBRATION_CLI_DEFAULT_MAX_ROWS: usize = 2048;
const CALIBRATION_DEFAULT_LAYER_PREFETCH_BYTES: u64 = 16 * 1024 * 1024 * 1024;
pub const CALIBRATION_PREFETCH_HOST_RESERVE_BYTES: u64 = 32 * 1024 * 1024 * 1024;
pub const CALIBRATION_PREFETCH_MIN_SWAP_FREE_DENOMINATOR: u64 = 4;

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrateCommand {
    pub model: PathBuf,
    pub corpus: PathBuf,
    pub output: PathBuf,
    pub sequences: usize,
    pub context: usize,
    pub sampling_seed: u64,
    pub sequence_batch: Option<usize>,
    pub time_tile: Option<usize>,
    pub max_rows: usize,
    pub min_expert_activations: u64,
    pub expert_capture_target: u64,
    pub expert_capture_tile_rows: usize,
    pub required_expert_fraction: f64,
    pub expert_coverage_policy: ExpertCoveragePolicy,
    pub kldref: bool,
    pub kldref_top_k: usize,
    pub kldref_rows: Option<usize>,
    pub layer_prefetch_bytes: u64,
    pub boundary_ram: bool,
    pub boundary_directory: Option<PathBuf>,
    pub resume: bool,
    pub pause_after_layers: Option<usize>,
    pub residual_probe_output: Option<PathBuf>,
    pub residual_probe_rows: usize,
    /// Publish an already-complete resumed spool without re-executing any
    /// layer. Requires `--resume`; carries no side outputs.
    pub finalize_completed: bool,
    /// Emit a native CASK (TriAttention centers) sidecar alongside the run.
    pub cask_output: Option<PathBuf>,
    /// CASK-only: the calibration artifact is scratch (capture skipped) and is
    /// removed after the CASK sidecar is written. Requires `--cask-output`.
    pub cask_only: bool,
    pub dry_run: bool,
}

impl CalibrateCommand {
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut model = None;
        let mut corpus = None;
        let mut output = None;
        let mut sequences = 128usize;
        let mut context = 2048usize;
        let mut sampling_seed = 1u64;
        let mut sequence_batch = None;
        let mut time_tile = None;
        let mut max_rows = CALIBRATION_CLI_DEFAULT_MAX_ROWS;
        let mut min_expert_activations = 2048u64;
        let mut expert_capture_target = 4096u64;
        let mut expert_capture_tile_rows = 256usize;
        let mut required_expert_fraction = 1.0f64;
        let mut expert_coverage_policy = ExpertCoveragePolicy::Strict;
        let mut kldref = true;
        let mut kldref_top_k = 64usize;
        let mut kldref_rows: Option<usize> = None;
        let mut layer_prefetch_bytes = CALIBRATION_DEFAULT_LAYER_PREFETCH_BYTES;
        let mut boundary_ram = false;
        let mut boundary_directory = None;
        // Resume is the default: an interrupted calibration should always be
        // continuable from its layer checkpoint rather than silently restarting
        // from layer 0. `resume_explicit` keeps `--resume`'s incompatibility
        // errors (RAM boundaries, residual probes) hard while letting the
        // *default* quietly step aside for those modes.
        let mut resume = true;
        let mut resume_explicit = false;
        let mut pause_after_layers = None;
        let mut residual_probe_output = None;
        let mut residual_probe_rows = 16usize;
        let mut finalize_completed = false;
        let mut cask_output = None;
        let mut cask_only = false;
        let mut dry_run = false;
        let mut index = 0usize;
        while index < args.len() {
            let flag = args[index].as_str();
            match flag {
                "--dry-run" => dry_run = true,
                "--kldref" => kldref = true,
                "--no-kldref" => kldref = false,
                "--boundary-ram" => boundary_ram = true,
                "--finalize-completed" => finalize_completed = true,
                "--cask-only" => cask_only = true,
                "--resume" => {
                    resume = true;
                    resume_explicit = true;
                }
                "--no-resume" => {
                    resume = false;
                    resume_explicit = false;
                }
                _ => {
                    let value = args
                        .get(index + 1)
                        .ok_or_else(|| format!("calibrate: {flag} requires a value"))?;
                    match flag {
                        "--model" => model = Some(PathBuf::from(value)),
                        "--corpus" => corpus = Some(PathBuf::from(value)),
                        "--output" => output = Some(PathBuf::from(value)),
                        "--sequences" => sequences = parse_value(flag, value)?,
                        "--context" => context = parse_value(flag, value)?,
                        "--sampling-seed" => sampling_seed = parse_value(flag, value)?,
                        "--sequence-batch" => sequence_batch = parse_auto_usize(flag, value)?,
                        "--time-tile" => time_tile = parse_auto_usize(flag, value)?,
                        "--max-rows" => max_rows = parse_value(flag, value)?,
                        "--min-expert-activations" => {
                            min_expert_activations = parse_value(flag, value)?
                        }
                        "--expert-capture-target" => {
                            expert_capture_target = parse_value(flag, value)?
                        }
                        "--expert-capture-tile-rows" => {
                            expert_capture_tile_rows = parse_value(flag, value)?
                        }
                        "--required-expert-fraction" => {
                            required_expert_fraction = parse_value(flag, value)?
                        }
                        "--expert-coverage-policy" => {
                            expert_coverage_policy = match value.as_str() {
                                "strict" => ExpertCoveragePolicy::Strict,
                                "preserve-undercovered" => {
                                    ExpertCoveragePolicy::PreserveUndercovered
                                }
                                _ => {
                                    return Err(format!(
                                        "calibrate: {flag} must be strict or preserve-undercovered"
                                    ))
                                }
                            }
                        }
                        "--kldref-topk" => kldref_top_k = parse_value(flag, value)?,
                        "--kldref-rows" => kldref_rows = Some(parse_value(flag, value)?),
                        "--layer-prefetch-bytes" => {
                            layer_prefetch_bytes = parse_value(flag, value)?
                        }
                        "--boundary-dir" => boundary_directory = Some(PathBuf::from(value)),
                        "--pause-after-layers" => {
                            pause_after_layers = Some(parse_value(flag, value)?)
                        }
                        "--residual-probe-output" => {
                            residual_probe_output = Some(PathBuf::from(value))
                        }
                        "--residual-probe-rows" => residual_probe_rows = parse_value(flag, value)?,
                        "--cask-output" => cask_output = Some(PathBuf::from(value)),
                        _ => return Err(format!("calibrate: unknown flag {flag}")),
                    }
                    index += 1;
                }
            }
            index += 1;
        }
        // Modes that cannot checkpoint: RAM boundaries have nothing to resume
        // from, residual probes require one uninterrupted pass, and CASK
        // generation currently requires a fresh run. An explicit `--resume`
        // with any of them is a user error; the default just yields.
        // `--finalize-completed` is deliberately excluded — it *requires*
        // resume, so it must not step aside.
        let unresumable =
            boundary_ram || residual_probe_output.is_some() || cask_output.is_some();
        if unresumable && !resume_explicit {
            resume = false;
        }
        let command = Self {
            model: model.ok_or("calibrate: --model <path> is required")?,
            corpus: corpus.ok_or("calibrate: --corpus <path> is required")?,
            output: output.ok_or("calibrate: --output <path> is required")?,
            sequences,
            context,
            sampling_seed,
            sequence_batch,
            time_tile,
            max_rows,
            min_expert_activations,
            expert_capture_target,
            expert_capture_tile_rows,
            required_expert_fraction,
            expert_coverage_policy,
            kldref,
            kldref_top_k,
            kldref_rows,
            layer_prefetch_bytes,
            boundary_ram,
            boundary_directory,
            resume,
            pause_after_layers,
            residual_probe_output,
            residual_probe_rows,
            finalize_completed,
            cask_output,
            cask_only,
            dry_run,
        };
        command
            .options()
            .and_then(|options| options.validate())
            .map_err(|error| error.to_string())?;
        if command.sequences == 0 || command.context < 2 {
            return Err(
                "calibrate: --sequences must be nonzero and --context must be at least 2".into(),
            );
        }
        if command.boundary_ram && command.boundary_directory.is_some() {
            return Err("calibrate: --boundary-ram conflicts with --boundary-dir".into());
        }
        if command.resume && command.boundary_ram {
            return Err("calibrate: --resume requires an mmap boundary store".into());
        }
        if command.pause_after_layers == Some(0) {
            return Err("calibrate: --pause-after-layers must be nonzero".into());
        }
        if command.residual_probe_rows == 0 {
            return Err("calibrate: --residual-probe-rows must be nonzero".into());
        }
        if command.residual_probe_output.is_some()
            && (command.resume || command.pause_after_layers.is_some())
        {
            return Err(
                "calibrate: residual probes require a fresh, uninterrupted run (no --resume or --pause-after-layers)"
                    .into(),
            );
        }
        if command.residual_probe_output.as_ref() == Some(&command.output) {
            return Err("calibrate: --residual-probe-output must differ from --output".into());
        }
        if command.finalize_completed && !command.resume {
            return Err("calibrate: --finalize-completed requires --resume".into());
        }
        if command.finalize_completed
            && (command.cask_output.is_some()
                || command.pause_after_layers.is_some()
                || command.residual_probe_output.is_some())
        {
            return Err(
                "calibrate: --finalize-completed only publishes already-complete calibration spools"
                    .into(),
            );
        }
        if command.cask_output.is_some() && (command.resume || command.pause_after_layers.is_some())
        {
            return Err(
                "calibrate: CASK generation currently requires a fresh, uninterrupted run (no --resume or --pause-after-layers)"
                    .into(),
            );
        }
        if let Some(cask) = command.cask_output.as_ref() {
            if cask == &command.output || command.residual_probe_output.as_ref() == Some(cask) {
                return Err("calibrate: --cask-output must be distinct from other outputs".into());
            }
        }
        if command.cask_only
            && (command.cask_output.is_none()
                || !command.boundary_ram
                || command.kldref
                || command.resume
                || command.pause_after_layers.is_some()
                || command.residual_probe_output.is_some())
        {
            return Err(
                "calibrate: --cask-only requires --cask-output, --boundary-ram, --no-kldref, and a fresh uninterrupted run"
                    .into(),
            );
        }
        Ok(command)
    }

    pub fn options(&self) -> Result<CalibrationOptions, CalibError> {
        Ok(CalibrationOptions {
            sequence_batch: self.sequence_batch,
            time_tile: self.time_tile,
            max_rows: self.max_rows,
            boundary_precision: BoundaryPrecision::F32,
            expert_quota: ExpertCaptureQuota {
                min_rows: self.min_expert_activations,
                target_rows: self.expert_capture_target,
                tile_rows: self.expert_capture_tile_rows,
                sampling: ExpertSamplingPolicy::DeterministicFirst {
                    seed: self.sampling_seed,
                },
            },
            required_expert_fraction: self.required_expert_fraction,
            // CASK-only skips capture entirely, so undercovered experts must not
            // fail the run — the calibration artifact is scratch either way.
            expert_coverage_policy: if self.cask_only {
                ExpertCoveragePolicy::PreserveUndercovered
            } else {
                self.expert_coverage_policy
            },
            kldref: self.kldref,
            kldref_top_k: self.kldref_top_k,
            kldref_rows: self.kldref_rows,
        })
    }
}

fn parse_value<T>(flag: &str, value: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse::<T>()
        .map_err(|_| format!("calibrate: invalid value {value:?} for {flag}"))
}

fn parse_auto_usize(flag: &str, value: &str) -> Result<Option<usize>, String> {
    if value == "auto" {
        Ok(None)
    } else {
        Ok(Some(parse_value(flag, value)?))
    }
}

pub struct ResolvedAdapter {
    pub family: &'static str,
    pub version: &'static str,
    pub adapter: Box<dyn CalibrationFamilyAdapter>,
}

pub fn resolve_adapter(source: &dyn ModelSource) -> Result<ResolvedAdapter, CalibError> {
    let adapter = registered_calibration_adapter(source.arch_id())?.ok_or_else(|| {
        CalibError::InvalidSourcePlan(format!(
            "no native calibration adapter is registered for architecture {}",
            source.arch_id()
        ))
    })?;
    let family = adapter.family();
    let version = adapter.adapter_version();
    Ok(ResolvedAdapter {
        family,
        version,
        adapter,
    })
}


pub fn resolve_hf_snapshot(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if path.join("config.json").is_file() {
        return Ok(path.to_path_buf());
    }
    let snapshots = path.join("snapshots");
    if !snapshots.is_dir() {
        return Err(format!(
            "{} is neither a Hugging Face snapshot nor a cache root",
            path.display()
        )
        .into());
    }
    if let Ok(revision) = fs::read_to_string(path.join("refs").join("main")) {
        let candidate = snapshots.join(revision.trim());
        if candidate.join("config.json").is_file() {
            return Ok(candidate);
        }
    }
    let mut candidates = fs::read_dir(&snapshots)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|candidate| candidate.join("config.json").is_file())
        .collect::<Vec<_>>();
    candidates.sort();
    match candidates.len() {
        1 => Ok(candidates.remove(0)),
        0 => Err(format!("{} contains no complete snapshot", path.display()).into()),
        count => Err(format!(
            "{} contains {count} snapshots and refs/main does not select one",
            path.display()
        )
        .into()),
    }
}

pub fn load_corpus_samples(
    corpus_path: &Path,
    tokenizer: &Tokenizer,
    wanted: usize,
    context: usize,
) -> Result<Vec<CalibrationSample>, Box<dyn Error>> {
    let corpus = fs::read_to_string(corpus_path)?;
    // A `.jsonl` corpus carries a label per record, which becomes the sample's
    // stratum and lets the router profiler characterise semantically-routed
    // layers by *what kind of document* reaches an expert. Plain text has no
    // labels, so every sample shares one stratum and the profile reports no
    // signal rather than a spurious finding.
    if corpus_path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("jsonl"))
    {
        return load_labelled_corpus_samples(&corpus, corpus_path, tokenizer, wanted, context);
    }
    let mut samples = Vec::with_capacity(wanted);
    for paragraph in corpus
        .split("\n\n")
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        let mut pending = Vec::<u32>::new();
        let mut byte_start = 0usize;
        while byte_start < paragraph.len() && samples.len() < wanted {
            let window_bytes = corpus_tokenize_window_bytes(wanted, samples.len(), context);
            let mut byte_end = (byte_start + window_bytes).min(paragraph.len());
            while !paragraph.is_char_boundary(byte_end) {
                byte_end -= 1;
            }
            pending.extend(tokenizer.encode(&paragraph[byte_start..byte_end]));
            byte_start = byte_end;
            while pending.len() >= context && samples.len() < wanted {
                let tail = pending.split_off(context);
                let tokens = std::mem::replace(&mut pending, tail);
                push_corpus_sample(&mut samples, tokens);
            }
        }
        if samples.len() < wanted && pending.len() >= 2 {
            push_corpus_sample(&mut samples, pending);
        }
        if samples.len() == wanted {
            break;
        }
    }
    if samples.len() != wanted {
        return Err(format!(
            "corpus {} produced {} independent sequences with at least two tokens; requested {wanted}",
            corpus_path.display(),
            samples.len()
        )
        .into());
    }
    Ok(samples)
}

fn corpus_tokenize_window_bytes(wanted: usize, produced: usize, context: usize) -> usize {
    // Tokenizers vary from roughly one byte per token for byte fallback to
    // several bytes per token for prose. Eight bytes per remaining requested
    // token bounds work without starving ordinary text; the fixed floor avoids
    // pathological tiny encode calls. The hard cap prevents an intermediate
    // token vector from approaching a model's context limit before we split it.
    wanted
        .saturating_sub(produced)
        .saturating_mul(context)
        .saturating_mul(8)
        .clamp(256, CORPUS_TOKENIZE_WINDOW_BYTES)
}

/// Field names a labelled corpus may use for its text and its label, in
/// preference order. `messages` is the chat-style shape both `Moe-lab/DBES` and
/// `allenai/tulu-3-sft-mixture` ship; `source` is their label field.
const CORPUS_TEXT_FIELDS: &[&str] = &["text", "content", "messages"];
const CORPUS_LABEL_FIELDS: &[&str] = &["stratum", "label", "source", "domain", "language"];

/// Flatten one JSONL record's text field, accepting either a plain string or a
/// chat-style `messages` array of `{role, content}`.
fn corpus_record_text(value: &serde_json::Value) -> Option<String> {
    for field in CORPUS_TEXT_FIELDS {
        match value.get(field) {
            Some(serde_json::Value::String(text)) => return Some(text.clone()),
            Some(serde_json::Value::Array(turns)) => {
                let joined = turns
                    .iter()
                    .filter_map(|turn| turn.get("content").and_then(serde_json::Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !joined.trim().is_empty() {
                    return Some(joined);
                }
            }
            _ => {}
        }
    }
    None
}

fn corpus_record_label(value: &serde_json::Value) -> String {
    for field in CORPUS_LABEL_FIELDS {
        if let Some(label) = value.get(field).and_then(serde_json::Value::as_str) {
            if !label.trim().is_empty() {
                return label.trim().to_string();
            }
        }
    }
    "unlabelled".to_string()
}

/// One sample per JSONL record, each carrying that record's label as its
/// stratum. Records are consumed in file order and short records are skipped
/// rather than concatenated, so a sample never straddles two labels — that would
/// make the per-expert stratum profile a lie.
fn load_labelled_corpus_samples(
    corpus: &str,
    corpus_path: &Path,
    tokenizer: &Tokenizer,
    wanted: usize,
    context: usize,
) -> Result<Vec<CalibrationSample>, Box<dyn Error>> {
    let mut samples = Vec::with_capacity(wanted);
    let mut skipped_short = 0usize;
    let mut skipped_unparsable = 0usize;
    for line in corpus.lines() {
        if samples.len() == wanted {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: serde_json::Value = match serde_json::from_str(line) {
            Ok(record) => record,
            Err(_) => {
                skipped_unparsable += 1;
                continue;
            }
        };
        let Some(text) = corpus_record_text(&record) else {
            skipped_unparsable += 1;
            continue;
        };
        let stratum = corpus_record_label(&record);
        // Bound the encode to what one sample can hold; a long record is
        // truncated, not split across samples.
        let budget = (context * 8).min(text.len());
        let mut end = budget;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        let mut tokens = tokenizer.encode(&text[..end]);
        if tokens.len() < 2 {
            skipped_short += 1;
            continue;
        }
        tokens.truncate(context);
        samples.push(CalibrationSample::new(
            format!("corpus-{:06}", samples.len()),
            tokens,
            stratum,
        ));
    }
    if samples.len() != wanted {
        return Err(format!(
            "labelled corpus {} produced {} usable records with at least two tokens; requested {wanted} \
             (skipped {skipped_short} too-short, {skipped_unparsable} unparsable)",
            corpus_path.display(),
            samples.len(),
        )
        .into());
    }
    let strata = samples
        .iter()
        .map(|sample| sample.stratum.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    eprintln!(
        "calibrate: labelled corpus supplied {} samples across {} strata: {}",
        samples.len(),
        strata.len(),
        strata.into_iter().collect::<Vec<_>>().join(", "),
    );
    Ok(samples)
}

fn push_corpus_sample(samples: &mut Vec<CalibrationSample>, tokens: Vec<u32>) {
    samples.push(CalibrationSample::new(
        format!("corpus-{:06}", samples.len()),
        tokens,
        "plain-text",
    ));
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceShardIdentity {
    pub file: String,
    pub bytes: u64,
    pub identity_kind: String,
    pub identity: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SourceManifestIdentity {
    pub fingerprint: String,
    pub shards: Vec<SourceShardIdentity>,
}

fn safetensors_header_identity(path: &Path) -> Result<String, CalibError> {
    use std::io::Read;
    let mut file = File::open(path).map_err(|error| {
        CalibError::InvalidSourcePlan(format!("open {}: {error}", path.display()))
    })?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix).map_err(|error| {
        CalibError::InvalidSourcePlan(format!("read {} header: {error}", path.display()))
    })?;
    let header_len = u64::from_le_bytes(prefix);
    let file_len = file
        .metadata()
        .map_err(|error| CalibError::InvalidSourcePlan(error.to_string()))?
        .len();
    if header_len > file_len.saturating_sub(8) || header_len > 512 * 1024 * 1024 {
        return Err(CalibError::InvalidSourcePlan(format!(
            "{} has invalid safetensors header length {header_len}",
            path.display()
        )));
    }
    let mut header = vec![0u8; header_len as usize + 8];
    header[..8].copy_from_slice(&prefix);
    file.read_exact(&mut header[8..]).map_err(|error| {
        CalibError::InvalidSourcePlan(format!("read {} header: {error}", path.display()))
    })?;
    Ok(stable_hash_bytes(&header))
}

fn source_shard_identity(path: &Path) -> Result<SourceShardIdentity, CalibError> {
    let bytes = fs::metadata(path)
        .map_err(|error| CalibError::InvalidSourcePlan(error.to_string()))?
        .len();
    let file = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string();
    if let Ok(target) = fs::read_link(path) {
        if let Some(blob) = target.file_name().and_then(|name| name.to_str()) {
            if matches!(blob.len(), 40 | 64) && blob.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Ok(SourceShardIdentity {
                    file,
                    bytes,
                    identity_kind: "huggingface_blob_digest".into(),
                    identity: blob.to_ascii_lowercase(),
                });
            }
        }
    }
    Ok(SourceShardIdentity {
        file,
        bytes,
        identity_kind: "safetensors_header_hash".into(),
        identity: safetensors_header_identity(path)?,
    })
}

pub fn source_manifest_identity(
    source: &dyn ModelSource,
) -> Result<SourceManifestIdentity, CalibError> {
    let mut names = source
        .tensor_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    names.sort();
    let mut shard_paths = std::collections::BTreeSet::new();
    let tensors = names
        .iter()
        .map(|name| {
            let info = source
                .tensor_info(name)
                .expect("tensor_names returned missing info");
            let storage = source.tensor_storage(name);
            if let Some(storage) = storage.as_ref() {
                shard_paths.insert(storage.path.clone());
            }
            serde_json::json!({
                "name": name,
                "dtype": info.dtype,
                "shape": info.shape,
                "storage": storage.map(|storage| serde_json::json!({
                    "file": storage.path.file_name().and_then(|name| name.to_str()),
                    "byte_offset": storage.byte_offset,
                    "byte_len": storage.byte_len,
                })),
            })
        })
        .collect::<Vec<_>>();
    let shards = shard_paths
        .iter()
        .map(|path| source_shard_identity(path))
        .collect::<Result<Vec<_>, _>>()?;
    let manifest = serde_json::to_vec(&serde_json::json!({
        "arch_id": source.arch_id(),
        "metadata": source.metadata_json(),
        "shards": shards,
        "tensors": tensors,
    }))
    .map_err(|error| CalibError::InvalidSourcePlan(error.to_string()))?;
    Ok(SourceManifestIdentity {
        fingerprint: stable_hash_bytes(&manifest),
        shards,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn default_boundary_directory(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("calib.hfq");
    output.with_file_name(format!(".{name}.boundary"))
}

pub struct CalibrationRunResult {
    pub model: ModelInspection,
    pub tensor_plan: TensorLoadPlan,
    pub read_ledger: ReadLedgerSnapshot,
    pub boundary_checkpoint: BoundaryCheckpoint,
    pub kldref: Option<KldRefPayload>,
    /// Number of KLD reference positions. This remains available when a
    /// completed artifact is recovered from its durable metadata without
    /// loading the potentially large KLD payload back into host memory.
    pub kldref_positions: Option<usize>,
    pub artifact_path: PathBuf,
    pub artifact: CalibSummary,
    pub expert_telemetry: Vec<ExpertLayerTelemetry>,
    pub geometry: MicrobatchGeometry,
    pub geometry_tuning: GeometryTuningReport,
    pub layer_timings: Vec<CalibrationLayerTiming>,
}

pub struct CalibrationPauseResult {
    pub model: ModelInspection,
    pub tensor_plan: TensorLoadPlan,
    pub read_ledger: ReadLedgerSnapshot,
    pub boundary_checkpoint: BoundaryCheckpoint,
    pub artifact_path: PathBuf,
    pub geometry: MicrobatchGeometry,
    pub geometry_tuning: GeometryTuningReport,
    pub layer_timings: Vec<CalibrationLayerTiming>,
}

pub enum CalibrationRunOutcome {
    Complete(CalibrationRunResult),
    Paused(CalibrationPauseResult),
}

struct RecoveredCalibrationArtifact {
    read_ledger: ReadLedgerSnapshot,
    kldref_positions: Option<usize>,
    artifact: CalibSummary,
    expert_telemetry: Vec<ExpertLayerTelemetry>,
    geometry: MicrobatchGeometry,
    geometry_tuning: GeometryTuningReport,
    layer_timings: Vec<CalibrationLayerTiming>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeometryCandidateProbe {
    pub geometry: MicrobatchGeometry,
    pub requested_bytes: u64,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeometryTuningReport {
    pub automatic: bool,
    pub selected: MicrobatchGeometry,
    pub free_vram_bytes: Option<u64>,
    pub reserved_headroom_bytes: u64,
    /// Which authority sized the candidate filter: `scheduler_budget` when the
    /// declared memory budget was used, `live_probe` for the instantaneous
    /// free-VRAM reading, `unavailable` when neither could be read.
    pub vram_ceiling_source: String,
    pub probes: Vec<GeometryCandidateProbe>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CalibrationLayerTiming {
    pub layer: usize,
    /// Time spent waiting for a lookahead read started by the previous layer.
    pub prefetch_wait_us: u64,
    /// Total background read duration for this layer's prefetched source bytes.
    pub prefetch_read_us: u64,
    pub prefetch_bytes: u64,
    /// Bytes retained in anonymous host staging and offered directly to the
    /// tensor reader. This may be smaller than `prefetch_bytes` after a partial
    /// read or bounded range that does not cover a complete tensor.
    pub prefetch_staged_bytes: u64,
    pub prefetch_errors: usize,
    /// Time required to plan and start the following layer's lookahead worker.
    pub prefetch_submit_us: u64,
    /// Why lookahead for the following layer was not started despite reaching
    /// the submission point. This distinguishes pressure admission from a
    /// configured-off run in persisted timing evidence.
    pub next_prefetch_disabled_reason: Option<String>,
    /// Successful source tensors materialized for this layer.
    pub source_tensor_count: u64,
    pub source_bytes: u64,
    pub gpu_upload_bytes: u64,
    pub staged_source_tensor_count: u64,
    pub staged_source_bytes: u64,
    /// Source lookup and logical-ledger accounting.
    pub source_view_us: u64,
    /// Host-side source dtype conversion/adjustment.
    pub source_decode_us: u64,
    /// HIP allocation/copy, including any mmap refaults during the copy.
    pub source_upload_us: u64,
    /// Mmap and page-cache release after synchronous upload.
    pub source_release_us: u64,
    /// Total adapter layer construction, including all source phases above,
    /// pointer tables, scratch allocation, and other family setup.
    pub load_upload_us: u64,
    pub execute_us: u64,
    pub capture_write_us: u64,
    pub finish_us: u64,
    pub part_sync_hash_us: u64,
    pub total_before_checkpoint_us: u64,
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct HostMemorySnapshot {
    pub available_bytes: Option<u64>,
    pub swap_total_bytes: Option<u64>,
    pub swap_free_bytes: Option<u64>,
    pub full_pressure_avg10: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPrefetchDecision {
    pub bytes: u64,
    pub disabled_reason: Option<&'static str>,
}

pub fn layer_prefetch_decision(
    configured_bytes: u64,
    layer_source_bytes: u64,
    host: HostMemorySnapshot,
) -> LayerPrefetchDecision {
    if configured_bytes == 0 {
        return LayerPrefetchDecision {
            bytes: 0,
            disabled_reason: Some("disabled_by_configuration"),
        };
    }
    if host.full_pressure_avg10.is_some_and(|avg10| avg10 > 0.0) {
        return LayerPrefetchDecision {
            bytes: 0,
            disabled_reason: Some("memory_psi_full"),
        };
    }
    if matches!(
        (host.swap_total_bytes, host.swap_free_bytes),
        (Some(total), Some(free))
            if total > 0
                && free < total.div_ceil(CALIBRATION_PREFETCH_MIN_SWAP_FREE_DENOMINATOR)
    ) {
        return LayerPrefetchDecision {
            bytes: 0,
            disabled_reason: Some("swap_free_below_25_percent"),
        };
    }
    // Staging and the next layer's GPU upload coexist until the upload is
    // synchronously complete. Reserve both that upload footprint and the live
    // host margin before retaining any anonymous lookahead bytes.
    let required_headroom =
        CALIBRATION_PREFETCH_HOST_RESERVE_BYTES.saturating_add(layer_source_bytes);
    let capacity = host
        .available_bytes
        .map(|available| available.saturating_sub(required_headroom))
        .unwrap_or(configured_bytes);
    let bytes = configured_bytes.min(layer_source_bytes).min(capacity);
    LayerPrefetchDecision {
        bytes,
        disabled_reason: (bytes == 0).then_some("host_headroom_below_reserve_plus_upload"),
    }
}

pub fn host_memory_snapshot() -> HostMemorySnapshot {
    #[cfg(target_os = "linux")]
    {
        let meminfo = fs::read_to_string("/proc/meminfo").ok();
        let kib = |label: &str| {
            meminfo.as_deref()?.lines().find_map(|line| {
                let mut fields = line.split_ascii_whitespace();
                (fields.next()? == label)
                    .then(|| fields.next()?.parse::<u64>().ok())
                    .flatten()
                    .and_then(|value| value.checked_mul(1024))
            })
        };
        let full_pressure_avg10 =
            fs::read_to_string("/proc/pressure/memory")
                .ok()
                .and_then(|pressure| {
                    pressure.lines().find_map(|line| {
                        let mut fields = line.split_ascii_whitespace();
                        (fields.next()? == "full").then(|| {
                            fields.find_map(|field| {
                                field
                                    .strip_prefix("avg10=")
                                    .and_then(|value| value.parse::<f64>().ok())
                            })
                        })?
                    })
                });
        HostMemorySnapshot {
            available_bytes: kib("MemAvailable:"),
            swap_total_bytes: kib("SwapTotal:"),
            swap_free_bytes: kib("SwapFree:"),
            full_pressure_avg10,
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        HostMemorySnapshot::default()
    }
}

fn estimated_remaining_layer_us(
    completed: &[CalibrationLayerTiming],
    remaining_layers: usize,
) -> Option<u64> {
    if completed.is_empty() {
        return None;
    }
    let total = completed.iter().fold(0u128, |sum, timing| {
        sum.saturating_add(timing.total_before_checkpoint_us as u128)
    });
    let mean = total / completed.len() as u128;
    u64::try_from(mean.saturating_mul(remaining_layers as u128)).ok()
}

fn format_duration_us(microseconds: u64) -> String {
    let seconds = microseconds / 1_000_000;
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m{seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CalibrationLayerProgress {
    schema_version: u32,
    engine_build: String,
    run_fingerprint: String,
    layer: usize,
    part_file: String,
    part_bytes: u64,
    part_hash: String,
    descriptors: Vec<CalibTensorDesc>,
    expert_telemetry: Option<ExpertLayerTelemetry>,
    preserve_high_precision: Vec<LayerExpert>,
    max_consistency: f32,
    read_ledger: ReadLedgerSnapshot,
    #[serde(default)]
    timing: CalibrationLayerTiming,
}

pub fn calibration_engine_build_identity() -> Result<String, CalibError> {
    #[cfg(target_os = "linux")]
    let executable = PathBuf::from(format!("/proc/{}/exe", std::process::id()));
    #[cfg(not(target_os = "linux"))]
    let executable = std::env::current_exe().map_err(|error| {
        CalibError::Checkpoint(format!(
            "resolve the running calibration executable for provenance: {error}"
        ))
    })?;
    let fingerprint = file_hash(&executable).ok_or_else(|| {
        CalibError::Checkpoint(format!(
            "fingerprint running calibration executable {}",
            executable.display()
        ))
    })?;
    if fingerprint == "unavailable" {
        return Err(CalibError::Checkpoint(format!(
            "fingerprint running calibration executable {} returned unavailable",
            executable.display()
        )));
    }
    Ok(format!("executable:{fingerprint}"))
}

fn calibration_checkpoint_execution_fingerprint(
    engine_build: &str,
    run_fingerprint: &str,
) -> String {
    let mut input = Vec::with_capacity(engine_build.len() + run_fingerprint.len() + 1);
    input.extend_from_slice(engine_build.as_bytes());
    input.push(0);
    input.extend_from_slice(run_fingerprint.as_bytes());
    stable_hash_bytes(&input)
}

pub fn calibration_run_fingerprint(
    adapter: &dyn CalibrationFamilyAdapter,
    model: &ModelInspection,
    tensor_plan: &TensorLoadPlan,
    job: &CalibrationJob,
    geometry: MicrobatchGeometry,
) -> Result<String, CalibError> {
    #[derive(serde::Serialize)]
    struct FingerprintModel<'a> {
        family: &'a str,
        arch_id: u32,
        hidden_width: usize,
        vocab_size: usize,
        num_layers: usize,
    }
    #[derive(serde::Serialize)]
    struct FingerprintInput<'a> {
        adapter_family: &'a str,
        adapter_version: &'a str,
        model: FingerprintModel<'a>,
        tensor_entries: &'a [crate::calibration::source::TensorLoadEntry],
        unique_source_bytes: u64,
        job: &'a CalibrationJob,
        geometry: MicrobatchGeometry,
    }
    let input = FingerprintInput {
        adapter_family: adapter.family(),
        adapter_version: adapter.adapter_version(),
        model: FingerprintModel {
            family: &model.family,
            arch_id: model.arch_id,
            hidden_width: model.hidden_width,
            vocab_size: model.vocab_size,
            num_layers: model.num_layers,
        },
        tensor_entries: tensor_plan.entries(),
        unique_source_bytes: tensor_plan.unique_source_bytes(),
        job,
        geometry,
    };
    let bytes =
        serde_json::to_vec(&input).map_err(|error| CalibError::Checkpoint(error.to_string()))?;
    Ok(stable_hash_bytes(&bytes))
}

fn calibration_progress_path(output: &Path, layer: usize) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("calib.hfq");
    output.with_file_name(format!(".{name}.layer-{layer:04}.progress.json"))
}

/// Recover the engine identity recorded by the run that produced a complete
/// spool, so `--finalize-completed` reconstructs the SAME checkpoint execution
/// fingerprint as the original process rather than this one. The layer-0
/// progress record is authoritative: it is the first thing every run commits.
fn completed_spool_engine_build(
    output: &Path,
    expected_run_fingerprint: &str,
) -> Result<String, CalibError> {
    let path = calibration_progress_path(output, 0);
    let bytes = fs::read(&path)
        .map_err(|error| CalibError::Checkpoint(format!("read {}: {error}", path.display())))?;
    let progress: CalibrationLayerProgress = serde_json::from_slice(&bytes)
        .map_err(|error| CalibError::Checkpoint(format!("parse {}: {error}", path.display())))?;
    if progress.schema_version != CALIBRATION_PROGRESS_SCHEMA_VERSION || progress.layer != 0 {
        return Err(CalibError::Checkpoint(format!(
            "completed-spool finalization requires a valid layer-0 progress record at {}",
            path.display()
        )));
    }
    if progress.run_fingerprint != expected_run_fingerprint {
        return Err(CalibError::Checkpoint(format!(
            "completed spools belong to run {}, expected {}",
            progress.run_fingerprint, expected_run_fingerprint
        )));
    }
    if progress.engine_build.is_empty() {
        return Err(CalibError::Checkpoint(
            "completed-spool engine identity is empty".into(),
        ));
    }
    Ok(progress.engine_build)
}

fn sync_file(path: &Path) -> Result<(), CalibError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| CalibError::Checkpoint(format!("sync {}: {error}", path.display())))
}

fn write_layer_progress(
    output: &Path,
    progress: &CalibrationLayerProgress,
) -> Result<(), CalibError> {
    let path = calibration_progress_path(output, progress.layer);
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(progress)
        .map_err(|error| CalibError::Checkpoint(error.to_string()))?;
    use std::io::Write;
    let mut file = File::create(&tmp)
        .map_err(|error| CalibError::Checkpoint(format!("create {}: {error}", tmp.display())))?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| CalibError::Checkpoint(format!("write {}: {error}", tmp.display())))?;
    fs::rename(&tmp, &path).map_err(|error| {
        CalibError::Checkpoint(format!(
            "commit progress {} -> {}: {error}",
            tmp.display(),
            path.display()
        ))
    })
}

fn read_layer_progress(
    output: &Path,
    layer: usize,
    expected_engine_build: &str,
    expected_run_fingerprint: &str,
) -> Result<CalibrationLayerProgress, CalibError> {
    let path = calibration_progress_path(output, layer);
    let bytes = fs::read(&path)
        .map_err(|error| CalibError::Checkpoint(format!("read {}: {error}", path.display())))?;
    let progress: CalibrationLayerProgress = serde_json::from_slice(&bytes)
        .map_err(|error| CalibError::Checkpoint(format!("parse {}: {error}", path.display())))?;
    if progress.schema_version != CALIBRATION_PROGRESS_SCHEMA_VERSION {
        return Err(CalibError::Checkpoint(format!(
            "unsupported calibration progress schema {} in {}",
            progress.schema_version,
            path.display()
        )));
    }
    if progress.engine_build != expected_engine_build {
        return Err(CalibError::Checkpoint(format!(
            "calibration progress {} was produced by engine {}, expected {}; resume with the original binary or restart the calibration",
            path.display(),
            progress.engine_build,
            expected_engine_build
        )));
    }
    if progress.run_fingerprint != expected_run_fingerprint {
        return Err(CalibError::Checkpoint(format!(
            "calibration progress {} belongs to run {}, expected {}",
            path.display(),
            progress.run_fingerprint,
            expected_run_fingerprint
        )));
    }
    if progress.layer != layer {
        return Err(CalibError::Checkpoint(format!(
            "calibration progress {} declares layer {}, expected {layer}",
            path.display(),
            progress.layer
        )));
    }
    let part = calibration_part_path(output, layer);
    let part_name = part
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if progress.part_file != part_name {
        return Err(CalibError::Checkpoint(format!(
            "calibration progress layer {layer} names part {:?}, expected {part_name:?}",
            progress.part_file
        )));
    }
    let part_bytes = fs::metadata(&part)
        .map_err(|error| CalibError::Checkpoint(format!("stat {}: {error}", part.display())))?
        .len();
    if part_bytes != progress.part_bytes {
        return Err(CalibError::Checkpoint(format!(
            "calibration part {} is {part_bytes} bytes, checkpoint requires {}",
            part.display(),
            progress.part_bytes
        )));
    }
    let part_hash = file_hash(&part).unwrap_or_else(|| "unavailable".into());
    if part_hash != progress.part_hash {
        return Err(CalibError::Checkpoint(format!(
            "calibration part {} hash {part_hash} differs from checkpoint {}",
            part.display(),
            progress.part_hash
        )));
    }
    Ok(progress)
}

fn remove_stale_layer_progress(
    output: &Path,
    start: usize,
    total: usize,
) -> Result<(), CalibError> {
    for layer in start..total {
        for path in [
            calibration_part_path(output, layer),
            calibration_progress_path(output, layer),
        ] {
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(CalibError::Checkpoint(format!(
                        "remove stale calibration spool {}: {error}",
                        path.display()
                    )))
                }
            }
        }
    }
    Ok(())
}

fn remove_if_exists(path: &Path) -> Result<(), CalibError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(CalibError::Checkpoint(format!(
            "remove {}: {error}",
            path.display()
        ))),
    }
}

fn cleanup_calibration_spools(output: &Path, total_layers: usize) -> Result<(), CalibError> {
    remove_stale_layer_progress(output, 0, total_layers)?;
    remove_if_exists(&assembling_artifact_path(output))
}

/// Remove every scratch spool and the (scratch) artifact of a `--cask-only` run.
///
/// CASK-only is deliberately non-resumable: its calibration parts are scratch
/// identity/ledger records, so retaining them after a failed run only makes the
/// next fresh retry fail on stale state. The CLI calls this on the cask-only
/// error path (the TriAttn sidecar, not the artifact, is the real output).
pub fn cleanup_cask_only_scratch(output: &Path, total_layers: usize) -> Result<(), CalibError> {
    cleanup_calibration_spools(output, total_layers)?;
    remove_if_exists(output)
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> Result<(), CalibError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            CalibError::Checkpoint(format!("sync directory {}: {error}", parent.display()))
        })
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> Result<(), CalibError> {
    Ok(())
}

fn metadata_usize(metadata: &serde_json::Value, key: &str) -> Result<usize, CalibError> {
    let value = metadata
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            CalibError::Checkpoint(format!(
                "completed calibration artifact has no unsigned {key} metadata"
            ))
        })?;
    usize::try_from(value).map_err(|_| {
        CalibError::Checkpoint(format!(
            "completed calibration artifact {key} does not fit usize"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_completed_artifact(
    path: &Path,
    expected_arch_id: u32,
    expected_family: &str,
    expected_adapter_version: &str,
    expected_engine_build: &str,
    expected_source_manifest: &SourceManifestIdentity,
    expected_run_fingerprint: &str,
    expected_job: &CalibrationJob,
    expected_geometry: MicrobatchGeometry,
) -> Result<RecoveredCalibrationArtifact, CalibError> {
    let package = HfqFile::open_index_only(path).map_err(|error| {
        CalibError::Checkpoint(format!(
            "open completed calibration artifact {}: {error}",
            path.display()
        ))
    })?;
    if package.arch_id != expected_arch_id {
        return Err(CalibError::Checkpoint(format!(
            "completed calibration artifact arch {} differs from expected {expected_arch_id}",
            package.arch_id
        )));
    }
    let metadata: serde_json::Value =
        serde_json::from_str(&package.metadata_json).map_err(|error| {
            CalibError::Checkpoint(format!(
                "parse completed calibration artifact metadata {}: {error}",
                path.display()
            ))
        })?;
    let expect_string = |key: &str, expected: &str| -> Result<(), CalibError> {
        let actual = metadata.get(key).and_then(serde_json::Value::as_str);
        if actual != Some(expected) {
            return Err(CalibError::Checkpoint(format!(
                "completed calibration artifact {key} {:?} differs from expected {expected:?}",
                actual
            )));
        }
        Ok(())
    };
    expect_string("artifact_kind", "calibration")?;
    expect_string("family", expected_family)?;
    expect_string("adapter_version", expected_adapter_version)?;
    expect_string("engine_build", expected_engine_build)?;
    expect_string("run_fingerprint", expected_run_fingerprint)?;
    if metadata.get("arch_id").and_then(serde_json::Value::as_u64)
        != Some(u64::from(expected_arch_id))
    {
        return Err(CalibError::Checkpoint(
            "completed calibration artifact arch_id metadata does not match the source plan".into(),
        ));
    }
    let expected_source = serde_json::to_value(expected_source_manifest)
        .map_err(|error| CalibError::Checkpoint(error.to_string()))?;
    if metadata.get("source_manifest") != Some(&expected_source) {
        return Err(CalibError::Checkpoint(
            "completed calibration artifact source manifest does not match this run".into(),
        ));
    }
    let expected_job_value = serde_json::to_value(expected_job)
        .map_err(|error| CalibError::Checkpoint(error.to_string()))?;
    if metadata.get("job") != Some(&expected_job_value) {
        return Err(CalibError::Checkpoint(
            "completed calibration artifact job does not match this run".into(),
        ));
    }

    let geometry: MicrobatchGeometry = serde_json::from_value(
        metadata
            .get("microbatch_geometry")
            .cloned()
            .ok_or_else(|| {
                CalibError::Checkpoint(
                    "completed calibration artifact lacks microbatch_geometry".into(),
                )
            })?,
    )
    .map_err(|error| CalibError::Checkpoint(format!("invalid microbatch geometry: {error}")))?;
    if geometry != expected_geometry {
        return Err(CalibError::Checkpoint(format!(
            "completed calibration artifact geometry {geometry:?} differs from expected {expected_geometry:?}"
        )));
    }
    let geometry_tuning: GeometryTuningReport =
        serde_json::from_value(metadata.get("geometry_tuning").cloned().ok_or_else(|| {
            CalibError::Checkpoint("completed calibration artifact lacks geometry_tuning".into())
        })?)
        .map_err(|error| CalibError::Checkpoint(format!("invalid geometry tuning: {error}")))?;
    if geometry_tuning.selected != geometry {
        return Err(CalibError::Checkpoint(
            "completed calibration artifact tuning selected a different geometry".into(),
        ));
    }

    let read_ledger: ReadLedgerSnapshot =
        serde_json::from_value(metadata.get("read_ledger").cloned().ok_or_else(|| {
            CalibError::Checkpoint("completed calibration artifact lacks read_ledger".into())
        })?)
        .map_err(|error| CalibError::Checkpoint(format!("invalid read ledger: {error}")))?;
    let expert_telemetry: Vec<ExpertLayerTelemetry> =
        serde_json::from_value(metadata.get("expert_telemetry").cloned().ok_or_else(|| {
            CalibError::Checkpoint("completed calibration artifact lacks expert_telemetry".into())
        })?)
        .map_err(|error| CalibError::Checkpoint(format!("invalid expert telemetry: {error}")))?;
    let layer_timings: Vec<CalibrationLayerTiming> = metadata
        .get("layer_timings")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| CalibError::Checkpoint(format!("invalid layer timings: {error}")))?
        .unwrap_or_default();

    let n_hessian = metadata_usize(&metadata, "n_hessian")?;
    let n_imatrix = metadata_usize(&metadata, "n_imatrix")?;
    let hessian_names = package
        .tensors()
        .iter()
        .filter_map(|tensor| tensor.name.strip_suffix(".hessian"))
        .collect::<std::collections::BTreeSet<_>>();
    let imatrix_names = package
        .tensors()
        .iter()
        .filter_map(|tensor| tensor.name.strip_suffix(".imatrix"))
        .collect::<std::collections::BTreeSet<_>>();
    if hessian_names.len() != n_hessian || imatrix_names.len() != n_imatrix {
        return Err(CalibError::Checkpoint(format!(
            "completed calibration artifact tensor counts hessian={} imatrix={} differ from metadata {n_hessian}/{n_imatrix}",
            hessian_names.len(),
            imatrix_names.len()
        )));
    }
    if !hessian_names.is_subset(&imatrix_names) {
        return Err(CalibError::Checkpoint(
            "completed calibration artifact has a Hessian without its imatrix".into(),
        ));
    }
    let per_tensor_tokens = metadata
        .get("per_tensor_tokens")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            CalibError::Checkpoint("completed calibration artifact lacks per_tensor_tokens".into())
        })?;
    if per_tensor_tokens.len() != n_imatrix
        || !imatrix_names
            .iter()
            .all(|name| per_tensor_tokens.contains_key(*name))
    {
        return Err(CalibError::Checkpoint(
            "completed calibration artifact per-tensor token metadata is incomplete".into(),
        ));
    }

    let max_consistency = metadata
        .get("max_consistency")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| {
            CalibError::Checkpoint("completed calibration artifact lacks max_consistency".into())
        })? as f32;
    if !max_consistency.is_finite() || max_consistency < 0.0 {
        return Err(CalibError::Checkpoint(
            "completed calibration artifact has invalid max_consistency".into(),
        ));
    }

    let kldref_positions = if expected_job.options.kldref {
        let kld = metadata
            .get("kldref")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| {
                CalibError::Checkpoint("completed artifact lacks KLDREF metadata".into())
            })?;
        let positions = kld
            .get("n_positions")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                CalibError::Checkpoint("completed artifact has invalid KLDREF positions".into())
            })?;
        if kld.get("top_k").and_then(serde_json::Value::as_u64)
            != Some(expected_job.options.kldref_top_k as u64)
        {
            return Err(CalibError::Checkpoint(
                "completed artifact KLDREF top-k differs from the job".into(),
            ));
        }
        let expected_matrix = vec![positions as u32, expected_job.options.kldref_top_k as u32];
        let expected_vector = vec![positions as u32];
        for (name, shape) in [
            ("lm_head.kldref_idx", &expected_matrix),
            ("lm_head.kldref_logit", &expected_matrix),
            ("lm_head.kldref_logz", &expected_vector),
        ] {
            let tensor = package.find_tensor_info(name).ok_or_else(|| {
                CalibError::Checkpoint(format!("completed artifact lacks {name}"))
            })?;
            if tensor.quant_type != 2 || tensor.shape != *shape {
                return Err(CalibError::Checkpoint(format!(
                    "completed artifact {name} has dtype/shape {}/{:?}, expected F32/{shape:?}",
                    tensor.quant_type, tensor.shape
                )));
            }
        }
        Some(positions)
    } else {
        if metadata.get("kldref").is_some()
            || [
                "lm_head.kldref_idx",
                "lm_head.kldref_logit",
                "lm_head.kldref_logz",
            ]
            .iter()
            .any(|name| package.find_tensor_info(name).is_some())
        {
            return Err(CalibError::Checkpoint(
                "completed artifact contains KLDREF data for a no-KLD job".into(),
            ));
        }
        None
    };

    Ok(RecoveredCalibrationArtifact {
        read_ledger,
        kldref_positions,
        artifact: CalibSummary {
            n_hessian,
            n_imatrix,
            max_consistency,
        },
        expert_telemetry,
        geometry,
        geometry_tuning,
        layer_timings,
    })
}

/// The long-lived state of a layer-streamed calibration run.
///
/// `LayerStreamEngine::run` is a wrapper over `begin` → `step`* → `finish`, and
/// this is what lives between those calls. It exists so a caller can treat one
/// layer as a schedulable quantum instead of holding the GPU for the whole run.
///
/// The read ledger is held as a [`ReadLedgerSnapshot`] rather than as a live
/// `ReadLedger`, which borrows the `TensorLoadPlan` it was built from: a struct
/// owning both would be self-referential. Each phase reconstitutes the ledger
/// from the snapshot and writes it back.
pub struct CalibrationSession {
    // Configuration, carried over from the engine.
    artifact_output: PathBuf,
    pause_after_layers: Option<usize>,
    layer_prefetch_bytes: u64,
    residual_probe_output: Option<PathBuf>,
    // Derived once by `begin`.
    model: ModelInspection,
    tensor_plan: TensorLoadPlan,
    capture: CaptureRegistry,
    geometry: MicrobatchGeometry,
    geometry_tuning: GeometryTuningReport,
    execution_job: CalibrationJob,
    batches: Vec<LayerMicrobatch>,
    resource_estimate: Option<CalibrationResourceEstimate>,
    effective_precision: serde_json::Value,
    engine_build: String,
    run_fingerprint: String,
    source_manifest: SourceManifestIdentity,
    resuming_existing_checkpoint: bool,
    // Advanced by `step`.
    boundary: BoundaryStore,
    ledger_snapshot: ReadLedgerSnapshot,
    residual_probe: Option<ResidualProbe>,
    next_layer: usize,
    pending_prefetch: Option<(usize, LayerPrefetch)>,
    part_paths: Vec<PathBuf>,
    descriptors: Vec<CalibTensorDesc>,
    expert_telemetry: Vec<ExpertLayerTelemetry>,
    preserve_high_precision: Vec<LayerExpert>,
    max_consistency: f32,
    layer_timings: Vec<CalibrationLayerTiming>,
}

/// What `begin` produced: an artifact recovered from a prior completed run, or
/// a live session parked at its first uncompleted layer.
pub enum CalibrationSessionStart {
    Complete(CalibrationRunResult),
    Session(CalibrationSession),
}

/// What one `step` did.
pub enum CalibrationStep {
    /// A layer was committed and more remain.
    Advanced,
    /// A layer was committed and `--pause-after-layers` was reached.
    Paused,
    /// No layer remained; the caller should call `finish`.
    LayersComplete,
}
pub struct LayerStreamEngine {
    boundary_backend: BoundaryBackend,
    artifact_output: PathBuf,
    resume: bool,
    finalize_completed: bool,
    cask_only: bool,
    pause_after_layers: Option<usize>,
    layer_prefetch_bytes: u64,
    residual_probe_output: Option<PathBuf>,
    residual_probe_rows: usize,
}

impl LayerStreamEngine {
    pub fn new(boundary_backend: BoundaryBackend, artifact_output: impl Into<PathBuf>) -> Self {
        Self {
            boundary_backend,
            artifact_output: artifact_output.into(),
            resume: false,
            finalize_completed: false,
            cask_only: false,
            pause_after_layers: None,
            layer_prefetch_bytes: CALIBRATION_DEFAULT_LAYER_PREFETCH_BYTES,
            residual_probe_output: None,
            residual_probe_rows: 16,
        }
    }

    pub const fn with_resume(mut self, resume: bool) -> Self {
        self.resume = resume;
        self
    }

    /// Publish an already-complete resumed spool without re-running any layer.
    /// The engine identity is recovered from the completed spool so the
    /// checkpoint fingerprint matches the original run rather than this process.
    pub const fn with_finalize_completed(mut self, finalize_completed: bool) -> Self {
        self.finalize_completed = finalize_completed;
        self
    }

    /// CASK-only: skip capture so the calibration artifact is scratch (the
    /// caller removes it after saving the CASK sidecar from the TriAttn tap).
    pub const fn with_cask_only(mut self, cask_only: bool) -> Self {
        self.cask_only = cask_only;
        self
    }

    pub const fn with_pause_after_layers(mut self, pause_after_layers: Option<usize>) -> Self {
        self.pause_after_layers = pause_after_layers;
        self
    }

    pub const fn with_layer_prefetch_bytes(mut self, layer_prefetch_bytes: u64) -> Self {
        self.layer_prefetch_bytes = layer_prefetch_bytes;
        self
    }

    pub fn with_residual_probe(mut self, output: Option<PathBuf>, rows: usize) -> Self {
        self.residual_probe_output = output;
        self.residual_probe_rows = rows;
        self
    }

    /// Run a calibration to completion, or to its `--pause-after-layers`
    /// boundary. A wrapper over the session phases: every caller that does not
    /// need to interleave other GPU work stays on this entry point.
    pub fn run(
        self,
        adapter: &mut dyn CalibrationFamilyAdapter,
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        job: &CalibrationJob,
    ) -> Result<CalibrationRunOutcome, CalibError> {
        let mut session = match self.begin(adapter, source, gpu, job)? {
            CalibrationSessionStart::Complete(result) => {
                return Ok(CalibrationRunOutcome::Complete(result))
            }
            CalibrationSessionStart::Session(session) => session,
        };
        loop {
            match session.step(adapter, source, gpu, job)? {
                CalibrationStep::Advanced => {}
                CalibrationStep::Paused => {
                    return Ok(CalibrationRunOutcome::Paused(session.into_paused()))
                }
                CalibrationStep::LayersComplete => break,
            }
        }
        session
            .finish(adapter, source, gpu, job)
            .map(CalibrationRunOutcome::Complete)
    }

    /// Validate the request, plan the run, open the boundary store, replay any
    /// resumable checkpoint, and execute the embedding pass.
    ///
    /// The completed-artifact recovery path returns before any layer work, so
    /// this yields either a finished result or a session.
    pub fn begin(
        self,
        adapter: &mut dyn CalibrationFamilyAdapter,
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        job: &CalibrationJob,
    ) -> Result<CalibrationSessionStart, CalibError> {
        job.options.validate()?;
        if self.artifact_output.exists() && !self.resume {
            return Err(CalibError::InvalidOptions(format!(
                "refusing to overwrite calibration artifact {}",
                self.artifact_output.display()
            )));
        }
        if let Some(path) = &self.residual_probe_output {
            if self.resume || self.pause_after_layers.is_some() {
                return Err(CalibError::InvalidOptions(
                    "residual probes require a fresh, uninterrupted calibration run".into(),
                ));
            }
            if path.exists() {
                return Err(CalibError::InvalidOptions(format!(
                    "refusing to overwrite residual probe {}",
                    path.display()
                )));
            }
            if path == &self.artifact_output {
                return Err(CalibError::InvalidOptions(
                    "residual probe output must differ from the calibration artifact".into(),
                ));
            }
        }
        if job.options.boundary_precision != BoundaryPrecision::F32 {
            return Err(CalibError::InvalidOptions(
                "the native layer-stream engine currently requires F32 boundaries".into(),
            ));
        }
        let model = adapter.inspect(source)?;
        model.validate()?;
        let residual_probe = self
            .residual_probe_output
            .as_ref()
            .map(|_| {
                ResidualProbe::new(
                    model.arch_id,
                    &model.family,
                    "layer-streamed",
                    job,
                    model.hidden_width,
                    model.num_layers,
                    self.residual_probe_rows,
                )
            })
            .transpose()?;
        if let Some(limit) = self.pause_after_layers {
            if limit > model.num_layers {
                return Err(CalibError::InvalidOptions(format!(
                    "pause-after-layers {limit} exceeds model layer count {}",
                    model.num_layers
                )));
            }
        }
        if adapter.family() != model.family {
            return Err(CalibError::InvalidSourcePlan(format!(
                "adapter family {} produced model family {}",
                adapter.family(),
                model.family
            )));
        }
        if source.arch_id() != model.arch_id {
            return Err(CalibError::InvalidSourcePlan(format!(
                "source arch {} differs from adapter plan arch {}",
                source.arch_id(),
                model.arch_id
            )));
        }
        let tensor_plan = TensorLoadPlan::build(source, model.tensor_requests.clone())?;
        let source_manifest = source_manifest_identity(source)?;
        if source_manifest.fingerprint != job.source_fingerprint {
            return Err(CalibError::InvalidSourcePlan(format!(
                "source manifest fingerprint {} differs from job {}",
                source_manifest.fingerprint, job.source_fingerprint
            )));
        }
        // CASK-only skips capture: the calibration artifact is scratch, only the
        // TriAttn tap output matters. Every other run captures normally.
        let capture = if self.cask_only {
            adapter.capture_plan(&model, job)?.skip_all()
        } else {
            adapter.capture_plan(&model, job)?
        };
        let geometry_tuning = tune_geometry_for_gpu(adapter, &model, &tensor_plan, job, gpu)?;
        let geometry = geometry_tuning.selected;
        let execution_job = execution_job(job, geometry);
        let planner = MicrobatchPlanner::new(geometry)?;
        let resource_estimate =
            adapter.resource_estimate(&model, &execution_job, planner.geometry())?;
        let effective_precision = adapter.effective_precision(gpu);
        let batches = planner.plan(&job.samples);
        let run_fingerprint =
            calibration_run_fingerprint(adapter, &model, &tensor_plan, job, geometry)?;
        // `--finalize-completed` publishes a spool produced by an EARLIER
        // process, so the checkpoint fingerprint must be reconstructed from that
        // run's recorded engine identity, not this process's. Every other run
        // uses its own identity.
        let engine_build = if self.finalize_completed {
            completed_spool_engine_build(&self.artifact_output, &run_fingerprint)?
        } else {
            calibration_engine_build_identity()?
        };
        let checkpoint_execution_fingerprint =
            calibration_checkpoint_execution_fingerprint(&engine_build, &run_fingerprint);
        let resuming_existing_checkpoint = match (&self.boundary_backend, self.resume) {
            (BoundaryBackend::Ram, true) => {
                return Err(CalibError::InvalidOptions(
                    "calibration resume requires an mmap boundary store".into(),
                ))
            }
            (BoundaryBackend::Mmap { directory }, true) => {
                BoundaryStore::mmap_checkpoint_exists(directory)?
            }
            _ => false,
        };
        if self.resume && !resuming_existing_checkpoint && self.artifact_output.exists() {
            return Err(CalibError::Checkpoint(format!(
                "cannot resume completed artifact {} without its boundary checkpoint",
                self.artifact_output.display()
            )));
        }
        if !resuming_existing_checkpoint {
            for layer in 0..model.num_layers {
                for path in [
                    calibration_part_path(&self.artifact_output, layer),
                    calibration_progress_path(&self.artifact_output, layer),
                ] {
                    if path.exists() {
                        return Err(CalibError::Checkpoint(format!(
                            "stale calibration spool exists at {} without a resumable boundary checkpoint; remove the stale run explicitly",
                            path.display()
                        )));
                    }
                }
            }
            let assembling = assembling_artifact_path(&self.artifact_output);
            if assembling.exists() {
                return Err(CalibError::Checkpoint(format!(
                    "stale assembling artifact exists at {} without a resumable boundary checkpoint; remove the stale run explicitly",
                    assembling.display()
                )));
            }
        }
        let mut boundary = match (&self.boundary_backend, self.resume) {
            (BoundaryBackend::Mmap { directory }, true) => {
                let (store, resumed) = BoundaryStore::resume_or_create_mmap(
                    directory,
                    job.samples.total_rows(),
                    model.hidden_width,
                    model.num_layers,
                    job.samples.fingerprint(),
                    &checkpoint_execution_fingerprint,
                )?;
                debug_assert_eq!(resumed, resuming_existing_checkpoint);
                store
            }
            (BoundaryBackend::Ram, true) => unreachable!("RAM resume rejected above"),
            (_, false) => BoundaryStore::create(
                self.boundary_backend,
                job.samples.total_rows(),
                model.hidden_width,
                model.num_layers,
                job.samples.fingerprint(),
                &checkpoint_execution_fingerprint,
            )?,
        };
        if boundary.checkpoint().rows != job.samples.total_rows()
            || boundary.checkpoint().width != model.hidden_width
            || boundary.checkpoint().total_layers != model.num_layers
        {
            return Err(CalibError::Checkpoint(format!(
                "boundary geometry [{}, {}, {}] differs from requested [{}, {}, {}]",
                boundary.checkpoint().rows,
                boundary.checkpoint().width,
                boundary.checkpoint().total_layers,
                job.samples.total_rows(),
                model.hidden_width,
                model.num_layers
            )));
        }
        if self.finalize_completed
            && (boundary.checkpoint().completed_layers != boundary.checkpoint().total_layers
                || !boundary.checkpoint().kld_finalized
                || boundary.checkpoint().artifact_complete)
        {
            return Err(CalibError::Checkpoint(
                "--finalize-completed requires all layers and KLD state committed, with the artifact still unpublished"
                    .into(),
            ));
        }
        if boundary.checkpoint().artifact_complete && !self.artifact_output.exists() {
            return Err(CalibError::Checkpoint(
                "calibration checkpoint declares a complete artifact, but the artifact is missing"
                    .into(),
            ));
        }
        if self.artifact_output.exists() {
            if boundary.checkpoint().completed_layers != boundary.checkpoint().total_layers
                || !boundary.checkpoint().kld_finalized
            {
                return Err(CalibError::Checkpoint(
                    "calibration artifact exists while its checkpoint is incomplete".into(),
                ));
            }
            let recovered = validate_completed_artifact(
                &self.artifact_output,
                model.arch_id,
                &model.family,
                adapter.adapter_version(),
                &engine_build,
                &source_manifest,
                &run_fingerprint,
                job,
                geometry,
            )?;
            let recovered_ledger = ReadLedger::resume(&tensor_plan, recovered.read_ledger.clone())?;
            recovered_ledger.assert_complete()?;
            if !boundary.checkpoint().artifact_complete {
                boundary.finalize_artifact()?;
            }
            cleanup_calibration_spools(&self.artifact_output, model.num_layers)?;
            return Ok(CalibrationSessionStart::Complete(CalibrationRunResult {
                model,
                tensor_plan,
                read_ledger: recovered.read_ledger,
                boundary_checkpoint: boundary.checkpoint().clone(),
                kldref: None,
                kldref_positions: recovered.kldref_positions,
                artifact_path: self.artifact_output,
                artifact: recovered.artifact,
                expert_telemetry: recovered.expert_telemetry,
                geometry: recovered.geometry,
                geometry_tuning: recovered.geometry_tuning,
                layer_timings: recovered.layer_timings,
            }));
        }
        let completed_layers = boundary.checkpoint().completed_layers;
        if self
            .pause_after_layers
            .is_some_and(|limit| completed_layers >= limit)
        {
            return Err(CalibError::InvalidOptions(format!(
                "checkpoint already has {completed_layers} completed layers; --pause-after-layers must be greater when resuming"
            )));
        }
        let mut part_paths = Vec::with_capacity(model.num_layers);
        let mut descriptors = Vec::<CalibTensorDesc>::new();
        let mut expert_telemetry = Vec::with_capacity(model.num_layers);
        let mut preserve_high_precision = Vec::<LayerExpert>::new();
        let mut max_consistency = 0.0f32;
        let mut layer_timings = Vec::with_capacity(model.num_layers);
        let mut resume_ledger = None;
        let pending_prefetch: Option<(usize, LayerPrefetch)> = None;

        if resuming_existing_checkpoint {
            let mut prior_consumed = std::collections::BTreeSet::new();
            for layer in 0..completed_layers {
                let progress = read_layer_progress(
                    &self.artifact_output,
                    layer,
                    &engine_build,
                    &run_fingerprint,
                )?;
                if !prior_consumed.is_subset(&progress.read_ledger.consumed_logical) {
                    return Err(CalibError::Checkpoint(format!(
                        "layer {layer} read ledger is not a monotonic continuation"
                    )));
                }
                ReadLedger::resume(&tensor_plan, progress.read_ledger.clone())?;
                prior_consumed = progress.read_ledger.consumed_logical.clone();
                descriptors.extend(progress.descriptors);
                if let Some(telemetry) = progress.expert_telemetry {
                    expert_telemetry.push(telemetry);
                }
                preserve_high_precision.extend(progress.preserve_high_precision);
                max_consistency = max_consistency.max(progress.max_consistency);
                layer_timings.push(progress.timing);
                resume_ledger = Some(progress.read_ledger);
                part_paths.push(calibration_part_path(&self.artifact_output, layer));
            }
            remove_stale_layer_progress(&self.artifact_output, completed_layers, model.num_layers)?;
            if completed_layers > 0 {
                eprintln!(
                    "calibrate: resuming {} at layer {completed_layers}/{}",
                    self.artifact_output.display(),
                    model.num_layers
                );
            }
        }
        // The ledger is carried between phases as its SNAPSHOT rather than as a
        // live `ReadLedger`, which borrows the `TensorLoadPlan` it was built from
        // (`calibration/source.rs`). That borrow is what stops the run state from
        // being hoisted into a session struct — a struct owning both the plan and
        // a ledger over it is self-referential. Reconstituting per phase costs one
        // clone of a few BTreeSets of tensor names against a layer of GPU work,
        // and `ReadLedger::resume` is already called repeatedly by the checkpoint
        // replay above, so round-tripping is an established operation rather than
        // a new one.
        let mut ledger_snapshot = match resume_ledger {
            Some(snapshot) => ReadLedger::resume(&tensor_plan, snapshot)?,
            None => ReadLedger::new(&tensor_plan),
        }
        .snapshot();

        if completed_layers == 0 {
            let mut ledger = ReadLedger::resume(&tensor_plan, ledger_snapshot.clone())?;
            let mut reader = PlannedTensorReader::new(source, &mut ledger, TensorOwner::Persistent);
            let mut embedding = adapter.load_embedding(&mut reader, gpu, &model, &execution_job)?;
            let execute_result = (|| {
                for batch in &batches {
                    let mut output = vec![0.0f32; batch.rows.len() * model.hidden_width];
                    embedding.execute(gpu, &batch.rows, &mut output)?;
                    boundary.write_active_indexed(&batch.boundary_rows, &output)?;
                }
                Ok::<(), CalibError>(())
            })();
            let finish_result = embedding.finish(gpu);
            execute_result?;
            finish_result?;
            ledger_snapshot = ledger.snapshot();
        }

        Ok(CalibrationSessionStart::Session(CalibrationSession {
            artifact_output: self.artifact_output,
            pause_after_layers: self.pause_after_layers,
            layer_prefetch_bytes: self.layer_prefetch_bytes,
            residual_probe_output: self.residual_probe_output,
            model,
            tensor_plan,
            capture,
            geometry,
            geometry_tuning,
            execution_job,
            batches,
            resource_estimate,
            effective_precision,
            engine_build,
            run_fingerprint,
            source_manifest,
            resuming_existing_checkpoint,
            boundary,
            ledger_snapshot,
            residual_probe,
            next_layer: completed_layers,
            pending_prefetch,
            part_paths,
            descriptors,
            expert_telemetry,
            preserve_high_precision,
            max_consistency,
            layer_timings,
        }))
    }
}

impl CalibrationSession {
    /// Run one layer: the calibration quantum. Loads the layer's weights,
    /// executes every microbatch through it, writes and syncs the capture part,
    /// and commits the boundary checkpoint.
    pub fn step(
        &mut self,
        adapter: &mut dyn CalibrationFamilyAdapter,
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        job: &CalibrationJob,
    ) -> Result<CalibrationStep, CalibError> {
        let layer_index = self.next_layer;
        if layer_index >= self.model.num_layers {
            return Ok(CalibrationStep::LayersComplete);
        }
        // One layer is the quantum: the ledger lives for exactly this step and
        // is handed back as a snapshot, so no borrow reaches across layers.
        let mut ledger = ReadLedger::resume(&self.tensor_plan, self.ledger_snapshot.clone())?;
        let layer_started = Instant::now();
        let prefetch_wait_started = Instant::now();
        let mut prefetch_report = match self.pending_prefetch.take() {
            Some((target_layer, prefetch)) if target_layer == layer_index => prefetch.wait(),
            Some((target_layer, prefetch)) => {
                drop(prefetch);
                LayerPrefetchReport {
                    errors: vec![format!(
                        "prefetch target layer {target_layer} reached engine layer {layer_index}"
                    )],
                    ..LayerPrefetchReport::default()
                }
            }
            None => LayerPrefetchReport::default(),
        };
        let prefetch_wait_us = if prefetch_report.requested_bytes == 0 {
            0
        } else {
            elapsed_us(prefetch_wait_started)
        };
        if !prefetch_report.errors.is_empty() {
            eprintln!(
                "calibrate: layer {layer_index} prefetch completed {}/{} bytes with {} error(s): {}",
                prefetch_report.completed_bytes,
                prefetch_report.requested_bytes,
                prefetch_report.errors.len(),
                prefetch_report.errors.join("; "),
            );
        }
        let load_started = Instant::now();
        let (mut layer, source_load) = {
            let mut reader = if prefetch_report.staging.is_empty() {
                PlannedTensorReader::new(source, &mut ledger, TensorOwner::Layer(layer_index))
            } else {
                PlannedTensorReader::new_with_staging(
                    source,
                    &mut ledger,
                    TensorOwner::Layer(layer_index),
                    &prefetch_report.staging,
                )
            };
            let layer = adapter.load_layer(
                &mut reader,
                gpu,
                &self.model,
                layer_index,
                &self.execution_job,
            )?;
            (layer, reader.timings())
        };
        let prefetch_staged_bytes = prefetch_report.staging.byte_len();
        // Layer weights now own their GPU copies; release the potentially
        // multi-gigabyte host staging before teacher execution begins.
        prefetch_report.staging = Default::default();
        let load_upload_us = elapsed_us(load_started);
        let prefetch_submit_started = Instant::now();
        let next_layer = layer_index + 1;
        let pausing_after_this_layer = self.pause_after_layers == Some(next_layer);
        let mut next_prefetch_disabled_reason = None;
        if next_layer < self.model.num_layers {
            if pausing_after_this_layer {
                next_prefetch_disabled_reason = Some("pause_boundary".into());
            } else if self.layer_prefetch_bytes == 0 {
                next_prefetch_disabled_reason = Some("disabled_by_configuration".into());
            } else {
                let layer_source_bytes = self.tensor_plan.bytes_for(TensorOwner::Layer(next_layer));
                let decision = layer_prefetch_decision(
                    self.layer_prefetch_bytes,
                    layer_source_bytes,
                    host_memory_snapshot(),
                );
                let ranges = self
                    .tensor_plan
                    .prefetch_ranges_for(TensorOwner::Layer(next_layer), decision.bytes);
                if !ranges.is_empty() {
                    match LayerPrefetch::spawn(ranges) {
                        Ok(prefetch) => self.pending_prefetch = Some((next_layer, prefetch)),
                        Err(error) => {
                            next_prefetch_disabled_reason = Some("worker_spawn_failed".into());
                            eprintln!(
                                "calibrate: layer {next_layer} prefetch disabled for this transition: {error}"
                            );
                        }
                    }
                } else {
                    let reason = decision
                        .disabled_reason
                        .unwrap_or("no_complete_tensor_fits_budget");
                    next_prefetch_disabled_reason = Some(reason.to_string());
                    eprintln!(
                        "calibrate: layer {next_layer} prefetch disabled for this transition: {reason}"
                    );
                }
            }
        }
        let prefetch_submit_us = elapsed_us(prefetch_submit_started);
        let execute_started = Instant::now();
        let execute_result = (|| {
            for batch in &self.batches {
                let input = self.boundary.read_active_indexed(&batch.boundary_rows)?;
                let mut output = vec![0.0f32; input.len()];
                layer.execute(gpu, batch, &input, &mut output, &self.capture)?;
                self.boundary
                    .write_next_indexed(&batch.boundary_rows, &output)?;
            }
            Ok::<(), CalibError>(())
        })();
        let execute_us = elapsed_us(execute_started);
        let part_path = calibration_part_path(&self.artifact_output, layer_index);
        let capture_started = Instant::now();
        let capture_result = if execute_result.is_ok() {
            let part_metadata = serde_json::json!({
                "artifact_kind": "calibration-part",
                "family": self.model.family,
                "layer": layer_index,
                "sample_fingerprint": job.samples.fingerprint(),
                "expert_capture_target": job.options.expert_quota.target_rows,
                "expert_capture_limit": job.options.expert_quota.limit_rows()?,
                "expert_capture_tile_rows": job.options.expert_quota.tile_rows,
            })
            .to_string();
            layer.write_capture_part(gpu, &part_path, self.model.arch_id, &part_metadata)
        } else {
            Err(CalibError::Runtime(
                "layer execution failed before capture part assembly".into(),
            ))
        };
        let capture_write_us = elapsed_us(capture_started);
        let finish_started = Instant::now();
        let finish_result = layer.finish(gpu);
        let finish_us = elapsed_us(finish_started);
        execute_result?;
        let capture_summary = capture_result?;
        finish_result?;
        let layer_descriptors = capture_summary.descriptors;
        let layer_telemetry = capture_summary.expert_telemetry;
        let mut layer_preserve = Vec::new();
        if let Some(telemetry) = layer_telemetry.as_ref() {
            let outcome = telemetry
                .coverage_report(
                    job.options.expert_coverage_policy,
                    job.options.required_expert_fraction,
                )
                .finalize()?;
            layer_preserve = outcome.preserve_high_precision;
        }
        let part_sync_started = Instant::now();
        sync_file(&part_path)?;
        let part_bytes = fs::metadata(&part_path)
            .map_err(|error| CalibError::Checkpoint(error.to_string()))?
            .len();
        let part_hash = file_hash(&part_path).unwrap_or_else(|| "unavailable".into());
        let part_file = part_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        let part_sync_hash_us = elapsed_us(part_sync_started);
        let timing = CalibrationLayerTiming {
            layer: layer_index,
            prefetch_wait_us,
            prefetch_read_us: prefetch_report.elapsed_us,
            prefetch_bytes: prefetch_report.completed_bytes,
            prefetch_staged_bytes,
            prefetch_errors: prefetch_report.errors.len(),
            prefetch_submit_us,
            next_prefetch_disabled_reason,
            source_tensor_count: source_load.tensor_count,
            source_bytes: source_load.source_bytes,
            gpu_upload_bytes: source_load.gpu_upload_bytes,
            staged_source_tensor_count: source_load.staged_tensor_count,
            staged_source_bytes: source_load.staged_source_bytes,
            source_view_us: source_load.view_us,
            source_decode_us: source_load.decode_us,
            source_upload_us: source_load.upload_us,
            source_release_us: source_load.release_us,
            load_upload_us,
            execute_us,
            capture_write_us,
            finish_us,
            part_sync_hash_us,
            total_before_checkpoint_us: elapsed_us(layer_started),
        };
        let progress = CalibrationLayerProgress {
            schema_version: CALIBRATION_PROGRESS_SCHEMA_VERSION,
            engine_build: self.engine_build.clone(),
            run_fingerprint: self.run_fingerprint.clone(),
            layer: layer_index,
            part_file,
            part_bytes,
            part_hash,
            descriptors: layer_descriptors.clone(),
            expert_telemetry: layer_telemetry.clone(),
            preserve_high_precision: layer_preserve.clone(),
            max_consistency: capture_summary.max_consistency,
            read_ledger: ledger.snapshot(),
            timing: timing.clone(),
        };
        self.ledger_snapshot = ledger.snapshot();
        write_layer_progress(&self.artifact_output, &progress)?;
        drop(layer);
        self.boundary.commit_layer(layer_index)?;
        if let Some(probe) = self.residual_probe.as_mut() {
            probe.push_layer(
                layer_index,
                self.boundary.read_active_rows(0, probe.row_count())?,
            )?;
        }
        self.max_consistency = self.max_consistency.max(capture_summary.max_consistency);
        self.descriptors.extend(layer_descriptors);
        if let Some(telemetry) = layer_telemetry {
            self.expert_telemetry.push(telemetry);
        }
        self.preserve_high_precision.extend(layer_preserve);
        self.part_paths.push(part_path);
        self.layer_timings.push(timing);
        self.next_layer = next_layer;
        let completed = self.boundary.checkpoint().completed_layers;
        let remaining = self.model.num_layers.saturating_sub(completed);
        let latest = self
            .layer_timings
            .last()
            .expect("the committed layer timing was just appended");
        let eta = estimated_remaining_layer_us(&self.layer_timings, remaining)
            .map(format_duration_us)
            .unwrap_or_else(|| "unknown".to_string());
        eprintln!(
            "calibrate: committed layer {completed}/{} in {} (prefetch {} read/{} wait, load {} [view {}, decode {}, upload {}, release {}], execute {}, capture+sync {}); rolling layer ETA {eta}",
            self.model.num_layers,
            format_duration_us(latest.total_before_checkpoint_us),
            format_duration_us(latest.prefetch_read_us),
            format_duration_us(latest.prefetch_wait_us),
            format_duration_us(latest.load_upload_us),
            format_duration_us(latest.source_view_us),
            format_duration_us(latest.source_decode_us),
            format_duration_us(latest.source_upload_us),
            format_duration_us(latest.source_release_us),
            format_duration_us(latest.execute_us),
            format_duration_us(
                latest
                    .capture_write_us
                    .saturating_add(latest.part_sync_hash_us)
            ),
        );
        if self.pause_after_layers == Some(completed) {
            return Ok(CalibrationStep::Paused);
        }
        Ok(CalibrationStep::Advanced)
    }

    /// Layers whose boundary has been committed so far — the resumable
    /// checkpoint frontier. Used by the daemon op to report progress between
    /// parked turns without exposing the session's internals.
    pub fn completed_layers(&self) -> usize {
        self.boundary.checkpoint().completed_layers
    }

    /// Total layers in the model under calibration.
    pub fn num_layers(&self) -> usize {
        self.model.num_layers
    }

    /// Consume a session that [`Self::step`] reported as `Paused`.
    pub fn into_paused(self) -> CalibrationPauseResult {
        CalibrationPauseResult {
            model: self.model,
            tensor_plan: self.tensor_plan,
            read_ledger: self.ledger_snapshot,
            boundary_checkpoint: self.boundary.checkpoint().clone(),
            artifact_path: self.artifact_output,
            geometry: self.geometry,
            geometry_tuning: self.geometry_tuning,
            layer_timings: self.layer_timings,
        }
    }

    /// Run the KLD finalizer and assemble the artifact. Call once [`Self::step`]
    /// has reported `LayersComplete`.
    pub fn finish(
        mut self,
        adapter: &mut dyn CalibrationFamilyAdapter,
        source: &dyn ModelSource,
        gpu: &mut Gpu,
        job: &CalibrationJob,
    ) -> Result<CalibrationRunResult, CalibError> {
        // Final phase: reconstitute once more for the finalizer's planned reads.
        let mut ledger = ReadLedger::resume(&self.tensor_plan, self.ledger_snapshot.clone())?;
        let kldref = if job.options.kldref {
            let mut finalizer = {
                let mut reader =
                    PlannedTensorReader::new(source, &mut ledger, TensorOwner::Persistent);
                adapter.load_finalizer(&mut reader, gpu, &self.model, &self.execution_job)?
            };
            let mut builder = KldRefBuilder::new(job.options.kldref_top_k)?;
            // Row subsetting has to happen BEFORE the finalizer, not after: the
            // cost is the full-vocabulary lm-head projection, so dropping rows
            // from the output would save nothing. `kldref_rows` selects a
            // deterministic even stride over the boundary row index, spreading
            // the reference across every sample and position rather than taking
            // a contiguous head.
            let kld_stride = kldref_row_stride(job.samples.total_rows(), job.options.kldref_rows);
            let execute_result = (|| {
                for batch in &self.batches {
                    let (batch, residual) = if kld_stride > 1 {
                        let Some(reduced) = subset_batch_by_stride(batch, kld_stride) else {
                            continue;
                        };
                        let residual = self.boundary.read_active_indexed(&reduced.boundary_rows)?;
                        (std::borrow::Cow::Owned(reduced), residual)
                    } else {
                        let residual = self.boundary.read_active_indexed(&batch.boundary_rows)?;
                        (std::borrow::Cow::Borrowed(batch), residual)
                    };
                    let mut rows = Vec::new();
                    finalizer.execute_kld(
                        gpu,
                        &batch,
                        &residual,
                        job.options.kldref_top_k,
                        &mut rows,
                    )?;
                    validate_kld_batch_rows(&batch, &rows)?;
                    for row in rows {
                        if kld_row_has_next_token(job, &row)? {
                            builder.push(row)?;
                        }
                    }
                }
                Ok::<(), CalibError>(())
            })();
            let finish_result = finalizer.finish(gpu);
            execute_result?;
            finish_result?;
            Some(builder.finish()?)
        } else {
            // The finalizer may still own planned final-norm/lm-head tensors.
            let mut reader = PlannedTensorReader::new(source, &mut ledger, TensorOwner::Persistent);
            let mut finalizer =
                adapter.load_finalizer(&mut reader, gpu, &self.model, &self.execution_job)?;
            finalizer.finish(gpu)?;
            None
        };

        let extra_tensors = kldref
            .as_ref()
            .map(KldRefPayload::to_hfq_tensors)
            .unwrap_or_default();
        self.boundary.finalize_kld()?;
        ledger.assert_complete()?;
        let read_ledger = ledger.snapshot();
        let static_meta = vec![
            ("artifact_kind", serde_json::json!("calibration")),
            ("engine_build", serde_json::json!(self.engine_build)),
            ("family", serde_json::json!(self.model.family)),
            (
                "adapter_version",
                serde_json::json!(adapter.adapter_version()),
            ),
            ("arch_id", serde_json::json!(self.model.arch_id)),
            (
                "source_manifest",
                serde_json::to_value(&self.source_manifest)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            ),
            ("effective_precision", self.effective_precision.clone()),
            (
                "resource_estimate",
                serde_json::to_value(&self.resource_estimate)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            ),
            (
                "microbatch_geometry",
                serde_json::to_value(self.geometry)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            ),
            (
                "geometry_tuning",
                serde_json::to_value(&self.geometry_tuning)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            ),
            (
                "layer_timings",
                serde_json::to_value(&self.layer_timings)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            ),
            ("run_fingerprint", serde_json::json!(self.run_fingerprint)),
            ("max_consistency", serde_json::json!(self.max_consistency)),
            (
                "job",
                serde_json::to_value(job)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            ),
            (
                "expert_telemetry",
                serde_json::to_value(&self.expert_telemetry)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            ),
            (
                "expert_capture_quota",
                serde_json::json!({
                    "minimum_rows": job.options.expert_quota.min_rows,
                    "target_rows": job.options.expert_quota.target_rows,
                    "limit_rows": job.options.expert_quota.limit_rows()?,
                    "tile_rows": job.options.expert_quota.tile_rows,
                    "maximum_batch_slack_rows": job.options.expert_quota.tile_rows - 1,
                    "sampling": job.options.expert_quota.sampling,
                }),
            ),
            (
                "preserve_high_precision",
                serde_json::to_value(&self.preserve_high_precision)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            ),
            (
                "read_ledger",
                serde_json::to_value(&read_ledger)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?,
            ),
        ];
        let static_meta_refs = static_meta
            .iter()
            .map(|(key, value)| (*key, value.clone()))
            .collect::<Vec<_>>();
        let extra_meta = kldref
            .as_ref()
            .map(|payload| vec![("kldref".to_string(), payload.metadata())])
            .unwrap_or_default();
        let metadata =
            build_calibration_metadata(&self.descriptors, Some(1), &static_meta_refs, &extra_meta)
                .map_err(CalibError::Runtime)?;
        let assembling_path = assembling_artifact_path(&self.artifact_output);
        if assembling_path.exists() {
            if self.resuming_existing_checkpoint {
                remove_if_exists(&assembling_path)?;
            } else {
                return Err(CalibError::Checkpoint(format!(
                    "stale assembling artifact already exists at {}",
                    assembling_path.display()
                )));
            }
        }
        combine_calib_parts(
            &assembling_path,
            self.model.arch_id,
            &metadata.json,
            &self.part_paths,
            &extra_tensors,
        )
        .map_err(|error| CalibError::Runtime(error.to_string()))?;
        if let (Some(path), Some(probe)) = (&self.residual_probe_output, &self.residual_probe) {
            if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
                fs::create_dir_all(parent)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?;
            }
            let probe_assembling = assembling_residual_probe_path(path);
            remove_if_exists(&probe_assembling)?;
            let write_result = (|| {
                probe.write(&probe_assembling)?;
                sync_file(&probe_assembling)?;
                fs::rename(&probe_assembling, path)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?;
                sync_parent_directory(path)
            })();
            if let Err(error) = write_result {
                let _ = remove_if_exists(&probe_assembling);
                let _ = remove_if_exists(path);
                return Err(error);
            }
        }
        if let Err(error) = std::fs::rename(&assembling_path, &self.artifact_output) {
            if let Some(path) = &self.residual_probe_output {
                let _ = remove_if_exists(path);
            }
            return Err(CalibError::Runtime(error.to_string()));
        }
        sync_file(&self.artifact_output)?;
        sync_parent_directory(&self.artifact_output)?;
        drop(ledger);
        self.boundary.finalize_artifact()?;
        cleanup_calibration_spools(&self.artifact_output, self.model.num_layers)?;
        let kldref_positions = kldref.as_ref().map(KldRefPayload::n_positions);
        Ok(CalibrationRunResult {
            model: self.model,
            tensor_plan: self.tensor_plan,
            read_ledger,
            boundary_checkpoint: self.boundary.checkpoint().clone(),
            kldref,
            kldref_positions,
            artifact_path: self.artifact_output,
            artifact: CalibSummary {
                n_hessian: metadata.n_hessian,
                n_imatrix: metadata.n_imatrix,
                max_consistency: self.max_consistency,
            },
            expert_telemetry: self.expert_telemetry,
            geometry: self.geometry,
            geometry_tuning: self.geometry_tuning,
            layer_timings: self.layer_timings,
        })
    }
}

fn validate_kld_batch_rows(
    batch: &crate::calibration::schedule::LayerMicrobatch,
    rows: &[KldRefRow],
) -> Result<(), CalibError> {
    if rows.len() != batch.rows.len() {
        return Err(CalibError::InvalidKldRef(format!(
            "finalizer returned {} rows for a {}-row scheduler batch",
            rows.len(),
            batch.rows.len()
        )));
    }
    for (expected, actual) in batch.rows.iter().zip(rows) {
        if expected.sample_index != actual.sample_index || expected.position != actual.position {
            return Err(CalibError::InvalidKldRef(format!(
                "finalizer row {}/{} does not match scheduler row {}/{}",
                actual.sample_index, actual.position, expected.sample_index, expected.position
            )));
        }
    }
    Ok(())
}

/// Stride between retained KLDREF rows. `None` (or a target at least as large as
/// the corpus) keeps every row, matching the historical behaviour.
pub fn kldref_row_stride(total_rows: usize, target: Option<usize>) -> usize {
    match target {
        Some(target) if target > 0 && target < total_rows => total_rows.div_ceil(target),
        _ => 1,
    }
}

/// Keep the rows of `batch` whose boundary index is on the stride. Returns `None`
/// when the stride selects nothing from this batch, which the caller skips
/// entirely so no lm-head projection runs for it.
fn subset_batch_by_stride(
    batch: &crate::calibration::schedule::LayerMicrobatch,
    stride: usize,
) -> Option<crate::calibration::schedule::LayerMicrobatch> {
    let mut rows = Vec::new();
    let mut boundary_rows = Vec::new();
    for (row, &boundary) in batch.rows.iter().zip(batch.boundary_rows.iter()) {
        if boundary % stride == 0 {
            rows.push(row.clone());
            boundary_rows.push(boundary);
        }
    }
    if rows.is_empty() {
        return None;
    }
    Some(crate::calibration::schedule::LayerMicrobatch {
        sequence_start: batch.sequence_start,
        sequence_end: batch.sequence_end,
        time_start: batch.time_start,
        time_end: batch.time_end,
        rows,
        boundary_rows,
    })
}

fn kld_row_has_next_token(job: &CalibrationJob, row: &KldRefRow) -> Result<bool, CalibError> {
    let sample = job.samples.samples().get(row.sample_index).ok_or_else(|| {
        CalibError::InvalidKldRef(format!(
            "KLD row references missing sample {}",
            row.sample_index
        ))
    })?;
    if row.position >= sample.tokens.len() {
        return Err(CalibError::InvalidKldRef(format!(
            "KLD row position {} is outside sample {} length {}",
            row.position,
            row.sample_index,
            sample.tokens.len()
        )));
    }
    Ok(row.position + 1 < sample.tokens.len())
}

fn calibration_part_path(output: &Path, layer: usize) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("calib.hfq");
    output.with_file_name(format!(".{name}.layer-{layer:04}.hfq"))
}

fn assembling_artifact_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("calib.hfq");
    output.with_file_name(format!(".{name}.assembling"))
}

fn assembling_residual_probe_path(output: &Path) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("residuals.hfq");
    output.with_file_name(format!(".{name}.assembling"))
}

pub fn resolve_geometry(job: &CalibrationJob) -> Result<MicrobatchGeometry, CalibError> {
    match (job.options.sequence_batch, job.options.time_tile) {
        (Some(sequence_batch), Some(time_tile)) => Ok(MicrobatchGeometry {
            sequence_batch,
            time_tile,
            row_budget: job.options.max_rows,
        }),
        (Some(sequence_batch), None) => Ok(MicrobatchGeometry {
            sequence_batch,
            time_tile: (job.options.max_rows / sequence_batch).max(1),
            row_budget: job.options.max_rows,
        }),
        (None, Some(time_tile)) => Ok(MicrobatchGeometry {
            sequence_batch: (job.options.max_rows / time_tile).max(1),
            time_tile,
            row_budget: job.options.max_rows,
        }),
        (None, None) => auto_geometry_candidates(job)?
            .last()
            .copied()
            .ok_or_else(|| CalibError::InvalidOptions("no automatic geometry candidate".into())),
    }
}

fn execution_job(job: &CalibrationJob, geometry: MicrobatchGeometry) -> CalibrationJob {
    let mut execution = job.clone();
    execution.options.sequence_batch = Some(geometry.sequence_batch);
    execution.options.time_tile = Some(geometry.time_tile);
    execution.options.max_rows = geometry.sequence_batch * geometry.time_tile;
    execution
}

fn auto_geometry_candidates(job: &CalibrationJob) -> Result<Vec<MicrobatchGeometry>, CalibError> {
    let sequence_cap = job.samples.samples().len().min(job.options.max_rows);
    let mut sequence_values = if let Some(sequence_batch) = job.options.sequence_batch {
        vec![sequence_batch]
    } else {
        vec![1usize]
    };
    if job.options.sequence_batch.is_none() {
        while sequence_values.last().copied().unwrap_or(1) < sequence_cap {
            let next = (sequence_values.last().copied().unwrap() * 2).min(sequence_cap);
            if next == *sequence_values.last().unwrap() {
                break;
            }
            sequence_values.push(next);
        }
    }
    let mut time_values = if let Some(time_tile) = job.options.time_tile {
        vec![time_tile]
    } else {
        vec![1usize]
    };
    if job.options.time_tile.is_none() {
        while time_values.last().copied().unwrap_or(1) < job.options.max_rows {
            let next = (time_values.last().copied().unwrap() * 2).min(job.options.max_rows);
            if next == *time_values.last().unwrap() {
                break;
            }
            time_values.push(next);
        }
    }
    let mut candidates = Vec::new();
    for sequence_batch in sequence_values {
        for &time_tile in &time_values {
            let geometry = MicrobatchGeometry {
                sequence_batch,
                time_tile,
                row_budget: job.options.max_rows,
            };
            if geometry.validate().is_ok() {
                candidates.push(geometry);
            }
        }
    }
    candidates.sort_by_key(|geometry| {
        (
            geometry.sequence_batch * geometry.time_tile,
            geometry.sequence_batch,
            geometry.time_tile,
        )
    });
    candidates.dedup();
    if candidates.is_empty() {
        return Err(CalibError::InvalidOptions(
            "no automatic calibration microbatch candidates fit the row budget".into(),
        ));
    }
    Ok(candidates)
}

/// The VRAM ceiling the automatic geometry search sizes candidates against.
struct VramCeiling {
    usable_bytes: Option<u64>,
    reserved_headroom_bytes: u64,
    source: &'static str,
}

/// Decide that ceiling from a declared budget when there is one, and only fall
/// back to the live probe when there is not.
///
/// A `get_vram_info()` probe is accurate only at the instant it runs. With a
/// daemon resident on the same GPU, free VRAM sampled before the daemon grows
/// its KV cache picks a geometry that collides later in the run — and a
/// calibration run is hours long. `HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES` /
/// `_HEADROOM_BYTES` are the same pair the daemon's `ResourceReservationManager`
/// reads, so declaring them makes both sides size against one number instead of
/// two independent guesses.
///
/// This is a ceiling on which candidates get *estimated*, not a replacement for
/// the live allocation probe below — that still runs and remains the hard check.
fn vram_ceiling(free_vram: Option<u64>, mut get: impl FnMut(&str) -> Option<u64>) -> VramCeiling {
    let budget = get("HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES").unwrap_or(0);
    if budget > 0 {
        let headroom = get("HIPFIRE_SCHEDULER_VRAM_HEADROOM_BYTES").unwrap_or(0);
        return VramCeiling {
            usable_bytes: Some(budget.saturating_sub(headroom)),
            reserved_headroom_bytes: headroom,
            source: "scheduler_budget",
        };
    }
    // No declared budget: the historical behaviour — hold back 5% of what is
    // free right now, at least 1 GiB.
    let reserved = free_vram
        .map(|free| (free / 20).max(1u64 << 30))
        .unwrap_or(0);
    VramCeiling {
        usable_bytes: free_vram.map(|free| free.saturating_sub(reserved)),
        reserved_headroom_bytes: reserved,
        source: if free_vram.is_some() {
            "live_probe"
        } else {
            "unavailable"
        },
    }
}

fn scheduler_budget_env(key: &str) -> Option<u64> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
}

fn tune_geometry_for_gpu(
    adapter: &dyn CalibrationFamilyAdapter,
    model: &ModelInspection,
    tensor_plan: &TensorLoadPlan,
    job: &CalibrationJob,
    gpu: &mut Gpu,
) -> Result<GeometryTuningReport, CalibError> {
    let free_vram = gpu.hip.get_vram_info().ok().map(|(free, _)| free as u64);
    let ceiling = vram_ceiling(free_vram, scheduler_budget_env);

    if job.options.sequence_batch.is_some() && job.options.time_tile.is_some() {
        let selected = resolve_geometry(job)?;
        return Ok(GeometryTuningReport {
            automatic: false,
            selected,
            free_vram_bytes: free_vram,
            reserved_headroom_bytes: 0,
            vram_ceiling_source: ceiling.source.to_string(),
            probes: Vec::new(),
        });
    }

    let reserved_headroom = ceiling.reserved_headroom_bytes;
    let usable_vram = ceiling.usable_bytes;
    let max_layer_bytes = (0..model.num_layers)
        .map(|layer| tensor_plan.bytes_for(TensorOwner::Layer(layer)))
        .max()
        .unwrap_or(0);
    let mut selected = None;
    let mut probes = Vec::new();
    for geometry in auto_geometry_candidates(job)? {
        let candidate_job = execution_job(job, geometry);
        let estimate = adapter.resource_estimate(model, &candidate_job, geometry)?;
        let Some(estimate) = estimate else {
            selected = Some(geometry);
            probes.push(GeometryCandidateProbe {
                geometry,
                requested_bytes: 0,
                accepted: true,
                reason: "adapter has no allocation estimate; selected by row geometry".into(),
            });
            continue;
        };
        let requested = max_layer_bytes
            .checked_add(estimate.scratch_bytes)
            .and_then(|bytes| bytes.checked_add(estimate.active_state_bytes))
            .ok_or_else(|| CalibError::InvalidOptions("geometry probe byte overflow".into()))?;
        if usable_vram.is_some_and(|usable| requested > usable) {
            probes.push(GeometryCandidateProbe {
                geometry,
                requested_bytes: requested,
                accepted: false,
                reason: format!(
                    "estimated layer+scratch+state exceeds usable VRAM ({requested} > {})",
                    usable_vram.unwrap()
                ),
            });
            continue;
        }
        let allocation_size = usize::try_from(requested).map_err(|_| {
            CalibError::InvalidOptions("geometry probe does not fit host usize".into())
        })?;
        match gpu.hip.malloc(allocation_size) {
            Ok(buffer) => {
                gpu.hip
                    .free(buffer)
                    .map_err(|error| CalibError::Runtime(error.to_string()))?;
                selected = Some(geometry);
                probes.push(GeometryCandidateProbe {
                    geometry,
                    requested_bytes: requested,
                    accepted: true,
                    reason: "live allocation probe succeeded".into(),
                });
            }
            Err(error) => probes.push(GeometryCandidateProbe {
                geometry,
                requested_bytes: requested,
                accepted: false,
                reason: format!("live allocation probe failed: {error}"),
            }),
        }
    }
    let selected = selected.ok_or_else(|| {
        CalibError::Runtime(format!(
            "no automatic calibration geometry fit the VRAM ceiling ({} usable, source {}); \
             lower --max-rows or raise HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES",
            usable_vram
                .map(|bytes| bytes.to_string())
                .unwrap_or_else(|| "unknown".into()),
            ceiling.source,
        ))
    })?;
    Ok(GeometryTuningReport {
        automatic: true,
        selected,
        free_vram_bytes: free_vram,
        reserved_headroom_bytes: reserved_headroom,
        vram_ceiling_source: ceiling.source.to_string(),
        probes,
    })
}

/// Everything [`LayerStreamEngine::begin`] borrows, built once from a parsed
/// [`CalibrateCommand`]. Shared by the CLI (`run_cli`) and the daemon op
/// ([`DaemonCalibration`]) so both feed the engine byte-identical inputs — the
/// factored form of the CLI's original inline source/adapter/job construction.
pub struct CalibrationRunInputs {
    pub snapshot: PathBuf,
    pub source: SafetensorsSource,
    pub adapter: Box<dyn CalibrationFamilyAdapter>,
    pub adapter_family: &'static str,
    pub adapter_version: &'static str,
    pub job: CalibrationJob,
    pub source_manifest: SourceManifestIdentity,
}

/// Resolve the model snapshot, tokenize the corpus, and build the calibration
/// job + native adapter. No GPU, no lock — pure, deterministic input
/// construction, identical for the CLI and daemon paths (that identity is what
/// makes the daemon-op artifact byte-for-byte the CLI's).
pub fn build_calibration_run_inputs(
    command: &CalibrateCommand,
) -> Result<CalibrationRunInputs, Box<dyn Error>> {
    let snapshot = resolve_hf_snapshot(&command.model)?;
    let source = SafetensorsSource::open(&snapshot)?;
    let tokenizer_path = source
        .tokenizer_json_path()
        .ok_or_else(|| format!("{} has no tokenizer.json", snapshot.display()))?;
    let tokenizer_json = fs::read_to_string(&tokenizer_path)?;
    let tokenizer = Tokenizer::from_hf_json(&tokenizer_json)?;
    let samples = load_corpus_samples(
        &command.corpus,
        &tokenizer,
        command.sequences,
        command.context,
    )?;
    let options = command.options()?;
    let source_manifest = source_manifest_identity(&source)?;
    let corpus_fingerprint = file_hash(&command.corpus).unwrap_or_else(|| "unavailable".into());
    let job = CalibrationJob::new(
        source_manifest.fingerprint.clone(),
        file_hash(&tokenizer_path).unwrap_or_else(|| "unavailable".into()),
        SampleSet::new(samples, command.context, command.sampling_seed)?,
        options,
    )?
    .with_corpus_fingerprint(corpus_fingerprint)?;
    let resolved = resolve_adapter(&source)?;
    Ok(CalibrationRunInputs {
        snapshot,
        source,
        adapter: resolved.adapter,
        adapter_family: resolved.family,
        adapter_version: resolved.version,
        job,
        source_manifest,
    })
}

/// Build the boundary backend the CLI and daemon both derive from a command.
pub fn boundary_backend_for(command: &CalibrateCommand) -> BoundaryBackend {
    if command.boundary_ram {
        BoundaryBackend::Ram
    } else {
        BoundaryBackend::Mmap {
            directory: command
                .boundary_directory
                .clone()
                .unwrap_or_else(|| default_boundary_directory(&command.output)),
        }
    }
}

fn engine_for(command: &CalibrateCommand) -> LayerStreamEngine {
    LayerStreamEngine::new(boundary_backend_for(command), &command.output)
        .with_resume(command.resume)
        .with_pause_after_layers(command.pause_after_layers)
        .with_layer_prefetch_bytes(command.layer_prefetch_bytes)
        .with_residual_probe(
            command.residual_probe_output.clone(),
            command.residual_probe_rows,
        )
}

/// A layer-stream calibration parked as a daemon session.
///
/// This owns everything the engine borrows — the boxed native adapter, the
/// safetensors source, and the calibration job — alongside the live
/// [`CalibrationSession`]. Nothing here is borrowed from a caller, so the daemon
/// parks the whole thing in `DaemonState` between GPU turns by move, then
/// borrows the fields disjointly on each [`Self::step`]. It runs **no** GPU
/// self-lock: the daemon already holds the single process-lifetime GPU lease and
/// drives this on its serial executor. The daemon-free CLI (`run_cli`) keeps its
/// own `acquire_gpu_lock`.
pub struct DaemonCalibration {
    adapter: Box<dyn CalibrationFamilyAdapter>,
    source: SafetensorsSource,
    job: CalibrationJob,
    session: CalibrationSession,
    output: PathBuf,
    num_layers: usize,
    family: String,
}

/// What [`DaemonCalibration::begin`] produced: an artifact recovered from a
/// prior completed run, or a parked session ready for its first
/// [`step`](DaemonCalibration::step).
pub enum DaemonCalibrationStart {
    Complete(Box<CalibrationRunResult>),
    Session(DaemonCalibration),
}

impl DaemonCalibration {
    /// Build the same inputs the CLI builds and run [`LayerStreamEngine::begin`]
    /// on the daemon's resident GPU. No lock is taken here (see the type doc).
    pub fn begin(
        command: &CalibrateCommand,
        gpu: &mut Gpu,
    ) -> Result<DaemonCalibrationStart, Box<dyn Error>> {
        let inputs = build_calibration_run_inputs(command)?;
        let CalibrationRunInputs {
            source,
            mut adapter,
            job,
            ..
        } = inputs;
        if let Some(parent) = command
            .output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        match engine_for(command).begin(adapter.as_mut(), &source, gpu, &job)? {
            CalibrationSessionStart::Complete(result) => {
                Ok(DaemonCalibrationStart::Complete(Box::new(result)))
            }
            CalibrationSessionStart::Session(session) => {
                let num_layers = session.num_layers();
                let family = session.model.family.clone();
                Ok(DaemonCalibrationStart::Session(DaemonCalibration {
                    adapter,
                    source,
                    job,
                    session,
                    output: command.output.clone(),
                    num_layers,
                    family,
                }))
            }
        }
    }

    /// Run exactly one layer — the calibration quantum. One daemon GPU turn.
    pub fn step(&mut self, gpu: &mut Gpu) -> Result<CalibrationStep, CalibError> {
        self.session
            .step(self.adapter.as_mut(), &self.source, gpu, &self.job)
    }

    /// Consume a session that [`Self::step`] reported as `Paused`.
    pub fn into_paused(self) -> CalibrationPauseResult {
        self.session.into_paused()
    }

    /// Finalize once [`Self::step`] reports `LayersComplete`.
    pub fn finish(self, gpu: &mut Gpu) -> Result<CalibrationRunResult, CalibError> {
        let DaemonCalibration {
            mut adapter,
            source,
            job,
            session,
            ..
        } = self;
        session.finish(adapter.as_mut(), &source, gpu, &job)
    }

    pub fn output(&self) -> &Path {
        &self.output
    }
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    pub fn completed_layers(&self) -> usize {
        self.session.completed_layers()
    }
    pub fn family(&self) -> &str {
        &self.family
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::contracts::{CalibrationOptions, CalibrationSample, SampleSet};
    use crate::hfq::{write_hfqm_package_mem, HfqMemTensor};

    fn job(options: CalibrationOptions) -> CalibrationJob {
        CalibrationJob::new(
            "source",
            "tokenizer",
            SampleSet::new(
                vec![CalibrationSample::new("one", vec![1, 2, 3], "text")],
                8,
                1,
            )
            .unwrap(),
            options,
        )
        .unwrap()
    }

    #[test]
    fn declared_scheduler_budget_outranks_the_live_vram_probe() {
        let env = |key: &str| match key {
            "HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES" => Some(16u64 << 30),
            "HIPFIRE_SCHEDULER_VRAM_HEADROOM_BYTES" => Some(2u64 << 30),
            _ => None,
        };
        // A live probe reporting 100 GiB free must not widen the ceiling past
        // the declared budget: this is the whole point of the change, since the
        // daemon's own reservation is invisible to `get_vram_info()`.
        let ceiling = vram_ceiling(Some(100 << 30), env);
        assert_eq!(ceiling.usable_bytes, Some(14 << 30));
        assert_eq!(ceiling.reserved_headroom_bytes, 2 << 30);
        assert_eq!(ceiling.source, "scheduler_budget");

        // The budget also applies when no probe is available at all.
        assert_eq!(vram_ceiling(None, env).usable_bytes, Some(14 << 30));
    }

    #[test]
    fn without_a_budget_the_ceiling_keeps_the_historical_probe_heuristic() {
        let none = |_: &str| None;
        // 5% held back, floored at 1 GiB.
        let ceiling = vram_ceiling(Some(100 << 30), none);
        assert_eq!(ceiling.reserved_headroom_bytes, 5 << 30);
        assert_eq!(ceiling.usable_bytes, Some(95 << 30));
        assert_eq!(ceiling.source, "live_probe");

        // Small card: the 1 GiB floor wins over 5%.
        let small = vram_ceiling(Some(8 << 30), none);
        assert_eq!(small.reserved_headroom_bytes, 1 << 30);
        assert_eq!(small.usable_bytes, Some(7 << 30));

        // No probe and no budget: no ceiling to apply, and say so.
        let blind = vram_ceiling(None, none);
        assert_eq!(blind.usable_bytes, None);
        assert_eq!(blind.source, "unavailable");
    }

    #[test]
    fn a_zero_budget_is_not_a_declaration() {
        // Unset parses as 0 in the daemon's reader too; 0 must mean "fall back",
        // never "no VRAM is usable".
        let zero = |key: &str| match key {
            "HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES" => Some(0u64),
            _ => None,
        };
        let ceiling = vram_ceiling(Some(8 << 30), zero);
        assert_eq!(ceiling.source, "live_probe");
        assert_eq!(ceiling.usable_bytes, Some(7 << 30));
    }

    #[test]
    fn a_budget_under_its_headroom_saturates_to_zero_rather_than_wrapping() {
        let inverted = |key: &str| match key {
            "HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES" => Some(1u64 << 30),
            "HIPFIRE_SCHEDULER_VRAM_HEADROOM_BYTES" => Some(4u64 << 30),
            _ => None,
        };
        assert_eq!(vram_ceiling(None, inverted).usable_bytes, Some(0));
    }

    // `linked_calibration_adapters_are_unique_and_resolve_by_architecture` lives
    // in the hipfire-coexistence tests: adapter registration is a property of a
    // binary that links the family crates as real dependencies (coexistence and
    // the daemon), not of this runtime lib, whose family crates are dev-deps a
    // unit-test binary does not reliably retain the inventory statics from.

    #[test]
    fn explicit_and_auto_geometry_obey_row_budget() {
        let mut options = CalibrationOptions::default();
        options.max_rows = 128;
        let auto = resolve_geometry(&job(options.clone())).unwrap();
        assert!(auto.sequence_batch * auto.time_tile <= 128);
        options.sequence_batch = Some(4);
        options.time_tile = Some(16);
        assert_eq!(
            resolve_geometry(&job(options)).unwrap(),
            MicrobatchGeometry {
                sequence_batch: 4,
                time_tile: 16,
                row_budget: 128,
            }
        );

        let mut partial = CalibrationOptions::default();
        partial.max_rows = 128;
        partial.sequence_batch = Some(4);
        let candidates = auto_geometry_candidates(&job(partial.clone())).unwrap();
        assert!(candidates
            .iter()
            .all(|geometry| geometry.sequence_batch == 4));
        let selected = *candidates.last().unwrap();
        assert_eq!(selected.sequence_batch * selected.time_tile, 128);
        let execution = execution_job(&job(partial), selected);
        assert_eq!(execution.options.max_rows, 128);
        assert_eq!(execution.options.sequence_batch, Some(4));
        assert_eq!(execution.options.time_tile, Some(32));
    }

    #[test]
    fn corpus_tokenization_window_scales_with_remaining_sample_geometry() {
        assert_eq!(corpus_tokenize_window_bytes(1, 0, 2), 256);
        assert_eq!(corpus_tokenize_window_bytes(8, 7, 128), 8 * 128);
        assert_eq!(
            corpus_tokenize_window_bytes(128, 0, 2048),
            CORPUS_TOKENIZE_WINDOW_BYTES
        );
        assert_eq!(corpus_tokenize_window_bytes(1, 1, 2), 256);
    }

    #[test]
    fn remaining_layer_eta_uses_persisted_completed_layer_mean() {
        let timings = [
            CalibrationLayerTiming {
                total_before_checkpoint_us: 10,
                ..CalibrationLayerTiming::default()
            },
            CalibrationLayerTiming {
                total_before_checkpoint_us: 20,
                ..CalibrationLayerTiming::default()
            },
        ];
        assert_eq!(estimated_remaining_layer_us(&timings, 3), Some(45));
        assert_eq!(estimated_remaining_layer_us(&[], 3), None);
        assert_eq!(format_duration_us(3_661_234_567), "1h01m01s");
    }

    #[test]
    fn layer_prefetch_budget_preserves_host_reserve_and_layer_bound() {
        let gib = 1024 * 1024 * 1024;
        let healthy = HostMemorySnapshot {
            available_bytes: Some(65 * gib),
            swap_total_bytes: Some(64 * gib),
            swap_free_bytes: Some(64 * gib),
            full_pressure_avg10: Some(0.0),
        };
        assert_eq!(
            layer_prefetch_decision(16 * gib, 13 * gib, healthy),
            LayerPrefetchDecision {
                bytes: 13 * gib,
                disabled_reason: None,
            }
        );
        let constrained = HostMemorySnapshot {
            available_bytes: Some(40 * gib),
            ..healthy
        };
        assert_eq!(
            layer_prefetch_decision(16 * gib, 13 * gib, constrained),
            LayerPrefetchDecision {
                bytes: 0,
                disabled_reason: Some("host_headroom_below_reserve_plus_upload"),
            }
        );
        assert_eq!(
            layer_prefetch_decision(16 * gib, 20 * gib, HostMemorySnapshot::default()).bytes,
            16 * gib,
        );
        assert_eq!(
            layer_prefetch_decision(
                16 * gib,
                13 * gib,
                HostMemorySnapshot {
                    full_pressure_avg10: Some(0.01),
                    ..healthy
                },
            )
            .disabled_reason,
            Some("memory_psi_full"),
        );
        assert_eq!(
            layer_prefetch_decision(
                16 * gib,
                13 * gib,
                HostMemorySnapshot {
                    swap_free_bytes: Some(12 * gib),
                    ..healthy
                },
            )
            .disabled_reason,
            Some("swap_free_below_25_percent"),
        );
        assert_eq!(
            layer_prefetch_decision(0, 20 * gib, healthy).disabled_reason,
            Some("disabled_by_configuration"),
        );
    }

    #[test]
    fn old_layer_timing_checkpoints_default_new_prefetch_fields() {
        let timing: CalibrationLayerTiming = serde_json::from_value(serde_json::json!({
            "layer": 3,
            "load_upload_us": 10,
            "execute_us": 20,
            "capture_write_us": 30,
            "finish_us": 40,
            "part_sync_hash_us": 50,
            "total_before_checkpoint_us": 150
        }))
        .unwrap();
        assert_eq!(timing.prefetch_wait_us, 0);
        assert_eq!(timing.prefetch_read_us, 0);
        assert_eq!(timing.prefetch_bytes, 0);
        assert_eq!(timing.prefetch_staged_bytes, 0);
        assert_eq!(timing.prefetch_errors, 0);
        assert_eq!(timing.next_prefetch_disabled_reason, None);
        assert_eq!(timing.source_tensor_count, 0);
        assert_eq!(timing.source_bytes, 0);
        assert_eq!(timing.gpu_upload_bytes, 0);
        assert_eq!(timing.staged_source_tensor_count, 0);
        assert_eq!(timing.staged_source_bytes, 0);
        assert_eq!(timing.source_view_us, 0);
        assert_eq!(timing.source_decode_us, 0);
        assert_eq!(timing.source_upload_us, 0);
        assert_eq!(timing.source_release_us, 0);
    }

    #[test]
    fn cli_parses_auto_geometry_and_unaligned_capture_target() {
        let args = [
            "--model",
            "model",
            "--corpus",
            "corpus.txt",
            "--output",
            "out.hfq",
            "--sequence-batch",
            "auto",
            "--time-tile",
            "8",
            "--expert-capture-target",
            "4100",
            "--expert-capture-tile-rows",
            "256",
            "--pause-after-layers",
            "2",
            "--dry-run",
        ]
        .map(str::to_string);
        let command = CalibrateCommand::parse(&args).unwrap();
        assert_eq!(command.sequence_batch, None);
        assert_eq!(command.time_tile, Some(8));
        assert_eq!(command.max_rows, 2048);
        assert_eq!(command.layer_prefetch_bytes, 16 * 1024 * 1024 * 1024);
        assert_eq!(
            command
                .options()
                .unwrap()
                .expert_quota
                .limit_rows()
                .unwrap(),
            4352
        );
        assert!(command.dry_run);
        assert_eq!(command.pause_after_layers, Some(2));
    }

    #[test]
    fn cli_accepts_an_explicit_layer_prefetch_budget() {
        let args = [
            "--model",
            "model",
            "--corpus",
            "corpus.txt",
            "--output",
            "out.hfq",
            "--layer-prefetch-bytes",
            "12345",
        ]
        .map(str::to_string);
        let command = CalibrateCommand::parse(&args).unwrap();
        assert_eq!(command.layer_prefetch_bytes, 12345);
    }

    #[test]
    fn cli_accepts_a_bounded_residual_probe_only_for_fresh_full_runs() {
        let base = [
            "--model",
            "model",
            "--corpus",
            "corpus.txt",
            "--output",
            "out.hfq",
            "--residual-probe-output",
            "residuals.hfq",
            "--residual-probe-rows",
            "8",
        ]
        .map(str::to_string);
        let command = CalibrateCommand::parse(&base).unwrap();
        assert_eq!(
            command.residual_probe_output,
            Some(PathBuf::from("residuals.hfq"))
        );
        assert_eq!(command.residual_probe_rows, 8);

        let mut resumed = base.to_vec();
        resumed.push("--resume".into());
        assert!(CalibrateCommand::parse(&resumed).is_err());
        let mut paused = base.to_vec();
        paused.extend(["--pause-after-layers", "1"].map(str::to_string));
        assert!(CalibrateCommand::parse(&paused).is_err());
    }

    #[test]
    fn kldref_row_cap_strides_evenly_and_keeps_every_row_by_default() {
        // Unset, zero, or a cap at least as large as the corpus keeps every row.
        assert_eq!(kldref_row_stride(1000, None), 1);
        assert_eq!(kldref_row_stride(1000, Some(0)), 1);
        assert_eq!(kldref_row_stride(1000, Some(1000)), 1);
        assert_eq!(kldref_row_stride(1000, Some(4000)), 1);
        // A cap below the corpus size rounds the stride up, so the retained count
        // never exceeds the request.
        assert_eq!(kldref_row_stride(1000, Some(100)), 10);
        assert_eq!(kldref_row_stride(1000, Some(300)), 4);
        for target in [1usize, 7, 99, 512] {
            let stride = kldref_row_stride(226_506, Some(target));
            let kept = (0..226_506).filter(|row| row % stride == 0).count();
            assert!(
                kept <= target.max(1) * 2 && kept >= target / 2,
                "target {target} kept {kept} with stride {stride}"
            );
        }
    }

    #[test]
    fn kldref_subset_spreads_across_samples_and_skips_empty_batches() {
        use crate::calibration::contracts::SampleRow;
        use crate::calibration::schedule::LayerMicrobatch;

        let batch = LayerMicrobatch {
            sequence_start: 0,
            sequence_end: 4,
            time_start: 0,
            time_end: 1,
            rows: (0..4)
                .map(|sample_index| SampleRow {
                    sample_index,
                    position: 0,
                    token: 1,
                    reset_state: true,
                })
                .collect(),
            // Sample-major boundary indices, as the planner emits.
            boundary_rows: vec![0, 10, 20, 30],
        };
        // Stride 20 keeps boundary rows 0 and 20 — one row from two different
        // samples, not a contiguous head.
        let reduced = subset_batch_by_stride(&batch, 20).expect("some rows selected");
        assert_eq!(reduced.boundary_rows, vec![0, 20]);
        assert_eq!(
            reduced
                .rows
                .iter()
                .map(|r| r.sample_index)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(reduced.rows.len(), reduced.boundary_rows.len());

        // A batch the stride misses entirely is skipped, so no lm-head
        // projection runs for it at all.
        let missed = LayerMicrobatch {
            boundary_rows: vec![3, 7],
            rows: batch.rows[..2].to_vec(),
            ..batch.clone()
        };
        assert!(subset_batch_by_stride(&missed, 20).is_none());
    }

    #[test]
    fn cli_resumes_by_default_and_yields_to_unresumable_modes() {
        let base = [
            "--model",
            "model",
            "--corpus",
            "corpus.txt",
            "--output",
            "out.hfq",
        ]
        .map(str::to_string);
        assert!(CalibrateCommand::parse(&base).unwrap().resume);

        let mut opted_out = base.to_vec();
        opted_out.push("--no-resume".into());
        assert!(!CalibrateCommand::parse(&opted_out).unwrap().resume);

        // RAM boundaries and residual probes cannot checkpoint; the default
        // steps aside rather than turning every such run into an error.
        let mut ram = base.to_vec();
        ram.push("--boundary-ram".into());
        assert!(!CalibrateCommand::parse(&ram).unwrap().resume);

        let mut probe = base.to_vec();
        probe.extend(["--residual-probe-output", "residuals.hfq"].map(str::to_string));
        assert!(!CalibrateCommand::parse(&probe).unwrap().resume);
    }

    #[test]
    fn cli_rejects_conflicting_boundary_backends_and_unknown_flags() {
        let base = [
            "--model",
            "model",
            "--corpus",
            "corpus.txt",
            "--output",
            "out.hfq",
        ]
        .map(str::to_string);
        let mut conflicting = base.to_vec();
        conflicting.extend(["--boundary-ram", "--boundary-dir", "state"].map(str::to_string));
        assert!(CalibrateCommand::parse(&conflicting).is_err());

        let mut unknown = base.to_vec();
        unknown.push("--qwen-only".into());
        assert!(CalibrateCommand::parse(&unknown).is_err());

        let mut ram_resume = base.to_vec();
        ram_resume.extend(["--boundary-ram", "--resume"].map(str::to_string));
        assert!(CalibrateCommand::parse(&ram_resume).is_err());

        let mut zero_pause = base.to_vec();
        zero_pause.extend(["--pause-after-layers", "0"].map(str::to_string));
        assert!(CalibrateCommand::parse(&zero_pause).is_err());
    }

    #[test]
    fn cli_limits_finalize_completed_to_resume_without_side_outputs() {
        let base = [
            "--model",
            "model",
            "--corpus",
            "corpus.txt",
            "--output",
            "out.hfq",
        ]
        .map(str::to_string);
        // Resume is the default, so `--finalize-completed` alone is admitted and
        // carries resume implicitly.
        let mut valid = base.to_vec();
        valid.push("--finalize-completed".into());
        let command = CalibrateCommand::parse(&valid).unwrap();
        assert!(command.resume);
        assert!(command.finalize_completed);

        // An explicit opt out of resume makes it a user error.
        let mut without_resume = base.to_vec();
        without_resume.extend(["--no-resume", "--finalize-completed"].map(str::to_string));
        assert!(CalibrateCommand::parse(&without_resume).is_err());

        // Finalization only publishes an existing spool — no side outputs.
        let mut with_cask = base.to_vec();
        with_cask.extend(
            ["--finalize-completed", "--cask-output", "sidecar.hfq"].map(str::to_string),
        );
        assert!(CalibrateCommand::parse(&with_cask).is_err());
    }

    #[test]
    fn cli_requires_cask_only_to_be_fresh_ram_backed_and_without_kld() {
        let valid = [
            "--model",
            "model",
            "--corpus",
            "corpus.txt",
            "--output",
            "scratch.hfq",
            "--cask-output",
            "sidecar.hfq",
            "--cask-only",
            "--boundary-ram",
            "--no-kldref",
        ]
        .map(str::to_string);
        let command = CalibrateCommand::parse(&valid).unwrap();
        assert!(command.cask_only);
        assert!(command.boundary_ram);
        assert!(!command.kldref);
        // `--boundary-ram` steps the default resume aside, so the run is fresh.
        assert!(!command.resume);

        // Dropping any of the three required companions is rejected.
        for omitted in ["--boundary-ram", "--no-kldref"] {
            let mut invalid = valid.to_vec();
            let index = invalid.iter().position(|arg| arg == omitted).unwrap();
            invalid.remove(index);
            assert!(
                CalibrateCommand::parse(&invalid).is_err(),
                "cask-only without {omitted} must be rejected"
            );
        }

        // `--cask-only` without `--cask-output` is rejected (drop the pair).
        let mut no_cask_output = valid.to_vec();
        let idx = no_cask_output
            .iter()
            .position(|arg| arg == "--cask-output")
            .unwrap();
        no_cask_output.drain(idx..idx + 2);
        assert!(CalibrateCommand::parse(&no_cask_output).is_err());
    }


    fn temp_output(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir()
            .join(format!(
                "hipfire-calibrate-{label}-{}-{nonce}",
                std::process::id()
            ))
            .join("fixture.calib.hfq")
    }

    #[test]
    fn calibration_engine_build_identity_is_deterministic_and_checkpoint_bound() {
        let identity = calibration_engine_build_identity().unwrap();
        assert_eq!(identity, calibration_engine_build_identity().unwrap());
        assert!(identity.starts_with("executable:"));
        assert_ne!(identity, "executable:unavailable");
        let checkpoint = calibration_checkpoint_execution_fingerprint(&identity, "run-a");
        assert_eq!(
            checkpoint,
            calibration_checkpoint_execution_fingerprint(&identity, "run-a")
        );
        assert_ne!(
            checkpoint,
            calibration_checkpoint_execution_fingerprint(&identity, "run-b")
        );
        assert_ne!(
            checkpoint,
            calibration_checkpoint_execution_fingerprint("engine-b", "run-a")
        );
    }

    #[test]
    fn layer_progress_binds_run_part_and_read_ledger() {
        let output = temp_output("progress");
        fs::create_dir_all(output.parent().unwrap()).unwrap();
        let part = calibration_part_path(&output, 0);
        fs::write(&part, b"stable part bytes").unwrap();
        let progress = CalibrationLayerProgress {
            schema_version: CALIBRATION_PROGRESS_SCHEMA_VERSION,
            engine_build: "engine-a".into(),
            run_fingerprint: "run-a".into(),
            layer: 0,
            part_file: part.file_name().unwrap().to_string_lossy().into_owned(),
            part_bytes: fs::metadata(&part).unwrap().len(),
            part_hash: file_hash(&part).unwrap(),
            descriptors: vec![CalibTensorDesc {
                name: "model.layers.0.q_proj".into(),
                has_hessian: true,
                k: 4,
                n_tokens: 8,
            }],
            expert_telemetry: None,
            preserve_high_precision: vec![],
            max_consistency: 0.0,
            read_ledger: ReadLedgerSnapshot {
                consumed_logical: std::collections::BTreeSet::from(["embedding".into()]),
                read_canonical: std::collections::BTreeSet::from(["embedding".into()]),
                logical_bytes_read: 16,
                ..ReadLedgerSnapshot::default()
            },
            timing: CalibrationLayerTiming::default(),
        };
        write_layer_progress(&output, &progress).unwrap();
        let restored = read_layer_progress(&output, 0, "engine-a", "run-a").unwrap();
        assert_eq!(restored.layer, 0);
        assert_eq!(restored.descriptors, progress.descriptors);
        assert_eq!(restored.read_ledger, progress.read_ledger);
        assert!(read_layer_progress(&output, 0, "engine-b", "run-a").is_err());
        assert!(read_layer_progress(&output, 0, "engine-a", "run-b").is_err());

        let mut legacy = progress.clone();
        legacy.schema_version = 1;
        write_layer_progress(&output, &legacy).unwrap();
        assert!(read_layer_progress(&output, 0, "engine-a", "run-a").is_err());
        write_layer_progress(&output, &progress).unwrap();

        fs::write(&part, b"corrupt").unwrap();
        assert!(read_layer_progress(&output, 0, "engine-a", "run-a").is_err());
        fs::remove_dir_all(output.parent().unwrap()).unwrap();
    }

    #[test]
    fn completed_artifact_recovers_rename_before_checkpoint_completion() {
        let output = temp_output("artifact-recovery");
        fs::create_dir_all(output.parent().unwrap()).unwrap();
        let boundary_dir = output.parent().unwrap().join("boundary");
        let mut options = CalibrationOptions::default();
        options.kldref = false;
        let job = job(options);
        let geometry = MicrobatchGeometry {
            sequence_batch: 1,
            time_tile: 1,
            row_budget: 1,
        };
        let tuning = GeometryTuningReport {
            automatic: false,
            selected: geometry,
            free_vram_bytes: None,
            reserved_headroom_bytes: 0,
            vram_ceiling_source: "unavailable".into(),
            probes: Vec::new(),
        };
        let source_manifest = SourceManifestIdentity {
            fingerprint: "source".into(),
            shards: Vec::new(),
        };
        let read_ledger = ReadLedgerSnapshot {
            consumed_logical: std::collections::BTreeSet::new(),
            read_canonical: std::collections::BTreeSet::new(),
            logical_bytes_read: 0,
            ..ReadLedgerSnapshot::default()
        };
        let metadata = serde_json::json!({
            "artifact_kind": "calibration",
            "engine_build": "engine-a",
            "family": "fixture",
            "adapter_version": "fixture.v1",
            "arch_id": 77,
            "source_manifest": source_manifest,
            "run_fingerprint": "run-fp",
            "job": job,
            "microbatch_geometry": geometry,
            "geometry_tuning": tuning,
            "read_ledger": read_ledger,
            "expert_telemetry": [],
            "n_hessian": 1,
            "n_imatrix": 1,
            "per_tensor_tokens": {"dense.0": 8},
            "max_consistency": 0.0,
        })
        .to_string();
        let tensors = vec![
            HfqMemTensor {
                name: "dense.0.hessian".into(),
                quant_type: 2,
                shape: vec![1, 1],
                group_size: 0,
                data: 1.0f32.to_le_bytes().to_vec(),
            },
            HfqMemTensor {
                name: "dense.0.imatrix".into(),
                quant_type: 2,
                shape: vec![1],
                group_size: 0,
                data: 1.0f32.to_le_bytes().to_vec(),
            },
        ];
        write_hfqm_package_mem(&output, 77, &metadata, &tensors).unwrap();

        let mut boundary = BoundaryStore::create(
            BoundaryBackend::Mmap {
                directory: boundary_dir,
            },
            1,
            1,
            1,
            "sample-fp",
            "checkpoint-a",
        )
        .unwrap();
        boundary.write_next_rows(0, &[2.0]).unwrap();
        boundary.commit_layer(0).unwrap();
        boundary.finalize_kld().unwrap();
        assert!(!boundary.checkpoint().artifact_complete);

        let recovered = validate_completed_artifact(
            &output,
            77,
            "fixture",
            "fixture.v1",
            "engine-a",
            &source_manifest,
            "run-fp",
            &job,
            geometry,
        )
        .unwrap();
        assert_eq!(recovered.artifact.n_hessian, 1);
        assert_eq!(recovered.artifact.n_imatrix, 1);
        assert_eq!(recovered.kldref_positions, None);
        assert!(validate_completed_artifact(
            &output,
            77,
            "fixture",
            "fixture.v1",
            "engine-a",
            &source_manifest,
            "wrong-run",
            &job,
            geometry,
        )
        .is_err());
        assert!(validate_completed_artifact(
            &output,
            77,
            "fixture",
            "fixture.v1",
            "engine-b",
            &source_manifest,
            "run-fp",
            &job,
            geometry,
        )
        .is_err());
        boundary.finalize_artifact().unwrap();
        assert!(boundary.checkpoint().artifact_complete);

        drop(boundary);
        fs::remove_dir_all(output.parent().unwrap()).unwrap();
    }

    #[test]
    fn kld_rows_require_scheduler_identity_and_exclude_last_sample_position() {
        let job = job(CalibrationOptions::default());
        let batch = &MicrobatchPlanner::new(MicrobatchGeometry {
            sequence_batch: 1,
            time_tile: 3,
            row_budget: 3,
        })
        .unwrap()
        .plan(&job.samples)[0];
        let rows = batch
            .rows
            .iter()
            .map(|row| KldRefRow {
                sample_index: row.sample_index,
                position: row.position,
                indices: vec![0],
                logits: vec![0.0],
                log_z: 0.0,
            })
            .collect::<Vec<_>>();
        validate_kld_batch_rows(batch, &rows).unwrap();
        assert!(kld_row_has_next_token(&job, &rows[0]).unwrap());
        assert!(kld_row_has_next_token(&job, &rows[1]).unwrap());
        assert!(!kld_row_has_next_token(&job, &rows[2]).unwrap());

        let mut reordered = rows;
        reordered.swap(0, 1);
        assert!(validate_kld_batch_rows(batch, &reordered).is_err());
    }
}

