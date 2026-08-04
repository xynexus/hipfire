// SPDX-License-Identifier: Apache-2.0
//! The top-level induction stage driver, ported from `induct_model.main`.
//!
//! Composes the DFlash converter, the two-pass target workflow ([`super::two_pass`]),
//! and the TriAttention validator into a resumable, fingerprint-gated run. The
//! load-bearing part here is the STAGE GATING: [`target_stage_complete`] skips
//! regeneration only when the recipe fingerprint and the two-pass manifest's
//! fingerprint set both match, so a stale run is never mistaken for a fresh one.
//!
//! Repo-tool auto-build (the `cargo build --release ...` glue) is deliberately
//! left to the caller / the Python wrapper, per the plan (build glue stays out).

use super::recipe::{Recipe, RecipeInputs};
use super::two_pass::{self, TwoPassConfig};
use super::{artifact_is_valid, atomic_json, python_resolve, utc_now};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

pub const STAGES: [&str; 3] = ["dflash", "target", "triattn"];

/// The canonical output paths for one induction, matching
/// `induct_model.artifact_layout`.
pub struct ArtifactLayout {
    pub model: PathBuf,
    pub triattn: PathBuf,
    pub calib: PathBuf,
    pub manifest: PathBuf,
    pub two_pass_manifest: PathBuf,
    /// dflash_format → output path, insertion-ordered (dedup preserved).
    pub dflash: Vec<(String, PathBuf)>,
}

impl ArtifactLayout {
    pub fn build(
        root: &Path,
        model_name: &str,
        quant_format: &str,
        dflash_formats: &[String],
    ) -> Result<Self, String> {
        let primary_stem = format!("{model_name}.{quant_format}");
        let mut dflash = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for fmt in dflash_formats {
            if !seen.insert(fmt.clone()) {
                continue;
            }
            dflash_format_args(fmt)?; // validate the format is known
            dflash.push((
                fmt.clone(),
                root.join("drafts")
                    .join(format!("{model_name}-{}.dflash.hfq", fmt.to_uppercase())),
            ));
        }
        if dflash.is_empty() {
            return Err("at least one DFlash format is required".into());
        }
        Ok(Self {
            model: root.join("models").join(format!("{primary_stem}.hfq")),
            triattn: root.join("triattn").join(format!("{model_name}.triattn.hfq")),
            calib: root.join("calib").join(format!("{model_name}.calib.hfq")),
            manifest: root.join("induction").join(&primary_stem).join("manifest.json"),
            two_pass_manifest: root.join("induction").join(&primary_stem).join("two-pass.json"),
            dflash,
        })
    }

    fn as_json(&self) -> Value {
        let mut map = BTreeMap::new();
        map.insert("model".to_string(), json!(self.model.to_string_lossy()));
        map.insert("triattn".to_string(), json!(self.triattn.to_string_lossy()));
        map.insert("calib".to_string(), json!(self.calib.to_string_lossy()));
        map.insert("manifest".to_string(), json!(self.manifest.to_string_lossy()));
        map.insert(
            "two_pass_manifest".to_string(),
            json!(self.two_pass_manifest.to_string_lossy()),
        );
        for (fmt, path) in &self.dflash {
            map.insert(format!("dflash_{fmt}"), json!(path.to_string_lossy()));
        }
        json!(map)
    }
}

/// DFlash converter dtype flags — the twin of `induct_model._dflash_format_args`.
pub fn dflash_format_args(dflash_format: &str) -> Result<Vec<String>, String> {
    Ok(match dflash_format {
        "bf16" => vec![],
        "f16" => vec!["--f16".into()],
        "f32" => vec!["--keep-f32".into()],
        "mq3" => vec!["--mq3".into()],
        "mq4" => vec!["--mq4".into()],
        "mq6" => vec!["--mq6".into()],
        other => return Err(format!("unknown dflash format {other:?}")),
    })
}

/// Default quantizer flags from the format suffix — twin of
/// `induct_model.default_quant_args`.
pub fn default_quant_args(quant_format: &str) -> Vec<String> {
    if quant_format.ends_with("++") {
        vec!["--awq".into(), "--ldlq".into()]
    } else if quant_format.ends_with('+') {
        vec!["--awq".into()]
    } else {
        vec![]
    }
}

/// The target recipe fingerprint used for stage gating — identical to the
/// two-pass recipe fingerprint (twin of `induct_model._target_recipe_fingerprint`).
pub fn target_recipe_fingerprint(inputs: &RecipeInputs) -> Result<String, Box<dyn Error>> {
    Ok(Recipe::build(inputs)?.recipe_fingerprint)
}

/// Whether the target stage is complete AND matches `recipe_fingerprint` and the
/// two-pass manifest's fingerprint set. Twin of `induct_model.target_stage_complete`.
pub fn target_stage_complete(layout: &ArtifactLayout, recipe_fingerprint: &str) -> bool {
    if !artifact_is_valid(&layout.calib, b"HFQM") || !artifact_is_valid(&layout.model, b"HFQM") {
        return false;
    }
    let Ok(text) = std::fs::read_to_string(&layout.two_pass_manifest) else {
        return false;
    };
    let Ok(manifest) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    let ledger = manifest.get("source_reads");
    let fingerprints = manifest.get("fingerprints");
    let audit = manifest.get("calibration_audit");
    let ledger_ok = ledger.and_then(|v| v.as_object()).is_some_and(|l| {
        !l.get("missing_logical").and_then(|v| v.as_array()).is_some_and(|a| !a.is_empty())
            && !l.get("duplicate_logical").and_then(|v| v.as_array()).is_some_and(|a| !a.is_empty())
    });
    let fingerprints_ok = fingerprints.and_then(|v| v.as_object()).is_some_and(|f| {
        f.get("calibration_artifact").and_then(|v| v.as_str()).is_some_and(|s| !s.is_empty())
            && f.get("quantized_artifact").and_then(|v| v.as_str()).is_some_and(|s| !s.is_empty())
    });
    let audit_ok = audit.and_then(|v| v.as_object()).is_some_and(|a| {
        a.get("schema").and_then(|v| v.as_str()) == Some("hipfire.calibration_audit.v1")
            && a.get("valid").and_then(|v| v.as_bool()) == Some(true)
            && !a.get("errors").and_then(|v| v.as_array()).is_some_and(|e| !e.is_empty())
            && a.get("artifact_fingerprint")
                == fingerprints.and_then(|f| f.get("calibration_artifact"))
    });
    manifest.get("status").and_then(|v| v.as_str()) == Some("complete")
        && manifest.get("recipe_fingerprint").and_then(|v| v.as_str()) == Some(recipe_fingerprint)
        && ledger_ok
        && fingerprints_ok
        && audit_ok
}

fn required_outputs(stage: &str, layout: &ArtifactLayout) -> Vec<(PathBuf, &'static [u8])> {
    match stage {
        "dflash" => layout
            .dflash
            .iter()
            .map(|(_, path)| (path.clone(), b"HFQM" as &[u8]))
            .collect(),
        "target" => vec![(layout.calib.clone(), b"HFQM"), (layout.model.clone(), b"HFQM")],
        "triattn" => vec![(layout.triattn.clone(), b"TRIA")],
        _ => vec![],
    }
}

fn stage_complete(stage: &str, layout: &ArtifactLayout, target_fingerprint: &str) -> bool {
    if stage == "target" {
        return target_stage_complete(layout, target_fingerprint);
    }
    required_outputs(stage, layout)
        .iter()
        .all(|(path, magic)| artifact_is_valid(path, magic))
}

/// Everything the induction driver needs.
pub struct InductConfig {
    pub target: PathBuf,
    pub dflash_source: PathBuf,
    pub model_name: String,
    pub artifact_root: PathBuf,
    pub quant_format: String,
    pub dflash_formats: Vec<String>,
    pub corpus: PathBuf,
    pub n_sequences: u64,
    pub ctx_len: u64,
    pub batch_size: u64,
    pub time_tile: u64,
    pub max_rows: u64,
    pub layer_prefetch_bytes: u64,
    pub kldref_topk: u64,
    pub min_expert_activations: u64,
    pub expert_capture_target: u64,
    pub expert_capture_tile_rows: u64,
    pub required_expert_fraction: f64,
    pub sampling_seed: u64,
    pub expert_coverage_policy: String,
    pub triattn_max_tokens: u64,
    pub triattn_chunk_len: u64,
    pub quant_args: Vec<String>,
    pub stages: Vec<String>,
    pub force: bool,
    pub dry_run: bool,
    // Tool binaries.
    pub hipfire: String,
    pub quantizer: String,
    pub dflash_converter: String,
    pub triattn_bin: String,
}

impl InductConfig {
    fn recipe_inputs(&self, layout: &ArtifactLayout) -> RecipeInputs {
        RecipeInputs {
            model: self.target.clone(),
            calib: layout.calib.clone(),
            output: layout.model.clone(),
            quant_format: self.quant_format.clone(),
            corpus: self.corpus.clone(),
            n_sequences: self.n_sequences,
            ctx_len: self.ctx_len,
            batch_size: self.batch_size,
            time_tile: self.time_tile,
            max_rows: self.max_rows,
            layer_prefetch_bytes: self.layer_prefetch_bytes,
            kldref_topk: self.kldref_topk,
            min_expert_activations: self.min_expert_activations,
            expert_capture_target: self.expert_capture_target,
            expert_capture_tile_rows: self.expert_capture_tile_rows,
            required_expert_fraction: self.required_expert_fraction,
            sampling_seed: self.sampling_seed,
            expert_coverage_policy: self.expert_coverage_policy.clone(),
            quant_args: self.quant_args.clone(),
        }
    }
}

fn run_subprocess(command: &[String]) -> Result<(), Box<dyn Error>> {
    let status = std::process::Command::new(&command[0])
        .args(&command[1..])
        .status()
        .map_err(|e| format!("spawn {:?}: {e}", command[0]))?;
    if !status.success() {
        return Err(format!("command failed ({status}): {}", command.join(" ")).into());
    }
    Ok(())
}

/// Run the induction. Resolves sources, lays out artifacts, gates each stage on
/// the recipe fingerprint, and drives DFlash / target / TriAttention.
pub fn run(cfg: &InductConfig) -> Result<Value, Box<dyn Error>> {
    let target = super::preflight::resolve_snapshot(&cfg.target)?;
    let selected: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        cfg.stages.iter().filter(|s| seen.insert((*s).clone())).cloned().collect()
    };

    let root = python_resolve(&cfg.artifact_root);
    let layout = ArtifactLayout::build(&root, &cfg.model_name, &cfg.quant_format, &cfg.dflash_formats)?;
    let recipe_fingerprint = target_recipe_fingerprint(&cfg.recipe_inputs(&layout))?;

    // Source preflight (compatibility contract when dflash is selected).
    let preflight = if selected.iter().any(|s| s == "dflash") {
        let draft = super::preflight::resolve_snapshot(&cfg.dflash_source)?;
        preflight_sources(&target, &draft)?
    } else {
        preflight_target_only(&target)?
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "preflight": preflight,
            "artifacts": layout.as_json(),
        }))?
    );

    if cfg.dry_run {
        for stage in &selected {
            if stage_complete(stage, &layout, &recipe_fingerprint) && !cfg.force {
                println!("{stage}: reuse valid artifact(s)");
            } else {
                println!("{stage}: would run");
            }
        }
        return Ok(json!({"dry_run": true, "recipe_fingerprint": recipe_fingerprint}));
    }

    // Top-level manifest.
    let previous = std::fs::read_to_string(&layout.manifest)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok());
    let created_at = previous
        .as_ref()
        .and_then(|p| p.get("created_at"))
        .cloned()
        .unwrap_or_else(|| json!(utc_now()));
    let mut stages_state = previous
        .as_ref()
        .and_then(|p| p.get("stages"))
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    let mut manifest = json!({
        "schema": 1,
        "created_at": created_at,
        "updated_at": utc_now(),
        "model_name": cfg.model_name,
        "quant_format": cfg.quant_format,
        "dflash_formats": cfg.dflash_formats,
        "corpus": cfg.corpus.to_string_lossy(),
        "sources": preflight,
        "artifacts": layout.as_json(),
        "admission": {
            "status": "pending",
            "required_evidence": [
                "finite-logit and coherence smoke",
                "KLD/PPL against BF16 or an accepted higher-precision reference",
                "DFlash acceptance/tau and decoded-output checks",
                "TriAttention/CASK long-context recall",
                "combined DFlash plus CASK coherence and recall",
                "Kernel Atlas AR and DFlash performance rows",
            ],
        },
    });
    manifest.as_object_mut().unwrap().insert("stages".into(), Value::Object(stages_state.clone()));
    atomic_json(&layout.manifest, &manifest)?;

    for stage in &selected {
        if stage_complete(stage, &layout, &recipe_fingerprint) && !cfg.force {
            println!("{stage}: reuse valid artifact(s)");
            stages_state.insert(
                stage.clone(),
                json!({"status": "reused", "completed_at": utc_now()}),
            );
            if stage == "target" {
                fold_two_pass(&mut manifest, &layout)?;
            }
            manifest.as_object_mut().unwrap().insert("stages".into(), Value::Object(stages_state.clone()));
            manifest.as_object_mut().unwrap().insert("updated_at".into(), json!(utc_now()));
            atomic_json(&layout.manifest, &manifest)?;
            continue;
        }
        if stage == "triattn" && !artifact_is_valid(&layout.model, b"HFQM") {
            return Err("TriAttention requires the completed target HFQ; run the target stage first".into());
        }
        for (output, _) in required_outputs(stage, &layout) {
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
        }
        stages_state.insert(stage.clone(), json!({"status": "running", "started_at": utc_now()}));
        manifest.as_object_mut().unwrap().insert("stages".into(), Value::Object(stages_state.clone()));
        manifest.as_object_mut().unwrap().insert("updated_at".into(), json!(utc_now()));
        atomic_json(&layout.manifest, &manifest)?;

        let result = run_stage(stage, cfg, &target, &layout, &recipe_fingerprint);
        if let Err(error) = result {
            stages_state.insert(
                stage.clone(),
                json!({"status": "failed", "failed_at": utc_now(), "error": error.to_string()}),
            );
            manifest.as_object_mut().unwrap().insert("stages".into(), Value::Object(stages_state.clone()));
            manifest.as_object_mut().unwrap().insert("updated_at".into(), json!(utc_now()));
            atomic_json(&layout.manifest, &manifest)?;
            return Err(error);
        }
        if !stage_complete(stage, &layout, &recipe_fingerprint) {
            return Err(format!("{stage} command returned success but its output artifact is invalid").into());
        }
        let outputs: Vec<Value> = required_outputs(stage, &layout)
            .iter()
            .map(|(path, _)| {
                json!({"path": path.to_string_lossy(), "bytes": std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)})
            })
            .collect();
        stages_state.insert(
            stage.clone(),
            json!({"status": "complete", "completed_at": utc_now(), "outputs": outputs}),
        );
        if stage == "target" {
            fold_two_pass(&mut manifest, &layout)?;
        }
        manifest.as_object_mut().unwrap().insert("stages".into(), Value::Object(stages_state.clone()));
        manifest.as_object_mut().unwrap().insert("updated_at".into(), json!(utc_now()));
        atomic_json(&layout.manifest, &manifest)?;
    }

    println!(
        "induction artifacts complete; admission remains pending: {}",
        layout.manifest.display()
    );
    Ok(manifest)
}

fn fold_two_pass(manifest: &mut Value, layout: &ArtifactLayout) -> Result<(), Box<dyn Error>> {
    let two_pass: Value = serde_json::from_str(&std::fs::read_to_string(&layout.two_pass_manifest)?)?;
    let obj = manifest.as_object_mut().unwrap();
    if let Some(reads) = two_pass.get("source_reads") {
        obj.insert("source_reads".into(), reads.clone());
    }
    if let Some(fp) = two_pass.get("fingerprints") {
        obj.insert("fingerprints".into(), fp.clone());
    }
    obj.insert("two_pass".into(), two_pass);
    Ok(())
}

fn run_stage(
    stage: &str,
    cfg: &InductConfig,
    target: &Path,
    layout: &ArtifactLayout,
    _recipe_fingerprint: &str,
) -> Result<(), Box<dyn Error>> {
    match stage {
        "dflash" => {
            let draft = super::preflight::resolve_snapshot(&cfg.dflash_source)?;
            for (fmt, output) in &layout.dflash {
                let mut command = vec![cfg.dflash_converter.clone()];
                command.extend(dflash_format_args(fmt)?);
                command.extend([
                    "--input".into(),
                    draft.to_string_lossy().into_owned(),
                    "--output".into(),
                    output.to_string_lossy().into_owned(),
                ]);
                run_subprocess(&command)?;
            }
            Ok(())
        }
        "target" => {
            let reuse = artifact_is_valid(&layout.calib, b"HFQM") && !cfg.force;
            let two_pass_cfg = TwoPassConfig {
                model: target.to_path_buf(),
                calib: layout.calib.clone(),
                output: layout.model.clone(),
                manifest: layout.two_pass_manifest.clone(),
                quant_format: cfg.quant_format.clone(),
                corpus: cfg.corpus.clone(),
                n_sequences: cfg.n_sequences,
                ctx_len: cfg.ctx_len,
                batch_size: cfg.batch_size,
                time_tile: cfg.time_tile,
                max_rows: cfg.max_rows,
                layer_prefetch_bytes: cfg.layer_prefetch_bytes,
                kldref_topk: cfg.kldref_topk,
                min_expert_activations: cfg.min_expert_activations,
                expert_capture_target: cfg.expert_capture_target,
                expert_capture_tile_rows: cfg.expert_capture_tile_rows,
                required_expert_fraction: cfg.required_expert_fraction,
                sampling_seed: cfg.sampling_seed,
                expert_coverage_policy: cfg.expert_coverage_policy.clone(),
                quant_args: cfg.quant_args.clone(),
                quantizer: cfg.quantizer.clone(),
                hipfire: cfg.hipfire.clone(),
                skip_calib: reuse,
                dry_run: false,
            };
            two_pass::run(&two_pass_cfg)?;
            Ok(())
        }
        "triattn" => {
            let command = vec![
                cfg.hipfire.clone(),
                "lock".into(),
                "run".into(),
                "induct-triattn".into(),
                "--".into(),
                cfg.triattn_bin.clone(),
                layout.model.to_string_lossy().into_owned(),
                "--sidecar".into(),
                layout.triattn.to_string_lossy().into_owned(),
                "--corpus".into(),
                cfg.corpus.to_string_lossy().into_owned(),
                "--max-tokens".into(),
                cfg.triattn_max_tokens.to_string(),
                "--chunk-len".into(),
                cfg.triattn_chunk_len.to_string(),
                "--gpu-calib".into(),
            ];
            run_subprocess(&command)
        }
        other => Err(format!("unknown induction stage {other}").into()),
    }
}

fn read_config(snapshot: &Path) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json"))?)?)
}

fn text_config(root: &Value) -> Value {
    root.get("text_config").cloned().unwrap_or_else(|| root.clone())
}

fn integer(config: &Value, key: &str) -> Result<i64, String> {
    config
        .get(key)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| format!("config field {key:?} must be an integer"))
}

fn opt_integer(config: &Value, key: &str) -> Value {
    config.get(key).and_then(|v| v.as_i64()).map(|v| json!(v)).unwrap_or(Value::Null)
}

fn source_summary(snapshot: &Path) -> Result<Value, Box<dyn Error>> {
    let config_bytes = std::fs::read(snapshot.join("config.json"))?;
    let mut safetensors: Vec<PathBuf> = std::fs::read_dir(snapshot)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    safetensors.sort();
    if safetensors.is_empty() {
        return Err(format!("no safetensors files under {}", snapshot.display()).into());
    }
    let bytes: u64 = safetensors.iter().filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len())).sum();
    Ok(json!({
        "snapshot": snapshot.to_string_lossy(),
        "config_sha256": super::recipe::sha256_hex(&config_bytes),
        "safetensors_files": safetensors.len(),
        "safetensors_bytes": bytes,
    }))
}

const PREFLIGHT_FIELDS: [&str; 5] = [
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
];

/// Twin of `induct_model.preflight_target_only`.
pub fn preflight_target_only(target: &Path) -> Result<Value, Box<dyn Error>> {
    let root = read_config(target)?;
    let text = text_config(&root);
    let mut summary = source_summary(target)?;
    let obj = summary.as_object_mut().unwrap();
    for field in PREFLIGHT_FIELDS {
        obj.insert(field.into(), opt_integer(&text, field));
    }
    obj.insert("num_hidden_layers".into(), opt_integer(&text, "num_hidden_layers"));
    Ok(json!({
        "target": summary,
        "compatibility": "not-applicable (no DFlash stage)",
    }))
}

/// Twin of `induct_model.preflight_sources` — the DFlash target/draft
/// compatibility contract (residual width, vocab, target-layer count).
pub fn preflight_sources(target: &Path, draft: &Path) -> Result<Value, Box<dyn Error>> {
    let target_root = read_config(target)?;
    let target_text = text_config(&target_root);
    let draft_config = read_config(draft)?;

    let architectures = draft_config.get("architectures").and_then(|v| v.as_array());
    if !architectures.is_some_and(|a| a.iter().any(|v| v.as_str() == Some("DFlashDraftModel"))) {
        return Err("DFlash source config does not declare DFlashDraftModel".into());
    }
    let dflash_config = draft_config
        .get("dflash_config")
        .and_then(|v| v.as_object())
        .ok_or("DFlash source config is missing dflash_config")?;
    let block_size = draft_config
        .get("block_size")
        .and_then(|v| v.as_i64())
        .or_else(|| dflash_config.get("block_size").and_then(|v| v.as_i64()))
        .filter(|v| *v >= 1)
        .ok_or("DFlash block_size must be a positive integer")?;

    let mut mismatches = Vec::new();
    for field in ["hidden_size", "vocab_size"] {
        let t = integer(&target_text, field)?;
        let d = integer(&draft_config, field)?;
        if t != d {
            mismatches.push(format!("{field}: target={t}, draft={d}"));
        }
    }
    let target_layers = integer(&target_text, "num_hidden_layers")?;
    let draft_target_layers = integer(&draft_config, "num_target_layers")?;
    if target_layers != draft_target_layers {
        mismatches.push(format!(
            "num_hidden_layers/num_target_layers: target={target_layers}, draft={draft_target_layers}"
        ));
    }
    let target_layer_ids = draft_config
        .get("dflash_config")
        .and_then(|v| v.get("target_layer_ids"))
        .and_then(|v| v.as_array())
        .ok_or("DFlash target_layer_ids must be an integer list")?;
    let ids: Vec<i64> = target_layer_ids
        .iter()
        .map(|v| v.as_i64().ok_or("DFlash target_layer_ids must be an integer list"))
        .collect::<Result<_, _>>()?;
    if ids.iter().any(|&l| l < 0 || l >= target_layers) {
        mismatches.push(format!("target_layer_ids outside target range 0..{}", target_layers - 1));
    }
    let mask_token_id = integer(&json!(dflash_config), "mask_token_id")?;
    let vocab = integer(&target_text, "vocab_size")?;
    if !(0..vocab).contains(&mask_token_id) {
        mismatches.push(format!("mask_token_id {mask_token_id} is outside the target vocabulary"));
    }
    if !mismatches.is_empty() {
        return Err(format!("target/DFlash incompatibility: {}", mismatches.join("; ")).into());
    }

    let mut target_summary = source_summary(target)?;
    {
        let obj = target_summary.as_object_mut().unwrap();
        for field in PREFLIGHT_FIELDS {
            obj.insert(field.into(), json!(integer(&target_text, field)?));
        }
        obj.insert("num_hidden_layers".into(), json!(target_layers));
    }
    let mut draft_summary = source_summary(draft)?;
    {
        let obj = draft_summary.as_object_mut().unwrap();
        for field in PREFLIGHT_FIELDS {
            obj.insert(field.into(), json!(integer(&draft_config, field)?));
        }
        obj.insert("num_hidden_layers".into(), json!(integer(&draft_config, "num_hidden_layers")?));
        obj.insert("num_target_layers".into(), json!(draft_target_layers));
        obj.insert("block_size".into(), json!(block_size));
        obj.insert("mask_token_id".into(), json!(mask_token_id));
        obj.insert("target_layer_ids".into(), json!(ids));
    }
    Ok(json!({
        "target": target_summary,
        "draft": draft_summary,
        "compatibility": "compatible",
    }))
}
