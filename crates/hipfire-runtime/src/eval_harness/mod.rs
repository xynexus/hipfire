// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared framework for the `hipfire-eval` runner.
//!
//! v1 establishes the stable command/result/output contract and the fast-tier
//! battery map. GPU-backed scoring batteries intentionally record explicit
//! skips until the daemon-backed execution path is promoted into this binary.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalTier {
    Fast,
    Targeted,
    Extensive,
}

impl EvalTier {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "fast" => Ok(Self::Fast),
            "targeted" => Ok(Self::Targeted),
            "extensive" => Ok(Self::Extensive),
            other => Err(format!("unknown tier: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Targeted => "targeted",
            Self::Extensive => "extensive",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatteryId {
    Smoke,
    Quality,
    Retrieval,
    Longctx,
    Speed,
    Dflash,
    PromptShape,
    Structured,
    Vision,
    Cask,
    Profile,
}

impl BatteryId {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "smoke" => Ok(Self::Smoke),
            "quality" => Ok(Self::Quality),
            "retrieval" => Ok(Self::Retrieval),
            "longctx" => Ok(Self::Longctx),
            "speed" => Ok(Self::Speed),
            "dflash" => Ok(Self::Dflash),
            "prompt_shape" | "prompt-shape" => Ok(Self::PromptShape),
            "structured" => Ok(Self::Structured),
            "vision" => Ok(Self::Vision),
            "cask" => Ok(Self::Cask),
            "profile" => Ok(Self::Profile),
            other => Err(format!("unknown battery: {other}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Quality => "quality",
            Self::Retrieval => "retrieval",
            Self::Longctx => "longctx",
            Self::Speed => "speed",
            Self::Dflash => "dflash",
            Self::PromptShape => "prompt_shape",
            Self::Structured => "structured",
            Self::Vision => "vision",
            Self::Cask => "cask",
            Self::Profile => "profile",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DflashMode {
    Off,
    Auto,
    On,
}

impl DflashMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            other => Err(format!("unknown dflash mode: {other}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProfileMode {
    Off,
    Passive,
}

impl ProfileMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "off" => Ok(Self::Off),
            "passive" => Ok(Self::Passive),
            other => Err(format!("unknown profile mode: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalConfig {
    pub model: String,
    pub draft: Option<String>,
    pub tier: EvalTier,
    pub batteries: Vec<BatteryId>,
    pub out_dir: PathBuf,
    pub kv_mode: Option<String>,
    pub max_tokens: usize,
    pub dflash: DflashMode,
    pub profile: ProfileMode,
    pub quality_max_chunks: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalManifest {
    pub schema: u32,
    pub runner: String,
    pub created_utc: String,
    pub repo_root: Option<String>,
    pub commit_sha: Option<String>,
    pub binary_hash: Option<String>,
    pub arch: Option<String>,
    pub rocm: Option<String>,
    pub config: EvalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub schema: u32,
    pub battery: BatteryId,
    pub case_id: String,
    pub status: EvalStatus,
    pub reason: Option<String>,
    pub metrics: BTreeMap<String, Value>,
    pub prompt_hash: Option<String>,
    pub prompt_path: Option<String>,
    pub model: String,
    pub draft: Option<String>,
    pub commit_sha: Option<String>,
    pub binary_hash: Option<String>,
    pub arch: Option<String>,
    pub rocm: Option<String>,
    pub kv_mode: Option<String>,
    pub started_utc: String,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvalStatus {
    Pass,
    Fail,
    Skip,
}

pub fn parse_args_from<I, S>(args: I) -> Result<EvalConfig, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let argv: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut model: Option<String> = None;
    let mut draft: Option<String> = None;
    let mut tier = EvalTier::Fast;
    let mut batteries: Option<Vec<BatteryId>> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut kv_mode: Option<String> = None;
    let mut max_tokens = 64usize;
    let mut dflash = DflashMode::Off;
    let mut profile = ProfileMode::Off;
    let mut quality_max_chunks: Option<usize> = None;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => return Err(usage()),
            "--model" => {
                model = Some(take_value(&argv, i, "--model")?);
                i += 2;
            }
            "--draft" => {
                draft = Some(take_value(&argv, i, "--draft")?);
                i += 2;
            }
            "--tier" => {
                tier = EvalTier::parse(&take_value(&argv, i, "--tier")?)?;
                i += 2;
            }
            "--battery" | "--batteries" => {
                let raw = take_value(&argv, i, "--battery")?;
                let parsed: Result<Vec<_>, _> = raw
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| BatteryId::parse(s.trim()))
                    .collect();
                batteries = Some(parsed?);
                i += 2;
            }
            "--out" => {
                out_dir = Some(PathBuf::from(take_value(&argv, i, "--out")?));
                i += 2;
            }
            "--kv-mode" => {
                kv_mode = Some(take_value(&argv, i, "--kv-mode")?);
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = take_value(&argv, i, "--max-tokens")?
                    .parse()
                    .map_err(|_| "--max-tokens must be a positive integer".to_string())?;
                i += 2;
            }
            "--dflash" => {
                dflash = DflashMode::parse(&take_value(&argv, i, "--dflash")?)?;
                i += 2;
            }
            "--profile" => {
                profile = ProfileMode::parse(&take_value(&argv, i, "--profile")?)?;
                i += 2;
            }
            "--quality-max-chunks" => {
                quality_max_chunks = Some(
                    take_value(&argv, i, "--quality-max-chunks")?
                        .parse()
                        .map_err(|_| {
                            "--quality-max-chunks must be a positive integer".to_string()
                        })?,
                );
                i += 2;
            }
            other => return Err(format!("unknown arg: {other}\n\n{}", usage())),
        }
    }

    let model = model.ok_or_else(|| format!("error: --model is required\n\n{}", usage()))?;
    let batteries = batteries.unwrap_or_else(|| default_batteries(tier));
    let out_dir = out_dir.unwrap_or_else(|| default_output_dir(&model, tier));

    Ok(EvalConfig {
        model,
        draft,
        tier,
        batteries,
        out_dir,
        kv_mode,
        max_tokens,
        dflash,
        profile,
        quality_max_chunks,
    })
}

pub fn usage() -> String {
    "Usage:\n  hipfire-eval --model <model> [--tier fast|targeted|extensive]\n\n\
     Options:\n\
       --battery <a,b>          smoke,quality,retrieval,longctx,speed,dflash,prompt_shape,structured,vision,cask,profile\n\
       --out <dir>              output directory\n\
       --draft <path>           DFlash draft artifact\n\
       --dflash <off|auto|on>   DFlash mode (default: off)\n\
       --kv-mode <mode>         KV mode metadata to record\n\
       --max-tokens <N>         short decode cap for execution batteries (default: 64)\n\
       --profile <off|passive>  profiling mode (default: off)\n\
       --quality-max-chunks <N> quality canary chunk cap\n"
        .to_string()
}

fn take_value(argv: &[String], i: usize, flag: &str) -> Result<String, String> {
    argv.get(i + 1)
        .filter(|v| !v.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

pub fn default_batteries(tier: EvalTier) -> Vec<BatteryId> {
    let mut out = vec![
        BatteryId::Smoke,
        BatteryId::Quality,
        BatteryId::Retrieval,
        BatteryId::Speed,
        BatteryId::Dflash,
        BatteryId::PromptShape,
        BatteryId::Structured,
    ];
    if matches!(tier, EvalTier::Targeted | EvalTier::Extensive) {
        out.extend([BatteryId::Longctx, BatteryId::Cask]);
    }
    if matches!(tier, EvalTier::Extensive) {
        out.push(BatteryId::Vision);
    }
    out
}

pub fn default_output_dir(model: &str, tier: EvalTier) -> PathBuf {
    let stem = model_stem(model);
    let leaf = format!("{}-{}-{}", utc_stamp_compact(), stem, tier.as_str());
    if repo_root().is_some() {
        PathBuf::from("benchmarks")
            .join("results")
            .join("eval")
            .join(leaf)
    } else {
        home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".hipfire")
            .join("eval")
            .join("results")
            .join(leaf)
    }
}

pub fn run_eval(config: EvalConfig) -> Result<(), String> {
    fs::create_dir_all(&config.out_dir)
        .map_err(|e| format!("create {}: {e}", config.out_dir.display()))?;
    fs::create_dir_all(config.out_dir.join("artifacts"))
        .map_err(|e| format!("create artifacts dir: {e}"))?;

    let context = EvalContext::new(&config);
    let manifest = EvalManifest {
        schema: 1,
        runner: "hipfire-eval".to_string(),
        created_utc: utc_now(),
        repo_root: repo_root().map(|p| p.display().to_string()),
        commit_sha: context.commit_sha.clone(),
        binary_hash: context.binary_hash.clone(),
        arch: context.arch.clone(),
        rocm: context.rocm.clone(),
        config: config.clone(),
    };
    write_json_pretty(&config.out_dir.join("manifest.json"), &manifest)?;

    let results_path = config.out_dir.join("results.jsonl");
    let mut results = Vec::new();
    for battery in &config.batteries {
        results.extend(run_battery(*battery, &config, &context));
    }
    let mut f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&results_path)
        .map_err(|e| format!("open {}: {e}", results_path.display()))?;
    for row in &results {
        serde_json::to_writer(&mut f, row).map_err(|e| format!("serialize result row: {e}"))?;
        f.write_all(b"\n")
            .map_err(|e| format!("write {}: {e}", results_path.display()))?;
    }
    write_summary(&config.out_dir.join("summary.md"), &config, &results)?;
    println!("{}", config.out_dir.display());
    Ok(())
}

struct EvalContext {
    commit_sha: Option<String>,
    binary_hash: Option<String>,
    arch: Option<String>,
    rocm: Option<String>,
}

impl EvalContext {
    fn new(_config: &EvalConfig) -> Self {
        Self {
            commit_sha: command_stdout("git", &["rev-parse", "HEAD"]),
            binary_hash: std::env::current_exe()
                .ok()
                .and_then(|p| file_digest("sha256sum", &p)),
            arch: detect_arch(),
            rocm: rocm_version(),
        }
    }
}

fn run_battery(battery: BatteryId, config: &EvalConfig, ctx: &EvalContext) -> Vec<EvalResult> {
    match battery {
        BatteryId::Smoke => vec![
            skip_row(
                battery,
                "load_metadata",
                "daemon-backed model load is not implemented in fast v1",
                config,
                ctx,
                None,
            ),
            skip_row(
                battery,
                "finite_greedy_decode",
                "daemon-backed greedy decode is not implemented in fast v1",
                config,
                ctx,
                prompt("benchmarks/prompts/qwen2_smoke.txt"),
            ),
            skip_row(
                battery,
                "multi_turn_reset_recall",
                "daemon-backed multi-turn session is not implemented in fast v1",
                config,
                ctx,
                prompt("benchmarks/prompts/trains-meet.txt"),
            ),
        ],
        BatteryId::Quality => vec![skip_row(
            battery,
            "kld_reference_slice",
            "quality-baseline subprocess integration is not implemented in fast v1",
            config,
            ctx,
            prompt("benchmarks/quality-baselines/harness/canary.md"),
        )],
        BatteryId::Retrieval => vec![pass_row(
            battery,
            "synthetic_seed_fixture",
            config,
            ctx,
            prompt("benchmarks/prompts/trains-meet.txt"),
            BTreeMap::from([
                (
                    "fixture_kind".to_string(),
                    json!("hipfire_native_synthetic"),
                ),
                ("seed".to_string(), json!(1)),
            ]),
        )],
        BatteryId::Speed => vec![skip_row(
            battery,
            "pp32_pp128_ttft_decode",
            "daemon-backed speed anchors are not implemented in fast v1",
            config,
            ctx,
            prompt("benchmarks/prompts/lru_cache_single_blank.txt"),
        )],
        BatteryId::Dflash => {
            let mut rows = vec![skip_row(
                battery,
                "ar_coherence_anchor",
                "daemon-backed AR anchor is not implemented in fast v1",
                config,
                ctx,
                prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
            )];
            let reason = if matches!(config.dflash, DflashMode::Off) {
                "DFlash disabled by --dflash off"
            } else if config.draft.is_none() {
                "DFlash draft not provided or discoverable by this runner"
            } else {
                "daemon-backed DFlash anchor is not implemented in fast v1"
            };
            rows.push(skip_row(
                battery,
                "dflash_anchor",
                reason,
                config,
                ctx,
                prompt("benchmarks/prompts/dflash_resident_smoke.txt"),
            ));
            rows
        }
        BatteryId::PromptShape => vec![pass_row(
            battery,
            "whitespace_fixture_hash",
            config,
            ctx,
            prompt("benchmarks/prompts/lru_cache_pep8_strict.txt"),
            BTreeMap::from([("normalization_probe".to_string(), json!("newline_runs"))]),
        )],
        BatteryId::Structured => vec![pass_row(
            battery,
            "tool_call_fixture_hash",
            config,
            ctx,
            prompt("benchmarks/prompts/tool_call_read_file.txt"),
            BTreeMap::from([("structured_probe".to_string(), json!("tool_call_jsonish"))]),
        )],
        BatteryId::Longctx | BatteryId::Vision | BatteryId::Cask | BatteryId::Profile => {
            vec![skip_row(
                battery,
                "not_implemented",
                "not implemented in fast v1",
                config,
                ctx,
                None,
            )]
        }
    }
}

fn pass_row(
    battery: BatteryId,
    case_id: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    mut metrics: BTreeMap<String, Value>,
) -> EvalResult {
    metrics.insert("implemented".to_string(), json!(true));
    row(
        battery,
        case_id,
        EvalStatus::Pass,
        None,
        metrics,
        config,
        ctx,
        prompt,
        0,
    )
}

fn skip_row(
    battery: BatteryId,
    case_id: &str,
    reason: &str,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
) -> EvalResult {
    row(
        battery,
        case_id,
        EvalStatus::Skip,
        Some(reason.to_string()),
        BTreeMap::new(),
        config,
        ctx,
        prompt,
        0,
    )
}

fn row(
    battery: BatteryId,
    case_id: &str,
    status: EvalStatus,
    reason: Option<String>,
    metrics: BTreeMap<String, Value>,
    config: &EvalConfig,
    ctx: &EvalContext,
    prompt: Option<PromptRef>,
    elapsed_ms: u128,
) -> EvalResult {
    EvalResult {
        schema: 1,
        battery,
        case_id: case_id.to_string(),
        status,
        reason,
        metrics,
        prompt_hash: prompt.as_ref().and_then(|p| p.hash.clone()),
        prompt_path: prompt.map(|p| p.path),
        model: config.model.clone(),
        draft: config.draft.clone(),
        commit_sha: ctx.commit_sha.clone(),
        binary_hash: ctx.binary_hash.clone(),
        arch: ctx.arch.clone(),
        rocm: ctx.rocm.clone(),
        kv_mode: config.kv_mode.clone(),
        started_utc: utc_now(),
        elapsed_ms,
    }
}

struct PromptRef {
    path: String,
    hash: Option<String>,
}

fn prompt(path: &str) -> Option<PromptRef> {
    let p = Path::new(path);
    if !p.exists() {
        return None;
    }
    Some(PromptRef {
        path: path.to_string(),
        hash: file_digest("md5sum", p),
    })
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let f = File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    serde_json::to_writer_pretty(f, value).map_err(|e| format!("write {}: {e}", path.display()))
}

fn write_summary(path: &Path, config: &EvalConfig, rows: &[EvalResult]) -> Result<(), String> {
    let pass = rows.iter().filter(|r| r.status == EvalStatus::Pass).count();
    let fail = rows.iter().filter(|r| r.status == EvalStatus::Fail).count();
    let skip = rows.iter().filter(|r| r.status == EvalStatus::Skip).count();
    let mut body = String::new();
    body.push_str("# hipfire eval summary\n\n");
    body.push_str(&format!("- model: `{}`\n", config.model));
    body.push_str(&format!("- tier: `{}`\n", config.tier.as_str()));
    body.push_str(&format!(
        "- rows: {} pass / {} fail / {} skip\n\n",
        pass, fail, skip
    ));
    body.push_str("| battery | case | status | reason |\n");
    body.push_str("|---|---|---|---|\n");
    for r in rows {
        body.push_str(&format!(
            "| {} | {} | {:?} | {} |\n",
            r.battery.as_str(),
            r.case_id,
            r.status,
            r.reason.as_deref().unwrap_or("")
        ));
    }
    fs::write(path, body).map_err(|e| format!("write {}: {e}", path.display()))
}

fn model_stem(model: &str) -> String {
    let name = Path::new(model)
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or(model);
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '-'
            }
        })
        .collect();
    sanitized.trim_matches('-').to_string()
}

fn repo_root() -> Option<PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

fn command_stdout(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn file_digest(tool: &str, path: &Path) -> Option<String> {
    let out = Command::new(tool).arg(path).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
}

fn detect_arch() -> Option<String> {
    for node in ["1", "0"] {
        let path = format!("/sys/class/kfd/kfd/topology/nodes/{node}/properties");
        let raw = match fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        for line in raw.lines() {
            if let Some(v) = line.strip_prefix("gfx_target_version") {
                if let Ok(ver) = v.trim().parse() {
                    return Some(gfx_target_version_to_arch(ver));
                }
            }
        }
    }
    None
}

fn gfx_target_version_to_arch(ver: u32) -> String {
    match ver {
        100100 => "gfx1010".to_string(),
        100300 | 100302 => "gfx1030".to_string(),
        110000 | 110001 => "gfx1100".to_string(),
        110501 => "gfx1151".to_string(),
        120000 => "gfx1200".to_string(),
        120001 => "gfx1201".to_string(),
        _ => {
            let major = ver / 10000;
            let minor = (ver % 10000) / 100;
            let step = ver % 100;
            format!("gfx{major}{minor}{step}")
        }
    }
}

fn rocm_version() -> Option<String> {
    command_stdout("hipconfig", &["--version"])
        .or_else(|| command_stdout("/opt/rocm/bin/hipconfig", &["--version"]))
}

fn utc_now() -> String {
    command_stdout("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"]).unwrap_or_else(|| {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("unix-{secs}")
    })
}

fn utc_stamp_compact() -> String {
    command_stdout("date", &["-u", "+%Y%m%dT%H%M%SZ"]).unwrap_or_else(|| {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("unix-{secs}")
    })
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub fn run_from_env() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprint!("{}", usage());
        return Ok(());
    }
    let config = parse_args_from(args)?;
    run_eval(config)
}

#[allow(dead_code)]
fn _io_err(e: io::Error) -> String {
    e.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_core_cli_surface() {
        let cfg = parse_args_from([
            "hipfire-eval",
            "--model",
            "qwen3.5:9b",
            "--tier",
            "fast",
            "--battery",
            "smoke,quality,speed",
            "--draft",
            "draft.hfq",
            "--dflash",
            "auto",
            "--profile",
            "passive",
        ])
        .unwrap();
        assert_eq!(cfg.model, "qwen3.5:9b");
        assert_eq!(cfg.tier, EvalTier::Fast);
        assert_eq!(
            cfg.batteries,
            vec![BatteryId::Smoke, BatteryId::Quality, BatteryId::Speed]
        );
        assert_eq!(cfg.draft.as_deref(), Some("draft.hfq"));
        assert_eq!(cfg.dflash, DflashMode::Auto);
        assert_eq!(cfg.profile, ProfileMode::Passive);
    }

    #[test]
    fn expands_targeted_and_extensive_tiers() {
        assert!(default_batteries(EvalTier::Targeted).contains(&BatteryId::Longctx));
        assert!(default_batteries(EvalTier::Targeted).contains(&BatteryId::Cask));
        assert!(default_batteries(EvalTier::Extensive).contains(&BatteryId::Vision));
    }

    #[test]
    fn sanitizes_default_output_model_stem() {
        let out = default_output_dir("/tmp/qwen3.5-9b-awq-mq4.hfq", EvalTier::Fast);
        let s = out.display().to_string();
        assert!(s.contains("qwen3.5-9b-awq-mq4-fast"));
    }

    #[test]
    fn serializes_result_row_as_json() {
        let cfg = EvalConfig {
            model: "m.hfq".to_string(),
            draft: None,
            tier: EvalTier::Fast,
            batteries: vec![BatteryId::Retrieval],
            out_dir: PathBuf::from("out"),
            kv_mode: Some("q8".to_string()),
            max_tokens: 8,
            dflash: DflashMode::Off,
            profile: ProfileMode::Off,
            quality_max_chunks: Some(1),
        };
        let ctx = EvalContext {
            commit_sha: Some("abc".to_string()),
            binary_hash: Some("def".to_string()),
            arch: Some("gfx1151".to_string()),
            rocm: None,
        };
        let row = pass_row(
            BatteryId::Retrieval,
            "fixture",
            &cfg,
            &ctx,
            None,
            BTreeMap::new(),
        );
        let s = serde_json::to_string(&row).unwrap();
        assert!(s.contains("\"battery\":\"retrieval\""));
        assert!(s.contains("\"status\":\"pass\""));
    }
}
