// SPDX-License-Identifier: Apache-2.0
//! hipfire-coexistence — offline import/export/conversion/interop tooling, kept
//! OUT of the inference binaries (daemon/server/runtime) per AGENTS.md.
//!
//! ```text
//! hipfire-coexistence lora export  --hfq <model.hfq> --data-dir <dir> \
//!     [--limit N] [--strength S] [--no-orthogonalize] [--max-seq N] --out <adapter.lora.{hfq,json}>
//! hipfire-coexistence lora merge   --hfq <base.hfq> --adapter <adapter.lora> --out <merged.hfq>
//! hipfire-coexistence lora convert --in <adapter.lora.{hfq,json}> --out <adapter.lora.{hfq,json}>
//! ```
//!
//! `export` derives a rank-1 abliteration adapter by capturing +/- residual means
//! through a `hipfire-daemon` (it needs the model forward); `merge` and `convert`
//! are pure offline file operations.

use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use hipfire_steer::driver::{load_prompts, ModelHarness, Prompt};
use hipfire_steer::{derive_directions, lora, SteerMode};
use hipfire_steer_harness::DaemonHarness;

const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let group = args.first().map(String::as_str);
    let op = args.get(1).map(String::as_str);
    match (group, op) {
        (Some("calibrate"), _) => hipfire_coexistence::calibrate::run_cli(&args[1..]),
        (Some("artifact"), Some("inspect")) => {
            hipfire_coexistence::artifact::run_inspect_cli(&args[2..])
        }
        (Some("artifact"), Some("audit-calibration")) => {
            hipfire_coexistence::calibration_audit::run_cli(&args[2..])
        }
        (Some("artifact"), Some("compare-calibration")) => {
            hipfire_coexistence::calibration_compare::run_cli(&args[2..])
        }
        (Some("artifact"), Some("compare-calibration-stability")) => {
            hipfire_coexistence::calibration_compare::run_stability_cli(&args[2..])
        }
        (Some("artifact"), Some("moe-router-profile")) => {
            hipfire_coexistence::router_profile::run_cli(&args[2..])
        }
        (Some("artifact"), Some("compare-residuals")) => {
            hipfire_coexistence::residual_compare::run_cli(&args[2..])
        }
        (Some("induct"), _) => induct_cli(&args[1..]),
        (Some("two-pass"), _) => two_pass_cli(&args[1..]),
        (Some("lora"), Some("export")) => lora_export(&args[2..]),
        (Some("lora"), Some("merge")) => lora_merge(&args[2..]),
        (Some("lora"), Some("convert")) => lora_convert(&args[2..]),
        (Some("import"), Some("gguf")) => import_gguf(&args[2..]),
        (Some("import"), Some("safetensors")) => {
            hipfire_coexistence::import_safetensors::run_cli(&args[2..])
        }
        #[cfg(target_os = "linux")]
        (Some("npu"), Some("pair-hfp")) => npu_pair_hfp(&args[2..]),
        _ => {
            usage();
            std::process::exit(2);
        }
    }
}

fn usage() {
    eprintln!(
        "usage: hipfire-coexistence <group> <op> [flags]\n\
         \n\
         calibrate --model <safetensors-dir-or-cache-root> --corpus <text> \
         --output <model.calib.hfq> [--sequences N] [--context N] \
         [--sequence-batch auto|N] [--time-tile auto|N] [--max-rows N] \
         [--min-expert-activations N] [--expert-capture-target N] \
         [--expert-capture-tile-rows N] [--expert-coverage-policy \
         strict|preserve-undercovered] [--kldref|--no-kldref] \
         [--kldref-topk N] [--boundary-dir DIR|--boundary-ram] [--resume] \
         [--finalize-completed] [--dry-run]\n\
         [--pause-after-layers N] \
         [--residual-probe-output PATH --residual-probe-rows N] \
         [--cask-output <model.triattn.hfq>] [--cask-only]\n\
         artifact inspect --input <artifact.hfq>\n\
         artifact audit-calibration --input <artifact.calib.hfq>\n\
         artifact compare-calibration --reference <resident.calib.hfq> \
         --candidate <streamed.calib.hfq> [--atol F] [--rtol F] \
         [--max-reports N] [--allow-unproven-provenance]\n\
         artifact compare-calibration-stability --reference <higher-cap.calib.hfq> \
         --candidate <lower-cap.calib.hfq>\n\
         artifact compare-residuals --reference <resident.residuals.hfq> \
         --candidate <streamed.residuals.hfq> [--atol F] [--rtol F] \
         [--max-reports N]\n\
         lora export  --hfq <model.hfq> --data-dir <dir> [--limit N] [--strength S] \
         [--no-orthogonalize] [--max-seq N] --out <adapter.lora.{{hfq,json}}>\n\
         lora merge   --hfq <base.hfq> --adapter <adapter.lora> --out <merged.hfq>\n\
         lora convert --in <adapter.lora.{{hfq,json}}> --out <adapter.lora.{{hfq,json}}>\n\
         import gguf  --in <model.gguf> --out <model.hfq> --format <FMT> \
         [--no-kmap] [--kmap-dense] [--kmap-mode full|alt|typed] [--arch-id N]\n\
         npu pair-hfp --in <whole-scaled.rdna2.hfp> --out <paired.rdna2.hfp>"
    );
}

#[cfg(target_os = "linux")]
fn npu_pair_hfp(args: &[String]) -> Result<(), Box<dyn Error>> {
    let flags = Flags::parse(args)?;
    let input = PathBuf::from(flags.req("in")?);
    let output = PathBuf::from(flags.req("out")?);
    let payload =
        hipfire_xdna::NpuOpusExecutor::prepack_paired_whole_scaled_cached(&output, &input)?;
    eprintln!(
        "wrote paired whole-scaled NPU HFP: {} payload bytes -> {}",
        payload.len(),
        output.display()
    );
    Ok(())
}

/// Import a GGUF checkpoint, re-quantizing its weights to a native `.hfq`
/// through the shared quant codecs (`--format`: bf16, fp16, hfq4, hfq6, mq4,
/// mq6, mq3, mq2, lloyd-mq*, hfp4, mfp4). The re-quant pipeline lives in the
/// `hipfire-quantize` library; this is the user-facing entry point, kept out of
/// the quantize binary per AGENTS.md. Pure offline file op.
fn import_gguf(args: &[String]) -> Result<(), Box<dyn Error>> {
    use hipfire_quantize::gguf_import::run_gguf_pipeline;
    use hipfire_quantize::quant_plan::GgufFormat;

    // --in/--out/--format take values; the kmap flags are parsed positionally
    // (mirrors the quantize CLI) since `--no-kmap`/`--kmap-dense` are presence
    // flags and `run_gguf_pipeline` also reads `--arch-id` from the process
    // args directly.
    let val = |k: &str| -> Option<&str> {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    let input = val("--in").ok_or("import gguf: --in <model.gguf> is required")?;
    let output = val("--out").ok_or("import gguf: --out <model.hfq> is required")?;
    let format = val("--format").ok_or("import gguf: --format <FMT> is required")?;
    let gguf_format = GgufFormat::from_flag(format).ok_or_else(|| {
        format!(
            "import gguf: --format '{format}' not recognized \
             (bf16, fp16, hfq4, hfq6, mq4, mq6, mq3, mq2, lloyd-mq*, hfp4, mfp4)"
        )
    })?;
    let no_kmap = args.iter().any(|a| a == "--no-kmap" || a == "--uniform");
    let kmap_dense = args.iter().any(|a| a == "--kmap-dense");
    let kmap_mode: u8 = val("--kmap-mode")
        .map(|v| match v {
            "full" | "0" => 0,
            "alternating" | "alt" | "1" => 1,
            "typed" | "2" => 2,
            other => {
                eprintln!("warning: unknown --kmap-mode '{other}', using alternating");
                1
            }
        })
        .unwrap_or(1);

    run_gguf_pipeline(
        Path::new(input),
        Path::new(output),
        gguf_format,
        format,
        no_kmap,
        kmap_dense,
        kmap_mode,
    )?;
    Ok(())
}

/// Minimal `--key value` flag bag. `--no-orthogonalize` is a presence-only flag.
struct Flags(HashMap<String, String>);

impl Flags {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut m = HashMap::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let key = a
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected arg {a}"))?;
            if key == "no-orthogonalize" {
                m.insert(key.to_string(), "1".to_string());
                continue;
            }
            let v = it.next().ok_or_else(|| format!("{a} needs a value"))?;
            m.insert(key.to_string(), v.clone());
        }
        Ok(Flags(m))
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(String::as_str)
    }
    fn req(&self, k: &str) -> Result<&str, String> {
        self.get(k).ok_or_else(|| format!("--{k} is required"))
    }
    fn has(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
}

/// Write an adapter to the binary container for `*.hfq` paths, JSON otherwise.
fn write_adapter_dispatch(out: &Path, adapter: &lora::LoraAdapter) -> Result<(), Box<dyn Error>> {
    if out.extension().and_then(|e| e.to_str()) == Some("hfq") {
        hipfire_lora_hfq::write_lora_hfq(out, adapter)?;
    } else {
        lora::write_adapter(out, adapter)?;
    }
    Ok(())
}

fn load_set(dir: &Path, name: &str, limit: usize) -> Result<Vec<Prompt>, Box<dyn Error>> {
    let path = dir.join(name);
    let mut prompts = load_prompts(&path, SYSTEM_PROMPT)
        .map_err(|e| format!("loading {}: {e}", path.display()))?;
    prompts.truncate(limit);
    Ok(prompts)
}

/// Capture +/- residual means through a daemon, derive per-block directions, and
/// write a rank-1 ablate adapter. `--strength` seeds the adapter's default scale.
fn lora_export(args: &[String]) -> Result<(), Box<dyn Error>> {
    let f = Flags::parse(args)?;
    let hfq = f.req("hfq")?;
    let data_dir = PathBuf::from(f.req("data-dir")?);
    let limit: usize = f
        .get("limit")
        .unwrap_or("16")
        .parse()
        .map_err(|_| "bad --limit")?;
    let strength: f32 = f
        .get("strength")
        .unwrap_or("0.2")
        .parse()
        .map_err(|_| "bad --strength")?;
    let max_seq: usize = f
        .get("max-seq")
        .unwrap_or("2048")
        .parse()
        .map_err(|_| "bad --max-seq")?;
    let orthogonalize = !f.has("no-orthogonalize");
    let out = PathBuf::from(f.req("out")?);

    let good = load_set(&data_dir, "good_prompts.txt", limit)?;
    let bad = load_set(&data_dir, "bad_prompts.txt", limit)?;

    let daemon_bin = hipfire_daemon_adapter::find_daemon_bin_or_error()?;
    let tmp = std::env::temp_dir().join(format!("hipfire-coex-{}", std::process::id()));
    let mut h = DaemonHarness::connect(
        &daemon_bin,
        Path::new(hfq),
        max_seq,
        64,
        SYSTEM_PROMPT.to_string(),
        tmp,
    )?;

    eprintln!(
        "capturing directions ({} good / {} bad, {} layers) ...",
        good.len(),
        bad.len(),
        h.num_layers()
    );
    h.begin_capture()
        .map_err(|e| format!("begin_capture: {e}"))?;
    h.capture(&good).map_err(|e| format!("capture good: {e}"))?;
    let good_means = h
        .finish_capture()
        .map_err(|e| format!("finish good: {e}"))?;
    h.begin_capture()
        .map_err(|e| format!("begin_capture: {e}"))?;
    h.capture(&bad).map_err(|e| format!("capture bad: {e}"))?;
    let bad_means = h.finish_capture().map_err(|e| format!("finish bad: {e}"))?;

    let directions = derive_directions(&good_means, &bad_means, orthogonalize);
    let layers = h.num_layers();
    let adapter = lora::abliteration_adapter(
        "abliterate",
        &directions,
        SteerMode::Ablate,
        strength,
        0..layers,
    )?;
    write_adapter_dispatch(&out, &adapter)?;
    eprintln!(
        "wrote LoRA adapter: {} rank-1 deltas, default scale {strength:.2} → {}",
        adapter.deltas.len(),
        out.display()
    );
    Ok(())
}

/// Bundle an adapter into a base model `.hfq` → a self-contained model the daemon
/// auto-applies on load. Pure offline file op.
fn lora_merge(args: &[String]) -> Result<(), Box<dyn Error>> {
    let f = Flags::parse(args)?;
    let base = PathBuf::from(f.req("hfq")?);
    let adapter_path = PathBuf::from(f.req("adapter")?);
    let out = PathBuf::from(f.req("out")?);
    let adapter = hipfire_lora_hfq::read_lora_any(&adapter_path)?;
    hipfire_lora_hfq::merge_lora_into_model(&base, &adapter, &out)?;
    eprintln!(
        "merged {} + {} ({} deltas, scale {:.2}) → {}",
        base.display(),
        adapter_path.display(),
        adapter.deltas.len(),
        adapter.scale,
        out.display()
    );
    Ok(())
}

/// Split `args` at a bare `--`: everything after it is the extra quantizer flags
/// (`quant_args`), mirroring argparse's `REMAINDER` capture.
fn split_quant_args(args: &[String]) -> (Vec<String>, Vec<String>) {
    match args.iter().position(|a| a == "--") {
        Some(i) => (args[..i].to_vec(), args[i + 1..].to_vec()),
        None => (args.to_vec(), Vec::new()),
    }
}

/// A tiny value/presence/repeatable flag bag for the induction CLIs.
struct ArgBag {
    values: HashMap<String, String>,
    repeated: HashMap<String, Vec<String>>,
    present: std::collections::HashSet<String>,
}

impl ArgBag {
    fn parse(args: &[String], presence: &[&str], repeatable: &[&str]) -> Result<Self, String> {
        let mut values = HashMap::new();
        let mut repeated: HashMap<String, Vec<String>> = HashMap::new();
        let mut present = std::collections::HashSet::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let key = a.strip_prefix("--").ok_or_else(|| format!("unexpected arg {a}"))?;
            if presence.contains(&key) {
                present.insert(key.to_string());
            } else {
                let v = it.next().ok_or_else(|| format!("--{key} needs a value"))?;
                if repeatable.contains(&key) {
                    repeated.entry(key.to_string()).or_default().push(v.clone());
                } else {
                    values.insert(key.to_string(), v.clone());
                }
            }
        }
        Ok(ArgBag { values, repeated, present })
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.values.get(k).map(String::as_str)
    }
    fn u64(&self, k: &str, default: u64) -> Result<u64, String> {
        match self.get(k) {
            Some(v) => v.parse().map_err(|_| format!("--{k} must be an integer")),
            None => Ok(default),
        }
    }
    fn f64(&self, k: &str, default: f64) -> Result<f64, String> {
        match self.get(k) {
            Some(v) => v.parse().map_err(|_| format!("--{k} must be a number")),
            None => Ok(default),
        }
    }
    fn has(&self, k: &str) -> bool {
        self.present.contains(k)
    }
}

const INDUCT_PRESENCE: &[&str] = &["force", "dry-run"];
const TWO_PASS_PRESENCE: &[&str] = &["skip-calib", "dry-run"];

/// `hipfire-coexistence induct` — the Rust port of `scripts/induct_model.py`.
fn induct_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    use hipfire_coexistence::induction::orchestrate::{
        default_quant_args, run, InductConfig, STAGES,
    };
    let (flags, quant_tail) = split_quant_args(args);
    let bag = ArgBag::parse(&flags, INDUCT_PRESENCE, &["dflash-format", "stage"])?;
    let quant_format = bag.get("format").unwrap_or("oq4.25++").to_string();
    let quant_args = if quant_tail.is_empty() {
        default_quant_args(&quant_format)
    } else {
        quant_tail
    };
    let dflash_formats = bag
        .repeated
        .get("dflash-format")
        .cloned()
        .unwrap_or_else(|| vec!["bf16".into(), "f16".into()]);
    let stages = bag
        .repeated
        .get("stage")
        .cloned()
        .unwrap_or_else(|| STAGES.iter().map(|s| s.to_string()).collect());
    let corpus = expand_user(bag.get("corpus").unwrap_or("benchmarks/calib/calib-5m.txt"));
    let cfg = InductConfig {
        target: expand_user(
            bag.get("target").unwrap_or("/srv/huggingface/models--Qwen--Qwen3.5-397B-A17B"),
        ),
        dflash_source: expand_user(
            bag.get("dflash-source")
                .unwrap_or("/srv/huggingface/models--z-lab--Qwen3.5-397B-A17B-DFlash"),
        ),
        model_name: bag.get("model-name").unwrap_or("Qwen3.5-397B-A17B").to_string(),
        artifact_root: expand_user(bag.get("artifact-root").unwrap_or("~/.hipfire")),
        quant_format,
        dflash_formats,
        corpus,
        n_sequences: bag.u64("n-sequences", 128)?,
        ctx_len: bag.u64("ctx-len", 2048)?,
        batch_size: bag.u64("batch-size", 64)?,
        time_tile: bag.u64("time-tile", 32)?,
        max_rows: bag.u64("max-rows", 2048)?,
        layer_prefetch_bytes: bag.u64("layer-prefetch-bytes", 16 * 1024 * 1024 * 1024)?,
        kldref_topk: bag.u64("kldref-topk", 64)?,
        min_expert_activations: bag.u64("min-expert-activations", 2048)?,
        expert_capture_target: bag.u64("expert-capture-target", 4096)?,
        expert_capture_tile_rows: bag.u64("expert-capture-tile-rows", 256)?,
        required_expert_fraction: bag.f64("required-expert-fraction", 1.0)?,
        sampling_seed: bag.u64("sampling-seed", 1)?,
        expert_coverage_policy: bag
            .get("expert-coverage-policy")
            .unwrap_or("preserve-undercovered")
            .to_string(),
        triattn_max_tokens: bag.u64("triattn-max-tokens", 100_000)?,
        triattn_chunk_len: bag.u64("triattn-chunk-len", 1024)?,
        quant_args,
        stages,
        force: bag.has("force"),
        dry_run: bag.has("dry-run"),
        hipfire: bag.get("hipfire").unwrap_or("target/release/hipfire").to_string(),
        quantizer: bag.get("quantizer").unwrap_or("target/release/hipfire-quantize").to_string(),
        dflash_converter: bag
            .get("dflash-converter")
            .unwrap_or("target/release/dflash_convert")
            .to_string(),
        triattn_bin: bag
            .get("triattn-bin")
            .unwrap_or("target/release/examples/triattn_validate")
            .to_string(),
    };
    run(&cfg)?;
    Ok(())
}

/// `hipfire-coexistence two-pass` — the Rust port of
/// `scripts/two_pass_quantize.py` (calibrate then quantize).
fn two_pass_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    use hipfire_coexistence::induction::two_pass::{default_manifest_path, run, TwoPassConfig};
    let (flags, quant_args) = split_quant_args(args);
    let bag = ArgBag::parse(&flags, TWO_PASS_PRESENCE, &[])?;
    let calib = expand_user(bag.get("calib").ok_or("two-pass requires --calib")?);
    let output = expand_user(bag.get("output").ok_or("two-pass requires --output")?);
    if !calib.to_string_lossy().ends_with(".calib.hfq") {
        return Err("--calib must end in .calib.hfq".into());
    }
    if !output.to_string_lossy().ends_with(".hfq") {
        return Err("--output must end in .hfq".into());
    }
    let manifest = bag
        .get("manifest")
        .map(expand_user)
        .unwrap_or_else(|| default_manifest_path(&output));
    let cfg = TwoPassConfig {
        model: expand_user(bag.get("model").ok_or("two-pass requires --model")?),
        calib,
        output,
        manifest,
        quant_format: bag.get("format").unwrap_or("oq4.25++").to_string(),
        corpus: expand_user(bag.get("corpus").unwrap_or("wikitext")),
        n_sequences: bag.u64("n-sequences", 128)?,
        ctx_len: bag.u64("ctx-len", 2048)?,
        batch_size: bag.u64("batch-size", 64)?,
        time_tile: bag.u64("time-tile", 32)?,
        max_rows: bag.u64("max-rows", 2048)?,
        layer_prefetch_bytes: bag.u64("layer-prefetch-bytes", 16 * 1024 * 1024 * 1024)?,
        kldref_topk: bag.u64("kldref-topk", 64)?,
        min_expert_activations: bag.u64("min-expert-activations", 2048)?,
        expert_capture_target: bag.u64("expert-capture-target", 4096)?,
        expert_capture_tile_rows: bag.u64("expert-capture-tile-rows", 256)?,
        required_expert_fraction: bag.f64("required-expert-fraction", 1.0)?,
        sampling_seed: bag.u64("sampling-seed", 1)?,
        expert_coverage_policy: bag
            .get("expert-coverage-policy")
            .unwrap_or("preserve-undercovered")
            .to_string(),
        quant_args,
        quantizer: bag.get("quantizer").unwrap_or("target/release/hipfire-quantize").to_string(),
        hipfire: bag.get("hipfire").unwrap_or("target/release/hipfire").to_string(),
        skip_calib: bag.has("skip-calib"),
        dry_run: bag.has("dry-run"),
    };
    run(&cfg)?;
    Ok(())
}

/// Expand a leading `~` to `$HOME`. Everything else is left verbatim; the
/// recipe layer resolves paths through `python_resolve`.
fn expand_user(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

/// Convert an adapter container between the binary (`.hfq`) and JSON forms
/// (dispatched by the output extension). Pure offline file op.
fn lora_convert(args: &[String]) -> Result<(), Box<dyn Error>> {
    let f = Flags::parse(args)?;
    let inp = PathBuf::from(f.req("in")?);
    let out = PathBuf::from(f.req("out")?);
    let adapter = hipfire_lora_hfq::read_lora_any(&inp)?;
    write_adapter_dispatch(&out, &adapter)?;
    eprintln!(
        "converted {} → {} ({} deltas)",
        inp.display(),
        out.display(),
        adapter.deltas.len()
    );
    Ok(())
}
