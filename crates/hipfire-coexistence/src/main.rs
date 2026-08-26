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
        (Some("export"), Some("safetensors")) => {
            hipfire_coexistence::export_safetensors::run_cli(&args[2..])
        }
        (Some("repack"), _) => hipfire_coexistence::repack::run_cli(&args[1..]),
        (Some("hub"), Some(op)) => hub_cli(op, &args[2..]),
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
         import safetensors --input <hf_dir> --output <model.hfq> [--arch <family>]\n\
         export safetensors --input <model.hfq> --output <hf_dir> \
         [--arch <family>] [--shard-size 5G]\n\
         hub fetch  --repo <org/name> [--revision <sha|main>] [--include <glob>] \
         [--dest <dir>] [--output <archive.hfa>] [--force] [--raw] [--jobs <n>]\n\
         \x20            default: streams into ~/.hipfire/models/models--Org--Name.hfa,\n\
         \x20            encoding as it downloads so the raw checkpoint is never staged.\n\
         \x20            --raw fetches a HuggingFace cache tree instead. --jobs (default 4)\n\
         \x20            opens that many connections: whole files in raw mode, ranged\n\
         \x20            windows within each file in archive mode.\n\
         hub verify --repo <org/name> [--revision <sha|main>] [--dest <dir>] [--raw] \
         [--only <glob>]\n\
         hub repair --repo <org/name> [--revision <sha|main>] [--dest <dir>] [--raw] \
         [--only <glob>]\n\
         \x20            --only restricts the sweep; both read every byte they cover.\n\
         repack --input <hf_dir> --output <archive.hfa>   (lossless, no arch needed)\n\
         repack --input <archive.hfa> --output <hf_dir>   (restore, byte-identical)\n\
         repack --input <archive.hfa> --check             (verify stored checksums)\n\
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
            let key = a
                .strip_prefix("--")
                .ok_or_else(|| format!("unexpected arg {a}"))?;
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
        Ok(ArgBag {
            values,
            repeated,
            present,
        })
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
            bag.get("target")
                .unwrap_or("/srv/huggingface/models--Qwen--Qwen3.5-397B-A17B"),
        ),
        dflash_source: expand_user(
            bag.get("dflash-source")
                .unwrap_or("/srv/huggingface/models--z-lab--Qwen3.5-397B-A17B-DFlash"),
        ),
        model_name: bag
            .get("model-name")
            .unwrap_or("Qwen3.5-397B-A17B")
            .to_string(),
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
        hipfire: bag
            .get("hipfire")
            .unwrap_or("target/release/hipfire")
            .to_string(),
        quantizer: bag
            .get("quantizer")
            .unwrap_or("target/release/hipfire-quantize")
            .to_string(),
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
        quantizer: bag
            .get("quantizer")
            .unwrap_or("target/release/hipfire-quantize")
            .to_string(),
        hipfire: bag
            .get("hipfire")
            .unwrap_or("target/release/hipfire")
            .to_string(),
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

/// The flat archive name for a repo, matching the store's directory naming so
/// `models--Qwen--Qwen3.5-0.8B/` and `models--Qwen--Qwen3.5-0.8B.hfa` sit side
/// by side and sort together.
fn archive_name(repo: &str) -> String {
    format!("models--{}.hfa", repo.replace('/', "--"))
}

/// Default root for archives: `~/.hipfire/models`, derived from `$HOME` the way
/// every other crate in the tree locates `~/.hipfire`. Deliberately not an env
/// var — no crate reads one for this, and `--dest` already overrides it.
///
/// Raw fetches keep defaulting to the HuggingFace cache root instead, since
/// producing that layout is the whole point of them.
fn archive_root() -> PathBuf {
    match std::env::var("HOME") {
        Ok(home) => PathBuf::from(home).join(".hipfire").join("models"),
        Err(_) => PathBuf::from(".hipfire/models"),
    }
}

/// `hub {fetch,verify,repair}`. Offline tooling — the runtime never links this.
///
/// `fetch` writes an `.hfa` archive by default, encoding as the bytes arrive so
/// the raw checkpoint is never staged. `--raw` restores the older behaviour of
/// materialising a HuggingFace cache tree, which is what you want when another
/// tool has to read the checkpoint as files.
fn hub_cli(op: &str, args: &[String]) -> Result<(), Box<dyn Error>> {
    let val = |k: &str| -> Option<String> {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let repo = val("--repo").ok_or("hub: --repo <org/name> is required")?;
    let revision = val("--revision").unwrap_or_else(|| "main".to_string());
    let raw = args.iter().any(|a| a == "--raw");
    let force = args.iter().any(|a| a == "--force");
    let root = match val("--dest") {
        Some(d) => PathBuf::from(d),
        None if raw => PathBuf::from(
            std::env::var("HF_HOME").unwrap_or_else(|_| "/srv/huggingface".to_string()),
        ),
        None => archive_root(),
    };
    let archive = match val("--output") {
        Some(o) => PathBuf::from(o),
        None => root.join(archive_name(&repo)),
    };
    let include = val("--include");
    // Verify and repair read every byte they cover, so restricting them to one
    // shard is the difference between seconds and hashing the whole repo.
    let only = val("--only");
    // Parallel connections: whole files in raw mode, ranged windows within a
    // file in archive mode (drained in order, so the stream stays sequential).
    let jobs = val("--jobs")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("hub: runtime: {e}"))?;

    rt.block_on(async {
        match op {
            "fetch" if !raw => {
                // These archives are routinely the only copy of their model on
                // an array with no redundancy, so overwriting one is never the
                // silent default.
                if archive.exists() && !force {
                    return Err(format!(
                        "hub: {} already exists — pass --force to replace it, \
                         or `repack --check` it first",
                        archive.display()
                    )
                    .into());
                }
                if let Some(p) = archive.parent() {
                    std::fs::create_dir_all(p)?;
                }
                let files = hipfire_hub::list_files(&repo, &revision).await?;
                hipfire_coexistence::hub_archive::fetch_to_archive(
                    &archive,
                    &repo,
                    &revision,
                    include.as_deref(),
                    files,
                    jobs,
                )
                .await?;
                eprintln!("hub: wrote {}", archive.display());
            }
            "fetch" => {
                let n = hipfire_hub::run::fetch(&root, &repo, &revision, include.as_deref(), jobs)
                    .await?;
                eprintln!("hub: {n} file(s) present and verified");
            }
            // An archive keeps a checksum per stored payload, not the hub's
            // per-file digest, so verifying one is `repack --check` rather than
            // a re-listing. Routing it here means `hub verify` answers the
            // question for whichever form the fetch actually produced.
            "verify" if !raw && archive.exists() => {
                hipfire_coexistence::repack::check(&archive)?;
            }
            "repair" if !raw && archive.exists() => {
                return Err(format!(
                    "hub: {} is an archive — a damaged payload cannot be patched in place. \
                     Re-run `hub fetch --repo {repo} --force`, or restore from it with \
                     `repack --input <archive> --output <dir>` if it still checks out",
                    archive.display()
                )
                .into());
            }
            "verify" => {
                let states =
                    hipfire_hub::run::verify(&root, &repo, &revision, only.as_deref()).await?;
                let mut good = 0;
                let mut gitok = 0;
                let mut bad = 0;
                let mut missing = 0;
                let mut lenonly = 0;
                for (f, s) in &states {
                    use hipfire_hub::run::FileState::*;
                    match s {
                        Good => good += 1,
                        // Reported apart from a SHA-256 match: the git blob
                        // hash is a real content check but a weaker one.
                        GoodGitOid => gitok += 1,
                        Corrupt { want, got, windows } => {
                            // Naming the windows is the point of recording a
                            // table: it turns "this shard is wrong" into the
                            // byte ranges `hub repair` will fetch.
                            if let Some(w) = windows {
                                let span: u64 = w.iter().map(|c| c.len).sum();
                                eprintln!(
                                    "  {} damaged window(s) in {} — {:.2} MB to refetch",
                                    w.len(),
                                    f.path,
                                    span as f64 / 1e6
                                );
                            }
                            bad += 1;
                            eprintln!(
                                "  CORRUPT {} expected {}… got {}…",
                                f.path,
                                &want[..16.min(want.len())],
                                &got[..16.min(got.len())]
                            );
                        }
                        Missing => {
                            missing += 1;
                            eprintln!("  MISSING {}", f.path);
                        }
                        Unreadable(e) => {
                            bad += 1;
                            eprintln!("  UNREADABLE {} ({e})", f.path);
                        }
                        // Reported separately: the hub gives no content hash for
                        // these, so calling them "verified" would overstate it.
                        LengthOnly => lenonly += 1,
                    }
                }
                eprintln!(
                    "hub: {good} verified (sha256), {gitok} verified (git blob sha1), \
{bad} corrupt, {missing} missing, {lenonly} length-only"
                );
                if bad > 0 || missing > 0 {
                    return Err(format!("{} file(s) need repair", bad + missing).into());
                }
            }
            "repair" => {
                let n = hipfire_hub::run::repair(&root, &repo, &revision, only.as_deref()).await?;
                eprintln!("hub: repaired {n} file(s)");
            }
            other => return Err(format!("hub: unknown op {other:?}").into()),
        }
        Ok::<(), Box<dyn Error>>(())
    })
}
