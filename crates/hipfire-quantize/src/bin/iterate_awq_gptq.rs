//! `iterate_awq_gptq` — orchestrator for the AWQ-aware-GPTQ v3 iterative
//! refinement algorithm in Rust. Tier-1-native counterpart to the Python
//! prototype at `scripts/mq4_masked_calib.py::run_iterative_awq_gptq`.
//!
//! ## Algorithm
//!
//! ```text
//! prev_scales = None
//! for round in 0..max_rounds:
//!     H_round = collect_hessian(model_prev_round_or_bf16)
//!     raw_scales = compute_awq_scales_from_hessian_diag(H_round, alpha)
//!     s_round = (1 - damp) * prev_scales + damp * raw_scales
//!                  (round 0 just uses raw; damp not yet applied)
//!     write s_round to AWQ scales sidecar (HFSC v1)
//!     hipfire-quantize --awq --gptq H_round --awq-scales s_round  →  model_N.hfq
//!     if round > 0 and relative_l2_delta(prev, s_round) < epsilon: break
//!     prev_scales = s_round
//! ```
//!
//! ## Pipeline integration
//!
//! Round 0 — BF16 forward (this is what existing infrastructure already
//! does):
//! 1. `collect_hessian --hf-model <dir> --output round_0/H.bin` (already
//!    in tree at `crates/hipfire-runtime/src/bin/collect_hessian.rs`).
//!
//! Round N+ — quantized forward (NOT YET IMPLEMENTED):
//! 1. Would need a `collect_hessian_quantized` bin that loads the round
//!    N-1 `.hfq` via the production loader and runs forward through MQ4
//!    quantized weights, capturing per-tensor x·xᵀ via the same
//!    `HessianCollector` capture hook (see
//!    `crates/hipfire-runtime/src/calibration.rs`). Recommended approach:
//!    dequantize MQ4 → BF16 in memory (mirrors Python's
//!    `install_candidate_mq4_weights`) and reuse `forward_prefill_bf16`.
//!
//! This orchestrator runs round 0 unconditionally and falls through to
//! round 1+ only when the user explicitly supplies `--round-1-hessian
//! <path>` (escape hatch for users who collect Hessians out-of-band, e.g.
//! the Python pipeline). When `--round-1-hessian` is omitted, round 1+
//! is skipped with a logged warning. See the report at
//! `docs/plans/imatrix-tier1-hipfire-native.md` for the planned
//! sub-binary.
//!
//! ## Usage
//!
//! ```ignore
//! cargo run --release -p hipfire-quantize --bin iterate_awq_gptq -- \
//!   --hf-model <bf16-hf-dir> \
//!   --base-output-dir <round-dir>/ \
//!   --quantize-format mq4 \
//!   --awq-alpha 0.55 \
//!   --damping 0.5 \
//!   --epsilon 0.01 \
//!   --max-rounds 3 \
//!   [--initial-hessian <H_round_0.bin>]  # skips round-0 collection
//!   [--corpus <calib.txt>]               # required if no --initial-hessian
//!   [--bf16-gguf <path.gguf>]            # forwarded to collect_hessian
//!   [--collect-hessian-bin <path>]       # default ./target/release/collect_hessian
//!   [--hipfire-quantize-bin <path>]      # default ./target/release/hipfire-quantize
//! ```

use byteorder::{ByteOrder, LittleEndian};
use hipfire_quantize::awq_compute::{
    compute_awq_scales_from_hessian_diag, damp_awq_scales, relative_l2_delta,
};
use hipfire_quantize::awq_scales_io::{write_awq_scales, AwqScaleEntry};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug)]
struct Args {
    /// HuggingFace BF16 model dir (`*.safetensors` + `config.json`).
    hf_model: PathBuf,
    /// Per-round output directory (will get `round_0/`, `round_1/`, ...).
    base_output_dir: PathBuf,
    /// `--format` arg forwarded to hipfire-quantize (default: mq4).
    quantize_format: String,
    /// AWQ alpha (default 0.55, hipfire production default).
    awq_alpha: f32,
    /// Damping coefficient: `s_round = (1 - β) * prev + β * raw`. Default 0.5
    /// matches the Python prototype's setting.
    damping: f32,
    /// Convergence threshold on the relative-L2 delta between consecutive
    /// scale vectors. Default 0.01 matches the Python prototype.
    epsilon: f32,
    /// Max iteration count. Default 3 (Python default is 6; we use 3 since
    /// the marginal KLD gain drops off rapidly past round 2 on 0.8B).
    max_rounds: usize,
    /// If provided, skip round-0 Hessian collection and use this HFHS file
    /// as the round-0 Hessian. Lets the orchestrator integrate with a
    /// pre-existing collect_hessian artifact.
    initial_hessian: Option<PathBuf>,
    /// Calibration corpus (path or HF dataset id). Required when
    /// `initial_hessian` is None — forwarded to collect_hessian.
    corpus: Option<PathBuf>,
    /// BF16 GGUF carrying the tokenizer metadata. Required when
    /// `initial_hessian` is None and the default tokenizer is llama-cpp.
    bf16_gguf: Option<PathBuf>,
    /// Path to the `collect_hessian` binary (round-0 collection).
    collect_hessian_bin: PathBuf,
    /// Path to the `hipfire-quantize` binary.
    hipfire_quantize_bin: PathBuf,
    /// Optional per-round Hessian for rounds 1..N. Each entry is consumed
    /// by the corresponding round (entry 0 → round 1's Hessian, etc.).
    /// When the orchestrator runs out of entries it stops the loop. Lets
    /// users supply round-1+ Hessians out-of-band while we don't have a
    /// `collect_hessian_quantized` bin yet.
    round_n_hessians: Vec<PathBuf>,
    /// AWQ scope (f1 or f2). Forwarded to hipfire-quantize. Default: f1
    /// (production winner per AWQ_SCOPE_F1 docs).
    awq_scope: String,
    /// AWQ formula (paper or autoawq). Forwarded to hipfire-quantize.
    awq_formula: String,
    /// GPTQ initial damping. Forwarded to hipfire-quantize.
    gptq_damp: f64,
    /// Optional imatrix path forwarded to hipfire-quantize. Required by
    /// the --awq path in hipfire-quantize for tensors NOT covered by the
    /// HFSC override (e.g. F1 sidecar emission for tensors absent from
    /// the round Hessian).
    imatrix: Option<PathBuf>,
    /// Optional. Pass through to hipfire-quantize so the kmap/quant
    /// pipeline knows the right tensor coverage list. Falls back to
    /// quantize defaults otherwise.
    extra_quantize_args: Vec<String>,
}

fn print_usage() {
    eprintln!(
        "Usage:\n  \
           iterate_awq_gptq --hf-model <dir> --base-output-dir <dir> \\\n    \
           [--quantize-format <fmt>]   (default: mq4)\n    \
           [--awq-alpha <f>]           (default: 0.55)\n    \
           [--damping <f>]             (default: 0.5)\n    \
           [--epsilon <f>]             (default: 0.01)\n    \
           [--max-rounds <N>]          (default: 3)\n    \
           [--initial-hessian <path>]  (skip round-0 collection)\n    \
           [--corpus <path>]           (required if --initial-hessian unset)\n    \
           [--bf16-gguf <path>]        (forwarded to collect_hessian)\n    \
           [--collect-hessian-bin <path>] (default: target/release/collect_hessian)\n    \
           [--hipfire-quantize-bin <path>] (default: target/release/hipfire-quantize)\n    \
           [--round-n-hessian <path>]  (repeatable; supplies round-1+ Hessians)\n    \
           [--awq-scope <f1|f2>]       (default: f1)\n    \
           [--awq-formula <paper|autoawq>] (default: paper)\n    \
           [--gptq-damp <f>]           (default: 0.01)\n    \
           [--imatrix <path>]          (forwarded to hipfire-quantize)\n    \
           [-- <extra args for hipfire-quantize>...]\n\
         \n\
         AWQ-aware-GPTQ v3 iterative refinement orchestrator. Drives the\n\
         round-0..N loop from `scripts/mq4_masked_calib.py::run_iterative_awq_gptq`.\n\
         Round 0 uses a BF16 Hessian (collect_hessian or --initial-hessian).\n\
         Round 1+ requires --round-n-hessian per round until a Tier 1\n\
         collect_hessian_quantized bin lands."
    );
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut hf_model: Option<PathBuf> = None;
    let mut base_output_dir: Option<PathBuf> = None;
    let mut quantize_format = "mq4".to_string();
    let mut awq_alpha: f32 = 0.55;
    let mut damping: f32 = 0.5;
    let mut epsilon: f32 = 0.01;
    let mut max_rounds: usize = 3;
    let mut initial_hessian: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut bf16_gguf: Option<PathBuf> = None;
    let mut collect_hessian_bin = PathBuf::from("target/release/collect_hessian");
    let mut hipfire_quantize_bin = PathBuf::from("target/release/hipfire-quantize");
    let mut round_n_hessians: Vec<PathBuf> = Vec::new();
    let mut awq_scope = "f1".to_string();
    let mut awq_formula = "paper".to_string();
    let mut gptq_damp: f64 = 0.01;
    let mut imatrix: Option<PathBuf> = None;
    let mut extra_quantize_args: Vec<String> = Vec::new();
    let mut in_passthrough = false;

    let mut i = 1;
    while i < argv.len() {
        if in_passthrough {
            extra_quantize_args.push(argv[i].clone());
            i += 1;
            continue;
        }
        match argv[i].as_str() {
            "--" => { in_passthrough = true; i += 1; }
            "--hf-model" => { hf_model = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--base-output-dir" => { base_output_dir = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--quantize-format" => { quantize_format = argv[i + 1].clone(); i += 2; }
            "--awq-alpha" => { awq_alpha = argv[i + 1].parse().unwrap_or_else(|_| {
                eprintln!("error: --awq-alpha must be float"); std::process::exit(1);
            }); i += 2; }
            "--damping" => { damping = argv[i + 1].parse().unwrap_or_else(|_| {
                eprintln!("error: --damping must be float"); std::process::exit(1);
            }); i += 2; }
            "--epsilon" => { epsilon = argv[i + 1].parse().unwrap_or_else(|_| {
                eprintln!("error: --epsilon must be float"); std::process::exit(1);
            }); i += 2; }
            "--max-rounds" => { max_rounds = argv[i + 1].parse().unwrap_or_else(|_| {
                eprintln!("error: --max-rounds must be positive int"); std::process::exit(1);
            }); i += 2; }
            "--initial-hessian" => { initial_hessian = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--corpus" => { corpus = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--bf16-gguf" => { bf16_gguf = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "--collect-hessian-bin" => { collect_hessian_bin = PathBuf::from(&argv[i + 1]); i += 2; }
            "--hipfire-quantize-bin" => { hipfire_quantize_bin = PathBuf::from(&argv[i + 1]); i += 2; }
            "--round-n-hessian" => { round_n_hessians.push(PathBuf::from(&argv[i + 1])); i += 2; }
            "--awq-scope" => { awq_scope = argv[i + 1].clone(); i += 2; }
            "--awq-formula" => { awq_formula = argv[i + 1].clone(); i += 2; }
            "--gptq-damp" => { gptq_damp = argv[i + 1].parse().unwrap_or_else(|_| {
                eprintln!("error: --gptq-damp must be float"); std::process::exit(1);
            }); i += 2; }
            "--imatrix" => { imatrix = Some(PathBuf::from(&argv[i + 1])); i += 2; }
            "-h" | "--help" => { print_usage(); std::process::exit(0); }
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
    }
    let hf_model = hf_model.unwrap_or_else(|| {
        eprintln!("error: --hf-model is required"); print_usage(); std::process::exit(1);
    });
    let base_output_dir = base_output_dir.unwrap_or_else(|| {
        eprintln!("error: --base-output-dir is required"); print_usage(); std::process::exit(1);
    });
    if initial_hessian.is_none() && corpus.is_none() {
        eprintln!("error: --corpus is required when --initial-hessian is unset");
        std::process::exit(1);
    }
    Args {
        hf_model, base_output_dir, quantize_format, awq_alpha, damping, epsilon,
        max_rounds, initial_hessian, corpus, bf16_gguf, collect_hessian_bin,
        hipfire_quantize_bin, round_n_hessians, awq_scope, awq_formula, gptq_damp,
        imatrix, extra_quantize_args,
    }
}

/// HFHS v1 reader, minimal subset — index + per-record name/K/payload. We
/// can't reuse `hessian_io::HessianSidecar` from a binary because the
/// quantize crate doesn't re-export its private TensorEntry struct fields
/// we need (payload offset). Single-purpose reader keeps the orchestrator
/// independent.
struct HfhsReader {
    mmap: Mmap,
    entries: Vec<HfhsEntry>,
}

struct HfhsEntry {
    name: String,
    expert_idx: u32,
    k: usize,
    payload_offset: usize,
    dtype_size: usize,
}

impl HfhsReader {
    fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("mmap {}: {e}", path.display()))?;
        if mmap.len() < 24 {
            return Err(format!("HFHS file too small: {} bytes", mmap.len()));
        }
        if &mmap[0..4] != b"HFHS" {
            return Err(format!("bad HFHS magic: {:?}", &mmap[0..4]));
        }
        let version = LittleEndian::read_u32(&mmap[4..8]);
        if version != 1 {
            return Err(format!("unsupported HFHS version {version}"));
        }
        let n_tensors = LittleEndian::read_u64(&mmap[8..16]) as usize;
        let mut entries = Vec::with_capacity(n_tensors);
        let mut pos = 24usize;
        for _ in 0..n_tensors {
            if pos + 4 > mmap.len() {
                return Err("truncated HFHS".to_string());
            }
            let name_len = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
            pos += 4;
            if pos + name_len + 12 > mmap.len() {
                return Err("truncated HFHS record".to_string());
            }
            let name = std::str::from_utf8(&mmap[pos..pos + name_len])
                .map_err(|e| format!("bad utf8: {e}"))?
                .to_string();
            pos += name_len;
            let expert_idx = LittleEndian::read_u32(&mmap[pos..pos + 4]);
            pos += 4;
            let k = LittleEndian::read_u32(&mmap[pos..pos + 4]) as usize;
            pos += 4;
            let dtype_flag = LittleEndian::read_u32(&mmap[pos..pos + 4]);
            pos += 4;
            let dtype_size = match dtype_flag {
                1 => 4usize, // F32
                2 => 8usize, // F64
                d => return Err(format!("unknown HFHS dtype {d}")),
            };
            let payload_bytes = k * k * dtype_size;
            if pos + payload_bytes > mmap.len() {
                return Err("truncated HFHS payload".to_string());
            }
            entries.push(HfhsEntry { name, expert_idx, k, payload_offset: pos, dtype_size });
            pos += payload_bytes;
        }
        Ok(Self { mmap, entries })
    }

    fn iter(&self) -> impl Iterator<Item = &HfhsEntry> {
        self.entries.iter()
    }

    fn payload(&self, e: &HfhsEntry) -> &[u8] {
        &self.mmap[e.payload_offset..e.payload_offset + e.k * e.k * e.dtype_size]
    }
}

#[derive(Debug, Clone)]
struct RoundRecord {
    round: usize,
    hessian_path: PathBuf,
    scales_path: PathBuf,
    model_path: PathBuf,
    n_scales: usize,
    scale_delta_vs_prev: f64,
    converged: bool,
    elapsed_seconds: f64,
}

fn write_summary_json(path: &Path, args: &Args, trace: &[RoundRecord], total_elapsed: f64) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"schema\": \"hipfire.iterate_awq_gptq.v1\",")?;
    writeln!(f, "  \"hf_model\": \"{}\",", args.hf_model.display())?;
    writeln!(f, "  \"base_output_dir\": \"{}\",", args.base_output_dir.display())?;
    writeln!(f, "  \"quantize_format\": \"{}\",", args.quantize_format)?;
    writeln!(f, "  \"awq_alpha\": {},", args.awq_alpha)?;
    writeln!(f, "  \"damping\": {},", args.damping)?;
    writeln!(f, "  \"epsilon\": {},", args.epsilon)?;
    writeln!(f, "  \"max_rounds\": {},", args.max_rounds)?;
    writeln!(f, "  \"awq_scope\": \"{}\",", args.awq_scope)?;
    writeln!(f, "  \"awq_formula\": \"{}\",", args.awq_formula)?;
    writeln!(f, "  \"elapsed_seconds\": {:.3},", total_elapsed)?;
    writeln!(f, "  \"rounds\": [")?;
    for (i, r) in trace.iter().enumerate() {
        let comma = if i + 1 < trace.len() { "," } else { "" };
        writeln!(f, "    {{")?;
        writeln!(f, "      \"round\": {},", r.round)?;
        writeln!(f, "      \"hessian_path\": \"{}\",", r.hessian_path.display())?;
        writeln!(f, "      \"scales_path\": \"{}\",", r.scales_path.display())?;
        writeln!(f, "      \"model_path\": \"{}\",", r.model_path.display())?;
        writeln!(f, "      \"n_scales\": {},", r.n_scales)?;
        writeln!(f, "      \"scale_delta_vs_prev\": {:.10e},", r.scale_delta_vs_prev)?;
        writeln!(f, "      \"converged\": {},", r.converged)?;
        writeln!(f, "      \"elapsed_seconds\": {:.3}", r.elapsed_seconds)?;
        writeln!(f, "    }}{comma}")?;
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}

/// Stage 1: round-0 BF16 Hessian. Either re-use a supplied HFHS file or
/// invoke the `collect_hessian` binary against the BF16 model.
fn obtain_round_0_hessian(args: &Args, round_dir: &Path) -> Result<PathBuf, String> {
    if let Some(initial) = &args.initial_hessian {
        eprintln!("[iterate] round 0: re-using initial Hessian {}", initial.display());
        // Symlink (or copy) into the round dir for hermetic traceability.
        let dst = round_dir.join("hessian.bin");
        if dst.exists() {
            std::fs::remove_file(&dst).ok();
        }
        // Hard link first (cheap), fallback to copy if cross-FS.
        if std::fs::hard_link(initial, &dst).is_err() {
            std::fs::copy(initial, &dst).map_err(|e| format!(
                "copy {} -> {}: {e}", initial.display(), dst.display()
            ))?;
        }
        return Ok(dst);
    }
    let corpus = args.corpus.as_ref().ok_or_else(|| {
        "internal error: corpus must be set when initial_hessian is None".to_string()
    })?;
    let output = round_dir.join("hessian.bin");
    let mut cmd = Command::new(&args.collect_hessian_bin);
    cmd.arg("--hf-model").arg(&args.hf_model)
       .arg("--corpus").arg(corpus)
       .arg("--output").arg(&output);
    if let Some(gguf) = &args.bf16_gguf {
        cmd.arg("--bf16-gguf").arg(gguf);
    } else {
        // Fallback: use the hipfire native tokenizer (no byte-parity with
        // PyTorch oracle but works without llama.cpp).
        cmd.arg("--tokenizer").arg("hipfire");
    }
    eprintln!(
        "[iterate] round 0: invoking collect_hessian:\n  {} {}",
        args.collect_hessian_bin.display(),
        cmd.get_args()
            .map(|s| s.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let status = cmd.status().map_err(|e| format!(
        "spawn collect_hessian: {e} (binary path: {})",
        args.collect_hessian_bin.display()
    ))?;
    if !status.success() {
        return Err(format!(
            "collect_hessian exited with status {} — see stderr above",
            status
        ));
    }
    Ok(output)
}

/// Compute AWQ scales from a Hessian sidecar by reading per-tensor
/// diagonals. Mirrors Python's `compute_awq_scale_dict` over the round
/// Hessian's tensors.
fn compute_round_scales(
    hessian: &HfhsReader,
    alpha: f32,
) -> Vec<(String, u32, Vec<f32>)> {
    let mut out = Vec::with_capacity(hessian.entries.len());
    for e in hessian.iter() {
        if e.dtype_size != 4 {
            eprintln!(
                "warning: tensor {} expert={} has dtype size {} (only F32 supported); skipping",
                e.name, e.expert_idx, e.dtype_size
            );
            continue;
        }
        let bytes = hessian.payload(e);
        let scales = compute_awq_scales_from_hessian_diag(bytes, e.k, alpha);
        out.push((e.name.clone(), e.expert_idx, scales));
    }
    out
}

fn run() -> Result<(), String> {
    let args = parse_args();
    eprintln!("iterate_awq_gptq — AWQ-aware-GPTQ v3 orchestrator");
    eprintln!("  hf-model:               {}", args.hf_model.display());
    eprintln!("  base-output-dir:        {}", args.base_output_dir.display());
    eprintln!("  quantize-format:        {}", args.quantize_format);
    eprintln!("  awq-alpha:              {}", args.awq_alpha);
    eprintln!("  damping:                {}", args.damping);
    eprintln!("  epsilon:                {}", args.epsilon);
    eprintln!("  max-rounds:             {}", args.max_rounds);
    eprintln!("  awq-scope:              {}", args.awq_scope);
    eprintln!("  awq-formula:            {}", args.awq_formula);
    eprintln!("  gptq-damp:              {}", args.gptq_damp);
    eprintln!("  round-n-hessians:       {} pre-supplied", args.round_n_hessians.len());
    eprintln!("  hipfire-quantize-bin:   {}", args.hipfire_quantize_bin.display());
    eprintln!("  collect-hessian-bin:    {}", args.collect_hessian_bin.display());
    if let Some(im) = &args.imatrix { eprintln!("  imatrix:                {}", im.display()); }
    if !args.extra_quantize_args.is_empty() {
        eprintln!("  extra quantize args:    {:?}", args.extra_quantize_args);
    }
    eprintln!();

    create_dir_all(&args.base_output_dir).map_err(|e| format!(
        "mkdir {}: {e}", args.base_output_dir.display()
    ))?;
    if !(0.0..=1.0).contains(&args.damping) {
        return Err(format!("--damping must be in [0, 1], got {}", args.damping));
    }

    let mut prev_scales_for_delta: Vec<(String, Vec<f32>)> = Vec::new();
    // Keyed scales for the damped update. Each entry: ((name, expert_idx), scales).
    let mut prev_scales_for_damp: HashMap<(String, u32), Vec<f32>> = HashMap::new();
    let mut trace: Vec<RoundRecord> = Vec::new();
    let started_all = Instant::now();

    for round_index in 0..args.max_rounds {
        let round_started = Instant::now();
        let round_dir = args.base_output_dir.join(format!("round_{round_index}"));
        create_dir_all(&round_dir).map_err(|e| format!(
            "mkdir {}: {e}", round_dir.display()
        ))?;

        // Step 1: obtain the round Hessian.
        let hessian_path = if round_index == 0 {
            obtain_round_0_hessian(&args, &round_dir)?
        } else {
            // Round 1+: look up a pre-supplied hessian (index = round_index - 1).
            let supply_idx = round_index - 1;
            if supply_idx < args.round_n_hessians.len() {
                let src = &args.round_n_hessians[supply_idx];
                let dst = round_dir.join("hessian.bin");
                if dst.exists() { std::fs::remove_file(&dst).ok(); }
                if std::fs::hard_link(src, &dst).is_err() {
                    std::fs::copy(src, &dst).map_err(|e| format!(
                        "copy {} -> {}: {e}", src.display(), dst.display()
                    ))?;
                }
                dst
            } else {
                eprintln!(
                    "[iterate] round {round_index}: no --round-n-hessian supplied; \
                     stopping. To run round 1+, supply --round-n-hessian per round \
                     (orchestrator does not yet ship a `collect_hessian_quantized` \
                     binary — see scripts/mq4_masked_calib.py for the Python ref)."
                );
                break;
            }
        };

        // Step 2: read the round Hessian, derive raw scales per tensor.
        eprintln!("[iterate] round {round_index}: reading Hessian {}", hessian_path.display());
        let h = HfhsReader::open(&hessian_path)?;
        eprintln!("[iterate] round {round_index}: Hessian has {} tensors", h.entries.len());
        let raw = compute_round_scales(&h, args.awq_alpha);
        eprintln!("[iterate] round {round_index}: derived {} raw scale vectors", raw.len());

        // Step 3: damped update.
        let mut damped: Vec<AwqScaleEntry> = Vec::with_capacity(raw.len());
        let mut flat_for_delta: Vec<(String, Vec<f32>)> = Vec::with_capacity(raw.len());
        for (name, expert_idx, raw_s) in &raw {
            let key = (name.clone(), *expert_idx);
            let prev_for_this = prev_scales_for_damp.get(&key).map(|v| v.as_slice());
            let damped_s = damp_awq_scales(prev_for_this, raw_s, args.damping);
            // For the delta calc we collapse (name, expert_idx) into a single
            // string key — matches the Python convention that treats per-expert
            // tensors as their own key (canonical name carries `.experts.X.`).
            let delta_key = if *expert_idx == 0 {
                name.clone()
            } else {
                format!("{name}#expert={expert_idx}")
            };
            flat_for_delta.push((delta_key, damped_s.clone()));
            damped.push(AwqScaleEntry {
                name: name.clone(),
                expert_idx: *expert_idx,
                scales: damped_s,
            });
        }

        // Step 4: relative-L2 delta vs previous round.
        let scale_delta = relative_l2_delta(&prev_scales_for_delta, &flat_for_delta);
        let converged = round_index > 0 && scale_delta < (args.epsilon as f64);
        eprintln!(
            "[iterate] round {round_index}: relative_l2_delta = {:.6e} \
             (epsilon = {:.6e}; converged = {converged})",
            scale_delta, args.epsilon
        );

        // Step 5: write HFSC scales file.
        let scales_path = round_dir.join("awq_scales.hfsc");
        write_awq_scales(&scales_path, &damped).map_err(|e| format!(
            "write HFSC scales {}: {e}", scales_path.display()
        ))?;
        eprintln!("[iterate] round {round_index}: wrote {} scales to {}",
            damped.len(), scales_path.display());

        // Step 6: invoke hipfire-quantize with the HFSC override + Hessian.
        let model_path = round_dir.join(format!("model.{}.hfq", args.quantize_format));
        let mut cmd = Command::new(&args.hipfire_quantize_bin);
        cmd.arg("--input").arg(&args.hf_model)
           .arg("--output").arg(&model_path)
           .arg("--format").arg(&args.quantize_format)
           .arg("--awq")
           .arg("--awq-alpha").arg(args.awq_alpha.to_string())
           .arg("--awq-scope").arg(&args.awq_scope)
           .arg("--awq-formula").arg(&args.awq_formula)
           .arg("--awq-scales").arg(&scales_path)
           .arg("--gptq").arg(&hessian_path)
           .arg("--gptq-damp").arg(args.gptq_damp.to_string());
        if let Some(im) = &args.imatrix {
            cmd.arg("--imatrix").arg(im);
        } else if args.extra_quantize_args.iter().all(|s| s != "--imatrix") {
            // The --awq path in hipfire-quantize requires --imatrix; without
            // it, the binary will exit early. Surface a clearer error.
            return Err(
                "hipfire-quantize --awq requires --imatrix. Pass --imatrix <path> \
                 to iterate_awq_gptq (forwarded through), or include it in -- <extras>.".to_string()
            );
        }
        for extra in &args.extra_quantize_args {
            cmd.arg(extra);
        }
        eprintln!(
            "[iterate] round {round_index}: invoking hipfire-quantize:\n  {} {}",
            args.hipfire_quantize_bin.display(),
            cmd.get_args().map(|s| s.to_string_lossy().to_string()).collect::<Vec<_>>().join(" ")
        );
        let status = cmd.status().map_err(|e| format!(
            "spawn hipfire-quantize: {e} (binary path: {})",
            args.hipfire_quantize_bin.display()
        ))?;
        if !status.success() {
            return Err(format!(
                "hipfire-quantize exited with status {status} on round {round_index}"
            ));
        }

        let elapsed = round_started.elapsed().as_secs_f64();
        let record = RoundRecord {
            round: round_index,
            hessian_path: hessian_path.clone(),
            scales_path: scales_path.clone(),
            model_path: model_path.clone(),
            n_scales: damped.len(),
            scale_delta_vs_prev: scale_delta,
            converged,
            elapsed_seconds: elapsed,
        };
        // Per-round summary file (lets the orchestrator be resumable / inspectable).
        let summary_path = round_dir.join("summary.json");
        if let Err(e) = File::create(&summary_path).and_then(|mut f| {
            writeln!(f, "{{")?;
            writeln!(f, "  \"round\": {},", record.round)?;
            writeln!(f, "  \"hessian_path\": \"{}\",", record.hessian_path.display())?;
            writeln!(f, "  \"scales_path\": \"{}\",", record.scales_path.display())?;
            writeln!(f, "  \"model_path\": \"{}\",", record.model_path.display())?;
            writeln!(f, "  \"n_scales\": {},", record.n_scales)?;
            writeln!(f, "  \"scale_delta_vs_prev\": {:.10e},", record.scale_delta_vs_prev)?;
            writeln!(f, "  \"converged\": {},", record.converged)?;
            writeln!(f, "  \"elapsed_seconds\": {:.3}", record.elapsed_seconds)?;
            writeln!(f, "}}")
        }) {
            eprintln!("warning: failed to write {}: {e}", summary_path.display());
        }
        trace.push(record);

        // Update state for the next round.
        prev_scales_for_delta = flat_for_delta;
        prev_scales_for_damp.clear();
        for ent in &damped {
            prev_scales_for_damp.insert((ent.name.clone(), ent.expert_idx), ent.scales.clone());
        }

        if converged {
            eprintln!("[iterate] converged at round {round_index} — stopping early");
            break;
        }
    }

    let total = started_all.elapsed().as_secs_f64();
    let summary_path = args.base_output_dir.join("iterate-summary.json");
    if let Err(e) = write_summary_json(&summary_path, &args, &trace, total) {
        eprintln!("warning: failed to write summary {}: {e}", summary_path.display());
    } else {
        eprintln!("[iterate] wrote summary to {}", summary_path.display());
    }
    eprintln!("\nround\tscale_delta\tconverged\telapsed_s\tmodel");
    for r in &trace {
        eprintln!(
            "{}\t{:.6e}\t{}\t{:.2}\t{}",
            r.round, r.scale_delta_vs_prev, r.converged, r.elapsed_seconds, r.model_path.display()
        );
    }
    eprintln!("\n[iterate] total elapsed = {:.2}s ({} rounds executed)", total, trace.len());
    Ok(())
}

fn main() {
    if let Err(msg) = run() {
        eprintln!("iterate_awq_gptq: ERROR: {msg}");
        std::process::exit(1);
    }
}
