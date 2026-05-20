//! `collect_hessian` — Tier 1 hipfire-native Hessian collector for GPTQ.
//!
//! Tier 1 driver for the Hessian-collection pipeline that replaces the
//! Tier 2 PyTorch wrapper at `scripts/collect_hessian.py`.
//!
//! Pipeline:
//!
//! 1. Initialize the GPU (HIP runtime + kernel cache).
//! 2. Load a BF16 HuggingFace model directly from `<dir>/*.safetensors`
//!    via `bf16_loader::load_bf16_model`.
//! 3. Install a `HessianCollector` on `gpu.capture_handler`. The
//!    collector accumulates a per-tensor `H = Σ x · xᵀ` K×K F32 outer
//!    product via subagent A's on-GPU rank-1-update kernel
//!    (`gpu.hessian_outer_product_bf16`). Only tensors matching the
//!    GPTQ target whitelist (`bf16_loader::is_gptq_target`) are
//!    accumulated — saves K×K F32 for norms / embed / lm_head.
//! 4. Tokenize the calibration corpus via `tokenize_corpus()` (uses the
//!    HF `tokenizer.json` for parity with the production hipfire path).
//! 5. Loop over `n_sequences` chunks of `ctx_len` tokens each (optionally
//!    multiple passes); run the BF16 prefill forward pass through
//!    subagent C's `forward_prefill_bf16` so the linear-layer dispatch
//!    sites fire the capture hook.
//! 6. Drain the collector to `Vec<HessianEntry>`.
//! 7. Write the HFHS-v1 binary via subagent B's `hfhs_writer`,
//!    byte-compatible with `scripts/collect_hessian.py` and consumed
//!    by `crates/hipfire-quantize/src/hessian_io.rs`.
//!
//! Target speedup vs Tier 2: 20× (~8h → ~25min for a 27B-class model on
//! MI300x). The Python path is currently bottlenecked on HF transformers
//! eager forward + CPU-side `x.T @ x` accumulator with the BF16→FP32
//! cast + per-token `.cpu()` PCIe transfer in
//! `scripts/collect_hessian.py:213-222`.
//!
//! Tier 1 keeps the outer-product accumulator on-GPU for the full
//! calibration pass; final HFHS dump is the only host-side step.
//!
//! Usage:
//!
//! ```ignore
//! cargo run --release -p hipfire-runtime --bin collect_hessian -- \
//!     --hf-model    <path-to-bf16-hf-model-dir> \
//!     --corpus      <path-to-corpus.txt-or-hf-dataset-id> \
//!     --output      <path-to-out.hessian.bin> \
//!     [--n-sequences 128] [--ctx-len 2048] [--n-passes 1]
//! ```

use hipfire_runtime::bf16_forward;
use hipfire_runtime::bf16_loader;
use hipfire_runtime::calibration::{tokenize_corpus, HessianCollector};
use rdna_compute::Gpu;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug)]
#[allow(dead_code)]
struct Args {
    /// HuggingFace BF16 model dir (`*.safetensors` + `config.json`).
    hf_model: PathBuf,
    /// Plain-text corpus file OR HuggingFace dataset id
    /// (e.g. `wikitext-2-raw-v1`). The Tier 2 Python path defaults to
    /// `wikitext`; we mirror that once corpus-loading lands.
    corpus: PathBuf,
    /// Output HFHS-v1 binary path (`*.hessian.bin` by convention).
    /// Consumed by `crates/hipfire-quantize/src/hessian_io.rs` at
    /// GPTQ quantize-time.
    output: PathBuf,
    /// Number of calibration sequences (GPTQ paper default: 128).
    n_sequences: usize,
    /// Tokens per calibration sequence (GPTQ paper default: 2048).
    ctx_len: usize,
    /// Number of full passes over the calibration corpus. Default 1.
    /// Multiple passes are useful for noisy GPTQ convergence on MoE
    /// models with sparse expert activations.
    n_passes: usize,
}

fn print_usage() {
    eprintln!(
        "Usage:\n  collect_hessian --hf-model <dir> --corpus <file-or-hf-id> --output <bin>\n\
         \n\
         Optional flags:\n\
           --n-sequences <N>     calibration sequences (default: 128)\n\
           --ctx-len <N>         tokens per calibration sequence (default: 2048)\n\
           --n-passes <N>        passes over the corpus (default: 1)\n\
         \n\
         Tier 1 hipfire-native Hessian collector. See\n\
         docs/investigations/2026-05-19-tier1-bf16-mfma/ for the foundation POC.\n\
         Output format: HFHS v1 (see scripts/collect_hessian.py:25-43)."
    );
}

fn parse_args() -> Args {
    let mut hf_model: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut n_sequences: usize = 128;
    let mut ctx_len: usize = 2048;
    let mut n_passes: usize = 1;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--hf-model" => {
                hf_model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--corpus" => {
                corpus = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--n-sequences" => {
                n_sequences = argv[i + 1]
                    .parse()
                    .expect("--n-sequences must be a positive integer");
                i += 2;
            }
            "--ctx-len" => {
                ctx_len = argv[i + 1]
                    .parse()
                    .expect("--ctx-len must be a positive integer");
                i += 2;
            }
            "--n-passes" => {
                n_passes = argv[i + 1]
                    .parse()
                    .expect("--n-passes must be a positive integer");
                i += 2;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                print_usage();
                std::process::exit(1);
            }
        }
    }

    let hf_model = hf_model.unwrap_or_else(|| {
        eprintln!("error: --hf-model is required");
        print_usage();
        std::process::exit(1);
    });
    let corpus = corpus.unwrap_or_else(|| {
        eprintln!("error: --corpus is required");
        print_usage();
        std::process::exit(1);
    });
    let output = output.unwrap_or_else(|| {
        eprintln!("error: --output is required");
        print_usage();
        std::process::exit(1);
    });

    Args {
        hf_model,
        corpus,
        output,
        n_sequences,
        ctx_len,
        n_passes,
    }
}

fn main() {
    let args = parse_args();
    eprintln!("collect_hessian (Tier 1)");
    eprintln!("  hf-model:    {}", args.hf_model.display());
    eprintln!("  corpus:      {}", args.corpus.display());
    eprintln!("  output:      {}", args.output.display());
    eprintln!("  n-sequences: {}", args.n_sequences);
    eprintln!("  ctx-len:     {}", args.ctx_len);
    eprintln!("  n-passes:    {}", args.n_passes);
    eprintln!();

    if let Err(msg) = run(&args) {
        eprintln!("collect_hessian: ERROR: {msg}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    // 1. GPU init.
    eprintln!("[1/7] initializing GPU...");
    let mut gpu = Gpu::init().map_err(|e| format!("Gpu::init failed: {e}"))?;

    // 2. BF16 model load (subagent D's load_bf16_model — currently
    //    unimplemented!(); the binary stops here with a clear panic
    //    message at the safetensors-parse TODO until D lands).
    eprintln!("[2/7] loading BF16 model from {}...", args.hf_model.display());
    let trunk = bf16_loader::load_bf16_model(&mut gpu, &args.hf_model)
        .map_err(|e| format!("bf16_loader::load_bf16_model failed: {e}"))?;
    eprintln!(
        "[2/7] loaded trunk: {} tensors, model_type={}, {} bytes",
        trunk.tensors.len(),
        trunk.model_type,
        trunk.total_bytes
    );

    // 3. Install the Hessian capture handler.
    eprintln!("[3/7] installing HessianCollector capture handler...");
    let collector = Arc::new(HessianCollector::new());
    gpu.capture_handler = Some(collector.clone());

    // 4. Tokenize corpus.
    eprintln!("[4/7] tokenizing corpus...");
    let corpus_text = std::fs::read_to_string(&args.corpus)
        .map_err(|e| format!("failed to read corpus at {}: {e}", args.corpus.display()))?;
    let tokens = tokenize_corpus(&args.hf_model, &corpus_text)?;
    eprintln!("[4/7] tokenized: {} tokens", tokens.len());

    // 5. Loop over N passes × N sequences of ctx_len tokens each.
    if tokens.len() < args.ctx_len {
        return Err(format!(
            "corpus has {} tokens but ctx_len={}; need at least one full sequence",
            tokens.len(),
            args.ctx_len
        ));
    }
    let actual_sequences = std::cmp::min(args.n_sequences, tokens.len() / args.ctx_len);
    if actual_sequences < args.n_sequences {
        eprintln!(
            "[5/7] WARNING: corpus only supplies {} sequences (requested {}); proceeding with {}",
            actual_sequences, args.n_sequences, actual_sequences
        );
    }
    eprintln!(
        "[5/7] running BF16 forward over {} passes × {} sequences × {} tokens \
         = {} total tokens...",
        args.n_passes,
        actual_sequences,
        args.ctx_len,
        args.n_passes * actual_sequences * args.ctx_len
    );
    for pass in 0..args.n_passes {
        for seq_idx in 0..actual_sequences {
            let start = seq_idx * args.ctx_len;
            let end = start + args.ctx_len;
            let chunk = &tokens[start..end];
            // BF16 prefill forward — fires per-linear capture hooks. Each
            // `gpu.gemm_bf16` site routes (input_ptr, dtype=BF16, shape)
            // into HessianCollector::capture via `gpu.capture_handler`.
            // The collector accumulates `Σ x · xᵀ` in a K×K F32 GPU
            // buffer per tensor name (only for `is_gptq_target` tensors).
            bf16_forward::forward_prefill_bf16(&mut gpu, &trunk, chunk)
                .map_err(|e| format!(
                    "bf16 prefill failed at pass={pass} seq_idx={seq_idx}: {e}"
                ))?;
            if seq_idx % 8 == 0 {
                eprintln!(
                    "[5/7] pass {}/{}, sequence {}/{}",
                    pass + 1,
                    args.n_passes,
                    seq_idx + 1,
                    actual_sequences
                );
            }
        }
    }

    // 6. Drain collector.
    eprintln!("[6/7] draining HessianCollector...");
    gpu.capture_handler = None;
    let entries = match Arc::try_unwrap(collector) {
        Ok(c) => c.drain(&gpu)?,
        Err(arc) => arc.drain(&gpu)?,
    };
    eprintln!("[6/7] drained {} Hessian entries", entries.len());

    // 7. Write HFHS-v1 output via subagent B's writer.
    eprintln!("[7/7] writing HFHS-v1 binary to {}...", args.output.display());
    // TODO(subagent-B): wire `hfhs_writer::write_hfhs(&args.output, &entries)`
    // HFHS-v1 byte-layout per `scripts/collect_hessian.py:25-43`:
    //
    //   magic "HFHS" + u32 version=1 + u32 n_tensors
    //   per-tensor: u32 name_len + name + u32 k + u64 n_tokens + k*k F32 row-major
    //
    // The Tier 2 Python path writes this format unchanged; subagent-B's
    // Rust writer must produce byte-identical output so
    // `crates/hipfire-quantize/src/hessian_io.rs` reads it without changes.
    let _ = &entries;
    Err(format!(
        "[7/7] HFHS writer not yet wired — subagent-B owns \
         `hfhs_writer::write_hfhs(&output, &entries)`. \
         Drained {} entries are ready to write.",
        entries.len()
    ))
}
