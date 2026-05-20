//! `collect_imatrix` — Tier 1 hipfire-native activation-magnitude collector.
//!
//! Tier 1 driver for the imatrix-collection pipeline that replaces the
//! Tier 2 subprocess wrapper at `examples/imatrix_collect.rs` (which
//! shells out to `llama-imatrix`).
//!
//! Pipeline:
//!
//! 1. Initialize the GPU (HIP runtime + kernel cache).
//! 2. Load a BF16 HuggingFace model directly from `<dir>/*.safetensors`
//!    via `bf16_loader::load_bf16_model`.
//! 3. Install an `ImatrixCollector` on `gpu.capture_handler`. The
//!    collector accumulates per-channel `Σ act²` per linear-layer input
//!    via subagent A's on-GPU reduction kernel
//!    (`gpu.sumsq_reduce_bf16`).
//! 4. Tokenize the calibration corpus via `tokenize_corpus()` (uses the
//!    HF `tokenizer.json` for parity with the production hipfire path).
//! 5. Loop over `n_sequences` chunks of `n_ctx` tokens each; run the
//!    BF16 prefill forward pass through subagent C's `forward_prefill_bf16`
//!    so the linear-layer dispatch sites fire the capture hook.
//! 6. Drain the collector to `Vec<ImatrixEntry>`.
//! 7. Write the GGUF imatrix via subagent B's `gguf_imatrix_writer`,
//!    byte-compatible with `llama-imatrix --output-format gguf`.
//!
//! Target speedup vs Tier 2: 20× (~8h → ~25min for a 27B-class model on
//! MI300x). See `docs/investigations/2026-05-19-tier1-bf16-mfma/README.md`
//! for the foundation-POC validating the BF16 MFMA GEMM building block.
//!
//! Usage:
//!
//! ```ignore
//! cargo run --release -p hipfire-runtime --bin collect_imatrix -- \
//!     --hf-model    <path-to-bf16-hf-model-dir> \
//!     --corpus      <path-to-calibration-corpus.txt> \
//!     --output      <path-to-output.imatrix.gguf> \
//!     [--n-ctx 2048] [--n-sequences 128] [--process-output]
//! ```
//!
//! TODO: replace this stdlib-only arg parser with `clap` once the
//! crate accepts a clap workspace-dep. Matched the imatrix_collect.rs
//! example style for now to keep the workspace dep graph clean.

use hipfire_runtime::bf16_loader;
use hipfire_runtime::calibration::{tokenize_corpus, ImatrixCollector};
use rdna_compute::Gpu;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug)]
#[allow(dead_code)]
struct Args {
    /// Path to a HuggingFace model directory containing
    /// `model.safetensors[.index.json]` + `config.json` + `tokenizer.json`.
    hf_model: PathBuf,
    /// Plain-text calibration corpus (one document or one file = one
    /// concatenated sequence; the binary chunks into `n_sequences ×
    /// n_ctx` tokens internally). Tier 2 used `wikitext-2-raw-v1` by
    /// default; Tier 1 will mirror that once corpus-loading lands.
    corpus: PathBuf,
    /// Output GGUF path (must end in `.imatrix.gguf` by convention).
    output: PathBuf,
    /// Tokens per calibration sequence.
    n_ctx: usize,
    /// Calibration sequences (matches GPTQ paper's 128-seq scale).
    n_sequences: usize,
    /// Also collect data for the `output` / `lm_head` tensor. Mirrors
    /// llama-imatrix's `--process-output` flag.
    process_output: bool,
}

fn print_usage() {
    eprintln!(
        "Usage:\n  collect_imatrix --hf-model <dir> --corpus <file> --output <gguf>\n\
         \n\
         Optional flags:\n\
           --n-ctx <N>           tokens per calibration sequence (default: 2048)\n\
           --n-sequences <N>     calibration sequences (default: 128)\n\
           --process-output      also collect data for lm_head / output tensor\n\
         \n\
         Tier 1 hipfire-native imatrix collector. See\n\
         docs/investigations/2026-05-19-tier1-bf16-mfma/ for the foundation POC."
    );
}

fn parse_args() -> Args {
    let mut hf_model: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut n_ctx: usize = 2048;
    let mut n_sequences: usize = 128;
    let mut process_output = false;

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
            "--n-ctx" => {
                n_ctx = argv[i + 1].parse().expect("--n-ctx must be a positive integer");
                i += 2;
            }
            "--n-sequences" => {
                n_sequences = argv[i + 1]
                    .parse()
                    .expect("--n-sequences must be a positive integer");
                i += 2;
            }
            "--process-output" => {
                process_output = true;
                i += 1;
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
        n_ctx,
        n_sequences,
        process_output,
    }
}

fn main() {
    let args = parse_args();
    eprintln!("collect_imatrix (Tier 1)");
    eprintln!("  hf-model:       {}", args.hf_model.display());
    eprintln!("  corpus:         {}", args.corpus.display());
    eprintln!("  output:         {}", args.output.display());
    eprintln!("  n-ctx:          {}", args.n_ctx);
    eprintln!("  n-sequences:    {}", args.n_sequences);
    eprintln!("  process-output: {}", args.process_output);
    eprintln!();

    if let Err(msg) = run(&args) {
        eprintln!("collect_imatrix: ERROR: {msg}");
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

    // 3. Install the imatrix capture handler.
    eprintln!("[3/7] installing ImatrixCollector capture handler...");
    let collector = Arc::new(ImatrixCollector::new(args.process_output));
    gpu.capture_handler = Some(collector.clone());

    // 4. Tokenize corpus.
    eprintln!("[4/7] tokenizing corpus...");
    let corpus_text = std::fs::read_to_string(&args.corpus)
        .map_err(|e| format!("failed to read corpus at {}: {e}", args.corpus.display()))?;
    let tokens = tokenize_corpus(&args.hf_model, &corpus_text)?;
    eprintln!("[4/7] tokenized: {} tokens", tokens.len());

    // 5. Loop over N sequences of ctx_len tokens each.
    let total_needed = args.n_sequences * args.n_ctx;
    if tokens.len() < args.n_ctx {
        return Err(format!(
            "corpus has {} tokens but n_ctx={}; need at least one full sequence",
            tokens.len(),
            args.n_ctx
        ));
    }
    let actual_sequences = std::cmp::min(args.n_sequences, tokens.len() / args.n_ctx);
    if actual_sequences < args.n_sequences {
        eprintln!(
            "[5/7] WARNING: corpus only supplies {} sequences (requested {}); proceeding with {}",
            actual_sequences, args.n_sequences, actual_sequences
        );
    }
    eprintln!(
        "[5/7] running BF16 forward over {} sequences × {} tokens = {} total tokens...",
        actual_sequences, args.n_ctx, actual_sequences * args.n_ctx
    );
    let _ = total_needed;
    for seq_idx in 0..actual_sequences {
        let start = seq_idx * args.n_ctx;
        let end = start + args.n_ctx;
        let chunk = &tokens[start..end];
        let _ = chunk;
        // TODO(subagent-C): wire `forward_prefill_bf16(&mut gpu, &trunk, chunk)`
        // here. Reference shape (per orchestrator dispatch):
        //
        //     bf16_forward::forward_prefill_bf16(&mut gpu, &trunk, chunk)
        //         .map_err(|e| format!("bf16 prefill failed: {e}"))?;
        //
        // The forward pass dispatches one linear-layer GEMM at a time; each
        // GEMM site fires `gpu.capture_handler.as_ref().map(|h| h.capture(...))`,
        // which routes into ImatrixCollector::capture. The collector
        // accumulates per-channel Σ x² in a K-sized F32 GPU buffer per
        // tensor name. n_tokens is incremented per call.
        return Err(format!(
            "[5/7] forward pass not yet wired — subagent-C owns \
             `bf16_forward::forward_prefill_bf16(&mut gpu, &trunk, chunk)`. \
             Until then, this binary stops here at seq_idx={} (out of {}).",
            seq_idx, actual_sequences
        ));
    }

    // 6. Drain collector.
    eprintln!("[6/7] draining ImatrixCollector...");
    // The Arc::try_unwrap path lets us reclaim sole ownership before
    // drain so the caller doesn't see a poisoned mutex if the
    // collector's clone outlives the forward loop.
    // For the scaffold flow the above loop's early-return prevents us
    // reaching here; once subagent-C wires the forward pass, the
    // returned Arc clone count is 2 (collector + gpu.capture_handler).
    // We need to clear the gpu handler first so try_unwrap succeeds.
    gpu.capture_handler = None;
    let entries = match Arc::try_unwrap(collector) {
        Ok(c) => c.drain(&gpu)?,
        Err(arc) => arc.drain(&gpu)?,
    };
    eprintln!("[6/7] drained {} imatrix entries", entries.len());

    // 7. Write GGUF imatrix output via subagent B's writer.
    eprintln!("[7/7] writing GGUF imatrix to {}...", args.output.display());
    // TODO(subagent-B): wire `gguf_imatrix_writer::write_gguf_imatrix(&args.output,
    //                                                                 &entries,
    //                                                                 Some("calibration"))`
    // The dataset name "calibration" matches the convention used by
    // llama-imatrix --output-format gguf (it embeds the source file name;
    // here we use the generic "calibration" so downstream tooling that
    // keys on dataset != "wikitext" doesn't false-trigger).
    let _ = &entries;
    Err(format!(
        "[7/7] GGUF writer not yet wired — subagent-B owns \
         `gguf_imatrix_writer::write_gguf_imatrix(&output, &entries, Some(\"calibration\"))`. \
         Drained {} entries are ready to write.",
        entries.len()
    ))
}
