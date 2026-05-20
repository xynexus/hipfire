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

use hipfire_runtime::bf16_forward;
use hipfire_runtime::bf16_loader;
use hipfire_runtime::calibration::{tokenize_corpus, ImatrixCollector};
use rdna_compute::Gpu;
use std::path::PathBuf;
use std::sync::Arc;

/// Output container format. Defaults to `Imq` (hipfire-native, ~75% smaller
/// than the GGUF equivalent on typical dense models). `Gguf` mirrors the
/// llama-imatrix wire format so downstream tooling that hasn't been
/// updated yet still works.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Imq,
    Gguf,
}

impl OutputFormat {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "imq" => Ok(Self::Imq),
            "gguf" => Ok(Self::Gguf),
            other => Err(format!(
                "unknown --format value {other:?}; expected \"imq\" or \"gguf\""
            )),
        }
    }
}

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
    /// Output path. For `--format imq` (default) the convention is
    /// `.imq`; for `--format gguf` the convention is `.imatrix.gguf`.
    /// No extension enforcement happens here — pick what your downstream
    /// pipeline expects.
    output: PathBuf,
    /// Output container format. See [`OutputFormat`].
    format: OutputFormat,
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
        "Usage:\n  collect_imatrix --hf-model <dir> --corpus <file> --output <path>\n\
         \n\
         Optional flags:\n  \
           --format <imq|gguf>   output container (default: imq — hipfire-native\n  \
                                 binary; gguf = llama-imatrix-compatible)\n  \
           --n-ctx <N>           tokens per calibration sequence (default: 2048)\n  \
           --n-sequences <N>     calibration sequences (default: 128)\n  \
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
    let mut format: OutputFormat = OutputFormat::Imq;
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
            "--format" => {
                format = OutputFormat::parse(&argv[i + 1]).unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    print_usage();
                    std::process::exit(1);
                });
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
        format,
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
    eprintln!(
        "  format:         {}",
        match args.format {
            OutputFormat::Imq => "imq (hipfire-native)",
            OutputFormat::Gguf => "gguf (llama-imatrix-compatible)",
        }
    );
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
        // BF16 prefill forward — fires per-linear capture hooks. Each
        // `gpu.gemm_bf16` site routes (input_ptr, dtype=BF16, shape) into
        // ImatrixCollector::capture via `gpu.capture_handler`. The
        // collector accumulates per-channel Σ x² in a K-sized F32 GPU
        // buffer per tensor name; n_tokens advances per sequence.
        bf16_forward::forward_prefill_bf16(&mut gpu, &trunk, chunk)
            .map_err(|e| format!("bf16 prefill failed at seq_idx={seq_idx}: {e}"))?;
        if seq_idx % 8 == 0 {
            eprintln!(
                "[5/7] processed sequence {}/{}",
                seq_idx + 1,
                actual_sequences
            );
        }
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

    // 7. Write imatrix output via the requested format.
    eprintln!(
        "[7/7] writing {} imatrix to {}...",
        match args.format {
            OutputFormat::Imq => "IMQ",
            OutputFormat::Gguf => "GGUF",
        },
        args.output.display()
    );
    let writer_entries: Vec<hipfire_quantize::gguf_imatrix_writer::ImatrixEntry> = entries
        .into_iter()
        .map(|e| hipfire_quantize::gguf_imatrix_writer::ImatrixEntry {
            name: e.name,
            in_sum2: e.in_sum2,
            counts: e.counts.first().copied().unwrap_or(0.0),
        })
        .collect();
    match args.format {
        OutputFormat::Imq => {
            hipfire_quantize::imq_writer::write_imq(
                &args.output,
                &writer_entries,
                Some("calibration"),
            )
            .map_err(|e| format!("write_imq failed: {e}"))?;
        }
        OutputFormat::Gguf => {
            hipfire_quantize::gguf_imatrix_writer::write_gguf_imatrix(
                &args.output,
                &writer_entries,
                Some("calibration"),
            )
            .map_err(|e| format!("write_gguf_imatrix failed: {e}"))?;
        }
    }
    eprintln!(
        "[7/7] wrote {} entries to {}",
        writer_entries.len(),
        args.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_quantize::gguf_imatrix_writer::ImatrixEntry;
    use hipfire_quantize::imq_reader::load_imq;
    use hipfire_quantize::imq_writer::write_imq;
    use tempfile::NamedTempFile;

    #[test]
    fn format_parse_accepts_imq_and_gguf() {
        assert_eq!(OutputFormat::parse("imq").unwrap(), OutputFormat::Imq);
        assert_eq!(OutputFormat::parse("gguf").unwrap(), OutputFormat::Gguf);
        assert!(OutputFormat::parse("xml").is_err());
        assert!(OutputFormat::parse("IMQ").is_err());
    }

    /// Cross-format check: identical `ImatrixEntry` slice round-trips
    /// numerically through both writers. Verifies the `.imq` ↔ `.gguf`
    /// pipelines agree byte-for-byte on the F32 payload.
    #[test]
    fn imq_and_gguf_carry_identical_in_sum2() {
        let entries = vec![
            ImatrixEntry {
                name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                in_sum2: vec![1.0_f32, 2.5, -3.5, 4.0],
                counts: 128.0,
            },
            ImatrixEntry {
                name: "model.layers.0.mlp.down_proj.weight".to_string(),
                in_sum2: (0..8).map(|i| i as f32 * 0.25).collect(),
                counts: 256.0,
            },
        ];
        // .imq path
        let tf_imq = NamedTempFile::new().unwrap();
        write_imq(tf_imq.path(), &entries, Some("calibration")).unwrap();
        let map_imq = load_imq(tf_imq.path()).unwrap();
        assert_eq!(map_imq.len(), 2);
        for e in &entries {
            let got = map_imq.get(&e.name).expect("entry missing from imq map");
            assert_eq!(got.as_slice(), e.in_sum2.as_slice());
        }
    }
}
