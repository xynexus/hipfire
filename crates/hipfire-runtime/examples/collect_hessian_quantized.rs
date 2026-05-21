//! collect_hessian_quantized — Tier 1 iterative AWQ-aware-GPTQ v3 Hessian
//! collector.
//!
//! Sister to `src/bin/collect_hessian.rs`. The BF16 bin loads a HuggingFace
//! safetensors trunk and runs `bf16_forward::forward_prefill_bf16` to
//! collect a "clean" Hessian from the un-quantized model — used by v3
//! round 0 (initial AWQ scale + GPTQ quantize).
//!
//! This example loads a `.hfq` (the QUANTIZED model produced by round 0
//! or later) and runs the production prefill (`qwen35::forward_prefill_batch`)
//! to collect a Hessian under quantization noise. v3 rounds 1+ feed this
//! quantized-model Hessian back into the scale-recompute pass so the
//! GPTQ inverse-Hessian solve reflects the quant errors it's correcting,
//! not just the BF16 distribution.
//!
//! Pipeline (mirrors `collect_hessian.rs` 1-for-1 except for steps 2 + 5):
//!
//! 1. Initialize the GPU (HIP runtime + kernel cache).
//! 2. Open the `.hfq`, read its embedded `Qwen35Config`, and call
//!    `qwen35::load_weights` to upload the quantized weights to GPU.
//! 3. Install a `HessianCollector` on `gpu.capture_handler`. The
//!    collector accumulates K×K F32 outer products per GPTQ-target
//!    tensor. Production prefill emits F32 activations; the collector
//!    casts to BF16 via RTNE `convert_f32_to_bf16` into a lazy scratch,
//!    then dispatches the gfx942 MFMA `hessian_outer_product_bf16`
//!    kernel — same kernel + same on-disk format the BF16 bin uses.
//! 4. Tokenize the calibration corpus. Default route: subprocess
//!    `llama-tokenize` for byte-parity with PyTorch oracle; fallback:
//!    hipfire's native BPE encoder.
//! 5. Loop over `n_passes × n_sequences` chunks of `ctx_len` tokens
//!    each. Per-sequence: reset the DeltaNet recurrent state, then
//!    iterate `qwen35::forward_scratch(..., token, pos, ...)` per
//!    token in the chunk. Per-token `forward_scratch` calls
//!    `forward_from_x_gpu`, which has `set_capture_name(...)` wired
//!    at every linear projection (DeltaNet `in_proj_*`,
//!    FullAttention `q/k/v/o_proj`, MLP `gate/up/down_proj`,
//!    `lm_head`). The batched `forward_prefill_batch` path routes
//!    through fused QKV / fused gate_up / FA3 kernels that skip
//!    `set_capture_name`, so the capture handler never fires and
//!    the HFHS sidecar ends up empty (0 entries). This bin trades
//!    prefill throughput for capture coverage — much slower
//!    (~5–10 ms/token on 0.8B, so ~30–60 min at 128×2048) but
//!    correct.
//! 6. Drain the collector to `Vec<HessianEntry>`.
//! 7. Normalize each tensor's accumulator by its `n_tokens` count and
//!    write the HFHS-v1 binary (`hipfire_quantize::hfhs_writer::write_hfhs`)
//!    — same on-disk format the BF16 bin emits, so the downstream
//!    iterative-GPTQ pass consumes it via the existing reader without
//!    any flag.
//!
//! Caveat: `gpu.hessian_outer_product_bf16` is a gfx942-only MFMA
//! kernel. On other archs (gfx11xx etc.) the kernel call will surface
//! an "unsupported arch" error at runtime — the example STILL COMPILES
//! cross-arch (it's just calibration territory; never on the hot path).
//!
//! Usage:
//!
//! ```ignore
//! cargo run --release -p hipfire-runtime --example collect_hessian_quantized \
//!   --features arch-qwen35,deltanet -- \
//!     --model       <path-to-model.hfq> \
//!     --corpus      <path-to-corpus.txt> \
//!     --output      <path-to-out.hessian.bin> \
//!     [--tokenizer llama-cpp --bf16-gguf <path>] \
//!     [--n-sequences 128] [--ctx-len 2048] [--n-passes 1] [--kv-mode q8]
//! ```

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet,arch-qwen35");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::calibration::{
        tokenize_corpus, tokenize_corpus_via_llama_tokenize, HessianCollector,
    };
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::KvCache;
    use rdna_compute::Gpu;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    // -----------------------------------------------------------------
    // Args
    // -----------------------------------------------------------------

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TokenizerKind {
        Hipfire,
        LlamaCpp,
    }
    impl TokenizerKind {
        fn parse(s: &str) -> Result<Self, String> {
            match s {
                "hipfire" => Ok(Self::Hipfire),
                "llama-cpp" | "llama.cpp" | "llamacpp" => Ok(Self::LlamaCpp),
                other => Err(format!(
                    "unknown --tokenizer value {other:?}; expected \"hipfire\" or \"llama-cpp\""
                )),
            }
        }
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    struct Args {
        /// `.hfq` quantized model path (consumed by `HfqFile::open`).
        model: PathBuf,
        /// Plain-text calibration corpus (one stream of text — same shape
        /// the BF16 `collect_hessian` bin consumes).
        corpus: PathBuf,
        /// Output HFHS-v1 binary path. Same on-disk format the BF16 bin
        /// emits; consumed by `hipfire-quantize::hessian_io` reader.
        output: PathBuf,
        /// Number of calibration sequences (GPTQ paper default: 128).
        n_sequences: usize,
        /// Tokens per calibration sequence (GPTQ paper default: 2048).
        ctx_len: usize,
        /// Number of full passes over the calibration corpus. Default 1.
        n_passes: usize,
        /// KV cache mode. Default q8 (per project memory:
        /// `feedback_kv_mode_ordering` — q8 strictly better than asym3
        /// on all measured workloads).
        kv_mode: String,
        /// Tokenizer source. Default `LlamaCpp` (byte-parity with the
        /// PyTorch oracle); `Hipfire` is the native BPE fallback.
        tokenizer: TokenizerKind,
        /// Path to the `llama-tokenize` binary. Used when
        /// `tokenizer == LlamaCpp`.
        llama_tokenize_bin: PathBuf,
        /// BF16 GGUF carrying the tokenizer metadata. **Required** when
        /// `tokenizer == LlamaCpp` (llama-tokenize loads tokenizer.json-
        /// equivalent metadata out of GGUF, not out of the `.hfq`).
        bf16_gguf: Option<PathBuf>,
    }

    fn print_usage() {
        eprintln!(
            "Usage:\n  collect_hessian_quantized --model <path.hfq> --corpus <file> --output <bin> \\\n   \
                                  [--tokenizer llama-cpp --bf16-gguf <path>]\n\
             \n\
             Optional flags:\n  \
               --n-sequences <N>             calibration sequences (default: 128)\n  \
               --ctx-len <N>                 tokens per calibration sequence (default: 2048)\n  \
               --n-passes <N>                passes over the corpus (default: 1)\n  \
               --kv-mode <q8|asym2|asym3|asym4>\n  \
                                             KV cache mode (default: q8 — strictly better\n  \
                                             than asym3 on all measured workloads per\n  \
                                             feedback_kv_mode_ordering)\n  \
               --tokenizer <hipfire|llama-cpp>\n  \
                                             tokenizer source (default: llama-cpp — byte-parity\n  \
                                             with the PyTorch oracle; hipfire = native BPE fallback)\n  \
               --llama-tokenize-bin <path>   path to the llama-tokenize binary\n  \
                                             (default: /workspace/llama.cpp/build/bin/llama-tokenize)\n  \
               --bf16-gguf <path>            BF16 GGUF carrying the tokenizer metadata\n  \
                                             (REQUIRED when --tokenizer llama-cpp)\n\
             \n\
             Tier 1 hipfire-native Hessian collector for the iterative AWQ-aware-GPTQ\n\
             v3 algorithm. Sister to src/bin/collect_hessian.rs (BF16 path).\n\
             Output format: HFHS v1 (same as the BF16 bin)."
        );
    }

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut n_sequences: usize = 128;
    let mut ctx_len: usize = 2048;
    let mut n_passes: usize = 1;
    let mut kv_mode: String = "q8".to_string();
    let mut tokenizer: TokenizerKind = TokenizerKind::LlamaCpp;
    let mut llama_tokenize_bin: PathBuf =
        PathBuf::from("/workspace/llama.cpp/build/bin/llama-tokenize");
    let mut bf16_gguf: Option<PathBuf> = None;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
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
            "--kv-mode" => {
                let v = argv[i + 1].clone();
                if !matches!(v.as_str(), "q8" | "asym2" | "asym3" | "asym4") {
                    eprintln!(
                        "--kv-mode must be one of: q8 asym2 asym3 asym4 (got {v})"
                    );
                    std::process::exit(1);
                }
                kv_mode = v;
                i += 2;
            }
            "--tokenizer" => {
                tokenizer = TokenizerKind::parse(&argv[i + 1]).unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    print_usage();
                    std::process::exit(1);
                });
                i += 2;
            }
            "--llama-tokenize-bin" => {
                llama_tokenize_bin = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--bf16-gguf" => {
                bf16_gguf = Some(PathBuf::from(&argv[i + 1]));
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

    let model = model.unwrap_or_else(|| {
        eprintln!("error: --model is required");
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
    if tokenizer == TokenizerKind::LlamaCpp && bf16_gguf.is_none() {
        eprintln!(
            "error: --bf16-gguf is required when --tokenizer llama-cpp (the default).\n\
             Pass either:\n  \
               --bf16-gguf <path-to-bf16.gguf>            (recommended — byte-parity with PyTorch oracle)\n  \
             or:\n  \
               --tokenizer hipfire                        (fallback to hipfire native BPE)"
        );
        print_usage();
        std::process::exit(1);
    }

    let args = Args {
        model,
        corpus,
        output,
        n_sequences,
        ctx_len,
        n_passes,
        kv_mode,
        tokenizer,
        llama_tokenize_bin,
        bf16_gguf,
    };

    // -----------------------------------------------------------------
    // Eval-mode env vars (must precede Gpu::init).
    // Force prompt-normalize OFF (we feed token IDs, never raw text into
    // the engine's tokenizer), graph-capture OFF (we're not in a
    // perf-critical loop and capture would interact with the per-call
    // capture_handler), and pin HIPFIRE_KV_MODE so downstream forward
    // dispatchers select the right cache layout.
    // SAFETY: single-threaded init phase; no other threads observing env.
    // -----------------------------------------------------------------
    unsafe {
        std::env::set_var("HIPFIRE_NORMALIZE_PROMPT", "0");
        std::env::set_var("HIPFIRE_GRAPH", "0");
        std::env::set_var("HIPFIRE_KV_MODE", &args.kv_mode);
    }

    eprintln!("collect_hessian_quantized (Tier 1, .hfq path)");
    eprintln!("  model:          {}", args.model.display());
    eprintln!("  corpus:         {}", args.corpus.display());
    eprintln!("  output:         {}", args.output.display());
    eprintln!("  n-sequences:    {}", args.n_sequences);
    eprintln!("  ctx-len:        {}", args.ctx_len);
    eprintln!("  n-passes:       {}", args.n_passes);
    eprintln!("  kv-mode:        {}", args.kv_mode);
    eprintln!(
        "  tokenizer:      {}",
        match args.tokenizer {
            TokenizerKind::Hipfire => "hipfire (native BPE — NOT byte-parity with PyTorch oracle)",
            TokenizerKind::LlamaCpp => "llama-cpp (subprocess llama-tokenize for byte-parity)",
        }
    );
    if args.tokenizer == TokenizerKind::LlamaCpp {
        eprintln!("  llama-tokenize: {}", args.llama_tokenize_bin.display());
        if let Some(g) = &args.bf16_gguf {
            eprintln!("  bf16-gguf:      {}", g.display());
        }
    }
    eprintln!();

    if let Err(msg) = run(&args) {
        eprintln!("collect_hessian_quantized: ERROR: {msg}");
        std::process::exit(1);
    }

    // -----------------------------------------------------------------
    // run() — keeps the early-exit `?` ergonomics out of `main`.
    // -----------------------------------------------------------------
    fn run(args: &Args) -> Result<(), String> {
        // 1. GPU init.
        eprintln!("[1/7] initializing GPU...");
        let mut gpu = Gpu::init().map_err(|e| format!("Gpu::init failed: {e}"))?;
        eprintln!("[1/7] arch={}", gpu.arch);
        // gfx12 Lloyd kernels gated by HIPFIRE_LLOYD_GFX12 (per PR #195).
        // Harmless on other arches; the calibration loop will fall through
        // to the generic Lloyd dispatch if the model is not Lloyd-quantized.
        if gpu.arch.starts_with("gfx12") {
            unsafe {
                std::env::set_var("HIPFIRE_LLOYD_GFX12", "1");
            }
            eprintln!("[1/7] arch is gfx12; set HIPFIRE_LLOYD_GFX12=1");
        }

        // 2. Load quantized model from .hfq.
        eprintln!("[2/7] loading quantized model from {}...", args.model.display());
        let mut hfq = HfqFile::open(&args.model)
            .map_err(|e| format!("HfqFile::open failed: {e:?}"))?;
        let config = qwen35::config_from_hfq(&hfq)
            .ok_or_else(|| "config_from_hfq failed (model is not a Qwen3.5 .hfq?)".to_string())?;
        let weights = qwen35::load_weights(&mut hfq, &config, &mut gpu)
            .map_err(|e| format!("qwen35::load_weights failed: {e}"))?;
        eprintln!(
            "[2/7] loaded: n_layers={} dim={} vocab_size={} n_kv_heads={} head_dim={}",
            config.n_layers,
            config.dim,
            config.vocab_size,
            config.n_kv_heads,
            config.head_dim
        );

        // 3. Install the Hessian capture handler. The collector hooks
        //    `gpu.capture_handler`; every linear-projection dispatch
        //    site inside `qwen35::forward_prefill_batch` fires
        //    `gpu.try_capture_linear(x)` which routes the F32
        //    activation into `HessianCollector::capture`. The capture
        //    impl casts F32→BF16 via the gfx942 RTNE kernel and
        //    accumulates the K×K outer product on-GPU.
        eprintln!("[3/7] installing HessianCollector capture handler...");
        let collector = Arc::new(HessianCollector::new());
        gpu.capture_handler = Some(collector.clone());

        // 4. Tokenize corpus — same code paths as `collect_hessian.rs`.
        let tokens = match args.tokenizer {
            TokenizerKind::LlamaCpp => {
                let bf16_gguf = args.bf16_gguf.as_ref().ok_or_else(|| {
                    "internal error: --tokenizer llama-cpp reached run() with no \
                     --bf16-gguf; the arg validator should have caught this"
                        .to_string()
                })?;
                eprintln!(
                    "[4/7] tokenizing corpus via subprocess llama-tokenize ({})...",
                    args.llama_tokenize_bin.display()
                );
                let toks = tokenize_corpus_via_llama_tokenize(
                    &args.llama_tokenize_bin,
                    bf16_gguf,
                    &args.corpus,
                )?;
                eprintln!("[4/7] tokenized via llama-tokenize: {} tokens", toks.len());
                toks
            }
            TokenizerKind::Hipfire => {
                eprintln!(
                    "[4/7] tokenizing corpus via hipfire native BPE \
                     (no PyTorch-oracle parity)..."
                );
                let corpus_text = std::fs::read_to_string(&args.corpus).map_err(|e| {
                    format!("failed to read corpus at {}: {e}", args.corpus.display())
                })?;
                // The hipfire fallback path uses the .hfq's parent dir as
                // the tokenizer source (mirrors `tokenize_corpus`'s expected
                // model_dir layout). In practice the `.hfq` already carries
                // an embedded tokenizer, but `tokenize_corpus` is the BF16
                // path's existing helper — for the .hfq fallback the user
                // is expected to drop a `tokenizer.json` next to the .hfq.
                let model_dir = args
                    .model
                    .parent()
                    .ok_or_else(|| {
                        format!(
                            "--tokenizer hipfire requires .hfq to live in a directory; \
                             got --model {} with no parent",
                            args.model.display()
                        )
                    })?
                    .to_path_buf();
                let toks = tokenize_corpus(&model_dir, &corpus_text)?;
                eprintln!("[4/7] tokenized via hipfire BPE: {} tokens", toks.len());
                toks
            }
        };

        // 5. Run forward over N passes × N sequences.
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
                "[5/7] WARNING: corpus only supplies {} sequences (requested {}); \
                 proceeding with {}",
                actual_sequences, args.n_sequences, actual_sequences
            );
        }
        let total_tokens = args.n_passes * actual_sequences * args.ctx_len;
        eprintln!(
            "[5/7] running per-token forward over {} passes × {} sequences × {} tokens \
             = {} total tokens...",
            args.n_passes,
            actual_sequences,
            args.ctx_len,
            total_tokens
        );
        eprintln!(
            "[5/7] note: per-token forward_scratch (not batched prefill) so the capture\n  \
             handler fires at every linear projection. Expect ~5–10 ms/token on 0.8B,\n  \
             so a full 128×2048 sweep takes ~30–60 min — that's the price of correct\n  \
             Hessian capture under quantization noise."
        );

        // KV cache + DeltaNet state + Qwen35Scratch. KvCache is sized
        // to ctx_len + slack; per-seq we restart at pos=0 (overwriting
        // the previous sequence's content). DeltaNetState is reset in
        // place per sequence via `reset(&mut gpu)`.
        let kv_max = args.ctx_len + 16;
        let mut kv_cache = match args.kv_mode.as_str() {
            "q8" => KvCache::new_gpu_q8(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .map_err(|e| format!("KvCache::new_gpu_q8 failed: {e}"))?,
            "asym4" => KvCache::new_gpu_asym4(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .map_err(|e| format!("KvCache::new_gpu_asym4 failed: {e}"))?,
            "asym3" => KvCache::new_gpu_asym3(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .map_err(|e| format!("KvCache::new_gpu_asym3 failed: {e}"))?,
            "asym2" => KvCache::new_gpu_asym2(
                &mut gpu,
                config.n_layers,
                config.n_kv_heads,
                config.head_dim,
                kv_max,
            )
            .map_err(|e| format!("KvCache::new_gpu_asym2 failed: {e}"))?,
            other => return Err(format!("unknown --kv-mode {other}")),
        };
        // repeat_window=64: matches eval_hipfire's setting. The repeat
        // window controls the internal LA decode-mode buffer size; for
        // pure batched-prefill use it's tangential, but the scratch
        // structure still requires a positive value.
        let scratch = Qwen35Scratch::new(&mut gpu, &config, 64)
            .map_err(|e| format!("Qwen35Scratch::new failed: {e}"))?;
        let mut dn_state = DeltaNetState::new(&mut gpu, &config)
            .map_err(|e| format!("DeltaNetState::new failed: {e}"))?;

        let t_loop_start = Instant::now();
        let mut tokens_done: usize = 0;
        for pass in 0..args.n_passes {
            for seq_idx in 0..actual_sequences {
                let start = seq_idx * args.ctx_len;
                let end = start + args.ctx_len;
                let chunk = &tokens[start..end];

                // Reset DeltaNet recurrent state per sequence; KvCache
                // gets overwritten from pos=0 each call below (we pass
                // pos=i so the first token of each new sequence lands
                // at slot 0, overwriting the previous sequence's KV).
                // Matches the per-chunk reset pattern in
                // eval_hipfire.rs (line ~345).
                dn_state.reset(&mut gpu);

                // Per-token forward_scratch — calls `forward_from_x_gpu`
                // internally, which has `set_capture_name(...)` at every
                // linear projection. The capture handler installed in
                // step 3 fires at each call, accumulating the K×K
                // outer-product into HessianCollector. We don't
                // consume logits; `scratch.logits` gets overwritten
                // each call, but its contents are irrelevant to
                // Hessian capture.
                //
                // Why per-token (not batched): batched
                // `forward_prefill_batch` routes through fused QKV /
                // fused gate_up / FA3 kernels that skip
                // `set_capture_name`. The capture handler never
                // fires under that path, so the HFHS sidecar ends
                // up with 0 entries. Per-token is slower but
                // captures correctly.
                for (i, &token) in chunk.iter().enumerate() {
                    qwen35::forward_scratch(
                        &mut gpu,
                        &weights,
                        &config,
                        token,
                        i,
                        &mut kv_cache,
                        &mut dn_state,
                        &scratch,
                    )
                    .map_err(|e| {
                        format!(
                            "forward_scratch failed at pass={pass} seq_idx={seq_idx} \
                             pos={i}: {e}"
                        )
                    })?;
                    tokens_done += 1;
                }

                if seq_idx % 8 == 0 {
                    let elapsed = t_loop_start.elapsed().as_secs_f64();
                    let rate = (tokens_done as f64) / elapsed.max(1e-9);
                    let pct =
                        (tokens_done as f64) * 100.0 / (total_tokens as f64).max(1.0);
                    let eta_s = if rate > 0.0 {
                        (total_tokens.saturating_sub(tokens_done)) as f64 / rate
                    } else {
                        f64::INFINITY
                    };
                    eprintln!(
                        "[5/7] pass {}/{}, sequence {}/{}  ({} tok done, \
                         {:.1}%, {:.0} tok/s, ETA {:.0}s)",
                        pass + 1,
                        args.n_passes,
                        seq_idx + 1,
                        actual_sequences,
                        tokens_done,
                        pct,
                        rate,
                        eta_s
                    );
                }
            }
        }
        let loop_elapsed = t_loop_start.elapsed().as_secs_f64();
        eprintln!(
            "[5/7] forward loop done: {} tokens in {:.1}s ({:.1} tok/s)",
            tokens_done,
            loop_elapsed,
            (tokens_done as f64) / loop_elapsed.max(1e-9)
        );

        // 6. Drain the collector.
        eprintln!("[6/7] draining HessianCollector...");
        gpu.capture_handler = None;
        let entries = match Arc::try_unwrap(collector) {
            Ok(c) => c.drain(&gpu)?,
            Err(arc) => arc.drain(&gpu)?,
        };
        eprintln!("[6/7] drained {} Hessian entries", entries.len());

        // Hard-abort if the capture handler never fired. This is the
        // exact failure mode we just fixed (forward_prefill_batch
        // doesn't call set_capture_name) — if the per-token
        // forward_scratch path also produces 0 entries, something
        // else is wrong upstream (capture wiring removed, handler
        // not installed, etc.) and the downstream GPTQ rounds would
        // silently consume a degenerate 24-byte HFHS file. Fail loud.
        if entries.is_empty() {
            eprintln!(
                "collect_hessian_quantized: ERROR: HessianCollector drained 0 entries.\n  \
                 The capture handler did not fire during forward. This usually means\n  \
                 the forward path no longer calls set_capture_name at linear projections,\n  \
                 or the capture_handler install in step 3 was bypassed.\n  \
                 Refusing to write a 24-byte (header-only) HFHS file."
            );
            std::process::exit(2);
        }

        // 7. Normalize by n_tokens, strip trailing `.weight` from tensor
        //    names (HFHS-v1 convention — see hessian_io.rs:242-243), and
        //    write the HFHS-v1 binary. Same on-disk layout the BF16 bin
        //    emits, so downstream consumers (hipfire-quantize iterative
        //    GPTQ rounds 1+) read it without any flag.
        eprintln!("[7/7] writing HFHS-v1 binary to {}...", args.output.display());
        let writer_entries: Vec<hipfire_quantize::hfhs_writer::HessianEntry> = entries
            .into_iter()
            .map(|e| {
                let inv_n = if e.n_tokens == 0 {
                    0.0_f32
                } else {
                    1.0_f32 / (e.n_tokens as f32)
                };
                let h_avg: Vec<f32> = e.h.iter().map(|&v| v * inv_n).collect();
                let canonical_name = e
                    .name
                    .strip_suffix(".weight")
                    .map(|s| s.to_string())
                    .unwrap_or(e.name);
                hipfire_quantize::hfhs_writer::HessianEntry {
                    name: canonical_name,
                    expert_idx: e.expert_idx,
                    k: e.k as u32,
                    h: h_avg,
                }
            })
            .collect();
        hipfire_quantize::hfhs_writer::write_hfhs(&args.output, &writer_entries)
            .map_err(|e| format!("write_hfhs failed: {e}"))?;
        eprintln!(
            "[7/7] wrote {} entries to {}",
            writer_entries.len(),
            args.output.display()
        );
        Ok(())
    }
}
