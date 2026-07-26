#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! DSpark training-LABEL generation for the **gemma3** text target (M2.1).
//!
//! The gemma3 analog of `hipfire-train/examples/dspark_labels.rs` (which is
//! llama-family only, via `load_llama_from_hfq`). Produces the identical `DSLB v1`
//! label cache the DSpark drafter trains against, but drives gemma3's own forward:
//! per-token capture via [`gemma3::forward::forward_step_capture`] (the M1a
//! `HiddenCaptureSink` path) and the block lm-head via
//! [`SpecTarget::lm_head_logits`] (gemma3 final-norm + output proj).
//!
//! Pipeline, per prompt (a pre-tokenized u32 sequence):
//!   1. reset state, per-token capturing forward over the whole prompt with
//!      `extract = target_layers ∪ final_layer`; the sink appends the post-FFN
//!      residual at each extract layer, `[n_pos × num_extract × dim]`.
//!   2. slide `ctx_len`-context + `block`-draft anchor windows by `--stride`.
//!   3. per window: main_hidden (target layers), block final-layer hidden →
//!      lm-head → target_logits, hard next-tokens (argmax), block/prev tokens,
//!      eval_mask; stream to the DSLB cache.
//!
//! DSLB v1 binary format is byte-identical to the llama emitter (see that file's
//! header doc) so the existing trainer/reader consume it unchanged; only the
//! `target_path` in the header points at the gemma3 `.hfq`.
//!
//! ```text
//! hipfire lock acquire gemma3-dspark-labels
//! cargo run --release --example dspark_labels -p hipfire-arch-gemma3 -- \
//!   --hfq ~/.hipfire/models/medgemma-1.5-4b-it-q8f16.hfq \
//!   --prompts ~/.hipfire/datasets/gemma3-4b-dspark/corpus.jsonl \
//!   --target-layers 1,9,17,25,33 --block 7 --ctx-len 128 --stride 64 \
//!   --max-windows 20000 --out ~/.hipfire/calib/gemma3-4b.dslb
//! hipfire lock release
//! ```
//! `--prompts`: one token sequence per line — a JSON array / `{"tokens":[...]}`
//! (`.jsonl`) or whitespace ints (`.txt`).

use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use hipfire_arch_gemma3 as gemma3;
use hipfire_arch_gemma3::arch::Gemma3Backend;
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::HiddenCaptureSink;
use hipfire_specdecode_dspark::spec::SpecTarget;

fn w_u32(w: &mut impl Write, x: u32) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_i32(w: &mut impl Write, x: i32) -> std::io::Result<()> {
    w.write_all(&x.to_le_bytes())
}
fn w_f32s(w: &mut impl Write, v: &[f32]) -> std::io::Result<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    w.write_all(bytes)
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bestv = f32::NEG_INFINITY;
    for (j, &x) in row.iter().enumerate() {
        if x > bestv {
            bestv = x;
            best = j;
        }
    }
    best
}

struct Args {
    hfq: String,
    prompts: String,
    target_layers: Vec<usize>,
    block: usize,
    ctx_len: usize,
    stride: usize,
    max_windows: usize,
    out: String,
}

fn parse_args() -> Args {
    let mut hfq = None;
    let mut prompts = None;
    let mut target_layers = vec![1usize, 9, 17, 25, 33];
    let mut block = 7usize;
    let mut ctx_len = 128usize;
    let mut stride: Option<usize> = None;
    let mut max_windows = 20000usize;
    let mut out = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--hfq" => hfq = it.next(),
            "--prompts" => prompts = it.next(),
            "--target-layers" => {
                target_layers = it
                    .next()
                    .unwrap()
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect()
            }
            "--block" => block = it.next().and_then(|s| s.parse().ok()).unwrap_or(7),
            "--ctx-len" => ctx_len = it.next().and_then(|s| s.parse().ok()).unwrap_or(128),
            "--stride" => stride = it.next().and_then(|s| s.parse().ok()),
            "--max-windows" => {
                max_windows = it.next().and_then(|s| s.parse().ok()).unwrap_or(20000)
            }
            "--out" => out = it.next(),
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    Args {
        hfq: hfq.expect("--hfq required"),
        prompts: prompts.expect("--prompts required"),
        target_layers,
        block,
        ctx_len,
        stride: stride.unwrap_or(block),
        max_windows,
        out: out.expect("--out required"),
    }
}

fn load_prompts(path: &str) -> Vec<Vec<u32>> {
    let text = std::fs::read_to_string(path).expect("read prompts");
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let toks: Vec<u32> = if t.starts_with('[') || t.starts_with('{') {
            let v: serde_json::Value = match serde_json::from_str(t) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let arr = if v.is_array() {
                v.as_array().cloned().unwrap_or_default()
            } else {
                v["tokens"].as_array().cloned().unwrap_or_default()
            };
            arr.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect()
        } else {
            t.split_whitespace()
                .filter_map(|s| s.parse::<u32>().ok())
                .collect()
        };
        if !toks.is_empty() {
            out.push(toks);
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    // ── load the gemma3 target ────────────────────────────────────────────────
    let mut hfq = HfqFile::open(Path::new(&args.hfq))?;
    let config = gemma3::config_from_hfq(&hfq).ok_or("gemma3 config parse failed")?;
    let dim = config.hidden_size;
    let vocab = config.vocab_size;
    let n_layers = config.num_hidden_layers;
    eprintln!("target: dim={dim} layers={n_layers} vocab={vocab}");

    // multimodal wrapper nests the decoder under "language_model.".
    let arch_str = serde_json::from_str::<serde_json::Value>(&hfq.metadata_json)
        .ok()
        .and_then(|v| {
            v.get("architecture")
                .and_then(|a| a.as_str())
                .map(String::from)
        })
        .unwrap_or_default();
    let prefix = if arch_str == "gemma3_text" {
        ""
    } else {
        "language_model."
    };

    // Target layers → ascending-unique valid indices; extract = ∪ final layer.
    let mut target_layers: Vec<usize> = args
        .target_layers
        .iter()
        .map(|&l| l.min(n_layers - 1))
        .collect();
    target_layers.sort_unstable();
    target_layers.dedup();
    let n_targets = target_layers.len();
    assert!(n_targets > 0, "no target layers");
    let lm_layer = n_layers - 1;
    let mut extract = target_layers.clone();
    if !extract.contains(&lm_layer) {
        extract.push(lm_layer);
    }
    extract.sort_unstable();
    extract.dedup();
    let num_extract = extract.len();
    let col_of = |layer: usize| extract.iter().position(|&x| x == layer).unwrap();
    let target_cols: Vec<usize> = target_layers.iter().map(|&l| col_of(l)).collect();
    let lm_col = col_of(lm_layer);

    let prompts = load_prompts(&args.prompts);
    assert!(!prompts.is_empty(), "no prompts");
    let max_len = prompts.iter().map(|p| p.len()).max().unwrap();
    eprintln!(
        "{} prompts (max len {max_len}); ctx_len={} block={} stride={} extract={:?}",
        prompts.len(),
        args.ctx_len,
        args.block,
        args.stride,
        extract
    );

    // ── GPU setup ─────────────────────────────────────────────────────────────
    let mut gpu = Gpu::init()?;
    let weights = gemma3::weights::load_weights_prefixed(&mut hfq, &config, &mut gpu, prefix)?;
    let max_seq = max_len.max(args.ctx_len + args.block);
    let state = gemma3::Gemma3State::new_with_max_seq(
        &mut gpu,
        &config,
        max_seq,
        hipfire_runtime::kv::KvQuantMode::Unquantized,
        4,
    )
    .map_err(|e| format!("state: {e:?}"))?;
    let mut backend = Gemma3Backend::new(config.clone(), weights, state);

    // ── output cache: header, stream windows, patch n_windows ─────────────────
    let mut f = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    f.write_all(b"DSLB")?;
    w_u32(&mut f, 1)?; // version
    w_u32(&mut f, vocab as u32)?;
    w_u32(&mut f, dim as u32)?;
    w_u32(&mut f, n_targets as u32)?;
    w_u32(&mut f, args.block as u32)?;
    w_u32(&mut f, args.ctx_len as u32)?;
    w_u32(&mut f, 0)?; // flags
    w_u32(&mut f, 0)?; // n_windows placeholder @ offset 32
    w_u32(&mut f, n_targets as u32)?;
    for &l in &target_layers {
        w_u32(&mut f, l as u32)?;
    }
    let tpath = args.hfq.as_bytes();
    w_u32(&mut f, tpath.len() as u32)?;
    f.write_all(tpath)?;

    let mut n_windows: u32 = 0;
    let row_stride = num_extract * dim;

    'outer: for tokens in &prompts {
        let l = tokens.len();
        if l < args.ctx_len + args.block {
            continue;
        }

        // One capturing per-token prefill over the whole prompt (start_pos=0).
        backend.state.reset();
        let mut hidden: Vec<f32> = Vec::with_capacity(l * row_stride);
        for &tok in tokens.iter() {
            let mut sink = HiddenCaptureSink {
                extract_layers: &extract,
                hidden: &mut hidden,
                hidden_gpu: None,
            };
            gemma3::forward::forward_step_capture(
                &mut gpu,
                &backend.weights,
                &backend.config,
                &mut backend.state,
                tok,
                Some(&mut sink),
            )
            .map_err(|e| format!("capture forward: {e:?}"))?;
        }
        assert_eq!(hidden.len(), l * row_stride, "capture size mismatch");

        let last_off = l - args.ctx_len - args.block;
        let mut off = 0usize;
        while off <= last_off {
            // main_hidden [ctx_len * n_targets * dim].
            let mut main_hidden = vec![0.0f32; args.ctx_len * n_targets * dim];
            for p in 0..args.ctx_len {
                let src_base = (off + p) * row_stride;
                let dst_base = p * n_targets * dim;
                for (t, &col) in target_cols.iter().enumerate() {
                    let s = src_base + col * dim;
                    main_hidden[dst_base + t * dim..dst_base + (t + 1) * dim]
                        .copy_from_slice(&hidden[s..s + dim]);
                }
            }

            // Final-layer hidden at block positions → gemma3 lm-head → target_logits.
            let mut block_final = vec![0.0f32; args.block * dim];
            for i in 0..args.block {
                let s = (off + args.ctx_len + i) * row_stride + lm_col * dim;
                block_final[i * dim..(i + 1) * dim].copy_from_slice(&hidden[s..s + dim]);
            }
            let block_final_gpu = gpu.upload_f32(&block_final, &[args.block * dim])?;
            let target_logits = backend
                .lm_head_logits(&mut gpu, &block_final_gpu, args.block)
                .map_err(|e| format!("lm_head: {e}"))?;
            gpu.free_tensor(block_final_gpu)?;
            debug_assert_eq!(target_logits.len(), args.block * vocab);

            let mut next_tokens = vec![0i32; args.block];
            let mut block_tokens = vec![0u32; args.block];
            let mut prev_tokens = vec![0u32; args.block];
            let mut eval_mask = vec![0u8; args.block];
            for i in 0..args.block {
                let pos = off + args.ctx_len + i;
                let valid = pos < l;
                eval_mask[i] = valid as u8;
                block_tokens[i] = tokens[pos];
                prev_tokens[i] = tokens[pos - 1];
                next_tokens[i] = if valid {
                    argmax(&target_logits[i * vocab..(i + 1) * vocab]) as i32
                } else {
                    -100
                };
            }

            w_f32s(&mut f, &main_hidden)?;
            w_f32s(&mut f, &target_logits)?;
            for &t in &next_tokens {
                w_i32(&mut f, t)?;
            }
            for &t in &block_tokens {
                w_u32(&mut f, t)?;
            }
            for &t in &prev_tokens {
                w_u32(&mut f, t)?;
            }
            f.write_all(&eval_mask)?;

            n_windows += 1;
            if n_windows as usize >= args.max_windows {
                break 'outer;
            }
            off += args.stride;
        }
        if n_windows % 500 == 0 && n_windows > 0 {
            eprintln!("  {n_windows} windows...");
        }
    }

    f.flush()?;
    let mut inner = f.into_inner()?;
    inner.seek(SeekFrom::Start(32))?;
    inner.write_all(&n_windows.to_le_bytes())?;
    inner.flush()?;

    eprintln!(
        "wrote {n_windows} windows to {} (vocab={vocab} dim={dim} n_targets={n_targets} block={} ctx_len={})",
        args.out, args.block, args.ctx_len
    );
    Ok(())
}
