// SPDX-License-Identifier: Apache-2.0
// hipfire — LFM2.5 per-layer hidden-state dump for HF-reference bisection.
//
//! Runs `prefill_batch_with_hidden` over a fixed prompt capturing EVERY layer's
//! post-layer residual, and writes them plus the final-position logits to disk
//! for offline comparison against an HF `Lfm2ForCausalLM` reference
//! (`scripts/dump_lfm2_hf_reference.py` + `scripts/compare_lfm2_hidden.py`).
//!
//! hipfire post-layer residual for layer L aligns with HF
//! `output_hidden_states[L+1]` (HF index 0 is the post-embedding state).
//!
//! Usage:
//!   cargo run -p hipfire-arch-lfm2moe --features deltanet \
//!     --example dump_lfm2_hidden -- \
//!     --model <model.hfq> --tokens <tokens.json> --out-prefix /tmp/lfm2

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_lfm2moe::config::Lfm2MoeConfig;
    use hipfire_arch_lfm2moe::forward::{prefill_batch_with_hidden, Lfm2HiddenCapture};
    use hipfire_arch_lfm2moe::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::tokenizer::Tokenizer;
    use std::io::Write;
    use std::path::PathBuf;

    let argv: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt = "The capital of France is".to_string();
    let mut tokens_path: Option<PathBuf> = None;
    let mut out_prefix = "/tmp/lfm2".to_string();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--prompt" => {
                prompt = argv[i + 1].clone();
                i += 2;
            }
            "--tokens" => {
                tokens_path = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out-prefix" => {
                out_prefix = argv[i + 1].clone();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let model = model.expect("--model required");

    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let mut hfq = HfqFile::open(&model).expect("open model");
    let cfg = Lfm2MoeConfig::from_hfq(&hfq).expect("config");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let weights = Lfm2MoeWeights::load(&mut hfq, &cfg, &mut gpu).expect("weights");

    let prompt_ids: Vec<u32> = if let Some(path) = &tokens_path {
        let s = std::fs::read_to_string(path).expect("read --tokens");
        let v: Vec<i64> = serde_json::from_str(&s).expect("parse --tokens json");
        v.into_iter().map(|t| t as u32).collect()
    } else {
        tok.encode(&prompt)
    };
    let n = prompt_ids.len();
    assert!(n >= 2, "need >= 2 tokens");
    eprintln!(
        "lfm2 dump: hidden={} layers={} vocab={} tokens={:?}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size, prompt_ids
    );
    for (l, k) in (0..cfg.num_hidden_layers).map(|l| (l, cfg.mixer(l))) {
        eprint!("L{l}={k:?} ");
        let _ = l;
    }
    eprintln!();

    let max_seq = n + 8;
    let mut state = Lfm2MoeState::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state");
    let layers: Vec<usize> = (0..cfg.num_hidden_layers).collect();
    let mut cap = Lfm2HiddenCapture::new(cfg.num_hidden_layers, cfg.hidden_size, layers.clone())
        .expect("hidden capture");
    let logits = prefill_batch_with_hidden(&cfg, &weights, &mut state, &mut gpu, &prompt_ids, &mut cap)
        .expect("prefill_batch_with_hidden");

    // hidden.bin: magic, n_layers, n_pos, hidden, then [pos][layer][hidden] f32.
    let hidden_path = format!("{out_prefix}.hipfire.hidden.bin");
    let mut f = std::fs::File::create(&hidden_path).expect("create hidden");
    f.write_all(b"LFM2HID0").unwrap();
    f.write_all(&(cfg.num_hidden_layers as u32).to_le_bytes()).unwrap();
    f.write_all(&(n as u32).to_le_bytes()).unwrap();
    f.write_all(&(cfg.hidden_size as u32).to_le_bytes()).unwrap();
    f.write_all(&0u32.to_le_bytes()).unwrap();
    for &x in cap.rows() {
        f.write_all(&x.to_le_bytes()).unwrap();
    }

    // logits.json: final-position logits stats + top-20.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top: Vec<(u32, f32, String)> = idx
        .iter()
        .take(20)
        .map(|&j| (j as u32, logits[j], tok.decode(&[j as u32])))
        .collect();
    let mn = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    let logits_path = format!("{out_prefix}.hipfire.logits.json");
    let top_json: Vec<serde_json::Value> = top
        .iter()
        .map(|(id, v, s)| serde_json::json!({"id": id, "logit": v, "text": s}))
        .collect();
    let j = serde_json::json!({
        "tokens": prompt_ids,
        "logit_min": mn, "logit_max": mx, "logit_mean": mean,
        "argmax": idx[0] as u32,
        "argmax_text": tok.decode(&[idx[0] as u32]),
        "top20": top_json,
    });
    std::fs::write(&logits_path, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    eprintln!("wrote {hidden_path}");
    eprintln!("wrote {logits_path}");
    eprintln!(
        "final logits: min={mn:.3} max={mx:.3} mean={mean:.3} argmax={} ({:?})",
        idx[0],
        tok.decode(&[idx[0] as u32])
    );
}
