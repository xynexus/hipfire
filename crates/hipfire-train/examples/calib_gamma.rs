// SPDX-License-Identifier: Apache-2.0
//! Per-linear output-gradient ENERGY (`gamma`) driver, LLaMA-dense.
//!
//! Loads an fp32 LLaMA (safetensors dir), runs forward + cross-entropy backward
//! over calibration sequences, and writes one f32 per projection: the mean
//! squared OUTPUT gradient, `gamma_i = E[ ||dL/dy_i||^2 ] / n_out`.
//!
//! WHY: the mixed-precision allocator ranks tensors by `tr(dW^T H dW)` — input
//! covariance only — which implicitly sets the output side to the identity.
//! Measured, that ranks `o_proj` 79th-113th of 113 while it is the single
//! largest promotion win (-15.1% KLD alone). `gamma` is the missing factor; the
//! K-FAC form is `dL ~= 1/2 tr(dW^T G dW H)` and `G ~= gamma*I` makes it a
//! scalar multiplier on the objective already computed.
//!
//! This is NOT the same statistic `calib_guided` captures. That one normalises
//! the per-token weights to mean 1 within each tensor — correct for its own
//! purpose, but it discards exactly the cross-tensor magnitude wanted here.
//!
//!   calib_gamma <model_dir-or-.hfq> <out.json> [seq] [n_seq] [--text <file>]
//!
//! With `--text`, the model's `tokenizer.json` encodes the file into real
//! calibration sequences (meaningful Fisher weights). Without it, tokens are
//! seeded-synthetic (proves the pipeline only).

use hipfire_model::tokenizer::Tokenizer;
use hipfire_model::ModelSource;
use hipfire_rdna::Gpu;
use hipfire_train::loader::{load_llama_fp32, load_llama_fp32_hfq};
use hipfire_train::model::{
    free_model_acts, model_forward, model_gamma_backward, model_gamma_streamed, GammaAccum,
    LlamaModel,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Positional: <model_dir> <out.calib.hfq> [seq] [n_seq]; flag: --text <file>.
    let mut text_path: Option<String> = None;
    let mut tok_dir: Option<String> = None;
    // Page one layer at a time instead of holding the model. Must produce the
    // SAME gamma table as the whole-model path — that equality is the test.
    let mut streamed = false;
    let mut plain = false; // --plain ⇒ w≡1 baseline (plain XᵀX over the same tokens)
    let mut skip_seq = 0usize; // --skip N ⇒ drop the first N sequences (held-out split)
    let mut pos_args: Vec<String> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--text" => text_path = it.next(),
            "--tokenizer" => tok_dir = it.next(),
            "--streamed" => streamed = true,
            "--plain" => plain = true,
            "--skip" => skip_seq = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            _ => pos_args.push(a),
        }
    }
    let fisher = !plain;
    let _ = fisher;
    let dir = pos_args
        .first()
        .cloned()
        .expect("usage: calib_guided <model_dir> <out.calib.hfq> [seq] [n_seq] [--text <file>]");
    let out = pos_args.get(1).cloned().expect("missing <out.calib.hfq>");
    let seq: usize = pos_args.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);
    let n_seq: usize = pos_args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    let mut gpu = Gpu::init().expect("Gpu::init");
    // Accept either a safetensors dir or a .hfq artifact — on this box the
    // measured models are .hfq (HF snapshots ship Meta .pth).
    let dpath = Path::new(&dir);
    let (cfg, w) = if dpath.extension().is_some_and(|e| e == "hfq") {
        load_llama_fp32_hfq(&mut gpu, dpath)?
    } else {
        load_llama_fp32(&mut gpu, dpath)?
    };
    let vocab = cfg.vocab_size;
    // rank-1 LoRA (B=0 ⇒ zero contribution); the backward computes + discards
    // its grads — this path drives the weighted-Hessian capture, not training.
    let model = LlamaModel::from_f32_weights(&mut gpu, &cfg, w, seq, 1, 1.0)?;
    // MoE geometry from the artifact's metadata; 0 experts ⇒ a dense model.
    // Routed-ness is probed per LAYER inside the walk, since hybrid models
    // (dense layer 0, routed above) exist.
    let (n_experts, top_k) = {
        let meta: serde_json::Value =
            serde_json::from_str(hipfire_runtime::hfq::HfqFile::open(dpath)?.metadata_json())
                .unwrap_or(serde_json::Value::Null);
        let c = meta.get("config").unwrap_or(&meta);
        let g = |k: &str| c.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let ne = g("num_experts").max(g("num_local_experts"));
        let tk = g("num_experts_per_tok").max(g("experts_per_tok"));
        (ne, tk.max(if ne > 0 { 1 } else { 0 }))
    };
    if n_experts > 0 {
        eprintln!("MoE: {n_experts} experts, top_k {top_k}");
    }
    let mut acc = GammaAccum::default();

    // Build calibration sequences.
    let sequences: Vec<Vec<u32>> = if let Some(tp) = &text_path {
        // For a .hfq input there is no sibling tokenizer.json; allow
        // --tokenizer to point at the HF snapshot that has one.
        let tok_path = tok_dir
            .as_ref()
            .map(|d| Path::new(d).join("tokenizer.json"))
            .unwrap_or_else(|| Path::new(&dir).join("tokenizer.json"));
        let tok =
            Tokenizer::from_tokenizer_json(&tok_path)?.ok_or("no tokenizer.json in model dir")?;
        let text = std::fs::read_to_string(tp)?;
        let ids = tok.encode(&text);
        eprintln!("tokenized {} chars -> {} tokens", text.len(), ids.len());
        ids.chunks(seq)
            .filter(|c| c.len() == seq)
            .skip(skip_seq)
            .take(n_seq)
            .map(|c| c.to_vec())
            .collect()
    } else {
        let mut s = 0x1234_5678u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as usize
        };
        (0..n_seq)
            .map(|_| (0..seq).map(|_| (next() % vocab) as u32).collect())
            .collect()
    };
    if sequences.is_empty() {
        return Err("no calibration sequences (text too short for one seq?)".into());
    }
    eprintln!(
        "calibrating on {} sequence(s) × {} tokens ({}, {})",
        sequences.len(),
        seq,
        if text_path.is_some() {
            "real text"
        } else {
            "synthetic"
        },
        if fisher {
            "Fisher-weighted"
        } else {
            "plain XᵀX control"
        }
    );

    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();
    let mut total = 0.0f32;
    for (si, toks) in sequences.iter().enumerate() {
        // CE targets = next token; the final position is ignored (-1).
        let mut targets = vec![0.0f32; seq];
        for t in 0..seq - 1 {
            targets[t] = toks[t + 1] as f32;
        }
        targets[seq - 1] = -1.0;
        let loss = if streamed {
            let hfq = hipfire_runtime::hfq::HfqFile::open(dpath)?;
            model_gamma_streamed(
                &mut gpu,
                &hfq,
                &cfg,
                &model.embed,
                model.lm_head.as_ref(),
                &model.final_norm,
                &model.dims,
                toks,
                &pos,
                &targets,
                -1,
                n_experts,
                top_k,
                &mut acc,
            )?
        } else {
            let acts = model_forward(&mut gpu, &model, toks, &pos)?;
            let l = model_gamma_backward(&mut gpu, &model, &acts, &targets, -1, &mut acc)?;
            free_model_acts(&mut gpu, acts)?;
            l
        };
        total += loss;
        eprintln!(
            "seq {}/{}  ce/tok {:.3}",
            si + 1,
            sequences.len(),
            loss / seq as f32
        );
    }
    let gamma = acc.finish();
    eprintln!(
        "mean ce/tok {:.4} over {} tensors",
        total / (sequences.len() * seq) as f32,
        gamma.len()
    );

    // Plain JSON keyed by HFQ tensor name without `.weight`, matching the
    // imatrix/hessian convention so the quantizer joins it the same way.
    let mut keys: Vec<&String> = gamma.keys().collect();
    keys.sort();
    let body: Vec<String> = keys
        .iter()
        .map(|k| format!("  {:?}: {:e}", k, gamma[*k]))
        .collect();
    std::fs::write(&out, format!("{{\n{}\n}}\n", body.join(",\n")))?;
    eprintln!("wrote {out}  ({} gamma entries)", gamma.len());
    Ok(())
}
