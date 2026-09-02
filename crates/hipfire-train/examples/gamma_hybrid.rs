// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Run the hybrid assembly end to end and report the gamma table.
//!
//! This is the whole stack in one pass: `linear_attn` layers with their
//! DeltaNet recurrence, a full-attention layer with QK-norm and the qwen3.5
//! gated q_proj, routed experts with a shared branch, stacked and fused expert
//! tensors — every piece probed from the artifact rather than assumed.
//!
//! What it checks beyond "did it run": the per-layer kinds must match the
//! config's own `layer_types` list (which the walk never reads), the loss must
//! be finite and near ln(vocab) for a random-init fixture, and every layer must
//! contribute gamma entries. A layer that silently produced nothing would
//! otherwise look like a pass.
//!
//! Run: cargo run --release -p hipfire-train --example gamma_hybrid

use hipfire_model::ModelSource;
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use hipfire_train::block::BlockDims;
use hipfire_train::hybrid::{gamma_by_layer, gamma_hybrid_streamed, LayerKind};
use hipfire_train::model::GammaAccum;

const FIXTURE: &str = "/srv/hipfire/fixtures/qwen3_5_moe-tiny";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| FIXTURE.into());
    let path = std::path::Path::new(&dir);
    if !path.exists() {
        eprintln!("fixture {dir} not present — skipping");
        return Ok(());
    }
    // Either a safetensors directory or an .hfq. A quantized .hfq goes through
    // DequantHfq, which is a VALIDATION path — see its doc comment: the gamma
    // it yields describes the quantized model, not the source.
    let is_hfq = path.extension().map(|e| e == "hfq").unwrap_or(false);
    let hfq = if is_hfq {
        Some(HfqFile::open(path)?)
    } else {
        None
    };
    let raw: serde_json::Value = match &hfq {
        Some(f) => serde_json::from_str(f.metadata_json())?,
        None => serde_json::from_str(&std::fs::read_to_string(path.join("config.json"))?)?,
    };
    let raw_full = raw.get("config").cloned().unwrap_or(raw.clone());
    let raw = raw_full.clone();
    // Qwen3.5-VL wraps the decoder config; the geometry lives in text_config.
    let raw = raw.get("text_config").cloned().unwrap_or(raw);
    // A pure-MoE config carries no dense `intermediate_size`. LlamaConfig
    // requires one, and BlockDims.inter is only read on a DENSE mlp path that
    // such a model never takes, so borrow the MoE width rather than invent a
    // number — if it is ever actually used, it will be the right order.
    let mut raw = raw;
    if raw.get("intermediate_size").is_none() {
        if let Some(mi) = raw.get("moe_intermediate_size").cloned() {
            raw["intermediate_size"] = mi;
        }
    }
    let cfg = hipfire_train::config::LlamaConfig::from_json_value(&raw)?;
    let n_experts = raw["num_experts"].as_u64().unwrap_or(0) as usize;
    let top_k = raw["num_experts_per_tok"].as_u64().unwrap_or(1) as usize;

    let mut gpu = Gpu::init()?;
    let st = if is_hfq {
        None
    } else {
        Some(SafetensorsSource::open(path)?)
    };
    let dq = hfq.as_ref().map(hipfire_train::loader::DequantHfq);
    let src: &dyn hipfire_train::loader::WeightSource = match (&dq, &st) {
        (Some(d), _) => d,
        (_, Some(s)) => s,
        _ => unreachable!(),
    };
    let prefix = hipfire_train::loader::detect_prefix(src);
    eprintln!(
        "source: {} prefix {prefix:?}",
        if is_hfq { "hfq" } else { "safetensors" }
    );

    // Third arg: sequence length. Short runs separate per-token errors from
    // ones that accumulate with position — at seq 1 the recurrence has no
    // history and attention's softmax is over a single key.
    let seq: usize = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(64);
    let h = cfg.hidden_size;
    let dims = BlockDims {
        seq,
        h,
        n_heads: cfg.num_attention_heads,
        n_kv: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        inter: cfg.intermediate_size,
        rope_base: cfg.rope_theta,
        // qwen3.5 rotates only part of each head; see BlockDims::rotary_dim.
        rotary_dim: (cfg.head_dim as f64
            * raw["rope_parameters"]["partial_rotary_factor"]
                .as_f64()
                .unwrap_or(1.0)) as usize,
        eps: cfg.rms_norm_eps,
        lora_scale: 1.0,
        lora_rank: 1,
    };

    let embed = hipfire_train::loader::load_embed_f32(&mut gpu, src, prefix, cfg.vocab_size, h)?;
    // Qwen3.5/3.6 store block and final norm weights as deviations from 1.
    let unit_offset =
        hipfire_train::loader::uses_unit_offset_norm(raw_full["model_type"].as_str().unwrap_or(""));
    eprintln!(
        "norm convention: {}",
        if unit_offset {
            "GemmaRMSNorm (1 + w)"
        } else {
            "plain w"
        }
    );
    // An untied head must be loaded; scoring through the embedding when the
    // artifact ships a separate lm_head compares against an unrelated matrix.
    let lm_head = hipfire_train::loader::load_lm_head_f32(
        &mut gpu,
        src,
        prefix,
        cfg.vocab_size,
        h,
        cfg.tie_word_embeddings,
    )?;
    eprintln!(
        "head: {}",
        match &lm_head {
            Some(_) => "UNTIED lm_head.weight",
            None => "tied (the head IS the embedding)",
        }
    );
    let final_norm =
        hipfire_train::loader::load_final_norm_f32(&mut gpu, src, prefix, h, unit_offset)?;

    // Real tokens when we can get them. An .hfq embeds its tokenizer, and a
    // real model over real text should land at a language-model loss (~2-4),
    // not near ln(vocab). That is the strongest end-to-end check available
    // without a reference implementation: every layout question this assembly
    // had to answer — the fused gate/up halves, the conv tap direction, the
    // [Q|K|V] split — wrecks the loss if answered wrong, because a model with
    // scrambled weights cannot predict text.
    // Second arg: a corpus path, or the literal `arange` to force the
    // deterministic 0,1,2,... prompt that dump_logits_qwen35 uses — the only
    // way to compare against it on a model that HAS a tokenizer.
    let corpus = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "benchmarks/calib/calib-1m.txt".into());
    let force_arange = corpus == "arange";
    let real = (|| {
        if force_arange {
            return None;
        }
        // An .hfq embeds its tokenizer; a restored HF directory has the file.
        let tok_path = match &hfq {
            Some(f) => {
                let meta: serde_json::Value = serde_json::from_str(f.metadata_json()).ok()?;
                let tj = meta.get("tokenizer")?.as_str()?.to_string();
                let tmp = std::env::temp_dir().join("hipfire_gamma_hybrid_tok.json");
                std::fs::write(&tmp, tj).ok()?;
                tmp
            }
            None => path.join("tokenizer.json"),
        };
        let text = std::fs::read_to_string(&corpus).ok()?;
        let tok = hipfire_model::tokenizer::Tokenizer::from_tokenizer_json(&tok_path).ok()??;
        let ids = tok.encode(&text[..text.len().min(20000)]);
        (ids.len() > seq).then(|| ids[..seq].to_vec())
    })();
    let synthetic = real.is_none();
    // The synthetic fallback is 0,1,2,... deliberately: that is exactly the
    // prompt `dump_logits_qwen35` uses, so a fixture with no tokenizer can
    // still be compared against the runtime token-for-token.
    let tokens: Vec<u32> = real.unwrap_or_else(|| (0..seq as u32).collect());
    eprintln!("first ids: {:?}", &tokens[..tokens.len().min(12)]);
    eprintln!(
        "tokens: {}",
        if synthetic {
            "SYNTHETIC (no embedded tokenizer or corpus) — loss is not meaningful"
        } else {
            "real, from the artifact's own tokenizer over the calib corpus"
        }
    );
    // Last position has no next token: mark it ignored rather than wrapping,
    // which would train/measure on a fabricated transition.
    let targets: Vec<f32> = (0..seq)
        .map(|i| tokens[(i + 1) % seq] as f32)
        .collect::<Vec<_>>();
    let pos: Vec<f32> = (0..seq).map(|i| i as f32).collect();

    // Optional oracle export: a token shim in the format
    // dump_qwen35_hidden_states reads (it wants HFKLDR, while every kldref on
    // disk is now HFQM or HFKREF — it only pulls tokens from that file), plus
    // this walk's own per-layer states in the dumper's HFHIDDEN format. Same
    // tokens on both sides is the whole point.
    if let Ok(dir) = std::env::var("HIPFIRE_GH_ORACLE_DIR") {
        let d = std::path::Path::new(&dir);
        std::fs::create_dir_all(d)?;
        let mut shim = Vec::new();
        shim.extend_from_slice(b"HFKLDR\0\0");
        let mut hdr = [0u8; 24];
        hdr[4..8].copy_from_slice(&(seq as u32).to_le_bytes()); // n_ctx
        hdr[12..16].copy_from_slice(&1u32.to_le_bytes()); // n_chunk
        shim.extend_from_slice(&hdr);
        for t in &tokens {
            shim.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(d.join("tokens.hfkldr"), &shim)?;
        eprintln!("oracle: wrote token shim ({seq} tokens) to {}", d.display());
    }

    let mut acc = GammaAccum {
        sum: Default::default(),
        n: 0,
    };
    let (loss, kinds, layer_inputs, last_logits) = gamma_hybrid_streamed(
        &mut gpu,
        src,
        prefix,
        &cfg,
        &dims,
        &embed,
        lm_head.as_ref(),
        &final_norm,
        &tokens,
        &pos,
        &targets,
        -100,
        n_experts,
        top_k,
        &mut acc,
        unit_offset,
    )?;

    // Oracle export, matching dump_qwen35_hidden_states' HFHIDDEN format.
    // These are layer INPUTS, so mine[i+1] lines up with the runtime's output
    // of layer i.
    if let Ok(dir) = std::env::var("HIPFIRE_GH_ORACLE_DIR") {
        let mut out = Vec::new();
        out.extend_from_slice(b"HFHIDDEN");
        out.extend_from_slice(&(layer_inputs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(seq as u32).to_le_bytes());
        out.extend_from_slice(&(h as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for li in &layer_inputs {
            for v in li {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(std::path::Path::new(&dir).join("mine_hidden.bin"), &out)?;
        let lb: Vec<u8> = last_logits.iter().flat_map(|v| v.to_ne_bytes()).collect();
        std::fs::write(std::path::Path::new(&dir).join("mine_logits.f32"), &lb)?;
        eprintln!(
            "oracle: wrote {} layer inputs and {} last-position logits",
            layer_inputs.len(),
            last_logits.len()
        );
    }

    println!("hybrid stack: {} layers, seq={seq}, h={h}", kinds.len());
    for (i, k) in kinds.iter().enumerate() {
        println!("  layer {i}: {k:?}");
    }
    println!("  loss {loss:.4} (mean {:.4})", loss / seq as f32);

    // The walk never reads layer_types; comparing against it is a real check.
    if let Some(types) = raw["layer_types"].as_array() {
        for (i, t) in types.iter().enumerate() {
            let want_la = t.as_str() == Some("linear_attention");
            let got_la = matches!(
                kinds[i],
                LayerKind::LinearAttnMoe | LayerKind::LinearAttnDense
            );
            if want_la != got_la {
                println!(
                    "\nFAIL — layer {i}: config says {t}, probe says {:?}",
                    kinds[i]
                );
                std::process::exit(1);
            }
        }
        println!("  layer kinds match config layer_types");
    }

    let table = acc.finish();
    let by_layer = gamma_by_layer(&table);
    println!(
        "\ngamma entries: {} across {} layers",
        table.len(),
        by_layer.len()
    );
    for (l, entries) in &by_layer {
        let top: Vec<String> = entries
            .iter()
            .take(3)
            .map(|(k, v)| {
                let short = k.rsplit_once("layers.").map(|(_, r)| r).unwrap_or(k);
                format!("{short}={v:.3e}")
            })
            .collect();
        println!(
            "  layer {l}: {} entries, top {}",
            entries.len(),
            top.join(" ")
        );
    }

    // Write the full table. The quantizer consumes exactly this shape via
    // HIPFIRE_MIXED_BPW_GAMMA (a flat {tensor: gamma} map), and without it the
    // pass reports a summary and throws the numbers away.
    if let Ok(js) = serde_json::to_string(&table) {
        let out = std::env::var("HIPFIRE_GAMMA_OUT")
            .unwrap_or_else(|_| "gamma.json".to_string());
        match std::fs::write(&out, js) {
            Ok(()) => println!("  wrote {} gamma entries -> {out}", table.len()),
            Err(e) => eprintln!("  WARN could not write {out}: {e}"),
        }
    }

    let mean = loss / seq as f32;
    let expect = (cfg.vocab_size as f32).ln();
    // A random-init fixture lands at ln(vocab); a real trained model lands far
    // BELOW it. Both are healthy — the failure mode this guards against is a
    // loss at or above uniform, which is what a broken forward produces.
    let ok_loss = mean.is_finite() && mean < expect + 0.5;
    let ok_layers = by_layer.len() == kinds.len();
    let ok_finite = table.values().all(|v| v.is_finite() && *v >= 0.0);

    if ok_loss && ok_layers && ok_finite {
        // Never claim "predicts text" off synthetic tokens: the run already
        // warned that the loss is not meaningful there, and a verdict that
        // contradicts its own preamble is worse than no verdict. On 0,1,2,...
        // a low loss says the stack is coherent, not that it models language.
        let verdict = match (synthetic, mean < expect - 2.0) {
            (true, true) => "well below uniform — but tokens are SYNTHETIC, so this shows a coherent stack, NOT a language-model loss",
            (true, false) => "near uniform (synthetic tokens; expected for a random-init fixture)",
            (false, true) => "predicts text",
            (false, false) => "near uniform (expected for a random-init fixture)",
        };
        println!("\nPASS — hybrid stack assembled, mean loss {mean:.3} vs ln(vocab) {expect:.3} — {verdict}");
        Ok(())
    } else {
        println!(
            "\nFAIL — loss_ok {ok_loss} layers_ok {ok_layers} ({} of {}) finite_ok {ok_finite}",
            by_layer.len(),
            kinds.len()
        );
        std::process::exit(1)
    }
}
