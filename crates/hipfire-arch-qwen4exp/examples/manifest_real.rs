//! Validate the qwen4_exp tensor manifest against a RESTORED checkpoint directory.
//!
//! ⚠️ Does not run on the shipped Qwen3.8-Flash-Next today: `HfqFile::from_safetensors`
//! refuses it, because the n-gram derived tables (`layer_multipliers`,
//! `ngram_heads_offsets`, `ngram_heads_vocab_sizes`) are **I64** and that reader
//! handles bf16/f16/f32 only. `hipfire-quantize` reads the same directory happily,
//! so the two readers of one format disagree — see BUGS.md. This example is what
//! found that, and it reports the gap rather than panicking on it.
use hipfire_arch_qwen4exp::{LayerType, Qwen4ExpConfig};
use hipfire_arch_qwen4exp_spec::manifest::{qwen4exp_manifest, Qwen4ExpGeometry};
use std::path::Path;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: <hf_dir>");
    let hfq = match hipfire_runtime::hfq::HfqFile::from_safetensors(Path::new(&dir)) {
        Ok(f) => f,
        Err(e) => {
            println!("manifest_real: cannot open {dir} through from_safetensors:\n  {e}");
            println!("manifest_real: SKIPPED (see BUGS.md — I64 tensors are refused)");
            return;
        }
    };
    let cfg_json = std::fs::read(Path::new(&dir).join("config.json")).unwrap();
    let cfg = Qwen4ExpConfig::from_slice(&cfg_json).expect("parse real config");

    let layer_types: Vec<String> = cfg
        .layer_types
        .iter()
        .map(|t| match t {
            LayerType::LinearAttention => "linear_attention".to_string(),
            LayerType::SparseAttention => "qwen_sparse_attention".to_string(),
        })
        .collect();
    let geom = Qwen4ExpGeometry {
        layers: cfg.layers,
        experts: cfg.moe.num_experts,
        ngram_layer: cfg.ngram.as_ref().map(|n| n.layer_idx),
        ngram_shards: cfg.ngram.as_ref().map(|n| n.shards).unwrap_or(1),
        mtp_layers: cfg.mtp_layers,
        vision_blocks: cfg.vision.as_ref().map(|v| v.depth).unwrap_or(0),
    };
    let names: Vec<&str> = hfq.tensors().iter().map(|t| t.name.as_str()).collect();
    println!("restored checkpoint carries {} tensors", names.len());
    let report = qwen4exp_manifest(&layer_types, geom).validate(names);
    if report.is_ok() {
        println!("manifest_real: OK — nothing missing, nothing unclaimed");
    } else {
        println!("{}", report.render("qwen4_exp"));
        std::process::exit(1);
    }
}
