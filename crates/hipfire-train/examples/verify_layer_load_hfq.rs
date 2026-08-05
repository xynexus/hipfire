// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Load one attention layer from a real `.hfq` and check what the loader
//! DERIVED against what the artifact actually contains.
//!
//! The hybrid assembly is verified end to end on `qwen3_5_moe-tiny`, but that
//! fixture is safetensors and random-init. This exercises the other
//! `WeightSource` — a real bf16 artifact, decoded through the hfq path — and
//! the two things the loader infers rather than reads from config:
//!
//!   * QK-norm presence, probed per layer.
//!   * `attn_out_gate`, derived from `q_proj`'s row count. The runtime derives
//!     it the same way (`infer_attn_output_gate_from_hfq`) because some Qwen3
//!     artifacts set the config flag while storing plain Q — so a config-driven
//!     loader would double-count the width and mis-slice every head.
//!
//! Both are cross-checked here against the tensor table directly, so the
//! derivation is compared with the artifact rather than with itself.
//!
//! Run: cargo run --release -p hipfire-train --example verify_layer_load_hfq [artifact.hfq]

use hipfire_model::ModelSource;
use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use hipfire_train::config::LlamaConfig;
use hipfire_train::loader::{free_llama_layer_fp32, load_llama_layer_fp32_hfq_pfx, WeightSource};

const DEFAULT: &str = "/srv/hipfire/models/Qwen3.5-122B-A10B-DFlash--bf16.hfq";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT.into());
    let p = std::path::Path::new(&path);
    if !p.exists() {
        eprintln!("artifact {path} not present — skipping");
        return Ok(());
    }
    let hfq = HfqFile::open(p)?;
    let cfg = LlamaConfig::from_hfq_metadata(hfq.metadata_json())?;
    let mut gpu = Gpu::init()?;

    // Tensor names here have no `model.` prefix, unlike the HF export.
    let prefix = if hfq
        .find_tensor_info("model.layers.0.self_attn.q_proj.weight")
        .is_some()
    {
        "model."
    } else {
        ""
    };
    let name = format!("{prefix}layers.0.self_attn.q_proj.weight");
    let q_rows = hfq.shape_of(&name).ok_or("no layer 0 q_proj")?[0];
    let has_qn = hfq.has(&format!("{prefix}layers.0.self_attn.q_norm.weight"));

    println!("{}", path.rsplit('/').next().unwrap_or(&path));
    println!(
        "  h={} heads={} kv={} head_dim={} q_dim={}",
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.q_dim()
    );
    println!("  q_proj rows {q_rows}, q_norm present: {has_qn}");

    let l = load_llama_layer_fp32_hfq_pfx(&mut gpu, &hfq, prefix, &cfg, 0, true)?;
    let want_gate = q_rows == 2 * cfg.q_dim();
    let ok_gate = l.attn_out_gate == want_gate;
    let ok_qn = l.q_norm.is_some() == has_qn && l.k_norm.is_some() == has_qn;

    // A layer that loaded all-zeros would satisfy every shape check above.
    let qw = gpu.download_f32(&l.q_proj)?;
    let mag = qw.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let finite = qw.iter().all(|v| v.is_finite());
    println!(
        "  derived attn_out_gate={} (expected {want_gate}), qk_norm={} ",
        l.attn_out_gate,
        l.q_norm.is_some()
    );
    println!("  q_proj max|w| {mag:.4} finite {finite}");

    free_llama_layer_fp32(&mut gpu, l)?;

    if ok_gate && ok_qn && finite && mag > 0.0 {
        println!("\nPASS — hfq path loads a real bf16 layer and derives both flags correctly");
        Ok(())
    } else {
        println!("\nFAIL — gate_ok {ok_gate} qknorm_ok {ok_qn} finite {finite} mag {mag}");
        std::process::exit(1)
    }
}
