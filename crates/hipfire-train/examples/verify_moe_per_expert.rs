#![allow(clippy::needless_range_loop)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Verify the PER-EXPERT-FUSED expert layout against the stacked one.
//!
//! Three expert layouts exist in the wild and this loader handles all three,
//! but only two had ever been checked. `verify_moe_stacked` covers the
//! stacked-3D form against an independent reslice; the per-expert-unfused form
//! is trivially named. The third — `experts.N.gate_up_proj [2*inter, h]`, what
//! `hipfire-quantize` emits for a routed MoE and what the 35B artifact
//! actually carries — was only ever checked against its own shapes.
//!
//! The check is a cross-load: the same fixture, read once through the STACKED
//! path (from the safetensors source) and once through the PER-EXPERT-FUSED
//! path (from a bf16 `.hfq` the quantizer produced from it). bf16 is lossless
//! here, so the two must agree bit-for-bit. Anything else — a wrong expert
//! stride, the halves swapped, gate and up transposed — shows up immediately,
//! and none of it would disturb a single shape.
//!
//! Run: cargo run --release -p hipfire-train --example verify_moe_per_expert

use hipfire_rdna::Gpu;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use hipfire_train::loader::{free_moe_layer_fp32, load_moe_layer_fp32};

const FIXTURE: &str = "/srv/hipfire/fixtures/qwen3_5_moe-tiny";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| FIXTURE.into());
    let hfq_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| format!("{}/../gu/bf16.hfq", std::env::temp_dir().display()));
    let (dp, hp) = (std::path::Path::new(&dir), std::path::Path::new(&hfq_path));
    if !dp.exists() || !hp.exists() {
        eprintln!("fixture or per-expert .hfq missing — skipping");
        eprintln!("  build with: hipfire-quantize --input {dir} --output <out.hfq> --format bf16");
        return Ok(());
    }
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dp.join("config.json"))?)?;
    let h = cfg["hidden_size"].as_u64().unwrap() as usize;
    let ne = cfg["num_experts"].as_u64().unwrap() as usize;

    let src = SafetensorsSource::open(dp)?;
    let hfq = HfqFile::open(hp)?;
    let mut gpu = Gpu::init()?;

    let (stacked, inter_a) = load_moe_layer_fp32(&mut gpu, &src, "model.", 0, h, ne)?;
    let (fused, inter_b) = load_moe_layer_fp32(&mut gpu, &hfq, "model.", 0, h, ne)?;
    println!("per-expert-fused vs stacked: experts={ne} h={h} inter={inter_a}/{inter_b}");
    assert_eq!(inter_a, inter_b, "intermediate width disagrees");

    let mut worst = 0.0f32;
    let mut which = String::new();
    for e in 0..ne {
        for (name, a, b) in [
            ("gate", &stacked.experts[e].0, &fused.experts[e].0),
            ("up", &stacked.experts[e].1, &fused.experts[e].1),
            ("down", &stacked.experts[e].2, &fused.experts[e].2),
        ] {
            let (x, y) = (gpu.download_f32(a)?, gpu.download_f32(b)?);
            assert_eq!(x.len(), y.len(), "expert {e} {name}: length differs");
            for i in 0..x.len() {
                let d = (x[i] - y[i]).abs();
                if d > worst {
                    worst = d;
                    which = format!("expert {e} {name}[{i}]");
                }
            }
        }
    }
    println!("  worst |stacked - per_expert_fused| = {worst:.3e}{}", {
        if which.is_empty() {
            String::new()
        } else {
            format!("  at {which}")
        }
    });

    // A shared branch that failed to load would make the comparison vacuous.
    let sh = stacked.shared.as_ref().expect("stacked shared expert");
    let fh = fused.shared.as_ref().expect("fused shared expert");
    assert_eq!(sh.inter, fh.inter, "shared intermediate disagrees");
    let (sg, fg) = (
        gpu.download_f32(&sh.scalar_gate)?,
        gpu.download_f32(&fh.scalar_gate)?,
    );
    let sworst = sg
        .iter()
        .zip(fg.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!(
        "  shared expert scalar gate: worst {sworst:.3e} (inter {})",
        sh.inter
    );

    free_moe_layer_fp32(&mut gpu, stacked)?;
    free_moe_layer_fp32(&mut gpu, fused)?;

    if worst == 0.0 && sworst == 0.0 {
        println!("\nPASS — the per-expert-fused split is bit-identical to the stacked one");
        Ok(())
    } else {
        println!("\nFAIL");
        std::process::exit(1)
    }
}
