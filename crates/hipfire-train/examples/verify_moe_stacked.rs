#![allow(clippy::needless_range_loop)]
// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Verify the stacked/fused MoE loader against real bytes.
//!
//! `qwen3_5_moe-tiny` is the only fixture with the layout the 35B uses — all
//! experts in one 3-D tensor and `gate_up_proj` fused as `(gate || up)`. It is
//! safetensors, and `import safetensors` implements 'zaya' alone, so there is
//! no `.hfq` build of it; `WeightSource` exists so this exercises the SAME
//! slicing code the artifact will rather than a parallel copy.
//!
//! What this proves and what it does not:
//!   * PROVES the index arithmetic — expert stride, the gate/up halving, and
//!     the `down_proj` orientation — by recomputing every element from the
//!     stacked tensor independently and comparing against what the loader
//!     uploaded and the GPU handed back. A wrong stride, a swapped half, or a
//!     transpose all fail here.
//!   * DOES NOT prove that gate is the first half rather than the second. That
//!     is a semantic claim about the checkpoint, and it rests on the layout
//!     comment in `qwen35/layout.rs` ("fused (gate || up)"). Only running the
//!     model against a reference would settle it, and no offline reference
//!     exists here. Both halves are the same shape, so nothing local can tell
//!     them apart — worth knowing if outputs ever look subtly wrong.
//!
//! Run: cargo run --release -p hipfire-train --example verify_moe_stacked

use hipfire_rdna::Gpu;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use hipfire_train::loader::{free_moe_layer_fp32, layer_is_moe, load_moe_layer_fp32, WeightSource};

const FIXTURE: &str = "/srv/hipfire/fixtures/qwen3_5_moe-tiny";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| FIXTURE.into());
    let path = std::path::Path::new(&dir);
    if !path.exists() {
        eprintln!("fixture {dir} not present — skipping");
        return Ok(());
    }
    let cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path.join("config.json"))?)?;
    let h = cfg["hidden_size"].as_u64().unwrap() as usize;
    let n_experts = cfg["num_experts"].as_u64().unwrap() as usize;

    let src = SafetensorsSource::open(path)?;
    let mut gpu = Gpu::init()?;

    assert!(
        layer_is_moe(&src, "model.", 0),
        "layer 0 should be detected as routed from the stacked tensor name"
    );

    let (layer, inter) = load_moe_layer_fp32(&mut gpu, &src, "model.", 0, h, n_experts)?;
    println!("loaded layer 0: h={h} experts={n_experts} inter={inter}");

    // Independent reference: re-slice the stacked tensors with index arithmetic
    // written here, not shared with the loader.
    let (gu_shape, gu) = src.fetch_f32("model.layers.0.mlp.experts.gate_up_proj")?;
    let (dn_shape, dn) = src.fetch_f32("model.layers.0.mlp.experts.down_proj")?;
    println!("  stacked gate_up {gu_shape:?}  down {dn_shape:?}");
    assert_eq!(gu_shape, vec![n_experts, 2 * inter, h]);
    assert_eq!(dn_shape, vec![n_experts, h, inter]);

    let mut worst = 0.0f32;
    for e in 0..n_experts {
        let (g, u, d) = &layer.experts[e];
        let (gh, uh, dh) = (
            gpu.download_f32(g)?,
            gpu.download_f32(u)?,
            gpu.download_f32(d)?,
        );
        assert_eq!(gh.len(), inter * h);
        assert_eq!(dh.len(), h * inter);
        for r in 0..inter {
            for c in 0..h {
                // gate: rows [0, inter) of expert e's slab; up: rows [inter, 2*inter)
                let want_g = gu[(e * 2 * inter + r) * h + c];
                let want_u = gu[(e * 2 * inter + inter + r) * h + c];
                worst = worst.max((gh[r * h + c] - want_g).abs());
                worst = worst.max((uh[r * h + c] - want_u).abs());
            }
        }
        for r in 0..h {
            for c in 0..inter {
                let want = dn[(e * h + r) * inter + c];
                worst = worst.max((dh[r * inter + c] - want).abs());
            }
        }
    }
    println!("  worst |loaded - independent reslice| over all {n_experts} experts: {worst:.3e}");

    let sh = layer
        .shared
        .as_ref()
        .expect("qwen3.5 MoE layer must have a shared expert");
    let sinter = cfg["shared_expert_intermediate_size"].as_u64().unwrap() as usize;
    assert_eq!(
        sh.inter, sinter,
        "shared intermediate from tensor vs config"
    );
    let sg = gpu.download_f32(&sh.scalar_gate)?;
    assert_eq!(sg.len(), h, "shared_expert_gate projects to ONE scalar");
    println!("  shared expert: inter={} scalar_gate=[1,{}]", sh.inter, h);

    // A zero or NaN slab would compare equal to a matching mis-slice, so check
    // the weights are actually live.
    let live = gpu.download_f32(&layer.experts[0].0)?;
    let mag = live.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(mag.is_finite() && mag > 0.0, "expert 0 gate is dead: {mag}");

    free_moe_layer_fp32(&mut gpu, layer)?;

    if worst == 0.0 {
        println!("\nPASS — stacked slicing is bit-exact against an independent reslice");
        Ok(())
    } else {
        println!("\nFAIL — {worst:.3e}");
        std::process::exit(1)
    }
}
