// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-check the host MoE router against `moe_softmax_topk_renorm_k8_batched`
//! at the REAL expert count.
//!
//! `verify_moe_per_expert` proves the expert WEIGHTS load bit-exactly, and the
//! MoE forward has been checked on a tiny fixture at seq 1. Neither touches
//! routing at the size the 35B runs it: 256 experts, top-8. That matters
//! because routing is where a disagreement is invisible to every shape check —
//! pick a different 8 experts out of 256 and the output is a sum of the wrong
//! eight, which is orthogonal to the right answer rather than close to it.
//!
//! What this settles:
//!
//!   * **The kept set.** Whether both sides select the same 8 expert ids, and
//!     in the same slot order. Order matters because the gate weight rides
//!     along in the same slot.
//!   * **Renormalisation.** Whether the gates are raw softmax probabilities or
//!     renormalised to sum to 1 over the kept set. Both are plausible, both
//!     produce a working model, and the difference is a per-token scale on the
//!     whole MoE branch.
//!   * **Ties.** The host tie-breaks on the lower expert index; a kernel doing
//!     iterative max-and-mask need not.
//!
//! Uses the pair the inference path actually runs — `softmax_f32` then
//! `moe_topk_renorm_k8` — NOT the fused `moe_softmax_topk_renorm_k8`, which
//! `moe_decode.rs` documents as abandoned: it differed from softmax-then-divide
//! by 1 ULP per element, which compounded across 30+ MoE layers into a
//! structural attractor on A3B and 122B-A10B at MQ4.
//!
//! Run: cargo run --release -p hipfire-train --example verify_moe_router [seq] [n_exp]

use hipfire_rdna::{DType, Gpu};
use hipfire_train::ops::moe::route;

fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    let seq = *a.first().unwrap_or(&8);
    let n_exp = *a.get(1).unwrap_or(&256);
    const TOP_K: usize = 8; // the kernel's compile-time K

    let mut gpu = Gpu::init()?;
    let mut s = 0x4d0e_u64;
    let logits: Vec<f32> = (0..seq * n_exp).map(|_| 2.0 * lcg(&mut s)).collect();

    let (h_idx, h_gate) = route(&logits, seq, n_exp, TOP_K);

    // The runtime routes one token at a time here, so match that exactly:
    // softmax in place over the row, then top-k + renorm on the PROBS.
    // norm_topk_prob is absent from the 35B config and both sides default it
    // to true, so renorm is on.
    let mut g_idx = Vec::with_capacity(seq * TOP_K);
    let mut g_gate = Vec::with_capacity(seq * TOP_K);
    for t in 0..seq {
        let lt = gpu.upload_f32(&logits[t * n_exp..(t + 1) * n_exp], &[n_exp])?;
        let it = gpu.alloc_tensor(&[TOP_K], DType::F32)?;
        let wt = gpu.zeros(&[TOP_K], DType::F32)?;
        gpu.softmax_f32(&lt)?;
        gpu.moe_topk_renorm_k8(&lt, &it, &wt, n_exp, true)?;
        // topk_indices is an f32-typed tensor holding raw i32 bits.
        let mut raw = vec![0i32; TOP_K];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, TOP_K * 4) };
        gpu.hip.memcpy_dtoh(bytes, &it.buf)?;
        g_idx.extend(raw);
        g_gate.extend(gpu.download_f32(&wt)?);
        for x in [lt, it, wt] {
            gpu.free_tensor(x)?;
        }
    }

    println!("MoE router host vs kernel: seq={seq} n_experts={n_exp} top_k={TOP_K}");

    let mut idx_mismatch = 0usize;
    let mut set_mismatch = 0usize;
    let mut worst_w = 0.0f32;
    for t in 0..seq {
        let hs: Vec<i32> = (0..TOP_K).map(|j| h_idx[t * TOP_K + j] as i32).collect();
        let gs: Vec<i32> = (0..TOP_K).map(|j| g_idx[t * TOP_K + j]).collect();
        if hs != gs {
            idx_mismatch += 1;
            // Same experts in a different slot order is a much milder finding
            // than a different set, so separate the two.
            let (mut a, mut b) = (hs.clone(), gs.clone());
            a.sort_unstable();
            b.sort_unstable();
            if a != b {
                set_mismatch += 1;
                if set_mismatch <= 3 {
                    println!("  token {t}: host set {a:?} != kernel set {b:?}");
                }
            } else if idx_mismatch <= 3 {
                println!("  token {t}: same set, slot order differs {hs:?} vs {gs:?}");
            }
        }
        for j in 0..TOP_K {
            worst_w = worst_w.max((h_gate[t * TOP_K + j] - g_gate[t * TOP_K + j]).abs());
        }
    }
    // Gates must sum to 1 per token on both sides if renorm is really happening.
    let sum_h: f32 = (0..TOP_K).map(|j| h_gate[j]).sum();
    let sum_g: f32 = (0..TOP_K).map(|j| g_gate[j]).sum();
    println!("  token 0 gate sum: host {sum_h:.6} kernel {sum_g:.6}");
    println!("  tokens with differing slot order: {idx_mismatch}/{seq}");
    println!("  tokens with a differing EXPERT SET: {set_mismatch}/{seq}");
    println!("  worst |host gate - kernel gate| = {worst_w:.3e}");

    if set_mismatch == 0 && idx_mismatch == 0 && worst_w < 1e-5 {
        println!("\nPASS — the host router is the inference path's, at {n_exp} experts");
        Ok(())
    } else {
        println!("\nFAIL");
        std::process::exit(1)
    }
}
