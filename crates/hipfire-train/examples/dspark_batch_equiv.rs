#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
//! Phase A equivalence gate: the window-BATCHED drafter body must reproduce, per
//! window, exactly what running each window ALONE produces.
//!
//! Because the f32 training path is deterministic (rmsnorm dw de-atomic'd), this
//! is a *bit-exact* test where the math allows it:
//!
//!   * forward `x_head` — every forward op is per-output-row independent
//!     (rmsnorm / rope / linear `Y=XWᵀ`) and attention is per-window, so the
//!     batched `x_head[i-slice]` must be BYTE-IDENTICAL to window `i` run alone.
//!   * `d_main_hidden` — `dX = dY·W` is per-output-row independent → BYTE-IDENTICAL
//!     per window slice.
//!   * param weight-grads — the batched dW contracts the full `wb*block` (resp.
//!     `wb*ctx_len`) rows in ONE reduction, whereas the reference sums per-window
//!     dW. Those differ ONLY by floating-point summation order, so at `wb>1` they
//!     match to a tight tolerance (not bit-exact); at `wb==1` they ARE bit-exact.
//!
//! A batching bug (wrong offset, cross-window attention leak, mis-tiled RoPE
//! positions) breaks the bit-exact checks immediately.
//!
//!   source ./scripts/rocm-env.sh && export ROCM_PATH=/opt/rocm
//!   hipfire lock acquire "dspark-batch-equiv"
//!   cargo run -p hipfire-train --release --example dspark_batch_equiv
//!   hipfire lock release

use hipfire_rdna::{Gpu, GpuTensor, HipResult};
use hipfire_train::dspark_drafter::{
    dspark_drafter_backward, dspark_drafter_forward_train, free_dspark_drafter_acts,
    free_dspark_drafter_grads, DsparkDrafterConfig, DsparkDrafterWeights, DsparkLayerWeights,
};

const H: usize = 16;
const NL: usize = 3;
const NH: usize = 2;
const NKV: usize = 1;
const HD: usize = 8;
const INTER: usize = 32;
const BLOCK: usize = 5;
const CTX: usize = 6;
const NT: usize = 2;
const QD: usize = NH * HD;
const KVD: usize = NKV * HD;
const FIN: usize = NT * H;

fn cfg() -> DsparkDrafterConfig {
    DsparkDrafterConfig {
        h: H,
        n_layers: NL,
        n_heads: NH,
        n_kv: NKV,
        head_dim: HD,
        inter: INTER,
        rope_base: 10000.0,
        eps: 1e-6,
        block_size: BLOCK,
        n_targets: NT,
        qk_norm: true,
        vocab: 32,
    }
}

fn seeded(n: usize, seed: u64, scale: f32, off: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as f32) / (1u64 << 31) as f32 - 1.0) * scale + off
        })
        .collect()
}

fn build(gpu: &mut Gpu) -> HipResult<DsparkDrafterWeights> {
    let mut seed = 1u64;
    let mut lin = |gpu: &mut Gpu, n: usize, scale: f32| -> HipResult<GpuTensor> {
        seed += 1;
        gpu.upload_f32(&seeded(n, seed, scale, 0.0), &[n])
    };
    let fc = lin(gpu, H * FIN, 0.06)?;
    let hidden_norm = lin(gpu, H, 0.05)?;
    let mut layers = Vec::with_capacity(NL);
    for _ in 0..NL {
        layers.push(DsparkLayerWeights {
            input_ln: lin(gpu, H, 0.05)?,
            wq: lin(gpu, QD * H, 0.06)?,
            wk: lin(gpu, KVD * H, 0.06)?,
            wv: lin(gpu, KVD * H, 0.06)?,
            wo: lin(gpu, H * QD, 0.06)?,
            q_norm: lin(gpu, HD, 0.05)?,
            k_norm: lin(gpu, HD, 0.05)?,
            post_ln: lin(gpu, H, 0.05)?,
            wgate: lin(gpu, INTER * H, 0.05)?,
            wup: lin(gpu, INTER * H, 0.05)?,
            wdown: lin(gpu, H * INTER, 0.05)?,
        });
    }
    let out_norm = lin(gpu, H, 0.05)?;
    Ok(DsparkDrafterWeights {
        fc,
        hidden_norm,
        layers,
        out_norm,
    })
}

fn param_names() -> Vec<String> {
    let mut v = vec!["fc".to_string(), "hidden_norm".to_string()];
    for li in 0..NL {
        for p in [
            "wq", "wk", "wv", "wo", "wgate", "wup", "wdown", "input_ln", "post_ln", "q_norm",
            "k_norm",
        ] {
            v.push(format!("L{li}.{p}"));
        }
    }
    v.push("out_norm".to_string());
    v
}

/// Max abs and max rel diff between two host vectors.
fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut ma = 0.0f32;
    let mut mr = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        ma = ma.max(d);
        let den = x.abs().max(y.abs());
        if den > 1e-12 {
            mr = mr.max(d / den);
        }
    }
    (ma, mr)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}", gpu.arch);
    let c = cfg();
    let ctx_pos: Vec<f32> = (0..CTX).map(|t| t as f32).collect();
    let blk_pos: Vec<f32> = (0..BLOCK).map(|t| (CTX + t) as f32).collect();
    let weights = build(&mut gpu)?;
    let names = param_names();

    // Run the equivalence check at a few window counts. wb==1 must be bit-exact
    // for EVERYTHING (single-window reference); wb>1 is bit-exact for x_head and
    // d_main_hidden, tight-tol for the reassociated weight grads.
    let mut all_ok = true;
    for &wb in &[1usize, 2, 4] {
        // Distinct per-window data so any cross-window leak shows up.
        let mh: Vec<Vec<f32>> = (0..wb)
            .map(|w| seeded(CTX * FIN, 100 + w as u64, 0.4, -0.05))
            .collect();
        let be: Vec<Vec<f32>> = (0..wb)
            .map(|w| seeded(BLOCK * H, 200 + w as u64, 0.4, -0.1))
            .collect();
        let gh: Vec<Vec<f32>> = (0..wb)
            .map(|w| seeded(BLOCK * H, 300 + w as u64, 0.3, 0.0))
            .collect();

        // ── Reference: each window alone (n_win=1); sum the weight grads. ──────
        let nparam = weights.params().len();
        let mut ref_xhead: Vec<Vec<f32>> = Vec::with_capacity(wb); // per window
        let mut ref_dmh: Vec<Vec<f32>> = Vec::with_capacity(wb); // per window
        let mut ref_grad_sum: Vec<Vec<f32>> = vec![Vec::new(); nparam];
        for w in 0..wb {
            let mhw = gpu.upload_f32(&mh[w], &[CTX * FIN])?;
            let bew = gpu.upload_f32(&be[w], &[BLOCK * H])?;
            let acts = dspark_drafter_forward_train(
                &mut gpu, &weights, &c, &mhw, &bew, &ctx_pos, &blk_pos, None, 1,
            )?;
            ref_xhead.push(gpu.download_f32(acts.x_head())?);
            let dxh = gpu.upload_f32(&gh[w], &[BLOCK * H])?;
            let grads = dspark_drafter_backward(&mut gpu, &weights, &c, &mhw, &acts, &dxh, 1)?;
            ref_dmh.push(gpu.download_f32(&grads.d_main_hidden)?);
            let gf = grads.flat();
            for p in 0..nparam {
                let hv = gpu.download_f32(gf[p])?;
                if ref_grad_sum[p].is_empty() {
                    ref_grad_sum[p] = hv;
                } else {
                    for (a, b) in ref_grad_sum[p].iter_mut().zip(hv.iter()) {
                        *a += *b;
                    }
                }
            }
            drop(gf);
            free_dspark_drafter_grads(&mut gpu, grads)?;
            free_dspark_drafter_acts(&mut gpu, acts)?;
            gpu.free_tensor(mhw)?;
            gpu.free_tensor(dxh)?;
            gpu.free_tensor(bew)?;
        }

        // ── Batched: all wb windows in one call (n_win=wb). ───────────────────
        let mut mh_cat = Vec::new();
        let mut be_cat = Vec::new();
        let mut gh_cat = Vec::new();
        for w in 0..wb {
            mh_cat.extend_from_slice(&mh[w]);
            be_cat.extend_from_slice(&be[w]);
            gh_cat.extend_from_slice(&gh[w]);
        }
        let mhb = gpu.upload_f32(&mh_cat, &[wb * CTX * FIN])?;
        let beb = gpu.upload_f32(&be_cat, &[wb * BLOCK * H])?;
        let acts = dspark_drafter_forward_train(
            &mut gpu, &weights, &c, &mhb, &beb, &ctx_pos, &blk_pos, None, wb,
        )?;
        let bat_xhead = gpu.download_f32(acts.x_head())?;
        let dxhb = gpu.upload_f32(&gh_cat, &[wb * BLOCK * H])?;
        let grads = dspark_drafter_backward(&mut gpu, &weights, &c, &mhb, &acts, &dxhb, wb)?;
        let bat_dmh = gpu.download_f32(&grads.d_main_hidden)?;
        let gf = grads.flat();
        let bat_grads: Vec<Vec<f32>> = (0..nparam)
            .map(|p| gpu.download_f32(gf[p]))
            .collect::<HipResult<_>>()?;
        drop(gf);
        free_dspark_drafter_grads(&mut gpu, grads)?;
        free_dspark_drafter_acts(&mut gpu, acts)?;
        gpu.free_tensor(mhb)?;
        gpu.free_tensor(beb)?;
        gpu.free_tensor(dxhb)?;

        // ── Compare. ───────────────────────────────────────────────────────────
        println!("\n=== wb = {wb} ===");
        // x_head per window must be BIT-EXACT.
        let mut xh_max = 0.0f32;
        for w in 0..wb {
            let slice = &bat_xhead[w * BLOCK * H..(w + 1) * BLOCK * H];
            let (ma, _) = diff(slice, &ref_xhead[w]);
            xh_max = xh_max.max(ma);
        }
        let xh_ok = xh_max == 0.0;
        all_ok &= xh_ok;
        println!(
            "  x_head        max|Δ| = {xh_max:.3e}   {}  (must be 0)",
            if xh_ok { "OK" } else { "XX" }
        );

        // d_main_hidden per window must be BIT-EXACT.
        let mut dmh_max = 0.0f32;
        for w in 0..wb {
            let slice = &bat_dmh[w * CTX * FIN..(w + 1) * CTX * FIN];
            let (ma, _) = diff(slice, &ref_dmh[w]);
            dmh_max = dmh_max.max(ma);
        }
        let dmh_ok = dmh_max == 0.0;
        all_ok &= dmh_ok;
        println!(
            "  d_main_hidden max|Δ| = {dmh_max:.3e}   {}  (must be 0)",
            if dmh_ok { "OK" } else { "XX" }
        );

        // Weight grads: bit-exact at wb==1, tight-tol (reassociation) at wb>1.
        let rtol = if wb == 1 { 0.0 } else { 2e-4 };
        let atol = if wb == 1 { 0.0 } else { 1e-5 };
        let mut worst_p = 0usize;
        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut grads_ok = true;
        for p in 0..nparam {
            let (ma, mr) = diff(&bat_grads[p], &ref_grad_sum[p]);
            if mr > worst_rel {
                worst_rel = mr;
                worst_abs = ma;
                worst_p = p;
            }
            let ok = ma
                <= atol
                    + rtol * {
                        // magnitude reference: max|ref| element
                        ref_grad_sum[p]
                            .iter()
                            .fold(0.0f32, |m, v| m.max(v.abs()))
                            .max(1e-6)
                    };
            grads_ok &= ok;
        }
        all_ok &= grads_ok;
        println!(
            "  weight grads  worst rel = {worst_rel:.3e} (abs {worst_abs:.3e}) at {}  {}  (tol rel {rtol:.0e})",
            names[worst_p],
            if grads_ok { "OK" } else { "XX" }
        );
    }

    if all_ok {
        println!("\n  PASS — batched body is per-window equivalent to the looped path");
        Ok(())
    } else {
        Err("DSpark batch-equivalence FAILED".into())
    }
}
