// SPDX-License-Identifier: Apache-2.0
//! Numerical parity for a WMMA (bf16x2) BACKWARD reformulated as NT via
//! `transpose_f32` + `gemm_bf16x2_train_nt` (+ `add_inplace_f32` for the
//! accumulate fan-in), vs the scalar f32 reference backward `gemm_f32_train` /
//! `gemm_f32_train_accum`.
//!
//! This is the blind spot both the gradcheck (toy dims → only the LDS-free GEMM,
//! no accumulate at scale) and the forward parity (GEMM output on clean tensors,
//! not the transpose/accumulate wrapper) miss — the §3.3 backward dead-end. It
//! exercises the backward mappings at TRAINING dims: the gfx1151 m128 LDS variant
//! (dX at M>=16), the LDS-free variant (dX at M<16), the transpose-of-`dY`/`X`
//! for dW, `sub_offset` views, and the scratch+add accumulate.
//!
//! Reference math (kernels/src/gemm_f32_train.hip), X:[M,K] W:[N,K] dY:[M,N]:
//!   dX[M,K]  = dY·W    contract N   (NT: dX = dY · (Wᵀ)ᵀ, Wᵀ=[K,N])
//!   dW[N,K]  = dYᵀ·X   contract M   (NT: dW = dYᵀ · (Xᵀ)ᵀ, dYᵀ=[N,M] Xᵀ=[K,M])
//! gemm_bf16x2_train_nt(a, b, c, m, k, n) computes c[m,n] = Σ_k a[m,k]·b[n,k].

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

fn rand_vec(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / (1u64 << 31) as f32; // ~[0,2)
            (u - 1.0) * scale
        })
        .collect()
}

/// Per-tensor power-of-two scale bringing `max|t|` to ~2^14 (f16s path).
fn pow2_scale(gpu: &mut Gpu, t: &GpuTensor) -> HipResult<f32> {
    let a = gpu.abs_max_f32(t)?;
    Ok(if a > 0.0 {
        (16384.0_f32 / a).log2().floor().exp2()
    } else {
        1.0
    })
}

/// cosine + max-rel-err of `cand` vs `refv`.
fn metrics(refv: &[f32], cand: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut max_rel) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for j in 0..refv.len() {
        let (av, bv) = (refv[j] as f64, cand[j] as f64);
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
        max_rel = max_rel.max((av - bv).abs() / av.abs().max(1e-6));
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-30), max_rel)
}

fn main() -> HipResult<()> {
    let mut gpu = Gpu::init().expect("GPU init");
    let mut worst_cos = 1.0f64;
    let mut report = |name: &str, cos: f64, rel: f64| {
        worst_cos = worst_cos.min(cos);
        let flag = if cos > 0.999 { "ok " } else { "FAIL" };
        println!("  [{flag}] {name:<28} cos={cos:.7}  max_rel={rel:.2e}");
    };

    // (M tokens, K contract/in, N out). M>=16 → dX hits the gfx1151 m128 LDS
    // variant; M<16 → LDS-free. Includes the batched lm-head (huge N).
    let shapes = [
        (56usize, 2560usize, 2560usize), // batched proj  (dX m128)
        (7, 2560, 10240),                // block MLP gate/up (dX LDS-free)
        (56, 2560, 262208),              // batched lm-head (dX m128, huge N)
    ];

    for (i, &(m, k, n)) in shapes.iter().enumerate() {
        println!("shape M={m} K={k} N={n}:");
        let xh = rand_vec(m * k, 0x100 + i as u64, 1.0 / (k as f32).sqrt());
        let wh = rand_vec(n * k, 0x200 + i as u64, 1.0 / (k as f32).sqrt());
        let dyh = rand_vec(m * n, 0x300 + i as u64, 1.0 / (n as f32).sqrt());
        let x = gpu.upload_f32(&xh, &[m, k])?;
        let w = gpu.upload_f32(&wh, &[n, k])?;
        let dy = gpu.upload_f32(&dyh, &[m, n])?;

        // ---- reference backward (f32) ----
        let dx_ref = gpu.zeros(&[m * k], DType::F32)?;
        gpu.gemm_f32_train(&dy, &w, &dx_ref, m, k, n, n, k, false, false)?;
        let dw_ref = gpu.zeros(&[n * k], DType::F32)?;
        gpu.gemm_f32_train(&dy, &x, &dw_ref, n, k, m, n, k, true, false)?;

        // ---- WMMA backward (bf16x2 NT + transpose) ----
        // dX = NT(dY, Wᵀ): Wᵀ = transpose(w[N,K]) → [K,N]; c[m,n=K] = Σ_{k=N}.
        let wt = gpu.zeros(&[k * n], DType::F32)?;
        gpu.transpose_f32(&w, &wt, n, k)?; // w is [N,K]: rows=N cols=K → [K,N]
        let dx_wmma = gpu.zeros(&[m * k], DType::F32)?;
        gpu.gemm_bf16x2_train_nt(&dy, &wt, &dx_wmma, m, n, k)?;

        // dW = NT(dYᵀ, Xᵀ): dYᵀ=transpose(dy)[N,M], Xᵀ=transpose(x)[K,M].
        let dyt = gpu.zeros(&[n * m], DType::F32)?;
        gpu.transpose_f32(&dy, &dyt, m, n)?; // dy [M,N] → [N,M]
        let xt = gpu.zeros(&[k * m], DType::F32)?;
        gpu.transpose_f32(&x, &xt, m, k)?; // x [M,K] → [K,M]
        let dw_wmma = gpu.zeros(&[n * k], DType::F32)?;
        gpu.gemm_bf16x2_train_nt(&dyt, &xt, &dw_wmma, n, m, k)?;

        let (cx, rx) = metrics(&gpu.download_f32(&dx_ref)?, &gpu.download_f32(&dx_wmma)?);
        report("dX bf16x2", cx, rx);
        let (cw, rw) = metrics(&gpu.download_f32(&dw_ref)?, &gpu.download_f32(&dw_wmma)?);
        report("dW bf16x2", cw, rw);

        // Single-pass bf16 (8-bit) — the cheap candidate. Non-gating; reported to
        // show the mantissa gap vs bf16x2's ~16 bits on the same backward shapes.
        let dx_bf16 = gpu.zeros(&[m * k], DType::F32)?;
        gpu.gemm_bf16c_train_nt(&dy, &wt, &dx_bf16, m, n, k)?;
        let dw_bf16 = gpu.zeros(&[n * k], DType::F32)?;
        gpu.gemm_bf16c_train_nt(&dyt, &xt, &dw_bf16, n, m, k)?;
        let (cxb, rxb) = metrics(&gpu.download_f32(&dx_ref)?, &gpu.download_f32(&dx_bf16)?);
        report("dX bf16   (1-pass)", cxb, rxb);
        let (cwb, rwb) = metrics(&gpu.download_f32(&dw_ref)?, &gpu.download_f32(&dw_bf16)?);
        report("dW bf16   (1-pass)", cwb, rwb);

        // Single-pass fp16 (10-bit mantissa, 5-bit exponent). More mantissa than
        // bf16 but a 30-binade range — watch for overflow (inf → cos NaN) on these
        // random inputs vs the real training operands.
        let dx_f16 = gpu.zeros(&[m * k], DType::F32)?;
        gpu.gemm_f16c_train_nt(&dy, &wt, &dx_f16, m, n, k)?;
        let dw_f16 = gpu.zeros(&[n * k], DType::F32)?;
        gpu.gemm_f16c_train_nt(&dyt, &xt, &dw_f16, n, m, k)?;
        let (cxf, rxf) = metrics(&gpu.download_f32(&dx_ref)?, &gpu.download_f32(&dx_f16)?);
        report("dX f16    (1-pass)", cxf, rxf);
        let (cwf, rwf) = metrics(&gpu.download_f32(&dw_ref)?, &gpu.download_f32(&dw_f16)?);
        report("dW f16    (1-pass)", cwf, rwf);

        // Scaled f16 (per-tensor pow2 scale into range) — the target format.
        let (s_dy, s_wt) = (pow2_scale(&mut gpu, &dy)?, pow2_scale(&mut gpu, &wt)?);
        let dx_f16s = gpu.zeros(&[m * k], DType::F32)?;
        gpu.gemm_f16s_train_nt(&dy, &wt, &dx_f16s, m, n, k, s_dy, s_wt)?;
        let (s_dyt, s_xt) = (pow2_scale(&mut gpu, &dyt)?, pow2_scale(&mut gpu, &xt)?);
        let dw_f16s = gpu.zeros(&[n * k], DType::F32)?;
        gpu.gemm_f16s_train_nt(&dyt, &xt, &dw_f16s, n, m, k, s_dyt, s_xt)?;
        let (cxs, rxs) = metrics(&gpu.download_f32(&dx_ref)?, &gpu.download_f32(&dx_f16s)?);
        report("dX f16s   (1-pass,scaled)", cxs, rxs);
        let (cws, rws) = metrics(&gpu.download_f32(&dw_ref)?, &gpu.download_f32(&dw_f16s)?);
        report("dW f16s   (1-pass,scaled)", cws, rws);

        // Phase C2: dedicated NN/TN kernels — NO transpose pass (read strided).
        let s_w = pow2_scale(&mut gpu, &w)?;
        let dx_nn = gpu.zeros(&[m * k], DType::F32)?;
        gpu.gemm_f16s_nn_train(&dy, &w, &dx_nn, m, n, k, s_dy, s_w)?;
        let (cxn, rxn) = metrics(&gpu.download_f32(&dx_ref)?, &gpu.download_f32(&dx_nn)?);
        report("dX f16s NN (C2,no-transpose)", cxn, rxn);
        let s_x = pow2_scale(&mut gpu, &x)?;
        let dw_tn = gpu.zeros(&[n * k], DType::F32)?;
        gpu.gemm_f16s_tn_train(&dy, &x, &dw_tn, m, n, k, s_dy, s_x)?;
        let (cwt, rwt) = metrics(&gpu.download_f32(&dw_ref)?, &gpu.download_f32(&dw_tn)?);
        report("dW f16s TN (C2,no-transpose)", cwt, rwt);
        gpu.free_tensor(dx_nn)?;
        gpu.free_tensor(dw_tn)?;

        // ---- accumulate fan-in: dX += dY·W onto a pre-seeded partial ----
        let seed = rand_vec(m * k, 0x400 + i as u64, 1.0 / (k as f32).sqrt());
        let dx_acc_ref = gpu.upload_f32(&seed, &[m, k])?;
        gpu.gemm_f32_train_accum(&dy, &w, &dx_acc_ref, m, k, n, n, k, false, false, 1.0)?;
        let dx_acc_wmma = gpu.upload_f32(&seed, &[m, k])?;
        let scratch = gpu.zeros(&[m * k], DType::F32)?;
        gpu.gemm_bf16x2_train_nt(&dy, &wt, &scratch, m, n, k)?;
        gpu.add_inplace_f32(&dx_acc_wmma, &scratch)?;
        let (ca, ra) = metrics(
            &gpu.download_f32(&dx_acc_ref)?,
            &gpu.download_f32(&dx_acc_wmma)?,
        );
        report("dX += dY·W (accum)", ca, ra);

        // ---- sub_offset view: dX from a dY that is a slice of a padded parent ----
        // Confirms views flow correctly through transpose + m128/LDS-free GEMM.
        let pad = 37usize;
        let mut padded = vec![0.0f32; pad + m * n];
        padded[pad..].copy_from_slice(&dyh);
        let dy_parent = gpu.upload_f32(&padded, &[pad + m * n])?;
        let dy_view: GpuTensor = dy_parent.sub_offset(pad, m * n);
        let dx_view_wmma = gpu.zeros(&[m * k], DType::F32)?;
        gpu.gemm_bf16x2_train_nt(&dy_view, &wt, &dx_view_wmma, m, n, k)?;
        let (cv, rv) = metrics(&gpu.download_f32(&dx_ref)?, &gpu.download_f32(&dx_view_wmma)?);
        report("dX from sub_offset dY", cv, rv);

        for t in [
            x, w, dy, dx_ref, dw_ref, wt, dx_wmma, dyt, xt, dw_wmma, dx_bf16, dw_bf16, dx_f16,
            dw_f16, dx_f16s, dw_f16s, dx_acc_ref, dx_acc_wmma, scratch, dy_parent, dx_view_wmma,
        ] {
            gpu.free_tensor(t)?;
        }
    }

    if worst_cos > 0.999 {
        println!("\nBACKWARD PARITY OK — worst cos {worst_cos:.6} (> 0.999)");
        Ok(())
    } else {
        println!("\nBACKWARD PARITY FAIL — worst cos {worst_cos:.6} (<= 0.999)");
        std::process::exit(1);
    }
}
