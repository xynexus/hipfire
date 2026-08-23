// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Differential test for the COMPACT-RESIDENT indexed MoE GEMVs against the
//! expanded-Oq8 kernels they replace.
//!
//! Routed experts were the last tensors still unpacked to one int8 per weight at
//! load, because these three GEMVs had no compact sibling. That expansion costs
//! 1.80x, and compounds with the driver's 2 MiB GTT rounding into 3.5x -- the
//! reason a 63.9 GiB 122B occupies ~137 GiB resident and will not load.
//!
//! Three ways, because two cannot separate "kernel is wrong" from "oracle is
//! wrong":
//!
//!   oracle  -- CPU f64 dot over the decoded logical weights. Ground truth.
//!   oq8     -- host-expand with `oqplus_compact_to_moe_oq8_blocks` (byte for
//!              byte what `load_moe_expert` does today) then run the shipping
//!              `gemv_oq8g256_moe_*` kernel. The path being replaced.
//!   compact -- `normalize_compact_overlays` + `split_compact_planes`, then the
//!              kernel under test.
//!
//! NOT bit-identical, deliberately, and this is the one place it differs from
//! `parity_gemm_oq_compact`: that GEMM accumulates int8xint8 in int32, which is
//! exact and reorder-proof. These GEMVs accumulate in f32, and an overlay
//! correction lands on the lane owning overlay slot `e` rather than the lane
//! owning weight index `idx` -- so the per-lane partial sums genuinely differ and
//! only the total is the same set of terms. Demanding bit-equality here would
//! fail on correct code. The bar instead: compact must track the f64 oracle at
//! least as well as the expanded path does, and the two must agree to ~1e-5
//! relative. A nibble-order, plane-offset, or overlay-precedence bug moves the
//! result by percent, not by an ulp, so this still catches every class the
//! stricter test would.
//!
//!   cargo run --release -p hipfire-runtime --example parity_gemv_oq_compact_moe [M K n_exp]

use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::oq8_arch::{normalize_compact_overlays, split_compact_planes};
use hipfire_runtime::oq_moe::oqplus_compact_to_moe_oq8_blocks;

const GROUP: usize = 256;
const K_TOP: usize = 8;
const N_OUT: usize = 3; // the shipped magnitude-tier width -> block_stride 136
const BLOCK_STRIDE: usize = 130 + 2 * N_OUT;

fn lcg(seed: u32) -> impl FnMut() -> u32 {
    let mut s = seed.max(1);
    move || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
        s
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = ((bits >> 13) & 0x3ff) as u16;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | mant
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    if exp == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

fn sext4(nib: u8) -> i32 {
    ((nib as i32) << 28) >> 28
}

/// One expert's raw interleaved compact blocks, exactly the on-disk shape:
/// `[f16 scale][128 nibbles][N_OUT x (u8 idx, i8 val)]` per 256-weight group.
fn build_expert(seed: u32, m: usize, k: usize) -> Vec<u8> {
    let ng = k / GROUP;
    let mut rng = lcg(seed);
    let mut blob = vec![0u8; m * ng * BLOCK_STRIDE];
    for b in 0..(m * ng) {
        let base = b * BLOCK_STRIDE;
        // Scales exactly representable in f16 so the oracle shares the value.
        let scale = f16_bits_to_f32(f32_to_f16_bits(0.004 + (rng() % 64) as f32 * 0.001));
        blob[base..base + 2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        for i in 0..128 {
            blob[base + 2 + i] = (rng() & 0xff) as u8;
        }
        for e in 0..N_OUT {
            // Deliberately allow duplicate indices: precedence (later entry
            // wins, earlier zeroed) is a real divergence risk between the two
            // paths and must be exercised, not designed around.
            blob[base + 130 + 2 * e] = (rng() % GROUP as u32) as u8;
            blob[base + 130 + 2 * e + 1] = (rng() & 0xff) as u8;
        }
    }
    blob
}

/// Decode raw compact blocks to logical f32 weights, [m*k]. Mirrors the on-disk
/// contract, not either kernel: outliers REPLACE the bulk nibble, last wins.
fn decode_logical(blob: &[u8], m: usize, k: usize) -> Vec<f32> {
    let ng = k / GROUP;
    let mut w = vec![0f32; m * k];
    for row in 0..m {
        for g in 0..ng {
            let base = (row * ng + g) * BLOCK_STRIDE;
            let scale = f16_bits_to_f32(u16::from_le_bytes([blob[base], blob[base + 1]]));
            let mut code = [0i32; GROUP];
            for i in 0..128 {
                let byte = blob[base + 2 + i];
                code[2 * i] = sext4(byte & 0x0f);
                code[2 * i + 1] = sext4(byte >> 4);
            }
            for e in 0..N_OUT {
                let idx = blob[base + 130 + 2 * e] as usize;
                code[idx] = blob[base + 130 + 2 * e + 1] as i8 as i32;
            }
            for i in 0..GROUP {
                w[row * k + g * GROUP + i] = scale * code[i] as f32;
            }
        }
    }
    w
}

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Largest relative difference between two vectors, scaled by the reference's
/// own magnitude so a near-zero output cannot manufacture a huge ratio.
fn max_rel(a: &[f32], b: &[f32]) -> f64 {
    let scale = a.iter().fold(0f64, |m, v| m.max((*v as f64).abs())).max(1e-30);
    a.iter()
        .zip(b)
        .fold(0f64, |m, (x, y)| m.max(((*x as f64) - (*y as f64)).abs() / scale))
}

#[allow(clippy::too_many_arguments)]
fn run_case(gpu: &mut Gpu, label: &str, m: usize, k: usize, n_exp: usize, batch: usize) -> bool {
    let ng = k / GROUP;
    let mi = m / 2;

    let raw: Vec<Vec<u8>> = (0..n_exp).map(|e| build_expert(11 + e as u32, m, k)).collect();
    let logical: Vec<Vec<f32>> = raw.iter().map(|b| decode_logical(b, m, k)).collect();

    // Every expert must actually be selected, or an addressing bug in an
    // unvisited slot passes silently.
    let topk: Vec<i32> = (0..batch * K_TOP)
        .map(|j| (j % n_exp) as i32)
        .collect();

    // x is PER-SLOT ([N x K_TOP x K]): routed experts carry different AWQ
    // scales, so a shared-x kernel cannot pass this.
    let mut rng = lcg(0xC0FFEE);
    let x: Vec<f32> = (0..batch * K_TOP * k)
        .map(|_| ((rng() % 2001) as f32 - 1000.0) / 1000.0)
        .collect();

    // ── oracle ──────────────────────────────────────────────────────────────
    let mut oracle = vec![0f32; batch * K_TOP * m];
    for b in 0..batch {
        for t in 0..K_TOP {
            let e = topk[b * K_TOP + t] as usize;
            let xr = &x[(b * K_TOP + t) * k..(b * K_TOP + t) * k + k];
            for row in 0..m {
                let mut acc = 0f64;
                for j in 0..k {
                    acc += logical[e][row * k + j] as f64 * xr[j] as f64;
                }
                oracle[(b * K_TOP + t) * m + row] = acc as f32;
            }
        }
    }

    // ── expanded Oq8 (the path being replaced) ──────────────────────────────
    let oq8_blobs: Vec<Vec<u8>> = raw
        .iter()
        .map(|b| oqplus_compact_to_moe_oq8_blocks(b, m, k).expect("expand"))
        .collect();
    // ── compact split planes (under test) ───────────────────────────────────
    let cmp_blobs: Vec<Vec<u8>> = raw
        .iter()
        .map(|b| {
            let mut owned = b.clone();
            normalize_compact_overlays(&mut owned, m, k, GROUP);
            split_compact_planes(&owned, m, k, GROUP)
        })
        .collect();
    assert_eq!(cmp_blobs[0].len(), m * ng * BLOCK_STRIDE, "split preserves bytes");
    let saving = oq8_blobs[0].len() as f64 / cmp_blobs[0].len() as f64;

    let up = |g: &mut Gpu, blobs: &[Vec<u8>]| -> (Vec<hipfire_rdna::GpuTensor>, hipfire_rdna::GpuTensor) {
        let ts: Vec<_> = blobs
            .iter()
            .map(|b| g.upload_raw(b, &[b.len()]).unwrap())
            .collect();
        let ptrs: Vec<f32> = ts
            .iter()
            .flat_map(|t| {
                let p = t.buf.as_ptr() as u64;
                [f32::from_bits(p as u32), f32::from_bits((p >> 32) as u32)]
            })
            .collect();
        let pt = g.upload_f32(&ptrs, &[2 * ts.len()]).unwrap();
        (ts, pt)
    };
    let (_keep8, ptr8) = up(gpu, &oq8_blobs);
    let (_keepc, ptrc) = up(gpu, &cmp_blobs);

    let idx_t = gpu
        .upload_raw(&topk.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[topk.len()])
        .unwrap();
    let x_t = gpu.upload_raw(&f32b(&x), &[x.len()]).unwrap();

    let mk = |g: &mut Gpu, n: usize| g.alloc_tensor(&[n], DType::F32).unwrap();
    let (g8, u8_) = (mk(gpu, batch * K_TOP * mi), mk(gpu, batch * K_TOP * mi));
    let (gc, uc) = (mk(gpu, batch * K_TOP * mi), mk(gpu, batch * K_TOP * mi));

    if batch == 1 {
        gpu.gemv_oq8g256_moe_gate_up_k8_indexed(&ptr8, &idx_t, &x_t, &g8, &u8_, m, k, true).unwrap();
        gpu.gemv_oq_compact_moe_gate_up_k8_indexed(&ptrc, &idx_t, &x_t, &gc, &uc, m, k, true, BLOCK_STRIDE).unwrap();
    } else {
        gpu.gemv_oq8g256_moe_gate_up_k8_indexed_batched(&ptr8, &idx_t, &x_t, &g8, &u8_, m, k, K_TOP, batch, true).unwrap();
        gpu.gemv_oq_compact_moe_gate_up_k8_indexed_batched(&ptrc, &idx_t, &x_t, &gc, &uc, m, k, K_TOP, batch, true, BLOCK_STRIDE).unwrap();
    }
    gpu.device_synchronize().unwrap();

    // Reassemble [.. x M] from the gate|up split the kernels write.
    let join = |g: &Gpu, gt: &hipfire_rdna::GpuTensor, ut: &hipfire_rdna::GpuTensor| -> Vec<f32> {
        let (gv, uv) = (g.download_f32(gt).unwrap(), g.download_f32(ut).unwrap());
        let mut out = vec![0f32; batch * K_TOP * m];
        for s in 0..batch * K_TOP {
            out[s * m..s * m + mi].copy_from_slice(&gv[s * mi..s * mi + mi]);
            out[s * m + mi..s * m + m].copy_from_slice(&uv[s * mi..s * mi + mi]);
        }
        out
    };
    let y8 = join(gpu, &g8, &u8_);
    let yc = join(gpu, &gc, &uc);

    let e8 = max_rel(&oracle, &y8);
    let ec = max_rel(&oracle, &yc);
    let ex = max_rel(&y8, &yc);
    let ok = ec <= e8.max(1e-6) * 4.0 && ex < 1e-5;
    println!(
        "  {label:<28} oq8-vs-oracle {e8:.3e}  compact-vs-oracle {ec:.3e}  \
         compact-vs-oq8 {ex:.3e}  bytes {saving:.3}x  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

/// `down` writes `expert_outputs[N x K_TOP x M]` with no gate|up split, so it
/// needs its own comparison rather than a flag on `run_case`.
fn run_down(gpu: &mut Gpu, m: usize, k: usize, n_exp: usize, batch: usize) -> bool {
    let raw: Vec<Vec<u8>> = (0..n_exp).map(|e| build_expert(71 + e as u32, m, k)).collect();
    let logical: Vec<Vec<f32>> = raw.iter().map(|b| decode_logical(b, m, k)).collect();
    let topk: Vec<i32> = (0..batch * K_TOP).map(|j| (j % n_exp) as i32).collect();

    let mut rng = lcg(0x00D0_0770);
    let x: Vec<f32> = (0..batch * K_TOP * k)
        .map(|_| ((rng() % 2001) as f32 - 1000.0) / 1000.0)
        .collect();

    let mut oracle = vec![0f32; batch * K_TOP * m];
    for slot in 0..batch * K_TOP {
        let e = topk[slot] as usize;
        let xr = &x[slot * k..slot * k + k];
        for row in 0..m {
            let mut acc = 0f64;
            for j in 0..k {
                acc += logical[e][row * k + j] as f64 * xr[j] as f64;
            }
            oracle[slot * m + row] = acc as f32;
        }
    }

    let oq8_blobs: Vec<Vec<u8>> = raw
        .iter()
        .map(|b| oqplus_compact_to_moe_oq8_blocks(b, m, k).expect("expand"))
        .collect();
    let cmp_blobs: Vec<Vec<u8>> = raw
        .iter()
        .map(|b| {
            let mut owned = b.clone();
            normalize_compact_overlays(&mut owned, m, k, GROUP);
            split_compact_planes(&owned, m, k, GROUP)
        })
        .collect();

    let up = |g: &mut Gpu, blobs: &[Vec<u8>]| -> (Vec<hipfire_rdna::GpuTensor>, hipfire_rdna::GpuTensor) {
        let ts: Vec<_> = blobs.iter().map(|b| g.upload_raw(b, &[b.len()]).unwrap()).collect();
        let ptrs: Vec<f32> = ts
            .iter()
            .flat_map(|t| {
                let p = t.buf.as_ptr() as u64;
                [f32::from_bits(p as u32), f32::from_bits((p >> 32) as u32)]
            })
            .collect();
        let pt = g.upload_f32(&ptrs, &[2 * ts.len()]).unwrap();
        (ts, pt)
    };
    let (_k8, ptr8) = up(gpu, &oq8_blobs);
    let (_kc, ptrc) = up(gpu, &cmp_blobs);
    let idx_t = gpu
        .upload_raw(&topk.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[topk.len()])
        .unwrap();
    let x_t = gpu.upload_raw(&f32b(&x), &[x.len()]).unwrap();
    let o8 = gpu.alloc_tensor(&[batch * K_TOP * m], DType::F32).unwrap();
    let oc = gpu.alloc_tensor(&[batch * K_TOP * m], DType::F32).unwrap();

    gpu.gemv_oq8g256_moe_down_k8_indexed_batched_expanded(&ptr8, &idx_t, &x_t, &o8, m, k, K_TOP, batch)
        .unwrap();
    gpu.gemv_oq_compact_moe_down_k8_indexed_batched_expanded(
        &ptrc, &idx_t, &x_t, &oc, m, k, K_TOP, batch, BLOCK_STRIDE,
    )
    .unwrap();
    gpu.device_synchronize().unwrap();

    let y8 = gpu.download_f32(&o8).unwrap();
    let yc = gpu.download_f32(&oc).unwrap();
    let e8 = max_rel(&oracle, &y8);
    let ec = max_rel(&oracle, &yc);
    let ex = max_rel(&y8, &yc);
    let ok = ec <= e8.max(1e-6) * 4.0 && ex < 1e-5;
    println!(
        "  {:<28} oq8-vs-oracle {e8:.3e}  compact-vs-oracle {ec:.3e}  compact-vs-oq8 {ex:.3e}  {}",
        format!("down M={m} K={k} N={batch}"),
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_exp: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    assert_eq!(m % 2, 0, "M must be even -- the kernel splits rows into gate|up");
    assert_eq!(k % GROUP, 0, "K must be a multiple of {GROUP}");

    let mut gpu = Gpu::init().unwrap();
    println!("compact-resident indexed MoE GEMV parity (M={m} K={k} experts={n_exp})");
    let mut all = true;
    all &= run_case(&mut gpu, "gate_up (batch=1)", m, k, n_exp, 1);
    all &= run_case(&mut gpu, "gate_up batched (N=3)", m, k, n_exp, 3);
    // A second K exercising a different group count, and a non-square shape
    // closer to a real `down` projection.
    all &= run_case(&mut gpu, "gate_up K=512", m, 512, n_exp, 2);
    all &= run_case(&mut gpu, "gate_up M=256 K=2048", 256, 2048, n_exp, 2);
    // `down` is the other shape in a real layer: K is the moe_intermediate, so
    // it is much smaller and the group count much lower than gate_up's.
    all &= run_down(&mut gpu, 512, 512, n_exp, 1);
    all &= run_down(&mut gpu, 2048, 512, n_exp, 3);
    all &= run_down(&mut gpu, 1024, 1024, n_exp, 2);

    println!("{}", if all { "ALL PASS" } else { "FAILURES PRESENT" });
    if !all {
        std::process::exit(1);
    }
}
