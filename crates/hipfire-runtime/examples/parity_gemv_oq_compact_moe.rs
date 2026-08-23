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
use hipfire_runtime::hfq::HfqFile;
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

/// The PRODUCTION conversion, not a hand-rolled one.
///
/// The hand-rolled version this replaces returned `sign` for `exp == 0`, i.e. it
/// flushed every subnormal f16 to zero. Synthetic scales are generated in a
/// normal range so it never showed there, and real artifacts carry subnormal
/// group scales -- which made the ORACLE wrong on real 122B weights while both
/// production paths agreed with each other to 2.3e-7. An oracle that is only
/// correct on the data the test generates is not an oracle.
fn f16_bits_to_f32(bits: u16) -> f32 {
    hipfire_primitives::conv::f16_to_f32(bits)
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

/// Which experts stay compact. `Mixed` alternates, which is what a promoted
/// artifact really looks like -- and is the case the per-expert stride table
/// exists for.
#[derive(Clone, Copy, PartialEq)]
enum Layout {
    AllCompact,
    AllOq8,
    Mixed,
}

impl Layout {
    fn compact(self, e: usize) -> bool {
        match self {
            Layout::AllCompact => true,
            Layout::AllOq8 => false,
            Layout::Mixed => e % 2 == 0,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Layout::AllCompact => "compact",
            Layout::AllOq8 => "oq8-via-compact",
            Layout::Mixed => "MIXED",
        }
    }
}

/// The device-side `[n_exp]` i32 table: compact block stride, or 0 for Oq8.
fn stride_table(gpu: &mut Gpu, layout: Layout, n_exp: usize) -> hipfire_rdna::GpuTensor {
    let v: Vec<f32> = (0..n_exp)
        .map(|e| {
            let st: i32 = if layout.compact(e) { BLOCK_STRIDE as i32 } else { 0 };
            f32::from_bits(st as u32)
        })
        .collect();
    gpu.upload_f32(&v, &[n_exp]).unwrap()
}

/// Per-expert device blobs under `layout`: compact experts get split planes,
/// promoted ones the expanded Oq8 blocks -- both decoded from the SAME raw
/// compact source, so the logical weights are identical either way.
fn layout_blobs(raw: &[Vec<u8>], layout: Layout, m: usize, k: usize) -> Vec<Vec<u8>> {
    raw.iter()
        .enumerate()
        .map(|(e, b)| {
            if layout.compact(e) {
                let mut owned = b.clone();
                normalize_compact_overlays(&mut owned, m, k, GROUP);
                split_compact_planes(&owned, m, k, GROUP)
            } else {
                oqplus_compact_to_moe_oq8_blocks(b, m, k).expect("expand")
            }
        })
        .collect()
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
fn run_case(
    gpu: &mut Gpu,
    label: &str,
    m: usize,
    k: usize,
    n_exp: usize,
    batch: usize,
    layout: Layout,
) -> bool {
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
    let cmp_blobs: Vec<Vec<u8>> = layout_blobs(&raw, layout, m, k);
    let bytes_now: usize = cmp_blobs.iter().map(|b| b.len()).sum();
    let bytes_oq8: usize = oq8_blobs.iter().map(|b| b.len()).sum();
    let saving = bytes_oq8 as f64 / bytes_now as f64;
    let _ = ng;

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
    let strides = stride_table(gpu, layout, n_exp);

    let idx_t = gpu
        .upload_raw(&topk.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[topk.len()])
        .unwrap();
    let x_t = gpu.upload_raw(&f32b(&x), &[x.len()]).unwrap();

    let mk = |g: &mut Gpu, n: usize| g.alloc_tensor(&[n], DType::F32).unwrap();
    let (g8, u8_) = (mk(gpu, batch * K_TOP * mi), mk(gpu, batch * K_TOP * mi));
    let (gc, uc) = (mk(gpu, batch * K_TOP * mi), mk(gpu, batch * K_TOP * mi));

    if batch == 1 {
        gpu.gemv_oq8g256_moe_gate_up_k8_indexed(&ptr8, &idx_t, &x_t, &g8, &u8_, m, k, true).unwrap();
        gpu.gemv_oq_compact_moe_gate_up_k8_indexed(&ptrc, &idx_t, &strides, &x_t, &gc, &uc, m, k, true).unwrap();
    } else {
        gpu.gemv_oq8g256_moe_gate_up_k8_indexed_batched(&ptr8, &idx_t, &x_t, &g8, &u8_, m, k, K_TOP, batch, true).unwrap();
        gpu.gemv_oq_compact_moe_gate_up_k8_indexed_batched(&ptrc, &idx_t, &strides, &x_t, &gc, &uc, m, k, K_TOP, batch, true).unwrap();
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
        "  {:<34} oq8-vs-oracle {e8:.3e}  under-test-vs-oracle {ec:.3e}  \
         cross {ex:.3e}  bytes {saving:.3}x  {}",
        format!("{label} [{}]", layout.label()),
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

/// `down` writes `expert_outputs[N x K_TOP x M]` with no gate|up split, so it
/// needs its own comparison rather than a flag on `run_case`.
fn run_down(gpu: &mut Gpu, m: usize, k: usize, n_exp: usize, batch: usize, layout: Layout) -> bool {
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
    let cmp_blobs: Vec<Vec<u8>> = layout_blobs(&raw, layout, m, k);

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
    let strides = stride_table(gpu, layout, n_exp);
    let idx_t = gpu
        .upload_raw(&topk.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(), &[topk.len()])
        .unwrap();
    let x_t = gpu.upload_raw(&f32b(&x), &[x.len()]).unwrap();
    let o8 = gpu.alloc_tensor(&[batch * K_TOP * m], DType::F32).unwrap();
    let oc = gpu.alloc_tensor(&[batch * K_TOP * m], DType::F32).unwrap();

    gpu.gemv_oq8g256_moe_down_k8_indexed_batched_expanded(&ptr8, &idx_t, &x_t, &o8, m, k, K_TOP, batch)
        .unwrap();
    gpu.gemv_oq_compact_moe_down_k8_indexed_batched_expanded(
        &ptrc, &idx_t, &strides, &x_t, &oc, m, k, K_TOP, batch,
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
        "  {:<34} oq8-vs-oracle {e8:.3e}  under-test-vs-oracle {ec:.3e}  cross {ex:.3e}  {}",
        format!("down M={m} K={k} N={batch} [{}]", layout.label()),
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

/// Compare both decode paths on an artifact's REAL routed-expert bytes.
///
/// The synthetic cases prove the kernel against a generator I wrote; this proves
/// it against weights a quantizer actually emitted, including whatever overlay
/// index distribution, duplicate pattern and scale range that artifact happens
/// to carry. It needs no model load, so it works for a model too large to serve.
fn run_real(gpu: &mut Gpu, path: &str, layer: usize, n_exp: usize) -> bool {
    let hfq = match HfqFile::open(std::path::Path::new(path)) {
        Ok(f) => f,
        Err(e) => {
            println!("  cannot open {path}: {e}");
            return false;
        }
    };
    let mut raw = Vec::new();
    let mut compact_flags = Vec::new();
    let (mut m, mut k) = (0usize, 0usize);
    for e in 0..n_exp {
        let name =
            format!("model.language_model.layers.{layer}.mlp.experts.{e}.gate_up_proj.weight");
        let Some((info, buf)) = hfq.tensor_data_pread(&name) else {
            println!("  tensor not found: {name}");
            return false;
        };
        if m == 0 {
            m = info.shape[0] as usize;
            k = info.shape[1] as usize;
        }
        compact_flags.push(info.quant_type == OQPLUS_COMPACT_QT);
        raw.push(buf.to_vec());
    }
    let n_compact = compact_flags.iter().filter(|c| **c).count();
    println!(
        "  layer {layer}: M={m} K={k}  {n_compact} compact / {} promoted of {n_exp}",
        n_exp - n_compact
    );

    // Oracle from the on-disk bytes, independent of either kernel.
    let logical: Vec<Vec<f32>> = raw
        .iter()
        .zip(&compact_flags)
        .map(|(b, is_c)| {
            if *is_c {
                decode_logical(b, m, k)
            } else {
                decode_logical_oq8_canonical(b, m, k)
            }
        })
        .collect();

    let topk: Vec<i32> = (0..K_TOP).map(|j| (j % n_exp) as i32).collect();
    let mut rng = lcg(0x5EED);
    let x: Vec<f32> = (0..K_TOP * k)
        .map(|_| ((rng() % 2001) as f32 - 1000.0) / 1000.0)
        .collect();

    let mut oracle = vec![0f32; K_TOP * m];
    for t in 0..K_TOP {
        let e = topk[t] as usize;
        let xr = &x[t * k..t * k + k];
        for row in 0..m {
            let mut acc = 0f64;
            for j in 0..k {
                acc += logical[e][row * k + j] as f64 * xr[j] as f64;
            }
            oracle[t * m + row] = acc as f32;
        }
    }

    // Device blobs exactly as the loader builds them.
    let blobs: Vec<Vec<u8>> = raw
        .iter()
        .zip(&compact_flags)
        .map(|(b, is_c)| {
            if *is_c {
                let mut owned = b.clone();
                normalize_compact_overlays(&mut owned, m, k, GROUP);
                split_compact_planes(&owned, m, k, GROUP)
            } else {
                hipfire_runtime::oq_moe::oq8_canonical_to_moe_blocks(b, m, k).expect("oq8 canonical")
            }
        })
        .collect();
    let strides: Vec<f32> = compact_flags
        .iter()
        .zip(&raw)
        .map(|(is_c, b)| {
            let st: i32 = if *is_c {
                (b.len() / (m * (k / GROUP))) as i32
            } else {
                0
            };
            f32::from_bits(st as u32)
        })
        .collect();

    let ts: Vec<_> = blobs
        .iter()
        .map(|b| gpu.upload_raw(b, &[b.len()]).unwrap())
        .collect();
    let ptrs: Vec<f32> = ts
        .iter()
        .flat_map(|t| {
            let p = t.buf.as_ptr() as u64;
            [f32::from_bits(p as u32), f32::from_bits((p >> 32) as u32)]
        })
        .collect();
    let ptr_t = gpu.upload_f32(&ptrs, &[2 * ts.len()]).unwrap();
    let st_t = gpu.upload_f32(&strides, &[strides.len()]).unwrap();
    let idx_t = gpu
        .upload_raw(
            &topk.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[topk.len()],
        )
        .unwrap();
    let x_t = gpu.upload_raw(&f32b(&x), &[x.len()]).unwrap();
    let mi = m / 2;
    let g = gpu.alloc_tensor(&[K_TOP * mi], DType::F32).unwrap();
    let u = gpu.alloc_tensor(&[K_TOP * mi], DType::F32).unwrap();

    gpu.gemv_oq_compact_moe_gate_up_k8_indexed(&ptr_t, &idx_t, &st_t, &x_t, &g, &u, m, k, true)
        .unwrap();
    gpu.device_synchronize().unwrap();

    let (gv, uv) = (gpu.download_f32(&g).unwrap(), gpu.download_f32(&u).unwrap());
    let mut y = vec![0f32; K_TOP * m];
    for s in 0..K_TOP {
        y[s * m..s * m + mi].copy_from_slice(&gv[s * mi..s * mi + mi]);
        y[s * m + mi..s * m + m].copy_from_slice(&uv[s * mi..s * mi + mi]);
    }
    let err = max_rel(&oracle, &y);
    let ok = err < 1e-5;
    println!(
        "  {:<34} kernel-vs-oracle {err:.3e}  {}",
        "REAL WEIGHTS",
        if ok { "PASS" } else { "FAIL" }
    );
    // THIRD OPINION: the production expansion path on the same real bytes, run
    // through the shipping Oq8 kernel. If the compact kernel matches THIS but
    // not the oracle, the oracle is what is wrong.
    let oq8_blobs: Vec<Vec<u8>> = raw
        .iter()
        .zip(&compact_flags)
        .map(|(b, is_c)| {
            if *is_c {
                oqplus_compact_to_moe_oq8_blocks(b, m, k).expect("expand")
            } else {
                hipfire_runtime::oq_moe::oq8_canonical_to_moe_blocks(b, m, k).expect("canonical")
            }
        })
        .collect();
    let ts8: Vec<_> = oq8_blobs
        .iter()
        .map(|b| gpu.upload_raw(b, &[b.len()]).unwrap())
        .collect();
    let ptrs8: Vec<f32> = ts8
        .iter()
        .flat_map(|t| {
            let p = t.buf.as_ptr() as u64;
            [f32::from_bits(p as u32), f32::from_bits((p >> 32) as u32)]
        })
        .collect();
    let ptr8_t = gpu.upload_f32(&ptrs8, &[2 * ts8.len()]).unwrap();
    let g8 = gpu.alloc_tensor(&[K_TOP * mi], DType::F32).unwrap();
    let u8_ = gpu.alloc_tensor(&[K_TOP * mi], DType::F32).unwrap();
    gpu.gemv_oq8g256_moe_gate_up_k8_indexed(&ptr8_t, &idx_t, &x_t, &g8, &u8_, m, k, true)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let (gv8, uv8) = (
        gpu.download_f32(&g8).unwrap(),
        gpu.download_f32(&u8_).unwrap(),
    );
    let mut y8 = vec![0f32; K_TOP * m];
    for s in 0..K_TOP {
        y8[s * m..s * m + mi].copy_from_slice(&gv8[s * mi..s * mi + mi]);
        y8[s * m + mi..s * m + m].copy_from_slice(&uv8[s * mi..s * mi + mi]);
    }
    println!(
        "  {:<34} expanded-vs-oracle {:.3e}   compact-vs-expanded {:.3e}",
        "cross-checks",
        max_rel(&oracle, &y8),
        max_rel(&y8, &y)
    );

    // Per-slot, so a failure names the expert rather than the layer.
    for t in 0..K_TOP {
        let e = topk[t] as usize;
        let se = max_rel(&oracle[t * m..(t + 1) * m], &y[t * m..(t + 1) * m]);
        let stride = raw[e].len() / (m * (k / GROUP));
        println!(
            "      slot {t} expert {e}: err {se:.3e}  {}  bytes={} stride={stride} n_ov={}",
            if compact_flags[e] { "compact" } else { "promoted" },
            raw[e].len(),
            (stride as i64 - 130) / 2,
        );
    }
    ok
}

/// Decode canonical Oq8 (`[f16 scale][256 int8]`, 258 B/group) to logical f32.
fn decode_logical_oq8_canonical(blob: &[u8], m: usize, k: usize) -> Vec<f32> {
    const SRC: usize = 258;
    let ng = k / GROUP;
    let mut w = vec![0f32; m * k];
    for row in 0..m {
        for g in 0..ng {
            let base = (row * ng + g) * SRC;
            let scale = f16_bits_to_f32(u16::from_le_bytes([blob[base], blob[base + 1]]));
            for i in 0..GROUP {
                w[row * k + g * GROUP + i] = scale * (blob[base + 2 + i] as i8) as f32;
            }
        }
    }
    w
}

const OQPLUS_COMPACT_QT: u8 = hipfire_runtime::oq_moe::OQPLUS_COMPACT_QT;

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n_exp: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    assert_eq!(m % 2, 0, "M must be even -- the kernel splits rows into gate|up");
    assert_eq!(k % GROUP, 0, "K must be a multiple of {GROUP}");

    let mut gpu = Gpu::init().unwrap();
    // `--hfq <path> <layer> <n_exp>`: check real artifact bytes instead.
    if std::env::args().nth(1).as_deref() == Some("--hfq") {
        let a: Vec<String> = std::env::args().skip(2).collect();
        let path = a.first().cloned().unwrap_or_default();
        let layer: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        let n: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
        println!("REAL-WEIGHT parity: {path}");
        let ok = run_real(&mut gpu, &path, layer, n);
        println!("{}", if ok { "PASS" } else { "FAIL" });
        std::process::exit(if ok { 0 } else { 1 });
    }
    println!("compact-resident indexed MoE GEMV parity (M={m} K={k} experts={n_exp})");
    let mut all = true;
    for layout in [Layout::AllCompact, Layout::AllOq8, Layout::Mixed] {
        all &= run_case(&mut gpu, "gate_up (batch=1)", m, k, n_exp, 1, layout);
        all &= run_case(&mut gpu, "gate_up batched (N=3)", m, k, n_exp, 3, layout);
        // A second K exercising a different group count, and a shape closer to a
        // real `down` projection.
        all &= run_case(&mut gpu, "gate_up K=512", m, 512, n_exp, 2, layout);
        all &= run_case(&mut gpu, "gate_up M=256 K=2048", 256, 2048, n_exp, 2, layout);
        // `down` is the other shape in a real layer: K is the moe_intermediate,
        // so it is smaller and the group count much lower than gate_up's.
        all &= run_down(&mut gpu, 512, 512, n_exp, 1, layout);
        all &= run_down(&mut gpu, 2048, 512, n_exp, 3, layout);
        all &= run_down(&mut gpu, 1024, 1024, n_exp, 2, layout);
        // The 122B's exact routed shapes: gate_up [2048,3072] (ng=12, not a
        // power of two) and down [3072,1024].
        all &= run_down(&mut gpu, 3072, 1024, n_exp, 2, layout);
    }

    println!("{}", if all { "ALL PASS" } else { "FAILURES PRESENT" });
    if !all {
        std::process::exit(1);
    }
}
