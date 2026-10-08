// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.
//! `oq_compact_overlay_correct_tr` against a host evaluation of the same sparse
//! overlay sum, at batch widths past the small-B (`_trs`, B <= 64) routing and row
//! counts that leave a partial row block. The host differs only by FMA contraction,
//! so the gate is error relative to the output scale; the printed hash of the GPU
//! output is for comparing two kernel versions bit for bit. Then again through the
//! b-tiled `_t` kernel (HIPFIRE_OQ_OVERLAY_ROWC=0).
//!
//! B % 4 != 0 with K*B a whole number of pages is the served-shape fault: a lane
//! gathers 4 b at once, so the last one read past XT -- into an unmapped page when
//! the exactly-sized buffer ended on one (B=78 at K=6144 is 117 pages; six
//! coalesced 27B prompts) -- and `_t` also stored up to 3 rows past Y. So inputs
//! carry OQ_OVERLAY_SLACK and exactly-sized ones must be refused, and rows past B
//! are a sentinel band that must come back untouched.
//!
//!   cargo run --release -p hipfire-rdna --example parity_oq_overlay_tr

use hipfire_rdna::{DType, Gpu, OQ_OVERLAY_SLACK};

fn main() {
    let mut gpu = Gpu::init().unwrap();
    let mut ok = true;
    let mut tr3_hashes = Vec::new();
    // "tr3" is what a 4.25-bit model takes (n_ov = 3); "tr" is the generic kernel
    // it must reproduce bit for bit.
    for pass in ["tr3", "tr", "t"] {
        match pass {
            "tr" => std::env::set_var("HIPFIRE_OQ_OVERLAY_TR3", "0"),
            "t" => std::env::set_var("HIPFIRE_OQ_OVERLAY_ROWC", "0"),
            _ => {}
        }
        println!("-- {pass}");
        for (i, &(m, k, b)) in [
            (200usize, 1024usize, 65usize),
            (128, 2048, 100),
            (77, 1024, 257),
            (300, 512, 513),
            (5120, 6144, 78),
            (6144, 6144, 30),
        ]
        .iter()
        .enumerate()
        {
            let (pass_ok, hash) = case(&mut gpu, m, k, b);
            ok &= pass_ok;
            match pass {
                "tr3" => tr3_hashes.push(hash),
                "tr" if tr3_hashes[i] != hash => {
                    println!("   tr3 hash {:016x} differs from tr", tr3_hashes[i]);
                    ok = false;
                }
                _ => {}
            }
        }
    }
    if !ok {
        std::process::exit(1);
    }
    println!("ALL PASS");
}

/// f16 bits of n / 4096 for 1 <= n < 1024 (exactly representable).
fn f16_bits(n: u32) -> u16 {
    let e = 31 - n.leading_zeros(); // n = 1.m * 2^e
    let mant = ((n << (10 - e)) & 0x3ff) as u16;
    let exp = (e as i32 - 12 + 15) as u16;
    (exp << 10) | mant
}

fn case(gpu: &mut Gpu, m: usize, k: usize, b: usize) -> (bool, u64) {
    let group = 256usize;
    let n_groups = k / group;
    let nib_bytes = group / 2;
    let n_ov = 3usize;
    let side_stride = 2 + 2 * n_ov;
    let block_stride = nib_bytes + side_stride;
    let mut s = 12345u32;
    let mut rnd = || {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        s >> 8
    };
    // W: nibble region (unused by the correction) then the side tables.
    let mut w = vec![0u8; m * n_groups * nib_bytes + m * n_groups * side_stride];
    let side_base = m * n_groups * nib_bytes;
    let mut sw = vec![0f32; m * n_groups];
    let mut ov = vec![(0usize, 0i32); m * n_groups * n_ov];
    for i in 0..m * n_groups {
        // n / 4096 with 4 <= n < 1024: exact in f16, so the host uses the same value.
        let n = 4 + rnd() % 1020;
        sw[i] = n as f32 / 4096.0;
        let base = side_base + i * side_stride;
        w[base..base + 2].copy_from_slice(&f16_bits(n).to_le_bytes());
        for e in 0..n_ov {
            let idx = (rnd() % group as u32) as usize;
            let val = (rnd() % 255) as i32 - 127;
            w[base + 2 + 2 * e] = idx as u8;
            w[base + 3 + 2 * e] = val as i8 as u8;
            ov[i * n_ov + e] = (idx, val);
        }
    }
    let xt: Vec<i8> = (0..k * b + OQ_OVERLAY_SLACK)
        .map(|_| ((rnd() % 255) as i32 - 127) as i8)
        .collect();
    let xst: Vec<f32> = (0..n_groups * b + OQ_OVERLAY_SLACK)
        .map(|_| 0.01 + (rnd() % 1000) as f32 * 1e-4)
        .collect();
    const SENTINEL_ROWS: usize = 4;
    let y0: Vec<f32> = (0..(b + SENTINEL_ROWS) * m)
        .map(|_| (rnd() % 1000) as f32 * 1e-3)
        .collect();

    // Host: Y[b][row] += sum_g (isum * sw) * xs, groups in order.
    let mut want = y0.clone();
    for row in 0..m {
        for bb in 0..b {
            let mut acc = 0f32;
            for g in 0..n_groups {
                let mut isum = 0i32;
                for e in 0..n_ov {
                    let (idx, val) = ov[(row * n_groups + g) * n_ov + e];
                    isum += val * xt[(g * group + idx) * b + bb] as i32;
                }
                acc += isum as f32 * sw[row * n_groups + g] * xst[g * b + bb];
            }
            want[bb * m + row] += acc;
        }
    }

    let wd = gpu.upload_raw(&w, &[w.len()]).unwrap();
    let xtd = gpu
        .upload_raw(
            &xt.iter().map(|v| *v as u8).collect::<Vec<_>>(),
            &[xt.len()],
        )
        .unwrap();
    let xsd = gpu
        .upload_raw(
            &xst.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[xst.len()],
        )
        .unwrap();
    let yd = gpu
        .upload_raw(
            &y0.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            &[y0.len()],
        )
        .unwrap();
    let _ = DType::F32;
    let exact = gpu.upload_raw(&vec![0u8; k * b], &[k * b]).unwrap();
    let refused = gpu
        .oq_compact_overlay_correct_t(&wd, &exact, &xsd, &yd, m, k, b, group, block_stride)
        .is_err();
    gpu.oq_compact_overlay_correct_t(&wd, &xtd, &xsd, &yd, m, k, b, group, block_stride)
        .unwrap();
    gpu.device_synchronize().unwrap();
    let got = gpu.download_f32(&yd).unwrap();
    let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let max_err = got
        .iter()
        .zip(&want)
        .fold(0f32, |a, (g, w)| a.max((g - w).abs()));
    let hash = got.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3)
    });
    let clobbered = got[b * m..] != y0[b * m..];
    let pass = max_err <= scale * 1e-6 && !clobbered && refused;
    println!(
        "M={m} K={k} B={b}: max_err/scale={:.2e} gpu_hash={hash:016x}{}{} -> {}",
        max_err / scale,
        if clobbered { " WROTE PAST B" } else { "" },
        if refused { "" } else { " TOOK AN UNPADDED XT" },
        if pass { "PASS" } else { "FAIL" }
    );
    (pass, hash)
}
