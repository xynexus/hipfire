// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! `gemv_oq8_grouped` at G128 vs G256, both differenced against a CPU reference.
//!
//! The kernel was G256-only: its packing assumed `group/32 = 8 int8 per lane =
//! two aligned int32 loads`. G128 gives 4 per lane and one load. The two arms are
//! selected by a wave-uniform branch on `group`, so this checks BOTH — G128 for
//! the new path, G256 as the control that the port did not disturb the old one.
//!
//! G256 bit-exactness is the thing most worth protecting here: the G256 arm is
//! kept verbatim precisely because its accumulation ORDER (qw0/qw1 interleaved
//! per n) differs from the G128 arm's, and merging them would have re-rounded
//! every recorded oq8 baseline in the tree.
//!
//! Note `gemv_oq8_grouped_v2` is G256-only (it hard-requires the 128-bit load
//! shape), so K is chosen here to be a multiple of 512 — that routes G256 through
//! v2 and G128 through v1, exercising the dispatch split as well as the kernels.
//!
//! Usage: parity_gemv_oq8_g128 [M] [K]

use hipfire_rdna::Gpu;

/// y[m] = Σ_g s[m,g] · Σ_{k∈g} w[m,k]·x[k], in f32, group by group — the same
/// shape the kernel accumulates in, so a mismatch means the kernel, not the order.
fn cpu_ref(w: &[i8], s: &[f32], x: &[f32], m: usize, k: usize, group: usize) -> Vec<f32> {
    let ng = k / group;
    let mut y = vec![0.0f32; m];
    for (row, yr) in y.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for g in 0..ng {
            let mut gsum = 0.0f32;
            for i in 0..group {
                let kk = g * group + i;
                gsum += w[row * k + kk] as f32 * x[kk];
            }
            acc += gsum * s[row * ng + g];
        }
        *yr = acc;
    }
    y
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    assert_eq!(k % 512, 0, "K must be a multiple of 512 to cover v1 and v2");

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(_) => {
            println!("parity_gemv_oq8_g128: no GPU — skipped");
            return;
        }
    };

    // Deterministic pseudo-random inputs; no Date/rand dependency.
    let mut st = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        st
    };
    let w: Vec<i8> = (0..m * k).map(|_| (next() % 255) as i8).collect();
    let x: Vec<f32> = (0..k)
        .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
        .collect();

    let to_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
    let wb: Vec<u8> = w.iter().map(|&c| c as u8).collect();
    let wd = gpu.upload_raw(&wb, &[m * k]).unwrap();
    let xd = gpu.upload_raw(&to_bytes(&x), &[1, k]).unwrap();

    let mut fail = 0;
    for group in [256usize, 128] {
        let ng = k / group;
        let s: Vec<f32> = (0..m * ng)
            .map(|i| 0.002 + (i % 17) as f32 * 1e-4)
            .collect();
        let sd = gpu.upload_raw(&to_bytes(&s), &[m * ng]).unwrap();
        let yd = gpu.upload_raw(&vec![0u8; m * 4], &[m]).unwrap();
        gpu.gemv_oq8_grouped(&wd, &sd, &xd, &yd, m, k, group)
            .unwrap();
        gpu.device_synchronize().unwrap();
        let got = gpu.download_f32(&yd).unwrap();
        let want = cpu_ref(&w, &s, &x, m, k, group);

        let mut worst = 0.0f32;
        let mut mag = 0.0f32;
        for row in 0..m {
            worst = worst.max((got[row] - want[row]).abs());
            mag = mag.max(want[row].abs());
        }
        // f32 reassociation only: the kernel splits each group across 32 lanes and
        // reduces by shuffle, so exact equality is not expected — but the error must
        // stay at the f32 rounding floor, not at a "read the wrong bytes" scale.
        let rel = if mag > 0.0 { worst / mag } else { worst };
        let path = if group == 256 { "v2" } else { "v1" };
        let ok = rel < 1e-5;
        if !ok {
            fail += 1;
        }
        println!(
            "  G{group:<4} ({path})  worst {worst:.6e}  magnitude {mag:.4}  rel {rel:.3e}  {}",
            if ok { "OK" } else { "FAIL" }
        );
    }

    // Negative control: a G128 weight read as G256 must NOT agree, or the test
    // above would pass even if `group` were being ignored entirely.
    {
        let ng = k / 128;
        let s: Vec<f32> = (0..m * ng)
            .map(|i| 0.002 + (i % 17) as f32 * 1e-4)
            .collect();
        let want128 = cpu_ref(&w, &s, &x, m, k, 128);
        let want256_scales: Vec<f32> = s.iter().step_by(2).copied().collect();
        let want256 = cpu_ref(&w, &want256_scales, &x, m, k, 256);
        let differ = want128
            .iter()
            .zip(&want256)
            .any(|(a, b)| (a - b).abs() > 1e-3);
        println!(
            "  control: G128 vs G256 interpretations differ = {differ} {}",
            if differ {
                "OK"
            } else {
                "FAIL (test is vacuous)"
            }
        );
        if !differ {
            fail += 1;
        }
    }

    if fail == 0 {
        println!("parity_gemv_oq8_g128: OK");
    } else {
        println!("parity_gemv_oq8_g128: FAILED ({fail})");
        std::process::exit(1);
    }
}
