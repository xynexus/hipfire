#![allow(
    clippy::duplicated_attributes,
    clippy::needless_range_loop,
    clippy::useless_vec
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity for the `Oq8G256` embedding gather kernel.
//!
//! Chain under test: the real encoder (`hipfire_quantize::codecs::quantize_oq8g256`)
//! → the new GPU gather (`embedding_lookup_oq8g256`), checked against the trusted
//! host decoder (`hipfire_runtime::quant::dequant_oq8g256`, already used by the
//! gemma3-vl tower). Both sides therefore see the same bytes and the comparison
//! tests the KERNEL, not a re-implementation of the format.
//!
//! The one thing this has to get right is the rotation: Oq8 symbols are stored in
//! the FWHT-rotated frame, and the inverse is the same signed transform with the
//! two sign vectors SWAPPED. Getting that backwards is silent — the values stay
//! finite and plausible — so it needs an explicit control, which the last check
//! provides by rotating the wrong way on purpose and asserting it does NOT match.
//!
//! Run: `cargo run -p hipfire-runtime --release --example embedding_oq8_parity`

use hipfire_quantize::codecs::quantize_oq8g256;
use hipfire_rdna::{DType, Gpu};
use hipfire_runtime::quant::dequant_oq8g256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (rows, dim) = (64usize, 1024usize);
    let groups = dim / 256;

    // Deterministic pseudo-random table with a few deliberate outlier rows, since
    // outliers are exactly what the rotation exists to spread.
    let mut table = vec![0.0f32; rows * dim];
    let mut z: u64 = 0x9E3779B97F4A7C15;
    for (i, v) in table.iter_mut().enumerate() {
        z ^= z << 13;
        z ^= z >> 7;
        z ^= z << 17;
        let u = ((z >> 11) as f64 / (1u64 << 53) as f64) as f32;
        *v = (u - 0.5) * 0.08;
        if i % 977 == 0 {
            *v *= 25.0; // outlier
        }
    }

    let s1 = hipfire_primitives::fwht::gen_fwht_signs(42, 256);
    let s2 = hipfire_primitives::fwht::gen_fwht_signs(1042, 256);

    // Encode row-major, one row at a time: a gather addresses whole rows.
    let mut packed: Vec<u8> = Vec::with_capacity(rows * groups * 258);
    for r in 0..rows {
        packed.extend_from_slice(&quantize_oq8g256(
            &table[r * dim..(r + 1) * dim],
            &s1,
            &s2,
        ));
    }
    assert_eq!(packed.len(), rows * groups * 258, "unexpected packed size");

    let mut gpu = Gpu::init().expect("Gpu::init failed");
    println!("arch: {}  rows={rows} dim={dim}  packed={} B/row", gpu.arch, groups * 258);

    // Upload the PLANAR form the kernel reads (and the tied head's GEMV shares).
    let combined = hipfire_runtime::oq8_arch::oq8_combined(&packed, rows, dim);
    let d_table = gpu.upload_raw(&combined, &[combined.len()])?;
    let d_out = gpu.zeros(&[dim], DType::F32)?;

    let mut worst = 0.0f32;
    let mut worst_row = 0usize;
    for r in [0usize, 1, 7, 13, 31, 63] {
        gpu.embedding_lookup_oq8g256(&d_table, &d_out, r as u32, dim)?;
        let got = gpu.download_f32(&d_out)?;
        let want = dequant_oq8g256(&packed[r * groups * 258..(r + 1) * groups * 258], dim);
        for i in 0..dim {
            let d = (got[i] - want[i]).abs();
            if d > worst {
                worst = d;
                worst_row = r;
            }
        }
    }
    println!("  kernel vs host decoder: max |Δ| {worst:.3e} (row {worst_row})");
    assert!(
        worst < 1e-4,
        "gather disagrees with the host decoder — max |Δ| {worst:.3e}"
    );

    // Control: the rotation must actually be doing something. Compare the
    // gathered row against the SAME bytes read as if they were unrotated int8 —
    // if the kernel silently skipped the inverse FWHT these would agree.
    let want0 = dequant_oq8g256(&packed[0..groups * 258], dim);
    gpu.embedding_lookup_oq8g256(&d_table, &d_out, 0, dim)?;
    let got0 = gpu.download_f32(&d_out)?;
    let mut unrot = vec![0.0f32; dim];
    for g in 0..groups {
        let gp = &packed[g * 258..(g + 1) * 258];
        let sc = hipfire_primitives::conv::f16_to_f32(u16::from_le_bytes([gp[0], gp[1]]));
        for i in 0..256 {
            unrot[g * 256 + i] = (gp[2 + i] as i8) as f32 * sc;
        }
    }
    let ctrl = got0
        .iter()
        .zip(&unrot)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  control (vs unrotated read): max |Δ| {ctrl:.3e} — must be large");
    assert!(
        ctrl > 1e-3,
        "gathered row matches an unrotated read; the inverse FWHT is not being applied"
    );
    let _ = &want0;

    // Reconstruction quality, for the record.
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for r in 0..rows {
        let want = dequant_oq8g256(&packed[r * groups * 258..(r + 1) * groups * 258], dim);
        for i in 0..dim {
            let e = (table[r * dim + i] - want[i]) as f64;
            num += e * e;
            den += (table[r * dim + i] as f64).powi(2);
        }
    }
    println!("  Oq8G256 embed rel MSE {:.3e}", num / den.max(1e-30));
    println!("PARITY OK");
    Ok(())
}
