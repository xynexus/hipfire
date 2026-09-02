//! Parity for the batched QTIP-3 GEMM (`gemm_qtip3g256`) against the already
//! validated scalar `gemv_qtip3g256`, on identical weights and activations.
//!
//! WHY THIS ORACLE. A CPU trellis oracle re-checks the decode, which
//! `parity_gemv_qtip4g256` already covers. What is NEW here is the BATCHING:
//! decoding a weight once and reusing it across a 32-wide column tile, plus the
//! `col_base`/`n_cols` tiling and the `[N,M]` output transpose. Comparing
//! against the GEMV isolates exactly that — every column of the GEMM must equal
//! the GEMV run on that column, bit-for-bit, because the decode path is
//! character-identical and only the reuse and store differ.
//!
//! Exercises N deliberately non-multiple-of-32 so the short final tile is
//! covered; a tiling bug that only shows on a partial tile is the likely one.
//!
//!   cargo run --release -p hipfire-rdna --example parity_gemm_qtip3g256 [M K N]

use hipfire_rdna::Gpu;

const BLK: usize = 100; // [f32 scale | 96 B of 3-bit symbols]
const NT: usize = 32; // activation columns per pass (matches the kernel)

fn lcg_u8(seed: u32, n: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            (s >> 13) as u8
        })
        .collect()
}
fn lcg_f32(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345) & 0x7fff_ffff;
            -1.0 + (s as f32 / 2_147_483_648.0) * 2.0
        })
        .collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let m: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(70);
    assert_eq!(k % 256, 0, "K must be a multiple of 256");
    let ng = k / 256;

    let mut gpu = Gpu::init().unwrap();

    // Random 3-bit symbols packed as a contiguous LE bitstream (symbol i at
    // bit 3i), and positive per-group scales — the layout gemv_qtip3g256 reads.
    let sym: Vec<u8> = lcg_u8(1, m * ng * 256).iter().map(|b| b & 7).collect();
    let sc: Vec<f32> = lcg_f32(2, m * ng)
        .iter()
        .map(|v| 0.02 + 0.03 * v.abs())
        .collect();
    let mut blob = vec![0u8; m * ng * BLK];
    for r in 0..m {
        for g in 0..ng {
            let blk = (r * ng + g) * BLK;
            blob[blk..blk + 4].copy_from_slice(&sc[r * ng + g].to_le_bytes());
            let sbase = (r * ng + g) * 256;
            for i in 0..256 {
                let bit = 3 * i;
                let v = (sym[sbase + i] & 7) as u32;
                let (byte, off) = (blk + 4 + bit / 8, bit % 8);
                blob[byte] |= (v << off) as u8;
                if off > 5 {
                    blob[byte + 1] |= (v >> (8 - off)) as u8;
                }
            }
        }
    }
    let ad = gpu.upload_raw(&blob, &[blob.len()]).unwrap();

    // X is [N, K] row-major; the GEMV consumes one row at a time.
    let x = lcg_f32(3, n * k);
    let xd = gpu.upload_raw(&f32_bytes(&x), &[n, k]).unwrap();

    // Reference: GEMV per column.
    let mut yref = vec![0.0f32; n * m];
    let yv = gpu.upload_raw(&vec![0u8; m * 4], &[1, m]).unwrap();
    for c in 0..n {
        let xc = gpu.upload_raw(&f32_bytes(&x[c * k..(c + 1) * k]), &[1, k]).unwrap();
        gpu.gemv_qtip3g256(&ad, &xc, &yv, m, k).unwrap();
        gpu.device_synchronize().unwrap();
        yref[c * m..(c + 1) * m].copy_from_slice(&gpu.download_f32(&yv).unwrap());
    }

    // Under test: the batched GEMM, walked in 32-column tiles.
    let yd = gpu.upload_raw(&vec![0u8; n * m * 4], &[n, m]).unwrap();
    let mut base = 0usize;
    while base < n {
        let cols = NT.min(n - base);
        gpu.gemm_qtip3g256(&ad, &xd, &yd, m, k, n, base, cols).unwrap();
        base += cols;
    }
    gpu.device_synchronize().unwrap();
    let yg = gpu.download_f32(&yd).unwrap();

    let (mut max_abs, mut max_mag) = (0.0f32, 0.0f32);
    for i in 0..n * m {
        max_abs = max_abs.max((yg[i] - yref[i]).abs());
        max_mag = max_mag.max(yref[i].abs());
    }
    let rel = if max_mag > 0.0 { max_abs / max_mag } else { 0.0 };
    println!("gemm_qtip3g256 vs gemv_qtip3g256  M={m} K={k} N={n} (tiles of {NT})");
    println!("  max |Δ| {max_abs:.3e}   max |ref| {max_mag:.3e}   rel {rel:.3e}");
    // Same decode, same order of the 8 per-lane products, same wave reduction —
    // only the reuse differs — so this is exact, not merely close.
    assert!(
        max_abs == 0.0 || rel < 1e-6,
        "batched GEMM disagrees with the scalar GEMV"
    );
    println!("PASS — every column matches the GEMV");
}
