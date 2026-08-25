//! Does the paper's Hadamard rotation measurably improve KVarN reconstruction,
//! and does the gain behave the way theory says?
//!
//! Reports MANY random tiles, not one — a single deterministic tile gave
//! 1.17 / 1.21 / 1.18 at 2/4/8 bits, which is non-monotonic and therefore noise:
//! there is no mechanism by which 4-bit gains more than 2-bit. If the rotation
//! helps, the gain must DECREASE with bit width, because a coarser grid is what
//! benefits from spreading outliers.
//!
//! Tile layout is [channel(row) x token(col)], matching kvarn_gather_k_tiles.
use hipfire_kvquant::kvarn::*;

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*state >> 33) as f32) / (u32::MAX as f32 / 2.0)) - 1.0
}

fn main() {
    let (r, c) = (128usize, 32usize);
    let n_tiles = 200usize;
    let mut seed = 0x9E3779B97F4A7C15u64;

    let err = |q: &QuantTile, rf: &[f32]| -> f64 {
        let d = dequantize_tile(q);
        let num: f64 = d.iter().zip(rf).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = rf.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().max(1e-12);
        (num / den).sqrt()
    };

    println!("{r}x{c} tiles, n={n_tiles}, outlier channels ~1/8, gain = plain/rotated");
    println!("{:>5} {:>9} {:>9} {:>9} {:>9} {:>7}", "bits", "mean", "std", "min", "max", "n>1");
    for (bits, qmax) in [(2usize, 3.0f32), (4, 15.0), (8, 255.0)] {
        let mut gains = Vec::with_capacity(n_tiles);
        let mut s = seed;
        for _ in 0..n_tiles {
            let mut tile = vec![0f32; r * c];
            for ch in 0..r {
                let outlier = lcg(&mut s) > 0.75; // ~1 in 8 channels
                let mag = if outlier { 8.0 + 8.0 * lcg(&mut s).abs() } else { 1.0 };
                for t in 0..c {
                    tile[ch * c + t] = lcg(&mut s) * mag;
                }
            }
            let mut rot_ref = tile.clone();
            hadamard_channels(&mut rot_ref, r, c);
            let p = err(&quantize_tile_qmax(&tile, r, c, qmax), &tile);
            let q = err(&quantize_tile_rotated(&tile, r, c, qmax), &rot_ref);
            gains.push(p / q.max(1e-12));
        }
        seed = seed.wrapping_add(0x1234567);
        let n = gains.len() as f64;
        let mean = gains.iter().sum::<f64>() / n;
        let std = (gains.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / n).sqrt();
        let min = gains.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let wins = gains.iter().filter(|g| **g > 1.0).count();
        println!("{bits:>5} {mean:>9.4} {std:>9.4} {min:>9.4} {max:>9.4} {wins:>4}/{n_tiles}");
    }
}
