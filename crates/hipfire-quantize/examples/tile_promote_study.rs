//! Can outlier correction be made INDEX-FREE by promoting whole K-tiles?
//!
//! The shipped overlay stores `(u8 idx, i8 val)` per outlier, and `val`
//! multiplies `x[idx]` — so applying it is a GATHER, which is why it costs 23.9%
//! of prefill to do 1.2% of the arithmetic. Two fusion attempts and a wider
//! format both failed to beat it, because the gather is intrinsic to a sparse
//! correction in a dense matrix engine.
//!
//! The escape is to make the correction POSITIONAL instead of indexed: if the
//! extra bits sit in the same K slot as the base weight, the WMMA consumes them
//! with no index and no second pass. Per-weight that costs a whole extra pass.
//! Per 16-wide K-TILE it does not: promote only the tiles containing outliers to
//! int8 (`v_wmma_i32_16x16x16_iu8`, 32 cycles) and leave the rest int4 (16
//! cycles). A group of 256 is 16 tiles, so promoting t of them costs
//! (16 - t)*16 + t*32 cycles = +6.25% per promoted tile, with NO gather.
//!
//! This measures the QUALITY side of that trade: how much of the shipped
//! overlay's error reduction survives when corrections must be tile-aligned?
//!
//!     cargo run --release -p hipfire-quantize --example tile_promote_study \
//!         -- <model.safetensors> [rows]

use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

const G: usize = 256;
const TILE: usize = 16;
const N_OUT: usize = 3;

/// SSE of a group under plain int4 with one scale.
fn sse_int4(g: &[f32], scale: f32) -> f64 {
    let inv = 1.0 / scale;
    g.iter()
        .map(|&v| {
            let q = (v * inv).round().clamp(-7.0, 7.0);
            let d = (v - q * scale) as f64;
            d * d
        })
        .sum()
}

/// SSE with `keep` positions carried at int8 **in the group's own scale** —
/// codes [-127,127] against int4's [-7,7]. That is the shipped overlay's actual
/// contract: the loader zeroes the base nibble and the i8 value replaces it, so
/// what an outlier buys is 18x more RANGE (clipping repair), not a finer step.
/// Modelling it as `scale/16` measures step refinement only and makes the
/// overlay look worthless, which is how this study was wrong the first time.
fn sse_with_int8_at(g: &[f32], scale: f32, keep: &[usize]) -> f64 {
    let inv = 1.0 / scale;
    let mut on = vec![false; g.len()];
    for &p in keep {
        on[p] = true;
    }
    g.iter()
        .enumerate()
        .map(|(i, &v)| {
            let d = if on[i] {
                let q = (v * inv).round().clamp(-127.0, 127.0);
                (v - q * scale) as f64
            } else {
                let q = (v * inv).round().clamp(-7.0, 7.0);
                (v - q * scale) as f64
            };
            d * d
        })
        .sum()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let Some(path) = a.get(1).cloned() else {
        eprintln!("usage: tile_promote_study <model.safetensors> [rows]");
        std::process::exit(2);
    };
    let rows_cap: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(256);

    let file = std::fs::File::open(&path).expect("open");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen]).expect("hdr");
    let base = 8 + hlen;
    let obj = header.as_object().expect("obj");
    let signs1 = gen_fwht_signs(42, G);
    let signs2 = gen_fwht_signs(1042, G);

    println!("tile-promotion study: G={G}, tile={TILE}, n_out={N_OUT}, <= {rows_cap} rows\n");
    println!(
        "  {:<14} {:>10} {:>11} {:>11} {:>11} {:>11} {:>9}",
        "tensor", "int4", "overlay-3", "1 tile", "2 tiles", "3 tiles", "1tile/ov"
    );

    for sfx in ["gate_proj.weight", "down_proj.weight", "q_proj.weight"] {
        let Some(name) = obj.keys().find(|n| n.ends_with(sfx)) else {
            continue;
        };
        let info = &obj[name];
        let sh: Vec<usize> = info["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if sh.len() != 2 || sh[1] % G != 0 {
            continue;
        }
        let (rows, k) = (sh[0], sh[1]);
        let dt = info["dtype"].as_str().unwrap_or("");
        let start = base
            + info["data_offsets"].as_array().unwrap()[0]
                .as_u64()
                .unwrap() as usize;
        let mut acc = [0.0f64; 5];
        let mut energy = 0.0f64;

        for r in 0..rows.min(rows_cap) {
            let mut row: Vec<f32> = (0..k)
                .map(|c| {
                    let i = start + (r * k + c) * 2;
                    let raw = u16::from_le_bytes([mmap[i], mmap[i + 1]]);
                    assert_eq!(dt, "BF16");
                    f32::from_bits((raw as u32) << 16)
                })
                .collect();
            for ch in row.chunks_mut(G) {
                cpu_fwht_256(ch, &signs1, &signs2);
            }
            energy += row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();

            for grp in row.chunks(G) {
                let amax = grp.iter().fold(0f32, |m, v| m.max(v.abs()));
                if amax <= 0.0 {
                    continue;
                }
                // The shipped packer clip-searches: a smaller scale refines the
                // bulk and deliberately clips the top few, which the overlay then
                // repairs. Without this the overlay has nothing to repair.
                let scale = hipfire_quantize::codecs::symmetric_clipsearch(grp, 7.0);
                acc[0] += sse_int4(grp, scale);

                // Shipped shape: the N_OUT largest positions, anywhere.
                let mut order: Vec<usize> = (0..G).collect();
                order.sort_by(|&i, &j| grp[j].abs().partial_cmp(&grp[i].abs()).unwrap());
                acc[1] += sse_with_int8_at(grp, scale, &order[..N_OUT]);

                // Tile-aligned: rank the 16 tiles by the error int4 leaves in
                // them, promote the worst t tiles WHOLE (all 16 lanes to int8).
                let mut tiles: Vec<(f64, usize)> = (0..G / TILE)
                    .map(|t| {
                        let s = &grp[t * TILE..(t + 1) * TILE];
                        (sse_int4(s, scale), t)
                    })
                    .collect();
                tiles.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                for (n, slot) in [(1usize, 2usize), (2, 3), (3, 4)] {
                    let keep: Vec<usize> = tiles[..n]
                        .iter()
                        .flat_map(|&(_, t)| (t * TILE..(t + 1) * TILE))
                        .collect();
                    acc[slot] += sse_with_int8_at(grp, scale, &keep);
                }
            }
        }
        if energy <= 0.0 {
            continue;
        }
        let rel: Vec<f64> = acc.iter().map(|v| v / energy).collect();
        println!(
            "  {:<14} {:>10.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>11.3e} {:>8.2}x",
            sfx.trim_end_matches(".weight"),
            rel[0],
            rel[1],
            rel[2],
            rel[3],
            rel[4],
            rel[2] / rel[1]
        );
    }
    println!("\ncost, per 256-group (no gather, no second pass, no index):");
    println!("  baseline  16 tiles x 2 iu4 passes = 512 WMMA cycles");
    for t in 1..=3 {
        let cyc = (16 - t) * 32 + t * 32;
        let bits = (136.0 + 8.0 * t as f64) * 8.0 / 256.0;
        println!("  {t} tile(s) promoted -> {cyc} cycles (CYCLE-NEUTRAL), {bits:.3} bits/weight");
    }
    println!("  shipped overlay      -> 512 cycles + a separate gather pass = 23.9% of PREFILL");
}
