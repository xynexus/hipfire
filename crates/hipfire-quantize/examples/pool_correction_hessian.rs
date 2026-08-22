//! Re-rank the pooled-correction frontier under the metric that actually
//! predicts output damage.
//!
//! `pool_correction_study` ranked arms by weight SSE and put pool P=64 at 76% of
//! the shipped overlay's benefit. But weight SSE weights every input channel
//! equally, and the pool is CHOSEN by a summed-benefit criterion -- exactly the
//! kind of decision an activation-weighted metric can reorder. So this re-scores
//! the same arms under `(dw)^T H (dw) / (w^T H w)` with the model's real
//! calibration Hessian, and additionally tests whether picking the pool by
//! H-weighted benefit beats picking it by raw SSE benefit.
//!
//!     cargo run --release -p hipfire-quantize --example pool_correction_hessian \
//!         -- <model.safetensors> <pkg.calib.hfq> [rows]

use hipfire_quantize::codecs::symmetric_clipsearch;
use hipfire_quantize::hessian_io::HessianSidecar;
use hipfire_quantize::ldlq::rotate_hessian;
use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

const G: usize = 256;
const N_OUT: usize = 3;

/// Per-element delta for a group under a wide-position mask.
fn deltas_into(grp: &[f32], scale: f32, wide: &[bool], out: &mut [f64]) {
    let inv = 1.0 / scale;
    for (i, &v) in grp.iter().enumerate() {
        let q = if wide[i] {
            (v * inv).round().clamp(-127.0, 127.0)
        } else {
            (v * inv).round().clamp(-7.0, 7.0)
        };
        out[i] = (v - q * scale) as f64;
    }
}

fn hquad(d: &[f64], h: &[f64], k: usize) -> f64 {
    let mut acc = 0.0;
    for i in 0..k {
        if d[i] == 0.0 {
            continue;
        }
        let row = &h[i * k..(i + 1) * k];
        let mut s = 0.0;
        for j in 0..k {
            s += row[j] * d[j];
        }
        acc += d[i] * s;
    }
    acc
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (Some(st), Some(cal)) = (a.get(1).cloned(), a.get(2).cloned()) else {
        eprintln!("usage: pool_correction_hessian <model.safetensors> <pkg.calib.hfq> [rows]");
        std::process::exit(2);
    };
    let cap: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    // How many output rows SHARE one pool. In production every row that reuses
    // the compacted activation X_pool must share it, so this is not free to set:
    // a narrow share means recompacting X_pool more often. Measuring the quality
    // side of that knob is the point -- a pool tuned on 32 rows is optimistic.
    let share: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(cap);

    let pkg = HessianSidecar::open(std::path::Path::new(&cal)).expect("calib");
    let file = std::fs::File::open(&st).expect("open");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen]).expect("hdr");
    let base = 8 + hlen;
    let obj = hdr.as_object().unwrap();

    let mut pick = None;
    for h in pkg.tensors() {
        let stem = h.name.trim_end_matches(".weight");
        if let Some(n) = obj.keys().find(|n| {
            let ws = n.trim_end_matches(".weight");
            (ws == stem || ws.ends_with(stem))
                && obj[*n]["shape"]
                    .as_array()
                    .map(|s| s.len() == 2 && s[1].as_u64().unwrap() as usize == h.k)
                    .unwrap_or(false)
        }) {
            pick = Some((n.clone(), h.name.to_string(), h.k));
            break;
        }
    }
    let Some((wname, hname, k)) = pick else {
        eprintln!("no name-matched Hessian in this shard");
        std::process::exit(1);
    };
    println!("MATCHED '{hname}' K={k} -> '{wname}'");

    let s1 = gen_fwht_signs(42, G);
    let s2 = gen_fwht_signs(1042, G);
    let href = pkg.tensors().find(|h| h.name == hname).unwrap();
    let mut h: Vec<f64> = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            h.push(href.at(i, j));
        }
    }
    rotate_hessian(&mut h, k, &s1, &s2);
    // Diagonal of the rotated H: the activation-energy weight per input channel,
    // and the right criterion for choosing which positions deserve extra bits.
    let hdiag: Vec<f64> = (0..k).map(|i| h[i * k + i]).collect();
    println!("rotated H; scoring {cap} rows\n");

    let info = &obj[&wname];
    let start = base
        + info["data_offsets"].as_array().unwrap()[0]
            .as_u64()
            .unwrap() as usize;
    let nrows = info["shape"].as_array().unwrap()[0].as_u64().unwrap() as usize;
    let take = nrows.min(cap);

    let mut rowsv: Vec<Vec<f32>> = Vec::with_capacity(take);
    for r in 0..take {
        let mut row: Vec<f32> = (0..k)
            .map(|c| {
                let i = start + (r * k + c) * 2;
                f32::from_bits((u16::from_le_bytes([mmap[i], mmap[i + 1]]) as u32) << 16)
            })
            .collect();
        for ch in row.chunks_mut(G) {
            cpu_fwht_256(ch, &s1, &s2);
        }
        rowsv.push(row);
    }
    let ng = k / G;
    let scales: Vec<Vec<f32>> = rowsv
        .iter()
        .map(|r| {
            (0..ng)
                .map(|g| symmetric_clipsearch(&r[g * G..(g + 1) * G], 7.0))
                .collect()
        })
        .collect();

    // Pool orders per (group, share-block).
    let nblocks = take.div_ceil(share);
    let mut order_sse: Vec<Vec<usize>> = Vec::with_capacity(ng * nblocks);
    let mut order_h: Vec<Vec<usize>> = Vec::with_capacity(ng * nblocks);
    for gb in 0..ng * nblocks {
        let g = gb / nblocks;
        let blk = gb % nblocks;
        let lo = blk * share;
        let hi = (lo + share).min(take);
        let _ = (lo, hi);
        let mut b_sse = vec![0.0f64; G];
        let mut b_h = vec![0.0f64; G];
        for ri in lo..hi {
            let r = &rowsv[ri];
            let grp = &r[g * G..(g + 1) * G];
            let sc = scales[ri][g];
            let inv = 1.0 / sc;
            for (p, &v) in grp.iter().enumerate() {
                let e4 = (v - (v * inv).round().clamp(-7.0, 7.0) * sc) as f64;
                let e8 = (v - (v * inv).round().clamp(-127.0, 127.0) * sc) as f64;
                let gain = e4 * e4 - e8 * e8;
                b_sse[p] += gain;
                b_h[p] += gain * hdiag[g * G + p];
            }
        }
        let mut o1: Vec<usize> = (0..G).collect();
        o1.sort_by(|&i, &j| b_sse[j].partial_cmp(&b_sse[i]).unwrap());
        order_sse.push(o1);
        let mut o2: Vec<usize> = (0..G).collect();
        o2.sort_by(|&i, &j| b_h[j].partial_cmp(&b_h[i]).unwrap());
        order_h.push(o2);
    }

    // `seg1` = ONE scale per row over the whole K (Kairic's kSegments=1). It
    // removes the per-group fold from the WMMA inner loop, which is what forces
    // the second accumulator set and caps WNt at 4 -- so it is what would buy the
    // 1-pass twin's WNt=8 shape. Pooling is the natural partner: a coarse scale
    // fails exactly where dynamic range is unusual, which is what pooled extra
    // bits restore, and it restores them WITHOUT a gather.
    let arms: Vec<(&str, usize, bool, bool)> = vec![
        ("int4 G256", 0, false, false),
        ("overlay-3 G256", usize::MAX, false, false),
        ("pool32 G256", 32, false, false),
        ("pool64 G256", 64, false, false),
        ("int4 seg1", 0, false, true),
        ("pool32 seg1", 32, false, true),
        ("pool64 seg1", 64, false, true),
        ("pool128 seg1", 128, false, true),
    ];
    let mut energy = 0.0f64;
    for r in &rowsv {
        let d: Vec<f64> = r.iter().map(|&v| v as f64).collect();
        energy += hquad(&d, &h, k);
    }

    println!(
        "  {:<18} {:>12} {:>10} {:>12}",
        "arm", "H-weighted", "vs ov", "recovered"
    );
    let mut results = Vec::new();
    for (name, p, hpick, seg1) in &arms {
        let mut tot = 0.0f64;
        for (ri, r) in rowsv.iter().enumerate() {
            let mut d = vec![0.0f64; k];
            // One scale for the entire row when seg1.
            let row_scale = symmetric_clipsearch(r, 7.0);
            for g in 0..ng {
                let grp = &r[g * G..(g + 1) * G];
                let sc = if *seg1 { row_scale } else { scales[ri][g] };
                let mut wide = vec![false; G];
                if *p == usize::MAX {
                    let mut own: Vec<usize> = (0..G).collect();
                    own.sort_by(|&i, &j| grp[j].abs().partial_cmp(&grp[i].abs()).unwrap());
                    for &q in &own[..N_OUT] {
                        wide[q] = true;
                    }
                } else {
                    let blk = ri / share;
                    let gb = g * nblocks + blk;
                    let ord = if *hpick { &order_h[gb] } else { &order_sse[gb] };
                    for &q in &ord[..*p] {
                        wide[q] = true;
                    }
                }
                deltas_into(grp, sc, &wide, &mut d[g * G..(g + 1) * G]);
            }
            tot += hquad(&d, &h, k);
        }
        results.push((name.to_string(), tot / energy));
    }
    let base_i4 = results[0].1;
    let ov = results[1].1;
    for (n, v) in &results {
        let rec = if (base_i4 - ov).abs() > 0.0 {
            (base_i4 - v) / (base_i4 - ov) * 100.0
        } else {
            0.0
        };
        println!("  {:<18} {:>12.4e} {:>9.2}x {:>11.0}%", n, v, v / ov, rec);
    }
}
