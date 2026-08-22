//! Does LDLQ error feedback pay for the "one long K segment" contract?
//!
//! `coarse_scale_study` priced the contract in WEIGHT MSE: one scale per output
//! row across all of K costs ~1.73x G=256. But weight MSE is the wrong metric
//! for LDLQ, which minimizes Hessian-weighted OUTPUT error and can happily raise
//! weight MSE while lowering the damage that actually matters.
//!
//! So this measures the right thing: proxy error `(dw)^T H (dw)`, normalized by
//! `w^T H w`, using the REAL calibration Hessian for this model, in the same
//! FWHT-rotated domain the shipped packers work in.
//!
//!     cargo run --release -p hipfire-quantize --example coarse_scale_ldlq_study \
//!         -- <model.safetensors> <pkg.calib.hfq> [rows]
//!
//! Arms, all int4 symmetric on identical rotated weights:
//!   G=256 RTN      — our bulk, no overlay, no feedback
//!   G=256 LDLQ     — our `++` bulk
//!   1-seg RTN      — Kairic's contract, no feedback
//!   1-seg LDLQ     — Kairic's contract with the feedback we already own

use hipfire_quantize::hessian_io::HessianSidecar;
use hipfire_quantize::ldlq::{inv_cholesky_lower_rotated_fast, rotate_hessian};
use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};

const G: usize = 256;

/// Quantize `w` (already rotated) and return the per-column deltas `w - q`.
/// `seg == 0` means one scale over the whole row. `l` enables LDLQ feedback.
fn deltas(w: &[f32], seg: usize, l: Option<&dyn Fn(usize, usize) -> f64>) -> Vec<f64> {
    let k = w.len();
    let span = if seg == 0 { k } else { seg };
    let mut resid: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let mut q = vec![0.0f64; k];
    // Scales come from the ORIGINAL weights, as an offline packer would compute
    // them, not from the running residual.
    let mut scales = Vec::new();
    for chunk in w.chunks(span) {
        let amax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
        scales.push((amax / 7.0).max(1e-12) as f64);
    }
    for c in 0..k {
        let s = scales[c / span];
        let qc = (resid[c] / s).round().clamp(-7.0, 7.0) * s;
        q[c] = qc;
        if let Some(l) = l {
            let d = l(c, c);
            if d > 0.0 {
                let err = (resid[c] - qc) / d;
                if err != 0.0 {
                    for f in (c + 1)..k {
                        let lf = l(f, c);
                        if lf != 0.0 {
                            resid[f] -= err * lf;
                        }
                    }
                }
            }
        }
    }
    (0..k).map(|c| w[c] as f64 - q[c]).collect()
}

/// (dw)^T H (dw) for a row, H row-major f64.
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
        eprintln!("usage: coarse_scale_ldlq_study <model.safetensors> <pkg.calib.hfq> [rows]");
        std::process::exit(2);
    };
    let rows_cap: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);

    let pkg = HessianSidecar::open(std::path::Path::new(&cal)).expect("open calib");
    // Pick the first Hessian whose K matches a tensor we can find in the shard.
    let file = std::fs::File::open(&st).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + hlen]).expect("header");
    let base = 8 + hlen;
    let obj = header.as_object().expect("obj");

    // Match the Hessian to the ACTUAL tensor, not merely to a matching K --
    // a mismatched Hessian keeps the arms comparable but is not this tensor's
    // calibration, and would not be a defensible number.
    let mut pick: Option<(String, String, usize)> = None;
    for h in pkg.tensors() {
        let stem = h.name.trim_end_matches(".weight");
        if let Some(name) = obj.keys().find(|n| {
            let ws = n.trim_end_matches(".weight");
            (ws == stem || ws.ends_with(stem) || stem.ends_with(ws))
                && obj[*n]["shape"]
                    .as_array()
                    .map(|s| s.len() == 2 && s[1].as_u64().unwrap() as usize == h.k)
                    .unwrap_or(false)
        }) {
            println!(
                "MATCHED hessian '{}' K={} -> weight '{}'",
                h.name, h.k, name
            );
            pick = Some((name.clone(), h.name.to_string(), h.k));
            break;
        }
    }
    let Some((wname, hname, k)) = pick else {
        eprintln!(
            "no Hessian NAME matched a weight in this shard; pass a shard whose\n\
                   tensors appear in the calib package"
        );
        std::process::exit(1);
    };
    let href = pkg.tensors().find(|h| h.name == hname).expect("hessian");

    let signs1 = gen_fwht_signs(42, G);
    let signs2 = gen_fwht_signs(1042, G);

    // Rotate H into the same incoherent domain the packers use.
    let mut h: Vec<f64> = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            h.push(href.at(i, j));
        }
    }
    rotate_hessian(&mut h, k, &signs1, &signs2);
    println!("rotated H ({k}x{k}) from '{hname}'; factoring...");
    let l = inv_cholesky_lower_rotated_fast(&h, k, 1e-2).expect("cholesky");
    println!("factored. quantizing {rows_cap} rows of {wname}\n");

    let info = &obj[&wname];
    let dt = info["dtype"].as_str().unwrap_or("");
    let start = base
        + info["data_offsets"].as_array().unwrap()[0]
            .as_u64()
            .unwrap() as usize;
    let nrows = info["shape"].as_array().unwrap()[0].as_u64().unwrap() as usize;

    let mut tot = [0.0f64; 4];
    let mut tot_energy = 0.0f64;
    for r in 0..nrows.min(rows_cap) {
        let mut w: Vec<f32> = (0..k)
            .map(|c| {
                let i = start + (r * k + c) * 2;
                let raw = u16::from_le_bytes([mmap[i], mmap[i + 1]]);
                assert_eq!(dt, "BF16", "this study expects a BF16 shard");
                f32::from_bits((raw as u32) << 16)
            })
            .collect();
        for ch in w.chunks_mut(G) {
            cpu_fwht_256(ch, &signs1, &signs2);
        }
        let wd: Vec<f64> = w.iter().map(|&v| v as f64).collect();
        tot_energy += hquad(&wd, &h, k);
        let lf = |i: usize, j: usize| l[(i, j)];
        let lref: &dyn Fn(usize, usize) -> f64 = &lf;
        tot[0] += hquad(&deltas(&w, G, None), &h, k);
        tot[1] += hquad(&deltas(&w, G, Some(lref)), &h, k);
        tot[2] += hquad(&deltas(&w, 0, None), &h, k);
        tot[3] += hquad(&deltas(&w, 0, Some(lref)), &h, k);
    }

    let rel: Vec<f64> = tot.iter().map(|t| t / tot_energy).collect();
    println!("Hessian-weighted proxy error  (dw)^T H (dw) / (w^T H w)");
    println!("  {:<16} {:>12} {:>10}", "arm", "rel error", "vs G256 RTN");
    let names = ["G=256 RTN", "G=256 LDLQ", "1-seg RTN", "1-seg LDLQ"];
    for (n, v) in names.iter().zip(&rel) {
        println!("  {:<16} {:>12.4e} {:>9.2}x", n, v, v / rel[0]);
    }
    println!(
        "\n  LDLQ recovery at 1 segment: {:.2}x  ({:.4e} -> {:.4e})",
        rel[2] / rel[3],
        rel[2],
        rel[3]
    );
    println!(
        "  1-seg LDLQ vs our ++ bulk (G=256 LDLQ): {:.2}x",
        rel[3] / rel[1]
    );
}
