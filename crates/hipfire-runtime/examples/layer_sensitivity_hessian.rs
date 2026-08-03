// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-layer quantization sensitivity, Hessian/imatrix-weighted — the cheap
//! proxy that ranks layers WITHOUT quantizing one artifact per layer.
//!
//! # What it computes
//!
//! For a linear layer `y = W x`, replacing `W` by its quantization `Ŵ` costs
//! output error `E‖δy‖² = tr(ΔW · H · ΔWᵀ)` with `H = E[x xᵀ]`. Using only the
//! diagonal of `H` (the captured imatrix `d_j = E[x_j²]`) gives the standard
//! GPTQ/imatrix proxy `Σ_ij d_j ΔW_ij²`.
//!
//! `ΔW` is taken as `W_oq8 − W_oq4`: oq8++ measures 3.5e-4 KLD from bf16 (plan
//! §13j), i.e. near-lossless, so it stands in for the unquantized reference —
//! which is not on disk for this model.
//!
//! # Why this quantizes oq8 rather than differencing two artifacts
//!
//! The obvious approach — `ΔW = W_oq8 − W_oq4` from the two on-disk artifacts —
//! does not survive contact with the storage format. Stored weights are
//! AWQ-scaled then FWHT-rotated, `W_st = (W·S)·Rᵀ`, and the two artifacts do NOT
//! share `S`: oq4++ carries AWQ scales on down_proj only, oq8++ on all 186
//! projections. Recovering the original basis needs `W = W_st·R·S⁻¹` — and since
//! diagonal `S` and orthogonal `R` do not commute, the `R` cannot be skipped.
//! (Skipping it produces relative errors around 50%, which is how the mistake
//! announces itself.)
//!
//! So instead of differencing bases, this stays inside ONE: take oq8++'s stored
//! weights as the reference (3.5e-4 KLD from bf16, plan §13j — near-lossless)
//! and simulate 4-bit quantization of them *in that same basis*. `ΔW = W₈ −
//! Q₄(W₈)` needs no inverse transform, and it measures the quantity allocation
//! actually needs: **what 4-bit costs at this layer**, rather than what one
//! particular oq4 artifact's AWQ/LDLQ choices happened to cost.
//!
//! The imatrix applies in that basis too. AWQ divides the activation by `s`, so
//! its second moment becomes `d_j/s_j²`; the FWHT then mixes each 256-group, and
//! a random-sign Hadamard leaves the rotated diagonal ≈ the group mean — which
//! is exactly the equalization the rotation exists to produce. Hence the weight
//! is `mean(d/s² over group)` per group.
//!
//! # Output
//!
//! Per-tensor and per-layer sensitivity, ranked, plus concentration statistics —
//! what share of total error the top-k layers carry. That share is the number
//! that decides mixed-precision-within-layers vs promote-whole-layers at equal
//! bits: concentrated ⇒ promotion competes, spread ⇒ it cannot.
//!
//!   cargo run --release -p hipfire-runtime --example layer_sensitivity_hessian \
//!     -- <oq4.hfq> <oq8.hfq> [calib.hfq]

use hipfire_runtime::hfq::HfqFile;
use std::collections::BTreeMap;
use std::path::Path;

const GROUP: usize = 256;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal → normalize
            let mut e = -1i32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            let exp32 = (127 - 15 + e + 1) as u32;
            (sign << 31) | (exp32 << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

fn read_f16_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn read_f32_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Dequantize an on-disk Opus-quant weight tensor.
///
/// The ON-DISK layout is block-interleaved per 256-group — oq4
/// `[f16 scale][128 nibbles]` = 130 B, oq8 `[f16 scale][256 int8]` = 258 B
/// (`hipfire-quant-format`). The flat `[payload | scales]` form the GEMM kernels
/// take is what the LOADER repacks to; reading the on-disk bytes with that
/// assumption silently reinterprets half of each value and produces a convincing
/// pile of inf/NaN.
fn dequant_opus(data: &[u8], m: usize, k: usize, bits: usize) -> Vec<f32> {
    let ng = k / GROUP;
    let payload = if bits == 4 { GROUP / 2 } else { GROUP };
    let block = 2 + payload;
    assert_eq!(
        data.len(),
        m * ng * block,
        "on-disk size mismatch: expected {} bytes ({m}x{ng} blocks of {block})",
        m * ng * block
    );
    let mut out = vec![0f32; m * k];
    for row in 0..m {
        for g in 0..ng {
            let base = (row * ng + g) * block;
            let scale = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
            let p = &data[base + 2..base + block];
            for j in 0..GROUP {
                let q = if bits == 4 {
                    let byte = p[j / 2];
                    let nib = if j % 2 == 0 { byte & 0xf } else { byte >> 4 };
                    if nib >= 8 {
                        nib as i32 - 16
                    } else {
                        nib as i32
                    }
                } else {
                    p[j] as i8 as i32
                };
                out[row * k + g * GROUP + j] = q as f32 * scale;
            }
        }
    }
    out
}

/// Outlier counts per 256-group to evaluate. Each costs a (u8 idx, i8 val) pair
/// = 16 bits, so N outliers add N/16 bits/weight. N=7 reproduces the shipped
/// `oq4.5++` budget exactly: 4.0625 + 7/16 = 4.500 bits/weight.
const OUTLIER_NS: [usize; 4] = [2, 4, 7, 12];

struct Row {
    layer: usize,
    site: String,
    n: usize,
    sse: f64,      // ‖ΔW‖²_F, isotropic
    ref_energy: f64, // ‖W8‖²_F, for a relative figure
    sse_w: Option<f64>, // imatrix-weighted, where a calib imatrix exists
    sse8: f64,
    sse8_w: f64,
    sse_out: [f64; OUTLIER_NS.len()],
    sse_out_w: [f64; OUTLIER_NS.len()],
}

fn parse_layer_site(name: &str) -> Option<(usize, String)> {
    let idx = name.find(".layers.")? + ".layers.".len();
    let rest = &name[idx..];
    let dot = rest.find('.')?;
    let layer: usize = rest[..dot].parse().ok()?;
    let site = rest[dot + 1..].trim_end_matches(".weight").to_string();
    Some((layer, site))
}

fn main() {
    let mut a = std::env::args().skip(1);
    let p4 = a.next().expect("usage: layer_sensitivity_hessian <oq4.hfq> <oq8.hfq> [calib.hfq]");
    let p8 = a.next().expect("usage: layer_sensitivity_hessian <oq4.hfq> <oq8.hfq> [calib.hfq]");
    let pc = a.next();

    let h4 = HfqFile::open(Path::new(&p4)).expect("open oq4");
    let h8 = HfqFile::open(Path::new(&p8)).expect("open oq8");
    let hc = pc.as_ref().map(|p| HfqFile::open(Path::new(p)).expect("open calib"));

    // imatrix per weight-tensor name, where captured.
    let mut imat: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    if let Some(c) = hc.as_ref() {
        for t in c.tensors() {
            if let Some(base) = t.name.strip_suffix(".imatrix") {
                let (_, bytes) = c.tensor_data_vec(&t.name).expect("imatrix data");
                imat.insert(format!("{base}.weight"), read_f32_vec(&bytes));
            }
        }
        eprintln!("calib: {} imatrix tensors", imat.len());
    }

    // quant_type bytes for the two Opus formats, read off the artifacts
    // themselves so this does not hard-code a codec table.
    let qt_of = |f: &HfqFile, want: &str| -> Option<u8> {
        f.tensors()
            .iter()
            .find(|t| t.name.ends_with(want))
            .map(|t| t.quant_type)
    };
    let qt4 = qt_of(&h4, "mlp.down_proj.weight").expect("oq4 down_proj");
    let qt8 = qt_of(&h8, "mlp.down_proj.weight").expect("oq8 down_proj");
    eprintln!("quant_type: oq4={qt4} oq8={qt8}");

    let names: Vec<String> = h8
        .tensors()
        .iter()
        .filter(|t| t.quant_type == qt8 && t.shape.len() == 2)
        .map(|t| t.name.clone())
        .collect();
    eprintln!("comparing {} quantized 2-D tensors", names.len());

    let mut rows: Vec<Row> = Vec::new();
    for name in &names {
        let t8 = h8.tensors().iter().find(|t| &t.name == name).unwrap();
        let (m, k) = (t8.shape[0] as usize, t8.shape[1] as usize);
        if k % GROUP != 0 {
            continue;
        }
        let (_, b8) = h8.tensor_data_vec(name).expect("oq8 data");
        let w8 = dequant_opus(&b8, m, k, 8);

        // Per-input-channel second moment in THIS basis: d_j / s_j², group-mean
        // (see header). None where no imatrix was captured.
        let awq = format!("{}.awq_scale.weight", name.trim_end_matches(".weight"));
        let s_awq = h8.tensor_data_vec(&awq).map(|(ti, b)| {
            assert_eq!(ti.shape[0] as usize, k, "awq length != K on {awq}");
            read_f16_vec(&b)
        });
        let gmean: Option<Vec<f64>> = imat.get(name).map(|d| {
            assert_eq!(d.len(), k, "imatrix length != K on {name}");
            let ng = k / GROUP;
            (0..ng)
                .map(|g| {
                    let mut acc = 0f64;
                    for j in g * GROUP..(g + 1) * GROUP {
                        let sj = s_awq.as_ref().map(|s| s[j]).unwrap_or(1.0) as f64;
                        acc += if sj.abs() > 1e-12 { d[j] as f64 / (sj * sj) } else { 0.0 };
                    }
                    acc / GROUP as f64
                })
                .collect()
        });

        // Simulate the oq4 codec's core step: symmetric absmax int4 per 256-group.
        let ng = k / GROUP;
        let mut sse = 0f64;
        let mut refe = 0f64;
        let mut sse_w = 0f64;
        // Equal-bit alternatives, evaluated in the SAME metric:
        //   int8            = promote the whole tensor (+4.0 bits/weight)
        //   int4 + N outlier = the OqPlusCompact shape, N exact (u8 idx, i8 val)
        //                      pairs per 256-group = +N/16 bits/weight
        let mut sse8 = 0f64;
        let mut sse8_w = 0f64;
        let mut sse_out = [0f64; OUTLIER_NS.len()];
        let mut sse_out_w = [0f64; OUTLIER_NS.len()];
        let mut err_buf: Vec<(f64, usize)> = Vec::with_capacity(GROUP);
        for row in 0..m {
            for g in 0..ng {
                let seg = &w8[row * k + g * GROUP..row * k + (g + 1) * GROUP];
                let amax = seg.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let scale4 = if amax > 0.0 { amax / 7.0 } else { 1.0 };
                let scale8 = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                let mut gsse = 0f64;
                let mut gsse8 = 0f64;
                err_buf.clear();
                for (j, &v) in seg.iter().enumerate() {
                    let q4 = (v / scale4).round().clamp(-7.0, 7.0);
                    let d4 = (v - q4 * scale4) as f64;
                    gsse += d4 * d4;
                    err_buf.push((d4 * d4, j));
                    let q8 = (v / scale8).round().clamp(-127.0, 127.0);
                    let d8 = (v - q8 * scale8) as f64;
                    gsse8 += d8 * d8;
                    refe += (v as f64) * (v as f64);
                }
                sse += gsse;
                sse8 += gsse8;
                if let Some(gm) = gmean.as_ref() {
                    sse_w += gm[g] * gsse;
                    sse8_w += gm[g] * gsse8;
                }
                // Outliers: patch the worst-error elements exactly (i8 payload —
                // residual after patching is the int8 error, not zero).
                err_buf.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                for (ni, &nout) in OUTLIER_NS.iter().enumerate() {
                    let mut g_out = gsse;
                    for e in err_buf.iter().take(nout) {
                        let j = e.1;
                        let v = seg[j];
                        let q8 = (v / scale8).round().clamp(-127.0, 127.0);
                        let d8 = (v - q8 * scale8) as f64;
                        g_out += d8 * d8 - e.0; // replace int4 error with int8 error
                    }
                    sse_out[ni] += g_out;
                    if let Some(gm) = gmean.as_ref() {
                        sse_out_w[ni] += gm[g] * g_out;
                    }
                }
            }
        }
        if !sse.is_finite() || !refe.is_finite() {
            eprintln!("  NON-FINITE {name} — dropped");
            continue;
        }
        let (layer, site) = parse_layer_site(name).unwrap_or((usize::MAX, name.clone()));
        rows.push(Row {
            layer,
            site,
            n: m * k,
            sse,
            ref_energy: refe,
            sse_w: gmean.as_ref().map(|_| sse_w),
            sse8,
            sse8_w,
            sse_out,
            sse_out_w,
        });
    }

    let total_sse: f64 = rows.iter().map(|r| r.sse).sum();
    let total_w: f64 = rows.iter().filter_map(|r| r.sse_w).sum();

    // ── Per-site rollup ────────────────────────────────────────────────────
    let mut by_site: BTreeMap<String, (f64, usize, f64)> = BTreeMap::new();
    for r in &rows {
        let e = by_site.entry(r.site.clone()).or_insert((0.0, 0, 0.0));
        e.0 += r.sse;
        e.1 += r.n;
        e.2 += r.ref_energy;
    }
    println!("\n=== per-SITE isotropic 4-bit sensitivity (‖W₈ − Q₄(W₈)‖²_F) ===");
    println!("{:<28} {:>12} {:>9} {:>12} {:>10}", "site", "SSE", "share", "elements", "rel err");
    let mut sites: Vec<_> = by_site.into_iter().collect();
    sites.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
    for (site, (sse, n, refe)) in &sites {
        println!(
            "{:<28} {:>12.4e} {:>8.1}% {:>12} {:>9.2}%",
            site,
            sse,
            100.0 * sse / total_sse,
            n,
            100.0 * (sse / refe).sqrt()
        );
    }

    // ── Per-layer rollup ───────────────────────────────────────────────────
    let mut by_layer: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for r in &rows {
        let e = by_layer.entry(r.layer).or_insert((0.0, 0.0));
        e.0 += r.sse;
        e.1 += r.sse_w.unwrap_or(0.0);
    }
    println!("\n=== per-LAYER isotropic sensitivity, ranked ===");
    let mut layers: Vec<_> = by_layer.iter().map(|(l, v)| (*l, v.0, v.1)).collect();
    layers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("{:<8} {:>12} {:>9} {:>12} {:>9}", "layer", "SSE", "share", "SSE(imat)", "share");
    for (l, sse, w) in &layers {
        println!(
            "{:<8} {:>12.4e} {:>8.2}% {:>12.4e} {:>8.2}%",
            l,
            sse,
            100.0 * sse / total_sse,
            w,
            if total_w > 0.0 { 100.0 * w / total_w } else { 0.0 }
        );
    }

    // ── Concentration: the number that decides the allocation question ─────
    let concentration = |mut v: Vec<f64>, total: f64, label: &str| {
        if total <= 0.0 {
            return;
        }
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let n = v.len();
        println!("\n--- concentration ({label}, {n} layers) ---");
        for frac in [0.10f64, 0.25, 0.50] {
            let take = ((n as f64 * frac).ceil() as usize).max(1);
            let s: f64 = v[..take].iter().sum();
            println!(
                "  top {:>3} layers ({:>4.0}% of layers) carry {:>5.1}% of the error",
                take,
                frac * 100.0,
                100.0 * s / total
            );
        }
        let even = total / n as f64;
        let gini_ish: f64 = v.iter().map(|x| (x - even).abs()).sum::<f64>() / (2.0 * total);
        println!("  Gini ≈ {gini_ish:.3}   (0 = perfectly even, 1 = one layer holds everything)");
    };
    concentration(layers.iter().map(|x| x.1).collect(), total_sse, "isotropic");
    if total_w > 0.0 {
        concentration(layers.iter().map(|x| x.2).collect(), total_w, "imatrix-weighted, down_proj");
    }

    // Does the imatrix weighting change the RANKING? If not, the isotropic
    // proxy (available for every site) can be trusted where no calib exists.
    let mut both: Vec<(usize, f64, f64)> =
        rows.iter().filter_map(|r| r.sse_w.map(|w| (r.layer, r.sse, w))).collect();
    if both.len() > 2 {
        both.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let iso_rank: Vec<usize> = both.iter().map(|x| x.0).collect();
        both.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        let w_rank: Vec<usize> = both.iter().map(|x| x.0).collect();
        let n = iso_rank.len() as f64;
        let pos = |v: &Vec<usize>, l: usize| v.iter().position(|&x| x == l).unwrap() as f64;
        let d2: f64 = iso_rank.iter().map(|&l| (pos(&iso_rank, l) - pos(&w_rank, l)).powi(2)).sum();
        let rho = 1.0 - 6.0 * d2 / (n * (n * n - 1.0));
        println!(
            "\n--- isotropic vs imatrix-weighted ranking (down_proj, {} layers) ---\n  Spearman ρ = {rho:.3}",
            iso_rank.len()
        );
    }
    // ── The allocation question, both strategies in ONE metric ─────────────
    // A: uniform int4 + N outliers per group (the OqPlusCompact shape).
    // B: int4 everywhere, then promote whole tensors to int8, greedily by
    //    sensitivity-per-parameter, until the same average bit budget is spent.
    let alloc = |weighted: bool, label: &str| {
        let sel = |r: &Row| if weighted { r.sse_w.unwrap_or(0.0) } else { r.sse };
        let sel8 = |r: &Row| if weighted { r.sse8_w } else { r.sse8 };
        let sel_out = |r: &Row, i: usize| if weighted { r.sse_out_w[i] } else { r.sse_out[i] };
        let pool: Vec<&Row> =
            rows.iter().filter(|r| !weighted || r.sse_w.is_some()).collect();
        if pool.is_empty() {
            return;
        }
        let base: f64 = pool.iter().map(|r| sel(r)).sum();
        let floor: f64 = pool.iter().map(|r| sel8(r)).sum();
        let total_params: f64 = pool.iter().map(|r| r.n as f64).sum();
        println!("\n=== EQUAL-BIT ALLOCATION ({label}, {} tensors) ===", pool.len());
        println!("  int4 baseline error {base:.4e}   int8 floor {floor:.4e}");
        println!(
            "{:>10} {:>12} {:>26} {:>26}",
            "bits/wt", "budget", "A: outliers/group", "B: promote tensors"
        );
        for (i, &nout) in OUTLIER_NS.iter().enumerate() {
            let extra = nout as f64 / 16.0;
            let a: f64 = pool.iter().map(|r| sel_out(r, i)).sum();
            // B: greedy by (error removed) / (params spent), promote to int8.
            let mut cand: Vec<(f64, f64, f64)> = pool
                .iter()
                .map(|r| {
                    let gain = sel(r) - sel8(r);
                    (gain / r.n.max(1) as f64, gain, r.n as f64)
                })
                .collect();
            cand.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            let budget_params = total_params * (extra / 4.0); // +4 bits per promoted weight
            let (mut spent, mut removed) = (0f64, 0f64);
            for (_, gain, n) in &cand {
                if spent + n > budget_params {
                    continue; // whole-tensor granularity: cannot part-promote
                }
                spent += n;
                removed += gain;
            }
            let b = base - removed;
            println!(
                "{:>10.4} {:>12} {:>17.4e} ({:>4.1}%) {:>17.4e} ({:>4.1}%)",
                4.0625 + extra,
                format!("N={nout}"),
                a,
                100.0 * (base - a) / (base - floor),
                b,
                100.0 * (base - b) / (base - floor),
            );
        }
        println!("  (% = share of the int4→int8 error gap closed at that budget)");
    };
    alloc(false, "isotropic, all sites");
    alloc(true, "imatrix-weighted, down_proj");

    println!("\ntotal isotropic SSE {total_sse:.4e}   total weighted SSE {total_w:.4e}");
}
