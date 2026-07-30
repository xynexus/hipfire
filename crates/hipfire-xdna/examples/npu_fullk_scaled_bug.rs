//! Isolate the scaled full-K discrepancy against the unscaled kernel.
//!
//! `run_resident_scaled` applies activation and weight scales ON THE ARRAY and
//! returns f32; `run_resident` returns raw per-group i32 the host then scales.
//! For the SAME weights and activations the two must agree:
//!
//!   scaled[row][col] == sum_group partial[group][row][col]
//!                       * act_scale[group][row] * weight_scale[group][col]
//!
//! `examples/npu_embeddinggemma_fullk_sweep.rs` runs the scaled kernel with
//! all-1.0 scales, which cannot detect a scale-application bug. This walks three
//! cases so the failure mode is pinned rather than guessed:
//!
//!   1. all scales 1.0        -> isolates "are scales applied at all"
//!   2. weight scales only    -> isolates the per-column term
//!   3. activation scales only-> isolates the per-(group,row) term
//!
//! Usage: npu_fullk_scaled_bug SCALED_CACHE UNSCALED_CACHE COLS

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::NpuGemmFullK;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 3 {
        return Err("usage: npu_fullk_scaled_bug SCALED_CACHE UNSCALED_CACHE COLS".into());
    }
    let cols: usize = args[2].parse()?;
    let mut scaled = NpuGemmFullK::load_cached(&args[0], cols)?;
    let mut plain = NpuGemmFullK::load_cached(&args[1], cols)?;
    if !scaled.scaled_output() {
        return Err("first cache is not a scaled build".into());
    }
    if plain.scaled_output() {
        return Err("second cache must be an unscaled build".into());
    }
    let (rows, k, n) = (scaled.rows(), scaled.k(), scaled.n());
    let groups = k / 256;
    if (plain.rows(), plain.k(), plain.n()) != (rows, k, n) {
        return Err("caches must share geometry".into());
    }
    println!("rows={rows} k={k} n={n} groups={groups} cols={cols}");

    // Deterministic small int4-range weights and int8 activations.
    let w = |g: usize, i: usize| (((g * 7 + i * 3) % 15) as i8) - 7;
    let base: Vec<Vec<i8>> = (0..groups)
        .map(|g| (0..256 * n).map(|i| w(g, i)).collect())
        .collect();
    let residual: Vec<Vec<i8>> = Vec::new();
    let activations: Vec<i8> = (0..rows * k).map(|i| (((i * 5) % 31) as i8) - 15).collect();

    let base_refs: Vec<&[i8]> = base.iter().map(Vec::as_slice).collect();
    let residual_refs: Vec<&[i8]> = residual.iter().map(Vec::as_slice).collect();

    // Unscaled reference: raw i32 partials, scaled on the host.
    let packed_plain = plain.prepack_weights(&base_refs, &residual_refs)?;
    let resident_plain = plain.upload_resident_weights(&packed_plain)?;
    let mut partials = vec![0i32; groups * rows * n];
    plain.run_resident(&resident_plain, &activations, &mut partials)?;

    // HIPFIRE_CASE_ORDER=reverse puts a non-unit case FIRST. If only the first
    // case of a run is ever correct, the fault is dispatch desynchronisation
    // (the cores loop forever over the fifos and a dispatch leaves them mid
    // slab), not scale handling — and "all-1.0 passes" was an artifact of it
    // always being tested first.
    // HIPFIRE_REPEAT_ONE separates the two remaining explanations for the
    // order-dependent corruption. Each normal case does prepack ->
    // upload_resident_weights -> run, so a failure at position 2+ could be
    // caused by the re-upload OR by the second dispatch. This mode uploads ONCE
    // and dispatches the identical input three times: if runs 2 and 3 still
    // diverge, the dispatch itself desyncs the array and the schedule is at
    // fault; if they agree, the fault is in upload_resident_weights.
    if std::env::var("HIPFIRE_REPEAT_ONE").is_ok() {
        // Vary ONE axis at a time. All-1.0 passes, so any axis that fails names
        // the index the kernel gets wrong. HIPFIRE_AXIS=wcol|wgroup|arow|agroup.
        let axis = std::env::var("HIPFIRE_AXIS").unwrap_or_else(|_| "wcol".into());
        let weight_scales: Vec<Vec<f32>> = (0..groups)
            .map(|g| {
                (0..n)
                    .map(|c| match axis.as_str() {
                        "wcol" => 1.0 + (c % 8) as f32,
                        "wgroup" => 1.0 + g as f32,
                        _ => 1.0,
                    })
                    .collect()
            })
            .collect();
        let act_scales: Vec<f32> = (0..groups * rows)
            .map(|i| match axis.as_str() {
                // laid out [group][row]
                "arow" => 1.0 + (i % rows) as f32,
                "agroup" => 1.0 + (i / rows) as f32,
                _ => 1.0,
            })
            .collect();
        let scale_refs: Vec<&[f32]> = weight_scales.iter().map(Vec::as_slice).collect();
        let packed = scaled.prepack_weights_with_scales(&base_refs, &residual_refs, &scale_refs)?;
        let resident = scaled.upload_resident_weights(&packed)?;
        let mut first: Option<Vec<f32>> = None;
        for run in 0..3 {
            let mut out = vec![0.0f32; rows * n];
            scaled.run_resident_scaled(&resident, &activations, &act_scales, &mut out)?;
            let mut want_err = 0.0f32;
            for row in 0..rows {
                for col in 0..n {
                    let mut want = 0.0f32;
                    for g in 0..groups {
                        want += partials[(g * rows + row) * n + col] as f32
                            * act_scales[g * rows + row]
                            * weight_scales[g][col];
                    }
                    want_err = want_err.max((out[row * n + col] - want).abs());
                }
            }
            let drift = first
                .as_ref()
                .map(|f| {
                    f.iter()
                        .zip(&out)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max)
                })
                .unwrap_or(0.0);
            println!(
                "run={run} max_abs_vs_reference={want_err:.6} max_abs_vs_run0={drift:.6} \
                 out[0]={:.4}",
                out[0]
            );
            first.get_or_insert(out);
        }
        return Ok(());
    }
    let mut cases = vec![
        ("all 1.0", 1.0f32, 1.0f32),
        ("weight only", 0.5f32, 1.0f32),
        ("activation only", 1.0f32, 0.25f32),
    ];
    if std::env::var("HIPFIRE_CASE_ORDER").as_deref() == Ok("reverse") {
        cases.reverse();
    }
    for (case, wscale, ascale) in cases {
        // Distinct-but-known values so a transposed or mis-strided index shows up.
        let weight_scales: Vec<Vec<f32>> = (0..groups)
            .map(|g| {
                (0..n)
                    .map(|c| wscale * (1.0 + (g + c) as f32 * 0.0))
                    .collect()
            })
            .collect();
        let act_scales: Vec<f32> = (0..groups * rows).map(|_| ascale).collect();
        let scale_refs: Vec<&[f32]> = weight_scales.iter().map(Vec::as_slice).collect();

        let packed = scaled.prepack_weights_with_scales(&base_refs, &residual_refs, &scale_refs)?;
        let resident = scaled.upload_resident_weights(&packed)?;
        let mut out = vec![0.0f32; rows * n];
        scaled.run_resident_scaled(&resident, &activations, &act_scales, &mut out)?;

        if std::env::var("HIPFIRE_DUMP_STAGE").is_ok() {
            let f = |r: std::ops::Range<usize>| {
                out[r]
                    .iter()
                    .take(4)
                    .map(|v| (v * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>()
            };
            println!(
                "  stage case={case}: to_float[0..4]={:?} weight_scale[0..4]={:?} after_mul1[0..4]={:?}",
                f(0..16), f(16..32), f(32..48)
            );
            continue;
        }
        if std::env::var("HIPFIRE_DUMP_PAYLOAD").is_ok() {
            // With the dump probe kernel loaded, row 0 of the output holds the
            // scale payload the core received: ROWS activation scales then
            // SLAB_N weight scales.
            let rows_per_core = rows / cols;
            let act: Vec<f32> = out[..rows_per_core].to_vec();
            let wt: Vec<f32> = out[rows_per_core..rows_per_core + 8].to_vec();
            println!(
                "  payload case={case}: act[0..{}]={:?} wt[0..8]={:?}",
                rows_per_core, act, wt
            );
            continue;
        }
        let mut worst = 0.0f32;
        let mut worst_at = (0usize, 0usize);
        let mut ratio_sum = 0.0f64;
        let mut ratio_n = 0usize;
        for row in 0..rows {
            for col in 0..n {
                let mut want = 0.0f32;
                for g in 0..groups {
                    want += partials[(g * rows + row) * n + col] as f32
                        * act_scales[g * rows + row]
                        * weight_scales[g][col];
                }
                let got = out[row * n + col];
                let err = (got - want).abs();
                if err > worst {
                    worst = err;
                    worst_at = (row, col);
                }
                if want.abs() > 1e-3 {
                    ratio_sum += (got / want) as f64;
                    ratio_n += 1;
                }
            }
        }
        // Diagnose the mechanism at the worst cell: print every group's
        // contribution so "only the last group survived" (init overwriting
        // instead of accumulating) is distinguishable from a scale-indexing
        // fault or from reading scales out of the wrong buffer.
        {
            // Fixed cell across all cases so the three are comparable.
            let (r, c) = (11usize, 2usize);
            let mut parts = Vec::new();
            for g in 0..groups {
                parts.push(
                    partials[(g * rows + r) * n + c] as f32
                        * act_scales[g * rows + r]
                        * weight_scales[g][c],
                );
            }
            let sum: f32 = parts.iter().sum();
            let last = *parts.last().unwrap();
            let first = parts[0];
            println!(
                "  probe row={r} col={c} got={:.4} sum={sum:.4} first={first:.4} last={last:.4} \
                 per_group={:?}",
                out[r * n + c],
                parts
                    .iter()
                    .map(|v| (v * 100.0).round() / 100.0)
                    .collect::<Vec<_>>()
            );
        }
        let (r, c) = worst_at;
        let mut want0 = 0.0f32;
        for g in 0..groups {
            want0 += partials[(g * rows + r) * n + c] as f32
                * act_scales[g * rows + r]
                * weight_scales[g][c];
        }
        println!(
            "case={case:16} max_abs={worst:.6} at(row={r},col={c}) got={:.6} want={want0:.6} \
             mean_ratio={:.6}",
            out[r * n + c],
            if ratio_n > 0 {
                ratio_sum / ratio_n as f64
            } else {
                f64::NAN
            }
        );
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_fullk_scaled_bug requires Linux + XDNA");
}
