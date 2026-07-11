//! Hardware parity and timing for the resident EmbeddingGemma FFN schedules.
//! Usage: `npu_resident_ffn_verify CACHE GATEUP_PACKER GATE_EXEC_OR_DOWN_PACKER [ITERS]`

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_primitives::fwht::gen_fwht_signs;
    use hipfire_xdna::{NpuGemmWholeScaled, NpuKernel};

    const M: usize = 256;
    const PAD_M: usize = 288;
    const K: usize = 768;
    const GROUP: usize = 256;
    const GATE_GROUPS: usize = 3;
    const INTER: usize = 1152;
    const PHYSICAL_GATE_N: usize = 2304;
    const DOWN_K: usize = 1280;
    const DOWN_GROUPS: usize = 5;
    const N: usize = 768;
    const COLS: usize = 8;
    const AB: usize = 6656;
    const PACKER_AB: usize = 8192;
    const WB: usize = 15872;
    const PACKER_WB: usize = 16384;
    const W_BLOCKS: usize = 42;

    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err(
            "usage: npu_resident_ffn_verify CACHE GATEUP_PACKER GATE_EXEC_OR_DOWN_PACKER [ITERS]"
                .into(),
        );
    }
    let resident_shape = std::fs::read_to_string(format!("{}/shape.txt", args[0]))?;
    let dense_w8 = resident_shape.lines().any(|line| line == "mode=dense-w8");
    let iterations = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(20);
    let timing_only = std::env::var_os("HIPFIRE_R25_TIMING_ONLY").is_some();
    let isolated_group = std::env::var("HIPFIRE_R25_GROUP")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let identity_down = std::env::var_os("HIPFIRE_R25_IDENTITY").is_some();

    let gate_activations: Vec<i8> = (0..M * K)
        .map(|index| signed_sample(index as u64 ^ 0xa17e_5eed, 63))
        .collect();
    let gate_activation_scales: Vec<f32> = (0..GATE_GROUPS)
        .flat_map(|group| (0..M).map(move |row| 0.015 + ((row + group * 7) % 19) as f32 * 0.0003))
        .collect();
    let mut gate_weights = vec![vec![0i8; GROUP * PHYSICAL_GATE_N]; GATE_GROUPS];
    let mut gate_scales = vec![vec![0.0f32; PHYSICAL_GATE_N]; GATE_GROUPS];
    for group in 0..GATE_GROUPS {
        for physical_col in 0..PHYSICAL_GATE_N {
            let (role, logical_col) = decode_gate_col(physical_col);
            gate_scales[group][physical_col] =
                0.008 + ((logical_col + role * 11 + group * 5) % 23) as f32 * 0.0002;
            for kk in 0..GROUP {
                let seed = kk as u64
                    ^ (logical_col as u64).wrapping_mul(0x9e37_79b9)
                    ^ (role as u64) << 47
                    ^ (group as u64) << 53;
                gate_weights[group][kk * PHYSICAL_GATE_N + physical_col] = signed_sample(seed, 15);
            }
        }
    }

    let gate_packer = NpuGemmWholeScaled::load_cached(&args[1])?;
    let packed_a = gate_packer.prepack_activations(&gate_activations, &gate_activation_scales)?;
    let mut resident_a = vec![0u8; packed_a.len() / PACKER_AB * AB];
    for block in 0..packed_a.len() / PACKER_AB {
        resident_a[block * AB..(block + 1) * AB]
            .copy_from_slice(&packed_a[block * PACKER_AB..block * PACKER_AB + AB]);
    }
    let gate_weight_refs = gate_weights.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let gate_scale_refs = gate_scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let packed_gate_w = gate_packer.prepack_weights(&gate_weight_refs, &gate_scale_refs)?;

    let down_weights: Vec<Vec<i8>> = (0..DOWN_GROUPS)
        .map(|group| {
            (0..GROUP * N)
                .map(|index| {
                    if isolated_group.is_none_or(|selected| selected == group) {
                        if identity_down {
                            i8::from(index / N == index % N % GROUP)
                        } else {
                            ((index * 11 + index / N * 5 + group * 7) % 15) as i8 - 7
                        }
                    } else {
                        0
                    }
                })
                .collect()
        })
        .collect();
    let down_scales: Vec<Vec<f32>> = (0..DOWN_GROUPS)
        .map(|group| {
            (0..N)
                .map(|col| {
                    if isolated_group.is_some_and(|selected| selected != group) {
                        0.0
                    } else if identity_down {
                        1.0
                    } else {
                        0.008 + ((col + group * 3) % 31) as f32 * 0.0001
                    }
                })
                .collect()
        })
        .collect();
    let awq: Vec<f32> = (0..DOWN_K)
        .map(|col| 0.7 + (col % 23) as f32 * 0.027)
        .collect();
    let signs1 = gen_fwht_signs(42, GROUP);
    let signs2 = gen_fwht_signs(1042, GROUP);

    if dense_w8 {
        return run_dense_w8(
            &args[0],
            &args[2],
            iterations,
            &packed_a,
            &packed_gate_w,
            &gate_activations,
            &gate_activation_scales,
            &gate_weights,
            &gate_scales,
            &down_weights,
            &down_scales,
            &awq,
            &signs1,
            &signs2,
        );
    }

    let mut packed_w = vec![0u8; COLS * W_BLOCKS * WB];
    for stripe in 0..COLS {
        let mut destination_block = 0;
        for m_macro in 0..3 {
            for n_macro in 0..3 {
                let gate_outblock = m_macro * 3 + n_macro;
                for group in 0..GATE_GROUPS {
                    let source_block =
                        (stripe * 27 + gate_outblock * GATE_GROUPS + group) * PACKER_WB;
                    let destination = (stripe * W_BLOCKS + destination_block) * WB;
                    packed_w[destination..destination + WB]
                        .copy_from_slice(&packed_gate_w[source_block..source_block + WB]);
                    destination_block += 1;
                }
                let ready: &[usize] = match n_macro {
                    0 => &[0],
                    1 => &[1, 2],
                    _ => &[3, 4],
                };
                for &group in ready {
                    let destination = (stripe * W_BLOCKS + destination_block) * WB;
                    pack_down_block(
                        &mut packed_w[destination..destination + WB],
                        stripe,
                        group,
                        &down_weights,
                        &down_scales,
                        &awq,
                        &signs1,
                        &signs2,
                    );
                    destination_block += 1;
                }
            }
        }
        assert_eq!(destination_block, W_BLOCKS);
    }

    let cpu_geglu = gate_reference(
        &gate_activations,
        &gate_activation_scales,
        &gate_weights,
        &gate_scales,
    );
    let geglu = {
        let xclbin = std::fs::read(format!("{}/final.xclbin", args[2]))?;
        let insts = std::fs::read(format!("{}/insts.bin", args[2]))?;
        let kernel = NpuKernel::load(&xclbin, &insts)?;
        let mut a = kernel.alloc_arg(packed_a.len())?;
        let mut w = kernel.alloc_arg(packed_gate_w.len())?;
        let o = kernel.alloc_arg(PAD_M * INTER * size_of::<f32>())?;
        a.as_mut_slice().copy_from_slice(&packed_a);
        w.as_mut_slice().copy_from_slice(&packed_gate_w);
        kernel.dispatch_synced(&[&a, &w, &o], &[true, true, true])?;
        kernel.sync_output(&o)?;
        unsafe { as_f32(o.as_slice())[..M * INTER].to_vec() }
    };
    let (gate_cosine, gate_max_abs, _, _) = metrics(&geglu, &cpu_geglu);
    let (down_activations, down_activation_scales) = prepare_down(&geglu, &awq, &signs1, &signs2);
    let reference = down_reference(
        &down_activations,
        &down_activation_scales,
        &down_weights,
        &down_scales,
    );

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let insts = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let mut kernel = NpuKernel::load(&xclbin, &insts)?;
    let mut a = kernel.alloc_arg(resident_a.len())?;
    let gate_only_stream = std::env::var_os("HIPFIRE_R25_GATE_ONLY_STREAM").is_some();
    let w_payload = if gate_only_stream {
        packed_gate_w.as_slice()
    } else {
        packed_w.as_slice()
    };
    let mut w = kernel.alloc_arg(w_payload.len())?;
    let c = kernel.alloc_arg(PAD_M * N * size_of::<f32>())?;
    a.as_mut_slice().copy_from_slice(&resident_a);
    w.as_mut_slice().copy_from_slice(w_payload);
    kernel.dispatch_synced(&[&a, &w, &c], &[true, true, true])?;
    kernel.sync_output(&c)?;
    let output = unsafe { as_f32(c.as_slice()) };
    if std::env::var_os("HIPFIRE_R25_RAW_GATE_PROBE").is_some() {
        let raw_group = std::env::var("HIPFIRE_R25_RAW_GROUP")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(2);
        let raw_nblock = std::env::var("HIPFIRE_R25_RAW_NBLOCK")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let probe_row = std::env::var("HIPFIRE_R25_PROBE_GLOBAL_ROW")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(219);
        let raw_source_jn = std::env::var("HIPFIRE_R25_RAW_SOURCE_JN")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?;
        let expected = raw_gate_reference(
            &gate_activations,
            &gate_activation_scales,
            &gate_weights,
            &gate_scales,
            raw_group,
            raw_nblock,
        );
        let full_got_row = &output[probe_row * N..(probe_row + 1) * N];
        let full_expected_row = &expected[probe_row * N..(probe_row + 1) * N];
        let (got_values, expected_values) = if let Some(source_jn) = raw_source_jn {
            let got = (0..COLS)
                .flat_map(|stripe| full_got_row[stripe * 96..stripe * 96 + 16].iter().copied())
                .collect::<Vec<_>>();
            let want = (0..COLS)
                .flat_map(|stripe| {
                    let start = stripe * 96 + source_jn * 16;
                    full_expected_row[start..start + 16].iter().copied()
                })
                .collect::<Vec<_>>();
            (got, want)
        } else {
            (full_got_row.to_vec(), full_expected_row.to_vec())
        };
        let got_row = got_values.as_slice();
        let expected_row = expected_values.as_slice();
        let (cosine, max_abs, mean_abs, max_reference) = metrics(got_row, expected_row);
        let allowed = 2.0e-5 + max_reference * 2.0e-5;
        let mismatches = got_row
            .iter()
            .zip(expected_row)
            .enumerate()
            .filter_map(|(index, (&got, &want))| {
                ((got - want).abs() > allowed).then_some((index, got, want))
            })
            .take(24)
            .collect::<Vec<_>>();
        let closest = (0..16)
            .map(|expected_col| {
                let want = expected_row[expected_col];
                let (got_col, error) = got_row
                    .iter()
                    .enumerate()
                    .map(|(col, &got)| (col, (got - want).abs()))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap();
                (expected_col, got_col, error)
            })
            .collect::<Vec<_>>();
        let nonzero_ranges = {
            let mut ranges = Vec::new();
            let mut start = None;
            for (col, &value) in got_row.iter().chain(std::iter::once(&0.0)).enumerate() {
                if value != 0.0 && start.is_none() {
                    start = Some(col);
                } else if value == 0.0 && start.is_some() {
                    ranges.push((start.take().unwrap(), col));
                }
            }
            ranges
        };
        let nonzero_pairs = got_row
            .iter()
            .zip(expected_row)
            .filter(|(got, _)| **got != 0.0)
            .map(|(&got, &want)| (got, want))
            .collect::<Vec<_>>();
        let nonzero_got = nonzero_pairs
            .iter()
            .map(|&(got, _)| got)
            .collect::<Vec<_>>();
        let nonzero_expected = nonzero_pairs
            .iter()
            .map(|&(_, want)| want)
            .collect::<Vec<_>>();
        let nonzero_metrics = metrics(&nonzero_got, &nonzero_expected);
        let block_candidates = raw_source_jn.map(|_| {
            (0..COLS)
                .map(|stripe| {
                    let got = &got_row[stripe * 16..stripe * 16 + 16];
                    (0..N / 16)
                        .map(|block| {
                            let want = &full_expected_row[block * 16..block * 16 + 16];
                            let (block_cosine, block_max_abs, _, _) = metrics(got, want);
                            (stripe, block, block_cosine, block_max_abs)
                        })
                        .max_by(|a, b| a.2.total_cmp(&b.2))
                        .unwrap()
                })
                .collect::<Vec<_>>()
        });
        let mut row_candidates = (0..M)
            .map(|row| {
                let candidate = if let Some(source_jn) = raw_source_jn {
                    (0..COLS)
                        .flat_map(|stripe| {
                            let start = row * N + stripe * 96 + source_jn * 16;
                            expected[start..start + 16].iter().copied()
                        })
                        .collect::<Vec<_>>()
                } else {
                    expected[row * N..(row + 1) * N].to_vec()
                };
                let (candidate_cosine, candidate_max_abs, _, _) = metrics(got_row, &candidate);
                let nonzero = candidate.iter().filter(|&&value| value != 0.0).count();
                (row, candidate_cosine, candidate_max_abs, nonzero)
            })
            .collect::<Vec<_>>();
        row_candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        row_candidates.truncate(12);
        let k_prefix_candidates = raw_source_jn.map(|source_jn| {
            [16usize, 32, 64, 128, 192, 256]
                .into_iter()
                .map(|k_prefix| {
                    let mut candidate = Vec::with_capacity(COLS * 16);
                    for stripe in 0..COLS {
                        for lane in 0..16 {
                            let pcol = raw_nblock * N + stripe * 96 + source_jn * 16 + lane;
                            let dot: i32 = (0..k_prefix)
                                .map(|kk| {
                                    gate_activations[probe_row * K + kk] as i32
                                        * gate_weights[0][kk * PHYSICAL_GATE_N + pcol] as i32
                                })
                                .sum();
                            candidate.push(
                                dot as f32
                                    * gate_activation_scales[probe_row]
                                    * gate_scales[0][pcol],
                            );
                        }
                    }
                    let (candidate_cosine, candidate_max_abs, _, _) = metrics(got_row, &candidate);
                    (k_prefix, candidate_cosine, candidate_max_abs)
                })
                .collect::<Vec<_>>()
        });
        eprintln!(
            "raw-gate probe group={raw_group} nblock={raw_nblock} row={probe_row} source_jn={raw_source_jn:?} cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} allowed={allowed:.7} nonzero_ranges={nonzero_ranges:?} nonzero_metrics={nonzero_metrics:?} block_candidates={block_candidates:?} row_candidates={row_candidates:?} k_prefix_candidates={k_prefix_candidates:?} closest={closest:?} mismatches={mismatches:?}"
        );
        return if cosine >= 0.999_999 && max_abs <= allowed {
            Ok(())
        } else {
            Err("resident FFN raw-gate probe failed".into())
        };
    }
    if std::env::var_os("HIPFIRE_R25_INPUT_PROBE").is_some() {
        let physical = unsafe { as_i32(c.as_slice()) };
        let probe_mblock = std::env::var("HIPFIRE_R25_PROBE_MBLOCK")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(1);
        let probe_core_row = std::env::var("HIPFIRE_R25_PROBE_CORE_ROW")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let probe_nblock = std::env::var("HIPFIRE_R25_PROBE_NBLOCK")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let row = probe_mblock * 96
            + probe_core_row * 24
            + std::env::var("HIPFIRE_R25_PROBE_ROW")
                .ok()
                .map(|value| value.parse::<usize>())
                .transpose()?
                .unwrap_or(0);
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(2166136261u32, |hash, &byte| {
                (hash ^ byte as u32).wrapping_mul(16777619)
            })
        };
        let mut results = Vec::with_capacity(COLS * GATE_GROUPS);
        for stripe in 0..COLS {
            for group in 0..GATE_GROUPS {
                let got_a = physical[row * N + stripe * 96 + 2 * group] as u32;
                let got_w = physical[row * N + stripe * 96 + 2 * group + 1] as u32;
                let a_base =
                    (probe_core_row * 27 + probe_mblock * 9 + probe_nblock * 3 + group) * AB;
                let nblock_offset = [0, 4, 9][probe_nblock];
                let w_base = (stripe * W_BLOCKS + probe_mblock * 14 + nblock_offset + group) * WB;
                let expected_a = fnv(&resident_a[a_base..a_base + 6240]);
                let expected_w = fnv(&packed_w[w_base..w_base + WB]);
                results.push((stripe, group, got_a, expected_a, got_w, expected_w));
            }
        }
        eprintln!("gate-input probe row={row} results={results:#x?}");
        let first_a = results[0].2;
        let a_candidates = resident_a
            .chunks_exact(AB)
            .enumerate()
            .filter_map(|(block, bytes)| (fnv(&bytes[..6240]) == first_a).then_some(block))
            .collect::<Vec<_>>();
        eprintln!("gate-input A candidates for first hash={a_candidates:?}");
        return if results
            .iter()
            .all(|(_, _, got_a, expected_a, got_w, expected_w)| {
                got_a == expected_a && got_w == expected_w
            }) {
            Ok(())
        } else {
            Err("resident FFN gate-input probe failed".into())
        };
    }
    if std::env::var_os("HIPFIRE_R25_GATE_PROBE").is_some() {
        let probe_row = std::env::var("HIPFIRE_R25_PROBE_ROW")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let global_row = std::env::var("HIPFIRE_R25_PROBE_GLOBAL_ROW")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?;
        let expected_gate = geglu.as_slice();
        let mismatches = (0..M)
            .filter(|row| row % 24 == probe_row && global_row.is_none_or(|target| *row == target))
            .flat_map(|row| {
                (0..384).filter_map(move |col| {
                    let stripe = col / 48;
                    let local_col = col % 48;
                    let got = output[row * N + stripe * 96 + local_col];
                    let expected = expected_gate[row * INTER + col];
                    (got.to_bits() != expected.to_bits()).then_some((row, col, got, expected))
                })
            })
            .take(24)
            .collect::<Vec<_>>();
        eprintln!(
            "gate-row probe local_row={probe_row} global_row={global_row:?} mismatches={mismatches:?}"
        );
        if !timing_only {
            return if mismatches.is_empty() {
                Ok(())
            } else {
                Err("resident FFN gate-row probe failed".into())
            };
        }
    }
    if std::env::var_os("HIPFIRE_R25_PROBE").is_some() {
        let physical = unsafe { as_i32(c.as_slice()) };
        let probe_group = std::env::var("HIPFIRE_R25_PROBE_GROUP")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(1);
        let probe_row = std::env::var("HIPFIRE_R25_PROBE_ROW")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let expected_activations = down_activations.as_slice();
        let mismatches = (0..M)
            .filter(|row| row % 24 == probe_row)
            .flat_map(|row| {
                (0..GROUP).filter_map(move |inner| {
                    let stripe = inner / 32;
                    let lane = inner % 32;
                    let got = physical[row * N + stripe * 96 + lane] as i8;
                    let expected = expected_activations[row * DOWN_K + probe_group * GROUP + inner];
                    (got != expected).then_some((row, inner, got, expected))
                })
            })
            .take(24)
            .collect::<Vec<_>>();
        eprintln!(
            "activation-row probe group={probe_group} local_row={probe_row} mismatches={mismatches:?}"
        );
        for &row in mismatches
            .iter()
            .map(|(row, ..)| row)
            .collect::<std::collections::BTreeSet<_>>()
        {
            let got = (0..GROUP)
                .map(|inner| physical[row * N + inner / 32 * 96 + inner % 32] as i8)
                .collect::<Vec<_>>();
            let (closest_row, equal_lanes) = (0..M)
                .map(|candidate| {
                    let expected = &expected_activations[candidate * DOWN_K + probe_group * GROUP
                        ..candidate * DOWN_K + (probe_group + 1) * GROUP];
                    (
                        candidate,
                        got.iter().zip(expected).filter(|(a, b)| a == b).count(),
                    )
                })
                .max_by_key(|&(_, equal)| equal)
                .unwrap();
            eprintln!(
                "activation-row probe row={row} closest_expected_row={closest_row} equal_lanes={equal_lanes}/{GROUP}"
            );
        }
        return if mismatches.is_empty() {
            Ok(())
        } else {
            Err("resident FFN activation-row probe failed".into())
        };
    }
    let (cosine, max_abs, mean_abs, max_reference) = metrics(&output[..M * N], &reference);
    let max_allowed = 0.02 + max_reference * 0.015;
    if !timing_only && (cosine < 0.999 || max_abs > max_allowed) {
        let first = output[..M * N].iter().zip(&reference).enumerate().max_by(
            |(_, (got_a, ref_a)), (_, (got_b, ref_b))| {
                (*got_a - *ref_a).abs().total_cmp(&(*got_b - *ref_b).abs())
            },
        );
        if let Some((index, (&got, &expected))) = first {
            eprintln!(
                "largest mismatch row={} col={} got={got:.7} expected={expected:.7}",
                index / N,
                index % N
            );
            if identity_down {
                let row = index / N;
                let group = isolated_group.unwrap_or(0);
                let scale = down_activation_scales[row * DOWN_GROUPS + group];
                let closest = (0..GROUP)
                    .min_by(|&left, &right| {
                        let l = (down_activations[row * DOWN_K + group * GROUP + left] as f32
                            * scale
                            - got)
                            .abs();
                        let r = (down_activations[row * DOWN_K + group * GROUP + right] as f32
                            * scale
                            - got)
                            .abs();
                        l.total_cmp(&r)
                    })
                    .unwrap();
                eprintln!(
                    "identity expected_inner={} closest_inner={closest} closest_value={:.7} scale={scale:.7}",
                    index % N % GROUP,
                    down_activations[row * DOWN_K + group * GROUP + closest] as f32 * scale
                );
                let aie_scale = output[row * N..row * N + GROUP]
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0f32, f32::max)
                    / 127.0;
                let quant_mismatches = (0..GROUP)
                    .filter_map(|inner| {
                        let got_q = (output[row * N + inner] / aie_scale).round() as i32;
                        let expected_q =
                            down_activations[row * DOWN_K + group * GROUP + inner] as i32;
                        (got_q != expected_q).then_some((inner, got_q, expected_q))
                    })
                    .take(24)
                    .collect::<Vec<_>>();
                eprintln!(
                    "identity inferred_aie_scale={aie_scale:.7} cpu_scale={scale:.7} quant_mismatches={quant_mismatches:?}"
                );
            }
        }
        return Err(format!(
            "resident FFN parity failed: cosine={cosine:.8} max_abs={max_abs:.7} allowed={max_allowed:.7}"
        )
        .into());
    }

    for _ in 0..2 {
        kernel.dispatch_synced(&[&a, &w, &c], &[false, false, false])?;
    }
    let started = std::time::Instant::now();
    let trace_timing = std::env::var_os("HIPFIRE_R25_TIMING_TRACE").is_some();
    let fresh_command = std::env::var_os("HIPFIRE_R25_FRESH_COMMAND").is_some();
    let recycle_every = std::env::var("HIPFIRE_R25_RECYCLE_EVERY")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    for iteration in 0..iterations {
        let dispatch_started = std::time::Instant::now();
        if recycle_every.is_some_and(|every| every != 0 && iteration % every == 0) {
            kernel.recreate_hwctx()?;
        }
        if fresh_command {
            let inflight = kernel.submit_inflight(&[&a, &w, &c])?;
            kernel.wait_inflight(inflight)?;
        } else {
            kernel.dispatch_synced(&[&a, &w, &c], &[false, false, false])?;
        }
        if trace_timing {
            eprintln!(
                "resident-ffn dispatch[{iteration}]={:.4}ms",
                dispatch_started.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "resident-ffn-w4 M={M} K={K} I={INTER} N={N} group={isolated_group:?}: gate_cosine={gate_cosine:.8} gate_max_abs={gate_max_abs:.7} cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} dispatch_ms={dispatch_ms:.4}"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn run_dense_w8(
    cache: &str,
    down_packer_cache: &str,
    iterations: usize,
    packed_gate_a: &[u8],
    packed_gate_w: &[u8],
    gate_activations: &[i8],
    gate_activation_scales: &[f32],
    gate_weights: &[Vec<i8>],
    gate_scales: &[Vec<f32>],
    down_weights: &[Vec<i8>],
    down_scales: &[Vec<f32>],
    awq: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::{NpuGemmWholeScaled, NpuKernel, NpuWholeMode};

    const M: usize = 256;
    const PAD_M: usize = 288;
    const INTER: usize = 1152;
    const N: usize = 768;
    const COLS: usize = 8;
    const ROW_STRIPES: usize = 4;
    const GATE_BLOCKS: usize = 54;
    const DOWN_BLOCKS: usize = 30;
    const WEIGHT_BLOCKS: usize = GATE_BLOCKS + DOWN_BLOCKS;
    const PACKED_A_BLOCK: usize = 8192;
    const DATA_PAIR: usize = 9216;
    const DATA_JOIN: usize = 4 * DATA_PAIR;
    const W_BLOCK: usize = 16384;
    const W_DATA: usize = 12288;
    const W_COLS: usize = 48;
    const PARAM_OFFSET: usize = W_DATA + W_COLS * size_of::<f32>();
    const T_ROWS: usize = 296;
    const T_STRIDE: usize = 5376;

    if iterations == 0 {
        return Err("R26 verification needs at least one iteration".into());
    }
    if packed_gate_a.len() != ROW_STRIPES * GATE_BLOCKS * PACKED_A_BLOCK
        || packed_gate_w.len() != COLS * GATE_BLOCKS * W_BLOCK
    {
        return Err("R26 gate packer geometry mismatch".into());
    }

    // Each memory-tile input is linked to four core-column pairs. Replicate the
    // canonical W8 activation block into those four pair windows, retaining a
    // 1 KiB tail per pair for the later 288-float down gather.
    let mut data = vec![0u8; ROW_STRIPES * GATE_BLOCKS * DATA_JOIN];
    for stripe in 0..ROW_STRIPES {
        for block in 0..GATE_BLOCKS {
            let source = (stripe * GATE_BLOCKS + block) * PACKED_A_BLOCK;
            let destination = (stripe * GATE_BLOCKS + block) * DATA_JOIN;
            for pair in 0..4 {
                data[destination + pair * DATA_PAIR
                    ..destination + pair * DATA_PAIR + PACKED_A_BLOCK]
                    .copy_from_slice(&packed_gate_a[source..source + PACKED_A_BLOCK]);
            }
        }
    }

    let down_packer = NpuGemmWholeScaled::load_cached(down_packer_cache)?;
    if down_packer.mode() != NpuWholeMode::W8
        || down_packer.rows() != M
        || down_packer.k() != 1280
        || down_packer.n() != N
    {
        return Err("R26 down packer must be W8 M=256 K=1280 N=768".into());
    }
    let down_weight_refs = down_weights.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let down_scale_refs = down_scales.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let packed_down_w = down_packer.prepack_weights(&down_weight_refs, &down_scale_refs)?;
    if packed_down_w.len() != COLS * DOWN_BLOCKS * W_BLOCK {
        return Err("R26 packed down-weight geometry mismatch".into());
    }

    // Gate blocks are already in [mblock,nmacro,group] order. R26 consumes the
    // down blocks as [mblock,group,nmacro], so transpose those two stream axes.
    let mut weights = vec![0u8; COLS * WEIGHT_BLOCKS * W_BLOCK];
    for stripe in 0..COLS {
        let destination = stripe * WEIGHT_BLOCKS * W_BLOCK;
        for mblock in 0..3 {
            for nblock in 0..6 {
                for group in 0..3 {
                    let block = (mblock * 6 + nblock) * 3 + group;
                    let base = destination + block * W_BLOCK;
                    for ln in 0..3 {
                        for kt in 0..32 {
                            for kk in 0..8 {
                                for nn in 0..16 {
                                    let local = ln * 16 + nn;
                                    let (role, tail) = match local {
                                        0..16 => (0, local),
                                        16..32 => (1, local - 16),
                                        32..40 => (0, local - 16),
                                        _ => (1, local - 24),
                                    };
                                    let logical_col = (nblock * COLS + stripe) * 24 + tail;
                                    let physical_col =
                                        (logical_col / 48) * 96 + role * 48 + logical_col % 48;
                                    let index =
                                        (ln * 32 + kt) * 128 + (nn / 8) * 64 + kk * 8 + nn % 8;
                                    weights[base + index] = gate_weights[group]
                                        [(kt * 8 + kk) * 2304 + physical_col]
                                        as u8;
                                }
                            }
                        }
                    }
                    for local in 0..48 {
                        let (role, tail) = match local {
                            0..16 => (0, local),
                            16..32 => (1, local - 16),
                            32..40 => (0, local - 16),
                            _ => (1, local - 24),
                        };
                        let logical_col = (nblock * COLS + stripe) * 24 + tail;
                        let physical_col = (logical_col / 48) * 96 + role * 48 + logical_col % 48;
                        let offset = base + W_DATA + local * size_of::<f32>();
                        weights[offset..offset + size_of::<f32>()]
                            .copy_from_slice(&gate_scales[group][physical_col].to_ne_bytes());
                    }
                }
            }
        }
        for mblock in 0..3 {
            for group in 0..5 {
                for nmacro in 0..2 {
                    let source_block = (mblock * 2 + nmacro) * 5 + group;
                    let destination_block = GATE_BLOCKS + (mblock * 5 + group) * 2 + nmacro;
                    let source = (stripe * DOWN_BLOCKS + source_block) * W_BLOCK;
                    let destination = (stripe * WEIGHT_BLOCKS + destination_block) * W_BLOCK;
                    weights[destination..destination + W_BLOCK]
                        .copy_from_slice(&packed_down_w[source..source + W_BLOCK]);
                    let mut params = Vec::with_capacity(3 * 256);
                    params.extend_from_slice(&awq[group * 256..(group + 1) * 256]);
                    params.extend_from_slice(signs1);
                    params.extend_from_slice(signs2);
                    weights
                        [destination + PARAM_OFFSET..destination + PARAM_OFFSET + params.len() * 4]
                        .copy_from_slice(unsafe { as_bytes(&params) });
                }
            }
        }
    }

    let gate = gate_reference(
        gate_activations,
        gate_activation_scales,
        gate_weights,
        gate_scales,
    );
    let (down_activations, down_activation_scales) = prepare_down(&gate, awq, signs1, signs2);
    let reference = down_reference(
        &down_activations,
        &down_activation_scales,
        down_weights,
        down_scales,
    );

    let xclbin = std::fs::read(format!("{cache}/final.xclbin"))?;
    let insts = std::fs::read(format!("{cache}/insts.bin"))?;
    let mut kernel = NpuKernel::load(&xclbin, &insts)?;
    let mut d = kernel.alloc_arg(data.len())?;
    let mut w = kernel.alloc_arg(weights.len())?;
    let mut t = kernel.alloc_arg(T_ROWS * T_STRIDE * size_of::<f32>())?;
    let mut o = kernel.alloc_arg(PAD_M * N * size_of::<f32>())?;
    d.as_mut_slice().copy_from_slice(&data);
    w.as_mut_slice().copy_from_slice(&weights);
    t.as_mut_slice().fill(0);

    let warmups = std::env::var("HIPFIRE_R26_WARMUPS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let recycle_every = std::env::var("HIPFIRE_R26_RECYCLE_EVERY")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(6);
    if recycle_every < 2 {
        return Err(
            "HIPFIRE_R26_RECYCLE_EVERY must leave room for prime + measured command".into(),
        );
    }
    for attempt in 0..=warmups {
        if attempt != 0 {
            t.as_mut_slice().fill(0);
            o.as_mut_slice().fill(0);
        }
        kernel.dispatch_synced(
            &[&d, &w, &t, &o],
            &[attempt == 0, attempt == 0, true, attempt != 0],
        )?;
    }
    kernel.sync_output(&t)?;
    kernel.sync_output(&o)?;
    let physical_gate = unsafe { as_f32(t.as_slice()) };
    let mut retained_gate = vec![0.0f32; M * INTER];
    for row in 0..M {
        for col in 0..INTER {
            retained_gate[row * INTER + col] =
                physical_gate[row * T_STRIDE + (col / 24) * 96 + col % 24];
        }
    }
    let (gate_cosine, gate_max_abs, gate_mean_abs, _) = metrics(&retained_gate, &gate);
    let output = unsafe { as_f32(o.as_slice()) };
    let (cosine, max_abs, mean_abs, max_reference) = metrics(&output[..M * N], &reference);
    if !cosine.is_finite() || cosine < 0.999 || max_abs > max_reference * 0.03 + 1.0e-4 {
        return Err(format!(
            "resident dense-W8 FFN parity failed: gate_cosine={gate_cosine:.8} gate_max_abs={gate_max_abs:.7} gate_mean_abs={gate_mean_abs:.8} cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} max_reference={max_reference:.7}"
        )
        .into());
    }

    let started = std::time::Instant::now();
    let mut context_commands = warmups + 1;
    for _dispatch in 0..iterations {
        if context_commands >= recycle_every {
            kernel.recreate_hwctx()?;
            t.as_mut_slice().fill(0);
            o.as_mut_slice().fill(0);
            kernel.dispatch_synced(&[&d, &w, &t, &o], &[false, false, true, true])?;
            context_commands = 1;
            t.as_mut_slice().fill(0);
            o.as_mut_slice().fill(0);
            kernel.sync_to_device(&t)?;
            kernel.sync_to_device(&o)?;
        }
        kernel.dispatch_synced(&[&d, &w, &t, &o], &[false, false, false, false])?;
        context_commands += 1;
    }
    kernel.sync_output(&o)?;
    let (final_cosine, final_max_abs, _, _) =
        metrics(&unsafe { as_f32(o.as_slice()) }[..M * N], &reference);
    if final_cosine < 0.999 || final_max_abs > max_reference * 0.03 + 1.0e-4 {
        return Err(format!(
            "resident dense-W8 FFN sustained parity failed: cosine={final_cosine:.8} max_abs={final_max_abs:.7}"
        )
        .into());
    }
    let dispatch_ms = started.elapsed().as_secs_f64() * 1e3 / iterations as f64;
    println!(
        "resident-ffn-dense-w8 M={M} I={INTER} N={N}: gate_cosine={gate_cosine:.8} gate_max_abs={gate_max_abs:.7} cosine={cosine:.8} max_abs={max_abs:.7} mean_abs={mean_abs:.8} dispatch_ms={dispatch_ms:.4}"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn pack_down_block(
    block: &mut [u8],
    stripe: usize,
    group: usize,
    weights: &[Vec<i8>],
    scales: &[Vec<f32>],
    awq: &[f32],
    signs1: &[f32],
    signs2: &[f32],
) {
    const N: usize = 768;
    const GROUP: usize = 256;
    const W_DATA: usize = 12288;
    for ln in 0..6 {
        for kt in 0..16 {
            for kk in 0..16 {
                for nn in 0..16 {
                    let col = stripe * 96 + ln * 16 + nn;
                    let value = weights[group][(kt * 16 + kk) * N + col];
                    let index = (ln * 16 + kt) * 256 + kk * 16 + nn;
                    let nibble = (value & 0x0f) as u8;
                    block[index / 2] |= if index % 2 == 0 { nibble } else { nibble << 4 };
                }
            }
        }
    }
    for local_col in 0..96 {
        let col = stripe * 96 + local_col;
        let offset = W_DATA + local_col * 4;
        block[offset..offset + 4].copy_from_slice(&scales[group][col].to_ne_bytes());
    }
    let mut params = Vec::with_capacity(3 * GROUP);
    params.extend_from_slice(&awq[group * GROUP..(group + 1) * GROUP]);
    params.extend_from_slice(signs1);
    params.extend_from_slice(signs2);
    let offset = W_DATA + 96 * 4;
    block[offset..offset + params.len() * 4].copy_from_slice(unsafe { as_bytes(&params) });
}

#[cfg(target_os = "linux")]
fn decode_gate_col(physical: usize) -> (usize, usize) {
    let stripe = physical / 96;
    let local = physical % 96;
    if local < 48 {
        (0, stripe * 48 + local)
    } else {
        (1, stripe * 48 + local - 48)
    }
}

#[cfg(target_os = "linux")]
fn gate_reference(
    activations: &[i8],
    activation_scales: &[f32],
    weights: &[Vec<i8>],
    scales: &[Vec<f32>],
) -> Vec<f32> {
    const M: usize = 256;
    const K: usize = 768;
    const GROUP: usize = 256;
    const INTER: usize = 1152;
    const PN: usize = 2304;
    let mut output = vec![0.0; M * INTER];
    for row in 0..M {
        for col in 0..INTER {
            let mut projected = [0.0f32; 2];
            for (role, value) in projected.iter_mut().enumerate() {
                let stripe = col / 48;
                let pcol = stripe * 96 + role * 48 + col % 48;
                for group in 0..K / GROUP {
                    let dot: i32 = (0..GROUP)
                        .map(|kk| {
                            activations[row * K + group * GROUP + kk] as i32
                                * weights[group][kk * PN + pcol] as i32
                        })
                        .sum();
                    *value += dot as f32 * activation_scales[group * M + row] * scales[group][pcol];
                }
            }
            let gate = projected[0];
            let up = projected[1];
            let tanh = hipfire_primitives::conv::round_f32_to_bf16(
                (0.797_884_6 * (gate + 0.044_715 * gate.powi(3))).tanh(),
            );
            output[row * INTER + col] = 0.5 * gate * (1.0 + tanh) * up;
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn raw_gate_reference(
    activations: &[i8],
    activation_scales: &[f32],
    weights: &[Vec<i8>],
    scales: &[Vec<f32>],
    last_group: usize,
    nblock: usize,
) -> Vec<f32> {
    const M: usize = 256;
    const K: usize = 768;
    const GROUP: usize = 256;
    const PN: usize = 2304;
    const N: usize = 768;
    assert!(last_group < K / GROUP);
    assert!(nblock < PN / N);
    let mut output = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let pcol = nblock * N + col;
            let mut value = 0.0f32;
            for group in 0..=last_group {
                let dot: i32 = (0..GROUP)
                    .map(|kk| {
                        activations[row * K + group * GROUP + kk] as i32
                            * weights[group][kk * PN + pcol] as i32
                    })
                    .sum();
                value += dot as f32 * activation_scales[group * M + row] * scales[group][pcol];
            }
            output[row * N + col] = value;
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn prepare_down(input: &[f32], awq: &[f32], signs1: &[f32], signs2: &[f32]) -> (Vec<i8>, Vec<f32>) {
    use hipfire_primitives::fwht::cpu_fwht_256;
    const M: usize = 256;
    const INTER: usize = 1152;
    const K: usize = 1280;
    const GROUP: usize = 256;
    const GROUPS: usize = 5;
    let mut quantized = vec![0i8; M * K];
    let mut scales = vec![0.0f32; M * GROUPS];
    for row in 0..M {
        for group in 0..GROUPS {
            let mut rotated = vec![0.0; GROUP];
            for inner in 0..GROUP {
                let col = group * GROUP + inner;
                rotated[inner] = if col < INTER {
                    input[row * INTER + col] / awq[col]
                } else {
                    0.0
                };
            }
            cpu_fwht_256(&mut rotated, signs1, signs2);
            let max_abs = rotated
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            scales[row * GROUPS + group] = scale;
            if scale > 0.0 {
                for inner in 0..GROUP {
                    quantized[row * K + group * GROUP + inner] =
                        (rotated[inner] / scale).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }
    (quantized, scales)
}

#[cfg(target_os = "linux")]
fn down_reference(
    activations: &[i8],
    activation_scales: &[f32],
    weights: &[Vec<i8>],
    scales: &[Vec<f32>],
) -> Vec<f32> {
    const M: usize = 256;
    const K: usize = 1280;
    const GROUP: usize = 256;
    const GROUPS: usize = 5;
    const N: usize = 768;
    let mut output = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            for group in 0..GROUPS {
                let dot: i32 = (0..GROUP)
                    .map(|inner| {
                        activations[row * K + group * GROUP + inner] as i32
                            * weights[group][inner * N + col] as i32
                    })
                    .sum();
                output[row * N + col] +=
                    dot as f32 * activation_scales[row * GROUPS + group] * scales[group][col];
            }
        }
    }
    output
}

#[cfg(target_os = "linux")]
fn signed_sample(mut value: u64, modulus: i16) -> i8 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    ((value % modulus as u64) as i16 - modulus / 2) as i8
}

#[cfg(target_os = "linux")]
fn metrics(got: &[f32], expected: &[f32]) -> (f64, f32, f64, f32) {
    let mut dot = 0.0;
    let mut got_norm = 0.0;
    let mut ref_norm = 0.0;
    let mut max_abs = 0.0f32;
    let mut max_reference = 0.0f32;
    let mut sum_abs = 0.0;
    for (&got, &expected) in got.iter().zip(expected) {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        max_reference = max_reference.max(expected.abs());
        sum_abs += error as f64;
        dot += got as f64 * expected as f64;
        got_norm += (got as f64).powi(2);
        ref_norm += (expected as f64).powi(2);
    }
    (
        dot / (got_norm.sqrt() * ref_norm.sqrt()),
        max_abs,
        sum_abs / got.len() as f64,
        max_reference,
    )
}

#[cfg(target_os = "linux")]
unsafe fn as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

#[cfg(target_os = "linux")]
unsafe fn as_f32(values: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() / 4) }
}

#[cfg(target_os = "linux")]
unsafe fn as_i32(values: &[u8]) -> &[i32] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() / 4) }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("resident FFN verification is Linux-only");
}
