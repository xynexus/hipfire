//! Resident one-dispatch full-K projection-inventory sweep for EmbeddingGemma.
//!
//! This times the production `NpuGemmFullK` seam: row-major host activation
//! packing, one XRT/AIE dispatch, and exact int32 partial reconstruction. It
//! excludes FWHT/quantization, group-scale reconstruction, attention, norms,
//! pooling, and Dense activation, so it is a hybrid projection measurement and
//! never a full-model throughput claim.

#[cfg(target_os = "linux")]
use std::{path::Path, time::Instant};

#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    k: usize,
    n: usize,
    repeats: usize,
}

#[cfg(target_os = "linux")]
const SHAPES: &[Shape] = &[
    Shape {
        name: "q_proj",
        k: 768,
        n: 768,
        repeats: 24,
    },
    Shape {
        name: "k_proj",
        k: 768,
        n: 256,
        repeats: 24,
    },
    Shape {
        name: "v_proj",
        k: 768,
        n: 256,
        repeats: 24,
    },
    Shape {
        name: "o_proj",
        k: 768,
        n: 768,
        repeats: 24,
    },
    Shape {
        name: "gate_proj",
        k: 768,
        n: 1152,
        repeats: 24,
    },
    Shape {
        name: "up_proj",
        k: 768,
        n: 1152,
        repeats: 24,
    },
    Shape {
        name: "down_proj",
        k: 1152,
        n: 768,
        repeats: 24,
    },
    Shape {
        name: "dense.0",
        k: 768,
        n: 3072,
        repeats: 1,
    },
    Shape {
        name: "dense.1",
        k: 3072,
        n: 768,
        repeats: 1,
    },
];

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::{NpuFullKMode, NpuGemmFullK};

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: npu_embeddinggemma_fullk_sweep CACHE_ROOT [MODES] [ITERS]".into());
    }
    let root = Path::new(&args[0]);
    let modes: Vec<&str> = args
        .get(1)
        .map_or("w4,mixed,w8", String::as_str)
        .split(',')
        .collect();
    let iterations: usize = args.get(2).map_or(Ok(10), |value| value.parse())?;
    let rows = 256usize;

    println!("mode,shape,m,k,padded_k,n,repeats,ms,logical_tops,physical_tops");
    for mode_name in modes {
        let expected_mode = match mode_name {
            "w4" => NpuFullKMode::W4,
            "mixed" => NpuFullKMode::Mixed,
            "w8" => NpuFullKMode::W8,
            _ => return Err(format!("unknown mode {mode_name}").into()),
        };
        let mut total_ms = 0.0f64;
        let mut total_logical_macs = 0.0f64;
        for shape in SHAPES {
            let padded_k = shape.k.div_ceil(256) * 256;
            let cache = root.join(format!(
                "embgemma_aie2p_fullk_submit_{mode_name}_m256_kg{}_n{}",
                padded_k / 256,
                shape.n
            ));
            let mut gemm = NpuGemmFullK::load_cached(&cache.to_string_lossy(), 8)?;
            if gemm.mode() != expected_mode {
                return Err(format!("{} mode mismatch", cache.display()).into());
            }
            let groups = padded_k / 256;
            let base: Vec<Vec<i8>> = (0..groups)
                .map(|group| {
                    (0..256 * shape.n)
                        .map(|index| {
                            let value = pseudo(group * 1_000_003 + index);
                            match expected_mode {
                                NpuFullKMode::W8 => value,
                                NpuFullKMode::W4 | NpuFullKMode::Mixed => value.clamp(-7, 7),
                            }
                        })
                        .collect()
                })
                .collect();
            let residual: Vec<Vec<i8>> = if expected_mode == NpuFullKMode::Mixed {
                (0..groups)
                    .map(|group| {
                        (0..256 * shape.n)
                            .map(|index| {
                                if (index + group) % 53 == 0 {
                                    pseudo(index + group * 97) / 4
                                } else {
                                    0
                                }
                            })
                            .collect()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let base_refs: Vec<&[i8]> = base.iter().map(Vec::as_slice).collect();
            let residual_refs: Vec<&[i8]> = residual.iter().map(Vec::as_slice).collect();
            let packed = gemm.prepack_weights(&base_refs, &residual_refs)?;
            let resident = gemm.upload_resident_weights(&packed)?;
            let activations: Vec<i8> = (0..rows * padded_k).map(pseudo).collect();
            let mut partials = vec![0i32; groups * rows * shape.n];
            for _ in 0..2 {
                gemm.run_resident(&resident, &activations, &mut partials)?;
            }
            let started = Instant::now();
            for _ in 0..iterations {
                gemm.run_resident(&resident, &activations, &mut partials)?;
            }
            let seconds = started.elapsed().as_secs_f64() / iterations as f64;
            let logical_macs = rows as f64 * shape.k as f64 * shape.n as f64;
            let physical_macs = rows as f64 * padded_k as f64 * shape.n as f64;
            let ms = seconds * 1e3;
            total_ms += ms * shape.repeats as f64;
            total_logical_macs += logical_macs * shape.repeats as f64;
            println!(
                "{mode_name},{},{rows},{},{padded_k},{},{},{ms:.4},{:.4},{:.4}",
                shape.name,
                shape.k,
                shape.n,
                shape.repeats,
                2.0 * logical_macs / seconds / 1e12,
                2.0 * physical_macs / seconds / 1e12,
            );
        }
        let aggregate = 2.0 * total_logical_macs / (total_ms / 1e3) / 1e12;
        eprintln!(
            "hybrid-projection summary mode={mode_name} M=256 weighted_ms={total_ms:.3} aggregate_logical_tops={aggregate:.3}"
        );
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn pseudo(index: usize) -> i8 {
    let value = (index as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(0x9e37_79b9);
    (((value >> 9) & 0x7f) as i16 - 63) as i8
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("amdxdna is Linux-only");
}
