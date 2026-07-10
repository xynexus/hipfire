//! EmbeddingGemma-300M Opus GEMM sweep on AIE2P through `NpuGemmMp`.
//!
//! This is a benchmark/quality-gate building block, not a serving path. It runs the
//! model's projection shapes as cache-width Opus groups, padding tail groups.
//! Packed groups are uploaded once, then all group/M-tile commands are queued
//! with resident weights and one final timeline wait. The per-group int32
//! outputs remain separate; this is a projection-inventory dispatch benchmark,
//! not the scaled full-K reconstruction or a full-model result.
//!
//! Usage:
//!   cargo run --release -p hipfire-xdna --example npu_embeddinggemma_opus_sweep -- \
//!     --cache w4:256:$HOME/.hipfire/npu/embgemma_aie2p_w4_4x4x16_c8_nb4 \
//!     --cache w4:768:$HOME/.hipfire/npu/embgemma_aie2p_w4_4x4x16_c8_nb12 \
//!     --cache w4:1152:$HOME/.hipfire/npu/embgemma_aie2p_w4_4x4x16_c8_nb18 \
//!     --cache w4:3072:$HOME/.hipfire/npu/embgemma_aie2p_w4_4x4x16_c8_nb48 \
//!     --cache w8:256:$HOME/.hipfire/npu/embgemma_aie2p_w8_2x4x32_c8_nb4_m8k8_w8 \
//!     --cache w8:768:$HOME/.hipfire/npu/embgemma_aie2p_w8_2x4x32_c8_nb12_m8k8_w8 \
//!     --cache w8:1152:$HOME/.hipfire/npu/embgemma_aie2p_w8_2x4x32_c8_nb18_m8k8_w8 \
//!     --cache w8:3072:$HOME/.hipfire/npu/embgemma_aie2p_w8_2x4x32_c8_nb48_m8k8_w8

#[cfg(target_os = "linux")]
use std::{collections::HashMap, time::Instant};

#[cfg(target_os = "linux")]
#[derive(Clone)]
struct Shape {
    name: &'static str,
    k: usize,
    n: usize,
    repeats: usize,
    sensitive: bool,
}

#[cfg(target_os = "linux")]
#[derive(Clone)]
struct CacheSpec {
    dir: String,
}

#[cfg(target_os = "linux")]
fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "q_proj",
            k: 768,
            n: 768,
            repeats: 24,
            sensitive: true,
        },
        Shape {
            name: "k_proj",
            k: 768,
            n: 256,
            repeats: 24,
            sensitive: true,
        },
        Shape {
            name: "v_proj",
            k: 768,
            n: 256,
            repeats: 24,
            sensitive: true,
        },
        Shape {
            name: "o_proj",
            k: 768,
            n: 768,
            repeats: 24,
            sensitive: true,
        },
        Shape {
            name: "gate_proj",
            k: 768,
            n: 1152,
            repeats: 24,
            sensitive: false,
        },
        Shape {
            name: "up_proj",
            k: 768,
            n: 1152,
            repeats: 24,
            sensitive: false,
        },
        Shape {
            name: "down_proj",
            k: 1152,
            n: 768,
            repeats: 24,
            sensitive: false,
        },
        Shape {
            name: "dense.0",
            k: 768,
            n: 3072,
            repeats: 1,
            sensitive: true,
        },
        Shape {
            name: "dense.1",
            k: 3072,
            n: 768,
            repeats: 1,
            sensitive: true,
        },
    ]
}

#[cfg(target_os = "linux")]
fn parse_list(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|v| v.trim().parse::<usize>().ok())
        .collect()
}

#[cfg(target_os = "linux")]
fn rnd_a(i: usize) -> i8 {
    let s = (i as u32)
        .wrapping_mul(2654435761)
        .wrapping_add(0x9e37_79b9);
    (((s >> 13) & 0x7f) as i32 - 63) as i8
}

#[cfg(target_os = "linux")]
fn rnd_w(i: usize, weight_bits: usize) -> i8 {
    let s = (i as u32)
        .wrapping_mul(2654435761)
        .wrapping_add(0x9e37_79b9);
    if weight_bits == 8 {
        (((s >> 9) & 0xff) as i32 - 128) as i8
    } else {
        (((s >> 13) & 0xf) as i32 - 8) as i8
    }
}

#[cfg(target_os = "linux")]
fn cache_tag_for(format: &str, shape: &Shape) -> &'static str {
    match format {
        "oq8++" | "op8++" | "w8" => "w8",
        "oq4.25-policy" => {
            if shape.sensitive {
                "w8"
            } else {
                "w4"
            }
        }
        _ => "w4",
    }
}

#[cfg(target_os = "linux")]
fn usage() -> ! {
    eprintln!(
        "usage: npu_embeddinggemma_opus_sweep --cache TAG:N:DIR [--format oq4++,oq8++,oq4.25-policy] [--batches 32,128,512] [--iters N] [--warmup N]"
    );
    std::process::exit(2);
}

#[cfg(target_os = "linux")]
fn select_cache<'a>(
    caches: &'a HashMap<(String, usize), Vec<CacheSpec>>,
    tag: &str,
    n: usize,
    batch: usize,
) -> Option<(&'a CacheSpec, hipfire_xdna::NpuGemmMp, usize)> {
    let specs = caches.get(&(tag.to_string(), n))?;
    specs
        .iter()
        .filter_map(|spec| {
            let gemm = hipfire_xdna::NpuGemmMp::load_cached(&spec.dir).ok()?;
            let rows_per = gemm.rows_per_dispatch();
            let padded = batch.div_ceil(rows_per) * rows_per;
            Some((spec, gemm, padded))
        })
        .min_by_key(|(_, gemm, padded)| (*padded, std::cmp::Reverse(gemm.rows_per_dispatch())))
}

#[cfg(target_os = "linux")]
fn main() {
    let mut caches: HashMap<(String, usize), Vec<CacheSpec>> = HashMap::new();
    let mut formats = vec![
        "oq4++".to_string(),
        "oq8++".to_string(),
        "oq4.25-policy".to_string(),
    ];
    let mut batches = vec![32usize, 128, 512];
    let mut warmup = 2usize;
    let mut iters = 10usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--cache" => {
                let spec = args.next().unwrap_or_else(|| usage());
                let mut parts = spec.splitn(3, ':');
                let tag = parts.next().unwrap_or("").to_string();
                let n = parts
                    .next()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or_else(|| usage());
                let dir = parts.next().unwrap_or_else(|| usage()).to_string();
                caches.entry((tag, n)).or_default().push(CacheSpec { dir });
            }
            "--format" | "--formats" => {
                formats = args
                    .next()
                    .map(|s| s.split(',').map(|v| v.trim().to_string()).collect())
                    .unwrap_or_else(|| usage());
            }
            "--batches" => {
                batches = args
                    .next()
                    .map(|s| parse_list(&s))
                    .unwrap_or_else(|| usage());
            }
            "--warmup" => warmup = args.next().and_then(|s| s.parse().ok()).unwrap_or(warmup),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown arg: {other}");
                usage();
            }
        }
    }
    if caches.is_empty() {
        usage();
    }

    println!("format,cache_tag,shape,batch,padded_batch,k,n,group_k,groups,repeats,weight_bits,ms_per_shape,ms_weighted,tops");
    for format in &formats {
        for &batch in &batches {
            let mut total_ms = 0.0f64;
            let mut total_macs = 0.0f64;
            for shape in shapes() {
                let tag = cache_tag_for(format, &shape);
                let Some((cache, mut gemm, padded_batch)) =
                    select_cache(&caches, tag, shape.n, batch)
                else {
                    eprintln!(
                        "skip format={format} shape={} N={} missing cache tag={tag}",
                        shape.name, shape.n
                    );
                    continue;
                };
                assert_eq!(gemm.n(), shape.n, "cache {} N mismatch", cache.dir);
                let group_k = gemm.k();
                let groups = shape.k.div_ceil(group_k);
                let weight_bits = gemm.weight_bits();
                assert_eq!(
                    if tag == "w8" { 8 } else { 4 },
                    weight_bits,
                    "cache tag does not match xclbin weight bits"
                );

                let mut resident_groups = Vec::with_capacity(groups);
                for group in 0..groups {
                    let effective_k = (shape.k - group * group_k).min(group_k);
                    let weights: Vec<i8> = (0..group_k * shape.n)
                        .map(|i| {
                            if i / shape.n < effective_k {
                                rnd_w(group * 1_000_003 + i, weight_bits)
                            } else {
                                0
                            }
                        })
                        .collect();
                    let packed = gemm.prepack_weights(group_k, shape.n, &weights);
                    resident_groups.push(
                        gemm.upload_resident_weights(&packed)
                            .expect("upload resident weights"),
                    );
                }
                let acts: Vec<Vec<i8>> = (0..groups)
                    .map(|group| {
                        let effective_k = (shape.k - group * group_k).min(group_k);
                        (0..padded_batch * group_k)
                            .map(|i| {
                                if i % group_k < effective_k {
                                    rnd_a(group * 2_000_003 + i)
                                } else {
                                    0
                                }
                            })
                            .collect()
                    })
                    .collect();
                let activation_groups: Vec<&[i8]> = acts.iter().map(Vec::as_slice).collect();
                let mut outputs = vec![vec![0i32; padded_batch * shape.n]; groups];

                for _ in 0..warmup {
                    let resident_refs = resident_groups.iter().collect::<Vec<_>>();
                    let mut output_refs = outputs
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>();
                    gemm.run_resident_batch(
                        &resident_refs,
                        padded_batch,
                        group_k,
                        shape.n,
                        &activation_groups,
                        &mut output_refs,
                    )
                    .expect("warmup");
                }
                let t0 = Instant::now();
                for _ in 0..iters {
                    let resident_refs = resident_groups.iter().collect::<Vec<_>>();
                    let mut output_refs = outputs
                        .iter_mut()
                        .map(Vec::as_mut_slice)
                        .collect::<Vec<_>>();
                    gemm.run_resident_batch(
                        &resident_refs,
                        padded_batch,
                        group_k,
                        shape.n,
                        &activation_groups,
                        &mut output_refs,
                    )
                    .expect("bench");
                }
                let dt = t0.elapsed().as_secs_f64() / iters as f64;
                let macs = padded_batch as f64 * shape.n as f64 * shape.k as f64;
                let tops = 2.0 * macs / dt / 1e12;
                let ms = dt * 1e3;
                let weighted_ms = ms * shape.repeats as f64;
                total_ms += weighted_ms;
                total_macs += macs * shape.repeats as f64;
                println!(
                    "{format},{tag},{},{batch},{padded_batch},{},{},{group_k},{groups},{},{weight_bits},{ms:.4},{weighted_ms:.4},{tops:.4}",
                    shape.name, shape.k, shape.n, shape.repeats
                );
            }
            let total_tops = if total_ms > 0.0 {
                2.0 * total_macs / (total_ms / 1e3) / 1e12
            } else {
                0.0
            };
            eprintln!(
                "summary format={format} batch={batch}: matmul_ms_per_encode={total_ms:.3} aggregate_tops={total_tops:.3}"
            );
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("amdxdna is Linux-only");
}
