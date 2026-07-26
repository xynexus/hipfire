#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! dflash_ref_dump: golden-reference dumper for the DFlash NPU bring-up
//! (`docs/npu/dflash-drafter-npu-plan.md`, Phase A).
//!
//! Loads a converted DFlash draft `.hfq` (ideally the F32 sidecar so the
//! reference carries no quant error), synthesizes a DETERMINISTIC block
//! input, runs `dflash::draft_forward` on the GPU, and writes the inputs
//! plus the final `[block_size, hidden]` block hidden to `.npy` files.
//!
//! The numpy reference (`tools/npu/dflash_ref.py`) reads the SAME safetensors
//! weights (bf16→f32, bit-exact to the F32 HFQ the runtime loads) and these
//! dumped inputs, reproduces every per-op intermediate, and validates its
//! final against `block_hidden.npy`. Those numpy intermediates are the
//! per-primitive golden slices Phase B checks each NPU kernel against.
//!
//! Determinism: the input RNG is a fixed-seed xorshift; two runs produce
//! byte-identical `.npy` files (Gate A).
//!
//! Usage:
//!   dflash_ref_dump <draft.hfq> [--block B] [--ctx L] [--out DIR] [--seed S]

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::dflash::{self, DflashConfig, DflashScratch, DflashWeights};
    use hipfire_runtime::hfq::HfqFile;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    // ── Minimal .npy writer (v1.0, C-order, little-endian f32/i32) ──────────
    fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", dims.join(", "))
        };
        let dict =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
        // Total header = 10 (magic+ver+len) + dict + padding, padded to 64.
        let mut header = dict.into_bytes();
        let unpadded = 10 + header.len() + 1; // +1 for trailing '\n'
        let pad = (64 - (unpadded % 64)) % 64;
        header.extend(std::iter::repeat(b' ').take(pad));
        header.push(b'\n');
        let mut out = Vec::with_capacity(10 + header.len());
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        out
    }
    fn write_npy_f32(dir: &Path, name: &str, data: &[f32], shape: &[usize]) {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "{name}: shape {shape:?} != len {}",
            data.len()
        );
        let mut f = std::fs::File::create(dir.join(format!("{name}.npy"))).unwrap();
        f.write_all(&npy_header("<f4", shape)).unwrap();
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&bytes).unwrap();
    }
    fn write_npy_i32(dir: &Path, name: &str, data: &[i32], shape: &[usize]) {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "{name}: shape {shape:?} != len {}",
            data.len()
        );
        let mut f = std::fs::File::create(dir.join(format!("{name}.npy"))).unwrap();
        f.write_all(&npy_header("<i4", shape)).unwrap();
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&bytes).unwrap();
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: dflash_ref_dump <draft.hfq> [--block B] [--ctx L] [--out DIR] [--seed S]"
        );
        std::process::exit(1);
    }
    let path = args[1].clone();
    let mut block_size: usize = 16;
    let mut ctx_len: usize = 32;
    let mut out_dir =
        PathBuf::from(std::env::var("DFLASH_REF_OUT").unwrap_or_else(|_| "dflash_ref".to_string()));
    let mut seed: u64 = 0xD1FEA55Eu64;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--block" => {
                block_size = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--ctx" => {
                ctx_len = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--out" => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    std::fs::create_dir_all(&out_dir).expect("mkdir out dir");

    eprintln!("=== dflash_ref_dump ===");
    eprintln!("draft: {path}");
    eprintln!("block_size: {block_size}  ctx_len: {ctx_len}  seed: {seed:#x}");
    eprintln!("out: {}", out_dir.display());

    let hfq = HfqFile::open(Path::new(&path)).expect("open draft .hfq");
    let cfg = DflashConfig::from_hfq(&hfq).expect("parse DflashConfig");
    eprintln!(
        "config: layers={} hidden={} heads={} kv_heads={} head_dim={} block={} eps={} theta={} target_layers={:?}",
        cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
        cfg.block_size, cfg.norm_eps, cfg.rope_theta, cfg.target_layer_ids,
    );

    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu init");
    let weights = DflashWeights::load(&mut gpu, &hfq, &cfg).expect("load dflash weights");
    let mut scratch =
        DflashScratch::new_with_mq(&mut gpu, &cfg, block_size, ctx_len, weights.has_mq)
            .expect("alloc scratch");

    // Deterministic seeded inputs (xorshift64*). Same values every run.
    let mut s = seed | 1;
    let mut rng = || -> f32 {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let x = s.wrapping_mul(0x2545F4914F6CDD1D);
        // top 24 bits → [0,1), then center + scale to a realistic hidden range.
        let u = ((x >> 40) as f32) / ((1u64 << 24) as f32);
        (u - 0.5) * 0.08
    };

    let noise_embedding: Vec<f32> = (0..block_size * cfg.hidden).map(|_| rng()).collect();
    let target_hidden: Vec<f32> = (0..ctx_len * cfg.num_extract() * cfg.hidden)
        .map(|_| rng())
        .collect();
    let positions_q: Vec<i32> = (ctx_len as i32..ctx_len as i32 + block_size as i32).collect();
    let positions_k: Vec<i32> = (0..(ctx_len + block_size) as i32).collect();

    dflash::draft_forward(
        &mut gpu,
        &weights,
        &cfg,
        Some(&noise_embedding),
        Some(&target_hidden),
        &positions_q,
        &positions_k,
        block_size,
        ctx_len,
        &mut scratch,
    )
    .expect("draft_forward");
    gpu.hip.device_synchronize().expect("sync");

    let block_hidden = gpu.download_f32(&scratch.x).expect("download x");

    // ── Dump inputs + final output ──────────────────────────────────────────
    write_npy_f32(
        &out_dir,
        "noise_embedding",
        &noise_embedding,
        &[block_size, cfg.hidden],
    );
    write_npy_f32(
        &out_dir,
        "target_hidden",
        &target_hidden,
        &[ctx_len, cfg.num_extract(), cfg.hidden],
    );
    write_npy_i32(&out_dir, "positions_q", &positions_q, &[block_size]);
    write_npy_i32(
        &out_dir,
        "positions_k",
        &positions_k,
        &[ctx_len + block_size],
    );
    write_npy_f32(
        &out_dir,
        "block_hidden",
        &block_hidden,
        &[block_size, cfg.hidden],
    );

    // Config sidecar for the numpy reference (shapes + rope/eps).
    let meta = serde_json::json!({
        "n_layers": cfg.n_layers,
        "hidden": cfg.hidden,
        "intermediate": cfg.intermediate,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "block_size": block_size,
        "ctx_len": ctx_len,
        "num_extract": cfg.num_extract(),
        "target_layer_ids": cfg.target_layer_ids,
        "seed": seed,
    });
    std::fs::write(
        out_dir.join("ref_meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )
    .unwrap();

    let (mn, mx) = block_hidden
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    eprintln!(
        "wrote {} .npy tensors + ref_meta.json (block_hidden min/max {mn:.5e}/{mx:.5e})",
        5
    );
    let finite = block_hidden.iter().filter(|v| v.is_finite()).count();
    assert_eq!(finite, block_hidden.len(), "non-finite in block_hidden");
    eprintln!("OK");
}
