//! Run one real `.hfq` linear through the NPU with FULL oq4 dequantization, check it
//! against the CPU reference, and report throughput.
//!
//! This is the piece the earlier `npu_artifact_gemv` probe did NOT cover. That one
//! verified the *integer* GEMM against a reference that also omitted scales. Oq4G256
//! carries a per-256-element group scale, so correct output needs
//! per-group `int32 -> * act_scale[group][row] * weight_scale[col] -> f32 accumulate`.
//! `NpuOpusExecutor::run_f32` already implements exactly that (per-group
//! `run_resident_batch` + `accumulate_scaled`), so the wire-in path is this call, not
//! `NpuGemm::run_resident`.
//!
//! Usage:
//!   npu_linear_oq4 MODEL.hfq TENSOR [--m 1] [--iters 20]
//!
//! The K=256 cache trio is resolved from N (= NB*NT*MN = NB*64), e.g. N=2048 -> nb32.
//! `NpuGemmMp` only accepts the M-parallel `r6mp_` builds (its cache-name parser
//! rejects a bare `r6_` prefix), so these are R6_GEN=r6_gen_mp.py outputs.
//!
//! Reports `logical_tops` (2*M*K*N over wall time) and the weight bytes/second the
//! call sustains — decode is weight-bandwidth-bound, so the second number is the one
//! that predicts tok/s.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    use std::time::Instant;

    use hipfire_runtime::hfq::HfqFile;
    use hipfire_xdna::{NpuOpusExecutor, OpusMatrixEncoding};

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 2 {
        return Err("usage: npu_linear_oq4 MODEL.hfq TENSOR [--m M] [--iters N]".into());
    }
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
    };
    let m: usize = opt("--m").map(|v| v.parse()).transpose()?.unwrap_or(1);
    let iters: usize = opt("--iters").map(|v| v.parse()).transpose()?.unwrap_or(20);

    let hfq = HfqFile::open(Path::new(&args[0]))?;
    let name = &args[1];
    let (info, payload) = hfq
        .tensor_data_vec(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if info.shape.len() != 2 {
        return Err(format!("{name} must be rank two, got {:?}", info.shape).into());
    }
    let n = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    let quant_type = info.quant_type;
    let encoding = OpusMatrixEncoding::classify(quant_type, payload.len(), k, n)?;

    // AWQ sidecar, if the artifact carries one for this tensor.
    let awq = hfq
        .tensor_data_vec(&format!("{name}.awq_scale.weight"))
        .or_else(|| {
            let base = name.strip_suffix(".weight").unwrap_or(name);
            hfq.tensor_data_vec(&format!("{base}.awq_scale.weight"))
        })
        .map(|(_, bytes)| {
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        })
        .filter(|v| v.len() == k);

    // N = NB * NT * MN = NB * 64. The trio must all be K=256 with this N.
    if n % 64 != 0 {
        return Err(format!("N={n} is not a multiple of 64; no cache shape covers it").into());
    }
    let nb = n / 64;
    let home = std::env::var("HOME")?;
    // Matches the validated EmbeddingGemma cache recipe exactly
    // (benchmarks/npu_gemm_tuning/embeddinggemma_aie2p/build_opus_caches.sh): MT=4
    // across the trio, R6_GEN=r6_gen_mp.py, and — the part that is easy to miss — the
    // W4 build needs R6_KERNEL_SRC=r6_gemm_ts.cc. The default r6_gemm.cc is the
    // pre-tiled load_v/store_v variant; it builds and runs but reads a different
    // A/W layout, so `run_f32` returns plausible-looking noise instead of the
    // reference. Same trap for MT: 16 builds fine and is also wrong.
    let w4 = format!("{home}/.hipfire/npu/r6ts_4x4x16_c8_nb{nb}");
    // The m8k8 W8 kernel has MK=8, so K=256 needs KCHUNK=32. MT=16 at that KCHUNK
    // overruns L1 ("allocated buffers exceeded available memory").
    let w8 = format!("{home}/.hipfire/npu/r6mp_4x4x32_c8_nb{nb}_m8k8_w8");
    let sp3 = format!("{home}/.hipfire/npu/r6mp_4x4x16_c8_nb{nb}_sparse3");
    for dir in [&w4, &w8, &sp3] {
        if !Path::new(&format!("{dir}/final.xclbin")).exists() {
            return Err(format!(
                "missing cache {dir}; build with benchmarks/npu_gemm_tuning/r6/r6_cache.sh"
            )
            .into());
        }
    }
    let mut executor = NpuOpusExecutor::load_cached(&w4, &w8, &sp3, n)?;
    let matrix = executor.pack_matrix(quant_type, k, n, &payload, awq)?;

    let input = (0..m * k)
        .map(|i| ((i as f32 * 0.013).sin() * 2.0) + ((i % 7) as f32 - 3.0) * 0.1)
        .collect::<Vec<_>>();
    let reference = matrix.reference_f32(m, &input)?;
    let mut output = vec![0.0f32; m * n];
    executor.run_f32(&matrix, m, &input, &mut output)?;

    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut first = None;
    for (i, (&got, &want)) in output.iter().zip(&reference).enumerate() {
        let err = (got - want).abs();
        max_abs = max_abs.max(err);
        // Same tolerance shape the EmbeddingGemma HFP verifier uses.
        if err > 1e-4 + want.abs() * 1e-5 {
            mismatches += 1;
            first.get_or_insert((i, got, want, err));
        }
    }
    println!(
        "{name}: qt={quant_type} encoding={encoding:?} M={m} K={k} N={n} awq={} \
         mismatches={mismatches}/{} max_abs={max_abs:.7}",
        matrix.awq_scale().is_some(),
        output.len()
    );
    if let Some((i, got, want, err)) = first {
        println!("first_mismatch index={i} got={got:.7} want={want:.7} abs={err:.7}");
    }
    if mismatches != 0 {
        return Err("oq4 NPU dequant parity failed".into());
    }

    if iters > 0 {
        for _ in 0..2 {
            executor.run_f32(&matrix, m, &input, &mut output)?;
        }
        let started = Instant::now();
        for _ in 0..iters {
            executor.run_f32(&matrix, m, &input, &mut output)?;
        }
        let secs = started.elapsed().as_secs_f64() / iters as f64;
        // Weight bytes actually streamed per call: 4 bits/weight for W4.
        let weight_bytes = (k * n) as f64 / 2.0;
        println!(
            "  m={m} ms={:.4} logical_tops={:.4} weight_GBs={:.2}",
            secs * 1e3,
            2.0 * (m * k * n) as f64 / secs / 1e12,
            weight_bytes / secs / 1e9,
        );
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_linear_oq4 requires Linux + XDNA");
}
