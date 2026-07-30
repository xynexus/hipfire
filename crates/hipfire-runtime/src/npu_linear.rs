//! Offload decode-shape Oq4G256 linears to the NPU.
//!
//! Opt-in with `HIPFIRE_NPU_DECODE=1`. Every eligible weight is packed and made
//! resident on the array on first use; subsequent tokens move only activations.
//! Anything ineligible — wrong dtype, no cache for that N, K not a multiple of
//! 256 — returns `None` and the caller runs its normal GPU path, so the flag is
//! safe to leave off and safe to turn on with a partially-covered model.
//!
//! ## Why the weights are reconstructed rather than packed at load
//!
//! `NpuOpusExecutor::pack_matrix` wants the CANONICAL on-disk oq4 payload
//! (`[f16 scale][128 nibbles]` per 256-group). By the time a linear reaches
//! `weight_gemv` the loader has already repacked it into the `oq4_arch` combined
//! device layout — `[split nibbles m*(k/2)][split f32 scales m*ng][interleaved
//! m*ng*132]`. The interleaved region is the same block stream in the same
//! (row, group) order, differing only in scale width, so canonical bytes are
//! recoverable from the uploaded buffer with an f32->f16 narrowing. That keeps
//! this module self-contained instead of threading payloads through every
//! loader.
//!
//! ## Cost shape (measured, llama-3.2-1B q_proj K=2048 N=2048, M=1)
//!
//! ~0.53 ms per linear on the best per-linear cache (full-K, COLS=1/M=8), of
//! which the array is ~74%. That is ~17 tok/s for a 112-linear model against
//! FastFlowLM's 60.1, so this path is a CORRECTNESS and measurement vehicle,
//! not a win. The structural fix is a fused decoder layer; see
//! `docs/npu/decoder-layer-npu-scope.md`.

use std::cell::RefCell;
use std::collections::HashMap;

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};
use hipfire_xdna::{NpuOpusExecutor, OpusPackedMatrix};

use crate::oq4_arch::OQ4_CANONICAL_QT;

fn bytemuck_cast(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any bit pattern is a valid u8 sequence.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

const GROUP: usize = 256;
/// `[f32 scale][128 nibbles]` — the interleaved record in the arch layout.
const ARCH_BLOCK: usize = 132;
/// `[f16 scale][128 nibbles]` — what `pack_matrix` consumes.
const CANONICAL_BLOCK: usize = 130;

/// HIPFIRE_NPU_DECODE_TIMING=1 prints a per-phase breakdown at process exit, so
/// the delivered cost can be split into GPU readback / NPU / GPU writeback
/// instead of inferred from a microbenchmark that kept activations on the host.
#[derive(Default)]
struct Timing {
    calls: u64,
    download_ns: u64,
    run_ns: u64,
    upload_ns: u64,
}

thread_local! {
    static TIMING: RefCell<Timing> = RefCell::new(Timing::default());
}

pub fn timing_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("HIPFIRE_NPU_DECODE_TIMING").is_ok_and(|v| v != "0"))
}

/// Print and reset the accumulated per-phase breakdown.
pub fn report_timing() {
    TIMING.with(|t| {
        let t = t.borrow();
        if t.calls == 0 {
            return;
        }
        let per = |ns: u64| ns as f64 / t.calls as f64 / 1e6;
        eprintln!(
            "[npu-decode] calls={} download_ms={:.4} npu_ms={:.4} upload_ms={:.4} total_ms={:.4}",
            t.calls,
            per(t.download_ns),
            per(t.run_ns),
            per(t.upload_ns),
            per(t.download_ns + t.run_ns + t.upload_ns),
        );
    });
}

pub fn npu_decode_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("HIPFIRE_NPU_DECODE").is_ok_and(|v| v != "0" && !v.is_empty())
    })
}

struct Resident {
    matrix: OpusPackedMatrix,
}

#[derive(Default)]
struct Registry {
    executors: HashMap<usize, NpuOpusExecutor>,
    weights: HashMap<usize, Resident>,
    rejected: HashMap<usize, ()>,
}

// Thread-local, not a global: the XRT handles inside `NpuOpusExecutor` are `Rc`
// and deliberately not `Send`. The inference worker is single-threaded, so one
// registry per thread is both correct and what the device wants.
thread_local! {
    static REGISTRY: RefCell<Registry> = RefCell::new(Registry::default());
}

/// Recover canonical oq4 bytes from the `oq4_arch` combined device layout.
fn canonical_from_arch(combined: &[u8], m: usize, k: usize) -> Option<Vec<u8>> {
    let ng = k / GROUP;
    let groups = m.checked_mul(ng)?;
    let packed_bytes = m * (k / 2);
    let scales_bytes = groups * 4;
    let interleaved = combined.get(packed_bytes + scales_bytes..)?;
    if interleaved.len() < groups * ARCH_BLOCK {
        return None;
    }
    let mut out = vec![0u8; groups * CANONICAL_BLOCK];
    for block in 0..groups {
        let src = block * ARCH_BLOCK;
        let dst = block * CANONICAL_BLOCK;
        let scale = f32::from_le_bytes([
            interleaved[src],
            interleaved[src + 1],
            interleaved[src + 2],
            interleaved[src + 3],
        ]);
        out[dst..dst + 2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        out[dst + 2..dst + CANONICAL_BLOCK]
            .copy_from_slice(&interleaved[src + 4..src + ARCH_BLOCK]);
    }
    Some(out)
}

/// Narrow to IEEE half. The quantizer wrote these from f16 originally, so this
/// round-trips exactly for every scale a real artifact carries.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    // Round-to-nearest-even on the 13 dropped mantissa bits.
    let mut half_mantissa = (mantissa >> 13) as u16;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_mantissa & 1) == 1) {
        half_mantissa += 1;
        if half_mantissa == 0x400 {
            half_mantissa = 0;
            exponent += 1;
            if exponent >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exponent as u16) << 10) | half_mantissa
}

fn cache_dir(n: usize, k: usize) -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let dir = format!(
        "{home}/.hipfire/npu/embgemma_aie2p_fullk_submit_w4_m8_kg{}_n{n}",
        k / GROUP
    );
    std::path::Path::new(&format!("{dir}/final.xclbin"))
        .exists()
        .then_some(dir)
}

fn have(dir: &str) -> bool {
    std::path::Path::new(&format!("{dir}/final.xclbin")).exists()
}

/// Build the executor for an `N`-column linear, preferring the per-group trio.
///
/// Both paths read back `groups * block_m * N` int32 partials and scale on the
/// host, so for decode (M=1) it is `block_m`, not the kernel, that dominates:
/// every row past the first is computed and transferred for nothing. The trio's
/// MT=1 build has `block_m = 4` against the full-K `m8` build's 8, and spreads
/// over 8 columns rather than the 1 the full-K path is loaded with. Measured on
/// Llama-3.2-1B oq4++ at M=1, identical output (0 mismatches):
/// gate_proj K=2048 N=8192 9.37 -> 2.07 ms, q_proj N=2048 1.13 -> 0.80 ms.
///
/// Falls back to full-K when the trio is not built for this N — the trio needs
/// all three caches present, and only the shapes we have built are covered.
fn load_executor(n: usize, k: usize) -> Option<NpuOpusExecutor> {
    let home = std::env::var("HOME").ok()?;
    // Opt-in, NOT the default: measured end to end on Llama-3.2-1B the trio is
    // 4.0 tok/s against full-K's 6.2. The trio's MT=1 build beats the trio's
    // MT=4 build 4.5x on a single linear (9.37 -> 2.07 ms on gate_proj), which
    // is real, but full-K does that same linear in 1.54 ms — so the per-linear
    // win does not survive contact with the path the runtime actually uses.
    // Kept behind a flag because MT=1 is still the right shape for decode and
    // will matter once the scale-application work removes the partial readback.
    let trio_ok = std::env::var("HIPFIRE_NPU_TRIO").is_ok_and(|v| v != "0");
    if trio_ok && n % 64 == 0 {
        let nb = n / 64;
        let w4 = format!("{home}/.hipfire/npu/r6ts_1x4x16_c8_nb{nb}");
        let w8 = format!("{home}/.hipfire/npu/r6mp_4x4x32_c8_nb{nb}_m8k8_w8");
        let sp3 = format!("{home}/.hipfire/npu/r6mp_4x4x16_c8_nb{nb}_sparse3");
        if have(&w4) && have(&w8) && have(&sp3) {
            if let Ok(exec) = NpuOpusExecutor::load_cached(&w4, &w8, &sp3, n) {
                if timing_enabled() {
                    eprintln!("[npu-decode] trio MT=1 N={n}");
                }
                return Some(exec);
            }
        }
    }
    // N-parallel scaled full-K: each column owns N/COLS slabs and all M rows
    // with its OWN weight stream, so per-tile ingress drops by COLS instead of
    // every tile absorbing the whole matrix through its 14.4 GB/s port. On real
    // Llama-3.2-1B oq4++ at M=1, all bit-exact: q_proj 0.546 -> 0.325 ms
    // (1.68x), gate_proj 1.555 -> 0.786 (1.98x), down_proj 2.043 -> 0.794
    // (2.57x). It also builds at N=8192, which the M-parallel schedule cannot
    // (its output BD carries a size=NB dimension that exceeds the DMA range).
    let np = format!(
        "{home}/.hipfire/npu/embgemma_aie2p_fullk_submit_w4-np-scaled_m8_kg{}_n{n}_c8",
        k / GROUP
    );
    // OPT-IN (HIPFIRE_NPU_NP=1). Every per-linear measurement favours this path,
    // but end to end it measured 5.5 tok/s against the full-K default's 6.2 —
    // the opposite direction. Per-linear wins therefore do not compose here and
    // the cause is not yet known; a candidate is that N-parallel duplicates the
    // activation into every (column, slab) entry, so the host repacks
    // cols*spc*groups*(rows*256 + scale_bytes) = ~600 KB per call against the
    // full-K path's ~16 KB, which a fixed-activation microbenchmark loop hides.
    // Not made default until that is measured, not guessed.
    if have(&np) && std::env::var("HIPFIRE_NPU_NP").is_ok_and(|v| v != "0") {
        if let Ok(exec) = NpuOpusExecutor::load_fullk_cached(&[(&np, 8)], n) {
            if timing_enabled() {
                eprintln!("[npu-decode] N-parallel scaled K={k} N={n}");
            }
            return Some(exec);
        }
    }
    let dir = cache_dir(n, k)?;
    NpuOpusExecutor::load_fullk_cached(&[(&dir, 1)], n).ok()
}

/// Run `y = W x` on the NPU. `None` = not eligible; caller falls back.
pub fn try_npu_gemv(
    gpu: &mut Gpu,
    w_buf: &GpuTensor,
    dtype: DType,
    m: usize,
    k: usize,
    awq_scale: Option<&GpuTensor>,
    x: &GpuTensor,
    y: &GpuTensor,
) -> Option<HipResult<()>> {
    if !npu_decode_enabled() || dtype != DType::Oq4G256 || k % GROUP != 0 || m % 64 != 0 {
        return None;
    }
    let key = w_buf.buf.as_ptr() as usize;
    // `weight_gemv` is the decode path: exactly one row. `x` is a scratch
    // tensor whose declared extent can exceed the live activation, so read a
    // K-element view rather than the whole thing.
    let rows = 1usize;
    // Not every caller's activation tensor is exactly `w.k` long — some are
    // sub-views or reuse a smaller scratch. Fall back rather than over-read.
    if x.buf.size() < k * std::mem::size_of::<f32>() {
        return None;
    }
    if y.buf.size() < m * std::mem::size_of::<f32>() {
        return None;
    }
    let x_view = GpuTensor {
        buf: unsafe { x.buf.alias() },
        shape: vec![k],
        dtype: DType::F32,
    };
    let t_dl = std::time::Instant::now();
    let input = match gpu.download_f32(&x_view) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let dl_ns = t_dl.elapsed().as_nanos() as u64;
    let weight_bytes = w_buf.buf.size();
    let combined_if_needed = REGISTRY.with(|reg| reg.borrow().weights.contains_key(&key));
    let combined = if combined_if_needed {
        None
    } else {
        match gpu.download_raw(w_buf, weight_bytes) {
            Ok(bytes) => Some(bytes),
            Err(e) => return Some(Err(e)),
        }
    };
    // AWQ sidecars are uploaded raw, so their declared extent is a BYTE count,
    // not an element count — `download_f32` on the tensor itself asks for 4x the
    // buffer. Read a correctly-sized element view instead.
    let awq = match awq_scale {
        Some(t) => {
            // These are uploaded raw (dtype Raw, size 1), so `sub_offset` would
            // scale by 1 byte per element and produce a view four times too
            // small. Alias the buffer with an explicit f32 view instead.
            let elems = t.buf.size() / std::mem::size_of::<f32>();
            let view = GpuTensor {
                buf: unsafe { t.buf.alias() },
                shape: vec![elems],
                dtype: DType::F32,
            };
            match gpu.download_f32(&view) {
                Ok(v) if v.len() >= k => Some(v[..k].to_vec()),
                _ => None,
            }
        }
        None => None,
    };

    let outcome = REGISTRY.with(|reg| {
        let reg = &mut *reg.borrow_mut();
        if reg.rejected.contains_key(&key) {
            return None;
        }
        if !reg.weights.contains_key(&key) {
            let Some(payload) = canonical_from_arch(combined.as_deref()?, m, k) else {
                reg.rejected.insert(key, ());
                return None;
            };
            if !reg.executors.contains_key(&m) {
                match load_executor(m, k) {
                    Some(exec) => {
                        reg.executors.insert(m, exec);
                    }
                    None => {
                        reg.rejected.insert(key, ());
                        return None;
                    }
                }
            }
            let exec = reg.executors.get(&m)?;
            match exec.pack_matrix(OQ4_CANONICAL_QT, k, m, &payload, awq) {
                Ok(matrix) => {
                    if timing_enabled() {
                        eprintln!(
                            "[npu-decode] resident K={k} N={m} (weights={})",
                            reg.weights.len() + 1
                        );
                    }
                    reg.weights.insert(key, Resident { matrix });
                }
                Err(_) => {
                    reg.rejected.insert(key, ());
                    return None;
                }
            }
        }
        let Registry {
            executors, weights, ..
        } = reg;
        let resident = weights.get(&key)?;
        let exec = executors.get_mut(&m)?;
        let mut output = vec![0.0f32; rows * m];
        let t_run = std::time::Instant::now();
        let run_result = exec.run_f32(&resident.matrix, rows, &input[..rows * k], &mut output);
        let run_ns = t_run.elapsed().as_nanos() as u64;
        TIMING.with(|t| {
            let t = &mut *t.borrow_mut();
            t.calls += 1;
            t.download_ns += dl_ns;
            t.run_ns += run_ns;
        });
        match run_result {
            Ok(()) => Some(Ok(output)),
            Err(e) => Some(Err(hipfire_rdna::HipError::new(0, &e.to_string()))),
        }
    })?;
    let output = match outcome {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let t_up = std::time::Instant::now();
    let wrote = gpu.memcpy_htod_auto(&y.buf, bytemuck_cast(&output));
    TIMING.with(|t| t.borrow_mut().upload_ns += t_up.elapsed().as_nanos() as u64);
    Some(wrote)
}
