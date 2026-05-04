//! convert_hfq4_to_hfq4v3 — convert an HFQ4-G256 (or MQ4-G256) `.hfq` model
//! file to the HFQ4v3 / MQ4v3 K=64 + FP16-(d, m) format in-place (writes
//! a new `.hfq` file).
//!
//! HFQ4v3 layout (per K=64 group):
//!   bytes  0..1  half d  (FP16 scale  = range / 15)
//!   bytes  2..3  half m  (FP16 zero   = group min)
//!   bytes  4..35 32 B nibbles (64 weights, lo nibble = even index)
//! Total 36 B per group → 4.5 b/w (vs HFQ4-G256's 4.25 b/w).
//!
//! Why a separate format: the gfx11 iu8 MMQ kernel (see
//! `gemm_hfq4v3_residual_iu8_mmq.gfx11.hip`) consumes K=64 metadata
//! granularity and FP16 (d, m). HFQ4-G256's per-K=256 FP32 metadata
//! over-amortizes for the iu8 wmma pipeline and inflates VGPR pressure.
//!
//! With `--rotate`, the converter dequantizes each row, applies the
//! engine's FWHT-64 sign-rotation per K=64 chunk, and re-quantizes.
//! That produces MQ4v3 (FWHT-64-rotated HFQ4v3 — analogous to MQ4 vs
//! HFQ4-G256 today). FWHT-64 = 6 levels of butterflies and stays in
//! registers, so the kernel-side counter-rotation on the activation
//! side is essentially free.
//!
//! Important: only weight tensors stored as HFQ4-G256 (quant_type=6)
//! or MQ4-G256 (quant_type=13) are converted. F16/F32 norms, Q8
//! embeddings, etc. are passed through unchanged. Source K must be a
//! multiple of 256 (which is true for every weight produced by the
//! existing quantizer — Qwen3.5/3.6 hidden_size is always 256-divisible).
//!
//! Usage:
//!   cargo run --release -p engine --example convert_hfq4_to_hfq4v3 -- \
//!     --input  ~/.hipfire/models/qwen3.5-9b.mq4 \
//!     --output ~/.hipfire/models/qwen3.5-9b.mq4v3 \
//!     [--rotate]
//!
//! With `--rotate`, the output magic is `MQ4V3` (quant_type=20). Without,
//! it is `HFQ4V3` (quant_type=19). Either way the kernel + dispatch
//! routing is identical — `--rotate` simply embeds the FWHT-64 rotation
//! into the weights, and the engine kernel-dispatch must apply the
//! matching FWHT-64 to activations (handled separately).

use memmap2::Mmap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

const HFQ_MAGIC: &[u8] = b"HFQM";
const HFQ_VERSION: u32 = 1;

const QT_HFQ4G256: u8 = 6;
const QT_MQ4G256: u8 = 13;
const QT_HFQ4V3G64: u8 = 19;
const QT_MQ4V3G64: u8 = 20;

// ─── HFQ source reader ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data_offset: usize,
    data_size: usize,
}

struct HfqReader {
    _file: File,
    mmap: Mmap,
    arch_id: u32,
    metadata_json: String,
    tensors: Vec<TensorInfo>,
}

impl HfqReader {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if &mmap[0..4] != HFQ_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not an HFQ file (magic mismatch)",
            ));
        }
        let _version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        let arch_id = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(mmap[24..32].try_into().unwrap()) as usize;

        // Find end of JSON metadata by brace-matching.
        let meta_bytes = &mmap[metadata_offset..data_offset];
        let mut depth = 0i32;
        let mut in_str = false;
        let mut esc = false;
        let mut json_end = 0usize;
        for (i, &b) in meta_bytes.iter().enumerate() {
            if esc { esc = false; continue; }
            if b == b'\\' && in_str { esc = true; continue; }
            if b == b'"' { in_str = !in_str; continue; }
            if !in_str {
                if b == b'{' { depth += 1; }
                if b == b'}' {
                    depth -= 1;
                    if depth == 0 { json_end = i + 1; break; }
                }
            }
        }
        let metadata_json = String::from_utf8_lossy(&meta_bytes[..json_end]).into_owned();

        // Tensor index follows the JSON.
        let mut pos = metadata_offset + json_end;
        let idx_n = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()) as usize;
        assert_eq!(idx_n, n_tensors, "index count mismatch");
        pos += 4;

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut cum = data_offset;
        for _ in 0..n_tensors {
            let nlen = u16::from_le_bytes(mmap[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let name = String::from_utf8_lossy(&mmap[pos..pos + nlen]).into_owned();
            pos += nlen;
            let quant_type = mmap[pos]; pos += 1;
            let n_dims = mmap[pos] as usize; pos += 1;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()));
                pos += 4;
            }
            let group_size = u32::from_le_bytes(mmap[pos..pos + 4].try_into().unwrap()); pos += 4;
            let data_size = u64::from_le_bytes(mmap[pos..pos + 8].try_into().unwrap()) as usize; pos += 8;
            tensors.push(TensorInfo { name, quant_type, shape, group_size, data_offset: cum, data_size });
            cum += data_size;
        }

        Ok(Self { _file: file, mmap, arch_id, metadata_json, tensors })
    }

    fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
        &self.mmap[info.data_offset..info.data_offset + info.data_size]
    }
}

// ─── HFQ writer (mirrors hipfire-quantize::write_hfq) ──────────────────────

struct OutTensor {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

fn write_hfq(path: &Path, arch: u32, metadata_json: &str, tensors: &[OutTensor]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let metadata_bytes = metadata_json.as_bytes();

    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;
    let index_offset = metadata_offset + metadata_size;

    let mut index_bytes = Vec::new();
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        index_bytes.push(t.quant_type);
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        index_bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }

    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    f.write_all(metadata_bytes)?;
    f.write_all(&index_bytes)?;

    let pad = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad])?;

    for t in tensors {
        f.write_all(&t.data)?;
    }

    Ok(())
}

// ─── FP16 helpers (IEEE-754, no f16 intrinsic) ─────────────────────────────

fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let exp_unbiased = exp - 127;
    if exp_unbiased < -24 {
        return (sign << 15) as u16;
    }
    if exp_unbiased < -14 {
        // Subnormal
        let shift = -14 - exp_unbiased;
        let mantissa = (frac | 0x800000) >> (13 + shift);
        let round_bit = ((frac | 0x800000) >> (12 + shift)) & 1;
        let m16 = mantissa + round_bit;
        return ((sign << 15) | m16) as u16;
    }
    if exp_unbiased > 15 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    let exp16 = (exp_unbiased + 15) as u32;
    let frac16 = frac >> 13;
    let round_bit = (frac >> 12) & 1;
    let f16_low = (sign << 15) | (exp16 << 10) | frac16;
    (f16_low + round_bit) as u16
}

// ─── HFQ4-G256 dequant (reference) ─────────────────────────────────────────

/// Dequantize one row of HFQ4-G256 to f32. K must be a multiple of 256.
fn dequant_hfq4g256_row(row_bytes: &[u8], k: usize) -> Vec<f32> {
    assert!(k % 256 == 0);
    let groups = k / 256;
    assert_eq!(row_bytes.len(), groups * 136);
    let mut out = vec![0.0f32; k];
    for g in 0..groups {
        let off = g * 136;
        let scale = f32::from_le_bytes(row_bytes[off..off + 4].try_into().unwrap());
        let zp = f32::from_le_bytes(row_bytes[off + 4..off + 8].try_into().unwrap());
        for i in 0..128 {
            let byte = row_bytes[off + 8 + i];
            let lo = (byte & 0xF) as f32;
            let hi = (byte >> 4) as f32;
            out[g * 256 + 2 * i + 0] = lo * scale + zp;
            out[g * 256 + 2 * i + 1] = hi * scale + zp;
        }
    }
    out
}

// ─── FWHT-64: 6 levels of butterflies ──────────────────────────────────────
//
// Sign tables match the engine convention but are sized to 64 (vs 256 for
// the existing FWHT-256 path). Same RNG (linear_congruential, seeds 42 and
// 1042) as `gen_fwht_signs` in hipfire-quantize/src/main.rs and dispatch.rs,
// so the rotation is reproducible from the seeds alone.
//
// Block scale = 1/sqrt(64) = 0.125. The kernel-side activation rotation
// uses the same butterfly + scale + signs, ensuring (x · H)(H^T · w) = x · w.

fn gen_fwht_signs64(seed: u32) -> [f32; 64] {
    let mut state = seed;
    let mut out = [0.0f32; 64];
    for i in 0..64 {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        out[i] = if (state >> 16) & 1 == 1 { 1.0 } else { -1.0 };
    }
    out
}

fn cpu_fwht_64(x: &mut [f32; 64], signs1: &[f32; 64], signs2: &[f32; 64]) {
    for i in 0..64 { x[i] *= signs1[i]; }
    let mut stride = 1usize;
    while stride < 64 {
        let mut i = 0usize;
        while i < 64 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.125f32; // 1/sqrt(64)
    for i in 0..64 { x[i] = x[i] * scale * signs2[i]; }
}

// ─── HFQ4v3 quant: K=64 groups with FP16 (d, m) ────────────────────────────

fn quantize_hfq4v3_row(f32_data: &[f32], k: usize, rotate: bool, signs1: &[f32; 64], signs2: &[f32; 64]) -> Vec<u8> {
    assert_eq!(f32_data.len(), k);
    assert!(k % 64 == 0);
    let groups = k / 64;
    let mut out = vec![0u8; groups * 36];

    for g in 0..groups {
        let mut group = [0.0f32; 64];
        for i in 0..64 {
            group[i] = f32_data[g * 64 + i];
        }
        if rotate {
            cpu_fwht_64(&mut group, signs1, signs2);
        }

        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for &v in &group {
            if v < min_v { min_v = v; }
            if v > max_v { max_v = v; }
        }

        let range = max_v - min_v;
        let d_f32 = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv = if range > 0.0 { 1.0 / d_f32 } else { 0.0 };
        let m_f32 = min_v;

        // Round-trip the FP16 (d, m) before quantizing nibbles so the
        // dequant on GPU exactly inverts the same (d, m) values the
        // weights were quantized against. Using the f32 d/m here would
        // leave a small precision drift between converter and kernel
        // (~0.05% per element typical) for no benefit.
        let d_h = f32_to_f16(d_f32);
        let m_h = f32_to_f16(m_f32);
        // Decode back to f32 for nibble quant.
        let d_round = f16_to_f32_local(d_h);
        let m_round = f16_to_f32_local(m_h);
        let inv_round = if d_round > 0.0 { 1.0 / d_round } else { 0.0 };

        let off = g * 36;
        out[off + 0] = (d_h & 0xFF) as u8;
        out[off + 1] = (d_h >> 8) as u8;
        out[off + 2] = (m_h & 0xFF) as u8;
        out[off + 3] = (m_h >> 8) as u8;

        for i in 0..32 {
            let v_lo = group[2 * i + 0];
            let v_hi = group[2 * i + 1];
            let q_lo = ((v_lo - m_round) * inv_round + 0.5).clamp(0.0, 15.0) as u8;
            let q_hi = ((v_hi - m_round) * inv_round + 0.5).clamp(0.0, 15.0) as u8;
            out[off + 4 + i] = q_lo | (q_hi << 4);
            // inv unused here on purpose — using inv_round (FP16 d round-trip)
            // keeps the on-GPU dequant exact.
            let _ = inv;
        }
    }

    out
}

fn f16_to_f32_local(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { (frac << 13) | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    let exp32 = exp + 127 - 15;
    f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
}

// ─── Conversion driver ──────────────────────────────────────────────────────

fn parse_args() -> (PathBuf, PathBuf, bool) {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut rotate = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--input" | "-i" => { input = args.next().map(PathBuf::from); }
            "--output" | "-o" => { output = args.next().map(PathBuf::from); }
            "--rotate" | "-r" => { rotate = true; }
            "-h" | "--help" => {
                eprintln!("usage: convert_hfq4_to_hfq4v3 --input <hfq4-g256.hfq> --output <out.hfq> [--rotate]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    let input = input.expect("missing --input");
    let output = output.expect("missing --output");
    (input, output, rotate)
}

fn main() -> std::io::Result<()> {
    let (in_path, out_path, rotate) = parse_args();

    eprintln!("converting {} → {}", in_path.display(), out_path.display());
    eprintln!("  --rotate = {rotate}");

    let r = HfqReader::open(&in_path)?;
    eprintln!("  arch_id={} ntensors={}", r.arch_id, r.tensors.len());

    let signs1 = gen_fwht_signs64(42);
    let signs2 = gen_fwht_signs64(1042);

    let mut out_tensors: Vec<OutTensor> = Vec::with_capacity(r.tensors.len());
    let mut converted = 0usize;
    let mut passed_through = 0usize;
    let mut bytes_in = 0usize;
    let mut bytes_out = 0usize;

    for info in &r.tensors {
        let raw = r.tensor_bytes(info);
        bytes_in += raw.len();

        let convert_this = matches!(info.quant_type, QT_HFQ4G256 | QT_MQ4G256);
        if !convert_this {
            // Pass through unchanged.
            out_tensors.push(OutTensor {
                name: info.name.clone(),
                quant_type: info.quant_type,
                shape: info.shape.clone(),
                group_size: info.group_size,
                data: raw.to_vec(),
            });
            passed_through += 1;
            bytes_out += raw.len();
            continue;
        }

        // Determine M (output rows) and K (input cols) from shape.
        // HFQ tensors are stored as [out_features, in_features] = [M, K]
        // for weight matrices (matches PyTorch/safetensors convention).
        let (m, k) = match info.shape.as_slice() {
            &[m, k] => (m as usize, k as usize),
            other => {
                // Fall through for higher-rank tensors: treat last dim as K
                // and the product of leading dims as M.
                let k = *other.last().expect("zero-rank tensor") as usize;
                let m: usize = other[..other.len() - 1].iter().map(|&d| d as usize).product();
                (m, k)
            }
        };
        if k % 256 != 0 {
            eprintln!("  SKIP {}: K={k} not multiple of 256, cannot dequant", info.name);
            out_tensors.push(OutTensor {
                name: info.name.clone(),
                quant_type: info.quant_type,
                shape: info.shape.clone(),
                group_size: info.group_size,
                data: raw.to_vec(),
            });
            passed_through += 1;
            bytes_out += raw.len();
            continue;
        }

        let row_bytes_in = (k / 256) * 136;
        let row_bytes_out = (k / 64) * 36;
        assert_eq!(raw.len(), m * row_bytes_in);

        let mut new_data = vec![0u8; m * row_bytes_out];
        // Process row-by-row to keep memory bounded.
        for row in 0..m {
            let row_in = &raw[row * row_bytes_in..(row + 1) * row_bytes_in];
            let f32_row = dequant_hfq4g256_row(row_in, k);
            let row_out = quantize_hfq4v3_row(&f32_row, k, rotate, &signs1, &signs2);
            assert_eq!(row_out.len(), row_bytes_out);
            new_data[row * row_bytes_out..(row + 1) * row_bytes_out].copy_from_slice(&row_out);
        }

        let new_qt = if rotate { QT_MQ4V3G64 } else { QT_HFQ4V3G64 };
        let label = if rotate { "MQ4V3G64" } else { "HFQ4V3G64" };
        eprintln!(
            "  {} {} [{}×{}] {} B → {} B ({:.1}%)",
            label,
            info.name,
            m, k,
            raw.len(),
            new_data.len(),
            100.0 * new_data.len() as f32 / raw.len() as f32,
        );

        out_tensors.push(OutTensor {
            name: info.name.clone(),
            quant_type: new_qt,
            shape: info.shape.clone(),
            group_size: 64,
            data: new_data,
        });
        converted += 1;
        bytes_out += m * row_bytes_out;
    }

    eprintln!("\n  converted: {converted}, passed through: {passed_through}");
    eprintln!("  total: {} MB → {} MB ({:.1}%)",
        bytes_in / 1_000_000,
        bytes_out / 1_000_000,
        100.0 * bytes_out as f32 / bytes_in.max(1) as f32);

    write_hfq(&out_path, r.arch_id, &r.metadata_json, &out_tensors)?;
    eprintln!("\nwrote {}", out_path.display());
    Ok(())
}
