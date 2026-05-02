//! Synthetic KV quantization parity for Qwen-style head dimensions.
//!
//! Compares the existing asym4 K + Q8 V path against asym4 K + TQV4/TQV2 V
//! on deterministic Q/K/V tensors. This isolates cache write + flash attention
//! layout/packing/dequant bugs before text generation enters the picture.
//!
//! Usage:
//!   cargo run --release -p engine --features deltanet --example kv_quant_parity
//!   cargo run --release -p engine --features deltanet --example kv_quant_parity -- --seq 128,512,2048 --head-dim 128,256

use engine::llama::KvCache;
use std::error::Error;
use std::time::Instant;

fn parse_list(arg: Option<&String>, default: &[usize]) -> Vec<usize> {
    arg.map(|s| {
        s.split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect::<Vec<_>>()
    })
    .filter(|v| !v.is_empty())
    .unwrap_or_else(|| default.to_vec())
}

fn metrics(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut dot = 0.0f32;
    let mut aa = 0.0f32;
    let mut bb = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    let mean_abs = sum_abs / a.len() as f32;
    let cosine = dot / ((aa.sqrt() * bb.sqrt()).max(1e-20));
    (max_abs, mean_abs, cosine)
}

fn make_vec(len: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed as u64;
    (0..len)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 32) as u32) as f32 / (u32::MAX as f32);
            let wave = ((i as f32) * 0.037 + seed as f32 * 0.001).sin();
            scale * (0.7 * (u * 2.0 - 1.0) + 0.3 * wave)
        })
        .collect()
}

fn run_case(
    gpu: &mut rdna_compute::Gpu,
    head_dim: usize,
    seq: usize,
    value_bits: usize,
) -> Result<(f32, f32, f32, f64), Box<dyn Error>> {
    let n_layers = 1usize;
    let n_heads = 1usize;
    let n_kv_heads = 1usize;
    let max_seq = seq.max(1);

    let base = KvCache::new_gpu_asym4(gpu, n_layers, n_kv_heads, head_dim, max_seq)?;
    let tqv = if value_bits == 2 {
        KvCache::new_gpu_asym4_tqv2_capped(gpu, n_layers, n_kv_heads, head_dim, max_seq, max_seq)?
    } else {
        KvCache::new_gpu_asym4_tqv4_capped(gpu, n_layers, n_kv_heads, head_dim, max_seq, max_seq)?
    };

    let pos_buf = gpu.hip.malloc(4)?;
    let mut write_ms = 0.0f64;
    for pos in 0..seq {
        let k = make_vec(head_dim, 0x1234_0000u32.wrapping_add(pos as u32), 1.0);
        let v = make_vec(head_dim, 0x5678_0000u32.wrapping_add(pos as u32), 0.8);
        let dk = gpu.upload_f32(&k, &[head_dim])?;
        let dv = gpu.upload_f32(&v, &[head_dim])?;
        gpu.hip.memcpy_htod(&pos_buf, &(pos as i32).to_ne_bytes())?;
        let t0 = Instant::now();
        gpu.kv_cache_write_asym4_fused(
            &base.k_gpu[0],
            &base.v_gpu[0],
            &dk,
            &dv,
            &pos_buf,
            base.givens_cos.as_ref().unwrap(),
            base.givens_sin.as_ref().unwrap(),
            n_kv_heads,
            head_dim,
        )?;
        gpu.kv_cache_write_asym4_tqv4_fused(
            &tqv.k_gpu[0],
            &tqv.v_gpu[0],
            &dk,
            &dv,
            &pos_buf,
            tqv.givens_cos.as_ref().unwrap(),
            tqv.givens_sin.as_ref().unwrap(),
            tqv.fwht_signs1.as_ref().unwrap(),
            tqv.fwht_signs2.as_ref().unwrap(),
            n_kv_heads,
            head_dim,
            value_bits,
        )?;
        gpu.hip.device_synchronize()?;
        write_ms += t0.elapsed().as_secs_f64() * 1000.0;
        gpu.free_tensor(dk).ok();
        gpu.free_tensor(dv).ok();
    }

    let q = make_vec(head_dim, 0x9abc_def0, 1.0);
    let dq = gpu.upload_f32(&q, &[head_dim])?;
    let out_base = gpu.zeros(&[head_dim], rdna_compute::DType::F32)?;
    let out_tqv = gpu.zeros(&[head_dim], rdna_compute::DType::F32)?;
    let partial_elems = n_heads * ((max_seq + 127) / 128) * (2 + head_dim);
    let partial_base = gpu.zeros(&[partial_elems], rdna_compute::DType::F32)?;
    let partial_tqv = gpu.zeros(&[partial_elems], rdna_compute::DType::F32)?;
    gpu.hip
        .memcpy_htod(&pos_buf, &((seq.saturating_sub(1)) as i32).to_ne_bytes())?;

    gpu.attention_flash_asym4(
        &dq,
        &base.k_gpu[0],
        &base.v_gpu[0],
        &out_base,
        &pos_buf,
        base.givens_cos.as_ref().unwrap(),
        base.givens_sin.as_ref().unwrap(),
        seq,
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq,
        &partial_base,
    )?;
    gpu.attention_flash_asym4_tqv4(
        &dq,
        &tqv.k_gpu[0],
        &tqv.v_gpu[0],
        &out_tqv,
        &pos_buf,
        tqv.givens_cos.as_ref().unwrap(),
        tqv.givens_sin.as_ref().unwrap(),
        tqv.fwht_signs1.as_ref().unwrap(),
        tqv.fwht_signs2.as_ref().unwrap(),
        seq,
        n_heads,
        n_kv_heads,
        head_dim,
        value_bits,
        max_seq,
        &partial_tqv,
    )?;
    gpu.hip.device_synchronize()?;

    let base_vals = gpu.download_f32(&out_base)?;
    let tqv_vals = gpu.download_f32(&out_tqv)?;
    Ok({
        let (max_abs, mean_abs, cosine) = metrics(&base_vals, &tqv_vals);
        (max_abs, mean_abs, cosine, write_ms / seq.max(1) as f64)
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut seqs = None;
    let mut head_dims = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seq" => {
                seqs = args.get(i + 1).cloned();
                i += 2;
            }
            "--head-dim" => {
                head_dims = args.get(i + 1).cloned();
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("Usage: kv_quant_parity [--seq 128,512,2048] [--head-dim 128,256]");
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}").into()),
        }
    }
    let seqs = parse_list(seqs.as_ref(), &[128, 512, 2048]);
    let head_dims = parse_list(head_dims.as_ref(), &[128, 256]);

    let mut gpu = rdna_compute::Gpu::init()?;
    eprintln!("GPU: {}", gpu.arch);
    println!("head_dim,seq,mode,max_abs,mean_abs,cosine,write_ms_per_token");
    for hd in head_dims {
        for &seq in &seqs {
            for bits in [4usize, 2usize] {
                let (max_abs, mean_abs, cosine, write_ms) = run_case(&mut gpu, hd, seq, bits)?;
                println!(
                    "{hd},{seq},asym4_tqv{bits},{max_abs:.8},{mean_abs:.8},{cosine:.8},{write_ms:.4}"
                );
                if !max_abs.is_finite() || !mean_abs.is_finite() || !cosine.is_finite() {
                    return Err(
                        format!("non-finite metric for head_dim={hd} seq={seq} tqv{bits}").into(),
                    );
                }
            }
        }
    }
    Ok(())
}
