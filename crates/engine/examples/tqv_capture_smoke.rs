//! Smoke test for the TQV calibration capture primitive.
//!
//! Verifies that tqv_capture_values emits normalized FWHT/sign-rotated values
//! with per-head norm ~1 for head_dim 128 and 256.

use engine::llama::KvCache;
use std::error::Error;

fn make_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 32) as u32) as f32 / u32::MAX as f32;
            let wave = ((i as f32) * 0.013).sin();
            0.8 * (u * 2.0 - 1.0) + 0.2 * wave
        })
        .collect()
}

fn run_case(gpu: &mut rdna_compute::Gpu, head_dim: usize) -> Result<(), Box<dyn Error>> {
    let n_kv_heads = 3usize;
    let kv = KvCache::new_gpu_asym4_tqv4_capped(gpu, 1, n_kv_heads, head_dim, 8, 8)?;
    let host = make_vec(n_kv_heads * head_dim, 0x5451_5600 + head_dim as u64);
    let src = gpu.upload_f32(&host, &[n_kv_heads * head_dim])?;
    let dst = gpu.zeros(&[n_kv_heads * head_dim], rdna_compute::DType::F32)?;
    gpu.tqv_capture_values(
        &dst,
        &src,
        kv.fwht_signs1.as_ref().unwrap(),
        kv.fwht_signs2.as_ref().unwrap(),
        n_kv_heads,
        head_dim,
    )?;
    gpu.hip.device_synchronize()?;
    let got = gpu.download_f32(&dst)?;
    for h in 0..n_kv_heads {
        let start = h * head_dim;
        let norm = got[start..start + head_dim]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        if (norm - 1.0).abs() > 0.0005 {
            return Err(format!("head_dim={head_dim} head={h} norm={norm}").into());
        }
    }
    let rms = (got.iter().map(|v| v * v).sum::<f32>() / got.len() as f32).sqrt();
    println!("head_dim={head_dim},samples={},rms={rms:.8}", got.len());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut gpu = rdna_compute::Gpu::init()?;
    run_case(&mut gpu, 128)?;
    run_case(&mut gpu, 256)?;
    Ok(())
}
