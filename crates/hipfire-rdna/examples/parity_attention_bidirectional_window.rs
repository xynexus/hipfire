// SPDX-License-Identifier: Apache-2.0

//! Hardware parity for the symmetric bidirectional attention window used by
//! long EmbeddingGemma encoder inputs.

use hipfire_rdna::{DType, Gpu};

fn values(seed: u32, count: usize) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.2
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sequence: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; q.len()];
    let repeat = heads / kv_heads;
    let q_stride = heads * head_dim;
    let kv_stride = kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for query in 0..sequence {
        let first = query.saturating_sub(window - 1);
        let last = sequence.min(query + window);
        for head in 0..heads {
            let kv_head = head / repeat;
            let q_row = &q[query * q_stride + head * head_dim..][..head_dim];
            let mut scores = (first..last)
                .map(|key| {
                    let k_row = &k[key * kv_stride + kv_head * head_dim..][..head_dim];
                    q_row.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            scores
                .iter_mut()
                .for_each(|score| *score = (*score - maximum).exp());
            let denominator = scores.iter().sum::<f32>();
            let out = &mut output[query * q_stride + head * head_dim..][..head_dim];
            for (score, key) in scores.into_iter().zip(first..last) {
                let v_row = &v[key * kv_stride + kv_head * head_dim..][..head_dim];
                for (target, value) in out.iter_mut().zip(v_row) {
                    *target += score * value / denominator;
                }
            }
        }
    }
    output
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sequence = 17;
    let heads = 3;
    let kv_heads = 1;
    let head_dim = 64;
    let window = 4;
    let q = values(1, sequence * heads * head_dim);
    let k = values(2, sequence * kv_heads * head_dim);
    let v = values(3, sequence * kv_heads * head_dim);
    let expected = reference(&q, &k, &v, sequence, heads, kv_heads, head_dim, window);

    let mut gpu = Gpu::init()?;
    let q_gpu = gpu.upload_owned_f32(&q, &[q.len()])?;
    let k_gpu = gpu.upload_owned_f32(&k, &[k.len()])?;
    let v_gpu = gpu.upload_owned_f32(&v, &[v.len()])?;
    let output_gpu = gpu.zeros_owned(&[expected.len()], DType::F32)?;
    gpu.attention_dflash_bidirectional_window_f32(
        &q_gpu,
        &k_gpu,
        &v_gpu,
        &output_gpu,
        sequence,
        heads,
        kv_heads,
        head_dim,
        window,
    )?;
    let actual = gpu.download_f32(&output_gpu)?;
    let max_abs = expected
        .iter()
        .zip(&actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    println!("sequence={sequence} window={window} max_abs={max_abs:.9}");
    if max_abs >= 1.0e-5 {
        return Err(format!("bidirectional window parity failed: max_abs={max_abs}").into());
    }
    Ok(())
}
