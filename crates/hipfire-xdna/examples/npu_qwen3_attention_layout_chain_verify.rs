//! End-to-end verification of the NPU Q/K/V pack → attention → unpack chain.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::{
        NpuQwen3AttentionUnpack, NpuQwen3KvPack, NpuQwen3QueryPack, NpuSegmentedAttention,
        SegmentedAttentionGeometry,
    };

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 6 {
        return Err("usage: npu_qwen3_attention_layout_chain_verify Q_CACHE KV_CACHE ATTENTION_CACHE UNPACK_CACHE BUCKET BATCH".into());
    }
    let bucket = args[4].parse::<usize>()?;
    let batch = args[5].parse::<usize>()?;
    let geometry = SegmentedAttentionGeometry {
        sequence_bucket: bucket,
        dispatch_batch: batch,
        query_heads: 16,
        kv_heads: 8,
        head_dim: 128,
    }
    .validate()?;
    let q_elements = batch * bucket * geometry.query_heads * geometry.head_dim;
    let kv_elements = batch * bucket * geometry.kv_heads * geometry.head_dim;
    let queries = values(q_elements, 0.001_731, 0.21);
    let keys = values(kv_elements, 0.002_117, 0.18);
    let values = values(kv_elements, 0.001_337, 0.29);
    let lengths = (0..batch)
        .map(|document| bucket.saturating_sub(document * 17 + 1).max(1) as u32)
        .collect::<Vec<_>>();

    let (q_xclbin, q_instructions) = image(&args[0])?;
    let (kv_xclbin, kv_instructions) = image(&args[1])?;
    let (attention_xclbin, attention_instructions) = image(&args[2])?;
    let (unpack_xclbin, unpack_instructions) = image(&args[3])?;
    let mut query_pack = NpuQwen3QueryPack::load(&q_xclbin, &q_instructions, geometry)?;
    let mut kv_pack = NpuQwen3KvPack::load(&kv_xclbin, &kv_instructions, geometry)?;
    let mut attention =
        NpuSegmentedAttention::load(&attention_xclbin, &attention_instructions, geometry)?;
    let mut unpack = NpuQwen3AttentionUnpack::load(&unpack_xclbin, &unpack_instructions, geometry)?;

    let packed_q = query_pack.run(&queries, &lengths)?;
    let packed_kv = kv_pack.run(&keys, &values)?;
    let packed_output = attention.run_packed(&packed_q, &packed_kv)?;
    let chained = unpack.run(&packed_output)?;

    let head_major_q = token_to_head_major(&queries, geometry.query_heads, geometry);
    let head_major_k = token_to_head_major(&keys, geometry.kv_heads, geometry);
    let head_major_v = token_to_head_major(&values, geometry.kv_heads, geometry);
    let canonical = attention.run(&head_major_q, &head_major_k, &head_major_v, &lengths)?;
    let reference = head_to_token_major(&canonical, geometry.query_heads, geometry);
    if chained != reference {
        let first = chained
            .iter()
            .zip(&reference)
            .position(|(actual, reference)| actual != reference)
            .unwrap_or(chained.len().min(reference.len()));
        return Err(format!(
            "NPU attention layout chain differs at value {first}: actual={:?} reference={:?}",
            &chained[first..chained.len().min(first + 8)],
            &reference[first..reference.len().min(first + 8)]
        )
        .into());
    }
    println!(
        "qwen3-attention-layout-chain S={bucket} B={batch} byte_exact=true lengths={lengths:?}"
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn image(cache: &str) -> Result<(Vec<u8>, Vec<u8>), std::io::Error> {
    Ok((
        std::fs::read(format!("{cache}/final.xclbin"))?,
        std::fs::read(format!("{cache}/insts.bin"))?,
    ))
}

#[cfg(target_os = "linux")]
fn values(length: usize, frequency: f32, scale: f32) -> Vec<u16> {
    (0..length)
        .map(|index| {
            hipfire_primitives::conv::f32_to_bf16_bits(
                (index as f32 * frequency).sin() * scale + (index % 23) as f32 * 0.001,
            )
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn token_to_head_major(
    token_major: &[u16],
    heads: usize,
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
) -> Vec<u16> {
    transpose(token_major, heads, geometry, true)
}

#[cfg(target_os = "linux")]
fn head_to_token_major(
    head_major: &[u16],
    heads: usize,
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
) -> Vec<u16> {
    transpose(head_major, heads, geometry, false)
}

#[cfg(target_os = "linux")]
fn transpose(
    source: &[u16],
    heads: usize,
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
    token_to_head: bool,
) -> Vec<u16> {
    let mut destination = vec![0u16; source.len()];
    for document in 0..geometry.dispatch_batch {
        for token in 0..geometry.sequence_bucket {
            for head in 0..heads {
                for dim in 0..geometry.head_dim {
                    let token_index = (((document * geometry.sequence_bucket + token) * heads
                        + head)
                        * geometry.head_dim)
                        + dim;
                    let head_index = (((document * heads + head) * geometry.sequence_bucket
                        + token)
                        * geometry.head_dim)
                        + dim;
                    if token_to_head {
                        destination[head_index] = source[token_index];
                    } else {
                        destination[token_index] = source[head_index];
                    }
                }
            }
        }
    }
    destination
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_qwen3_attention_layout_chain_verify is Linux-only");
}
