//! Byte-exact AIE2P verification for token-major Qwen3 query packing.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::{NpuQwen3QueryPack, SegmentedAttentionGeometry};

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: npu_qwen3_query_pack_verify CACHE BUCKET BATCH".into());
    }
    let bucket = args[1].parse::<usize>()?;
    let batch = args[2].parse::<usize>()?;
    let geometry = SegmentedAttentionGeometry {
        sequence_bucket: bucket,
        dispatch_batch: batch,
        query_heads: 16,
        kv_heads: 8,
        head_dim: 128,
    }
    .validate()?;
    validate_manifest(&args[0], geometry)?;

    let elements = batch * bucket * geometry.query_heads * geometry.head_dim;
    let token_major = (0..elements)
        .map(|index| {
            let value = index as u32;
            value
                .wrapping_mul(0x9e37)
                .wrapping_add(value.rotate_left(11))
                .wrapping_add(0x5a17) as u16
        })
        .collect::<Vec<_>>();
    let head_major = token_to_head_major(&token_major, geometry);
    let lengths = (0..batch)
        .map(|document| bucket.saturating_sub(document * 19 + 1).max(1) as u32)
        .collect::<Vec<_>>();

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let instructions = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let mut packer = NpuQwen3QueryPack::load(&xclbin, &instructions, geometry)?;
    verify_dispatch(&mut packer, &token_major, &head_major, &lengths)?;

    let second_lengths = (0..batch)
        .map(|document| (bucket / 2 + document * 7).min(bucket).max(1) as u32)
        .collect::<Vec<_>>();
    verify_dispatch(&mut packer, &token_major, &head_major, &second_lengths)?;
    println!(
        "qwen3-query-pack-bf16 S={bucket} B={batch} Hq={} D={} byte_exact=true dynamic_lengths={lengths:?}->{second_lengths:?}",
        geometry.query_heads, geometry.head_dim
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn verify_dispatch(
    packer: &mut hipfire_xdna::NpuQwen3QueryPack,
    token_major: &[u16],
    head_major: &[u16],
    lengths: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let reference = packer
        .geometry()
        .pack_q_bf16_with_lengths(head_major, lengths)?;
    let actual = packer.run(token_major, lengths)?;
    if actual != reference {
        let first = actual
            .iter()
            .zip(&reference)
            .position(|(actual, reference)| actual != reference)
            .unwrap_or(actual.len().min(reference.len()));
        return Err(format!(
            "query-pack output differs at byte {first}: actual={:?} reference={:?}",
            &actual[first..actual.len().min(first + 16)],
            &reference[first..reference.len().min(first + 16)]
        )
        .into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn token_to_head_major(
    token_major: &[u16],
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
) -> Vec<u16> {
    let mut head_major = vec![0u16; token_major.len()];
    for document in 0..geometry.dispatch_batch {
        for token in 0..geometry.sequence_bucket {
            for head in 0..geometry.query_heads {
                for dim in 0..geometry.head_dim {
                    let source = (((document * geometry.sequence_bucket + token)
                        * geometry.query_heads
                        + head)
                        * geometry.head_dim)
                        + dim;
                    let destination = (((document * geometry.query_heads + head)
                        * geometry.sequence_bucket
                        + token)
                        * geometry.head_dim)
                        + dim;
                    head_major[destination] = token_major[source];
                }
            }
        }
    }
    head_major
}

#[cfg(target_os = "linux")]
fn validate_manifest(
    cache: &str,
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = std::fs::read_to_string(format!("{cache}/manifest.json"))?;
    for field in [
        "\"schema\": \"hipfire.npu_qwen3_query_pack.v1\"".to_string(),
        format!("\"sequence_bucket\": {}", geometry.sequence_bucket),
        format!("\"dispatch_batch\": {}", geometry.dispatch_batch),
        format!("\"query_heads\": {}", geometry.query_heads),
        format!("\"kv_heads\": {}", geometry.kv_heads),
        format!("\"head_dim\": {}", geometry.head_dim),
        "\"input_layout\": \"token_major_b_s_qh_d_bf16\"".to_string(),
        "\"output_layout\": \"segmented_attention_q_with_length_trailers\"".to_string(),
    ] {
        if !manifest.contains(&field) {
            return Err(format!("query-pack manifest missing {field}").into());
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_qwen3_query_pack_verify is Linux-only");
}
