//! Byte-exact AIE2P verification for segmented Qwen3 attention unpacking.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::{NpuQwen3AttentionUnpack, SegmentedAttentionGeometry};

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(3..=4).contains(&args.len()) {
        return Err(
            "usage: npu_qwen3_attention_unpack_verify CACHE BUCKET BATCH [QUERY_HEADS]".into(),
        );
    }
    let bucket = args[1].parse::<usize>()?;
    let batch = args[2].parse::<usize>()?;
    let query_heads = args
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16);
    let geometry = SegmentedAttentionGeometry {
        sequence_bucket: bucket,
        dispatch_batch: batch,
        query_heads,
        kv_heads: 8,
        head_dim: 128,
    }
    .validate()?;
    validate_manifest(&args[0], geometry)?;

    let elements = batch * bucket * query_heads * geometry.head_dim;
    let token_major = (0..elements)
        .map(|index| {
            let value = index as u32;
            value
                .wrapping_mul(0x4d2d)
                .wrapping_add(value.rotate_left(7))
                .wrapping_add(0xa531) as u16
        })
        .collect::<Vec<_>>();
    let head_major = token_to_head_major(&token_major, geometry);
    let segmented = geometry.pack_output_bf16(&head_major)?;

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let instructions = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let mut unpacker = NpuQwen3AttentionUnpack::load(&xclbin, &instructions, geometry)?;
    compare(&unpacker.run(&segmented)?, &token_major)?;
    compare(&unpacker.run(&segmented)?, &token_major)?;
    println!(
        "qwen3-attention-unpack-bf16 S={bucket} B={batch} Hq={query_heads} D={} byte_exact=true repeat_stable=true",
        geometry.head_dim
    );
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
fn compare(actual: &[u16], reference: &[u16]) -> Result<(), Box<dyn std::error::Error>> {
    if actual == reference {
        return Ok(());
    }
    let first = actual
        .iter()
        .zip(reference)
        .position(|(actual, reference)| actual != reference)
        .unwrap_or(actual.len().min(reference.len()));
    Err(format!(
        "attention-unpack output differs at value {first}: actual={:?} reference={:?}",
        &actual[first..actual.len().min(first + 8)],
        &reference[first..reference.len().min(first + 8)]
    )
    .into())
}

#[cfg(target_os = "linux")]
fn validate_manifest(
    cache: &str,
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = std::fs::read_to_string(format!("{cache}/manifest.json"))?;
    for field in [
        "\"schema\": \"hipfire.npu_qwen3_attention_unpack.v1\"".to_string(),
        format!("\"sequence_bucket\": {}", geometry.sequence_bucket),
        format!("\"dispatch_batch\": {}", geometry.dispatch_batch),
        format!("\"query_heads\": {}", geometry.query_heads),
        format!("\"head_dim\": {}", geometry.head_dim),
        "\"input_layout\": \"segmented_attention_output\"".to_string(),
        "\"output_layout\": \"token_major_b_s_qh_d_bf16\"".to_string(),
    ] {
        if !manifest.contains(&field) {
            return Err(format!("attention-unpack manifest missing {field}").into());
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_qwen3_attention_unpack_verify is Linux-only");
}
