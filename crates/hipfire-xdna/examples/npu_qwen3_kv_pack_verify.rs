//! Byte-exact AIE2P verification for token-major Qwen3 K/V packing.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hipfire_xdna::{NpuQwen3KvPack, SegmentedAttentionGeometry};

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: npu_qwen3_kv_pack_verify CACHE BUCKET BATCH".into());
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

    let elements = batch * bucket * geometry.kv_heads * geometry.head_dim;
    let keys = patterned_bits(elements, 0x37a5, 0x1357);
    let values = patterned_bits(elements, 0x71d3, 0x9bdf);
    let head_major_keys = token_to_head_major(&keys, geometry);
    let head_major_values = token_to_head_major(&values, geometry);
    let reference = geometry.pack_kv_bf16(&head_major_keys, &head_major_values)?;

    let xclbin = std::fs::read(format!("{}/final.xclbin", args[0]))?;
    let instructions = std::fs::read(format!("{}/insts.bin", args[0]))?;
    let mut packer = NpuQwen3KvPack::load(&xclbin, &instructions, geometry)?;
    let actual = packer.run(&keys, &values)?;
    compare(&actual, &reference)?;
    let repeated = packer.run(&keys, &values)?;
    compare(&repeated, &reference)?;
    println!(
        "qwen3-kv-pack-bf16 S={bucket} B={batch} Hkv={} D={} byte_exact=true repeat_stable=true",
        geometry.kv_heads, geometry.head_dim
    );
    Ok(())
}

#[cfg(target_os = "linux")]
fn patterned_bits(length: usize, multiplier: u32, bias: u32) -> Vec<u16> {
    (0..length)
        .map(|index| {
            let value = index as u32;
            value
                .wrapping_mul(multiplier)
                .wrapping_add(value.rotate_left(9))
                .wrapping_add(bias) as u16
        })
        .collect()
}

#[cfg(target_os = "linux")]
fn token_to_head_major(
    token_major: &[u16],
    geometry: hipfire_xdna::SegmentedAttentionGeometry,
) -> Vec<u16> {
    let mut head_major = vec![0u16; token_major.len()];
    for document in 0..geometry.dispatch_batch {
        for token in 0..geometry.sequence_bucket {
            for head in 0..geometry.kv_heads {
                for dim in 0..geometry.head_dim {
                    let source = (((document * geometry.sequence_bucket + token)
                        * geometry.kv_heads
                        + head)
                        * geometry.head_dim)
                        + dim;
                    let destination = (((document * geometry.kv_heads + head)
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
fn compare(actual: &[u8], reference: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    if actual == reference {
        return Ok(());
    }
    let first = actual
        .iter()
        .zip(reference)
        .position(|(actual, reference)| actual != reference)
        .unwrap_or(actual.len().min(reference.len()));
    Err(format!(
        "K/V-pack output differs at byte {first}: actual={:?} reference={:?}",
        &actual[first..actual.len().min(first + 16)],
        &reference[first..reference.len().min(first + 16)]
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
        "\"schema\": \"hipfire.npu_qwen3_kv_pack.v1\"".to_string(),
        format!("\"sequence_bucket\": {}", geometry.sequence_bucket),
        format!("\"dispatch_batch\": {}", geometry.dispatch_batch),
        format!("\"kv_heads\": {}", geometry.kv_heads),
        format!("\"head_dim\": {}", geometry.head_dim),
        "\"input_layout\": \"token_major_b_s_hkv_d_bf16\"".to_string(),
        "\"output_layout\": \"segmented_attention_kv\"".to_string(),
    ] {
        if !manifest.contains(&field) {
            return Err(format!("K/V-pack manifest missing {field}").into());
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("npu_qwen3_kv_pack_verify is Linux-only");
}
