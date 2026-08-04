// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic canonical `.hfq` → HuggingFace snapshot export driver.
//!
//! The mirror of [`crate::import_safetensors`], and split the same way: the
//! container work (decode, shard, index, sidecars) lives here, the per-family
//! name remap lives in the arch crate (`hipfire_arch_zaya::ingest::hf_name`).
//!
//! **Fidelity.** A tensor stored in a source precision (`BF16`/`F16`/`F32`)
//! exports byte-for-byte. So does one stored under a lossless recoding
//! (`Bf16Huff`, `Bf16Lut3`): `HfqFile` rewrites the index to the logical type at
//! open and expands on read, so a DFloat11-style artifact round-trips
//! bit-identically through this driver. A *lossy* quant (`mq4`, `oq4`, `hfp4`, …)
//! is refused rather than silently dequantized — a 4-bit artifact expanded to
//! BF16 is not the checkpoint it was made from, and emitting one as a plain
//! safetensors snapshot invites it being published as a base model.
//!
//! **Memory.** Like the import side, nothing here may hold the model. Every
//! payload length is known from the (already logical) index before any byte is
//! read, so each shard's safetensors header is written up front and payloads are
//! streamed one tensor at a time. Peak RSS is the largest single tensor.

use std::collections::BTreeMap;
use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};

use hipfire_quant_format::QuantType;
use hipfire_runtime::hfq::{hf_sidecars_from_metadata, HfqFile};

/// HF's own default `max_shard_size`, and what the reference checkpoints in the
/// wild are cut at.
const DEFAULT_SHARD_SIZE: u64 = 5_000_000_000;

/// `hipfire-coexistence export safetensors --input <model.hfq> --output <hf_dir>`
pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut arch: Option<String> = None;
    let mut shard_size: u64 = DEFAULT_SHARD_SIZE;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next().map(PathBuf::from),
            "--output" => output = it.next().map(PathBuf::from),
            "--arch" => arch = it.next().cloned(),
            "--shard-size" => {
                let raw = it.next().ok_or("--shard-size requires a byte count")?;
                shard_size = parse_size(raw)?;
            }
            other => {
                return Err(format!("export safetensors: unexpected argument {other:?}").into())
            }
        }
    }
    let input = input.ok_or("export safetensors requires --input <model.hfq>")?;
    let output = output.ok_or("export safetensors requires --output <hf_dir>")?;
    if shard_size == 0 {
        return Err("export safetensors: --shard-size must be non-zero".into());
    }

    let hfq = HfqFile::open(&input)?;
    let family = match arch {
        Some(a) => a.to_ascii_lowercase(),
        None => family_for_arch_id(hfq.arch_id)?,
    };
    match family.as_str() {
        "zaya" => export_zaya(&hfq, &output, shard_size),
        other => Err(format!(
            "export safetensors: unsupported family {other:?} (only 'zaya' is implemented)"
        )
        .into()),
    }
}

/// Accepts a plain byte count or a `K`/`M`/`G` suffix (decimal, matching how HF
/// states shard sizes).
fn parse_size(raw: &str) -> Result<u64, Box<dyn Error>> {
    // Drop a trailing byte marker first, so "4GB" reaches the suffix match as
    // "4G" rather than failing to parse.
    let s = raw.trim().trim_end_matches(['B', 'b']);
    let (digits, mult) = match s.chars().last() {
        Some('K' | 'k') => (&s[..s.len() - 1], 1_000),
        Some('M' | 'm') => (&s[..s.len() - 1], 1_000_000),
        Some('G' | 'g') => (&s[..s.len() - 1], 1_000_000_000),
        _ => (s, 1),
    };
    let n: u64 = digits
        .trim()
        .parse()
        .map_err(|_| format!("export safetensors: cannot parse size {raw:?}"))?;
    Ok(n * mult)
}

fn family_for_arch_id(arch_id: u32) -> Result<String, Box<dyn Error>> {
    match arch_id {
        hipfire_arch_zaya::ingest::ZAYA_ARCH_ID => Ok("zaya".to_string()),
        other => Err(format!(
            "export safetensors: arch id {other} has no export mapping; pass --arch <family>"
        )
        .into()),
    }
}

/// One tensor's place in the export: where it came from and where it lands.
struct Planned {
    canonical: String,
    hf_name: String,
    dtype: &'static str,
    shape: Vec<u32>,
    /// Logical (expanded) byte length — the index is already in logical terms.
    len: u64,
}

fn export_zaya(hfq: &HfqFile, out_dir: &Path, shard_size: u64) -> Result<(), Box<dyn Error>> {
    use hipfire_arch_zaya::ingest;

    // Canonical block count from the collapsed names, mirroring how the import
    // side derives it from the raw half-layer indices.
    let num_blocks = hfq
        .tensors()
        .iter()
        .filter_map(|t| t.name.strip_prefix("model.layers."))
        .filter_map(|r| r.split_once('.').and_then(|(i, _)| i.parse::<usize>().ok()))
        .max()
        .map(|max_idx| max_idx + 1)
        .ok_or("export safetensors zaya: no model.layers.* tensors found")?;

    let mut planned: Vec<Planned> = Vec::with_capacity(hfq.tensors().len());
    let mut unmapped: Vec<String> = Vec::new();
    for t in hfq.tensors() {
        let dtype = source_dtype(t.quant_type, &t.name)?;
        match ingest::hf_name(&t.name, num_blocks) {
            Some(hf_name) => planned.push(Planned {
                canonical: t.name.clone(),
                hf_name,
                dtype,
                shape: t.shape.clone(),
                len: t.data_size as u64,
            }),
            None => unmapped.push(t.name.clone()),
        }
    }
    if !unmapped.is_empty() {
        return Err(format!(
            "export safetensors zaya: {} canonical tensors have no HF mapping; first few: {:?}",
            unmapped.len(),
            &unmapped[..unmapped.len().min(5)]
        )
        .into());
    }

    // Deterministic order so a re-export is byte-comparable with the last one.
    planned.sort_by(|a, b| a.hf_name.cmp(&b.hf_name));

    std::fs::create_dir_all(out_dir)?;
    let shards = plan_shards(&planned, shard_size);
    let total_size: u64 = planned.iter().map(|p| p.len).sum();

    eprintln!(
        "export safetensors zaya: {} blocks, {} tensors, {:.2} GB across {} shard(s) -> {}",
        num_blocks,
        planned.len(),
        total_size as f64 / 1e9,
        shards.len(),
        out_dir.display()
    );

    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    for (i, range) in shards.iter().enumerate() {
        let file_name = shard_file_name(i, shards.len());
        write_shard(hfq, &out_dir.join(&file_name), &planned[range.clone()])?;
        for p in &planned[range.clone()] {
            weight_map.insert(p.hf_name.clone(), file_name.clone());
        }
    }

    // A single-file model carries no index, matching the HF convention.
    if shards.len() > 1 {
        let index = serde_json::json!({
            "metadata": { "total_size": total_size },
            "weight_map": weight_map,
        });
        std::fs::write(
            out_dir.join("model.safetensors.index.json"),
            serde_json::to_vec_pretty(&index)?,
        )?;
    }

    write_sidecars(hfq, out_dir)?;
    Ok(())
}

/// Map a stored quant type to its safetensors dtype, refusing anything that
/// would need dequantization.
fn source_dtype(quant_type: u8, name: &str) -> Result<&'static str, Box<dyn Error>> {
    let qt = QuantType::from_code(quant_type)
        .ok_or_else(|| format!("export safetensors: tensor {name:?} has unknown quant code {quant_type}"))?;
    match qt {
        QuantType::BF16 => Ok("BF16"),
        QuantType::F16 => Ok("F16"),
        QuantType::F32 => Ok("F32"),
        // Still a recoding here means the index was not expanded — the residency
        // opt-out. Exporting the packed bytes under a BF16 label would emit a
        // corrupt checkpoint, so refuse and name the cause.
        other if other.is_lossless_recoding() => Err(format!(
            "export safetensors: tensor {name:?} is still stored as {other:?}; \
             unset HIPFIRE_BF16L3_RESIDENT so the loader expands it"
        )
        .into()),
        other => Err(format!(
            "export safetensors: tensor {name:?} is quantized ({other:?}). Exporting it would \
             require dequantizing to BF16, which does not reproduce the source checkpoint and is \
             not implemented. Export a source-precision or losslessly-recoded .hfq instead."
        )
        .into()),
    }
}

/// Greedy fill to `shard_size`, never emitting an empty shard, so a single
/// tensor larger than the target still gets its own file rather than failing.
fn plan_shards(planned: &[Planned], shard_size: u64) -> Vec<std::ops::Range<usize>> {
    let mut shards = Vec::new();
    let mut start = 0usize;
    let mut acc = 0u64;
    for (i, p) in planned.iter().enumerate() {
        if i > start && acc + p.len > shard_size {
            shards.push(start..i);
            start = i;
            acc = 0;
        }
        acc += p.len;
    }
    if start < planned.len() {
        shards.push(start..planned.len());
    }
    shards
}

fn shard_file_name(i: usize, total: usize) -> String {
    if total == 1 {
        "model.safetensors".to_string()
    } else {
        format!("model-{:05}-of-{:05}.safetensors", i + 1, total)
    }
}

/// Write one safetensors shard: 8-byte little-endian header length, the JSON
/// header, then the payloads at their declared offsets.
///
/// The header is built entirely from the index, so it can be written before any
/// tensor is read and each payload streamed straight out afterwards.
fn write_shard(hfq: &HfqFile, path: &Path, tensors: &[Planned]) -> Result<(), Box<dyn Error>> {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for p in tensors {
        let end = offset + p.len;
        header.insert(
            p.hf_name.clone(),
            serde_json::json!({
                "dtype": p.dtype,
                "shape": p.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let mut header_bytes = serde_json::to_vec(&header)?;
    // Pad the header with spaces so the data section starts 8-byte aligned;
    // readers mmap the payload region and misalignment costs them a copy.
    let pad = (8 - ((8 + header_bytes.len()) % 8)) % 8;
    header_bytes.extend(std::iter::repeat_n(b' ', pad));

    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);
    w.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&header_bytes)?;

    for p in tensors {
        // `tensor_data_vec` expands a losslessly-recoded payload; one tensor is
        // resident at a time, never the model.
        let (_, bytes) = hfq
            .tensor_data_vec(&p.canonical)
            .ok_or_else(|| format!("export safetensors: no data for tensor {:?}", p.canonical))?;
        if bytes.len() as u64 != p.len {
            return Err(format!(
                "export safetensors: tensor {:?} expanded to {} bytes, index declared {}",
                p.canonical,
                bytes.len(),
                p.len
            )
            .into());
        }
        w.write_all(&bytes)?;
    }
    w.flush()?;
    Ok(())
}

/// Restore the snapshot's non-weight files: the verbatim `hf_sidecars` blob
/// first, then anything the runtime keys can still fill that the blob did not
/// carry (an `.hfq` produced before sidecar capture, or by the quantizer).
fn write_sidecars(hfq: &HfqFile, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut written = 0usize;
    for (rel, bytes) in hf_sidecars_from_metadata(&hfq.metadata_json) {
        let dest = out_dir.join(&rel);
        // Defensive: a crafted key must not escape the output directory.
        if !dest.starts_with(out_dir) {
            eprintln!("export safetensors: skipping sidecar {rel:?} — escapes the output dir");
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&dest, bytes)?;
        written += 1;
    }

    let meta: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).unwrap_or_else(|_| serde_json::json!({}));

    // `tokenizer` is stored as raw text; the rest are parsed values.
    if let Some(text) = meta.get("tokenizer").and_then(|t| t.as_str()) {
        written += write_if_absent(&out_dir.join("tokenizer.json"), text.as_bytes())? as usize;
    }
    for (key, file) in [
        ("config", "config.json"),
        ("tokenizer_config", "tokenizer_config.json"),
        ("generation_config", "generation_config.json"),
    ] {
        if let Some(v) = meta.get(key) {
            let bytes = serde_json::to_vec_pretty(v)?;
            written += write_if_absent(&out_dir.join(file), &bytes)? as usize;
        }
    }

    eprintln!("export safetensors: {written} sidecar file(s) written");
    Ok(())
}

/// Never overwrite a verbatim sidecar with a re-serialized one — the captured
/// bytes are the source of truth when both exist.
fn write_if_absent(path: &Path, bytes: &[u8]) -> std::io::Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    std::fs::write(path, bytes)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planned(name: &str, len: u64) -> Planned {
        Planned {
            canonical: name.to_string(),
            hf_name: name.to_string(),
            dtype: "BF16",
            shape: vec![len as u32 / 2],
            len,
        }
    }

    #[test]
    fn shards_fill_greedily_and_never_emit_an_empty_shard() {
        // 6 alone (6+5 would overflow 10), then 5+4 fits.
        let items = vec![planned("a", 6), planned("b", 5), planned("c", 4)];
        assert_eq!(plan_shards(&items, 10), vec![0..1, 1..3]);

        // A single tensor larger than the target still gets a shard of its own.
        let big = vec![planned("a", 100), planned("b", 1)];
        assert_eq!(plan_shards(&big, 10), vec![0..1, 1..2]);

        assert_eq!(plan_shards(&items, 1_000), vec![0..3]);
    }

    #[test]
    fn shard_names_follow_the_hf_convention() {
        assert_eq!(shard_file_name(0, 1), "model.safetensors");
        assert_eq!(shard_file_name(0, 16), "model-00001-of-00016.safetensors");
        assert_eq!(shard_file_name(15, 16), "model-00016-of-00016.safetensors");
    }

    #[test]
    fn sizes_parse_with_and_without_suffixes() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("5G").unwrap(), 5_000_000_000);
        assert_eq!(parse_size("4GB").unwrap(), 4_000_000_000);
        assert_eq!(parse_size("512M").unwrap(), 512_000_000);
        assert!(parse_size("nonsense").is_err());
    }

    /// The fidelity claim, exercised rather than asserted: a tensor stored under
    /// a lossless recoding must leave the exporter as the exact BF16 bytes it
    /// was built from. This is what makes hfq → HF a true round trip for a
    /// DFloat11-style artifact instead of a lossy one.
    #[test]
    fn losslessly_recoded_bf16_exports_bit_identically() {
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

        // Exponents clustered the way real weights are, so the coder shrinks.
        let n = 8192usize;
        let mut raw = Vec::with_capacity(n * 2);
        for i in 0..n {
            // Exponent lives in bits 7..14; step it over a narrow range.
            let exp: u16 = 0x3F80u16.wrapping_add((((i * 7) % 5) as u16) << 7);
            let mantissa = (i * 31 % 128) as u16;
            raw.extend_from_slice(&(exp | mantissa).to_le_bytes());
        }
        let packed = hipfire_primitives::bf16_huff::encode_if_smaller(&raw)
            .expect("test data must compress, else this asserts nothing");
        assert!(packed.len() < raw.len());

        let dir = std::env::temp_dir().join(format!("hipfire-export-huff-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let hfq_path = dir.join("model.hfq");

        let entries = vec![
            HfqMemTensor {
                name: "model.norm.weight".into(),
                quant_type: QuantType::Bf16Huff as u8,
                shape: vec![n as u32],
                group_size: 0,
                data: packed.clone(),
            },
            HfqMemTensor {
                name: "model.layers.0.input_layernorm.weight".into(),
                quant_type: QuantType::Bf16Huff as u8,
                shape: vec![n as u32],
                group_size: 0,
                data: packed,
            },
        ];
        write_hfqm_package_mem(
            &hfq_path,
            hipfire_arch_zaya::ingest::ZAYA_ARCH_ID,
            "{}",
            &entries,
        )
        .unwrap();

        let out = dir.join("hf");
        let hfq = HfqFile::open(&hfq_path).unwrap();
        export_zaya(&hfq, &out, DEFAULT_SHARD_SIZE).expect("export");

        // Parse the shard and compare against the pre-compression bytes.
        let blob = std::fs::read(out.join("model.safetensors")).unwrap();
        let hlen = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&blob[8..8 + hlen]).unwrap();
        let data = &blob[8 + hlen..];

        for hf_name in ["model.final_norm.weight", "model.layers.0.input_norm.weight"] {
            let e = header.get(hf_name).unwrap_or_else(|| {
                panic!("exported header is missing {hf_name}: {header}")
            });
            assert_eq!(e["dtype"], "BF16", "recoded tensor must present as BF16");
            let s = e["data_offsets"][0].as_u64().unwrap() as usize;
            let end = e["data_offsets"][1].as_u64().unwrap() as usize;
            assert_eq!(
                &data[s..end],
                &raw[..],
                "{hf_name} did not expand to the original BF16 bytes"
            );
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A quantized tensor must be refused, not silently dequantized.
    #[test]
    fn quantized_tensors_are_refused() {
        let err = source_dtype(QuantType::Q8F16 as u8, "model.embed_tokens.weight")
            .expect_err("quantized export must be refused");
        let msg = err.to_string();
        assert!(msg.contains("quantized"), "unexpected message: {msg}");
        assert!(msg.contains("not implemented"), "unexpected message: {msg}");

        assert_eq!(source_dtype(QuantType::BF16 as u8, "x").unwrap(), "BF16");
        assert_eq!(source_dtype(QuantType::F32 as u8, "x").unwrap(), "F32");
    }
}
