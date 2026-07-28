// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Generic safetensors → canonical `.hfq` conversion driver.
//!
//! Per AGENTS.md the offline format-conversion machinery lives here; the
//! family-specific tensor remap (names, half-layer merge, synthesised tensors)
//! lives in the arch crate (e.g. `hipfire_arch_zaya::ingest`). This driver reads
//! the HF checkpoint once via [`HfqFile::from_safetensors`], applies the arch
//! remap to produce canonical-named source-precision tensors, and writes a
//! self-describing `.hfq` the standard loader + `hipfire-quantize` accept.
//!
//! The write streams. A source-precision checkpoint is far larger than host RAM
//! at the sizes this driver exists for — a 35B bf16 model is ~65 GB — so nothing
//! here may hold the model. Payload length for an uncompressed tensor is known
//! from the source index without touching bytes, which is what lets the HFQM
//! index be written up front and each payload be copied straight from the
//! shard mmap afterwards. Peak RSS is a page-cache working set, not a model.

use std::error::Error;
use std::path::{Path, PathBuf};

use hipfire_runtime::hfq::{write_hfqm_package_streaming, HfqFile, HfqStreamEntry};

/// `hipfire-coexistence import safetensors --input <hf_dir> --output <out.hfq> [--arch <family>]`
pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut arch: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => input = it.next().map(PathBuf::from),
            "--output" => output = it.next().map(PathBuf::from),
            "--arch" => arch = it.next().cloned(),
            other => {
                return Err(format!("import safetensors: unexpected argument {other:?}").into())
            }
        }
    }
    let input = input.ok_or("import safetensors requires --input <hf_dir>")?;
    let output = output.ok_or("import safetensors requires --output <out.hfq>")?;

    // Resolve the family: explicit --arch wins, else config.json's model_type.
    let family = match arch {
        Some(a) => a.to_ascii_lowercase(),
        None => read_model_type(&input)?,
    };

    match family.as_str() {
        "zaya" => convert_zaya(&input, &output),
        other => Err(format!(
            "import safetensors: unsupported family {other:?} (only 'zaya' is implemented)"
        )
        .into()),
    }
}

fn read_model_type(dir: &Path) -> Result<String, Box<dyn Error>> {
    let cfg = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read {}/config.json: {e}", dir.display()))?;
    let v: serde_json::Value = serde_json::from_str(&cfg)?;
    Ok(v.get("model_type")
        .and_then(|m| m.as_str())
        .unwrap_or_default()
        .to_ascii_lowercase())
}

fn convert_zaya(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    use hipfire_arch_zaya::ingest;

    let hf = HfqFile::from_safetensors(input)?;

    // Snapshot the raw tensor identities up front (name/shape/qt/byte length) so
    // the borrow of `hf.tensors()` ends before the per-tensor payload reads.
    // `data_size` comes from the source index, so every payload length is known
    // without reading a single weight byte.
    let raw: Vec<(String, u8, Vec<u32>, usize)> = hf
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.quant_type, t.shape.clone(), t.data_size))
        .collect();

    // Canonical block count = (highest raw half-layer index + 1) / 2. Needed to
    // route the model-level residual scale onto the last block.
    let num_blocks = raw
        .iter()
        .filter_map(|(n, _, _, _)| n.strip_prefix("model.layers."))
        .filter_map(|r| r.split_once('.').and_then(|(i, _)| i.parse::<usize>().ok()))
        .max()
        .map(|max_idx| (max_idx + 1) / 2)
        .ok_or("import safetensors zaya: no model.layers.* tensors found")?;

    let mut entries: Vec<HfqStreamEntry> = Vec::new();
    // Source name for each entry, index-aligned with `entries` so the streaming
    // callback can find the bytes for entry `i` without a second lookup table.
    let mut sources: Vec<String> = Vec::new();
    let mut unmapped: Vec<String> = Vec::new();

    for (name, quant_type, shape, data_size) in raw {
        match ingest::canonical_name(&name, num_blocks) {
            Some(canonical) => {
                entries.push(HfqStreamEntry {
                    name: canonical,
                    quant_type,
                    shape,
                    group_size: 0,
                    data_len: data_size as u64,
                });
                sources.push(name);
            }
            None => unmapped.push(name),
        }
    }

    if !unmapped.is_empty() {
        return Err(format!(
            "import safetensors zaya: {} source tensors have no canonical mapping \
             (unexpected checkpoint layout); first few: {:?}",
            unmapped.len(),
            &unmapped[..unmapped.len().min(5)]
        )
        .into());
    }

    eprintln!(
        "import safetensors zaya: {} blocks, {} canonical tensors -> {}",
        num_blocks,
        entries.len(),
        output.display()
    );

    // `tensor_data` borrows straight out of the shard mmap for a
    // safetensors-backed file, so a payload is copied page-by-page from the
    // source into the output with no intermediate heap buffer.
    write_hfqm_package_streaming(
        output,
        ingest::ZAYA_ARCH_ID,
        &hf.metadata_json,
        &entries,
        |i, w| {
            let name = &sources[i];
            let (_, bytes) = hf.tensor_data(name).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no data for source tensor {name:?}"),
                )
            })?;
            w.write_all(bytes)
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    // Force-link the arch-spec bundle so ZAYA's `register_arch!` registration
    // survives into the test binary; without it `from_safetensors` rejects the
    // source as having no linked architecture spec.
    use hipfire_arch_specs as _;

    /// Minimal single-shard safetensors writer: 8-byte LE header length, JSON
    /// header, then payloads at the declared offsets.
    fn write_safetensors(path: &Path, tensors: &[(&str, Vec<u32>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut blob: Vec<u8> = Vec::new();
        for (name, shape, data) in tensors {
            let start = blob.len();
            blob.extend_from_slice(data);
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [start, blob.len()],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&blob);
        std::fs::write(path, out).unwrap();
    }

    /// The streaming write must reproduce every source payload byte-for-byte
    /// under its canonical name. `write_hfqm_package_streaming` enforces the
    /// declared length, so this is what catches a payload routed to the wrong
    /// entry — the failure a length check alone cannot see.
    #[test]
    fn streams_payloads_to_their_canonical_names() {
        let dir = std::env::temp_dir().join(format!("hipfire-import-st-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["ZayaForCausalLM"],"model_type":"zaya"}"#,
        )
        .unwrap();

        // Distinct payloads so a mis-routed tensor is visible, not just a
        // length match. Highest raw layer index 1 => num_blocks = 1.
        let final_norm = vec![0xAAu8; 8];
        let input_norm = vec![0xBBu8; 8];
        let res_scale = vec![0xCCu8; 4];
        write_safetensors(
            &dir.join("model.safetensors"),
            &[
                ("model.final_norm.weight", vec![4], final_norm.clone()),
                ("model.layers.0.input_norm.weight", vec![4], input_norm.clone()),
                (
                    "model.layers.1.res_scale.residual_scale",
                    vec![2],
                    res_scale.clone(),
                ),
            ],
        );

        let out = dir.join("out.hfq");
        convert_zaya(&dir, &out).expect("streaming conversion");

        let hfq = hipfire_runtime::hfq::HfqFile::open(&out).expect("open converted .hfq");
        for (canonical, expected) in [
            ("model.norm.weight", &final_norm),
            ("model.layers.0.input_layernorm.weight", &input_norm),
            (
                "model.layers.0.post_attention_residual_scale.residual_scale",
                &res_scale,
            ),
        ] {
            let (_, bytes) = hfq
                .tensor_data(canonical)
                .unwrap_or_else(|| panic!("missing canonical tensor {canonical}"));
            assert_eq!(bytes, &expected[..], "payload mismatch for {canonical}");
        }

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
