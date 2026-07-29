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

use std::error::Error;
use std::path::{Path, PathBuf};

use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};

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

    // Snapshot the raw tensor identities up front (name/shape/qt) so the borrow
    // of `hf.tensors()` ends before the per-tensor `tensor_data_vec` reads.
    let raw: Vec<(String, u8, Vec<u32>)> = hf
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.quant_type, t.shape.clone()))
        .collect();

    // Canonical block count = (highest raw half-layer index + 1) / 2. Needed to
    // route the model-level residual scale onto the last block.
    let num_blocks = raw
        .iter()
        .filter_map(|(n, _, _)| n.strip_prefix("model.layers."))
        .filter_map(|r| r.split_once('.').and_then(|(i, _)| i.parse::<usize>().ok()))
        .max()
        .map(|max_idx| (max_idx + 1) / 2)
        .ok_or("import safetensors zaya: no model.layers.* tensors found")?;

    let mut entries: Vec<HfqMemTensor> = Vec::new();
    let mut unmapped: Vec<String> = Vec::new();

    for (name, quant_type, shape) in raw {
        match ingest::canonical_name(&name, num_blocks) {
            Some(canonical) => {
                let (_, data) = hf
                    .tensor_data_vec(&name)
                    .ok_or_else(|| format!("no data for source tensor {name:?}"))?;
                entries.push(HfqMemTensor {
                    name: canonical,
                    quant_type,
                    shape,
                    group_size: 0,
                    data,
                });
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

    write_hfqm_package_mem(output, ingest::ZAYA_ARCH_ID, &hf.metadata_json, &entries)?;
    Ok(())
}
