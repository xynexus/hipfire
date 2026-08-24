// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Is a quantized artifact FAITHFUL to its source? Compares decoded weights
//! against the bf16 originals in the `.hfa` archive.
//!
//! Every parity test in this tree checks that two decode paths agree. They all
//! read the SAME quantized bytes, so if the quantization itself is wrong they
//! agree on garbage and every check passes. This is the missing axis: the source
//! archive is the oracle, and nothing else in the tree compares against it.
//!
//! Reads tensors out of the archive by range -- no 168 GiB restore, no model
//! load -- so it works on a model far too large to serve.
//!
//!   cargo run --release -p hipfire-runtime --example oracle_quant_fidelity \
//!       <artifact.hfq> <source.hfa> [name-substring]

use hipfire_quant_format::hfa::HfaArchive;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::oq8_arch::{normalize_compact_overlays, split_compact_planes};

const GROUP: usize = 256;

fn f16(bits: u16) -> f32 {
    hipfire_primitives::conv::f16_to_f32(bits)
}
fn bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
fn sext4(n: u8) -> i32 {
    ((n as i32) << 28) >> 28
}

/// Decode OqPlusCompact interleaved blocks (`[f16 scale][128 nibbles][table]`).
fn decode_compact(blob: &[u8], m: usize, k: usize, stride: usize, rows: usize) -> Vec<f32> {
    let ng = k / GROUP;
    let n_out = (stride - 130) / 2;
    let mut w = vec![0f32; rows * k];
    for row in 0..rows {
        for g in 0..ng {
            let b = (row * ng + g) * stride;
            let sc = f16(u16::from_le_bytes([blob[b], blob[b + 1]]));
            let mut code = [0i32; GROUP];
            for i in 0..128 {
                let byte = blob[b + 2 + i];
                code[2 * i] = sext4(byte & 0x0f);
                code[2 * i + 1] = sext4(byte >> 4);
            }
            for e in 0..n_out {
                let idx = blob[b + 130 + 2 * e] as usize;
                code[idx] = blob[b + 130 + 2 * e + 1] as i8 as i32;
            }
            for i in 0..GROUP {
                w[row * k + g * GROUP + i] = sc * code[i] as f32;
            }
        }
    }
    let _ = m;
    w
}

/// Decode canonical Oq8 (`[f16 scale][256 int8]`, 258 B/group).
fn decode_oq8(blob: &[u8], k: usize, rows: usize) -> Vec<f32> {
    const SRC: usize = 258;
    let ng = k / GROUP;
    let mut w = vec![0f32; rows * k];
    for row in 0..rows {
        for g in 0..ng {
            let b = (row * ng + g) * SRC;
            let sc = f16(u16::from_le_bytes([blob[b], blob[b + 1]]));
            for i in 0..GROUP {
                w[row * k + g * GROUP + i] = sc * (blob[b + 2 + i] as i8) as f32;
            }
        }
    }
    w
}

/// Source bytes -> f32, for the dtypes a HuggingFace checkpoint actually uses.
fn source_to_f32(bytes: &[u8], dtype: &str) -> Option<Vec<f32>> {
    Some(match dtype {
        "BF16" => bytes
            .chunks_exact(2)
            .map(|c| bf16(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F16" => bytes
            .chunks_exact(2)
            .map(|c| f16(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => return None,
    })
}

/// Cosine similarity — the metric that survives quantization noise while still
/// collapsing when the weights are structurally wrong.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// hfq names carry a `language_model` segment the HF checkpoint does not.
fn source_name_candidates(hfq_name: &str) -> Vec<String> {
    let mut v = vec![hfq_name.to_string()];
    if let Some(stripped) = hfq_name.strip_prefix("model.language_model.") {
        v.push(format!("model.{stripped}"));
        v.push(stripped.to_string());
    }
    v.push(format!("model.{hfq_name}"));
    v
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (hfq_path, hfa_path) = (a[0].clone(), a[1].clone());
    let filter = a.get(2).cloned().unwrap_or_default();

    let hfq = HfqFile::open(std::path::Path::new(&hfq_path)).expect("open hfq");
    let hfa = HfaArchive::open(std::path::Path::new(&hfa_path)).expect("open hfa");
    let index = hfa.tensor_index().expect("hfa index");
    println!("source archive: {} tensors", index.len());

    let mut checked = 0;
    let mut worst = (1.0f64, String::new());
    for info in hfq.tensors() {
        if !filter.is_empty() && !info.name.contains(&filter) {
            continue;
        }
        // 1-D tensors (norms, biases) are exactly the ones worth checking: they
        // are stored UNROTATED, so a corrupted one is visible here while passing
        // every quant parity test in the tree.
        let (m, k) = match info.shape.len() {
            1 => (1usize, info.shape[0] as usize),
            2 => (info.shape[0] as usize, info.shape[1] as usize),
            _ => continue,
        };
        let rotated = matches!(info.quant_type, 35 | 36);
        if rotated && k % GROUP != 0 {
            continue;
        }
        let Some((rel, src_name)) = source_name_candidates(&info.name)
            .into_iter()
            .find_map(|n| index.get(&n).map(|r| (r.clone(), n)))
        else {
            continue;
        };
        let Some((_, data)) = hfq.tensor_data_pread(&info.name) else {
            continue;
        };
        // Cap rows: a 248320-row lm_head does not need all of them to answer this.
        let rows = m.min(64);
        let decoded = match info.quant_type {
            36 => {
                let stride = data.len() / (m * (k / GROUP));
                decode_compact(&data, m, k, stride, rows)
            }
            35 => decode_oq8(&data, k, rows),
            // Unrotated passthrough dtypes: the CONTROL. OQ weights are stored
            // FWHT-rotated and AWQ-pre-scaled, so they cannot match the source
            // basis and a low cosine there means nothing. A BF16/F16 tensor is
            // stored as-is, so it must match -- if it does not, the harness
            // (name mapping, shard offsets, dtype) is what is broken.
            16 => data
                .chunks_exact(2)
                .take(rows * k)
                .map(|c| bf16(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            1 => data
                .chunks_exact(2)
                .take(rows * k)
                .map(|c| f16(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            _ => continue,
        };
        let Ok((sbytes, dtype, sshape)) = hfa.tensor_bytes(&rel, &src_name) else {
            continue;
        };
        let Some(src) = source_to_f32(&sbytes, &dtype) else {
            continue;
        };
        // A 1-D hfq tensor is modelled as 1xK here; the source keeps it [K].
        let shape_ok = match sshape.len() {
            1 => m == 1 && sshape[0] == k,
            2 => sshape[0] == m && sshape[1] == k,
            _ => false,
        };
        if !shape_ok {
            println!("  {} SHAPE MISMATCH hfq {m}x{k} vs source {sshape:?}", info.name);
            continue;
        }
        let cos = cosine(&decoded[..rows * k], &src[..rows * k]);
        // OQ weights are stored FWHT-rotated and AWQ-pre-scaled, so a low cosine
        // against the source basis is EXPECTED and says nothing -- verified by
        // running this on a known-good 35B, which reads ~0 on every OQ tensor
        // exactly like the model under suspicion. Only unrotated dtypes carry a
        // verdict. Comparing the rotated ones needs the inverse transform, which
        // this tool does not implement.
        let tag = if rotated {
            "rotated-basis (no verdict)"
        } else if cos > 0.99 {
            "ok"
        } else if cos > 0.9 {
            "SUSPECT"
        } else {
            "BROKEN"
        };
        println!(
            "  {:<62} qt={} {m}x{k}  cosine {cos:.6}  {tag}",
            info.name.rsplit('.').take(4).collect::<Vec<_>>().join("."),
            info.quant_type
        );
        if !rotated && cos < worst.0 {
            worst = (cos, info.name.clone());
        }
        checked += 1;
    }
    println!(
        "\nchecked {checked}; worst UNROTATED cosine {:.6} on {}",
        worst.0,
        if worst.1.is_empty() { "(none)" } else { &worst.1 }
    );
}
