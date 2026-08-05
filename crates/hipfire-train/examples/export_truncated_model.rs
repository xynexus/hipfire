// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Export the first N layers of an `.hfq` as a bf16 safetensors directory.
//!
//! Two things this makes possible that nothing else does:
//!
//!   * **A per-depth oracle.** Truncating to N layers, quantising back to bf16
//!     and running both `dump_logits_qwen35` and this crate's walk over it gives
//!     a reference at depth N built entirely from paths already verified.
//!     Validated on Qwen3.5-0.8B truncated to 4 layers: cos 1.0001.
//!   * **Separating layer math from dequantisation.** The tensors written here
//!     are what THIS crate's loader produces — dequantised, AWQ folded. Both
//!     sides of the comparison then read identical weights, so a disagreement
//!     is in the layer math and cannot be blamed on the decode path (or the
//!     reverse, if they agree).
//!
//! The second is why this exists: the 35B is orthogonal to the runtime at one
//! token, every layer type ablates without isolating it, and quantisation is
//! verified only on a dense 0.8B.
//!
//! Run: cargo run --release -p hipfire-train --example export_truncated_model \
//!        <model.hfq> <out_dir> [n_layers]

use hipfire_runtime::hfq::HfqFile;
use hipfire_train::loader::{detect_prefix, DequantHfq, WeightSource};
use std::io::Write;

fn f32_to_bf16_bytes(v: &[f32]) -> Vec<u8> {
    // Round-to-nearest-even, matching what a bf16 cast should do rather than
    // truncating — truncation biases every weight toward zero.
    let mut out = Vec::with_capacity(v.len() * 2);
    for &x in v {
        let bits = x.to_bits();
        let rounded = ((bits >> 16) & 1).wrapping_add(0x7fff).wrapping_add(bits);
        out.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: export_truncated_model <model.hfq> <out_dir> [n_layers]");
        std::process::exit(2);
    }
    let n_layers: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(1);
    let hfq = HfqFile::open(std::path::Path::new(&args[1]))?;
    let dq = DequantHfq(&hfq);
    let src: &dyn WeightSource = &dq;
    let prefix = detect_prefix(src);
    let out = std::path::Path::new(&args[2]);
    std::fs::create_dir_all(out)?;

    // Keep every non-layer tensor, plus layers 0..n_layers.
    let names: Vec<String> = hfq
        .tensors()
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| !n.contains(".awq_scale.")) // folded into the weight already
        .filter(|n| match n.split(".layers.").nth(1) {
            None => true,
            Some(rest) => rest
                .split('.')
                .next()
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i < n_layers),
        })
        .collect();
    eprintln!(
        "exporting {} tensors ({} layers) from {}",
        names.len(),
        n_layers,
        args[1]
    );

    let mut header = serde_json::Map::new();
    let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(names.len());
    let mut off = 0usize;
    for (i, name) in names.iter().enumerate() {
        let (shape, v) = src.fetch_f32(name)?;
        let b = f32_to_bf16_bytes(&v);
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [off, off + b.len()],
            }),
        );
        off += b.len();
        blobs.push(b);
        if i % 200 == 0 {
            eprintln!("  {i}/{}", names.len());
        }
    }

    let mut hj = serde_json::to_vec(&header)?;
    while hj.len() % 8 != 0 {
        hj.push(b' ');
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out.join("model.safetensors"))?);
    f.write_all(&(hj.len() as u64).to_le_bytes())?;
    f.write_all(&hj)?;
    for b in &blobs {
        f.write_all(b)?;
    }
    f.flush()?;

    // Config: truncate the layer count and layer_types to match.
    let meta: serde_json::Value = {
        use hipfire_model::ModelSource;
        serde_json::from_str(hfq.metadata_json())?
    };
    let mut cfg = meta.get("config").cloned().unwrap_or(serde_json::json!({}));
    let tc_key = if cfg.get("text_config").is_some() {
        "text_config"
    } else {
        ""
    };
    {
        let tc = if tc_key.is_empty() {
            &mut cfg
        } else {
            cfg.get_mut(tc_key).unwrap()
        };
        tc["num_hidden_layers"] = serde_json::json!(n_layers);
        if let Some(lt) = tc.get("layer_types").and_then(|v| v.as_array()).cloned() {
            tc["layer_types"] = serde_json::json!(lt[..n_layers.min(lt.len())].to_vec());
        }
    }
    std::fs::write(out.join("config.json"), serde_json::to_vec_pretty(&cfg)?)?;
    if let Some(t) = meta.get("tokenizer").and_then(|v| v.as_str()) {
        std::fs::write(out.join("tokenizer.json"), t)?;
    }

    eprintln!(
        "wrote {} ({:.2} GB), prefix {prefix:?}",
        out.display(),
        off as f64 / 1e9
    );
    Ok(())
}
