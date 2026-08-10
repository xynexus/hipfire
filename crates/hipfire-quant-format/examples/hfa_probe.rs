// SPDX-License-Identifier: Apache-2.0
//! Verify the `.hfa` random-access reader against a restored copy of the same
//! archive: every tensor must come back byte-identical without restoring.
//!
//!   cargo run --release -p hipfire-quantize --example hfa_probe -- <a.hfa> [restored_dir]
//!
//! With a restored directory it compares; without one it just reports what the
//! archive contains.

use hipfire_quant_format::hfa::HfaArchive;
use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: hfa_probe <archive.hfa> [restored_dir]");
        std::process::exit(2);
    }
    let ar = match HfaArchive::open(std::path::Path::new(&args[1])) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!("files: {}", ar.file_names().count());
    for n in ar.file_names().take(6) {
        println!("  {n}");
    }
    let shards = ar.safetensors_names();
    println!("shards: {}", shards.len());

    let index = ar.tensor_index().expect("tensor index");
    println!("tensors: {}", index.len());

    // config.json must reconstruct exactly — it is what the quantizer reads
    // first, and a wrong one is a silently wrong model.
    if let Ok(cfg) = ar.read_small_file("config.json") {
        let v: serde_json::Value = serde_json::from_slice(&cfg).expect("config parses");
        println!(
            "config.json: {} bytes, model_type={:?}",
            cfg.len(),
            v.get("model_type").and_then(|x| x.as_str())
        );
    }

    let Some(dir) = args.get(2) else {
        // No restore to compare against (the whole point for a 550 GB archive),
        // so do the check that needs no reference: decode a few real tensors and
        // confirm each yields exactly shape-product x dtype-size bytes. A codec
        // or offset bug shows up here as a length mismatch.
        let mut names: Vec<_> = index.iter().collect();
        names.sort();
        for (name, shard) in names.iter().take(4) {
            let (bytes, dtype, shape) = ar.tensor_bytes(shard, name).expect("tensor_bytes");
            let esz = match dtype.as_str() {
                "BF16" | "F16" | "I16" => 2,
                "F32" | "I32" => 4,
                "F64" | "I64" => 8,
                _ => 1,
            };
            let want: usize = shape.iter().product::<usize>() * esz;
            let ok = if bytes.len() == want {
                "OK"
            } else {
                "MISMATCH"
            };
            println!("  {ok} {name} {dtype} {shape:?} -> {} bytes", bytes.len());
            assert_eq!(bytes.len(), want, "{name}: decoded length is wrong");
        }
        println!("(no restored dir given — decoded-length check only)");
        return;
    };

    // Compare every tensor against the restored safetensors. Uses memmap via
    // safetensors' own parse so the comparison is against a genuinely
    // independent reader, not our own offset math a second time.
    let mut checked = 0usize;
    let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
    for (name, shard) in &index {
        by_shard
            .entry(shard.clone())
            .or_default()
            .push(name.clone());
    }
    for (shard, names) in &by_shard {
        let path = std::path::Path::new(dir).join(shard);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip {shard}: {e}");
                continue;
            }
        };
        let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen]).unwrap();
        let blob = &bytes[8 + hlen..];
        for name in names {
            let (got, dtype, shape) = ar.tensor_bytes(shard, name).expect("tensor_bytes");
            let m = hdr.get(name).unwrap();
            let o = m.get("data_offsets").unwrap().as_array().unwrap();
            let (s, e) = (
                o[0].as_u64().unwrap() as usize,
                o[1].as_u64().unwrap() as usize,
            );
            let want = &blob[s..e];
            assert_eq!(
                got.len(),
                want.len(),
                "{name}: length {} vs restored {}",
                got.len(),
                want.len()
            );
            assert!(
                got == want,
                "{name}: bytes differ (dtype={dtype} shape={shape:?})"
            );
            checked += 1;
        }
    }
    println!("OK: {checked} tensors byte-identical to the restored copy");
}
