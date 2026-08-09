// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare two `ModelSource`s tensor-by-tensor through the TRAIT.
//!
//! The point is to prove that reading a model through one container yields the
//! same logical bytes as reading it through another — an `.hfq` against the
//! safetensors directory it was built from, most usefully. That is the property
//! `impl ModelSource for HfqFile` has to satisfy before anything downstream
//! (streamed calibration, per-arch decode removal) can rely on it.
//!
//! It exercises the decode path by construction: `hipfire-quantize --format
//! bf16` applies the `huff` BF16 codec by DEFAULT, so a bf16 `.hfq`'s stored
//! bytes are not its logical bytes, and `tensor()` has to expand them to match
//! the directory.
//!
//! Usage:
//!   cargo run --release --example modelsource_parity -- <a> <b>
//! where each of <a>/<b> is an `.hfq` file or a safetensors directory.

use hipfire_model::ModelSource;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::safetensors_source::SafetensorsSource;
use std::path::Path;

fn open(path: &Path) -> Box<dyn ModelSource> {
    if path.is_dir() {
        Box::new(SafetensorsSource::open(path).expect("open safetensors dir"))
    } else {
        Box::new(HfqFile::open(path).expect("open hfq"))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: modelsource_parity <a.hfq|dir> <b.hfq|dir>");
        std::process::exit(2);
    }
    let a = open(Path::new(&args[0]));
    let b = open(Path::new(&args[1]));

    let mut a_names: Vec<String> = a.tensor_names().iter().map(|s| s.to_string()).collect();
    let mut b_names: Vec<String> = b.tensor_names().iter().map(|s| s.to_string()).collect();
    a_names.sort();
    b_names.sort();

    println!("a: {} tensors  b: {} tensors", a_names.len(), b_names.len());

    let only_a: Vec<&String> = a_names.iter().filter(|n| !b_names.contains(n)).collect();
    let only_b: Vec<&String> = b_names.iter().filter(|n| !a_names.contains(n)).collect();
    if !only_a.is_empty() || !only_b.is_empty() {
        println!(
            "name sets differ: {} only in a, {} only in b",
            only_a.len(),
            only_b.len()
        );
        for n in only_a.iter().take(5) {
            println!("  only in a: {n}");
        }
        for n in only_b.iter().take(5) {
            println!("  only in b: {n}");
        }
    }

    let mut compared = 0usize;
    let mut mismatched = 0usize;
    let mut missing = 0usize;
    let mut storage_differs = 0usize;
    let mut owned_a = 0usize;
    let mut owned_b = 0usize;
    let mut bytes = 0u64;

    for name in a_names.iter().filter(|n| b_names.contains(n)) {
        let Some((ia, da)) = a.tensor(name) else {
            println!("MISSING in a via tensor(): {name}");
            missing += 1;
            continue;
        };
        let Some((ib, db)) = b.tensor(name) else {
            println!("MISSING in b via tensor(): {name}");
            missing += 1;
            continue;
        };
        if matches!(da, std::borrow::Cow::Owned(_)) {
            owned_a += 1;
        }
        if matches!(db, std::borrow::Cow::Owned(_)) {
            owned_b += 1;
        }
        if ia.shape != ib.shape {
            println!("SHAPE  {name}: {:?} vs {:?}", ia.shape, ib.shape);
            mismatched += 1;
            continue;
        }
        // A tensor stored in a GPU-decodable packed coding is a STORAGE
        // difference, not a defect. `tensor()` returns the payload for the
        // coding the artifact DECLARES, and a Bf16Lut3 (qt=49) head is
        // deliberately kept packed because `gemv_bf16l3` decodes it natively —
        // expanding it here would defeat the reason it is stored that way.
        // Note 0xFF is the safetensors sentinel ("use dtype"), NOT a coding, so
        // it must not by itself count as a difference.
        const PACKED_BF16: [u8; 2] = [49, 50]; // Bf16Lut3, Bf16Huff
        let packed_side =
            PACKED_BF16.contains(&ia.quant_type) || PACKED_BF16.contains(&ib.quant_type);
        if packed_side && da.len() != db.len() {
            println!(
                "STORAGE {name}: qt {} ({} B) vs qt {} ({} B) — packed coding, kept resident",
                ia.quant_type,
                da.len(),
                ib.quant_type,
                db.len()
            );
            storage_differs += 1;
            continue;
        }
        if da.len() != db.len() {
            println!("LEN    {name}: {} vs {} bytes", da.len(), db.len());
            mismatched += 1;
            continue;
        }
        if *da != *db {
            let first = da
                .iter()
                .zip(db.iter())
                .position(|(x, y)| x != y)
                .unwrap_or(0);
            println!(
                "BYTES  {name}: differ at offset {first} ({} bytes)",
                da.len()
            );
            mismatched += 1;
            continue;
        }
        compared += 1;
        bytes += da.len() as u64;
    }

    println!(
        "identical: {compared}  mismatched: {mismatched}  missing: {missing}  \
         storage-differs: {storage_differs}  bytes compared: {bytes}"
    );
    println!("decoded (Cow::Owned) — a: {owned_a}  b: {owned_b}");
    if mismatched == 0 && missing == 0 && compared > 0 {
        println!("PARITY OK");
    } else {
        println!("PARITY FAILED");
        std::process::exit(1);
    }
}
