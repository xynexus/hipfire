// SPDX-License-Identifier: Apache-2.0
//! Index-only HFQ inspection for offline induction manifests.

use hipfire_runtime::hfq::HfqFile;
use serde_json::{json, Value};
use std::error::Error;
use std::path::{Path, PathBuf};

/// Return provenance and an index fingerprint without reading tensor payloads.
///
/// The fingerprint is `HfqFile::index_fingerprint` — the SAME number
/// `hipfire inspect` reports. It used to be a second, independent hash over a
/// JSON serialization of the same facts, which meant two algorithms answering
/// one question and disagreeing by value. One of them had to go; the runtime's
/// is the one every other caller already used.
///
/// The embedded `quantization_hash`, when present, remains the payload-integrity
/// identity; this fingerprint identifies metadata and tensor layout for cheap
/// resume checks.
pub fn inspect_artifact(path: &Path) -> Result<Value, Box<dyn Error>> {
    let hfq = HfqFile::open_index_only(path)?;
    let metadata: Value = serde_json::from_str(&hfq.metadata_json)?;
    Ok(json!({
        "artifact": path,
        "bytes": std::fs::metadata(path)?.len(),
        "version": hfq.version,
        "arch_id": hfq.arch_id,
        "tensor_count": hfq.tensors().len(),
        "artifact_fingerprint": hfq.index_fingerprint(),
        "metadata": metadata,
    }))
}

pub fn run_inspect_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let input = args
        .windows(2)
        .find(|pair| pair[0] == "--input")
        .map(|pair| PathBuf::from(&pair[1]))
        .ok_or("artifact inspect requires --input <artifact.hfq>")?;
    if args.len() != 2 {
        return Err("artifact inspect accepts only --input <artifact.hfq>".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&inspect_artifact(&input)?)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::hfq::{
        write_hfqm_package_from_files, HfqPackageWriteEntry, HFQM_ARCH_NON_WEIGHT_PACKAGE,
    };

    #[test]
    fn inspection_is_index_only_and_exposes_embedded_provenance() {
        let root = std::env::temp_dir().join(format!(
            "hipfire-artifact-inspect-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&root).unwrap();
        let payload = root.join("payload.bin");
        std::fs::write(&payload, [1u8, 2, 3, 4]).unwrap();
        let artifact = root.join("fixture.calib.hfq");
        let entries = vec![HfqPackageWriteEntry {
            name: "calib/layer.0".into(),
            quant_type: 2,
            shape: vec![1],
            group_size: 0,
            source_path: payload,
            data_size: 4,
        }];
        write_hfqm_package_from_files(
            &artifact,
            HFQM_ARCH_NON_WEIGHT_PACKAGE,
            r#"{"artifact_kind":"calibration","run_fingerprint":"run-a","read_ledger":{"missing_logical":[]}}"#,
            &entries,
        )
        .unwrap();

        let inspected = inspect_artifact(&artifact).unwrap();
        assert_eq!(inspected["tensor_count"], 1);
        assert_eq!(inspected["metadata"]["run_fingerprint"], "run-a");
        assert!(inspected["artifact_fingerprint"]
            .as_str()
            .unwrap()
            .starts_with("fnv64:"));

        std::fs::remove_dir_all(root).unwrap();
    }
}
