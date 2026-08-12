// SPDX-License-Identifier: Apache-2.0
//! The two-pass recipe and its content fingerprint.
//!
//! Byte-identical to `two_pass_quantize.recipe_manifest` (and the
//! `induct_model._target_recipe_fingerprint` twin): the recipe is canonically
//! encoded with sorted keys and compact separators, and the `recipe_fingerprint`
//! is `sha256:` over those bytes. The fingerprint gates resume — a mismatch
//! means the geometry silently changed — so it MUST match the Python exactly.

use super::python_resolve;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

/// Hex-encoded SHA-256 of `bytes` (lower-case), no prefix.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// `sha256:`-prefixed streaming file digest — the twin of
/// `two_pass_quantize._sha256_file`.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    Ok(format!("sha256:{hex}"))
}

/// Every field that enters the recipe. Paths are pre-`python_resolve`d here so
/// the encoded form is stable regardless of the caller's working directory.
#[derive(Clone)]
pub struct RecipeInputs {
    pub model: std::path::PathBuf,
    pub calib: std::path::PathBuf,
    pub output: std::path::PathBuf,
    pub quant_format: String,
    pub corpus: std::path::PathBuf,
    pub n_sequences: u64,
    pub ctx_len: u64,
    pub batch_size: u64,
    pub time_tile: u64,
    pub max_rows: u64,
    pub layer_prefetch_bytes: u64,
    pub kldref_topk: u64,
    pub min_expert_activations: u64,
    pub expert_capture_target: u64,
    pub expert_capture_tile_rows: u64,
    pub required_expert_fraction: f64,
    pub sampling_seed: u64,
    pub expert_coverage_policy: String,
    pub quant_args: Vec<String>,
}

/// The recipe object built from `inputs`, computed the same way for the
/// fingerprint and for embedding in the manifest. The corpus SHA-256 is read
/// from disk (the corpus must exist), matching Python.
/// Recipe fields that are recorded but NOT hashed.
///
/// Every one is a resolved absolute path, and a path says where a thing sits on
/// one machine rather than what it is. Hashing them made `recipe_fingerprint`
/// identify a filesystem layout: the same logical recipe fingerprinted
/// differently on two boxes because a corpus was reached through a symlink or
/// an HF cache root sat elsewhere. In a repo whose expensive artifacts are
/// shared over NFS between machines, that is the wrong identity — it defeats
/// resume across exactly the boundary the share exists to cross.
///
/// What identifies the inputs instead is content: `corpus_sha256` already
/// pins the corpus, and the corpus is git-tracked, so the fingerprint is now
/// reproducible anywhere the repo is.
///
/// KNOWN GAP, deliberate: with `model` unhashed, two runs that differ ONLY by
/// which model they point at now share a fingerprint. Closing that wants a
/// cheap content identity for a model directory (an index digest, not a
/// 244 GB payload hash) — worth doing, but it is a separate change, and the
/// alternative of hashing the path is a worse answer than no answer.
///
/// Must stay in lockstep with `_UNHASHED_PATH_FIELDS` in
/// `scripts/two_pass_quantize.py`; the two fingerprints must agree byte for
/// byte or resume gating diverges silently.
const UNHASHED_PATH_FIELDS: [&str; 5] = [
    "cask_artifact",
    "calibration_artifact",
    "corpus",
    "model",
    "quantized_artifact",
];

pub struct Recipe {
    /// Sorted key→value map. Serialized directly (a `BTreeMap`, so key order is
    /// stable irrespective of serde_json's crate-wide `preserve_order`).
    fields: BTreeMap<String, Value>,
    pub recipe_fingerprint: String,
}

impl Recipe {
    pub fn build(inputs: &RecipeInputs) -> std::io::Result<Self> {
        let corpus_resolved = python_resolve(&inputs.corpus);
        let corpus_sha256 = sha256_file(&corpus_resolved)?;
        let mut fields: BTreeMap<String, Value> = BTreeMap::new();
        fields.insert(
            "model".into(),
            json!(python_resolve(&inputs.model).to_string_lossy()),
        );
        fields.insert(
            "calibration_artifact".into(),
            json!(python_resolve(&inputs.calib).to_string_lossy()),
        );
        fields.insert(
            "quantized_artifact".into(),
            json!(python_resolve(&inputs.output).to_string_lossy()),
        );
        fields.insert("quant_format".into(), json!(inputs.quant_format));
        fields.insert("corpus".into(), json!(corpus_resolved.to_string_lossy()));
        fields.insert("corpus_sha256".into(), json!(corpus_sha256));
        fields.insert("sequences".into(), json!(inputs.n_sequences));
        fields.insert("context".into(), json!(inputs.ctx_len));
        fields.insert("sequence_batch".into(), json!(inputs.batch_size));
        fields.insert("time_tile".into(), json!(inputs.time_tile));
        fields.insert("max_rows".into(), json!(inputs.max_rows));
        fields.insert(
            "layer_prefetch_bytes".into(),
            json!(inputs.layer_prefetch_bytes),
        );
        fields.insert("kldref_topk".into(), json!(inputs.kldref_topk));
        fields.insert(
            "min_expert_activations".into(),
            json!(inputs.min_expert_activations),
        );
        fields.insert(
            "expert_capture_target".into(),
            json!(inputs.expert_capture_target),
        );
        fields.insert(
            "expert_capture_tile_rows".into(),
            json!(inputs.expert_capture_tile_rows),
        );
        fields.insert(
            "required_expert_fraction".into(),
            json!(inputs.required_expert_fraction),
        );
        fields.insert("sampling_seed".into(), json!(inputs.sampling_seed));
        // Python normalizes the policy dashes → underscores before hashing.
        fields.insert(
            "expert_coverage_policy".into(),
            json!(inputs.expert_coverage_policy.replace('-', "_")),
        );
        fields.insert("quant_args".into(), json!(inputs.quant_args));

        // Canonical encoding: sorted keys (BTreeMap), compact separators — the
        // serde_json default compact form is `{"a":1,...}` with no spaces, the
        // twin of Python `separators=(",", ":")`.
        //
        // Path fields are EXCLUDED from the hash (see `UNHASHED_PATH_FIELDS`).
        // They remain in `fields`, so the manifest still records where
        // everything lived; they just stop deciding the fingerprint.
        let hashed: BTreeMap<&str, &Value> = fields
            .iter()
            .filter(|(key, _)| !UNHASHED_PATH_FIELDS.contains(&key.as_str()))
            .map(|(key, value)| (key.as_str(), value))
            .collect();
        let encoded = serde_json::to_vec(&hashed).expect("recipe encodes");
        let recipe_fingerprint = format!("sha256:{}", sha256_hex(&encoded));
        Ok(Self {
            fields,
            recipe_fingerprint,
        })
    }

    /// Hermetic constructor for tests that only need a recipe to spread into a
    /// manifest (no corpus on disk).
    #[cfg(test)]
    pub(crate) fn for_test(fields: BTreeMap<String, Value>, fingerprint: &str) -> Self {
        Self {
            fields,
            recipe_fingerprint: fingerprint.to_string(),
        }
    }

    /// The recipe object as it is spread into the manifest (`**recipe` in
    /// Python): all fields plus `recipe_fingerprint`.
    pub fn as_manifest_fields(&self) -> serde_json::Map<String, Value> {
        let mut map = serde_json::Map::new();
        for (key, value) in &self.fields {
            map.insert(key.clone(), value.clone());
        }
        map.insert("recipe_fingerprint".into(), json!(self.recipe_fingerprint));
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Golden values captured from `scripts/two_pass_quantize.py` on this repo
    // (nix1). If the encoding, key order, float format, or path resolution
    // diverges from CPython, this fails — which is exactly the silent
    // resume-gating divergence the M6 verification gate forbids.
    #[test]
    fn recipe_fingerprint_matches_python_golden() {
        let corpus = PathBuf::from("/home/sadara/hipfire/benchmarks/calib/calib-5m.txt");
        if !corpus.is_file() {
            eprintln!("skipping: golden corpus absent");
            return;
        }
        let inputs = RecipeInputs {
            model: PathBuf::from(
                "/srv/huggingface/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17",
            ),
            calib: PathBuf::from("/home/sadara/.hipfire/calib/Qwen3.5-0.8B.calib.hfq"),
            output: PathBuf::from("/home/sadara/.hipfire/models/Qwen3.5-0.8B.oq4.25++.hfq"),
            quant_format: "oq4.25++".into(),
            corpus,
            n_sequences: 4,
            ctx_len: 64,
            batch_size: 4,
            time_tile: 16,
            max_rows: 2048,
            layer_prefetch_bytes: 16 * 1024 * 1024 * 1024,
            kldref_topk: 64,
            min_expert_activations: 2048,
            expert_capture_target: 4096,
            expert_capture_tile_rows: 256,
            required_expert_fraction: 1.0,
            sampling_seed: 1,
            expert_coverage_policy: "preserve-undercovered".into(),
            quant_args: vec!["--awq".into(), "--ldlq".into()],
        };
        let recipe = Recipe::build(&inputs).unwrap();
        assert_eq!(
            recipe.recipe_fingerprint,
            "sha256:70dbbe3e2cfd58967fb0449c3f4c7ce51504670a7a6d4c723d91b6e13c5d1cf3"
        );

        // The paths above are this machine's. Point every one of them somewhere
        // else and the fingerprint must not move — that is the whole reason the
        // path fields left the hash. Previously this assertion was impossible:
        // the golden encoded nix1's filesystem, so the test failed on any box
        // where a path resolved differently (here, a symlinked corpus).
        let elsewhere = RecipeInputs {
            model: PathBuf::from("/some/other/box/models/Qwen3.5-0.8B"),
            calib: PathBuf::from("/elsewhere/Qwen3.5-0.8B.calib.hfq"),
            output: PathBuf::from("/elsewhere/Qwen3.5-0.8B.oq4.25++.hfq"),
            // Same corpus CONTENT reached by its non-symlinked path.
            corpus: PathBuf::from("/home/sadara/.hipfire/src/benchmarks/calib/calib-5m.txt"),
            ..inputs
        };
        assert_eq!(
            Recipe::build(&elsewhere).unwrap().recipe_fingerprint,
            recipe.recipe_fingerprint,
            "recipe_fingerprint must identify the recipe, not the filesystem"
        );

        // The manifest still records where everything was — dropping the paths
        // from the HASH must not drop them from the provenance record.
        let manifest = recipe.as_manifest_fields();
        for key in UNHASHED_PATH_FIELDS {
            if key == "cask_artifact" {
                continue; // only present when a CASK output was requested
            }
            assert!(
                manifest.contains_key(key),
                "{key} must stay in the manifest even though it is unhashed"
            );
        }
    }

    #[test]
    fn resolve_normalizes_nonexistent_tail_like_python() {
        // /tmp exists, the tail does not — Python resolve() keeps the tail.
        let got = python_resolve(std::path::Path::new("/tmp/hipfire-nonexist-xyz/a/../b.hfq"));
        assert_eq!(got, PathBuf::from("/tmp/hipfire-nonexist-xyz/b.hfq"));
    }
}
