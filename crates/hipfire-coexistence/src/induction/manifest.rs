// SPDX-License-Identifier: Apache-2.0
//! The two-pass provenance manifest, its fingerprint set, and fingerprint
//! gating. Ported from `two_pass_quantize.update_manifest`,
//! `_merge_calibration_execution`, and `accumulate_attempt_timing`.
//!
//! The `fingerprints` block is the gate: `induct_model.target_stage_complete`
//! skips regeneration when the recipe fingerprint and this set match, so the set
//! must be assembled from the same inspection fields the Python reads.

use super::{atomic_json, dig, recipe::Recipe, utc_now};
use serde_json::{json, Map, Value};
use std::path::Path;

/// Optional inputs to one manifest update. Absent (`None`) values carry the
/// previous manifest's value forward, exactly as the Python does.
#[derive(Default)]
pub struct ManifestUpdate {
    pub calibration: Option<Value>,
    pub calibration_audit: Option<Value>,
    pub storage_preflight: Option<Value>,
    pub quantized: Option<Value>,
    pub calibration_execution: Option<Value>,
    pub phase_timings: Option<Value>,
    pub failure: Option<Value>,
}

fn read_previous(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Object(Map::new()))
}

/// Merge a fresh `calibration_execution` with the previous one, deduplicating
/// segments — the twin of `_merge_calibration_execution`. A change in mode /
/// segment size / release seconds discards the old record.
fn merge_calibration_execution(previous: Option<&Value>, current: &Value) -> Value {
    let Some(previous) = previous.and_then(|v| v.as_object()) else {
        return current.clone();
    };
    let current_obj = current.as_object().cloned().unwrap_or_default();
    let identity = ["mode", "process_segment_layers", "release_seconds"];
    if identity
        .iter()
        .any(|key| previous.get(*key) != current_obj.get(*key))
    {
        return current.clone();
    }
    let mut merged = previous.clone();
    for (key, value) in &current_obj {
        merged.insert(key.clone(), value.clone());
    }
    let mut segments: Vec<Value> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let empty = Vec::new();
    let prev_segments = previous.get("segments").and_then(|v| v.as_array()).unwrap_or(&empty);
    let cur_segments = current_obj.get("segments").and_then(|v| v.as_array()).unwrap_or(&empty);
    for segment in prev_segments.iter().chain(cur_segments) {
        let Some(seg) = segment.as_object() else { continue };
        let key = format!(
            "{}|{}|{}|{}",
            seg.get("started_after_layer").unwrap_or(&Value::Null),
            seg.get("pause_after_layer").unwrap_or(&Value::Null),
            seg.get("completed_layers").unwrap_or(&Value::Null),
            seg.get("artifact_complete").unwrap_or(&Value::Null),
        );
        if seen.insert(key) {
            segments.push(segment.clone());
        }
    }
    if !segments.is_empty() {
        merged.insert("segments".into(), Value::Array(segments));
    }
    Value::Object(merged)
}

/// Cumulative + last-attempt timing for one phase. Twin of
/// `accumulate_attempt_timing`: reads the current manifest's prior seconds and
/// adds `elapsed`, rounding to 6 decimals like Python's `round(x, 6)`.
pub fn accumulate_attempt_timing(manifest: &Value, phase: &str, elapsed_seconds: f64) -> Value {
    let prior = manifest
        .get("phase_timings")
        .and_then(|v| v.get(format!("{phase}_seconds")))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let elapsed = round6(elapsed_seconds);
    json!({
        format!("{phase}_seconds"): round6(prior + elapsed),
        format!("last_{phase}_attempt_seconds"): elapsed,
    })
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// Write the atomic two-pass manifest for `phase`, carrying forward prior fields
/// and reassembling the `fingerprints` gate. Returns the written manifest.
pub fn update_manifest(
    path: &Path,
    recipe: &Recipe,
    phase: &str,
    update: ManifestUpdate,
) -> std::io::Result<Value> {
    let previous = read_previous(path);
    let prev = previous.as_object();

    let mut manifest = Map::new();
    manifest.insert("schema".into(), json!(1));
    manifest.insert(
        "created_at".into(),
        prev.and_then(|p| p.get("created_at")).cloned().unwrap_or_else(|| json!(utc_now())),
    );
    manifest.insert("updated_at".into(), json!(utc_now()));
    manifest.insert("status".into(), json!(phase));
    for (key, value) in recipe.as_manifest_fields() {
        manifest.insert(key, value);
    }

    // calibration (+ its read_ledger → source_reads), or carry forward.
    if let Some(calibration) = &update.calibration {
        manifest.insert("calibration".into(), calibration.clone());
        if let Some(ledger) = dig(Some(calibration), &["metadata", "read_ledger"]) {
            manifest.insert("source_reads".into(), ledger.clone());
        }
    } else if let Some(prev_cal) = prev.and_then(|p| p.get("calibration")) {
        manifest.insert("calibration".into(), prev_cal.clone());
        if let Some(reads) = prev.and_then(|p| p.get("source_reads")) {
            manifest.insert("source_reads".into(), reads.clone());
        }
    }

    if let Some(audit) = &update.calibration_audit {
        manifest.insert("calibration_audit".into(), audit.clone());
    } else if update.calibration.is_none() {
        if let Some(prev_audit) = prev.and_then(|p| p.get("calibration_audit")) {
            manifest.insert("calibration_audit".into(), prev_audit.clone());
        }
    }

    if let Some(preflight) = &update.storage_preflight {
        manifest.insert("pass_two_storage_preflight".into(), preflight.clone());
    } else if let Some(prev_pf) = prev.and_then(|p| p.get("pass_two_storage_preflight")) {
        manifest.insert("pass_two_storage_preflight".into(), prev_pf.clone());
    }

    if let Some(quantized) = &update.quantized {
        manifest.insert("quantized".into(), quantized.clone());
    } else if let Some(prev_q) = prev.and_then(|p| p.get("quantized")) {
        manifest.insert("quantized".into(), prev_q.clone());
    }

    if let Some(execution) = &update.calibration_execution {
        manifest.insert(
            "calibration_execution".into(),
            merge_calibration_execution(prev.and_then(|p| p.get("calibration_execution")), execution),
        );
    } else if let Some(prev_exec) = prev.and_then(|p| p.get("calibration_execution")) {
        manifest.insert("calibration_execution".into(), prev_exec.clone());
    }

    if let Some(timings) = &update.phase_timings {
        let mut merged = prev
            .and_then(|p| p.get("phase_timings"))
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        if let Some(obj) = timings.as_object() {
            for (key, value) in obj {
                merged.insert(key.clone(), value.clone());
            }
        }
        manifest.insert("phase_timings".into(), Value::Object(merged));
    } else if let Some(prev_t) = prev.and_then(|p| p.get("phase_timings")) {
        manifest.insert("phase_timings".into(), prev_t.clone());
    }

    if let Some(failure) = &update.failure {
        manifest.insert("failure".into(), failure.clone());
    }

    // Fingerprint set (None values filtered out), from the just-set-or-inherited
    // calibration / quantized values.
    let calibration_value = update
        .calibration
        .clone()
        .or_else(|| manifest.get("calibration").cloned());
    let quantized_value = update
        .quantized
        .clone()
        .or_else(|| manifest.get("quantized").cloned());
    let cal = calibration_value.as_ref();
    let quant = quantized_value.as_ref();
    let candidates: [(&str, Option<Value>); 7] = [
        ("calibration_artifact", dig(cal, &["artifact_fingerprint"]).cloned()),
        ("calibration_engine_build", dig(cal, &["metadata", "engine_build"]).cloned()),
        ("calibration_run", dig(cal, &["metadata", "run_fingerprint"]).cloned()),
        ("source", dig(cal, &["metadata", "source_manifest", "fingerprint"]).cloned()),
        ("samples", dig(cal, &["metadata", "job", "samples", "fingerprint"]).cloned()),
        ("quantized_artifact", dig(quant, &["artifact_fingerprint"]).cloned()),
        ("quantized_payload", dig(quant, &["metadata", "quantization_hash", "value"]).cloned()),
    ];
    let mut fingerprints = Map::new();
    for (key, value) in candidates {
        if let Some(value) = value.filter(|v| !v.is_null()) {
            fingerprints.insert(key.into(), value);
        }
    }
    manifest.insert("fingerprints".into(), Value::Object(fingerprints));

    let value = Value::Object(manifest);
    atomic_json(path, &value)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn fingerprint_set_matches_python_golden() {
        // Synthetic inspection values fed to the same update; golden from
        // scripts/two_pass_quantize.update_manifest.
        let mut fields = BTreeMap::new();
        fields.insert("model".to_string(), json!("/m"));
        fields.insert("quant_format".to_string(), json!("oq4.25++"));
        let recipe = Recipe::for_test(fields, "sha256:deadbeef");
        let calibration = json!({
            "artifact_fingerprint": "fnv64:aaa",
            "metadata": {
                "artifact_kind": "calibration",
                "engine_build": "engine-XYZ",
                "run_fingerprint": "run-123",
                "read_ledger": {"missing_logical": [], "duplicate_logical": []},
                "source_manifest": {"fingerprint": "src-999"},
                "job": {"samples": {"fingerprint": "samp-777"}},
            },
        });
        let quantized = json!({
            "artifact_fingerprint": "fnv64:bbb",
            "metadata": {"quantization_hash": {"value": "qhash-555"}, "calibration": {"x": 1}},
        });
        let dir = std::env::temp_dir().join(format!("hipfire-manifest-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two-pass.json");
        let manifest = update_manifest(
            &path,
            &recipe,
            "complete",
            ManifestUpdate {
                calibration: Some(calibration),
                quantized: Some(quantized),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(manifest["status"], "complete");
        assert_eq!(manifest["recipe_fingerprint"], "sha256:deadbeef");
        assert_eq!(
            manifest["source_reads"],
            json!({"missing_logical": [], "duplicate_logical": []})
        );
        let fp = &manifest["fingerprints"];
        assert_eq!(fp["calibration_artifact"], "fnv64:aaa");
        assert_eq!(fp["calibration_engine_build"], "engine-XYZ");
        assert_eq!(fp["calibration_run"], "run-123");
        assert_eq!(fp["source"], "src-999");
        assert_eq!(fp["samples"], "samp-777");
        assert_eq!(fp["quantized_artifact"], "fnv64:bbb");
        assert_eq!(fp["quantized_payload"], "qhash-555");
        assert_eq!(fp.as_object().unwrap().len(), 7);
        std::fs::remove_dir_all(&dir).ok();
    }
}
