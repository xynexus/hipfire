// SPDX-License-Identifier: Apache-2.0
//! Offline model-induction orchestration, ported from the load-bearing parts of
//! `scripts/induct_model.py` + `scripts/two_pass_quantize.py` (M6 of
//! `docs/plans/2026-07-25-daemon-merge-training-induction-scheduler.md`).
//!
//! What lives here (the load-bearing Python the plan names):
//! - [`recipe`]: the two-pass recipe JSON + its content `recipe_fingerprint`
//!   (`sha256:` over the canonical-encoded recipe). Byte-identical to
//!   `two_pass_quantize.recipe_manifest`.
//! - [`preflight`]: `pass_two_storage_preflight`, the quant-format byte math,
//!   reusing `hipfire-quant-format` block geometry rather than re-deriving it.
//! - [`manifest`]: `update_manifest` + the fingerprint set it assembles and the
//!   fingerprint-gating that skips a completed stage.
//! - [`two_pass`]: the pass-1 (in-process layer-stream engine) → pass-2
//!   (quantizer) orchestration that `two_pass_quantize.main` performs. The
//!   process-quantum `run_calibration_pass` respawn scheduler is REPLACED by the
//!   M6 in-process engine (`calibrate::run_from_command`) / daemon calibrate op.
//! - [`orchestrate`]: the `induct_model.main` stage driver (dflash / target /
//!   triattn), source preflight, artifact layout, and stage-completion gating.
//!
//! Deliberately left as thin subprocess shells over `hipfire-coexistence
//! artifact ...` and the converter/triattn binaries, exactly as the Python does:
//! index inspection, calibration audit, the DFlash converter, and the
//! TriAttention validator. The inference/quant byte-rewriting stays in their own
//! binaries per AGENTS.md.

pub mod manifest;
pub mod orchestrate;
pub mod preflight;
pub mod recipe;
pub mod two_pass;

use chrono::SecondsFormat;
use serde_json::Value;
use std::path::{Component, Path, PathBuf};

/// UTC ISO-8601 timestamp, matching Python's
/// `datetime.now(timezone.utc).isoformat()` closely enough for the manifest's
/// `created_at`/`updated_at` (which are excluded from every fingerprint and the
/// equivalence gate compares "modulo timestamps").
pub fn utc_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(SecondsFormat::Micros, true)
}

/// Reimplements Python `pathlib.Path.resolve()` (strict=False / `os.path.realpath`
/// semantics): make absolute, canonicalize the longest existing ancestor (which
/// resolves every symlink in the real prefix — a symlink is always an existing
/// file, so it can never appear in the non-existent tail), then append the
/// remaining components lexically, folding `.`/`..`.
///
/// `std::fs::canonicalize` cannot be used directly because the recipe resolves
/// the not-yet-created calibration/quantized output paths, and canonicalize errors
/// on a missing path. The resolved strings feed the `recipe_fingerprint`, so this
/// must match CPython byte-for-byte.
pub fn python_resolve(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(path)
    };

    // Peel components off the end until the remaining prefix exists on disk.
    let mut prefix = abs.clone();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    while !prefix.exists() {
        match prefix.file_name() {
            Some(name) => {
                tail.push(name.to_os_string());
                if !prefix.pop() {
                    break;
                }
            }
            None => break,
        }
    }
    let mut resolved = std::fs::canonicalize(&prefix).unwrap_or(prefix);
    for name in tail.iter().rev() {
        match name.to_str() {
            Some(".") => {}
            Some("..") => {
                resolved.pop();
            }
            _ => resolved.push(name),
        }
    }
    // Fold any `.`/`..` that lived inside the (non-canonicalized) absolute prefix
    // when canonicalize was unavailable. For real paths the canonicalize above
    // already did this; this keeps the fallback path honest.
    lexical_normalize(&resolved)
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Atomic pretty JSON write with sorted keys, matching Python's
/// `json.dumps(value, indent=2, sort_keys=True) + "\n"` then a temp-file rename.
pub fn atomic_json(path: &Path, value: &Value) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let sorted = sort_value(value);
    let mut text = serde_json::to_string_pretty(&sorted).unwrap_or_default();
    text.push('\n');
    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension().and_then(|e| e.to_str()).unwrap_or("json")
    ));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Recursively sort object keys so `to_string_pretty` mirrors Python
/// `sort_keys=True`. serde_json runs with `preserve_order` (insertion order)
/// crate-wide, so this is required for a stable, diff-friendly manifest file.
pub fn sort_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut btree = std::collections::BTreeMap::new();
            for (k, v) in map {
                btree.insert(k.clone(), sort_value(v));
            }
            Value::Object(btree.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_value).collect()),
        other => other.clone(),
    }
}

/// `value[k1][k2]...` walk that returns `None` at the first non-object or missing
/// key — the Rust twin of `two_pass_quantize._get`.
pub fn dig<'a>(mut value: Option<&'a Value>, keys: &[&str]) -> Option<&'a Value> {
    for key in keys {
        value = value?.as_object()?.get(*key);
    }
    value
}

/// The first four magic bytes of a file, or `None` if it is shorter than 32 bytes
/// or unreadable — the check `induct_model.artifact_is_valid` performs.
pub fn artifact_is_valid(path: &Path, magic: &[u8]) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() || meta.len() < 32 {
        return false;
    }
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    use std::io::Read;
    let mut head = vec![0u8; magic.len()];
    file.read_exact(&mut head).is_ok() && head == magic
}
