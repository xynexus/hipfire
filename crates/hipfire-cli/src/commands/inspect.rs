//! `hipfire inspect <artefact>` — detail the contents of a `.hfq` (HFQM)
//! container without touching tensor payload.
//!
//! Read-only companion to `hipfire model` (compose/decompose). It opens a
//! container with [`HfqFile::open_index_only`] — pread of the 32-byte header
//! plus the metadata/index region only, no tensor-payload mmap — so even a
//! 100GiB+ model is inspected cheaply. Quant-type bytes are decoded through the
//! canonical [`QuantType`] byte-contract; arch ids through the arch registry.

use std::collections::BTreeMap;
use std::path::PathBuf;

use clap::Args;
use hipfire_arch_api::ArchId;
use hipfire_config::LoadedConfig;
use hipfire_hfq_tooling::{
    ComposeManifest, HFQM_COMPOSE_FORMAT, HFQM_COMPOSE_FORMAT_V1, HFQM_COMPOSE_KEY,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::quant::QuantType;
use serde_json::{json, Value};

use crate::model::find_model;

#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Container to inspect: a `.hfq` file path or a local model alias.
    target: String,
    /// List every tensor (name, quant type, shape, group size, size).
    #[arg(long)]
    tensors: bool,
    /// Emit a machine-readable JSON object (includes the full tensor array and
    /// the raw metadata verbatim); ignores `--tensors`.
    #[arg(long)]
    json: bool,
}

/// Config keys decoded into the human "model shape" section, in print order.
/// `(metadata_key, display_label)`. Only keys that are present are shown.
const SHAPE_FIELDS: &[(&str, &str)] = &[
    ("hidden_size", "hidden_size"),
    ("num_hidden_layers", "layers"),
    ("num_attention_heads", "attn heads"),
    ("num_key_value_heads", "kv heads"),
    ("head_dim", "head_dim"),
    ("intermediate_size", "intermediate"),
    ("vocab_size", "vocab"),
    ("num_experts", "experts"),
    ("num_experts_per_tok", "experts/tok"),
    ("rope_theta", "rope_theta"),
    ("max_position_embeddings", "max_pos_embed"),
];

pub fn run(args: InspectArgs, loaded: LoadedConfig) -> anyhow::Result<()> {
    let path = resolve(&args.target, &loaded)?;
    let hfq = HfqFile::open_index_only(&path)
        .map_err(|e| anyhow::anyhow!("failed to open {}: {e}", path.display()))?;

    let meta: Value = serde_json::from_str(&hfq.metadata_json).unwrap_or_else(|_| json!({}));
    let components = component_summary(&meta)?;
    let arch_name = hipfire_archs::registry()
        .get(ArchId(hfq.arch_id as u16))
        .map(|a| a.family);

    if args.json {
        print_json(&args.target, &path, &hfq, &meta, arch_name, &components);
    } else {
        print_human(&path, &hfq, &meta, arch_name, args.tensors, &components);
    }
    Ok(())
}

/// Resolve an argument to a concrete path: an existing file path wins, else it
/// is treated as a model alias resolved against the models directory. Mirrors
/// `commands::model::resolve`.
fn resolve(arg: &str, loaded: &LoadedConfig) -> anyhow::Result<PathBuf> {
    let direct = PathBuf::from(arg);
    if direct.exists() {
        return Ok(direct);
    }
    find_model(arg, &loaded.config).ok_or_else(|| anyhow::anyhow!("no such file or model: {arg}"))
}

/// Decode a quant-type byte to its canonical variant name, or a marker for an
/// unknown/reserved id.
fn quant_name(code: u8) -> String {
    match QuantType::from_code(code) {
        Some(qt) => format!("{qt:?}"),
        None => format!("qt{code}?"),
    }
}

/// Human-readable byte count (GB/MB/KB/B).
fn fmt_bytes(b: u64) -> String {
    let f = b as f64;
    if f >= 1e9 {
        format!("{:.2} GB", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.2} MB", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.2} KB", f / 1e3)
    } else {
        format!("{b} B")
    }
}

/// Look up `key` at the top level of the metadata, then inside a nested
/// `text_config`/`config` object (mirrors `config_from_hfq`).
fn meta_get<'a>(meta: &'a Value, key: &str) -> Option<&'a Value> {
    if let Some(v) = meta.get(key) {
        return Some(v);
    }
    for nest in ["text_config", "config"] {
        if let Some(v) = meta.get(nest).and_then(|c| c.get(key)) {
            return Some(v);
        }
    }
    None
}

/// Render a scalar metadata value: bare string for strings, JSON form otherwise.
fn val_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Feature sidecars present in the metadata, as short tags.
fn sidecar_tags(meta: &Value) -> Vec<&'static str> {
    let mut tags = Vec::new();
    if meta.get("mtp").is_some() {
        tags.push("mtp");
    }
    if meta.get("dflash").is_some() {
        tags.push("dflash");
    }
    if meta.get("vl").is_some() || meta.get("vision_config").is_some() {
        tags.push("vl");
    }
    if meta.get("krot").is_some() {
        tags.push("krot");
    }
    if meta.get("roughquant_sidecar").is_some() {
        tags.push("roughquant");
    }
    tags
}

/// Per-quant-type `(tensor_count, total_bytes)`, sorted by bytes descending.
fn quant_histogram(hfq: &HfqFile) -> Vec<(u8, usize, u64)> {
    let mut by_code: BTreeMap<u8, (usize, u64)> = BTreeMap::new();
    for t in hfq.tensors() {
        let e = by_code.entry(t.quant_type).or_default();
        e.0 += 1;
        e.1 += t.data_size as u64;
    }
    let mut rows: Vec<(u8, usize, u64)> = by_code
        .into_iter()
        .map(|(code, (n, bytes))| (code, n, bytes))
        .collect();
    rows.sort_by(|a, b| b.2.cmp(&a.2));
    rows
}

/// Module-table counts per [`HfqModuleKind`], keyed by its debug name.
fn module_kind_counts(hfq: &HfqFile) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for m in hfq.modules() {
        *counts.entry(format!("{:?}", m.kind)).or_default() += 1;
    }
    counts
}

/// Decode the compose manifest using metadata/index bytes only. This remains
/// cheap for model-sized bundles and makes corrupt manifests visible instead
/// of silently presenting the artifact as an ordinary monolith.
fn component_summary(meta: &Value) -> anyhow::Result<Vec<Value>> {
    let Some(value) = meta.get(HFQM_COMPOSE_KEY) else {
        return Ok(Vec::new());
    };
    let manifest: ComposeManifest = serde_json::from_value(value.clone())
        .map_err(|error| anyhow::anyhow!("invalid {HFQM_COMPOSE_KEY} manifest: {error}"))?;
    if manifest.format != HFQM_COMPOSE_FORMAT && manifest.format != HFQM_COMPOSE_FORMAT_V1 {
        anyhow::bail!("unsupported compose manifest format {:?}", manifest.format);
    }
    Ok(manifest
        .components
        .into_iter()
        .map(|component| {
            let encoding = match component.source_format.as_str() {
                "tria-v1" => "opaque-bytes",
                "hfqm" => "hfqm-mapped-entries",
                _ => "unknown",
            };
            json!({
                "role": component.tag,
                "filename": component.filename,
                "source_format": component.source_format,
                "original_arch_id": component.arch_id,
                "encoding": encoding,
                "byte_len": component.byte_len,
                "sha256": component.sha256,
                "entries": component.stored_entries.len(),
            })
        })
        .collect())
}

fn print_human(
    path: &std::path::Path,
    hfq: &HfqFile,
    meta: &Value,
    arch_name: Option<&str>,
    list_tensors: bool,
    components: &[Value],
) {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    println!("artefact: {name}");
    match arch_name {
        Some(a) => println!("arch:     {} ({})", hfq.arch_id, a),
        None => println!("arch:     {}", hfq.arch_id),
    }
    println!("format:   hfqm v{}", hfq.version);

    // Quant family / KV mode / sidecars.
    if let Some(v) = meta_get(meta, "quant_family") {
        println!("quant:    {}", val_str(v));
    }
    if let Some(v) = meta_get(meta, "kv_mode") {
        println!("kv_mode:  {}", val_str(v));
    }
    let sidecars = sidecar_tags(meta);
    if !sidecars.is_empty() {
        println!("sidecars: {}", sidecars.join(", "));
    }
    if !components.is_empty() {
        println!("\ncomponents:");
        for component in components {
            println!(
                "  {}: {} ({}, arch {}, {}, {}, sha256 {})",
                component["role"].as_str().unwrap_or("?"),
                component["filename"].as_str().unwrap_or("?"),
                component["source_format"].as_str().unwrap_or("?"),
                component["original_arch_id"],
                component["encoding"].as_str().unwrap_or("?"),
                fmt_bytes(component["byte_len"].as_u64().unwrap_or(0)),
                component["sha256"]
                    .as_str()
                    .unwrap_or("legacy-v1-unavailable"),
            );
        }
    }

    // Model shape.
    let shape: Vec<(&str, String)> = SHAPE_FIELDS
        .iter()
        .filter_map(|(key, label)| meta_get(meta, key).map(|v| (*label, val_str(v))))
        .collect();
    if !shape.is_empty() {
        println!("\nmodel shape:");
        let w = shape.iter().map(|(l, _)| l.len()).max().unwrap_or(0);
        for (label, value) in &shape {
            println!("  {label:<w$}  {value}", w = w);
        }
    }

    // Module table (MoE grouping).
    let mods = hfq.modules();
    if !mods.is_empty() {
        let counts = module_kind_counts(hfq);
        let summary = counts
            .iter()
            .map(|(k, n)| format!("{n} {k}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("\nmodules: {} ({summary})", mods.len());
    }

    // Quant histogram + totals.
    let hist = quant_histogram(hfq);
    let total_tensors: usize = hist.iter().map(|(_, n, _)| n).sum();
    let total_bytes: u64 = hist.iter().map(|(_, _, b)| b).sum();
    println!("\nquant histogram:");
    let name_w = hist
        .iter()
        .map(|(c, _, _)| quant_name(*c).len())
        .max()
        .unwrap_or(0);
    for (code, n, bytes) in &hist {
        println!(
            "  {:<name_w$}  {:>6} tensors  {:>10}",
            quant_name(*code),
            n,
            fmt_bytes(*bytes),
            name_w = name_w,
        );
    }
    println!("total: {total_tensors} tensors  {}", fmt_bytes(total_bytes));

    if list_tensors {
        println!("\ntensors:");
        for t in hfq.tensors() {
            println!(
                "  {:60} {:<14} shape={:?} g={} {}",
                t.name,
                quant_name(t.quant_type),
                t.shape,
                t.group_size,
                fmt_bytes(t.data_size as u64),
            );
        }
    } else {
        println!("\n(--tensors for full per-tensor list, --json for machine-readable output)");
    }
}

fn print_json(
    target: &str,
    path: &std::path::Path,
    hfq: &HfqFile,
    meta: &Value,
    arch_name: Option<&str>,
    components: &[Value],
) {
    let shape: serde_json::Map<String, Value> = SHAPE_FIELDS
        .iter()
        .filter_map(|(key, _)| meta_get(meta, key).map(|v| (key.to_string(), v.clone())))
        .collect();

    let hist = quant_histogram(hfq);
    let total_tensors: usize = hist.iter().map(|(_, n, _)| n).sum();
    let total_bytes: u64 = hist.iter().map(|(_, _, b)| b).sum();
    let histogram: Vec<Value> = hist
        .iter()
        .map(|(code, n, bytes)| {
            json!({
                "quant_type": quant_name(*code),
                "code": code,
                "tensors": n,
                "bytes": bytes,
            })
        })
        .collect();

    let tensors: Vec<Value> = hfq
        .tensors()
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "quant_code": t.quant_type,
                "quant_type": quant_name(t.quant_type),
                "shape": t.shape,
                "group_size": t.group_size,
                "data_size": t.data_size,
            })
        })
        .collect();

    let modules = if hfq.modules().is_empty() {
        Value::Null
    } else {
        json!({
            "count": hfq.modules().len(),
            "by_kind": module_kind_counts(hfq),
        })
    };

    let out = json!({
        "target": target,
        "path": path.display().to_string(),
        "arch_id": hfq.arch_id,
        "arch_name": arch_name,
        "hfqm_version": hfq.version,
        "quant_family": meta_get(meta, "quant_family"),
        "kv_mode": meta_get(meta, "kv_mode"),
        "sidecars": sidecar_tags(meta),
        "components": components,
        "shape": shape,
        "quant_histogram": histogram,
        "totals": { "tensors": total_tensors, "bytes": total_bytes },
        "modules": modules,
        "tensors": tensors,
        "metadata": meta,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_component_summary_is_index_only_and_role_aware() {
        let metadata = json!({
            HFQM_COMPOSE_KEY: {
                "format": HFQM_COMPOSE_FORMAT,
                "components": [{
                    "tag": "dflash",
                    "filename": "Model.dflash.oq4+.hfq",
                    "arch_id": 20,
                    "tensors": ["fc.weight"],
                    "metadata_json": "{}",
                    "source_format": "hfqm",
                    "byte_len": 1024,
                    "sha256": "abc",
                    "stored_entries": [{
                        "stored_name": "__hipfire_component/dflash/1/fc.weight",
                        "original_name": "fc.weight",
                        "original_offset": 4096
                    }]
                }]
            }
        });
        let components = component_summary(&metadata).unwrap();
        assert_eq!(components[0]["role"], "dflash");
        assert_eq!(components[0]["original_arch_id"], 20);
        assert_eq!(components[0]["encoding"], "hfqm-mapped-entries");
        assert_eq!(components[0]["byte_len"], 1024);
        assert_eq!(components[0]["sha256"], "abc");
    }
}
