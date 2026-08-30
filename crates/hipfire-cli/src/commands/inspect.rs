//! `hipfire inspect <artefact>` — detail the contents of a `.hfq` (HFQM)
//! container without touching tensor payload.
//!
//! One command for every container kind: a diffusion artefact is autodetected
//! and additionally reports its pipeline summary, which used to require a
//! separate `hipfire diffusion inspect`. Detection routes through
//! `hipfire_diffusion::is_diffusion_hfq`, so `inspect` cannot disagree with the
//! diffusion runtime about what a diffusion container is.
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
use hipfire_diffusion::{inspect_hfq_with_runtime_support, DiffusionHfqInspection};
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
    let hfq = HfqFile::open_index_only(&path).map_err(|e| {
        // An .hfa is an HFAR archive of a HuggingFace directory, not an HFQM
        // container, so the raw "not an HFQM container at offset 0" leaks an
        // internal detail and names no way forward.
        if is_hfar_archive(&path) {
            anyhow::anyhow!(
                "{} is an HFAR archive, not an .hfq container. It holds a HuggingFace \
                 source directory; restore it first with `hipfire-coexistence repack \
                 --input {} --output <hf_dir>`.",
                path.display(),
                path.display(),
            )
        } else {
            anyhow::anyhow!("failed to open {}: {e}", path.display())
        }
    })?;

    let meta: Value = serde_json::from_str(&hfq.metadata_json).unwrap_or_else(|_| json!({}));
    let components = component_summary(&meta)?;
    // A calibration artefact carries arch 0 in its HFQM header and the real one
    // in `source_arch_id`; trusting the header prints "arch: 0 (llama)" for a
    // qwen35 calib, which is not merely uninformative but wrong.
    let arch_id = calib_source_arch_id(&meta).unwrap_or(hfq.arch_id);
    // The legacy generic-diffusion marker (0x3046_4944) does not fit the u16 the
    // registry keys on, so a bare `as u16` truncates it into an unrelated arch.
    let arch_name = u16::try_from(arch_id)
        .ok()
        .and_then(|id| hipfire_archs::registry().get(ArchId(id)))
        .map(|a| a.family)
        // The legacy marker is not a registry family, so it would otherwise
        // print as a bare nine-digit number on the one path that now surfaces
        // it routinely: a pre-A2 diffusion container.
        .or_else(|| {
            (arch_id == hipfire_arch_api::ARCH_ID_DIFFUSION_LEGACY)
                .then_some("diffusion, legacy marker")
        });

    // Autodetected: a diffusion container gets the pipeline summary that used to
    // need a separate `hipfire diffusion inspect`. Detection is the canonical
    // `is_diffusion_hfq` (parsing metadata, else a registered diffusion arch id
    // in the header), so `inspect` and the diffusion runtime always agree.
    //
    // A container the header calls diffusion but whose metadata will not parse
    // keeps the generic view plus the reason, rather than failing the whole
    // inspect: a broken artefact is exactly when its tensor index is wanted.
    let diffusion = hipfire_diffusion::is_diffusion_hfq(&path)
        .then(|| inspect_hfq_with_runtime_support(&path).map_err(|e| e.to_string()));

    // On-disk size. Distinct from `totals.bytes`, which sums tensor payloads and
    // so excludes the header, metadata and index.
    let file_bytes = std::fs::metadata(&path).map(|m| m.len()).ok();

    if args.json {
        print_json(
            &args.target,
            &path,
            &hfq,
            &meta,
            arch_id,
            arch_name,
            &components,
            diffusion,
            file_bytes,
        );
    } else {
        print_human(
            &path,
            &hfq,
            &meta,
            arch_id,
            arch_name,
            args.tensors,
            &components,
            diffusion,
            file_bytes,
        );
    }
    Ok(())
}

/// The `pipeline:` block for a diffusion container. Mirrors what `hipfire
/// diffusion inspect` printed, minus the JSON envelope.
fn print_diffusion(inspection: &Result<DiffusionHfqInspection, String>) {
    println!();
    let inspection = match inspection {
        Ok(inspection) => inspection,
        Err(reason) => {
            println!("pipeline: diffusion container, but its metadata did not parse: {reason}");
            return;
        }
    };
    let summary = &inspection.summary;
    println!("pipeline: {}", summary.pipeline_class);
    for (label, value) in [
        ("title", summary.title.clone()),
        ("model", summary.model_name.clone()),
        ("weights", summary.weight_format.clone()),
        ("max batch", summary.max_batch.to_string()),
    ] {
        println!("  {label:9} {value}");
    }
    let support = &inspection.runtime_support;
    match (&support.runtime_kind, &support.reason) {
        (Some(kind), _) => println!("  {:9} yes ({})", "runtime", kind.as_str()),
        (None, Some(reason)) => println!("  {:9} no ({reason})", "runtime"),
        (None, None) => println!("  {:9} no", "runtime"),
    }
}

/// True when the file begins with an HFAR archive magic (`repack.rs`).
fn is_hfar_archive(path: &std::path::Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 8];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok()
        && (&magic == b"HFAR0002" || &magic == b"HFAR0001")
}

/// One captured projection, aggregated across layers.
struct CalibProjection {
    /// Projection path with the layer index and any expert index elided,
    /// e.g. `mlp.experts.*.down_proj`.
    name: String,
    layers: std::collections::BTreeSet<u32>,
    n_hessian: usize,
    n_imatrix: usize,
    /// K (input width) seen for this projection; a Hessian is K x K.
    k: Option<u32>,
    /// Token counts across every tensor of this projection, sorted. Collected
    /// per BASE name so a tensor contributes once, not once per artifact kind.
    tokens: Vec<u64>,
    counted: std::collections::BTreeSet<String>,
}

/// Split a calibration tensor name into (projection, layer, artifact kind).
/// `model.language_model.layers.7.mlp.experts.3.down_proj.hessian`
///   -> ("mlp.experts.*.down_proj", Some(7), "hessian")
fn split_calib_tensor(name: &str) -> (String, Option<u32>, &str) {
    let (base, kind) = match name.rsplit_once('.') {
        Some((b, k @ ("hessian" | "imatrix"))) => (b, k),
        _ => (name, ""),
    };
    // Everything after `layers.<N>.` is the projection; before it is the model
    // prefix, which varies by arch (`model.`, `model.language_model.`).
    let (layer, proj) = match base.split_once(".layers.") {
        Some((_, rest)) => match rest.split_once('.') {
            Some((idx, proj)) => (idx.parse::<u32>().ok(), proj),
            None => (None, rest),
        },
        None => (None, base),
    };
    // Collapse per-expert captures so a 512-expert MoE reports one row.
    let mut out = String::with_capacity(proj.len());
    let mut parts = proj.split('.').peekable();
    while let Some(p) = parts.next() {
        if p.parse::<u32>().is_ok() {
            out.push('*');
        } else {
            out.push_str(p);
        }
        if parts.peek().is_some() {
            out.push('.');
        }
    }
    (out, layer, kind)
}

/// Aggregate the tensor index of a calibration artefact into per-projection
/// coverage. This is what answers "is this calib complete" — the question that
/// otherwise needs an ad-hoc script over `--json`.
fn calib_projections(hfq: &HfqFile, meta: &Value) -> Vec<CalibProjection> {
    use std::collections::BTreeMap;
    let tokens_by_tensor = meta_get(meta, "per_tensor_tokens").and_then(|v| v.as_object());
    let mut by_proj: BTreeMap<String, CalibProjection> = BTreeMap::new();
    for t in hfq.tensors() {
        let (proj, layer, kind) = split_calib_tensor(&t.name);
        if kind.is_empty() {
            continue;
        }
        let e = by_proj
            .entry(proj.clone())
            .or_insert_with(|| CalibProjection {
                name: proj,
                layers: Default::default(),
                n_hessian: 0,
                n_imatrix: 0,
                k: None,
                tokens: Vec::new(),
                counted: Default::default(),
            });
        if let Some(l) = layer {
            e.layers.insert(l);
        }
        match kind {
            // A Hessian is [K, K]; the imatrix is [K], so take K from whichever
            // is present without assuming both are.
            "hessian" => {
                e.n_hessian += 1;
                e.k = e.k.or_else(|| t.shape.first().copied());
            }
            "imatrix" => {
                e.n_imatrix += 1;
                e.k = e.k.or_else(|| t.shape.first().copied());
            }
            _ => {}
        }
        // Token counts are keyed by base name and shared by both artifact
        // kinds. Collect from whichever arrives first so an imatrix-only
        // projection (routed MoE experts) still reports its coverage.
        let base = t.name.trim_end_matches(&format!(".{kind}"));
        if !e.counted.contains(base) {
            if let Some(n) = tokens_by_tensor
                .and_then(|m| m.get(base))
                .and_then(|v| v.as_u64())
            {
                e.tokens.push(n);
                e.counted.insert(base.to_string());
            }
        }
    }
    let mut out: Vec<CalibProjection> = by_proj.into_values().collect();
    for p in &mut out {
        p.tokens.sort_unstable();
    }
    out
}

fn print_calib_coverage(hfq: &HfqFile, meta: &Value) {
    let projs = calib_projections(hfq, meta);
    if projs.is_empty() {
        return;
    }
    let all_layers: std::collections::BTreeSet<u32> = projs
        .iter()
        .flat_map(|p| p.layers.iter().copied())
        .collect();
    println!(
        "\ncoverage: {} projections across {} layers",
        projs.len(),
        all_layers.len()
    );
    let w = projs
        .iter()
        .map(|p| p.name.len())
        .max()
        .unwrap_or(0)
        .max(10);
    println!(
        "  {:<w$}  {:>6}  {:>7}  {:>6}  {:>5}",
        "projection",
        "layers",
        "H / I",
        "K",
        "tokens",
        w = w
    );
    for p in &projs {
        let tok = match (p.tokens.first(), p.tokens.last()) {
            (Some(lo), Some(hi)) if lo == hi => format!("{lo}"),
            (Some(lo), Some(hi)) => format!("{lo}-{hi}"),
            _ => "-".to_string(),
        };
        // A non-layer capture (embeddings, a pooling head) has no layer index;
        // printing 0 would read as "zero layers covered".
        let layers = if p.layers.is_empty() {
            "-".to_string()
        } else {
            p.layers.len().to_string()
        };
        println!(
            "  {:<w$}  {:>6}  {:>3} /{:>3}  {:>6}  {:>5}",
            p.name,
            layers,
            p.n_hessian,
            p.n_imatrix,
            p.k.map(|k| k.to_string()).unwrap_or_else(|| "-".into()),
            tok,
            w = w,
        );
    }

    // Factual warnings only — no arch-specific rules about which projections a
    // layer "should" have, since hybrid stacks legitimately differ per layer.
    let mut warnings: Vec<String> = Vec::new();
    for p in &projs {
        if p.n_hessian > 0 && p.n_imatrix > 0 && p.n_hessian != p.n_imatrix {
            warnings.push(format!(
                "{}: {} hessians but {} imatrix tensors",
                p.name, p.n_hessian, p.n_imatrix
            ));
        }
        // The failure mode that cost a session: `--ldlq` does not fail on a
        // missing Hessian, it logs `ldlq: skip <t>` and RTN-quantizes. For
        // routed MoE experts this may be deliberate (a K x K Hessian per
        // expert is enormous), so state it rather than calling it an error.
        if p.n_hessian == 0 && p.n_imatrix > 0 {
            warnings.push(format!(
                "{}: imatrix only, no Hessian — `--ldlq` / `oq*++` will fall back to RTN here",
                p.name
            ));
        }
        if p.n_imatrix == 0 && p.n_hessian > 0 {
            warnings.push(format!("{}: Hessian only, no imatrix", p.name));
        }
    }
    // Layer profiles: a layer's captured projection set. A hybrid arch has a
    // few; a profile covering one or two layers is worth a look.
    let mut profile_of: std::collections::BTreeMap<u32, Vec<&str>> = Default::default();
    for p in &projs {
        for l in &p.layers {
            profile_of.entry(*l).or_default().push(&p.name);
        }
    }
    let mut profiles: std::collections::BTreeMap<String, Vec<u32>> = Default::default();
    for (layer, mut kinds) in profile_of {
        kinds.sort_unstable();
        profiles.entry(kinds.join(",")).or_default().push(layer);
    }
    if profiles.len() > 1 {
        println!("  {} layer profiles:", profiles.len());
        for (_, layers) in &profiles {
            println!("    {:>4} layers  e.g. layer {}", layers.len(), layers[0]);
        }
    }
    // No "rare profile" warning: a hybrid stack legitimately has one (lfm2.5
    // carries 2 dense FFN layers among 22 MoE ones), so it fires on healthy
    // artefacts. The profile listing above already shows the shape.
    let starved: Vec<&CalibProjection> = projs
        .iter()
        .filter(|p| {
            p.tokens
                .last()
                .zip(p.tokens.first())
                .is_some_and(|(hi, lo)| *hi > 0 && *lo * 10 < *hi)
        })
        .collect();
    for p in starved {
        warnings.push(format!(
            "{}: token coverage spans {}-{} — some rows saw <10% of the best-covered row",
            p.name,
            p.tokens.first().copied().unwrap_or(0),
            p.tokens.last().copied().unwrap_or(0),
        ));
    }
    if !warnings.is_empty() {
        println!("\nwarnings:");
        for w in warnings {
            println!("  ! {w}");
        }
    }
}

/// `source_arch_id` of a calibration artefact, whose HFQM header arch is 0.
fn calib_source_arch_id(meta: &Value) -> Option<u32> {
    if meta_get(meta, "artifact_kind").and_then(|v| v.as_str()) != Some("calibration") {
        return None;
    }
    meta_get(meta, "source_arch_id")?.as_u64().map(|v| v as u32)
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
/// Codes that are lossless BF16 recodings rather than value quantisations.
/// A tensor still carrying one of these at inspect time was left packed
/// (residency) rather than expanded at open.
fn is_lossless_recoding_code(code: u8) -> bool {
    QuantType::from_code(code).is_some_and(|qt| qt.is_lossless_recoding())
}

/// Calibration-only HFQM quant_type, deliberately outside the `QuantType`
/// byte-contract: exact F32 Hessian diagonal + BF16 lower strict triangle.
/// Kept in sync with `QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32`
/// (`hipfire_runtime::calibration`), which is private to that module.
const QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32: u8 = 130;

fn quant_name(code: u8) -> String {
    if code == QUANT_TYPE_HESSIAN_BF16_TRIL_DIAG_F32 {
        return "HessianBf16TrilDiagF32".to_string();
    }
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
    arch_id: u32,
    arch_name: Option<&str>,
    list_tensors: bool,
    components: &[Value],
    diffusion: Option<Result<DiffusionHfqInspection, String>>,
    file_bytes: Option<u64>,
) {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());
    println!("artefact: {name}");
    let calib = calib_source_arch_id(meta).is_some();
    if calib {
        println!("kind:     calibration (hessian + imatrix)");
    }
    match arch_name {
        Some(a) if calib => println!("arch:     {arch_id} ({a}, from source_arch_id)"),
        Some(a) => println!("arch:     {arch_id} ({a})"),
        None => println!("arch:     {arch_id}"),
    }
    println!("format:   hfqm v{}", hfq.version);
    // Comparable with a calib's `source fp`: that is this value, computed over
    // the artefact the calib was captured from.
    println!("fingerprint {}", hfq.index_fingerprint());
    if let Some(b) = file_bytes {
        println!("file      {b} bytes on disk");
    }
    if calib {
        for (key, label) in [
            ("n_hessian", "hessians"),
            ("n_imatrix", "imatrix"),
            ("n_calib_tokens", "tokens"),
            ("source_model", "source"),
            ("source_fingerprint", "source fp"),
            ("corpus", "corpus"),
        ] {
            if let Some(v) = meta_get(meta, key) {
                println!("{label:9} {}", val_str(v));
            }
        }
        if meta_get(meta, "source_fingerprint").is_none() {
            println!("source fp (not recorded — artefact predates source fingerprinting)");
        }
        print_calib_coverage(hfq, meta);
    }

    if let Some(diffusion) = &diffusion {
        print_diffusion(diffusion);
        println!();
    }

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

    // Lossless storage codecs.
    //
    // The histogram above is the LOGICAL view: `expand_bf16_index` rewrites a
    // recoded entry's dtype and length at open, so a BF16L3-compressed tensor
    // shows as plain `BF16` with its expanded size. Without this section the
    // only way to tell a compressed artifact from an uncompressed one is to
    // compare the file size against the total — which is exactly how it was
    // missed while working on the BF16L3 lm_head.
    let mut codecs: std::collections::BTreeMap<u8, (usize, u64, u64)> = Default::default();
    for (i, t) in hfq.tensors().iter().enumerate() {
        // Two ways a tensor is compressed on disk, and reporting only the first
        // hides the biggest one.
        //
        //  1. Expanded at open. `expand_bf16_index` rewrote the entry to its
        //     logical view and `stored_recoding` remembers the physical extent.
        //  2. Left PACKED. A LUT3 head is resident by default, so its entry is
        //     never rewritten — `stored_recoding` is None and `quant_type` is
        //     still the codec. Its `data_size` is already the packed length.
        //
        // Case 2 is exactly the 379.74 MB embedding on a stock artifact, so
        // omitting it reported "saved 55.73 KB" against a real 145 MB.
        let (stored_qt, packed_len, logical_len) = match hfq.stored_recoding(i) {
            Some((qt, packed)) => (qt, packed as u64, t.data_size as u64),
            None if is_lossless_recoding_code(t.quant_type) => {
                let n: u64 = t.shape.iter().map(|&d| d as u64).product();
                (t.quant_type, t.data_size as u64, n * 2)
            }
            None => continue,
        };
        let e = codecs.entry(stored_qt).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += packed_len;
        e.2 += logical_len;
    }
    if !codecs.is_empty() {
        println!("\nlossless storage (compressed on disk, expanded or decoded in-kernel):");
        let (mut tot_p, mut tot_l) = (0u64, 0u64);
        for (qt, (n, packed, logical)) in &codecs {
            println!(
                "  {:<12}  {:>6} tensors  {:>10} on disk  from {:>10}  {:.3}x",
                quant_name(*qt),
                n,
                fmt_bytes(*packed),
                fmt_bytes(*logical),
                *logical as f64 / (*packed).max(1) as f64,
            );
            tot_p += packed;
            tot_l += logical;
        }
        if codecs.len() > 1 {
            println!(
                "  {:<12}  {:>6} tensors  {:>10} on disk  from {:>10}  {:.3}x",
                "combined",
                codecs.values().map(|v| v.0).sum::<usize>(),
                fmt_bytes(tot_p),
                fmt_bytes(tot_l),
                tot_l as f64 / tot_p.max(1) as f64,
            );
        }
        println!(
            "  saved {} against the logical size",
            fmt_bytes(tot_l - tot_p)
        );
    }

    if list_tensors {
        println!("\ntensors:");
        for (i, t) in hfq.tensors().iter().enumerate() {
            // Annotate the stored codec: the dtype column is the logical view,
            // so without this a compressed tensor is indistinguishable here.
            let stored = match hfq.stored_recoding(i) {
                Some((qt, packed)) => format!(
                    "  [stored {} {}, {:.3}x]",
                    quant_name(qt),
                    fmt_bytes(packed as u64),
                    t.data_size as f64 / (packed as f64).max(1.0)
                ),
                // Left packed rather than expanded — the dtype column already
                // names the codec, so say so and give the ratio it is saving.
                None if is_lossless_recoding_code(t.quant_type) => {
                    let n: u64 = t.shape.iter().map(|&d| d as u64).product();
                    format!(
                        "  [packed in place, from {}, {:.3}x]",
                        fmt_bytes(n * 2),
                        (n * 2) as f64 / (t.data_size as f64).max(1.0)
                    )
                }
                None => String::new(),
            };
            println!(
                "  {:60} {:<14} shape={:?} g={} {}{stored}",
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
    arch_id: u32,
    arch_name: Option<&str>,
    components: &[Value],
    diffusion: Option<Result<DiffusionHfqInspection, String>>,
    file_bytes: Option<u64>,
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
        .enumerate()
        .map(|(i, t)| {
            // `quant_type` / `data_size` are the LOGICAL view. When the tensor
            // is stored under a lossless recoding these report what it expands
            // to, not what is on disk, so emit the stored form alongside rather
            // than leaving a consumer to infer it from file size.
            let (stored_type, stored_code, stored_size) = match hfq.stored_recoding(i) {
                Some((qt, packed)) => (
                    Value::from(quant_name(qt)),
                    Value::from(qt),
                    Value::from(packed),
                ),
                None => (Value::Null, Value::Null, Value::Null),
            };
            json!({
                "name": t.name,
                "quant_code": t.quant_type,
                "quant_type": quant_name(t.quant_type),
                "shape": t.shape,
                "group_size": t.group_size,
                "data_size": t.data_size,
                "stored_quant_type": stored_type,
                "stored_quant_code": stored_code,
                "stored_data_size": stored_size,
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

    // Byte-identical to what `hipfire diffusion inspect` emitted, so a consumer
    // of that command only has to reach one key deeper after the merge.
    let diffusion = match diffusion {
        None => Value::Null,
        Some(Ok(inspection)) => super::diffusion::inspection_json(inspection),
        Some(Err(reason)) => json!({ "error": reason }),
    };

    let out = json!({
        "target": target,
        "path": path.display().to_string(),
        // For a calibration artefact this is `source_arch_id`; the raw HFQM
        // header value (always 0 there) stays available as `header_arch_id`.
        "arch_id": arch_id,
        "header_arch_id": hfq.arch_id,
        "artifact_kind": meta_get(meta, "artifact_kind").cloned().unwrap_or(Value::Null),
        "arch_name": arch_name,
        "hfqm_version": hfq.version,
        "fingerprint": hfq.index_fingerprint(),
        "quant_family": meta_get(meta, "quant_family"),
        "kv_mode": meta_get(meta, "kv_mode"),
        "sidecars": sidecar_tags(meta),
        "components": components,
        "shape": shape,
        "quant_histogram": histogram,
        "totals": { "tensors": total_tensors, "bytes": total_bytes },
        "modules": modules,
        "file_bytes": file_bytes,
        "diffusion": diffusion,
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

    #[test]
    fn calibration_arch_comes_from_source_not_the_zero_header() {
        // A calib artefact's HFQM header carries arch 0; trusting it printed
        // "arch: 0 (llama)" for a qwen35 calib.
        let calib = json!({"artifact_kind": "calibration", "source_arch_id": 5});
        assert_eq!(calib_source_arch_id(&calib), Some(5));
        // A model artefact must keep using its own header arch.
        assert_eq!(calib_source_arch_id(&json!({"source_arch_id": 5})), None);
        assert_eq!(calib_source_arch_id(&json!({})), None);
    }

    #[test]
    fn calib_tensor_names_split_into_projection_layer_and_kind() {
        // Arch prefixes vary (`model.`, `model.language_model.`); the layer
        // index and any expert index must both be elided so a 512-expert MoE
        // reports one row instead of 512.
        assert_eq!(
            split_calib_tensor("model.language_model.layers.7.mlp.down_proj.hessian"),
            ("mlp.down_proj".to_string(), Some(7), "hessian")
        );
        assert_eq!(
            split_calib_tensor("model.layers.31.mlp.experts.418.down_proj.imatrix"),
            ("mlp.experts.*.down_proj".to_string(), Some(31), "imatrix")
        );
        // A non-layer capture (embeddings, pooling head) has no layer index.
        assert_eq!(
            split_calib_tensor("model.embed_tokens.hessian"),
            ("model.embed_tokens".to_string(), None, "hessian")
        );
        // A weight tensor in a model artefact carries neither suffix.
        assert_eq!(
            split_calib_tensor("model.layers.0.mlp.down_proj.weight").2,
            ""
        );
    }

    #[test]
    fn hessian_quant_code_is_named_not_marked_unknown() {
        assert_eq!(quant_name(130), "HessianBf16TrilDiagF32");
        assert!(quant_name(200).ends_with('?'), "reserved codes stay marked");
    }
}
