// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! `hipfire list` — local models as a capability/artifact matrix. Mostly
//! presentation: inventory comes from the canonical local LLM registry,
//! capability detection lives in `hipfire_model::model_card` (GPU-free), and
//! the resident-memory estimate reuses the loader's own admission math in
//! `hipfire_runtime::weight_pager` so the table cannot disagree with what the
//! loader will actually admit.

use std::path::{Path, PathBuf};

use clap::Args;
use hipfire_config::{configured_models_dir, LoadedConfig};
use hipfire_model::{
    build_llm_registry_in, model_card, ArchFeatures, FeatureSupport, LlmModelRegistry,
    LlmModelRegistryEntry, ModelCard, Sidecars, ARCH_ID_EMBEDDINGGEMMA,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::weight_pager::estimated_module_resident_bytes;
use serde_json::json;

#[derive(Debug, Args)]
#[command(
    after_help = "Examples:\n  hipfire list\n  hipfire list --json\n  hipfire list --local\n"
)]
pub struct ListArgs {
    /// Emit a machine-readable JSON array instead of the table.
    #[arg(long)]
    pub json: bool,
    /// Skip the secondary (network) model store even when one is configured.
    #[arg(long)]
    pub local: bool,
}

/// Drafter/role sidecar arch ids. Not loadable base architectures — a `.hfq`
/// header only carries one of these when the file IS the sidecar.
const ARCH_ID_DFLASH_DRAFT: u32 = 20;
const ARCH_ID_MTP_HEAD: u32 = 21;
const ARCH_ID_DSPARK_DRAFT: u32 = 22;

/// Coarse model role, derived from the HFQ arch id. Drives both the `TYPE`
/// column and which capability columns are applicable at all: asking whether a
/// text-embedding encoder has a chat template or a speculative drafter is a
/// category error, and printing `·` there reads as "missing", not "moot".
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelType {
    TextGen,
    TextVl,
    TextEmbed,
    ImageGen,
    Draft,
    Unknown,
}

impl ModelType {
    fn label(self) -> &'static str {
        match self {
            ModelType::TextGen => "Text-gen",
            ModelType::TextVl => "Text-VL",
            ModelType::TextEmbed => "Text-embed",
            ModelType::ImageGen => "Image-gen",
            ModelType::Draft => "Draft",
            ModelType::Unknown => "?",
        }
    }

    /// Chat template, speculative drafters, and TriAttention are autoregressive
    /// text concepts. Everything else leaves those cells `n/a`.
    fn is_autoregressive_text(self) -> bool {
        matches!(self, ModelType::TextGen | ModelType::TextVl)
    }

    /// Vision is only a meaningful question for a text model that could carry it.
    fn can_have_vision(self) -> bool {
        matches!(self, ModelType::TextGen | ModelType::TextVl)
    }
}

fn model_type(card: &ModelCard) -> ModelType {
    let Some(arch_id) = card.arch_id else {
        return ModelType::Unknown;
    };
    // `is_diffusion_arch` is the shared answer: it covers the per-family ids AND
    // the legacy 0x3046_4944 marker, which does not fit the u16 the registry
    // keys on and would otherwise truncate into some unrelated arch.
    if hipfire_archs::is_diffusion_arch(arch_id) {
        return ModelType::ImageGen;
    }
    match arch_id {
        ARCH_ID_EMBEDDINGGEMMA => ModelType::TextEmbed,
        ARCH_ID_DFLASH_DRAFT | ARCH_ID_MTP_HEAD | ARCH_ID_DSPARK_DRAFT => ModelType::Draft,
        _ if card.features.vision.is_full() => ModelType::TextVl,
        _ => ModelType::TextGen,
    }
}

/// Tick/cross glyph for an on-disk artifact, or `n/a` when the question does
/// not apply to this model type.
fn yn(applicable: bool, v: bool) -> &'static str {
    if !applicable {
        "n/a"
    } else if v {
        "✓"
    } else {
        "·"
    }
}

/// Glyph for a tri-state arch-feature support level.
fn tri(applicable: bool, s: FeatureSupport) -> &'static str {
    if !applicable {
        return "n/a";
    }
    match s {
        FeatureSupport::Full => "✓",
        FeatureSupport::Partial => "~",
        FeatureSupport::None => "·",
        FeatureSupport::Unknown => "?",
    }
}

fn arch_cell(card: &ModelCard) -> String {
    match card.arch_id {
        Some(id) => format!("{} ({id})", card.features.label),
        None => "?".to_string(),
    }
}

fn nonempty_cell(values: &[String]) -> String {
    if values.is_empty() {
        "-".to_string()
    } else {
        values.join(",")
    }
}

fn optional_cell(value: Option<&str>) -> String {
    value
        .filter(|value| !value.is_empty())
        .unwrap_or("-")
        .to_string()
}

/// Everything left of the `--` machine-section boundary is the human-readable
/// model name (AGENTS.md "Artifact Names").
///
/// Older all-dotted artifacts (`zaya1-8b-native.oq4++`) have no boundary, so cut
/// at the header's own quant token instead — searched from the RIGHT, since the
/// token sits at the end and a model may legitimately repeat it earlier. The cut
/// lands before the token, so a name that spells it longer than the header does
/// (name `oq4++`, header `oq4`) still trims clean.
///
/// Deliberately NOT `hipfire_model::quant_token`: that falls back to the last
/// `-`-separated segment when it recognizes nothing, which turns `SomeModel-7B`
/// into `SomeModel`. An unrecognized name keeps its full name.
fn strip_quant(model: &str, metadata_quant: &str) -> String {
    if let Some((head, _)) = model.split_once("--") {
        return head.trim_end_matches(['-', '.']).to_string();
    }
    if metadata_quant.is_empty() || metadata_quant == "unknown" {
        return model.to_string();
    }
    model
        .to_ascii_lowercase()
        .rfind(&metadata_quant.to_ascii_lowercase())
        .filter(|cut| *cut > 0)
        .map(|cut| model[..cut].trim_end_matches(['-', '.']).to_string())
        .unwrap_or_else(|| model.to_string())
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Minimum and full resident host/device bytes for an artifact.
///
/// `full` is the loader's own admission estimate: routed-expert modules priced
/// at what they actually occupy (compact→Oq8 expansion plus per-tensor GTT
/// rounding), the rest of the file at its disk length. `min` is that same file
/// with every routed-expert module left on the host — the floor a paged load
/// has to fit. A dense artifact has no module table, so the two coincide.
fn resident_bytes(path: &Path, disk: u64) -> (u64, u64) {
    let Ok(index) = HfqFile::open_index_only(path) else {
        return (disk, disk);
    };
    let (resident, on_disk) = estimated_module_resident_bytes(&index);
    let backbone = disk.saturating_sub(on_disk);
    (backbone, backbone.saturating_add(resident))
}

struct ListRow {
    model: String,
    ty: ModelType,
    params: String,
    disk: u64,
    ram_min: u64,
    ram_full: u64,
    tags: String,
    features: String,
    quant: String,
    artifact_arch: String,
    store: &'static str,
    card: ModelCard,
}

fn card_from_registry_entry(entry: &LlmModelRegistryEntry) -> ModelCard {
    let mut card = model_card(Path::new(&entry.path));
    card.quant = entry.quant.clone();
    card.sidecars.template |= !entry.chat_templates.is_empty();
    card.sidecars.dflash |= !entry.drafts.is_empty();
    card.sidecars.triattn |= !entry.triattn.is_empty();
    card
}

fn row_from_registry_entry(entry: &LlmModelRegistryEntry, store: &'static str) -> ListRow {
    let card = card_from_registry_entry(entry);
    let (ram_min, ram_full) = resident_bytes(Path::new(&entry.path), entry.bytes);
    ListRow {
        model: strip_quant(&entry.model, &entry.quant),
        ty: model_type(&card),
        params: optional_cell(entry.size.as_deref()),
        disk: entry.bytes,
        ram_min,
        ram_full,
        tags: nonempty_cell(&entry.tags),
        features: nonempty_cell(&entry.features),
        quant: entry.quant.clone(),
        artifact_arch: optional_cell(entry.arch.as_deref()),
        store,
        card,
    }
}

/// A `.hfq` speculative sidecar sitting beside the base counts as spec support,
/// whether it is a DFlash drafter or an MTP head.
fn has_spec(card: &ModelCard) -> bool {
    card.sidecars.mtp || card.sidecars.dflash
}

/// The secondary store, if one is configured and is not the primary. Reading it
/// can mean an NFS round-trip per artifact, hence the progress line.
fn secondary_store(loaded: &LoadedConfig, primary: &Path) -> Option<PathBuf> {
    let dir = PathBuf::from(
        loaded
            .config
            .models_network_dir
            .as_deref()
            .filter(|dir| !dir.is_empty())?,
    );
    (dir != primary && dir.exists()).then_some(dir)
}

fn scan(dir: &Path) -> LlmModelRegistry {
    let hipfire = hipfire_config::hipfire_dir();
    build_llm_registry_in(
        dir,
        &hipfire.join("triattn"),
        &hipfire.join("drafts"),
        &hipfire.join("templates"),
    )
}

pub fn run(args: ListArgs, loaded: LoadedConfig) -> anyhow::Result<()> {
    let models_dir = configured_models_dir(&loaded.config);
    let secondary = (!args.local)
        .then(|| secondary_store(&loaded, &models_dir))
        .flatten();

    let mut rows: Vec<ListRow> = scan(&models_dir)
        .models
        .iter()
        .map(|entry| row_from_registry_entry(entry, "local"))
        .collect();

    if let Some(dir) = &secondary {
        // ponytail: one stderr line, not a progress bar — the scan is a single
        // opaque directory walk with no per-item callback to drive one from.
        // Swap in indicatif if `build_llm_registry_in` ever reports progress.
        eprint!("scanning {} ... ", dir.display());
        let found = scan(dir);
        eprintln!("{} models", found.models.len());
        // A model present in both stores is the same artifact; keep the local copy.
        let extra: Vec<ListRow> = found
            .models
            .iter()
            .filter(|entry| !rows.iter().any(|row| row.card.name == entry.id))
            .map(|entry| row_from_registry_entry(entry, "network"))
            .collect();
        rows.extend(extra);
    }

    if rows.is_empty() {
        if args.json {
            println!("[]");
        } else {
            println!("No valid HFQ models found in {}", models_dir.display());
        }
        return Ok(());
    }
    rows.sort_by(|a, b| a.model.cmp(&b.model).then_with(|| a.quant.cmp(&b.quant)));

    if args.json {
        return print_json(&rows);
    }
    print_table(&rows, secondary.is_some());
    Ok(())
}

fn print_json(rows: &[ListRow]) -> anyhow::Result<()> {
    let items: Vec<_> = rows
        .iter()
        .map(|row| {
            let card = &row.card;
            let ar = row.ty.is_autoregressive_text();
            json!({
                "model": row.model,
                "type": row.ty.label(),
                "store": row.store,
                "params": row.params,
                "disk_bytes": row.disk,
                "ram_min_bytes": row.ram_min,
                "ram_full_bytes": row.ram_full,
                "quant": row.quant,
                "tags": row.tags,
                "features": row.features,
                "artifact_arch": row.artifact_arch,
                "arch_id": card.arch_id,
                "arch": card.features.label,
                "on_disk": {
                    "template": ar.then_some(card.sidecars.template),
                    "spec": ar.then(|| has_spec(card)),
                    "triattn": ar.then_some(card.sidecars.triattn),
                    "hessian": card.sidecars.hessian,
                },
                "arch_features": {
                    "prefill": format!("{:?}", card.features.prefill),
                    "dflash": format!("{:?}", card.features.dflash),
                    "mtp": format!("{:?}", card.features.mtp),
                    "kv": card.features.kv,
                    "vision": row.ty.can_have_vision()
                        .then(|| format!("{:?}", card.features.vision)),
                },
            })
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&items)?);
    Ok(())
}

/// Terminal width, or a sane default when stdout is not a tty.
fn term_width() -> usize {
    if let Ok(cols) = std::env::var("COLUMNS") {
        if let Ok(cols) = cols.parse::<usize>() {
            return cols;
        }
    }
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    if unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) } == 0 && ws.ws_col > 0
    {
        return ws.ws_col as usize;
    }
    100
}

/// One rendered row, split at the point where it may be folded onto a second
/// line: identity + sizes stay together, capability cells go below.
struct Cells {
    head: Vec<String>,
    tail: Vec<String>,
}

const HEAD: &[&str] = &["MODEL", "TYPE", "PARAMS", "DISK", "RAM min/full"];
const TAIL: &[&str] = &[
    "QUANT", "ART", "ARCH", "tpl", "spec", "tri", "hess", "pfil", "dfl", "mtp", "kv", "vis",
];

fn cells(row: &ListRow) -> Cells {
    let card = &row.card;
    let Sidecars {
        template,
        triattn,
        hessian,
        ..
    } = card.sidecars;
    let ArchFeatures {
        prefill,
        dflash: f_dflash,
        mtp: f_mtp,
        kv,
        vision,
        ..
    } = card.features;
    let ar = row.ty.is_autoregressive_text();
    Cells {
        head: vec![
            row.model.clone(),
            row.ty.label().to_string(),
            row.params.clone(),
            format!("{:.1} GB", gb(row.disk)),
            format!("{:.1}/{:.1} GB", gb(row.ram_min), gb(row.ram_full)),
        ],
        tail: vec![
            row.quant.clone(),
            row.artifact_arch.clone(),
            arch_cell(card),
            yn(ar, template).into(),
            yn(ar, has_spec(card)).into(),
            yn(ar, triattn).into(),
            yn(true, hessian).into(),
            tri(true, prefill).into(),
            tri(ar, f_dflash).into(),
            tri(ar, f_mtp).into(),
            kv.to_string(),
            tri(row.ty.can_have_vision(), vision).into(),
        ],
    }
}

/// Per-column width: the widest of the header and every cell. Counts chars, not
/// bytes — `✓` and `·` are multi-byte, so `len()` would over-pad every glyph
/// column and shear the table.
fn widths(headers: &[&str], rows: &[Vec<String>]) -> Vec<usize> {
    headers
        .iter()
        .enumerate()
        .map(|(i, header)| {
            rows.iter()
                .map(|cells| cells[i].chars().count())
                .chain(std::iter::once(header.chars().count()))
                .max()
                .unwrap_or(0)
        })
        .collect()
}

fn join(cells: &[String], widths: &[usize]) -> String {
    cells
        .iter()
        .zip(widths)
        .map(|(cell, width)| {
            let pad = width.saturating_sub(cell.chars().count());
            format!("{cell}{:pad$}", "")
        })
        .collect::<Vec<_>>()
        .join("  ")
}

/// Trailing padding is only dead space at the end of a whole LINE. Trimming it
/// per column group instead would eat the last head column's padding and shear
/// every tail column right of it.
fn line(parts: &[String]) -> String {
    parts.join("  ").trim_end().to_string()
}

fn print_table(rows: &[ListRow], scanned_network: bool) {
    let rendered: Vec<Cells> = rows.iter().map(cells).collect();
    let head_cells: Vec<Vec<String>> = rendered.iter().map(|c| c.head.clone()).collect();
    let tail_cells: Vec<Vec<String>> = rendered.iter().map(|c| c.tail.clone()).collect();
    let head_w = widths(HEAD, &head_cells);
    let tail_w = widths(TAIL, &tail_cells);

    let head_len: usize = head_w.iter().map(|w| w + 2).sum();
    let tail_len: usize = tail_w.iter().map(|w| w + 2).sum();
    // Fold to two lines per model when one line will not fit. The continuation
    // is indented so a model's two lines read as one record.
    const INDENT: &str = "    ";
    let two_line = head_len + tail_len > term_width();

    let head_headers: Vec<String> = HEAD.iter().map(|h| h.to_string()).collect();
    let tail_headers: Vec<String> = TAIL.iter().map(|h| h.to_string()).collect();
    let emit = |head: &[String], tail: &[String]| {
        if two_line {
            println!("{}", line(&[join(head, &head_w)]));
            println!("{INDENT}{}", line(&[join(tail, &tail_w)]));
        } else {
            println!("{}", line(&[join(head, &head_w), join(tail, &tail_w)]));
        }
    };
    emit(&head_headers, &tail_headers);
    for cells in &rendered {
        emit(&cells.head, &cells.tail);
    }

    println!(
        "\nRAM min/full: min leaves routed experts on the host (paged); full is every module resident"
    );
    println!(
        "on disk: tpl=chat template · spec=MTP/DFlash drafter (sidecar or bundled) · tri=TriAttn · hess=Hessian/calib"
    );
    println!(
        "arch features (per MODEL-SUPPORT.md): ✓=full · ~=partial · ·=none · n/a=not applicable to this model type · pfil=batched prefill · kv=quant menu"
    );
    if scanned_network {
        println!(
            "stores: local + secondary (models_network_dir); `--local` skips the secondary scan"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_quant_uses_the_machine_section_boundary() {
        assert_eq!(
            strip_quant("Qwen3.5-122B-A10B--+mtp.+vl.mq2l", "mq2l"),
            "Qwen3.5-122B-A10B"
        );
        assert_eq!(
            strip_quant("Gemma-4-8B-E4B-it--oq4++", "oq4++"),
            "Gemma-4-8B-E4B-it"
        );
        // Legacy all-dotted artifacts carry no boundary.
        assert_eq!(strip_quant("Qwen3.5-0.8B.mq4", "mq4"), "Qwen3.5-0.8B");
        // Name says `oq4++`, header says `oq4`: cut at the name's own token.
        assert_eq!(
            strip_quant("zaya1-8b-native.oq4++", "oq4"),
            "zaya1-8b-native"
        );
        // Nothing to strip is left alone: an unknown quant, and a name whose
        // trailing segment is a parameter size rather than a format token.
        assert_eq!(strip_quant("SomeModel-7B", "unknown"), "SomeModel-7B");
        assert_eq!(strip_quant("SomeModel-7B", "mq4"), "SomeModel-7B");
    }

    #[test]
    fn widths_count_chars_not_bytes() {
        // "✓" is 3 bytes, 1 column. A byte-length width shears every glyph column.
        let rows = vec![vec!["✓".to_string()], vec!["n/a".to_string()]];
        assert_eq!(widths(&["tpl"], &rows), vec![3]);
        assert_eq!(join(&["✓".to_string()], &[3]), "✓  ");
    }

    #[test]
    fn column_groups_keep_their_padding_when_joined() {
        // The regression: trimming inside `join` ate the last head column's pad,
        // so every tail column shifted left by however short that cell was.
        let head = join(&["a".to_string(), "b".to_string()], &[3, 3]);
        let tail = join(&["c".to_string()], &[1]);
        assert_eq!(line(&[head, tail]), "a    b    c");
    }
}
