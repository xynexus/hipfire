//! `hipfire model {compose,decompose}` — reorganize `.hfq` packaging between a
//! single bundled container and separate base + role/feature sidecar files.
//!
//! Native `.hfq`->`.hfq` manipulation (no tensor-byte transform); the heavy
//! lifting lives in the offline `hipfire-hfq-tooling` crate.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use hipfire_arch_api::ArchId;
use hipfire_config::LoadedConfig;
use hipfire_hfq_tooling::{
    check_compose_inputs, compose_hfq_with_config_keys_options,
    decompose_hfq_auto_with_config_keys_options, sidecar_role_group, sidecar_tag_from_filename,
    RoleConfigKeys, EMBEDDED_ROLE_PREFIX, KNOWN_ROLES,
};
use hipfire_runtime::hfq::HfqPackage;

use crate::model::find_model;

/// Build the `role -> owned config-key` map for a container's arch by consulting
/// the arch registry (`Arch::sidecar_config_keys`). Empty when the arch isn't
/// registered or declares no sidecar-specific config — which reproduces the
/// pre-partition compose/decompose behavior.
fn role_config_keys_for(path: &Path) -> RoleConfigKeys {
    let Ok(pkg) = HfqPackage::open(path) else {
        return RoleConfigKeys::new();
    };
    let Some(arch) = hipfire_archs::registry().get(ArchId(pkg.arch_id as u16)) else {
        return RoleConfigKeys::new();
    };
    let mut map = RoleConfigKeys::new();
    for role in KNOWN_ROLES {
        let keys = arch.base.sidecar_config_keys(role);
        if !keys.is_empty() {
            map.insert(
                role.to_string(),
                keys.iter().map(|s| s.to_string()).collect(),
            );
        }
    }
    map
}

#[derive(Debug, Args)]
pub struct ModelArgs {
    #[command(subcommand)]
    command: ModelCommand,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Merge a base `.hfq` and its role/feature sidecars into one bundled
    /// container (records a provenance manifest so `decompose` is lossless).
    Compose(ComposeArgs),
    /// Split a bundled `.hfq` back into its base + sidecar files.
    Decompose(DecomposeArgs),
    /// Interactive wizard: bring an external model (HuggingFace repo or local
    /// safetensors dir) into a named `.hfq` — calibrate, quantize, fold sidecars.
    Induct(crate::commands::induct::InductArgs),
}

#[derive(Debug, Args)]
pub struct ComposeArgs {
    /// Base container first, then one or more sidecars (file paths or model
    /// aliases).
    #[arg(required = true, num_args = 2..)]
    inputs: Vec<String>,
    /// Output bundle path. Default: the base name with the sidecar feature
    /// dot-groups inserted before the quant token, each marked `+` because the
    /// role is now embedded rather than standalone (e.g. `Model--mq4.hfq` +
    /// `Model.mtp.hfq` -> `Model--+mtp.mq4.hfq`).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Validate component roles, formats, architectures, geometry, lengths,
    /// digests, and reserved namespaces without writing a bundle.
    #[arg(long)]
    check: bool,
    /// Emit a machine-readable JSON report.
    #[arg(long)]
    json: bool,
    /// Replace an existing output bundle. Without this flag compose fails
    /// closed when the destination exists.
    #[arg(long)]
    overwrite: bool,
}

#[derive(Debug, Args)]
pub struct DecomposeArgs {
    /// Bundle container to split (file path or model alias).
    bundle: String,
    /// Directory to write the reconstructed component files into.
    output_dir: PathBuf,
    /// Heuristically split a bundle that has no `hipfire_compose` manifest,
    /// using the filename's role dot-groups + tensor-name prefixes. Legacy
    /// bundles with a plain filename fall back to inferring roles from tensor
    /// names alone. Lossy: output files are not byte-identical to any originals.
    /// Bundles that DO carry a manifest still take the exact, lossless path.
    #[arg(long)]
    infer: bool,
    /// Emit a machine-readable JSON report.
    #[arg(long)]
    json: bool,
    /// Replace existing reconstructed component files. Without this flag
    /// decompose fails closed before replacing a destination.
    #[arg(long)]
    overwrite: bool,
}

pub fn run(args: ModelArgs, loaded: LoadedConfig) -> anyhow::Result<()> {
    match args.command {
        ModelCommand::Compose(a) => run_compose(a, &loaded),
        ModelCommand::Decompose(a) => run_decompose(a, &loaded),
        ModelCommand::Induct(a) => crate::commands::induct::run_induct(a, loaded),
    }
}

/// Resolve an argument to a concrete path: an existing file path wins, else it
/// is treated as a model alias resolved against the models directory.
fn resolve(arg: &str, loaded: &LoadedConfig) -> anyhow::Result<PathBuf> {
    let direct = PathBuf::from(arg);
    if direct.exists() {
        return Ok(direct);
    }
    find_model(arg, &loaded.config).ok_or_else(|| anyhow::anyhow!("no such file or model: {arg}"))
}

/// Default bundle path: insert sorted, de-duplicated sidecar feature tags as
/// dot-groups immediately before the base's quant token (the last stem
/// segment), per the artifact naming convention.
fn default_bundle_path(inputs: &[PathBuf]) -> PathBuf {
    let base = &inputs[0];
    let dir = base.parent().map(PathBuf::from);
    let fname = base
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "bundle.hfq".to_string());
    let stem = fname.strip_suffix(".hfq").unwrap_or(&fname);

    // Sidecar inputs are named by their own (unmarked) role; once folded into a
    // bundle the role becomes embedded, so it is written `+role`.
    let mut tags: Vec<String> = inputs[1..]
        .iter()
        .filter_map(|p| sidecar_tag_from_filename(p))
        .map(|tag| format!("{EMBEDDED_ROLE_PREFIX}{tag}"))
        .collect();
    tags.sort();
    tags.dedup();

    let out_name = if let Some((identity, machine)) = stem.split_once("--") {
        // New convention: features are dotted groups after the `--` boundary and
        // before the quant. Insert new tags at the front of the machine section
        // (still before the quant), skipping any already present. A group already
        // there in either spelling counts as present, so re-composing a bundle
        // does not double up `dflash` and `+dflash`.
        let mut groups: Vec<String> = tags;
        for g in machine.split('.') {
            let bare = g.strip_prefix(EMBEDDED_ROLE_PREFIX).unwrap_or(g);
            let already = groups.iter().any(|existing| {
                existing
                    .strip_prefix(EMBEDDED_ROLE_PREFIX)
                    .unwrap_or(existing)
                    == bare
            });
            if !already {
                // Carry an existing bare role group over as embedded.
                groups.push(match sidecar_role_group(g) {
                    Some(role) => format!("{EMBEDDED_ROLE_PREFIX}{role}"),
                    None => g.to_string(),
                });
            }
        }
        format!("{identity}--{}.hfq", groups.join("."))
    } else {
        // Legacy dotted base: insert features before the last (quant) segment.
        let mut segs: Vec<String> = stem.split('.').map(|s| s.to_string()).collect();
        let insert_at = if segs.len() >= 2 {
            segs.len() - 1
        } else {
            segs.len()
        };
        for (i, tag) in tags.into_iter().enumerate() {
            if !segs.contains(&tag) {
                segs.insert(insert_at + i, tag);
            }
        }
        format!("{}.hfq", segs.join("."))
    };
    match dir {
        Some(d) if !d.as_os_str().is_empty() => d.join(out_name),
        _ => PathBuf::from(out_name),
    }
}

fn run_compose(a: ComposeArgs, loaded: &LoadedConfig) -> anyhow::Result<()> {
    let inputs: Vec<PathBuf> = a
        .inputs
        .iter()
        .map(|s| resolve(s, loaded))
        .collect::<anyhow::Result<_>>()?;
    let report = check_compose_inputs(&inputs)?;
    if a.check {
        if a.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            println!(
                "compatible: {} component(s), bundle arch {}, manifest {}",
                report.components.len(),
                report.bundle_arch_id,
                report.manifest_format
            );
            for component in &report.components {
                println!(
                    "  {}: {} ({}, arch {}, {} entries, {} bytes, sha256 {})",
                    component.role,
                    component.filename,
                    component.source_format,
                    component
                        .arch_id
                        .map(|arch| arch.to_string())
                        .unwrap_or_else(|| "n/a".to_string()),
                    component.entries,
                    component.byte_len,
                    component.sha256
                );
            }
        }
        return Ok(());
    }
    let out = a.output.unwrap_or_else(|| default_bundle_path(&inputs));
    // Arch owns which config keys belong to each sidecar; the base (first input)
    // determines the arch.
    let role_keys = role_config_keys_for(&inputs[0]);
    let written = compose_hfq_with_config_keys_options(&inputs, &out, &role_keys, a.overwrite)?;
    if a.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "operation": "compose",
                "output": written,
                "inputs": inputs,
                "check": report,
            }))?
        );
    } else {
        println!("composed {} inputs -> {}", inputs.len(), written.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bundle_path_uses_double_hyphen_boundary() {
        // New-form base: the feature is inserted into the machine section,
        // before the quant, keeping the `--` boundary. The folded role is
        // `+`-marked because it is now embedded, not the artifact's own role.
        let out = default_bundle_path(&[
            PathBuf::from("/m/MiniCPM5-1B--mq4.hfq"),
            PathBuf::from("/m/MiniCPM5-1B.mtp.hfq"),
        ]);
        assert_eq!(out, PathBuf::from("/m/MiniCPM5-1B--+mtp.mq4.hfq"));

        // Several sidecars: one marked group each, sorted, quant last.
        let multi = default_bundle_path(&[
            PathBuf::from("/m/Qwen3.5-9B--mq4.hfq"),
            PathBuf::from("/m/Qwen3.5-9B--dflash.oq4+.hfq"),
            PathBuf::from("/m/Qwen3.5-9B.triattn.hfq"),
        ]);
        assert_eq!(
            multi,
            PathBuf::from("/m/Qwen3.5-9B--+dflash.+triattn.mq4.hfq")
        );

        // Re-composing a bundle does not double up `dflash` and `+dflash`.
        let again = default_bundle_path(&[
            PathBuf::from("/m/Qwen3.5-9B--+dflash.mq4.hfq"),
            PathBuf::from("/m/Qwen3.5-9B--dflash.oq4+.hfq"),
        ]);
        assert_eq!(again, PathBuf::from("/m/Qwen3.5-9B--+dflash.mq4.hfq"));

        // Legacy dotted base keeps its dotted shape; the folded role is still
        // marked, since the marker is orthogonal to the `--` boundary.
        let legacy = default_bundle_path(&[
            PathBuf::from("/m/Model.mq4.hfq"),
            PathBuf::from("/m/Model.mtp.hfq"),
        ]);
        assert_eq!(legacy, PathBuf::from("/m/Model.+mtp.mq4.hfq"));
    }

    #[test]
    fn embedded_marker_separates_bundles_from_sidecars() {
        use hipfire_hfq_tooling::{embedded_role_group, sidecar_role_group};
        // A bare group names the artifact's own role; a marked one says the role
        // is carried inside it.
        assert_eq!(sidecar_role_group("dflash"), Some("dflash"));
        assert_eq!(sidecar_role_group("+dflash"), None);
        assert_eq!(embedded_role_group("+dflash"), Some("dflash"));
        assert_eq!(embedded_role_group("dflash"), None);
        // The quant token's trailing `+` is a suffix and must not be read as a
        // marker.
        assert_eq!(embedded_role_group("oq4+"), None);
        assert_eq!(sidecar_role_group("oq4+"), None);
    }
}

fn run_decompose(a: DecomposeArgs, loaded: &LoadedConfig) -> anyhow::Result<()> {
    let bundle = resolve(&a.bundle, loaded)?;
    let role_keys = role_config_keys_for(&bundle);
    let written = decompose_hfq_auto_with_config_keys_options(
        &bundle,
        &a.output_dir,
        a.infer,
        &role_keys,
        a.overwrite,
    )?;
    if a.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "operation": "decompose",
                "bundle": bundle,
                "output_dir": a.output_dir,
                "infer": a.infer,
                "outputs": written,
            }))?
        );
        return Ok(());
    }
    println!(
        "decomposed{} {} -> {} file(s) in {}",
        if a.infer { " (heuristic)" } else { "" },
        bundle.display(),
        written.len(),
        a.output_dir.display()
    );
    for p in &written {
        println!("  {}", p.display());
    }
    Ok(())
}
