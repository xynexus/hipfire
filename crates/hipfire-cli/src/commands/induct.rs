//! `hipfire model induct` — interactive wizard to bring an external model into a
//! named `.hfq`.
//!
//! It orchestrates existing tools and adds no quant/format logic of its own:
//!   1. resolve a source (HuggingFace repo id or local safetensors dir; HF is
//!      checked against `/srv/huggingface` and the HF cache before any download),
//!   2. optionally run `hipfire-coexistence calibrate` to produce a `.calib.hfq`
//!      (Hessian/imatrix) when the chosen quant needs one,
//!   3. run `hipfire-quantize` to emit the `.hfq`,
//!   4. optionally fold in sidecars via `hipfire model compose`.
//!
//! QAT and DFlash-drafter *building* are out of scope for the wizard (no
//! productionized entrypoint); it points at the existing commands instead.
//!
//! Per AGENTS.md the download/convert *logic* is offline tooling — the wizard is
//! only the interactive shell that sequences the real tools.

use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context};
use clap::Args;
use hipfire_config::LoadedConfig;

/// (quant token, one-line help). The token is also the artifact quant field, so
/// it becomes the output filename's `.<token>.hfq` group directly.
const KNOWN_FORMATS: &[(&str, &str)] = &[
    (
        "oq4++",
        "Opus Quant 4-bit, clip + Hessian/LDLQ (needs calibration) — best 4-bit",
    ),
    (
        "oq8++",
        "Opus Quant 8-bit, Hessian/LDLQ (needs calibration) — near-lossless",
    ),
    ("oq4", "Opus Quant 4-bit symmetric (no calibration)"),
    ("oq8", "Opus Quant 8-bit symmetric (no calibration)"),
    (
        "mq4+",
        "Magnum Quant 4-bit, activation-aware clip (needs calibration)",
    ),
    ("mq4", "Magnum Quant 4-bit affine (no calibration)"),
    (
        "qtip3",
        "Trellis 3-bit — near-mq4 quality, ~26% less weight bandwidth",
    ),
    ("bf16", "No quantization — bf16 passthrough"),
];

#[derive(Debug, Args)]
pub struct InductArgs {
    /// Model source: a HuggingFace repo id (`org/name`) or a local safetensors
    /// directory. Omit to be prompted.
    source: Option<String>,
    /// Quant format token (e.g. `oq4++`, `mq4`, `qtip3`, `bf16`). Omit to be
    /// prompted from the known list.
    #[arg(long)]
    format: Option<String>,
}

// ── pure helpers (unit-tested) ─────────────────────────────────────────────

/// `+`/`++` formats are activation-/Hessian-aware and need a calibration
/// sidecar; symmetric formats need nothing. Both `+` and `++` are fed the
/// `.calib.hfq` via `hipfire-quantize --hessian` (it reads the bundled
/// Hessian+imatrix); `++` additionally enables LDLQ error feedback.
fn format_needs_calibration(format: &str) -> bool {
    format.contains('+')
}

fn format_uses_ldlq(format: &str) -> bool {
    format.contains("++")
}

/// HF cache directory key for a repo id, matching the `huggingface_hub` layout
/// (`models--org--name`).
fn hf_cache_key(repo: &str) -> String {
    format!("models--{}", repo.replace('/', "--"))
}

/// Last path/id segment — the artifact family+size stem.
/// `Qwen/Qwen3.5-0.8B` -> `Qwen3.5-0.8B`; `/models/foo` -> `foo`.
fn source_stem(source: &str) -> &str {
    let s = source.trim_end_matches('/');
    s.rsplit('/').next().unwrap_or(s)
}

/// Default output artifact name: `<stem>.<quant-token>.hfq`.
fn derive_output_name(source: &str, format: &str) -> String {
    format!("{}.{}.hfq", source_stem(source), format)
}

/// A source string is a local dir if it exists as one; otherwise it is treated
/// as an HF repo id.
fn is_local_dir(source: &str) -> bool {
    Path::new(source).is_dir()
}

/// Which sibling files are worth downloading: config/weights/tokenizer only —
/// skip pytorch `.bin`/`.pth`, `.gguf`, docs, images.
fn want_hf_file(name: &str) -> bool {
    let n = name.rsplit('/').next().unwrap_or(name);
    n.ends_with(".safetensors")
        || n.ends_with(".json")
        || n.ends_with(".model")
        || n.contains("tokenizer")
}

// ── filesystem / HF resolution ─────────────────────────────────────────────

fn hf_home() -> PathBuf {
    if let Some(h) = std::env::var_os("HF_HOME").filter(|h| !h.is_empty()) {
        return PathBuf::from(h);
    }
    if let Some(h) = std::env::var_os("HOME") {
        return Path::new(&h).join(".cache/huggingface");
    }
    PathBuf::from(".cache/huggingface")
}

fn hf_hub_dir() -> PathBuf {
    hf_home().join("hub")
}

fn dir_has_safetensors(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        })
        .unwrap_or(false)
}

/// Resolve a directory to the one actually holding safetensors: use it directly,
/// else if it is an HF cache dir with `snapshots/<hash>/`, descend into the
/// newest snapshot that has weights.
fn descend_to_model_dir(dir: &Path) -> Option<PathBuf> {
    if dir_has_safetensors(dir) {
        return Some(dir.to_path_buf());
    }
    let mut snaps: Vec<PathBuf> = std::fs::read_dir(dir.join("snapshots"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .collect();
    // Newest snapshot first.
    snaps.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    snaps.into_iter().rev().find(|p| dir_has_safetensors(p))
}

/// Local snapshot candidates for a repo, mount first (AGENTS.local.md: check
/// `/srv/huggingface` before downloading), then the on-disk HF cache. Both
/// `/srv/huggingface` (used directly as a hub cache here) and its `hub/`
/// subdir are treated as cache roots.
fn local_snapshot_candidates(repo: &str) -> Vec<PathBuf> {
    let key = hf_cache_key(repo);
    let mut out = vec![PathBuf::from("/srv/huggingface").join(repo)];
    for hub in [
        PathBuf::from("/srv/huggingface"),
        PathBuf::from("/srv/huggingface/hub"),
        hf_hub_dir(),
    ] {
        let snaps = hub.join(&key).join("snapshots");
        if let Ok(rd) = std::fs::read_dir(&snaps) {
            for e in rd.flatten() {
                out.push(e.path());
            }
        }
    }
    out
}

/// Resolve a source to a directory holding safetensors. Local dir wins; else an
/// HF repo id is checked against the mount/cache and only then downloaded.
fn resolve_source(source: &str) -> anyhow::Result<PathBuf> {
    if is_local_dir(source) {
        return descend_to_model_dir(Path::new(source)).ok_or_else(|| {
            anyhow::anyhow!("{source} has no *.safetensors (even under snapshots/)")
        });
    }
    if !source.contains('/') {
        bail!("'{source}' is neither a local directory nor an org/name HuggingFace repo id");
    }
    for cand in local_snapshot_candidates(source) {
        if let Some(dir) = descend_to_model_dir(&cand) {
            println!("• found local copy: {}", dir.display());
            return Ok(dir);
        }
    }
    println!("• not on /srv/huggingface or in the HF cache — downloading {source} …");
    hf_download(source)
}

fn read_hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.is_empty() {
            return Some(t);
        }
    }
    std::fs::read_to_string(hf_home().join("token"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Download a repo's config/weights/tokenizer into the standard HF cache
/// snapshot dir (so a later run's mount/cache check finds it) and return it.
fn hf_download(repo: &str) -> anyhow::Result<PathBuf> {
    let dest = hf_hub_dir()
        .join(hf_cache_key(repo))
        .join("snapshots")
        .join("main");
    std::fs::create_dir_all(&dest)?;
    let token = read_hf_token();
    let rt = tokio::runtime::Runtime::new().context("start async runtime for HF download")?;
    let dl_dest = dest.clone();
    rt.block_on(async move {
        let client = reqwest::Client::builder()
            .user_agent("hipfire-induct")
            .build()?;
        let files = hf_list_files(&client, repo, token.as_deref()).await?;
        let wanted: Vec<String> = files.into_iter().filter(|f| want_hf_file(f)).collect();
        if wanted.is_empty() {
            bail!("no config/safetensors/tokenizer files listed for {repo}");
        }
        for f in &wanted {
            let out = dl_dest.join(f);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if out.exists()
                && std::fs::metadata(&out)
                    .map(|m| m.len() > 0)
                    .unwrap_or(false)
            {
                continue; // resume: skip already-fetched files
            }
            hf_get_file(&client, repo, f, &out, token.as_deref()).await?;
        }
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(dest)
}

async fn hf_list_files(
    client: &reqwest::Client,
    repo: &str,
    token: Option<&str>,
) -> anyhow::Result<Vec<String>> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp =
        req.send().await?.error_for_status().with_context(|| {
            format!("HF API list failed for {repo} (private/gated? set HF_TOKEN)")
        })?;
    let json: serde_json::Value = resp.json().await?;
    let files = json["siblings"]
        .as_array()
        .map(|s| {
            s.iter()
                .filter_map(|e| e["rfilename"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    Ok(files)
}

async fn hf_get_file(
    client: &reqwest::Client,
    repo: &str,
    rfilename: &str,
    out: &Path,
    token: Option<&str>,
) -> anyhow::Result<()> {
    let url = format!("https://huggingface.co/{repo}/resolve/main/{rfilename}");
    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let mut resp = req
        .send()
        .await?
        .error_for_status()
        .with_context(|| format!("download {rfilename}"))?;
    let bar = resp
        .content_length()
        .map(|len| indicatif::ProgressBar::new(len))
        .unwrap_or_else(indicatif::ProgressBar::new_spinner);
    bar.set_message(rfilename.to_string());
    let tmp = out.with_extension("part");
    let mut file = std::fs::File::create(&tmp)?;
    while let Some(chunk) = resp.chunk().await? {
        file.write_all(&chunk)?;
        bar.inc(chunk.len() as u64);
    }
    file.flush()?;
    drop(file);
    std::fs::rename(&tmp, out)?;
    bar.finish_and_clear();
    println!("  ↓ {rfilename}");
    Ok(())
}

fn ensure_safetensors(dir: &Path) -> anyhow::Result<()> {
    if !dir_has_safetensors(dir) {
        bail!(
            "{} has no *.safetensors — not a usable model directory",
            dir.display()
        );
    }
    Ok(())
}

// ── interactive prompts ────────────────────────────────────────────────────

fn prompt(msg: &str, default: Option<&str>) -> anyhow::Result<String> {
    match default {
        Some(d) if !d.is_empty() => print!("{msg} [{d}]: "),
        _ => print!("{msg}: "),
    }
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let t = line.trim().to_string();
    if t.is_empty() {
        if let Some(d) = default {
            return Ok(d.to_string());
        }
    }
    Ok(t)
}

fn prompt_yes_no(msg: &str, default_yes: bool) -> anyhow::Result<bool> {
    let d = if default_yes { "Y/n" } else { "y/N" };
    let ans = prompt(
        &format!("{msg} ({d})"),
        Some(if default_yes { "y" } else { "n" }),
    )?;
    Ok(matches!(ans.to_ascii_lowercase().as_str(), "y" | "yes"))
}

fn choose_format(preset: Option<&str>) -> anyhow::Result<String> {
    if let Some(f) = preset {
        return Ok(f.to_string());
    }
    println!("Quant format:");
    for (i, (tok, help)) in KNOWN_FORMATS.iter().enumerate() {
        println!("  {:>2}) {:<7} {}", i + 1, tok, help);
    }
    let ans = prompt("Choose (number or token)", Some("oq4++"))?;
    if let Ok(n) = ans.parse::<usize>() {
        if (1..=KNOWN_FORMATS.len()).contains(&n) {
            return Ok(KNOWN_FORMATS[n - 1].0.to_string());
        }
    }
    // Accept an arbitrary token too — hipfire-quantize validates it.
    Ok(ans)
}

/// Prompt for optional sidecar files to fold in. Returns existing paths only.
fn collect_sidecars() -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for (role, hint) in [
        ("DFlash drafter", ".dflash.hfq"),
        ("MTP head", ".mtp.hfq"),
        ("TriAttention centers", ".triattn.hfq"),
        ("chat template", ".jinja"),
    ] {
        let p = prompt(
            &format!("{role} sidecar path ({hint}, blank to skip)"),
            Some(""),
        )?;
        if p.is_empty() {
            continue;
        }
        let pb = PathBuf::from(&p);
        if pb.exists() {
            out.push(pb);
        } else {
            println!("  ! {p} not found — skipping");
        }
    }
    Ok(out)
}

// ── sibling tool invocation ────────────────────────────────────────────────

/// Locate a sibling hipfire binary: `$ENV_VAR`, next to the current exe,
/// `target/{release,debug}`, then `~/.hipfire/bin` (mirrors forward.rs).
fn sibling_binary(release_name: &str, env_var: &str) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(env_var).filter(|p| !p.is_empty()) {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }
    let exe = std::env::consts::EXE_SUFFIX;
    let mut cands = Vec::new();
    if let Ok(cur) = std::env::current_exe() {
        if let Some(dir) = cur.parent() {
            cands.push(dir.join(format!("{release_name}{exe}")));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        cands.push(cwd.join(format!("target/release/{release_name}{exe}")));
        cands.push(cwd.join(format!("target/debug/{release_name}{exe}")));
    }
    if let Some(home) = std::env::var_os("HOME") {
        cands.push(
            Path::new(&home)
                .join(".hipfire/bin")
                .join(format!("{release_name}{exe}")),
        );
    }
    cands.into_iter().find(|p| p.exists())
}

fn run_tool(bin: &Path, args: &[OsString], build_hint: &str) -> anyhow::Result<()> {
    println!(
        "\n$ {} {}\n",
        bin.display(),
        args.iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let status = Command::new(bin)
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("run {}", bin.display()))?;
    if !status.success() {
        bail!(
            "{} failed ({}). Build it with: {build_hint}",
            bin.display(),
            status
        );
    }
    Ok(())
}

// ── wizard ─────────────────────────────────────────────────────────────────

pub fn run_induct(args: InductArgs, loaded: LoadedConfig) -> anyhow::Result<()> {
    println!("hipfire model induct — bring an external model into a named .hfq\n");

    // 1. Source → safetensors dir.
    let source = match args.source {
        Some(s) => s,
        None => prompt(
            "Model source (HuggingFace org/name or local safetensors dir)",
            None,
        )?,
    };
    if source.is_empty() {
        bail!("no source given");
    }
    let model_dir = resolve_source(&source)?;
    ensure_safetensors(&model_dir)?;
    println!("• source model: {}\n", model_dir.display());

    // 2. Quant format.
    let format = choose_format(args.format.as_deref())?;
    println!("• format: {format}");

    // 3. Calibration (only when the format needs it).
    let mut calib: Option<PathBuf> = None;
    if format_needs_calibration(&format) {
        let existing = prompt(
            "Existing .calib.hfq (Hessian/imatrix) path, blank to generate now",
            Some(""),
        )?;
        if !existing.is_empty() {
            calib = Some(PathBuf::from(existing));
        } else {
            let corpus = prompt("Calibration corpus text file", None)?;
            if corpus.is_empty() {
                bail!("{format} needs calibration but no corpus or existing .calib.hfq given");
            }
            let calib_out = model_dir.join(format!("{}.calib.hfq", source_stem(&source)));
            let coex = sibling_binary("hipfire-coexistence", "HIPFIRE_COEXISTENCE_BIN").context(
                "hipfire-coexistence not found (cargo build --release -p hipfire-coexistence)",
            )?;
            run_tool(
                &coex,
                &[
                    "calibrate".into(),
                    "--model".into(),
                    model_dir.clone().into_os_string(),
                    "--corpus".into(),
                    OsString::from(&corpus),
                    "--output".into(),
                    calib_out.clone().into_os_string(),
                ],
                "cargo build --release -p hipfire-coexistence",
            )?;
            calib = Some(calib_out);
        }
    }

    // 4. Output location + name.
    let default_dir = hipfire_config::configured_models_dir(&loaded.config);
    let out_dir = prompt("Output directory", Some(&default_dir.to_string_lossy()))?;
    let default_name = derive_output_name(&source, &format);
    let out_name = prompt("Output filename", Some(&default_name))?;
    let out_path = PathBuf::from(out_dir).join(out_name);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // 5. Quantize.
    let quantize = sibling_binary("hipfire-quantize", "HIPFIRE_QUANTIZE_BIN")
        .context("hipfire-quantize not found (cargo build --release -p hipfire-quantize)")?;
    let mut qargs: Vec<OsString> = vec![
        "--input".into(),
        model_dir.into_os_string(),
        "--format".into(),
        OsString::from(&format),
        "--output".into(),
        out_path.clone().into_os_string(),
    ];
    if let Some(c) = &calib {
        qargs.push("--hessian".into());
        qargs.push(c.clone().into_os_string());
        if format_uses_ldlq(&format) {
            qargs.push("--ldlq".into());
        }
    }
    run_tool(
        &quantize,
        &qargs,
        "cargo build --release -p hipfire-quantize",
    )?;
    println!("• wrote {}", out_path.display());

    // 6. Sidecars → compose (reuse `hipfire model compose` via the current exe).
    let sidecars = collect_sidecars()?;
    if sidecars.is_empty() {
        println!(
            "\n(no sidecars folded in.\n \
             Build a DFlash drafter later with `dflash_convert --input <dir> --output <name>.dflash.hfq`,\n \
             then `hipfire model compose {} <name>.dflash.hfq`.)",
            out_path.display()
        );
    } else if prompt_yes_no("Fold these sidecars into the artifact now?", true)? {
        let self_exe = std::env::current_exe().context("resolve hipfire executable")?;
        let mut cargs: Vec<OsString> = vec![
            "model".into(),
            "compose".into(),
            out_path.clone().into_os_string(),
        ];
        cargs.extend(sidecars.iter().map(|p| p.clone().into_os_string()));
        run_tool(&self_exe, &cargs, "cargo build --release -p hipfire-cli")?;
    }

    println!("\n✓ induct complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_gating_matches_quant_tokens() {
        assert!(format_needs_calibration("oq4++"));
        assert!(format_needs_calibration("mq4+"));
        assert!(!format_needs_calibration("oq4"));
        assert!(!format_needs_calibration("mq4"));
        assert!(!format_needs_calibration("qtip3"));
        assert!(!format_needs_calibration("bf16"));

        assert!(format_uses_ldlq("oq4++"));
        assert!(format_uses_ldlq("oq8++"));
        assert!(!format_uses_ldlq("mq4+")); // single + is not error-feedback
        assert!(!format_uses_ldlq("oq4"));
    }

    #[test]
    fn hf_cache_key_matches_hub_layout() {
        assert_eq!(
            hf_cache_key("Qwen/Qwen3.5-0.8B"),
            "models--Qwen--Qwen3.5-0.8B"
        );
        assert_eq!(hf_cache_key("gpt2"), "models--gpt2");
    }

    #[test]
    fn output_name_and_stem() {
        assert_eq!(source_stem("Qwen/Qwen3.5-0.8B"), "Qwen3.5-0.8B");
        assert_eq!(source_stem("/srv/models/foo/"), "foo");
        assert_eq!(
            derive_output_name("Qwen/Qwen3.5-0.8B", "oq4++"),
            "Qwen3.5-0.8B.oq4++.hfq"
        );
    }

    #[test]
    fn hf_file_filter_keeps_config_weights_tokenizer_only() {
        assert!(want_hf_file("model.safetensors"));
        assert!(want_hf_file("model-00001-of-00002.safetensors"));
        assert!(want_hf_file("config.json"));
        assert!(want_hf_file("tokenizer.model"));
        assert!(want_hf_file("tokenizer_config.json"));
        assert!(!want_hf_file("pytorch_model.bin"));
        assert!(!want_hf_file("model.gguf"));
        assert!(!want_hf_file("README.md"));
    }
}
