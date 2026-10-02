// SPDX-License-Identifier: Apache-2.0
// hipfire — `hfq`: HFQ container manipulation tool.
//
//! Inspect and edit HFQ containers (models, calibration-artifact sidecars,
//! `.mtp`/`.dflash`/`.triattn` sidecars). With these, "bundle everything vs keep
//! separate" is a runtime choice — split a Hessian out of a bundle, merge
//! sidecars, or embed a chat template into a model — not an upfront commitment.
//!
//! Subcommands:
//!   hfq list   <file>                          — arch, metadata keys, tensor table
//!   hfq extract <in> <out> --tensor <pat> ...  — copy matching tensors to a new HFQ
//!                                                 (pat: exact, `prefix*`, `*suffix`, `*sub*`)
//!   hfq meta-set <in> <out> --key <k> (--value <v> | --value-file <f>) [--json]
//!                                              — set a metadata JSON key (all tensors
//!                                                copied), e.g. embed a jinja2 template
//!   hfq meta-get <file> [--key <k>]            — dump metadata JSON (or one key)
//!   hfq rearch <in> <out> --arch-id <id>       — rewrite HFQM header arch_id and
//!                                                numeric metadata.arch_id
//!
//! Examples:
//!   hfq extract model.calib.hfq just.hessian.hfq --tensor '*.hessian'
//!   hfq meta-set model.hfq model+tmpl.hfq --key chat_template --value-file tmpl.jinja

use crate::hfq::HfqFile;
use std::io::Write;
use std::path::Path;

const HFQM_MAGIC: &[u8; 4] = b"HFQM";
const HFQM_VERSION: u32 = 1;

/// Owned tensor (name, quant_type, shape, group_size, data) for writing.
struct Tensor {
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
}

/// Drop metadata that describes the SOURCE file's byte layout.
///
/// `hfqm_modules` is a table of ABSOLUTE byte ranges into the file it was built
/// from. A subset extract keeps a handful of tensors and renumbers everything,
/// so every range in that table becomes wrong — and `HfqFile::open` validates
/// them, so the extract is unreadable by every other tool:
///
///   HFQM module layers.0.experts.0 invalid range 2038587392..2040263680
///   for file_len 22921216
///
/// Rebasing is not the fix: the modules describe pageable expert groups whose
/// tensors are mostly NOT in the extract, so there is nothing coherent to point
/// them at. A tensor subset is not a pageable artifact — the table should simply
/// not be there.
fn strip_layout_metadata(meta_json: &str) -> String {
    let Ok(mut meta) = serde_json::from_str::<serde_json::Value>(meta_json) else {
        return meta_json.to_string();
    };
    let removed = meta
        .as_object_mut()
        .and_then(|obj| obj.remove("hfqm_modules"))
        .is_some();
    if removed {
        eprintln!("  (dropped hfqm_modules: byte ranges do not survive a subset extract)");
    }
    serde_json::to_string(&meta).unwrap_or_else(|_| meta_json.to_string())
}

fn write_hfq(
    path: &str,
    arch: u32,
    metadata_json: &str,
    tensors: &[Tensor],
) -> std::io::Result<()> {
    let meta = metadata_json.as_bytes();
    let metadata_offset = 32u64;
    let index_offset = metadata_offset + meta.len() as u64;
    let mut index = Vec::new();
    index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        let nb = t.name.as_bytes();
        index.extend_from_slice(&(nb.len() as u16).to_le_bytes());
        index.extend_from_slice(nb);
        index.push(t.quant_type);
        index.push(t.shape.len() as u8);
        for &d in &t.shape {
            index.extend_from_slice(&d.to_le_bytes());
        }
        index.extend_from_slice(&t.group_size.to_le_bytes());
        index.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
    }
    let data_start = index_offset + index.len() as u64;
    let data_offset = (data_start + 4095) & !4095;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(HFQM_MAGIC)?;
    f.write_all(&HFQM_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;
    f.write_all(meta)?;
    f.write_all(&index)?;
    f.write_all(&vec![0u8; (data_offset - data_start) as usize])?;
    for t in tensors {
        f.write_all(&t.data)?;
    }
    f.flush()
}

/// Simple glob: exact, `prefix*`, `*suffix`, `*sub*`, or `*` (all).
fn matches(pat: &str, name: &str) -> bool {
    if pat == "*" {
        return true;
    }
    match (pat.starts_with('*'), pat.ends_with('*')) {
        (true, true) => name.contains(pat.trim_matches('*')),
        (true, false) => name.ends_with(pat.trim_start_matches('*')),
        (false, true) => name.starts_with(pat.trim_end_matches('*')),
        (false, false) => name == pat,
    }
}

fn flag<'a>(args: &'a [String], f: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == f)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}
fn flags_all(args: &[String], f: &str) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..args.len() {
        if args[i] == f {
            if let Some(v) = args.get(i + 1) {
                out.push(v.clone());
            }
        }
    }
    out
}

fn load_all(path: &str) -> (u32, String, Vec<Tensor>) {
    let hfq = HfqFile::open(Path::new(path)).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let arch = hfq.arch_id;
    let meta = hfq.metadata_json.clone();
    let names: Vec<(String, u8, Vec<u32>, u32)> = hfq
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.quant_type, t.shape.clone(), t.group_size))
        .collect();
    let tensors = names
        .into_iter()
        .map(|(name, qt, shape, gs)| {
            let (_, data) = hfq.tensor_data_cow(&name).expect("tensor data");
            Tensor {
                name,
                quant_type: qt,
                shape,
                group_size: gs,
                data: data.to_vec(),
            }
        })
        .collect();
    (arch, meta, tensors)
}

/// Entry point for the standalone `hfq` binary.
/// A filename for a tensor that is stable, ordered, and cannot collide.
///
/// The index prefix preserves container order (which `implode` must reproduce —
/// the index and the payload are written in the same order) and makes two
/// tensors that sanitise to the same string distinct anyway. Anything outside
/// `[A-Za-z0-9._-]` becomes `_`, so a name carrying a path separator cannot
/// escape the directory.
fn tensor_filename(i: usize, name: &str) -> String {
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{i:04}.{safe}.bin")
}

/// `explode`: one file per tensor plus a manifest, so the payload can be
/// inspected or edited with ordinary tools and put back with `implode`.
///
/// The manifest keeps the FULL source metadata, including any `hfqm_modules`
/// layout table. `implode` is what drops that (see `strip_layout_metadata`),
/// because it rewrites the payload at its own offsets and every absolute range
/// in that table would then point at the wrong bytes.
fn cmd_explode(input: &str, dir: &str) {
    let (arch, meta, tensors) = load_all(input);
    let root = Path::new(dir);
    let tdir = root.join("tensors");
    std::fs::create_dir_all(&tdir).unwrap_or_else(|e| panic!("create {}: {e}", tdir.display()));
    let mut entries = Vec::with_capacity(tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        let fname = tensor_filename(i, &t.name);
        std::fs::write(tdir.join(&fname), &t.data).unwrap_or_else(|e| panic!("write {fname}: {e}"));
        entries.push(serde_json::json!({
            "name": t.name,
            "quant_type": t.quant_type,
            "shape": t.shape,
            "group_size": t.group_size,
            "bytes": t.data.len(),
            "file": format!("tensors/{fname}"),
        }));
    }
    let manifest = serde_json::json!({
        "hfq_explode_version": 1,
        "arch_id": arch,
        "metadata_json": meta,
        "tensors": entries,
    });
    let mpath = root.join("manifest.json");
    std::fs::write(
        &mpath,
        serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", mpath.display()));
    println!(
        "exploded {} tensors (arch {arch}) to {}",
        tensors.len(),
        root.display()
    );
}

/// `implode`: rebuild a container from an `explode` directory.
///
/// Tensor order, names, quant types, shapes and group sizes come from the
/// manifest; only the payload bytes come from the files, so editing a tensor in
/// place round-trips as long as its length is unchanged. A length that no longer
/// matches the manifest is refused rather than written — the index would say one
/// thing and the payload another, and nothing downstream would notice.
fn cmd_implode(dir: &str, output: &str) {
    let root = Path::new(dir);
    let raw = std::fs::read(root.join("manifest.json"))
        .unwrap_or_else(|e| panic!("read {}/manifest.json: {e}", root.display()));
    let m: serde_json::Value = serde_json::from_slice(&raw).expect("parse manifest.json");
    let arch = m["arch_id"].as_u64().expect("manifest arch_id") as u32;
    let meta = m["metadata_json"].as_str().unwrap_or("{}").to_string();
    let list = m["tensors"].as_array().expect("manifest tensors");
    let mut tensors = Vec::with_capacity(list.len());
    for e in list {
        let name = e["name"].as_str().expect("tensor name").to_string();
        let file = e["file"].as_str().expect("tensor file");
        let data =
            std::fs::read(root.join(file)).unwrap_or_else(|err| panic!("read {file}: {err}"));
        if let Some(expected) = e["bytes"].as_u64() {
            assert_eq!(
                data.len() as u64,
                expected,
                "{name}: {file} is {} bytes, manifest says {expected} — the index would \
                 disagree with the payload",
                data.len()
            );
        }
        tensors.push(Tensor {
            name,
            quant_type: e["quant_type"].as_u64().expect("quant_type") as u8,
            shape: e["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|d| d.as_u64().expect("shape dim") as u32)
                .collect(),
            group_size: e["group_size"].as_u64().expect("group_size") as u32,
            data,
        });
    }
    // Same reason `extract` strips it: the payload is rewritten at this writer's
    // offsets, so a layout table of absolute ranges from the source file is wrong.
    let meta = strip_layout_metadata(&meta);
    write_hfq(output, arch, &meta, &tensors).unwrap_or_else(|e| panic!("write {output}: {e}"));
    println!(
        "imploded {} tensors (arch {arch}) into {output}",
        tensors.len()
    );
}

pub fn main() {
    main_with_args(&std::env::args().collect::<Vec<_>>());
}

/// Entry point for `hipfire hfq`, which must supply argv.
///
/// The subcommand selector is `argv[1]` and every operand is read by absolute
/// index, so the real process argv (`hipfire hfq list x.hfq`) would select on
/// "hfq" and fall through to usage. Callers pass the argv this would have had
/// as its own binary.
pub fn main_with_args(argv: &[String]) {
    let argv: Vec<String> = argv.to_vec();
    let cmd = argv.get(1).map(|s| s.as_str()).unwrap_or("");
    match cmd {
        // Decode every tensor and report the ones that fail, without retaining
        // any of them. `extract` cannot answer this: `load_all` materialises the
        // WHOLE file before filtering, so on a large artifact it either exhausts
        // memory or panics anonymously on the first bad tensor — which is how a
        // 122B decode failure showed up as `.expect("tensor data")` with no name.
        "verify" => {
            let path = argv.get(2).expect("usage: hfq verify <file>");
            let hfq = HfqFile::open(Path::new(path)).expect("open");
            let names: Vec<(String, u8, usize)> = hfq
                .tensors()
                .iter()
                .map(|t| (t.name.clone(), t.quant_type, t.data_size))
                .collect();
            println!("verifying {} tensors in {path}", names.len());
            let mut bad = 0usize;
            for (name, qt, size) in &names {
                match hfq.tensor_data_cow(name) {
                    Some((_, bytes)) => {
                        if bytes.len() != *size {
                            println!(
                                "  SHORT {name} (qt {qt}): {} bytes, index says {size}",
                                bytes.len()
                            );
                            bad += 1;
                        }
                    }
                    None => {
                        println!("  FAIL  {name} (qt {qt}, {size} bytes): decode returned None");
                        bad += 1;
                    }
                }
            }
            if bad == 0 {
                println!("all {} tensors decode", names.len());
            } else {
                println!("{bad} tensor(s) failed");
                std::process::exit(1);
            }
        }
        "explode" => {
            let input = argv.get(2).expect("usage: hfq explode <in.hfq> <dir>");
            let dir = argv.get(3).expect("usage: hfq explode <in.hfq> <dir>");
            cmd_explode(input, dir);
        }
        "implode" => {
            let dir = argv.get(2).expect("usage: hfq implode <dir> <out.hfq>");
            let output = argv.get(3).expect("usage: hfq implode <dir> <out.hfq>");
            cmd_implode(dir, output);
        }
        "list" => {
            let path = argv.get(2).expect("usage: hfq list <file>");
            let hfq = HfqFile::open(Path::new(path)).expect("open");
            println!("arch_id: {}", hfq.arch_id);
            let meta: serde_json::Value =
                serde_json::from_str(&hfq.metadata_json).unwrap_or(serde_json::json!({}));
            if let Some(o) = meta.as_object() {
                let mut keys: Vec<&String> = o.keys().collect();
                keys.sort();
                println!(
                    "metadata keys: {}",
                    keys.iter()
                        .map(|k| k.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            let ts = hfq.tensors();
            println!("tensors: {}", ts.len());
            let mut total = 0u64;
            let mut stored_total = 0u64;
            let mut recoded = 0usize;
            for t in ts {
                // The index reports the LOGICAL encoding — a losslessly-recoded
                // tensor is presented as the type it expands to. An inspection
                // tool must show what is actually on disk, or a compressed
                // artifact is indistinguishable from a plain one.
                let (stored_qt, stored_len) = hfq
                    .stored_encoding(&t.name)
                    .unwrap_or((t.quant_type, t.data_size));
                total += t.data_size as u64;
                stored_total += stored_len as u64;
                let qt = if stored_qt == t.quant_type {
                    format!("{}", t.quant_type)
                } else {
                    recoded += 1;
                    format!("{stored_qt}->{}", t.quant_type)
                };
                let size = if stored_len == t.data_size {
                    format!("{:.2} MB", t.data_size as f64 / 1e6)
                } else {
                    format!(
                        "{:.2} MB on disk (-> {:.2} MB)",
                        stored_len as f64 / 1e6,
                        t.data_size as f64 / 1e6
                    )
                };
                println!(
                    "  {:60} qt={:<6} shape={:?} g={} {size}",
                    t.name, qt, t.shape, t.group_size
                );
            }
            if recoded > 0 {
                println!(
                    "  ({recoded} tensor(s) losslessly recoded on disk: {:.2} MB stored, \
                     {:.2} MB expanded, {:.4}x; `qt=STORED->LOGICAL`)",
                    stored_total as f64 / 1e6,
                    total as f64 / 1e6,
                    total as f64 / stored_total.max(1) as f64
                );
            }
            // Report the on-disk total too, or a recoded artifact appears to
            // occupy its expanded size — which is not what the file costs.
            if stored_total == total {
                println!("total tensor bytes: {:.2} MB", total as f64 / 1e6);
            } else {
                println!(
                    "total tensor bytes: {:.2} MB on disk ({:.2} MB expanded)",
                    stored_total as f64 / 1e6,
                    total as f64 / 1e6
                );
            }
        }
        "extract" => {
            let inp = argv
                .get(2)
                .expect("usage: hfq extract <in> <out> --tensor <pat>...");
            let out = argv.get(3).expect("out path");
            let pats = flags_all(&argv, "--tensor");
            assert!(!pats.is_empty(), "need at least one --tensor <pattern>");
            let (arch, meta, tensors) = load_all(inp);
            let kept: Vec<Tensor> = tensors
                .into_iter()
                .filter(|t| pats.iter().any(|p| matches(p, &t.name)))
                .collect();
            assert!(!kept.is_empty(), "no tensors matched {pats:?}");
            eprintln!("extracting {} tensor(s) → {out}", kept.len());
            for t in &kept {
                eprintln!("  {}", t.name);
            }
            write_hfq(out, arch, &strip_layout_metadata(&meta), &kept).expect("write");
        }
        "meta-set" => {
            let inp = argv
                .get(2)
                .expect("usage: hfq meta-set <in> <out> --key <k> --value[-file] <v>");
            let out = argv.get(3).expect("out path");
            let key = flag(&argv, "--key").expect("--key required");
            let value: String = if let Some(vf) = flag(&argv, "--value-file") {
                std::fs::read_to_string(vf).expect("read value-file")
            } else {
                flag(&argv, "--value")
                    .expect("--value or --value-file required")
                    .to_string()
            };
            let (arch, meta_json, tensors) = load_all(inp);
            let mut meta: serde_json::Value =
                serde_json::from_str(&meta_json).unwrap_or(serde_json::json!({}));
            // `--json` stores the value as PARSED JSON rather than a string.
            //
            // The default is a string because the documented use is a jinja chat
            // template, which is one. But some metadata is structured — the
            // top-level `dflash` block a drafter needs is an OBJECT, and
            // `DflashConfig::from_hfq` does `df.get("num_hidden_layers")` on it.
            // Stored as a string that call returns `None` and the whole parse
            // fails with no indication which field was at fault.
            //
            // Opt-in rather than "parse if it looks like JSON": a template that
            // happens to be valid JSON must not silently change type.
            let encoded = if argv.iter().any(|a| a == "--json") {
                serde_json::from_str::<serde_json::Value>(&value)
                    .expect("--json given but the value is not valid JSON")
            } else {
                serde_json::Value::String(value)
            };
            meta.as_object_mut()
                .expect("metadata not an object")
                .insert(key.to_string(), encoded);
            eprintln!(
                "set metadata[{key}] ({} tensors copied) → {out}",
                tensors.len()
            );
            write_hfq(out, arch, &serde_json::to_string(&meta).unwrap(), &tensors).expect("write");
        }
        "meta-get" => {
            let path = argv.get(2).expect("usage: hfq meta-get <file> [--key <k>]");
            let hfq = HfqFile::open(Path::new(path)).expect("open");
            if let Some(k) = flag(&argv, "--key") {
                let meta: serde_json::Value =
                    serde_json::from_str(&hfq.metadata_json).unwrap_or(serde_json::json!({}));
                match meta.get(k) {
                    Some(v) => println!("{}", serde_json::to_string_pretty(v).unwrap()),
                    None => {
                        eprintln!("key {k} not found");
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", hfq.metadata_json);
            }
        }
        "rearch" => {
            let inp = argv
                .get(2)
                .expect("usage: hfq rearch <in> <out> --arch-id <id>");
            let out = argv.get(3).expect("out path");
            let arch: u32 = flag(&argv, "--arch-id")
                .expect("--arch-id required")
                .parse()
                .expect("--arch-id must be a u32");
            let (_old_arch, meta_json, tensors) = load_all(inp);
            let mut meta: serde_json::Value =
                serde_json::from_str(&meta_json).unwrap_or(serde_json::json!({}));
            let obj = meta.as_object_mut().expect("metadata not an object");
            obj.insert(
                "arch_id".to_string(),
                serde_json::Value::Number(serde_json::Number::from(arch)),
            );
            obj.remove("arch_id_semantics");
            eprintln!(
                "set header arch_id and metadata.arch_id to {arch} ({} tensors copied) → {out}",
                tensors.len()
            );
            write_hfq(out, arch, &serde_json::to_string(&meta).unwrap(), &tensors).expect("write");
        }
        _ => {
            eprintln!(
                "hfq — HFQ container tool\n\
                 usage:\n  hfq list <file>
  hfq verify <file>\n  hfq extract <in> <out> --tensor <pat>...\n\
                 \x20 hfq meta-set <in> <out> --key <k> (--value <v> | --value-file <f>) [--json]\n\
                 \x20 hfq meta-get <file> [--key <k>]\n\
                 \x20 hfq rearch <in> <out> --arch-id <id>\n\
                 \x20 hfq explode <in.hfq> <dir>     one file per tensor + manifest.json\n\
                 \x20 hfq implode <dir> <out.hfq>    rebuild a container from that directory"
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("hfq-explode-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// explode -> implode must preserve the payload AND the index, in order.
    /// The index is the half that fails silently: a wrong quant_type or group
    /// size still writes a container that opens, and only decodes to garbage.
    #[test]
    fn explode_implode_round_trips_payload_and_index() {
        let dir = scratch("roundtrip");
        let src = dir.join("src.hfq");
        let tensors = vec![
            Tensor {
                name: "model.embed_tokens.weight".to_string(),
                quant_type: 7,
                shape: vec![4, 8],
                group_size: 32,
                data: (0u8..32).collect(),
            },
            // A name with characters that must not reach the filesystem verbatim.
            Tensor {
                name: "model/layers.0..weight".to_string(),
                quant_type: 1,
                shape: vec![2],
                group_size: 0,
                data: vec![9, 9, 9, 9],
            },
        ];
        write_hfq(src.to_str().unwrap(), 5, r#"{"k":"v"}"#, &tensors).unwrap();

        let ex = dir.join("ex");
        cmd_explode(src.to_str().unwrap(), ex.to_str().unwrap());
        // The separator became `_`, so nothing escaped `tensors/`.
        assert!(ex.join("tensors/0001.model_layers.0..weight.bin").exists());

        let out = dir.join("out.hfq");
        cmd_implode(ex.to_str().unwrap(), out.to_str().unwrap());

        let (arch, _meta, back) = load_all(out.to_str().unwrap());
        assert_eq!(arch, 5);
        assert_eq!(back.len(), tensors.len());
        for (a, b) in tensors.iter().zip(back.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.quant_type, b.quant_type);
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.group_size, b.group_size);
            assert_eq!(a.data, b.data);
        }
    }

    /// An edited tensor of the wrong length is refused, not written: the index
    /// would claim one size and the payload hold another, and no reader checks.
    #[test]
    #[should_panic(expected = "manifest says")]
    fn implode_refuses_a_tensor_whose_length_changed() {
        let dir = scratch("badlen");
        let src = dir.join("src.hfq");
        write_hfq(
            src.to_str().unwrap(),
            5,
            "{}",
            &[Tensor {
                name: "w".to_string(),
                quant_type: 1,
                shape: vec![4],
                group_size: 0,
                data: vec![1, 2, 3, 4],
            }],
        )
        .unwrap();
        let ex = dir.join("ex");
        cmd_explode(src.to_str().unwrap(), ex.to_str().unwrap());
        std::fs::write(ex.join("tensors/0000.w.bin"), [1u8, 2, 3]).unwrap();
        cmd_implode(ex.to_str().unwrap(), dir.join("out.hfq").to_str().unwrap());
    }
}
