#![allow(
    clippy::duplicated_attributes,
    clippy::doc_lazy_continuation,
    clippy::doc_overindented_list_items,
    clippy::explicit_counter_loop,
    clippy::field_reassign_with_default,
    clippy::manual_checked_ops,
    clippy::manual_clamp,
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::ptr_arg,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::useless_vec,
    clippy::while_let_loop
)]
// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.

//! optimize — repackage a canonical OQ4 `.hfq` into the arch-optimal weight
//! layout ahead of load time (the `hipfire optimize` runner).
//!
//! Canonical OQ4 stores each weight as quant_type 34 (`[f16 scale][128 nibbles]`
//! per 256-group, row-contiguous) — the portable, arch-independent form that
//! ships everywhere. At load the qwen3.5 loader repacks each such tensor into the
//! arch *combined* device layout (split nibbles + split scales + interleaved
//! decode records). This tool performs that transform ONCE, writing a derived
//! `.hfq` whose OQ4 tensors are stored pre-packed as quant_type 37; the loader
//! then uploads them verbatim with no per-load repack.
//!
//! Design (matches the "general on disk, arch-optimal in use" split):
//!   * The canonical file is the source of truth and is never modified. The
//!     derived file is a machine/arch artifact.
//!   * The quant_type code IS the layout version. A future change to the combined
//!     layout takes a NEW code, so a stale derived artifact is refused at load
//!     (honest capability gap) rather than read as garbage.
//!   * Loading is UMA-safe with no `mmap` double-copy: the qwen3.5 loader reads
//!     tensor data via `pread` (the index-only mmap is dropped after parsing,
//!     `FADV_DONTNEED`), so the pre-packed bytes stream straight into the single
//!     device allocation — no page-cache shadow copy in the shared pool.
//!
//! The transform itself is `hipfire_runtime::oq4_arch::oq4_pack_arch_combined`,
//! the SAME function the loader's qt=34 (OQ4) path calls, so the tool and the
//! loader can never drift.
//!
//! Usage:
//!   optimize <in.hfq> [-o <out.hfq>] [--arch <gfx>]
//!
//! With no `-o`, the output is `<stem>.<arch>.hfq` beside the input.

use std::path::PathBuf;

use hipfire_quant_format::QuantType;
use hipfire_runtime::hfq::{
    oq4_pack_arch_combined, plan_hfqm_layout, write_hfqm_package_mem, HfqFile, HfqMemTensor,
    HfqStreamEntry, OQ4_ARCH_PACKED_QT, OQ4_CANONICAL_QT,
};
use hipfire_runtime::hfq_modules::{
    module_table_json, HfqModuleKind, HfqModuleRecord, HFQM_MODULE_TABLE_KEY,
};
use hipfire_runtime::oq_moe::oq4_canonical_to_moe_blocks;

/// Best-effort live GPU arch probe (e.g. "gfx1103"). Read-only
/// `hipGetDeviceProperties` on device 0 — no GPU lock, no compute context, so it
/// coexists with a running daemon. Returns `None` if HIP can't be loaded or no
/// device is present (then the caller requires an explicit `--arch`).
fn probe_gpu_arch() -> Option<String> {
    let hip = hip_bridge::HipRuntime::load().ok()?;
    let arch = hip.get_arch(0).ok()?;
    if arch.starts_with("gfx") {
        Some(arch)
    } else {
        None
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut arch: Option<String> = None;
    let mut moe_blocks = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => {
                i += 1;
                output = Some(PathBuf::from(args.get(i).expect("-o needs a path")));
            }
            "--arch" => {
                i += 1;
                arch = Some(args.get(i).expect("--arch needs a value").clone());
            }
            "--moe-blocks" => moe_blocks = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: optimize <in.hfq> [-o <out.hfq>] [--arch <gfx>] [--moe-blocks]\n\
                     \n  \
                     default      dense arch-pack: OQ4 tensors qt 34 -> 37, uploaded verbatim\n  \
                     --moe-blocks paged pre-pack:  routed-expert OQ4 qt 34 -> 53, fetched\n               \
                     verbatim by the weight pager instead of being repacked\n               \
                     on the CPU per page-in"
                );
                return;
            }
            other => {
                if input.is_none() {
                    input = Some(PathBuf::from(other));
                } else {
                    eprintln!("unexpected argument: {other}");
                    std::process::exit(2);
                }
            }
        }
        i += 1;
    }
    let input = input.unwrap_or_else(|| {
        eprintln!("usage: optimize <in.hfq> [-o <out.hfq>] [--arch <gfx>]");
        std::process::exit(2);
    });

    // Resolve the target arch: explicit --arch wins; otherwise probe the live
    // GPU (a read-only hipGetDeviceProperties — no GPU lock, no compute context).
    // This is just a device-name query, so it coexists with a running daemon.
    let arch = arch.unwrap_or_else(|| match probe_gpu_arch() {
        Some(a) => {
            eprintln!("detected GPU arch: {a} (override with --arch)");
            a
        }
        None => {
            eprintln!("could not detect a GPU arch; pass --arch <gfx> (e.g. --arch gfx1103)");
            std::process::exit(2);
        }
    });

    // Default output is arch-tagged beside the input: `<stem>.<arch>.hfq`
    // (e.g. qwen3.5-0.8b-oq4+.hfq -> qwen3.5-0.8b-oq4+.gfx1103.hfq). file_stem
    // strips only the trailing `.hfq`, so dotted model versions survive.
    let output = output.unwrap_or_else(|| {
        let stem = input
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "model".into());
        input.with_file_name(format!("{stem}.{arch}.hfq"))
    });

    // Index-only open: parse header/index without mmapping (let alone paging in)
    // the tensor payload — repack tooling must inspect arbitrarily large models
    // without a full-file copy. Per-tensor bytes are then read on demand below.
    let hfq = HfqFile::open(&input).unwrap_or_else(|e| {
        eprintln!("open {}: {e}", input.display());
        std::process::exit(1);
    });

    // v1 writes the whole derived package in memory (one Vec per tensor). Bundled
    // multi-module packages (MTP/DFlash sidecars) are not round-tripped by the
    // in-memory writer yet — refuse rather than silently drop them.
    // Routed-expert modules are the PAGED contract; arch-packing them would make
    // them unpageable (`WeightPager::register_expert_module` refuses qt 37 for
    // having no per-expert addressing). So a module-bearing artifact needs
    // `--moe-blocks`, which pre-applies the transform the pager would otherwise
    // run on the CPU per page-in.
    let routed_modules = hfq
        .modules()
        .iter()
        .filter(|m| m.kind == HfqModuleKind::RoutedExpert)
        .count();
    if !hfq.modules().is_empty() && !moe_blocks {
        eprintln!(
            "refusing: {} carries {} routed-expert module(s) of {} total.\n  \
             Arch-packing those would strip their per-expert addressing and the weight \
             pager would refuse the result.\n  Pass --moe-blocks to pre-pack them into \
             the MoE-block layout instead (page-in becomes a verbatim fetch).",
            input.display(),
            routed_modules,
            hfq.modules().len()
        );
        std::process::exit(1);
    }
    if moe_blocks && routed_modules == 0 {
        eprintln!(
            "refusing: --moe-blocks given but {} has no routed-expert modules",
            input.display()
        );
        std::process::exit(1);
    }

    // Informational for now: the combined layout is identical across current
    // RDNA/CDNA arches, so quant_type 37 is portable among them — the arch only
    // tags the output name. The flag/probe is the hook for future per-arch
    // layouts (group size / interleave width), at which point a distinct arch
    // will take a distinct quant_type code (and a header arch-gate at load).
    if moe_blocks {
        eprintln!(
            "packing routed experts into the MoE-block layout (qt {OQ4_CANONICAL_QT} -> {})",
            QuantType::Oq4G256MoeBlocks.code()
        );
    } else {
        eprintln!("packing for {arch} (combined layout v1, qt {OQ4_ARCH_PACKED_QT})");
    }

    // Names of every tensor that belongs to a routed-expert module — the only
    // ones the paged path addresses individually.
    let routed_tensor_names: std::collections::HashSet<String> = hfq
        .modules()
        .iter()
        .filter(|m| m.kind == HfqModuleKind::RoutedExpert)
        .flat_map(|m| m.tensors.iter().map(|t| t.name.clone()))
        .collect();

    let infos: Vec<_> = hfq.tensors().to_vec();
    let mut out_tensors: Vec<HfqMemTensor> = Vec::with_capacity(infos.len());
    let mut repacked = 0usize;
    let mut copied = 0usize;
    let mut canon_bytes = 0u64;
    let mut packed_bytes = 0u64;

    for info in &infos {
        let (_, data) = hfq
            .tensor_data_vec(&info.name)
            .unwrap_or_else(|| panic!("tensor data not found: {}", info.name));

        if moe_blocks {
            // Only routed-expert tensors are pre-packed; everything else is copied
            // byte-for-byte. The module table below is rebuilt against the new
            // layout, which is the part that makes the derived file loadable.
            if info.quant_type == OQ4_CANONICAL_QT && routed_tensor_names.contains(&info.name) {
                assert_eq!(
                    info.shape.len(),
                    2,
                    "OQ4 routed tensor {} expected 2D [m,k], got {:?}",
                    info.name,
                    info.shape
                );
                let m = info.shape[0] as usize;
                let k = info.shape[1] as usize;
                let packed = oq4_canonical_to_moe_blocks(&data, m, k).unwrap_or_else(|e| {
                    eprintln!("moe-block pack {}: {e}", info.name);
                    std::process::exit(1);
                });
                canon_bytes += data.len() as u64;
                packed_bytes += packed.len() as u64;
                out_tensors.push(HfqMemTensor {
                    name: info.name.clone(),
                    quant_type: QuantType::Oq4G256MoeBlocks.code(),
                    shape: info.shape.clone(),
                    group_size: info.group_size,
                    data: packed,
                });
                repacked += 1;
            } else {
                out_tensors.push(HfqMemTensor {
                    name: info.name.clone(),
                    quant_type: info.quant_type,
                    shape: info.shape.clone(),
                    group_size: info.group_size,
                    data,
                });
                copied += 1;
            }
        } else if info.quant_type == OQ4_CANONICAL_QT {
            // OQ4 weight matrices are stored [m, k] = [out, in].
            assert_eq!(
                info.shape.len(),
                2,
                "OQ4 tensor {} expected 2D [m,k], got shape {:?}",
                info.name,
                info.shape
            );
            let m = info.shape[0] as usize;
            let k = info.shape[1] as usize;
            let combined = oq4_pack_arch_combined(&data, m, k);
            canon_bytes += data.len() as u64;
            packed_bytes += combined.len() as u64;
            out_tensors.push(HfqMemTensor {
                name: info.name.clone(),
                quant_type: OQ4_ARCH_PACKED_QT,
                shape: info.shape.clone(),
                group_size: info.group_size,
                data: combined,
            });
            repacked += 1;
        } else {
            out_tensors.push(HfqMemTensor {
                name: info.name.clone(),
                quant_type: info.quant_type,
                shape: info.shape.clone(),
                group_size: info.group_size,
                data,
            });
            copied += 1;
        }
    }

    if repacked == 0 {
        eprintln!(
            "warning: no OQ4 (quant_type {OQ4_CANONICAL_QT}) tensors found in {} — nothing to repack",
            input.display()
        );
    }

    // Rewriting payloads moves every tensor, so a module-bearing artifact needs
    // its `hfqm_modules` table recomputed — records carry ABSOLUTE file offsets
    // (`weight_pager` hands `data_offset` straight to `read_host`/`fetch`).
    // Passing the source metadata through unchanged, as this tool used to, would
    // produce a file whose module table points into the OLD layout.
    let metadata_json = if moe_blocks {
        rebuild_metadata_with_modules(&hfq, &out_tensors).unwrap_or_else(|e| {
            eprintln!("rebuild module table: {e}");
            std::process::exit(1);
        })
    } else {
        hfq.metadata_json.clone()
    };
    let metadata_json = strip_stale_container_keys(&metadata_json).unwrap_or_else(|e| {
        eprintln!("sanitize metadata: {e}");
        std::process::exit(1);
    });

    write_hfqm_package_mem(&output, hfq.arch_id, &metadata_json, &out_tensors).unwrap_or_else(
        |e| {
            eprintln!("write {}: {e}", output.display());
            std::process::exit(1);
        },
    );

    let (to_qt, layout_note) = if moe_blocks {
        (
            QuantType::Oq4G256MoeBlocks.code(),
            "MoE-block layout, fetched verbatim by the weight pager",
        )
    } else {
        (
            OQ4_ARCH_PACKED_QT,
            "combined layout incl. interleaved decode region",
        )
    };
    eprintln!(
            "optimize: {} -> {}\n  repacked {repacked} OQ4 tensor(s) (qt {OQ4_CANONICAL_QT} -> {to_qt}), copied {copied} other(s)\n  OQ4 weight bytes {canon_bytes} -> {packed_bytes} on disk ({layout_note})",
        input.display(),
        output.display()
    );
}

/// Recompute `hfqm_modules` against the layout the rewritten tensors will get.
///
/// The layout depends on the metadata length, and the metadata contains the
/// offsets the layout produces — a circularity resolved by iterating to a fixed
/// point. Only decimal digit widths move between rounds, so it settles in two or
/// three; `hipfire-quantize`'s own writer does the same thing.
fn rebuild_metadata_with_modules(
    hfq: &HfqFile,
    out_tensors: &[HfqMemTensor],
) -> Result<String, String> {
    let entries: Vec<HfqStreamEntry> = out_tensors
        .iter()
        .map(|t| HfqStreamEntry {
            name: t.name.clone(),
            quant_type: t.quant_type,
            shape: t.shape.clone(),
            group_size: t.group_size,
            data_len: t.data.len() as u64,
        })
        .collect();
    let index_of: std::collections::HashMap<&str, usize> = out_tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();

    let base: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).map_err(|e| format!("source metadata: {e}"))?;

    let mut metadata_json = hfq.metadata_json.clone();
    for round in 0..8 {
        let layout = plan_hfqm_layout(metadata_json.len(), &entries)
            .map_err(|e| format!("plan layout: {e}"))?;

        let mut modules: Vec<HfqModuleRecord> = Vec::with_capacity(hfq.modules().len());
        for src in hfq.modules() {
            let mut start = u64::MAX;
            let mut end = 0u64;
            for t in &src.tensors {
                let i = *index_of
                    .get(t.name.as_str())
                    .ok_or_else(|| format!("module tensor {} missing from output", t.name))?;
                let off = layout.tensor_offsets[i];
                start = start.min(off);
                end = end.max(off + entries[i].data_len);
            }
            let mut record = src.clone();
            record.data_offset = start as usize;
            record.data_size = (end - start) as usize;
            for t in &mut record.tensors {
                let i = index_of[t.name.as_str()];
                t.rel_offset = (layout.tensor_offsets[i] - start) as usize;
                t.data_size = entries[i].data_len as usize;
                t.quant_type = out_tensors[i].quant_type;
            }
            modules.push(record);
        }

        let mut value = base.clone();
        value[HFQM_MODULE_TABLE_KEY] = serde_json::to_value(module_table_json(modules))
            .map_err(|e| format!("serialize module table: {e}"))?;
        let next = serde_json::to_string(&value).map_err(|e| format!("serialize metadata: {e}"))?;
        if next.len() == metadata_json.len() {
            // Same length ⇒ re-planning yields the same offsets ⇒ this is stable.
            return Ok(next);
        }
        metadata_json = next;
        if round == 7 {
            return Err("module-table layout did not converge in 8 rounds".to_string());
        }
    }
    unreachable!()
}

/// Drop the container-level keys the writer regenerates for the derived file.
///
/// `HfqFile::open` INLINES a v2 tail blob into `metadata_json` but leaves
/// `tail_metadata` in place, so the resolved string still carries an
/// `{offset, size, hash}` pointer describing the **source** file's tail. Writing
/// that through verbatim gives the derived file a pointer into bytes that are
/// something else entirely, and it fails to reopen with
/// "HFQM v2 tail hash mismatch" — which is what every artifact this tool emitted
/// from a v2 source has done, on the default path, independently of
/// `--moe-blocks`.
///
/// `hipfire-quantize` already strips these for exactly this reason (see
/// `merge_source_tail_metadata`); the reader in `hipfire-runtime` does not,
/// which is why a writer has to.
fn strip_stale_container_keys(metadata_json: &str) -> Result<String, String> {
    let mut value: serde_json::Value =
        serde_json::from_str(metadata_json).map_err(|e| format!("metadata is not JSON: {e}"))?;
    if let Some(map) = value.as_object_mut() {
        map.remove("tail_metadata");
        map.remove("hfq_format");
    }
    serde_json::to_string(&value).map_err(|e| format!("re-serialize metadata: {e}"))
}
