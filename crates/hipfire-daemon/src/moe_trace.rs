// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-request MoE routing trace and statistics (`moe_trace` setting).
//!
//! The routing records themselves are written by
//! `hipfire_dispatch::moe_telemetry` as the forward routes each row; this side
//! owns the request lifecycle: open when a generate request starts, and when the
//! request is no longer running, close it and write `<stem>.json` beside the
//! `<stem>.bin` with what the request did.

use hipfire_runtime::dispatch::moe_telemetry as telemetry;
use hipfire_runtime::weight_pager::ModulePagerStats;
use hipfire_serving_core::model::LoadedModel;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct OpenTrace {
    stem: PathBuf,
    model: String,
    session: Option<String>,
    started: Instant,
    started_unix_ms: u128,
    pager: Option<ModulePagerStats>,
}

#[derive(Default)]
pub(crate) struct MoeTraces {
    /// Model paths loaded with `moe_trace` on. Keyed by path rather than held
    /// as one flag so a `model_overrides` entry follows its model when several
    /// are resident.
    pub(crate) models: HashSet<String>,
    open: HashMap<String, OpenTrace>,
}

fn pager_stats(m: &LoadedModel) -> Option<ModulePagerStats> {
    if let Some(rb) = &m.registered_backend {
        return rb.backend.expert_pager_stats();
    }
    let pager = m.q35_weights.as_ref()?.pager.as_ref()?;
    Some(pager.borrow().module_stats())
}

/// A request id as a file-name component.
fn file_safe(id: &str) -> String {
    id.chars()
        .take(64)
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl MoeTraces {
    /// Start tracing request `id` if its model has the setting on.
    pub(crate) fn open(&mut self, m: &LoadedModel, id: &str, session: Option<&str>) {
        if !self.models.contains(&m.model_path) {
            return;
        }
        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let stem = hipfire_config::hipfire_dir()
            .join("moe_traces")
            .join(format!("{unix_ms}-{}", file_safe(id)));
        if let Err(e) = telemetry::trace_open(id, &stem) {
            tracing::warn!("moe_trace: cannot open {}: {e}", stem.display());
            return;
        }
        self.open.insert(
            id.to_owned(),
            OpenTrace {
                stem,
                model: m.model_path.clone(),
                session: session.map(str::to_owned),
                started: Instant::now(),
                started_unix_ms: unix_ms,
                pager: pager_stats(m),
            },
        );
    }

    /// Close every open trace whose request is not in `live`, writing its
    /// statistics. `m` supplies the pager counters when it is still the model
    /// the request ran on.
    pub(crate) fn sweep<'a>(
        &mut self,
        m: Option<&LoadedModel>,
        live: impl IntoIterator<Item = &'a str>,
    ) {
        if self.open.is_empty() {
            return;
        }
        let live: HashSet<&str> = live.into_iter().collect();
        let done: Vec<String> = self
            .open
            .keys()
            .filter(|id| !live.contains(id.as_str()))
            .cloned()
            .collect();
        for id in done {
            let t = self.open.remove(&id).expect("listed above");
            let Some(sum) = telemetry::trace_close(&id) else {
                continue;
            };
            let pager = m
                .filter(|m| m.model_path == t.model)
                .and_then(pager_stats)
                .zip(t.pager)
                .map(|(b, a)| {
                    serde_json::json!({
                        "cache_hits": b.module_cache_hits.saturating_sub(a.module_cache_hits),
                        "cold_loads": b.module_cold_loads.saturating_sub(a.module_cold_loads),
                        "evictions": b.module_evictions.saturating_sub(a.module_evictions),
                        "resident_modules": b.resident_modules,
                        "resident_bytes": b.resident_module_bytes,
                        "registered_modules": b.registered_modules,
                    })
                });
            let n = sum.records.max(1) as f64;
            let stats = serde_json::json!({
                "request_id": id,
                "session_id": t.session,
                "model": t.model,
                "started_unix_ms": t.started_unix_ms as u64,
                "wall_s": t.started.elapsed().as_secs_f64(),
                "trace": sum.path.file_name().map(|f| f.to_string_lossy().into_owned()),
                "record": "u32 position, u16 layer, u16 k, k x u16 expert, k x u16 weight/65535; little-endian; last record per (position, layer) wins",
                "records": sum.records,
                "unplaced_rows": sum.unplaced,
                "k": sum.k,
                "positions": sum.positions.map(|(a, b)| [a, b]),
                "layers": sum.layer_counts.len(),
                "experts_touched_per_layer": sum
                    .layer_counts
                    .iter()
                    .map(|c| c.iter().filter(|&&x| x > 0).count())
                    .collect::<Vec<_>>(),
                "mean_weight_by_rank": sum.weight_by_rank.iter().map(|w| w / n).collect::<Vec<_>>(),
                "expert_pager": pager,
            });
            let path = t.stem.with_extension("json");
            if let Err(e) = std::fs::write(&path, format!("{stats:#}\n")) {
                tracing::warn!("moe_trace: cannot write {}: {e}", path.display());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::file_safe;

    #[test]
    fn request_ids_become_plain_file_names() {
        assert_eq!(file_safe("chatcmpl-1a/../x y"), "chatcmpl-1a____x_y");
        assert_eq!(file_safe(&"z".repeat(100)).len(), 64);
    }
}
