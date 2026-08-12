// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire-coexistence artifact moe-router-profile` — what each MoE expert
//! actually specialises in, read out of a calibration artifact.
//!
//! Router *load* alone cannot tell you why an expert is starved. A rarely-routed
//! expert with a **low** mean gate is a weak expert; a rarely-routed expert with a
//! **high** mean gate is a specialist the corpus almost never triggers — and the
//! fix for the second case is corpus material, not a quantization policy. This
//! report joins three things to separate them:
//!
//! - load share and routing imbalance per layer (Gini, max/min),
//! - the winning gate's mean/σ per expert,
//! - the per-expert routed-token histogram recorded by
//!   [`hipfire_runtime::calibration::contracts::LayerRouterStats::token_counts`],
//!   decoded through the model's tokenizer and bucketed into character classes.
//!
//! The token histogram is truncated to the top
//! [`TOKEN_PROFILE_KEEP`](hipfire_runtime::calibration::contracts::TOKEN_PROFILE_KEEP)
//! ids per expert at snapshot time, so class fractions and concentration are
//! computed over the retained head and reported alongside the dropped-id count —
//! never presented as a complete tail. Families whose routed capture runs on the
//! shape-only grouped-MoE seam record no tokens at all; those layers report
//! `token_profile: absent` rather than a misleading empty profile.

use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::calibration::contracts::ExpertLayerTelemetry;
use hipfire_runtime::hfq::HfqFile;
use serde_json::Value;
use std::error::Error;
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: hipfire-coexistence artifact moe-router-profile \
--input <model.calib.hfq> [--tokenizer <hf-dir|model.hfq>] [--layer N] \
[--top N (default: 12)] [--min-activations N] [--json]";

/// Character-class buckets for a decoded token. Deliberately coarse and
/// tokenizer-independent: the question is "what kind of text is this expert for",
/// not a linguistic taxonomy.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct ClassProfile {
    pub word: u64,
    pub digit: u64,
    pub punct: u64,
    pub whitespace: u64,
    pub cjk: u64,
    pub other_non_ascii: u64,
    pub byte_fallback: u64,
    pub total: u64,
}

impl ClassProfile {
    fn record(&mut self, text: &str, count: u64) {
        self.total += count;
        // Byte-fallback tokens carry no linguistic signal; count them separately
        // rather than smearing them into `other_non_ascii`.
        if is_byte_fallback(text) {
            self.byte_fallback += count;
            return;
        }
        let core = text.trim_start_matches([' ', '\u{2581}', '\u{0120}']);
        if core.is_empty() || core.chars().all(char::is_whitespace) {
            self.whitespace += count;
            return;
        }
        if core.chars().any(is_cjk) {
            self.cjk += count;
            return;
        }
        if core.chars().any(|c| c.is_ascii_digit()) {
            self.digit += count;
            return;
        }
        if core.chars().any(|c| !c.is_ascii()) {
            self.other_non_ascii += count;
            return;
        }
        if core
            .chars()
            .all(|c| c.is_alphabetic() || c == '\'' || c == '-')
        {
            self.word += count;
            return;
        }
        self.punct += count;
    }

    /// Class shares, largest first, skipping empty buckets.
    fn dominant(&self) -> Vec<(&'static str, f64)> {
        if self.total == 0 {
            return Vec::new();
        }
        let mut shares: Vec<(&'static str, f64)> = [
            ("word", self.word),
            ("digit", self.digit),
            ("punct", self.punct),
            ("space", self.whitespace),
            ("cjk", self.cjk),
            ("non-ascii", self.other_non_ascii),
            ("byte", self.byte_fallback),
        ]
        .into_iter()
        .filter(|(_, count)| *count > 0)
        .map(|(name, count)| (name, count as f64 / self.total as f64))
        .collect();
        shares.sort_by(|left, right| right.1.total_cmp(&left.1));
        shares
    }
}

fn is_byte_fallback(text: &str) -> bool {
    // `<0xNN>` is the conventional spelling across HF tokenizers.
    let bytes = text.as_bytes();
    bytes.len() == 6 && bytes.starts_with(b"<0x") && bytes.ends_with(b">")
}

fn is_cjk(c: char) -> bool {
    // CJK *punctuation* (、。「」…) counts as CJK, not as generic punctuation:
    // for corpus-gap diagnosis what matters is the script the text belongs to.
    matches!(c as u32,
        0x3000..=0x303F      // CJK symbols + punctuation
        | 0x3040..=0x30FF    // hiragana + katakana
        | 0x3400..=0x4DBF    // CJK ext A
        | 0x4E00..=0x9FFF    // CJK unified
        | 0xAC00..=0xD7AF    // hangul
        | 0xF900..=0xFAFF    // CJK compatibility
        | 0xFF00..=0xFFEF    // fullwidth + halfwidth forms
        | 0x20000..=0x2FA1F) // CJK ext B+
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExpertProfile {
    pub expert: usize,
    pub routed_tokens: u64,
    pub share: f64,
    /// `None` when the expert was never routed — an unrouted expert has no gate,
    /// which is a different fact from a gate of zero.
    pub mean_gate: Option<f64>,
    pub gate_stddev: Option<f64>,
    pub undercovered: bool,
    /// Share of this expert's retained tokens covered by its top 10 ids — high
    /// means narrowly specialised, low means it absorbs generic traffic.
    pub top10_concentration: Option<f64>,
    pub distinct_tokens_kept: usize,
    pub distinct_tokens_dropped: u64,
    pub class_profile: Option<ClassProfile>,
    pub top_tokens: Vec<TokenSample>,
    /// Per-stratum share of this expert's routed tokens with enrichment against
    /// the layer's overall stratum mix. Enrichment, not raw share, is the signal:
    /// an expert taking 40% code tokens from a 40%-code corpus has learned
    /// nothing about code.
    pub strata: Vec<StratumShare>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StratumShare {
    pub stratum: String,
    pub tokens: u64,
    pub share: f64,
    /// `share / corpus share`. >1 means this stratum reaches the expert more
    /// often than chance; `None` when the layer never saw the stratum.
    pub enrichment: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenSample {
    pub id: u32,
    pub text: Option<String>,
    pub count: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LayerProfile {
    pub layer: usize,
    pub num_experts: usize,
    pub k_top: usize,
    pub routed_tokens: u64,
    /// Tokens whose top-1 slot was outside the real expert range — for ZAYA this
    /// is the Mixture-of-Depths skip route (the token bypasses the FFN).
    pub skipped_slots: u64,
    pub imbalance: f64,
    pub gini: f64,
    pub token_profile_present: bool,
    /// True only when the layer saw more than one stratum — a single-label corpus
    /// carries no stratum signal, and reporting one would be noise dressed as a
    /// finding.
    pub stratum_profile_present: bool,
    pub experts: Vec<ExpertProfile>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RouterProfileReport {
    pub artifact: PathBuf,
    pub arch_id: u32,
    pub family: Option<String>,
    pub tokenizer: Option<String>,
    pub min_activations: Option<u64>,
    pub layers: Vec<LayerProfile>,
    pub summary: ProfileSummary,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileSummary {
    pub layers_profiled: usize,
    pub layers_with_token_profile: usize,
    pub undercovered_capture_points: usize,
    /// (layer, expert) pairs below `min_activations`, worst first.
    pub starved: Vec<StarvedExpert>,
    /// Class shares over every retained token in every profiled layer — the
    /// corpus's own mix, for comparison against a starved expert's profile.
    pub corpus_class_profile: Option<ClassProfile>,
    /// Which strata disproportionately feed the *starved* experts. This is the
    /// actionable output for corpus composition: a stratum with lift 3x delivers
    /// three times as much of its traffic to under-covered experts as the corpus
    /// average does, so acquiring more of it buys coverage far more cheaply than
    /// scaling the whole corpus.
    pub stratum_guidance: Vec<StratumLift>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StratumLift {
    pub stratum: String,
    /// This stratum's share of all tokens routed to starved experts.
    pub starved_share: f64,
    /// This stratum's share of all routed tokens.
    pub overall_share: f64,
    /// `starved_share / overall_share`. >1 means over-represented in the tail.
    pub lift: f64,
    pub starved_tokens: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StarvedExpert {
    pub layer: usize,
    pub expert: usize,
    pub routed_tokens: u64,
    pub share: f64,
    pub mean_gate: Option<f64>,
    pub dominant_classes: Vec<(String, f64)>,
    pub top_tokens: Vec<String>,
}

struct Options {
    input: PathBuf,
    tokenizer: Option<PathBuf>,
    layer: Option<usize>,
    top: usize,
    min_activations: Option<u64>,
    json: bool,
}

fn gini(counts: &[u64]) -> f64 {
    let n = counts.len();
    if n == 0 {
        return 0.0;
    }
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let total: u64 = sorted.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let mut cumulative = 0u64;
    let mut area = 0u64;
    for value in &sorted {
        cumulative += value;
        area += cumulative;
    }
    1.0 - 2.0 * area as f64 / (n as f64 * total as f64) + 1.0 / n as f64
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer, Box<dyn Error>> {
    if path.is_dir() {
        let json = std::fs::read_to_string(path.join("tokenizer.json"))?;
        Ok(Tokenizer::from_hf_json(&json)?)
    } else if path.extension().is_some_and(|ext| ext == "json") {
        let json = std::fs::read_to_string(path)?;
        Ok(Tokenizer::from_hf_json(&json)?)
    } else {
        let hfq = HfqFile::open_index_only(path)?;
        Ok(Tokenizer::from_hfq_metadata(&hfq.metadata_json)?)
    }
}

fn profile_layer(
    telemetry: &ExpertLayerTelemetry,
    tokenizer: Option<&Tokenizer>,
    top: usize,
    min_activations: Option<u64>,
    corpus_classes: &mut ClassProfile,
) -> LayerProfile {
    let router = &telemetry.router;
    let hits = &router.top1_hits;
    let total: u64 = hits.iter().sum();
    let token_profile_present = router.token_counts.iter().any(|counts| !counts.is_empty());

    // Layer-wide stratum mix is the baseline every expert is scored against.
    let mut layer_strata: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    for counts in &router.stratum_counts {
        for (stratum, count) in counts {
            *layer_strata.entry(stratum.clone()).or_insert(0) += count;
        }
    }
    let layer_stratum_total: u64 = layer_strata.values().sum();

    let mut experts = Vec::with_capacity(telemetry.num_experts);
    for expert in 0..telemetry.num_experts {
        let routed = hits.get(expert).copied().unwrap_or(0);
        let weights = router.route_weights.get(expert);
        let (mean_gate, gate_stddev) = match weights {
            Some(stats) if stats.count > 0 => {
                let count = stats.count as f64;
                let mean = stats.sum / count;
                // Var = E[x²] − E[x]²; clamp the tiny negatives f64 can produce.
                let variance = (stats.sum_squared / count - mean * mean).max(0.0);
                (Some(mean), Some(variance.sqrt()))
            }
            _ => (None, None),
        };
        let counts = router.token_counts.get(expert);
        let (class_profile, top_tokens, concentration, kept) = match counts {
            Some(counts) if !counts.is_empty() => {
                let mut classes = ClassProfile::default();
                let mut ranked: Vec<(u32, u64)> = counts.iter().map(|(id, n)| (*id, *n)).collect();
                ranked
                    .sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
                if let Some(tokenizer) = tokenizer {
                    for (id, count) in &ranked {
                        let text = decode_token(tokenizer, *id);
                        classes.record(&text, *count);
                        corpus_classes.record(&text, *count);
                    }
                }
                let retained: u64 = ranked.iter().map(|(_, n)| *n).sum();
                let head: u64 = ranked.iter().take(10).map(|(_, n)| *n).sum();
                let concentration = (retained > 0).then(|| head as f64 / retained as f64);
                let samples = ranked
                    .iter()
                    .take(top)
                    .map(|(id, count)| TokenSample {
                        id: *id,
                        text: tokenizer.map(|t| decode_token(t, *id)),
                        count: *count,
                    })
                    .collect();
                let kept = ranked.len();
                (tokenizer.map(|_| classes), samples, concentration, kept)
            }
            _ => (None, Vec::new(), None, 0),
        };
        let strata = match router.stratum_counts.get(expert) {
            Some(counts) if !counts.is_empty() && layer_stratum_total > 0 => {
                let expert_total: u64 = counts.values().sum();
                let mut shares: Vec<StratumShare> = counts
                    .iter()
                    .map(|(stratum, count)| {
                        let share = *count as f64 / expert_total as f64;
                        let base = layer_strata.get(stratum).copied().unwrap_or(0) as f64
                            / layer_stratum_total as f64;
                        StratumShare {
                            stratum: stratum.clone(),
                            tokens: *count,
                            share,
                            enrichment: (base > 0.0).then(|| share / base),
                        }
                    })
                    .collect();
                shares.sort_by(|left, right| right.share.total_cmp(&left.share));
                shares
            }
            _ => Vec::new(),
        };
        experts.push(ExpertProfile {
            expert,
            routed_tokens: routed,
            share: if total > 0 {
                routed as f64 / total as f64
            } else {
                0.0
            },
            mean_gate,
            gate_stddev,
            undercovered: min_activations.is_some_and(|floor| routed < floor),
            top10_concentration: concentration,
            distinct_tokens_kept: kept,
            distinct_tokens_dropped: router
                .token_profile_dropped
                .get(expert)
                .copied()
                .unwrap_or(0),
            class_profile,
            top_tokens,
            strata,
        });
    }

    let min_hits = hits.iter().copied().min().unwrap_or(0);
    let max_hits = hits.iter().copied().max().unwrap_or(0);
    LayerProfile {
        layer: telemetry.layer,
        num_experts: telemetry.num_experts,
        k_top: telemetry.k_top,
        routed_tokens: router.routed_tokens,
        skipped_slots: router.dropped_indices,
        imbalance: if min_hits > 0 {
            max_hits as f64 / min_hits as f64
        } else {
            f64::INFINITY
        },
        gini: gini(hits),
        token_profile_present,
        stratum_profile_present: layer_strata.len() > 1,
        experts,
    }
}

fn decode_token(tokenizer: &Tokenizer, id: u32) -> String {
    tokenizer
        .vocab()
        .get(id as usize)
        .cloned()
        .unwrap_or_else(|| tokenizer.decode(&[id]))
}

pub fn build_report(
    input: &Path,
    tokenizer: Option<&Tokenizer>,
    tokenizer_label: Option<String>,
    layer_filter: Option<usize>,
    top: usize,
    min_activations: Option<u64>,
) -> Result<RouterProfileReport, Box<dyn Error>> {
    let hfq = HfqFile::open_index_only(input)?;
    let metadata: Value = serde_json::from_str(&hfq.metadata_json)?;
    if metadata.get("artifact_kind").and_then(Value::as_str) != Some("calibration") {
        return Err(format!("{} is not a calibration artifact", input.display()).into());
    }
    let telemetry: Vec<ExpertLayerTelemetry> = match metadata.get("expert_telemetry") {
        Some(value) => serde_json::from_value(value.clone())?,
        None => {
            return Err(format!(
                "{} carries no expert_telemetry; it is not a routed-MoE calibration",
                input.display()
            )
            .into())
        }
    };
    if telemetry.is_empty() {
        return Err(format!("{} has an empty expert_telemetry", input.display()).into());
    }

    let mut corpus_classes = ClassProfile::default();
    let mut layers = Vec::new();
    for entry in &telemetry {
        if layer_filter.is_some_and(|wanted| wanted != entry.layer) {
            continue;
        }
        layers.push(profile_layer(
            entry,
            tokenizer,
            top,
            min_activations,
            &mut corpus_classes,
        ));
    }
    if layers.is_empty() {
        return Err(format!(
            "layer filter selected none of the {} profiled layers",
            telemetry.len()
        )
        .into());
    }

    let mut starved: Vec<StarvedExpert> = Vec::new();
    let mut undercovered = 0usize;
    for layer in &layers {
        for expert in &layer.experts {
            if !expert.undercovered {
                continue;
            }
            undercovered += 1;
            starved.push(StarvedExpert {
                layer: layer.layer,
                expert: expert.expert,
                routed_tokens: expert.routed_tokens,
                share: expert.share,
                mean_gate: expert.mean_gate,
                dominant_classes: expert
                    .class_profile
                    .map(|classes| {
                        classes
                            .dominant()
                            .into_iter()
                            .take(3)
                            .map(|(name, share)| (name.to_string(), share))
                            .collect()
                    })
                    .unwrap_or_default(),
                top_tokens: expert
                    .top_tokens
                    .iter()
                    .take(8)
                    .map(|sample| {
                        sample
                            .text
                            .clone()
                            .unwrap_or_else(|| format!("<id {}>", sample.id))
                    })
                    .collect(),
            });
        }
    }
    starved.sort_by(|left, right| left.routed_tokens.cmp(&right.routed_tokens));

    // Which strata feed the tail. Aggregated across layers: a stratum's lift is
    // its share of starved-expert traffic over its share of all traffic.
    let mut starved_tokens: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    let mut all_tokens: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    for layer in &layers {
        for expert in &layer.experts {
            for share in &expert.strata {
                *all_tokens.entry(share.stratum.clone()).or_insert(0) += share.tokens;
                if expert.undercovered {
                    *starved_tokens.entry(share.stratum.clone()).or_insert(0) += share.tokens;
                }
            }
        }
    }
    let starved_total: u64 = starved_tokens.values().sum();
    let all_total: u64 = all_tokens.values().sum();
    let mut stratum_guidance: Vec<StratumLift> = if starved_total > 0 && all_total > 0 {
        all_tokens
            .iter()
            .map(|(stratum, total)| {
                let starved = starved_tokens.get(stratum).copied().unwrap_or(0);
                let starved_share = starved as f64 / starved_total as f64;
                let overall_share = *total as f64 / all_total as f64;
                StratumLift {
                    stratum: stratum.clone(),
                    starved_share,
                    overall_share,
                    lift: if overall_share > 0.0 {
                        starved_share / overall_share
                    } else {
                        0.0
                    },
                    starved_tokens: starved,
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    stratum_guidance.sort_by(|left, right| right.lift.total_cmp(&left.lift));

    let mut notes = Vec::new();
    let with_tokens = layers
        .iter()
        .filter(|layer| layer.token_profile_present)
        .count();
    if with_tokens == 0 {
        notes.push(
            "no layer recorded a token profile: this family's routed capture runs on the \
             shape-only grouped-MoE seam, which does not see corpus tokens. Load and gate \
             statistics below are still exact."
                .into(),
        );
    } else if with_tokens < layers.len() {
        notes.push(format!(
            "{with_tokens} of {} layers recorded a token profile",
            layers.len()
        ));
    }
    if tokenizer.is_none() && with_tokens > 0 {
        notes.push(
            "no --tokenizer supplied: token ids are reported undecoded and class profiles are \
             omitted."
                .into(),
        );
    }
    if layers.iter().any(|layer| layer.skipped_slots == 0)
        && layers.iter().all(|layer| layer.skipped_slots == 0)
    {
        notes.push(
            "no token took a skip/no-expert route in any profiled layer — for a \
             Mixture-of-Depths router that means the skip slot never won."
                .into(),
        );
    }
    if layers
        .iter()
        .flat_map(|layer| &layer.experts)
        .any(|expert| expert.distinct_tokens_dropped > 0)
    {
        notes.push(
            "token histograms are truncated to their most-routed ids; class shares and \
             concentration describe the retained head, not the full tail."
                .into(),
        );
    }

    Ok(RouterProfileReport {
        artifact: input.to_path_buf(),
        arch_id: hfq.arch_id,
        family: metadata
            .get("family")
            .and_then(Value::as_str)
            .map(str::to_string),
        tokenizer: tokenizer_label,
        min_activations,
        summary: ProfileSummary {
            layers_profiled: layers.len(),
            layers_with_token_profile: with_tokens,
            undercovered_capture_points: undercovered,
            starved,
            corpus_class_profile: (corpus_classes.total > 0).then_some(corpus_classes),
            stratum_guidance,
            notes,
        },
        layers,
    })
}

fn escape(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\n' => "\\n".to_string(),
            '\t' => "\\t".to_string(),
            '\r' => "\\r".to_string(),
            ' ' => "·".to_string(),
            '\u{2581}' => "·".to_string(),
            other => other.to_string(),
        })
        .collect()
}

fn render_text(report: &RouterProfileReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "MoE router profile: {} (arch {}, family {})",
        report.artifact.display(),
        report.arch_id,
        report.family.as_deref().unwrap_or("unknown"),
    );
    for note in &report.summary.notes {
        let _ = writeln!(out, "  note: {note}");
    }
    if let Some(classes) = &report.summary.corpus_class_profile {
        let shares = classes
            .dominant()
            .into_iter()
            .map(|(name, share)| format!("{name} {:.0}%", share * 100.0))
            .collect::<Vec<_>>()
            .join("  ");
        let _ = writeln!(out, "  corpus token mix: {shares}");
    }
    for layer in &report.layers {
        let _ = writeln!(
            out,
            "\nlayer {} — {} experts, k_top {}, {} routed tokens, imbalance {:.1}x, gini {:.3}, skip-route {}",
            layer.layer,
            layer.num_experts,
            layer.k_top,
            layer.routed_tokens,
            layer.imbalance,
            layer.gini,
            layer.skipped_slots,
        );
        let _ = writeln!(
            out,
            "  {:>2}  {:>9} {:>7}  {:>11}  {:>5}  {:<22} {}",
            "e", "tokens", "share", "gate mean±σ", "conc", "classes", "top tokens"
        );
        for expert in &layer.experts {
            let classes = expert
                .class_profile
                .map(|classes| {
                    classes
                        .dominant()
                        .into_iter()
                        .take(3)
                        .map(|(name, share)| format!("{name} {:.0}%", share * 100.0))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_else(|| "-".into());
            let tokens = expert
                .top_tokens
                .iter()
                .map(|sample| match &sample.text {
                    Some(text) => format!("{:?}", escape(text)),
                    None => format!("#{}", sample.id),
                })
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(
                out,
                "  {}{:>2} {:>9} {:>6.2}%  {:>12} {:>5}  {:<22} {}",
                if expert.undercovered { "!" } else { " " },
                expert.expert,
                expert.routed_tokens,
                expert.share * 100.0,
                match (expert.mean_gate, expert.gate_stddev) {
                    (Some(mean), Some(sigma)) => format!("{mean:.3}±{sigma:.3}"),
                    _ => "never routed".to_string(),
                },
                expert
                    .top10_concentration
                    .map(|value| format!("{:.2}", value))
                    .unwrap_or_else(|| "-".into()),
                classes,
                tokens,
            );
            // Stratum enrichment is the depth-robust lens: token identity says
            // little where routing is semantic rather than lexical.
            if layer.stratum_profile_present && !expert.strata.is_empty() {
                let enriched = expert
                    .strata
                    .iter()
                    .filter(|share| share.enrichment.is_some_and(|value| value >= 1.25))
                    .take(3)
                    .map(|share| {
                        format!(
                            "{} {:.0}% ({:.1}x)",
                            share.stratum,
                            share.share * 100.0,
                            share.enrichment.unwrap_or(1.0)
                        )
                    })
                    .collect::<Vec<_>>();
                if !enriched.is_empty() {
                    let _ = writeln!(out, "        strata: {}", enriched.join("  "));
                }
            }
        }
    }
    if !report.summary.stratum_guidance.is_empty() {
        let _ = writeln!(
            out,
            "\nstratum lift toward starved experts (acquire the high-lift strata first):"
        );
        let _ = writeln!(
            out,
            "  {:<20} {:>13} {:>13} {:>7} {:>14}",
            "stratum", "starved share", "corpus share", "lift", "starved tokens"
        );
        for lift in &report.summary.stratum_guidance {
            let _ = writeln!(
                out,
                "  {:<20} {:>12.1}% {:>12.1}% {:>6.2}x {:>14}",
                lift.stratum,
                lift.starved_share * 100.0,
                lift.overall_share * 100.0,
                lift.lift,
                lift.starved_tokens,
            );
        }
    }
    if !report.summary.starved.is_empty() {
        let _ = writeln!(
            out,
            "\nstarved capture points (below {} activations): {}",
            report.min_activations.unwrap_or(0),
            report.summary.undercovered_capture_points
        );
        for starved in &report.summary.starved {
            let classes = starved
                .dominant_classes
                .iter()
                .map(|(name, share)| format!("{name} {:.0}%", share * 100.0))
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(
                out,
                "  layer {:>2} expert {:>2}: {} tokens ({:.2}%), gate {}, {} | {}",
                starved.layer,
                starved.expert,
                starved.routed_tokens,
                starved.share * 100.0,
                starved
                    .mean_gate
                    .map(|gate| format!("{gate:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
                if classes.is_empty() { "-" } else { &classes },
                starved
                    .top_tokens
                    .iter()
                    .map(|text| format!("{:?}", escape(text)))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
    }
    out
}

fn parse(args: &[String]) -> Result<Options, Box<dyn Error>> {
    let mut input = None;
    let mut tokenizer = None;
    let mut layer = None;
    let mut top = 12usize;
    let mut min_activations = None;
    let mut json = false;
    let mut index = 0usize;
    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--json" => json = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| format!("moe-router-profile: {flag} needs a value"))?;
                match flag {
                    "--input" => input = Some(PathBuf::from(value)),
                    "--tokenizer" => tokenizer = Some(PathBuf::from(value)),
                    "--layer" => layer = Some(value.parse()?),
                    "--top" => top = value.parse()?,
                    "--min-activations" => min_activations = Some(value.parse()?),
                    _ => {
                        return Err(
                            format!("moe-router-profile: unknown flag {flag}\n{USAGE}").into()
                        )
                    }
                }
                index += 1;
            }
        }
        index += 1;
    }
    let input = input.ok_or_else(|| format!("moe-router-profile requires --input\n{USAGE}"))?;
    if top == 0 {
        return Err("moe-router-profile: --top must be nonzero".into());
    }
    Ok(Options {
        input,
        tokenizer,
        layer,
        top,
        min_activations,
        json,
    })
}

pub fn run_cli(args: &[String]) -> Result<(), Box<dyn Error>> {
    let options = parse(args)?;
    let tokenizer = options
        .tokenizer
        .as_deref()
        .map(load_tokenizer)
        .transpose()?;
    let report = build_report(
        &options.input,
        tokenizer.as_ref(),
        options
            .tokenizer
            .as_ref()
            .map(|path| path.display().to_string()),
        options.layer,
        options.top,
        options.min_activations,
    )?;
    if options.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render_text(&report));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_profile_separates_prose_digits_and_cjk() {
        let mut classes = ClassProfile::default();
        classes.record(" the", 10);
        classes.record("ing", 5);
        classes.record(" 2048", 3);
        classes.record("。", 4);
        classes.record("<0x1F>", 2);
        classes.record("\n", 1);
        classes.record("=>", 6);
        assert_eq!(classes.word, 15);
        assert_eq!(classes.digit, 3);
        assert_eq!(classes.cjk, 4);
        assert_eq!(classes.byte_fallback, 2);
        assert_eq!(classes.whitespace, 1);
        assert_eq!(classes.punct, 6);
        assert_eq!(classes.total, 31);
        let dominant = classes.dominant();
        assert_eq!(dominant[0].0, "word");
    }

    #[test]
    fn byte_fallback_is_recognised_without_swallowing_short_tags() {
        assert!(is_byte_fallback("<0x0A>"));
        assert!(!is_byte_fallback("<0xAB"));
        assert!(!is_byte_fallback("<eos>"));
        assert!(!is_byte_fallback("0x0A"));
    }

    #[test]
    fn gini_is_zero_for_uniform_and_high_for_concentrated_load() {
        assert!(gini(&[100, 100, 100, 100]).abs() < 1e-12);
        assert!(gini(&[0, 0, 0, 400]) > 0.7);
        // Degenerate inputs must not divide by zero.
        assert_eq!(gini(&[]), 0.0);
        assert_eq!(gini(&[0, 0]), 0.0);
    }

    #[test]
    fn stratum_enrichment_is_measured_against_the_layer_mix_not_raw_share() {
        use hipfire_runtime::calibration::contracts::{
            ExpertCaptureQuota, ExpertSamplingPolicy, LayerRouterStats,
        };
        use std::collections::BTreeMap;

        // Layer mix: 100 code, 100 prose. Expert 0 takes 90 code / 10 prose;
        // expert 1 takes 10 code / 90 prose.
        let mut router = LayerRouterStats::default();
        router.top1_hits = vec![100, 100];
        router.topk_hits = vec![100, 100];
        router.routed_tokens = 200;
        router.routed_slots = 200;
        router.route_weights = vec![Default::default(); 2];
        router.token_counts = vec![BTreeMap::new(); 2];
        router.token_profile_dropped = vec![0; 2];
        router.stratum_counts = vec![
            BTreeMap::from([("code".to_string(), 90), ("prose".to_string(), 10)]),
            BTreeMap::from([("code".to_string(), 10), ("prose".to_string(), 90)]),
        ];
        let telemetry = ExpertLayerTelemetry {
            layer: 0,
            num_experts: 2,
            k_top: 1,
            quota: ExpertCaptureQuota {
                min_rows: 1,
                target_rows: 1,
                tile_rows: 1,
                sampling: ExpertSamplingPolicy::DeterministicFirst { seed: 0 },
            },
            router,
            gate_up: vec![Default::default(); 2],
            down: vec![Default::default(); 2],
        };
        let mut corpus = ClassProfile::default();
        let profile = profile_layer(&telemetry, None, 4, None, &mut corpus);
        assert!(profile.stratum_profile_present);

        let code = profile.experts[0]
            .strata
            .iter()
            .find(|share| share.stratum == "code")
            .expect("code stratum");
        // 90% of the expert's tokens against a 50% layer baseline = 1.8x.
        assert!((code.share - 0.9).abs() < 1e-9);
        assert!((code.enrichment.unwrap() - 1.8).abs() < 1e-9);

        let prose = profile.experts[1]
            .strata
            .iter()
            .find(|share| share.stratum == "prose")
            .expect("prose stratum");
        assert!((prose.enrichment.unwrap() - 1.8).abs() < 1e-9);
    }

    #[test]
    fn single_stratum_corpus_reports_no_stratum_signal() {
        use hipfire_runtime::calibration::contracts::{
            ExpertCaptureQuota, ExpertSamplingPolicy, LayerRouterStats,
        };
        use std::collections::BTreeMap;

        let mut router = LayerRouterStats::default();
        router.top1_hits = vec![10];
        router.topk_hits = vec![10];
        router.routed_tokens = 10;
        router.routed_slots = 10;
        router.route_weights = vec![Default::default(); 1];
        router.token_counts = vec![BTreeMap::new(); 1];
        router.token_profile_dropped = vec![0; 1];
        router.stratum_counts = vec![BTreeMap::from([("plain-text".to_string(), 10)])];
        let telemetry = ExpertLayerTelemetry {
            layer: 0,
            num_experts: 1,
            k_top: 1,
            quota: ExpertCaptureQuota {
                min_rows: 1,
                target_rows: 1,
                tile_rows: 1,
                sampling: ExpertSamplingPolicy::DeterministicFirst { seed: 0 },
            },
            router,
            gate_up: vec![Default::default(); 1],
            down: vec![Default::default(); 1],
        };
        let mut corpus = ClassProfile::default();
        let profile = profile_layer(&telemetry, None, 4, None, &mut corpus);
        // One label carries no signal; enrichment would be a tautological 1.0.
        assert!(!profile.stratum_profile_present);
    }

    #[test]
    fn escape_makes_whitespace_and_word_boundaries_visible() {
        assert_eq!(escape(" the"), "·the");
        assert_eq!(escape("\u{2581}word"), "·word");
        assert_eq!(escape("a\nb"), "a\\nb");
    }
}
