// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! N-gram speculative decoding inside the fused (dense or grouped-MoE) batch
//! decode step.
//!
//! Each session drafts from its own n-gram table (the prompt and every committed
//! token), and the drafts ride the step as extra rows of the same fused forward:
//! row 0 is the token the step would have fed anyway, rows 1.. are drafts, and
//! the per-row argmax says how many drafts the model agrees with. The step
//! emits the accepted drafts plus the model's own next token (the "bonus").
//!
//! Rollback. A rejected draft leaves wrong K/V rows and a DeltaNet state that
//! has absorbed them. The KV rows above the cursor are harmless — the next write
//! overwrites them, and nothing attends past the cursor — as long as the step
//! never wraps the KVarN window ring into a new block while the old one is still
//! open, so drafts stop at the block end. The DeltaNet state is restored from a
//! snapshot taken before the forward, which also undoes the accepted rows; those
//! are "pending" (`SessionRegistry::spec_pending`) and ride the NEXT step as
//! rows, so a rejection costs rows, not a second forward pass. Any decode path
//! other than this one feeds pending tokens first (`flush_spec_pending`).
//!
//! Greedy only, like the batch decode path it extends.

use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::speculative::DeltaNetSnapshot;
use hipfire_generate::{GenerateBatchDecodeEnvelope, GenerateBatchDecodeSession};
use hipfire_specdecode_ngram::{NgramConfig, NgramSpec};

use crate::model::LoadedModel;
use crate::session::Qwen35RequestSessionState;

/// Batched n-gram speculation, on unless `HIPFIRE_BATCH_NGRAM_SPEC=0`.
pub fn batch_ngram_spec_enabled() -> bool {
    !matches!(
        std::env::var("HIPFIRE_BATCH_NGRAM_SPEC").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Longest draft one session proposes per step.
fn max_draft() -> usize {
    env_usize("HIPFIRE_BATCH_SPEC_MAX_DRAFT", 8)
}

/// Rows one fused step may carry, feeds and drafts together. Drafts only get
/// what the sessions' own feed rows leave. Measured on the 27B (gfx1151): a row
/// costs ~3-5 ms up to ~16 rows, then climbs steeply (19 ms/row from 16 to 32).
/// ponytail: first-come split of the spare rows; rank by draft confidence
/// (n-gram order / measured acceptance) if the budget binds often.
fn row_budget() -> usize {
    env_usize("HIPFIRE_BATCH_SPEC_ROW_BUDGET", 96)
}

/// Past this many pending tokens a session drafts nothing, so its next step
/// cannot reject and the pending run is fed for good.
const MAX_PENDING: usize = 8;

const KVARN_GROUP: usize = hipfire_runtime::kv::KvCache::KVARN_GROUP;

/// Below this running acceptance a session stops drafting, save a probe every
/// `PROBE_EVERY` steps (so it picks back up when its output turns repetitive).
/// Free-form prose drafts rarely and is mostly rejected: measured on the 27B, 8
/// essays ran 8% slower with ungated drafting, while code the model is copying
/// accepts ~95%.
const ACCEPT_FLOOR: f32 = 0.3;
const PROBE_EVERY: u32 = 8;

/// One session's speculation state.
pub struct BatchSpecSession {
    ngram: NgramSpec,
    /// Running fraction of drafted tokens accepted (starts optimistic).
    acceptance: f32,
    steps_since_draft: u32,
}

impl BatchSpecSession {
    fn new(history: &[u32]) -> Self {
        let mut ngram = NgramSpec::new(NgramConfig::default());
        ngram.observe(history);
        Self {
            ngram,
            acceptance: 1.0,
            steps_since_draft: 0,
        }
    }

    /// Whether to offer drafts this step (see `ACCEPT_FLOOR`).
    fn wants_draft(&mut self) -> bool {
        if self.acceptance >= ACCEPT_FLOOR || self.steps_since_draft >= PROBE_EVERY {
            self.steps_since_draft = 0;
            true
        } else {
            self.steps_since_draft += 1;
            false
        }
    }

    fn record(&mut self, drafted: usize, accepted: usize) {
        self.ngram.record_acceptance(accepted);
        self.acceptance = 0.7 * self.acceptance + 0.3 * accepted as f32 / drafted as f32;
    }
}

pub fn spec_pending_len(m: &LoadedModel, session_id: &str) -> usize {
    m.q35_registry
        .spec_pending
        .get(session_id)
        .map_or(0, Vec::len)
}

/// Feed `session_id`'s pending tokens one at a time, leaving its KV/DeltaNet
/// state and `logits` as if they had been decoded normally. For every decode
/// path that is not the speculative one; a no-op without pending tokens.
pub fn flush_spec_pending(
    m: &mut LoadedModel,
    gpu: &mut hipfire_rdna::Gpu,
    session_id: &str,
) -> Result<(), String> {
    let Some(pending) = m.q35_registry.spec_pending.remove(session_id) else {
        return Ok(());
    };
    let Some(mut state) = m.q35_registry.sessions.remove(session_id) else {
        return Err(format!(
            "session {session_id} has pending speculative tokens but is not resident"
        ));
    };
    let result = (|| {
        let weights = m.q35_weights.as_ref().ok_or("qwen35 weights missing")?;
        let config = m.q35_config.as_ref().ok_or("qwen35 config missing")?;
        let scratch = m.q35_scratch.as_ref().ok_or("qwen35 scratch missing")?;
        crate::qwen35_prefill::qwen35_prefill_owned_session_serial_segment(
            gpu, weights, config, scratch, &mut state, &pending,
        )?;
        gpu.memcpy_dtod_auto(
            &state.logits.buf,
            &scratch.logits.buf,
            scratch.logits.buf.size(),
        )
        .map_err(|e| format!("flush pending speculative tokens: logits copy: {e:?}"))
    })();
    m.q35_registry
        .sessions
        .insert(session_id.to_string(), state);
    result
}

/// What one session does this step.
struct SessionPlan {
    /// Committed tokens to feed: the pending run, or the one token this step
    /// emits when there is none.
    feed: Vec<u32>,
    drafts: Vec<u32>,
    /// Tokens emitted this step so far (the fresh token, when `feed` is it).
    emitted: Vec<u32>,
    stop: bool,
    remaining: usize,
}

/// Longest draft prefix the model agrees with, and the model's token after it.
/// `argmax[i]` is the model's choice after row i; row `feed_len - 1` is the
/// last committed token, so draft j is checked against row `feed_len - 1 + j`.
fn accept(drafts: &[u32], argmax_after: impl Fn(usize) -> u32, feed_len: usize) -> (usize, u32) {
    let base = feed_len - 1;
    let mut a = 0;
    while a < drafts.len() && drafts[a] == argmax_after(base + a) {
        a += 1;
    }
    (a, argmax_after(base + a))
}

/// Row index of `(session, token j)` in the fused forward's round-major order.
fn round_major_index(lens: &[usize]) -> Vec<Vec<usize>> {
    let mut idx: Vec<Vec<usize>> = lens.iter().map(|&n| Vec::with_capacity(n)).collect();
    let mut next = 0;
    for j in 0..lens.iter().copied().max().unwrap_or(0) {
        for (s, &n) in lens.iter().enumerate() {
            if j < n {
                idx[s].push(next);
                next += 1;
            }
        }
    }
    idx
}

/// Drafts `session` may propose: bounded by the draft cap, the remaining token
/// budget (every accepted draft and the bonus are emitted), the KVarN block end
/// (see module doc), and `MAX_PENDING`.
fn draft_cap(seq_pos: usize, feed_len: usize, fresh_token: bool, remaining: usize) -> usize {
    if feed_len > MAX_PENDING {
        return 0;
    }
    let budget = if fresh_token {
        remaining.saturating_sub(1)
    } else {
        remaining
    };
    let block_room = KVARN_GROUP.saturating_sub(seq_pos % KVARN_GROUP + feed_len);
    max_draft().min(budget).min(block_room)
}

fn is_terminator(
    config: &qwen35::Qwen35Config,
    tokenizer: &hipfire_model::tokenizer::Tokenizer,
    im_end_token: Option<u32>,
    token: u32,
) -> bool {
    token == config.eos_token || im_end_token == Some(token) || tokenizer.is_terminator(token)
}

/// One speculative fused decode step over `chunk`: 1+ resident sessions on a
/// dense model, 2+ on a grouped-MoE one (its forward keeps the two-session
/// minimum).
pub fn qwen35_decode_step_fused_dense_spec_chunk(
    m: &mut LoadedModel,
    gpu: &mut hipfire_rdna::Gpu,
    envelope: &GenerateBatchDecodeEnvelope,
    chunk: &[GenerateBatchDecodeSession],
    im_end_token: Option<u32>,
) -> Result<Vec<serde_json::Value>, String> {
    let mut states: Vec<(GenerateBatchDecodeSession, Qwen35RequestSessionState)> =
        Vec::with_capacity(chunk.len());
    for session in chunk {
        let state = m
            .q35_registry
            .sessions
            .remove(&session.session_id)
            .ok_or_else(|| {
                format!(
                    "decode session {} is not resident for speculative dense decode",
                    session.session_id
                )
            })?;
        states.push((session.clone(), state));
    }
    let mut pendings: Vec<Vec<u32>> = chunk
        .iter()
        .map(|s| {
            m.q35_registry
                .spec_pending
                .remove(&s.session_id)
                .unwrap_or_default()
        })
        .collect();

    let result = step(m, gpu, envelope, &mut states, &mut pendings, im_end_token);

    for ((session, state), pending) in states.into_iter().zip(pendings) {
        if !pending.is_empty() {
            m.q35_registry
                .spec_pending
                .insert(session.session_id.clone(), pending);
        }
        m.q35_registry.sessions.insert(session.session_id, state);
    }
    // State of sessions that are gone (released, evicted) goes with them.
    let registry = &mut m.q35_registry;
    let live: std::collections::HashSet<&String> = registry
        .sessions
        .keys()
        .chain(registry.active_session_id.iter())
        .collect();
    let dead: Vec<String> = registry
        .spec_sessions
        .keys()
        .filter(|k| !live.contains(k))
        .cloned()
        .collect();
    for k in dead {
        registry.spec_sessions.remove(&k);
    }
    // A request cancelled mid-run is released with its pending run unfed.
    let live: std::collections::HashSet<String> = live.into_iter().cloned().collect();
    registry.spec_pending.retain(|k, _| live.contains(k));
    result
}

fn step(
    m: &mut LoadedModel,
    gpu: &mut hipfire_rdna::Gpu,
    envelope: &GenerateBatchDecodeEnvelope,
    states: &mut [(GenerateBatchDecodeSession, Qwen35RequestSessionState)],
    pendings: &mut [Vec<u32>],
    im_end_token: Option<u32>,
) -> Result<Vec<serde_json::Value>, String> {
    let config = m.q35_config.as_ref().ok_or("qwen35 config missing")?;
    let tokenizer = m
        .tokenizer
        .as_ref()
        .ok_or("generate_batch_decode_step requires a tokenizer")?;
    let vocab = config.vocab_size;

    // 1. What each session feeds, and its drafts.
    let mut plans = Vec::with_capacity(states.len());
    for ((session, state), pending) in states.iter().zip(pendings.iter_mut()) {
        let physical = state.cursor.seq_pos + state.kv_cache().compact_offset;
        if physical + pending.len() != session.logical_position {
            return Err(format!(
                "decode session {} logical_position mismatch: expected={} resident={}+{} pending",
                session.session_id,
                session.logical_position,
                physical,
                pending.len()
            ));
        }
        let remaining = session.max_tokens_remaining;
        let mut plan = if pending.is_empty() {
            let token = gpu
                .argmax_f32(&state.logits, vocab)
                .map_err(|e| format!("qwen35 decode argmax: {e:?}"))?;
            let stop = is_terminator(config, tokenizer, im_end_token, token) || remaining <= 1;
            SessionPlan {
                feed: vec![token],
                drafts: Vec::new(),
                emitted: vec![token],
                stop,
                remaining,
            }
        } else {
            // The pending run was emitted last step; it only needs feeding.
            SessionPlan {
                feed: std::mem::take(pending),
                drafts: Vec::new(),
                emitted: Vec::new(),
                stop: false,
                remaining,
            }
        };
        if !plan.stop {
            let cap = draft_cap(
                state.cursor.seq_pos,
                plan.feed.len(),
                !plan.emitted.is_empty(),
                remaining,
            );
            // History = everything committed before this step's feed. A pending
            // run was observed when it was emitted.
            let spec = m
                .q35_registry
                .spec_sessions
                .entry(session.session_id.clone())
                .or_insert_with(|| BatchSpecSession::new(&state.cursor.conversation_tokens));
            spec.ngram.observe(&plan.emitted);
            if cap > 0 && spec.wants_draft() {
                if let Some(spine) = spec.ngram.draft() {
                    plan.drafts = spine.iter().copied().take(cap).collect();
                }
            }
        }
        plans.push(plan);
    }
    // Row budget: feeds are owed; drafts share what is left, in session order.
    let mut spare = row_budget().saturating_sub(plans.iter().map(|p| p.feed.len()).sum());
    for plan in &mut plans {
        let take = plan.drafts.len().min(spare);
        plan.drafts.truncate(take);
        spare -= take;
    }

    // 2. Snapshot the DeltaNet state of every session that may reject.
    let mut snapshots: Vec<Option<DeltaNetSnapshot>> = Vec::with_capacity(states.len());
    for ((_, state), plan) in states.iter().zip(&plans) {
        if plan.drafts.is_empty() {
            snapshots.push(None);
            continue;
        }
        let dn = qwen35_dn(state);
        let mut snap = DeltaNetSnapshot::new_for(gpu, dn)
            .map_err(|e| format!("speculative decode: DeltaNet snapshot alloc: {e:?}"))?;
        snap.save_from(dn, gpu)
            .map_err(|e| format!("speculative decode: DeltaNet snapshot: {e:?}"))?;
        snapshots.push(Some(snap));
    }

    // 3. One fused forward over every session's feed + drafts.
    let row_tokens: Vec<Vec<u32>> = plans
        .iter()
        .map(|p| p.feed.iter().chain(&p.drafts).copied().collect())
        .collect();
    let lens: Vec<usize> = row_tokens.iter().map(Vec::len).collect();
    let total_rows: usize = lens.iter().sum();
    crate::qwen35_decode::qwen35_ensure_decode_prefill_batch_scratch(m, gpu, total_rows)?;
    // No drafts anywhere: nothing to verify, so take the ordinary forward, which
    // writes each session's `logits` itself — for a lone session the per-token
    // decode kernels, which beat a one-row batch forward (measured: +35% wall on
    // single-request prose when it went through the batch forward).
    let verify = plans.iter().any(|p| !p.drafts.is_empty());
    let forward = (|| -> Result<Option<(Vec<u32>, hipfire_rdna::GpuTensor)>, String> {
        let weights = m.q35_weights.as_ref().ok_or("qwen35 weights missing")?;
        let config = m.q35_config.as_ref().ok_or("qwen35 config missing")?;
        let pbs = m
            .q35_scratch
            .as_ref()
            .and_then(|s| s.prefill_batch.as_ref())
            .ok_or("qwen35 decode batch scratch missing")?;
        let moe = config.num_experts != 0 || config.has_shared_expert;
        if !verify && states.len() == 1 {
            let scratch = m.q35_scratch.as_ref().ok_or("qwen35 scratch missing")?;
            let state = &mut states[0].1;
            for (i, &token) in row_tokens[0].iter().enumerate() {
                qwen35::forward_scratch(
                    gpu,
                    weights,
                    config,
                    token,
                    state.cursor.seq_pos + i,
                    state.sequence_state.kv.as_mut().expect("qwen35 session KV"),
                    state
                        .sequence_state
                        .recurrent
                        .as_mut()
                        .expect("qwen35 session dn")
                        .as_any_mut()
                        .downcast_mut::<qwen35::DeltaNetState>()
                        .expect("qwen35 session dn"),
                    scratch,
                )
                .map_err(|e| format!("qwen35 decode advance: {e:?}"))?;
            }
            gpu.memcpy_dtod_auto(
                &state.logits.buf,
                &scratch.logits.buf,
                scratch.logits.buf.size(),
            )
            .map_err(|e| format!("qwen35 decode logits copy: {e:?}"))?;
            return Ok(None);
        }
        let mut rows: Vec<qwen35::DensePrefillSessionBatchRow<'_>> = states
            .iter_mut()
            .zip(&row_tokens)
            .map(|((_, state), tokens)| qwen35::DensePrefillSessionBatchRow {
                tokens,
                start_pos: state.cursor.seq_pos,
                kv_cache: state.sequence_state.kv.as_mut().expect("qwen35 session KV"),
                dn_state: state
                    .sequence_state
                    .recurrent
                    .as_mut()
                    .expect("qwen35 session dn")
                    .as_any_mut()
                    .downcast_mut::<qwen35::DeltaNetState>()
                    .expect("qwen35 session dn"),
                logits: &state.logits,
            })
            .collect();
        if !verify {
            let scratch = m.q35_scratch.as_ref().ok_or("qwen35 scratch missing")?;
            if moe {
                qwen35::forward_prefill_grouped_moe_session_batch(
                    gpu, weights, config, &mut rows, scratch, pbs,
                )
            } else {
                qwen35::forward_prefill_dense_session_batch(
                    gpu, weights, config, &mut rows, scratch, pbs,
                )
            }
            .map_err(|e| format!("qwen35 decode forward: {e:?}"))?;
            return Ok(None);
        }
        let logits = if moe {
            qwen35::forward_prefill_grouped_moe_session_batch_all_row_logits(
                gpu, weights, config, &mut rows, pbs,
            )
        } else {
            qwen35::forward_prefill_dense_session_batch_all_row_logits(
                gpu, weights, config, &mut rows, pbs,
            )
        }
        .map_err(|e| format!("qwen35 speculative decode forward: {e:?}"))?;
        let ids = gpu
            .alloc_tensor(&[total_rows], hipfire_rdna::DType::F32)
            .map_err(|e| format!("speculative decode: argmax alloc: {e:?}"))?;
        let argmax = gpu
            .argmax_f32_batched(&logits, &ids, config.vocab_size, total_rows)
            .and_then(|()| gpu.download_raw(&ids, total_rows * 4));
        let _ = gpu.free_tensor(ids);
        match argmax {
            Ok(bytes) => Ok(Some((
                bytes
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect(),
                logits,
            ))),
            Err(e) => {
                let _ = gpu.free_tensor(logits);
                Err(format!("speculative decode: argmax: {e:?}"))
            }
        }
    })();
    let verified = match forward {
        Ok(v) => v,
        Err(e) => {
            for snap in snapshots.into_iter().flatten() {
                snap.free_gpu(gpu);
            }
            return Err(e);
        }
    };
    let row_index = round_major_index(&lens);

    // 4. Accept, then commit or roll back each session.
    let config = m.q35_config.as_ref().ok_or("qwen35 config missing")?;
    let tokenizer = m.tokenizer.as_ref().ok_or("tokenizer missing")?;
    let mut lines = Vec::with_capacity(states.len());
    let mut outcome: Result<(), String> = Ok(());
    for (s, (((session, state), mut plan), snap)) in
        states.iter_mut().zip(plans).zip(snapshots).enumerate()
    {
        let (a, bonus) = match &verified {
            Some((argmax, _)) if !plan.drafts.is_empty() => {
                accept(&plan.drafts, |i| argmax[row_index[s][i]], plan.feed.len())
            }
            _ => (0, 0),
        };
        let rejected = a < plan.drafts.len();
        if !plan.drafts.is_empty() {
            if let Some(spec) = m.q35_registry.spec_sessions.get_mut(&session.session_id) {
                spec.record(plan.drafts.len(), a);
            }
        }
        let mut fresh: Vec<u32> = plan.drafts[..a].to_vec();
        if rejected {
            fresh.push(bonus);
        }
        // Stop at a terminator or the token budget, whichever comes first.
        for (i, &t) in fresh.iter().enumerate() {
            if plan.emitted.len() + i + 1 >= plan.remaining
                || is_terminator(config, tokenizer, im_end_token, t)
            {
                fresh.truncate(i + 1);
                plan.stop = true;
                break;
            }
        }
        if plan.emitted.len() >= plan.remaining {
            plan.stop = true;
        }
        if let Some(spec) = m.q35_registry.spec_sessions.get_mut(&session.session_id) {
            spec.ngram.observe(&fresh);
        }

        if rejected {
            // Undo the whole step; everything it committed feeds next step.
            if let Some(snap) = &snap {
                if let Err(e) = snap.restore_to(qwen35_dn_mut(state), gpu) {
                    outcome = Err(format!("speculative decode: DeltaNet restore: {e:?}"));
                }
            }
            if !plan.stop {
                pendings[s] = plan.feed.iter().chain(&fresh).copied().collect();
            }
        } else {
            // The plain forward wrote `logits` already; a verify leaves them in
            // the all-rows tensor.
            if let Some((_, logits)) = &verified {
                let last = row_index[s][row_tokens[s].len() - 1];
                let row = logits.sub_offset(last * vocab, vocab);
                if let Err(e) = gpu.memcpy_dtod_auto(&state.logits.buf, &row.buf, vocab * 4) {
                    outcome = Err(format!("speculative decode: logits copy: {e:?}"));
                }
            }
            state.cursor.seq_pos += row_tokens[s].len();
            state
                .cursor
                .conversation_tokens
                .extend_from_slice(&row_tokens[s]);
        }
        if let Some(snap) = snap {
            snap.free_gpu(gpu);
        }
        plan.emitted.extend_from_slice(&fresh);

        let by_length = plan.stop
            && plan.emitted.len() >= plan.remaining
            && !plan
                .emitted
                .last()
                .is_some_and(|&t| is_terminator(config, tokenizer, im_end_token, t));
        let text: Vec<u32> = plan
            .emitted
            .iter()
            .copied()
            .filter(|&t| !is_terminator(config, tokenizer, im_end_token, t))
            .collect();
        let logical_position =
            state.cursor.seq_pos + state.kv_cache().compact_offset + pendings[s].len();
        lines.push(serde_json::json!({
            "type": "generate_batch_decode_step_session_done",
            "id": envelope.id,
            "batch_id": envelope.batch_id,
            "session_id": session.id,
            "runtime_state_handle": session.session_id,
            "token": plan.emitted.first(),
            "tokens": plan.emitted,
            "text": tokenizer.decode(&text),
            "stop": plan.stop,
            "finish_reason": if !plan.stop { serde_json::Value::Null }
                else if by_length { "length".into() } else { "stop".into() },
            "logical_position": logical_position,
            "spec": { "drafted": row_tokens[s].len() - plan.feed.len(), "accepted": a },
        }));
    }
    if let Some((_, logits)) = verified {
        let _ = gpu.free_tensor(logits);
    }
    outcome.map(|()| lines)
}

fn qwen35_dn(state: &Qwen35RequestSessionState) -> &qwen35::DeltaNetState {
    state
        .sequence_state
        .recurrent
        .as_ref()
        .expect("qwen35 session dn")
        .as_any()
        .downcast_ref::<qwen35::DeltaNetState>()
        .expect("qwen35 session dn")
}

fn qwen35_dn_mut(state: &mut Qwen35RequestSessionState) -> &mut qwen35::DeltaNetState {
    state
        .sequence_state
        .recurrent
        .as_mut()
        .expect("qwen35 session dn")
        .as_any_mut()
        .downcast_mut::<qwen35::DeltaNetState>()
        .expect("qwen35 session dn")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accept_counts_the_agreeing_prefix_and_takes_the_next_choice() {
        // feed = [t], drafts d1..d3; model after t says d1, after d1 says d2,
        // after d2 says 99 (disagrees with d3).
        let argmax = [11, 12, 99, 7];
        assert_eq!(accept(&[11, 12, 13], |i| argmax[i], 1), (2, 99));
        // All accepted: the bonus is the model's choice after the last draft.
        assert_eq!(accept(&[11, 12, 99], |i| argmax[i], 1), (3, 7));
        // No drafts: the bonus is the choice after the last fed token.
        assert_eq!(accept(&[], |i| argmax[i], 1), (0, 11));
        // A pending run of 2 fed first: verification starts after its last row.
        // Row 0's choice (0) belongs to the pending run, not to a draft.
        let argmax = [0, 5, 6, 8];
        assert_eq!(accept(&[5, 6], |i| argmax[i], 2), (2, 8));
        assert_eq!(accept(&[6, 8], |i| argmax[i], 2), (0, 5));
    }

    #[test]
    fn round_major_index_interleaves_sessions_by_token() {
        assert_eq!(
            round_major_index(&[3, 1, 2]),
            vec![vec![0, 3, 5], vec![1], vec![2, 4]]
        );
    }

    #[test]
    fn draft_cap_stops_at_the_block_end_and_the_budget() {
        std::env::remove_var("HIPFIRE_BATCH_SPEC_MAX_DRAFT");
        // Position 120, feeding 1: rows 120..=127 fit, so 7 drafts.
        assert_eq!(draft_cap(120, 1, true, 100), 7);
        // Last slot of the block: no room.
        assert_eq!(draft_cap(127, 1, true, 100), 0);
        // Budget: a fresh token plus drafts must fit in `remaining`.
        assert_eq!(draft_cap(0, 1, true, 3), 2);
        // A pending run was emitted already, so drafts get the whole budget.
        assert_eq!(draft_cap(0, 2, false, 3), 3);
        assert_eq!(draft_cap(0, MAX_PENDING + 1, false, 100), 0);
    }
}
