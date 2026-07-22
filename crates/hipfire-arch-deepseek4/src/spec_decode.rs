// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! DeepSeek V4 speculative decoding via the built-in MTP head (DeepSeek V3 §4
//! Multi-Token Prediction).
//!
//! Pipeline:
//! 1. Generate K candidate tokens by iterating `mtp_forward` K times,
//!    seeding each step with the previous step's predicted token and
//!    the previous step's MTP-layer hidden state.
//! 2. Run the main DeepSeek V4 forward as `forward_prefill_batch_chunk` at B=K
//!    on `[committed_token, draft_1, draft_2, ..., draft_{K-1}]`.
//! 3. Compare main's top-1 at each verify position against the next
//!    draft's predicted token; accept the longest matching prefix.
//! 4. Return `accepted_tokens` + the main model's preferred token at
//!    the divergence position (to keep generation moving forward even
//!    on a rejected suffix).
//!
//! When all K drafts are accepted, the next call starts from
//! `accepted_tokens[K-1]` and the previously-cached hidden state.
//!
//! ## Status
//!
//! Skeleton only. The MTP forward (`forward::mtp_forward`) is currently
//! a stub that returns an error until the standard layer block runs
//! against `weights.mtp_layer` (tracked as M3 in
//! `docs/plans/deepseek4-mtp-requant-2026-05-20.md`). This module compiles
//! and exposes the public API but errors out at the first MTP step
//! until M3 lands.

use crate::deepseek4::{DeepseekV4Config, DeepseekV4State, DeepseekV4Weights};
use crate::forward::{self};
use crate::grammar;
use hipfire_rdna::Gpu;

/// One acceptance window of DeepSeek4 MTP speculative decoding.
///
/// This is the lightweight MTP result and is intentionally distinct from the
/// core `hipfire_specdecode::SpecStepResult` (the DeltaNet/DFlash step, which
/// carries `bonus_token`/`drafted`/`committed`/rollback+verify-graph modes this
/// path never computes). Its `n_proposed`/`n_accepted`/`accepted_tokens.len()`
/// map directly onto the unified `SpecMetrics::record_window(proposed, accepted,
/// committed)` at the daemon call site — do not re-merge it with the core type.
#[derive(Debug, Clone)]
pub struct SpecWindow {
    /// Tokens accepted this window (in emission order). At minimum
    /// always contains the verifier's preferred token at the
    /// divergence position; on full acceptance contains all K drafts.
    pub accepted_tokens: Vec<u32>,
    /// How many of the K drafts were accepted (longest matching
    /// prefix between drafts and main-model top-1 logits).
    pub n_accepted: usize,
    /// How many draft tokens were proposed (= K).
    pub n_proposed: usize,
}

/// Caller-owned grammar state for tool-call constrained speculative decode.
///
/// The daemon owns the tokenizer/decoded-vocab cache and DSML matcher. Passing
/// them here lets the MTP draft path and the verifier path apply the same
/// structural mask used by plain DeepSeek4 decoding, without making this module
/// depend on the daemon's request machinery.
pub struct SpecGrammar<'a> {
    pub matcher: &'a mut grammar::Matcher,
    pub decoded_vocab: &'a [String],
    pub mask: &'a mut Vec<bool>,
}

/// Run one speculative-decode acceptance window.
///
/// Inputs:
/// - `cfg`/`weights`/`state`/`gpu`: standard DeepSeek V4 runtime
/// - `last_token`: the most-recently-committed token (position N)
/// - `last_hidden`: optional cached hidden state at position N
///   (populated from the prior main forward); if `None`, the function
///   will run a 1-token main forward to materialize it
/// - `k`: number of draft tokens to propose
///
/// Returns the acceptance result.
///
/// Stub status: returns an error from the first `mtp_forward` call
/// until M3 lands.
/// Same as [`speculative_decode_step`] but takes a caller-owned PBS scratch
/// instead of allocating one per call. Allocating PBS internally (~30 small
/// GpuTensor allocations) costs measurable milliseconds at small K — caching
/// it once at session setup and passing it in eliminates that.
#[allow(clippy::too_many_arguments)]
pub fn speculative_decode_step_with_pbs(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &forward::PrefillBatchScratch,
    last_token: u32,
    last_position: u32,
    last_hidden: Option<&hipfire_rdna::GpuTensor>,
    k: usize,
) -> Result<SpecWindow, String> {
    speculative_decode_impl(
        cfg,
        weights,
        state,
        gpu,
        Some(pbs),
        last_token,
        last_position,
        last_hidden,
        k,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn speculative_decode_step_with_pbs_grammar(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    pbs: &forward::PrefillBatchScratch,
    last_token: u32,
    last_position: u32,
    last_hidden: Option<&hipfire_rdna::GpuTensor>,
    k: usize,
    matcher: &mut grammar::Matcher,
    decoded_vocab: &[String],
    grammar_mask: &mut Vec<bool>,
) -> Result<SpecWindow, String> {
    speculative_decode_impl(
        cfg,
        weights,
        state,
        gpu,
        Some(pbs),
        last_token,
        last_position,
        last_hidden,
        k,
        Some(SpecGrammar {
            matcher,
            decoded_vocab,
            mask: grammar_mask,
        }),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn speculative_decode_step(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    last_token: u32,
    last_position: u32,
    last_hidden: Option<&hipfire_rdna::GpuTensor>,
    k: usize,
) -> Result<SpecWindow, String> {
    speculative_decode_impl(
        cfg,
        weights,
        state,
        gpu,
        None,
        last_token,
        last_position,
        last_hidden,
        k,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn speculative_decode_impl(
    cfg: &DeepseekV4Config,
    weights: &DeepseekV4Weights,
    state: &mut DeepseekV4State,
    gpu: &mut Gpu,
    cached_pbs: Option<&forward::PrefillBatchScratch>,
    last_token: u32,
    last_position: u32,
    last_hidden: Option<&hipfire_rdna::GpuTensor>,
    k: usize,
    mut grammar: Option<SpecGrammar<'_>>,
) -> Result<SpecWindow, String> {
    if k == 0 {
        return Err("speculative_decode_step: k must be > 0".to_string());
    }
    if cfg.num_nextn_predict_layers == 0 || weights.mtp_layer.is_none() {
        return Err("speculative_decode_step: MTP layer not loaded — \
            quantize with deepseek4-q8-mtp + addon, or set HIPFIRE_DEEPSEEK4_LOAD_MTP=1"
            .to_string());
    }

    // ── 1. Pick the initial hidden state h_n ───────────────────────────
    // V3 §4: the first MTP step takes the post-layer-block hidden at
    // position N (= `last_position`). Caller can supply it via
    // `last_hidden` (faster — from a prior main forward); else we fall
    // back to `state.mtp_last_hidden`, populated by every `decode_step`
    // / `mtp_forward` call.
    //
    // Stored as a raw pointer so subsequent iterations can borrow
    // `state` mutably without re-aliasing this initial hidden. SAFETY:
    // both candidate sources live in stable VRAM allocations for the
    // duration of this function; mtp_forward writes ONLY to scratch
    // buffers + mtp_last_hidden (which is what step 1+ reads), never
    // to the `last_hidden` caller-passed tensor.
    let h_n_ptr: *const hipfire_rdna::GpuTensor = match last_hidden {
        Some(h) => h as *const _,
        None => {
            let h = state.mtp_last_hidden.as_ref().ok_or_else(|| {
                "speculative_decode_step: no hidden state available — \
                 run decode_step(last_token, last_position) first or \
                 pass last_hidden explicitly"
                    .to_string()
            })?;
            h as *const _
        }
    };

    // ── 2. K draft iterations of mtp_forward ──────────────────────────
    // Each call writes the post-layer-block stream-0 hidden back to
    // `state.mtp_last_hidden`, so step k+1 chains from step k's output.
    //
    // **state.n_tokens bookkeeping** (subtle): `attn_stub` (called by
    // mtp_forward via the standard layer block) reads `state.n_tokens`
    // to pick the SWA ring slot. After the caller's most recent
    // decode_step at position N, `state.n_tokens == N+1`. For MTP step
    // s at position N+1+s, we need `state.n_tokens == N+1+s` so the
    // SWA write lands in the right MTP-layer ring slot. We increment
    // between iterations and restore based on `n_accept` at the end.
    //
    // (forward_prefill_batch_chunk uses an explicit start_pos parameter
    // and doesn't touch state.n_tokens — verified.)
    let initial_n_tokens = state.n_tokens;
    let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
    let mut draft_matcher = grammar.as_ref().map(|g| (*g.matcher).clone());
    for step in 0..k {
        let next_token = if step == 0 {
            last_token
        } else {
            draft_tokens[step - 1]
        };
        // V3 paper §4: h_i^k = M_k @ Concat(norm(h_i^{k-1}), norm(e_{i+k})).
        // The MTP transformer block operates at position i (not i+1).
        // For step k=0 predicting T_{N+1}: i = N-1 = last_position.
        // For step k=s predicting T_{N+1+s}: i = N-1+s = last_position+s.
        // So position passed to mtp_forward (which sets RoPE phase + SWA
        // slot) is `last_position + step`, NOT `last_position + 1 + step`.
        // The off-by-one earlier was causing MTP attn_stub to write the
        // wrong SWA slot and RoPE to encode the wrong phase — accepted
        // rate measured at ~50% K=2 with the bug; fix is being tested.
        let position = last_position + step as u32;
        state.n_tokens = position as u64;

        // For step 0 we use h_n_ptr; for step k>0 we point at the
        // freshly-written state.mtp_last_hidden. Both go through a raw
        // pointer to decouple from state's borrow. SAFETY: as above,
        // these GpuTensors live in stable allocations and are only
        // READ by the GEMV chain inside mtp_forward (which writes to
        // distinct scratch + state.mtp_last_hidden each iteration).
        let hidden_ptr: *const hipfire_rdna::GpuTensor = if step == 0 {
            h_n_ptr
        } else {
            state.mtp_last_hidden.as_ref().ok_or_else(|| {
                format!(
                    "spec_decode: mtp_last_hidden missing after step {}",
                    step - 1
                )
            })? as *const _
        };
        let hidden: &hipfire_rdna::GpuTensor = unsafe { &*hidden_ptr };
        let mut logits =
            forward::mtp_forward(cfg, weights, state, gpu, hidden, next_token, position)?;
        if let (Some(g), Some(matcher)) = (grammar.as_mut(), draft_matcher.as_ref()) {
            apply_grammar_mask(matcher, g.decoded_vocab, g.mask, &mut logits);
        }
        let argmax = logits_argmax(&logits) as u32;
        draft_tokens.push(argmax);
        if let (Some(g), Some(matcher)) = (grammar.as_ref(), draft_matcher.as_mut()) {
            advance_matcher_token(matcher, g.decoded_vocab, argmax);
        }
    }

    // ── 3. Single B=K main verify pass ────────────────────────────────
    // Tokens to feed the verifier: the last committed token plus the
    // first K-1 drafts. The verifier outputs logits at K positions,
    // each predicting "what comes after my input token" — these are
    // the predictions we compare to the drafts.
    //
    //   verify_tokens[0] = last_token   → predicts pos N+1's token (= draft[0]'s target)
    //   verify_tokens[1] = draft[0]     → predicts pos N+2's token (= draft[1]'s target)
    //   ...
    //   verify_tokens[K-1] = draft[K-2] → predicts pos N+K  's token
    let verify_tokens: Vec<u32> = std::iter::once(last_token)
        .chain(draft_tokens.iter().take(k - 1).copied())
        .collect();
    debug_assert_eq!(verify_tokens.len(), k);

    // Use the caller-provided PBS if available; otherwise allocate one.
    // The owned variant exists so single-shot callers / tests still work
    // without threading a PBS through; the cached variant is the perf-
    // critical path used by tight spec-decode loops.
    let owned_pbs: Option<forward::PrefillBatchScratch> = match cached_pbs {
        Some(_) => None,
        None => Some(forward::PrefillBatchScratch::new(gpu, cfg, k)?),
    };
    // Run the verify body in a closure so a locally-allocated `owned_pbs` is
    // returned to the pool on EVERY exit path — `PrefillBatchScratch` has no
    // Drop and the body has several `?` early returns. Cached-PBS callers pass
    // `Some`, so `owned_pbs` is `None` for them and the free below is a no-op.
    let result = (|| -> Result<SpecWindow, String> {
        let pbs: &forward::PrefillBatchScratch =
            cached_pbs.unwrap_or_else(|| owned_pbs.as_ref().unwrap());
        if pbs.max_batch < k {
            return Err(format!(
                "spec_decode: cached PBS max_batch ({}) < k ({})",
                pbs.max_batch, k
            ));
        }
        // Spec-decode SWA-ring rewind (opt-in, pending GPU validation): snapshot
        // the K soon-to-be-evicted main-layer ring slots before the verify pass
        // overwrites them in place, so a partial accept can revert the
        // uncommitted ones. `base` = the verify pass's first position below.
        if swa_rewind::enabled() {
            swa_rewind::snapshot(cfg, state, gpu, (last_position + 1) as u64, k)?;
        }
        forward::forward_prefill_batch_chunk(
            cfg,
            weights,
            state,
            gpu,
            pbs,
            &verify_tokens,
            last_position + 1,
        )?;

        // ── 4. Per-position top-1 from the verifier ───────────────────────
        let all_logits =
            forward::final_norm_and_head_all_batched(cfg, weights, state, pbs, gpu, k)?;
        let mut verify_matcher = grammar.as_ref().map(|g| (*g.matcher).clone());

        // ── 5. Longest matching prefix → acceptance ────────────────────────
        //
        // In tool-call mode the verifier's preferred token is chosen after the
        // same DSML grammar mask as the non-spec decode path. The verifier matcher
        // advances only along the actually accepted prefix; at divergence, the
        // appended verifier token is legal for the grammar state reached by that
        // prefix.
        let mut accepted_tokens: Vec<u32> = Vec::with_capacity(k);
        let mut n_accept = 0usize;
        for (idx, &draft) in draft_tokens.iter().enumerate() {
            let main = match (grammar.as_mut(), verify_matcher.as_ref()) {
                (Some(g), Some(matcher)) => {
                    let mut logits = all_logits[idx].clone();
                    apply_grammar_mask(matcher, g.decoded_vocab, g.mask, &mut logits);
                    logits_argmax(&logits) as u32
                }
                _ => logits_argmax(&all_logits[idx]) as u32,
            };
            if draft == main {
                accepted_tokens.push(draft);
                n_accept += 1;
                if let (Some(g), Some(matcher)) = (grammar.as_ref(), verify_matcher.as_mut()) {
                    advance_matcher_token(matcher, g.decoded_vocab, draft);
                }
            } else {
                accepted_tokens.push(main);
                if let (Some(g), Some(matcher)) = (grammar.as_ref(), verify_matcher.as_mut()) {
                    advance_matcher_token(matcher, g.decoded_vocab, main);
                }
                break;
            }
        }

        if let Some(g) = grammar.as_mut() {
            for &tok in &accepted_tokens {
                advance_matcher_token(g.matcher, g.decoded_vocab, tok);
            }
        }

        // Spec-decode SWA-ring rewind (opt-in): revert the ring slots the
        // verify pass wrote for uncommitted, next-decode-unfixed draft
        // positions (verify columns [accepted_len + 1, k)). No-op on full
        // accept or when the rewind path is disabled. Same `base`/`k` as the
        // snapshot above.
        if swa_rewind::enabled() {
            swa_rewind::restore(
                cfg,
                state,
                gpu,
                (last_position + 1) as u64,
                accepted_tokens.len(),
                k,
            )?;
        }

        // ── 6. Refresh state.mtp_last_hidden from the verify pass ──────────
        // Capture the FULL [hc_mult, hidden] residual stream of
        // pbs.streams_batch[accepted_tokens.len() - 1, :, :]. Matches the
        // antirez/ds4 reference MTP HC plumbing (see project memory entry
        // `project_deepseek4_mtp_hc_plumbing_gap`). Stream-0-only capture was what
        // discarded 75% of HC signal and pinned K=2 accept at ~50%.
        {
            let last_idx = accepted_tokens.len() - 1;
            let stream_len = cfg.hc_mult * cfg.hidden_size;
            let off = last_idx * stream_len;
            let last_full = pbs.streams_batch.sub_offset(off, stream_len);
            let need_realloc = state
                .mtp_last_hidden
                .as_ref()
                .map(|t| t.numel() != stream_len)
                .unwrap_or(true);
            if need_realloc {
                state.mtp_last_hidden = Some(
                    gpu.alloc_tensor(&[cfg.hc_mult, cfg.hidden_size], hipfire_rdna::DType::F32)
                        .map_err(|e| format!("alloc mtp_last_hidden: {e:?}"))?,
                );
            }
            let dst = state.mtp_last_hidden.as_ref().unwrap();
            gpu.memcpy_dtod_auto(&dst.buf, &last_full.buf, stream_len * 4)
                .map_err(|e| format!("capture verify-pass full HC streams: {e:?}"))?;
        }

        // ── 7. Restore state.n_tokens to the post-accept position ─────────
        // Caller's next forward expects `state.n_tokens` == (next position
        // to be processed). We emitted `accepted_tokens.len()` tokens
        // starting at position last_position+1, so the next free position
        // is last_position + 1 + accepted_tokens.len().
        //
        // Why this isn't simply `initial_n_tokens + accepted_tokens.len()`:
        // initial_n_tokens (== last_position + 1) is the position of the
        // FIRST emitted token. After emitting all accepted_tokens, next
        // position is initial_n_tokens + accepted_tokens.len(). Same thing.
        //
        // Stale-cache caveat: MTP layer's SWA cache has writes at positions
        // [N+1 .. N+K] from the draft loop; the main layers' SWA caches
        // have writes at the same positions from the verify pass. Positions
        // BEYOND n_accept were computed using rejected draft tokens (input
        // mismatch with what the caller will treat as committed). Those
        // entries get naturally invalidated when the caller's next forward
        // overwrites them via ring buffer. Bug only manifests when a
        // forward READS those stale slots before overwriting — happens
        // only in narrow windows and is documented as a production-hardening
        // follow-up.
        //
        // Correct fix shape: speculative verify must either write into scratch
        // cache state and commit only the accepted prefix, or explicitly
        // invalidate/rewind every per-layer cache slot beyond n_accept before
        // returning. Moving n_tokens alone is not sufficient, because SWA ring
        // indices can still alias rejected-token cache entries on a later read.
        state.n_tokens = initial_n_tokens + accepted_tokens.len() as u64;

        Ok(SpecWindow {
            accepted_tokens,
            n_accepted: n_accept,
            n_proposed: k,
        })
    })();
    if let Some(p) = owned_pbs {
        p.free_gpu(gpu);
    }
    result
}

/// Standalone helper: compute argmax of a [vocab] logits vector.
/// Used by `speculative_decode_step` to pick the verifier's preferred
/// token at the divergence position.
#[inline]
pub fn logits_argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = logits[0];
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// Spec-decode SWA-ring rewind for DeepSeek V4 — fixes the stale-slot
/// corruption tracked in BUGS.md ("Stale SWA ring-buffer slots after
/// speculative reject").
///
/// The verify pass writes K draft positions into the *real* per-layer SWA ring
/// (slot = `pos % sliding_window`). On a partial accept only `state.n_tokens`
/// is rewound; the uncommitted ring slots keep rejected-draft K/V. The next
/// decode step overwrites exactly ONE of them (the corrected token's slot, at
/// verify column `accepted_len`), so the still-stale columns are
/// `[accepted_len + 1, k)`. Post-wrap (context ≥ `sliding_window`) those alias
/// positions that are still inside the next forward's window and silently
/// corrupt attention. We snapshot the soon-to-be-evicted slots before the
/// verify and restore the uncommitted ones after the accept, so the ring
/// matches the pure-AR frontier.
///
/// Scope: only the modular SWA ring (`swa_k`/`swa_v`) aliases. `full_k_cache`
/// is absolute-position-indexed and causally safe (a future stale row is never
/// gathered, and the march overwrites it before it becomes past); the MTP
/// layer's ring only affects draft acceptance (verify still guarantees correct
/// output). Neither is touched here.
///
/// PENDING GPU VALIDATION — gated OFF by default behind
/// `HIPFIRE_DEEPSEEK4_SPEC_KV_REWIND=1`. The pure slot arithmetic
/// ([`slot_subranges`]) is unit-tested; the on-device copy must still be
/// validated with an AR-vs-spec losslessness A/B on a runnable deepseek4 model
/// (the divergence appears only post-wrap with k ≥ n_accept + 3) before this is
/// defaulted on.
pub mod swa_rewind {
    use super::{DeepseekV4Config, DeepseekV4State};
    use hipfire_rdna::{DType, Gpu};

    /// Snapshot column capacity = max supported spec K (spec K is small).
    const SNAP_COLS: usize = 16;

    #[inline]
    pub fn enabled() -> bool {
        std::env::var("HIPFIRE_DEEPSEEK4_SPEC_KV_REWIND")
            .ok()
            .as_deref()
            == Some("1")
    }

    /// Split the consecutive absolute positions `[p_start, p_end)` into
    /// `(ring_slot, pack_col, len)` runs. Position `p` maps to ring slot
    /// `p % win` and snapshot column `p - snap_base`. The ring wraps at `win`,
    /// so this yields 1..=2 contiguous runs. Pure arithmetic — unit-tested.
    pub fn slot_subranges(
        snap_base: u64,
        p_start: u64,
        p_end: u64,
        win: usize,
    ) -> Vec<(usize, usize, usize)> {
        let winu = win as u64;
        let mut out = Vec::new();
        let mut p = p_start;
        while p < p_end {
            let to_wrap = winu - (p % winu);
            let len = to_wrap.min(p_end - p);
            out.push(((p % winu) as usize, (p - snap_base) as usize, len as usize));
            p += len;
        }
        out
    }

    /// Save the K ring slots `[base, base + k)` that the verify pass is about
    /// to overwrite, for every main layer with an allocated SWA ring. Call
    /// immediately before the batched verify forward.
    pub fn snapshot(
        cfg: &DeepseekV4Config,
        state: &mut DeepseekV4State,
        gpu: &mut Gpu,
        base: u64,
        k: usize,
    ) -> Result<(), String> {
        if k == 0 || k > SNAP_COLS {
            return Ok(());
        }
        let rows = cfg.num_key_value_heads * cfg.head_dim;
        let win = cfg.sliding_window;
        let sub = slot_subranges(base, base, base + k as u64, win);
        for l in 0..cfg.num_hidden_layers {
            let attn = &mut state._attention[l];
            if attn.swa_k.is_none() || attn.swa_v.is_none() {
                continue;
            }
            if attn.swa_k_snap.is_none() {
                attn.swa_k_snap = Some(
                    gpu.alloc_tensor(&[rows, SNAP_COLS], DType::F32)
                        .map_err(|e| format!("alloc swa_k_snap l{l}: {e:?}"))?,
                );
                attn.swa_v_snap = Some(
                    gpu.alloc_tensor(&[rows, SNAP_COLS], DType::F32)
                        .map_err(|e| format!("alloc swa_v_snap l{l}: {e:?}"))?,
                );
            }
            let sk = attn.swa_k.as_ref().unwrap();
            let sv = attn.swa_v.as_ref().unwrap();
            let nk = attn.swa_k_snap.as_ref().unwrap();
            let nv = attn.swa_v_snap.as_ref().unwrap();
            for &(slot, pack, len) in &sub {
                gpu.strided_copy_2d(sk, slot, win, nk, pack, SNAP_COLS, rows, len, false)
                    .map_err(|e| format!("swa_k snapshot l{l}: {e:?}"))?;
                gpu.strided_copy_2d(sv, slot, win, nv, pack, SNAP_COLS, rows, len, false)
                    .map_err(|e| format!("swa_v snapshot l{l}: {e:?}"))?;
            }
        }
        Ok(())
    }

    /// Restore the uncommitted ring slots after a partial accept. The stale
    /// verify columns are `[accepted_len + 1, k)` (see module docs); a no-op on
    /// a full accept, when `accepted_len + 1 >= k`, or before any snapshot ran.
    /// `base` and `k` MUST match the preceding [`snapshot`] call.
    pub fn restore(
        cfg: &DeepseekV4Config,
        state: &mut DeepseekV4State,
        gpu: &mut Gpu,
        base: u64,
        accepted_len: usize,
        k: usize,
    ) -> Result<(), String> {
        if k == 0 || k > SNAP_COLS {
            return Ok(());
        }
        let lo = accepted_len + 1;
        if lo >= k {
            return Ok(());
        }
        let rows = cfg.num_key_value_heads * cfg.head_dim;
        let win = cfg.sliding_window;
        // Stale positions [base + lo, base + k); snapshot columns [lo, k).
        let sub = slot_subranges(base, base + lo as u64, base + k as u64, win);
        for l in 0..cfg.num_hidden_layers {
            let attn = &state._attention[l];
            let (Some(sk), Some(sv), Some(nk), Some(nv)) = (
                attn.swa_k.as_ref(),
                attn.swa_v.as_ref(),
                attn.swa_k_snap.as_ref(),
                attn.swa_v_snap.as_ref(),
            ) else {
                continue;
            };
            for &(slot, pack, len) in &sub {
                gpu.strided_copy_2d(nk, pack, SNAP_COLS, sk, slot, win, rows, len, false)
                    .map_err(|e| format!("swa_k restore l{l}: {e:?}"))?;
                gpu.strided_copy_2d(nv, pack, SNAP_COLS, sv, slot, win, rows, len, false)
                    .map_err(|e| format!("swa_v restore l{l}: {e:?}"))?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::slot_subranges;

        #[test]
        fn subranges_no_wrap() {
            // base=100, save k=3 → positions 100..103 (win=128): one run.
            assert_eq!(slot_subranges(100, 100, 103, 128), vec![(100, 0, 3)]);
        }

        #[test]
        fn subranges_wrap() {
            // base=126, k=4 → positions 126..130 wrap at 128.
            // slots 126,127,0,1 ; packed cols 0,1,2,3.
            assert_eq!(
                slot_subranges(126, 126, 130, 128),
                vec![(126, 0, 2), (0, 2, 2)]
            );
        }

        #[test]
        fn restore_subset_across_wrap() {
            // Snapshot base=126,k=4; restore stale cols [lo=2,4) → positions
            // 128..130 → slots 0,1 from packed cols 2,3.
            assert_eq!(slot_subranges(126, 128, 130, 128), vec![(0, 2, 2)]);
        }

        #[test]
        fn empty_range_yields_nothing() {
            // k=2 always leaves the stale range empty (lo=accepted_len+1>=k).
            assert!(slot_subranges(100, 102, 102, 128).is_empty());
        }
    }
}

fn apply_grammar_mask(
    matcher: &grammar::Matcher,
    decoded_vocab: &[String],
    mask: &mut Vec<bool>,
    logits: &mut [f32],
) {
    if matcher.is_free() || decoded_vocab.is_empty() {
        return;
    }
    if mask.len() < decoded_vocab.len() {
        mask.resize(decoded_vocab.len(), true);
    }
    matcher.token_mask(decoded_vocab, mask);
    grammar::Matcher::apply_mask_to_logits(mask, logits);
}

fn advance_matcher_token(matcher: &mut grammar::Matcher, decoded_vocab: &[String], token: u32) {
    if let Some(text) = decoded_vocab.get(token as usize) {
        matcher.advance(text);
    }
}
