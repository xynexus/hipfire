// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Logit-space sampling: top-p, temperature, repeat_penalty, and
//! single-token attractor blocking. Wraps GPU-dispatched sampling
//! kernels. One entry point for any caller (daemon, examples, future
//! arch ports).
//!
//! # Why this module
//!
//! Sampling primitives (top-p kernel call, repeat-penalty window upload,
//! attractor `-INF` writes, RNG threading) used to live across
//! `hipfire_runtime::llama` (CPU primitives + GPU launch wrappers) and
//! `crates/hipfire-daemon/src/main.rs` (call-site glue). New arch ports either
//! reached into llama.rs internals or duplicated the host-side prep.
//! This module gives every caller one entry point: [`sample`], with
//! [`SamplerConfig`] holding the policy knobs.
//!
//! # Behavior preservation
//!
//! [`sample`] is a pure call-site refactor. It delegates to the same
//! `Gpu::sample_top_p` kernel and the same `memcpy_htod` for the repeat
//! window and attractor `-INF` writes that the daemon used inline. The
//! same `(logits, history, temp, top_p, repeat_penalty, repeat_window,
//! blocked_tokens, rng_state)` tuple produces the same `next_token`
//! before and after PR 3.
//!
//! # Conditional vs unconditional blocking
//!
//! The unclosed-opener attractor block (#111) decides at the call site
//! which token to block (the opener) based on a depth count over recent
//! history. The decision lives at the call site; the resulting set of
//! token IDs is passed in as [`SamplerConfig::blocked_tokens`]. The
//! sampler treats them as unconditional `-INF` writes — it does not
//! reimplement the depth counter.

use hipfire_rdna::{Gpu, GpuTensor};

pub use hipfire_generate::sampler::{collect_unclosed_attractor_blocks, SamplerConfig};

/// Sample one token from a GPU-resident `logits` tensor.
///
/// Pre-dispatch host work, in order (matches the daemon's pre-PR3
/// inline sequence so byte-identical token streams are preserved):
///
///  1. Upload the trailing `min(history.len(), repeat_window,
///     repeat_buf_capacity)` tokens of `history` into `repeat_buf`.
///  2. Write `-INF` to `logits` at every offset in
///     `cfg.blocked_tokens` (one 4-byte H2D copy each).
///  3. Launch `Gpu::sample_top_p` (top-K + softmax + top-p + RNG +
///     argmax-on-greedy, all on GPU). One 8-byte D2H syncs the
///     `(token, new_rng)` result.
///
/// `rng_state` is mutated in place. For greedy (`temperature == 0.0`)
/// the value is unused but is still threaded through the kernel.
///
/// # Buffer types
///
/// `logits` is the model's output logits tensor (shape `[vocab_size]`,
/// dtype F32). `sample_buf` and `repeat_buf` are scratch buffers from
/// `llama::ForwardScratch`; the caller owns them. This matches the
/// existing pre-PR3 daemon signature exactly — we do not redesign the
/// argument shape.
/// The value a decode path should start its sampler RNG from.
///
/// Reads the `sampler_rng` config option once per process:
///
/// * `"fixed"` (default) — the constant `0x13579BDF` every call site used to
///   hardcode. Reproducible, and what every existing deployment already gets.
/// * `"random"` — a fresh entropy-derived value per call, so two concurrent
///   streams at `temperature>0` are independent rather than identical.
///
/// **Why this exists.** The constant was inlined at seven-plus sites
/// (`generate.rs`, `qwen35_prefill.rs`, …), which meant every `temperature>0`
/// request sampled the same stream of randomness: identical prompts produced
/// identical tokens by construction, and no request could be made independent of
/// another because nothing on the wire carries a seed. Routing every site
/// through one accessor is also what stops the two modes drifting apart, the way
/// the f16-state dither drifted from its tree kernel.
///
/// **Greedy decode never consults the RNG**, so `temperature == 0` output — and
/// every greedy baseline in the tiny gates — is identical under both modes. That
/// is the property to check first if this is ever suspected of moving numbers.
pub fn initial_rng_state() -> u32 {
    /// The historical constant. Kept as the `"fixed"` value rather than being
    /// re-derived, so `"fixed"` is bit-identical to the old inlined literal.
    const FIXED_SEED: u32 = 0x13579BDF;

    static MODE_IS_RANDOM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let random = *MODE_IS_RANDOM.get_or_init(|| {
        hipfire_config::load_config()
            .sampler_rng
            .eq_ignore_ascii_case("random")
    });
    if !random {
        return FIXED_SEED;
    }
    // Entropy per CALL, not per process: two streams admitted in the same
    // process must not share a stream of randomness, which is the whole point.
    // A OnceLock here would reintroduce exactly the sharing this replaces.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() ^ (d.as_secs() as u32))
        .unwrap_or(FIXED_SEED);
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // Mix the counter in: two calls inside the same nanosecond tick would
    // otherwise collide, and `subsec_nanos` is coarser than a nanosecond on
    // plenty of hosts.
    let mixed = nanos.wrapping_mul(2654435761).rotate_left(13) ^ n.wrapping_mul(2246822519);
    // 0 is a degenerate state for the xorshift the sampler advances.
    if mixed == 0 {
        FIXED_SEED
    } else {
        mixed
    }
}

pub fn sample(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    sample_buf: &GpuTensor,
    repeat_buf: &GpuTensor,
    vocab_size: usize,
    history: &[u32],
    cfg: &SamplerConfig,
    rng_state: &mut u32,
) -> u32 {
    // Step 1: upload the repeat-penalty window. The kernel reads
    // `repeat_tokens[0..effective_window]`, so we only have to upload
    // the tokens that will actually be read. An empty scope is a no-op
    // (matches the first-sample case in the daemon, which used to
    // skip the htod when `bytes0` was empty).
    let buf_cap_tokens = repeat_buf.buf.size() / 4;
    let window = cfg.repeat_window.min(buf_cap_tokens);
    let scope_start = history.len().saturating_sub(window);
    let scope = &history[scope_start..];
    if !scope.is_empty() {
        let bytes: Vec<u8> = scope.iter().flat_map(|t| t.to_ne_bytes()).collect();
        let _ = gpu.hip.memcpy_htod(&repeat_buf.buf, &bytes);
    }

    // Step 2: apply unconditional blocked tokens. One 4-byte H2D per
    // token. The daemon path used `gpu_block_attractor_unclosed` which
    // wrote `-INF` to a single offset only when the depth counter
    // tripped; here the caller has already done the depth math and
    // accumulated the token IDs into `cfg.blocked_tokens`.
    if !cfg.blocked_tokens.is_empty() {
        let neg_inf: [u8; 4] = f32::NEG_INFINITY.to_ne_bytes();
        for &tok in &cfg.blocked_tokens {
            if (tok as usize) < vocab_size {
                let _ = gpu
                    .hip
                    .memcpy_htod_offset(&logits.buf, (tok as usize) * 4, &neg_inf);
            }
        }
    }

    // Step 3: GPU sample. The kernel does:
    //   - top-K = 20 from raw logits
    //   - apply repeat_penalty over `repeat_buf[0..scope.len()]`
    //   - softmax(top-K) with temperature scaling
    //   - top-p truncation
    //   - RNG draw + argmax-on-greedy fallback
    //   - writeback (token_id, new_rng) to `sample_buf`
    //   - 8-byte D2H sync (returned by the wrapper)
    //
    // A GPU sample can fail with a transient kernel launch / readback error
    // (e.g. an APU MES hiccup / brief device wedge). Panicking here would take
    // down the whole daemon and every in-flight request, so instead retry a
    // few times and then fall back to CPU sampling on a host copy of the
    // logits. The blocked-token `-INF` writes from step 2 are already in the
    // logits buffer, so they survive the readback; `sample_cpu` re-applies the
    // repeat/presence/frequency penalties (the GPU kernel applies those
    // internally without mutating the buffer, so there is no double-penalty).
    const GPU_SAMPLE_ATTEMPTS: usize = 3;
    let mut last_err = None;
    for attempt in 1..=GPU_SAMPLE_ATTEMPTS {
        match gpu.sample_top_p_pf(
            logits,
            sample_buf,
            repeat_buf,
            vocab_size,
            cfg.temperature,
            cfg.top_p,
            cfg.top_k,
            *rng_state,
            scope.len(),
            cfg.repeat_penalty,
            cfg.presence_penalty,
            cfg.frequency_penalty,
        ) {
            Ok((tok, new_rng)) => {
                *rng_state = new_rng;
                return tok;
            }
            Err(e) => {
                eprintln!(
                    "sampler: GPU sample_top_p failed (attempt {attempt}/{GPU_SAMPLE_ATTEMPTS}): {e:?}"
                );
                last_err = Some(e);
            }
        }
    }

    // GPU sampling did not recover after retries. Degrade to CPU sampling so
    // the request fails soft (or continues) instead of aborting the process.
    match gpu.download_f32(logits) {
        Ok(mut host_logits) => {
            let n = host_logits.len().min(vocab_size);
            eprintln!(
                "sampler: GPU sample failed after {GPU_SAMPLE_ATTEMPTS} attempts ({last_err:?}); \
                 falling back to CPU sampling"
            );
            // Continue the CALLER's stream rather than a process-global one:
            // a fallback must not make this request's sampling depend on
            // whatever else happened to be sampling at the time.
            let mut fallback_rng = SamplerRng::from_seed(*rng_state);
            let tok = sample_cpu(&mut host_logits[..n], history, cfg, &mut fallback_rng);
            *rng_state = fallback_rng.state();
            tok
        }
        Err(readback_err) => {
            // The device is unusable (even the logits readback failed). Emit a
            // deterministic fallback token with a loud log rather than panic;
            // the decode loop's stop/EOS handling winds the request down.
            eprintln!(
                "sampler: GPU sample AND logits readback both failed \
                 (sample_err={last_err:?}, readback_err={readback_err:?}); emitting fallback token 0"
            );
            0
        }
    }
}

/// CPU-only fallback: same math as [`sample`] but operates on a host
/// `logits` slice. Used by the VL path (`generate_vl` in daemon.rs)
/// where the argmax/top-p selection runs after a CPU-side
/// `apply_ngram_block` + `apply_repeat_penalty` pass that has no GPU
/// equivalent.
///
/// This is a thin wrapper over `llama::apply_repeat_penalty` +
/// `llama::sample_top_p` that exists so call sites have one import
/// path; the math is unchanged.
pub fn sample_cpu(
    logits: &mut [f32],
    history: &[u32],
    cfg: &SamplerConfig,
    rng: &mut SamplerRng,
) -> u32 {
    if cfg.repeat_penalty != 1.0 && cfg.repeat_window > 0 {
        apply_repeat_penalty(logits, history, cfg.repeat_window, cfg.repeat_penalty);
    }
    if (cfg.presence_penalty > 0.0 || cfg.frequency_penalty > 0.0) && cfg.repeat_window > 0 {
        let start = history.len().saturating_sub(cfg.repeat_window);
        let mut counts = std::collections::HashMap::<u32, f32>::new();
        for &tok in &history[start..] {
            *counts.entry(tok).or_insert(0.0) += 1.0;
        }
        for (tok, count) in counts {
            if (tok as usize) < logits.len() {
                logits[tok as usize] -= cfg.frequency_penalty * count + cfg.presence_penalty;
            }
        }
    }
    for &tok in &cfg.blocked_tokens {
        if (tok as usize) < logits.len() {
            logits[tok as usize] = f32::NEG_INFINITY;
        }
    }
    sample_top_k_top_p(logits, cfg.temperature, cfg.top_k, cfg.top_p, rng)
}

// ─── CPU sampling primitives (relocated from llama.rs in the de-llama cleanup) ───

/// Sample the next token from logits using argmax (greedy).
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0 as u32
}

/// Sample the next token using temperature + top-k + top-p (nucleus) sampling.
/// Qwen3 recommended: temperature=0.7, top_k=20, top_p=0.8
///
/// Single pass over raw logits to find top-K by value (no softmax on 151K vocab).
/// Softmax only computed on the K=20 finalists.
// ─── Sampling configuration ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub think_temp: f32,
    pub answer_temp: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
}

impl SamplingConfig {
    /// Text-only thinking model (Qwen3.5 text inference).
    pub fn text_thinking() -> Self {
        Self {
            think_temp: 0.3,
            answer_temp: 0.3,
            top_p: 0.8,
            repeat_penalty: 1.15,
            repeat_window: 128,
        }
    }
    /// VL thinking model.
    pub fn vl_thinking() -> Self {
        Self {
            think_temp: 0.3,
            answer_temp: 0.3,
            top_p: 0.8,
            repeat_penalty: 1.15,
            repeat_window: 128,
        }
    }
    /// Simple greedy-ish sampling (no think/answer split).
    pub fn simple() -> Self {
        Self {
            think_temp: 0.7,
            answer_temp: 0.3,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_window: 64,
        }
    }
}

/// Apply repeat penalty to logits in-place.
pub fn apply_repeat_penalty(logits: &mut [f32], history: &[u32], window: usize, penalty: f32) {
    let start = history.len().saturating_sub(window);
    let recent = &history[start..];

    // Count frequency of each token in the window, then apply penalty
    // scaled by count. A token seen once gets penalty^1, seen 3 times
    // gets penalty^3. This lets common words reappear naturally while
    // strongly suppressing actual repetition loops.
    // Also apply recency decay: tokens near the end of the window get
    // full penalty, tokens near the start get reduced penalty.
    let window_len = recent.len() as f32;
    let mut counts = std::collections::HashMap::<u32, (u32, f32)>::new(); // (count, closest_position_ratio)
    for (i, &t) in recent.iter().enumerate() {
        let recency = (i as f32 + 1.0) / window_len; // 0→1, higher = more recent
        let entry = counts.entry(t).or_insert((0, 0.0));
        entry.0 += 1;
        if recency > entry.1 {
            entry.1 = recency;
        }
    }

    for (&t, &(count, recency)) in &counts {
        if (t as usize) < logits.len() {
            // Effective penalty: base^(count * recency), capped at 1.5x.
            // Without the cap, "the" appearing 8x recently gets 1.15^8 = 3x suppression,
            // which collectively flattens the distribution after ~400 tokens.
            // The cap ensures no single token is suppressed more than 50%, keeping
            // the natural vocabulary accessible even in long generation.
            let effective = penalty.powf(count as f32 * recency).min(1.5);
            if logits[t as usize] > 0.0 {
                logits[t as usize] /= effective;
            } else {
                logits[t as usize] *= effective;
            }
        }
    }
}

/// N-gram repeat detection: if the last `n` tokens in history match an earlier n-gram,
/// set the logit of the token that followed that earlier occurrence to -inf.
/// This breaks phrase-level loops that token-level repeat penalty misses.
/// Checks n-grams of sizes 3, 4, 5, 6 for robustness.
pub fn apply_ngram_block(logits: &mut [f32], history: &[u32]) {
    if history.len() < 4 {
        return;
    }
    for ngram_size in [3, 4, 5, 6] {
        if history.len() <= ngram_size {
            continue;
        }
        let suffix = &history[history.len() - ngram_size..];
        // Scan history for earlier occurrences of this n-gram
        let search_end = history.len() - ngram_size;
        for i in 0..search_end {
            if i + ngram_size >= history.len() {
                break;
            }
            if history[i..i + ngram_size] == *suffix {
                // Found a match — the token that followed this earlier n-gram
                // is what the model wants to repeat. Block it.
                let next_tok = history[i + ngram_size];
                if (next_tok as usize) < logits.len() {
                    logits[next_tok as usize] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// Single-token attractor block for special tokens. Counts how many times
/// `token_id` appears in the last `window` tokens of `history`; if it is
/// at or above `threshold`, sets that token's logit to `-INF` so the
/// next sample picks something else. Targets MQ4 single-token attractors
/// on tokens that have no paired closer (e.g. a runaway emit of a
/// solo special). For paired open/close tokens like `<tool_call>` /
/// `</tool_call>`, prefer `apply_unclosed_attractor_block` — it triggers
/// before the model can stack a second nested opener that breaks
/// downstream regex parsers (see #111 codex review).
pub fn apply_special_token_attractor_block(
    logits: &mut [f32],
    history: &[u32],
    token_id: u32,
    window: usize,
    threshold: usize,
) {
    if (token_id as usize) >= logits.len() || threshold == 0 || window == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    let count = history[start..].iter().filter(|&&t| t == token_id).count();
    if count >= threshold {
        logits[token_id as usize] = f32::NEG_INFINITY;
    }
}

/// Open/close-paired attractor block for structured special tokens
/// (`<tool_call>`/`</tool_call>`, `<think>`/`</think>`).
///
/// Counts unclosed openers in the last `window` tokens — `opens - closes`,
/// floored at zero. When the running depth reaches `threshold`, sets
/// `open_id`'s logit to `-INF` so the next sample cannot stack another
/// nested opener. With `threshold = 2`, a second consecutive opener
/// without an intervening closer is the last one the decoder is allowed
/// to emit; the third+ are blocked. The downstream Hermes JSON parser
/// tolerates a single nested opener by stripping the leading repeat before
/// JSON parse.
///
/// The depth saturates at 0 from below: a stray closer at the start of
/// the window doesn't push depth negative and create false-allow.
pub fn apply_unclosed_attractor_block(
    logits: &mut [f32],
    history: &[u32],
    open_id: u32,
    close_id: u32,
    window: usize,
    threshold: usize,
) {
    if (open_id as usize) >= logits.len() || threshold == 0 || window == 0 {
        return;
    }
    let start = history.len().saturating_sub(window);
    let mut depth: i32 = 0;
    for &t in &history[start..] {
        if t == open_id {
            depth += 1;
        } else if t == close_id && depth > 0 {
            depth -= 1;
        }
    }
    if depth >= threshold as i32 {
        logits[open_id as usize] = f32::NEG_INFINITY;
    }
}

pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32, rng: &mut SamplerRng) -> u32 {
    sample_top_k_top_p(logits, temperature, 20, top_p, rng)
}

/// Sample with a caller-selected top-k cutoff before nucleus truncation.
///
/// The shared GPU sampler supports up to 64 candidates. Clamp the CPU path to
/// the same range so CPU fallback and GPU execution retain the same candidate
/// set. `top_k == 0` selects the backend maximum (64).
pub fn sample_top_k_top_p(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut SamplerRng,
) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    let top_p = top_p.clamp(0.0, 1.0);
    let top_k = if top_k == 0 { 64 } else { top_k.clamp(1, 64) };

    let inv_temp = 1.0 / temperature;

    // Single pass: find max AND top-K indices from raw logits simultaneously.
    // Uses a fixed-size array (no heap alloc) with manual min-tracking.
    let mut topk_val = vec![f32::NEG_INFINITY; top_k];
    let mut topk_idx = vec![0u32; top_k];
    let mut min_pos = 0usize; // index of smallest element in topk
    let mut min_val = f32::NEG_INFINITY;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &l) in logits.iter().enumerate() {
        if l > max_logit {
            max_logit = l;
        }
        if l > min_val {
            topk_val[min_pos] = l;
            topk_idx[min_pos] = i as u32;
            // Find new min
            min_val = f32::INFINITY;
            for j in 0..top_k {
                if topk_val[j] < min_val {
                    min_val = topk_val[j];
                    min_pos = j;
                }
            }
        }
    }

    // Softmax only the K candidates (temperature-scaled)
    let mut probs = vec![0.0f32; top_k];
    let mut sum = 0.0f32;
    for i in 0..top_k {
        let p = ((topk_val[i] - max_logit) * inv_temp).exp();
        probs[i] = p;
        sum += p;
    }

    // Sort descending by probability (insertion sort on 20 elements)
    let mut order: Vec<usize> = (0..top_k).collect();
    for i in 1..top_k {
        let mut j = i;
        while j > 0 && probs[order[j]] > probs[order[j - 1]] {
            order.swap(j, j - 1);
            j -= 1;
        }
    }

    // Match the GPU kernel exactly: find the nucleus cutoff first, then draw
    // once from the renormalized retained mass with one xorshift32 advance.
    let threshold = top_p * sum;
    let mut trunc_sum = 0.0_f32;
    let mut trunc_k = top_k;
    for (i, &k) in order.iter().enumerate() {
        trunc_sum += probs[k];
        if trunc_sum >= threshold {
            trunc_k = i + 1;
            break;
        }
    }
    let r = rng.next_f32() * trunc_sum;
    let mut cumulative = 0.0_f32;
    for &k in order.iter().take(trunc_k) {
        cumulative += probs[k];
        if cumulative >= r {
            return topk_idx[k];
        }
    }
    topk_idx[order[trunc_k - 1]]
}

/// Apply the repeat penalty in-place to a specific subset of (token_id, value)
/// candidates, rather than the full 151k-entry logits vector. Used by the
/// GPU-assisted sampler path: the GPU produces a top-K=128 candidate set
/// from the raw logits, and the CPU then runs the existing repeat-penalty
/// math on just those 128 entries.
///
/// Math is identical to `apply_repeat_penalty` — same frequency count, same
/// recency decay, same 1.5× cap, same ">0 ? divide : multiply" branch.
/// The only difference is iteration scope.
pub fn apply_repeat_penalty_candidates(
    cand_ids: &[u32],
    cand_vals: &mut [f32],
    history: &[u32],
    window: usize,
    penalty: f32,
) {
    debug_assert_eq!(cand_ids.len(), cand_vals.len());

    let start = history.len().saturating_sub(window);
    let recent = &history[start..];
    let window_len = recent.len() as f32;
    if window_len == 0.0 {
        return;
    }

    let mut counts = std::collections::HashMap::<u32, (u32, f32)>::new();
    for (i, &t) in recent.iter().enumerate() {
        let recency = (i as f32 + 1.0) / window_len;
        let entry = counts.entry(t).or_insert((0, 0.0));
        entry.0 += 1;
        if recency > entry.1 {
            entry.1 = recency;
        }
    }

    for (i, &tok) in cand_ids.iter().enumerate() {
        if let Some(&(count, recency)) = counts.get(&tok) {
            let effective = penalty.powf(count as f32 * recency).min(1.5);
            if cand_vals[i] > 0.0 {
                cand_vals[i] /= effective;
            } else {
                cand_vals[i] *= effective;
            }
        }
    }
}

/// Sample from a pre-selected candidate set instead of the full logits.
///
/// Accepts (cand_ids, cand_vals): 128 raw (pre-penalty) candidate tokens
/// from the GPU `topk_logits_f32` kernel. Applies repeat penalty to just
/// those candidates, then runs the same top-K=20 → softmax → top-p
/// sampling pipeline as `sample_top_p` on the full logits array.
///
/// This is bit-exact with the full-CPU path PROVIDED that the pre-penalty
/// top-128 ⊇ the post-penalty top-20 from the full vocabulary. Since
/// `apply_repeat_penalty` monotonically decreases logits (divide-if-positive
/// or multiply-more-negative), a token outside the pre-penalty top-128 can
/// never climb into the top-20 after penalty, so the set relation holds.
pub fn sample_top_p_from_candidates(
    cand_ids: &[u32],
    cand_vals: &mut [f32],
    history: &[u32],
    repeat_window: usize,
    repeat_penalty: f32,
    temperature: f32,
    top_p: f32,
    rng: &mut SamplerRng,
) -> u32 {
    debug_assert_eq!(cand_ids.len(), cand_vals.len());

    // Step 1: apply repeat penalty to the candidate subset.
    apply_repeat_penalty_candidates(cand_ids, cand_vals, history, repeat_window, repeat_penalty);

    // Step 2: if greedy, just return the argmax of the penalized candidates.
    if temperature <= 0.0 {
        let mut best_idx = 0usize;
        let mut best_val = cand_vals[0];
        for i in 1..cand_vals.len() {
            if cand_vals[i] > best_val {
                best_val = cand_vals[i];
                best_idx = i;
            }
        }
        return cand_ids[best_idx];
    }

    // Step 3: top-K=20 selection from the candidate set, matching the
    // full-CPU path's selection logic exactly. The candidate set is already
    // ≤ 128, but we still pick the top 20 via the same min-tracking loop
    // the full path uses, so the resulting set ordering is identical.
    const TOP_K: usize = 20;
    let top_p = top_p.clamp(0.0, 1.0);
    let inv_temp = 1.0 / temperature;

    let mut topk_val = [f32::NEG_INFINITY; TOP_K];
    let mut topk_idx = [0u32; TOP_K];
    let mut min_pos = 0usize;
    let mut min_val = f32::NEG_INFINITY;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &l) in cand_vals.iter().enumerate() {
        let tok = cand_ids[i];
        if l > max_logit {
            max_logit = l;
        }
        if l > min_val {
            topk_val[min_pos] = l;
            topk_idx[min_pos] = tok;
            min_val = f32::INFINITY;
            for j in 0..TOP_K {
                if topk_val[j] < min_val {
                    min_val = topk_val[j];
                    min_pos = j;
                }
            }
        }
    }

    // Step 4: softmax over the K=20 winners (temperature-scaled).
    let mut probs = [0.0f32; TOP_K];
    let mut sum = 0.0f32;
    for i in 0..TOP_K {
        let p = ((topk_val[i] - max_logit) * inv_temp).exp();
        probs[i] = p;
        sum += p;
    }

    // Step 5: sort descending by probability (insertion sort on 20).
    let mut order: [usize; TOP_K] = core::array::from_fn(|i| i);
    for i in 1..TOP_K {
        let mut j = i;
        while j > 0 && probs[order[j]] > probs[order[j - 1]] {
            order.swap(j, j - 1);
            j -= 1;
        }
    }

    // Step 6: top-p filtering + sample. Draws from the caller-owned `rng`
    // state, so the RNG stream is identical across the full-CPU and
    // GPU-assisted paths.
    let r = rng.next_f32() * sum;
    let mut cumulative = 0.0f32;
    let mut sample_acc = 0.0f32;
    let threshold = top_p * sum;
    for &k in &order {
        cumulative += probs[k];
        sample_acc += probs[k];
        if sample_acc >= r {
            return topk_idx[k];
        }
        if cumulative >= threshold {
            let r2 = rng.next_f32() * cumulative;
            let mut acc2 = 0.0f32;
            for &k2 in &order {
                acc2 += probs[k2];
                if acc2 >= r2 {
                    return topk_idx[k2];
                }
                if acc2 >= cumulative {
                    break;
                }
            }
            return topk_idx[order[0]];
        }
    }
    topk_idx[order[0]]
}

/// A sampler RNG stream, owned by whoever is sampling.
///
/// This used to be `static SAMPLER_STATE: AtomicU32`, reset per request to the
/// constant `0x13579BDF` by both `generate` and `generate_vl`. That was already
/// fragile and becomes wrong under the v2 daemon (M1b): with several streams
/// interleaved at module granularity, stream *i*'s token depends on how many
/// other streams happened to draw before it. Per-request seeds stop meaning
/// anything, an identical request stops being reproducible, and — because both
/// resets used the *same* constant — concurrent streams did not merely share a
/// stream, they repeatedly reset each other onto the same one.
///
/// It is also why batched decode is restricted to greedy/argmax today: with no
/// per-row RNG there is no correct way to sample more than one row at a time.
///
/// xorshift32. Not crypto-quality; fine for token sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerRng(u32);

impl SamplerRng {
    /// A stream from an explicit seed. Seed 0 is remapped to 1, because
    /// xorshift32 has a fixed point at zero and would emit it forever.
    pub fn from_seed(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    /// A stream seeded from the clock, for callers with no reproducibility
    /// requirement (standalone examples and one-shot tools).
    pub fn from_entropy() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(1);
        Self::from_seed(nanos)
    }

    /// Raw state, for snapshot/restore. `HIPFIRE_SAMPLE_COMPARE` runs two
    /// samplers from one seed so that token differences reflect real divergence
    /// rather than RNG stream drift; this is how it pins them together.
    pub fn state(self) -> u32 {
        self.0
    }

    pub fn set_state(&mut self, state: u32) {
        *self = Self::from_seed(state);
    }

    /// Next draw in `[0, 1]`.
    pub fn next_f32(&mut self) -> f32 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.0 = s;
        (s as f32) / (u32::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// There is deliberately no RNG lock here any more.
    ///
    /// These tests used to share a `static SAMPLER_STATE: AtomicU32`, and cargo
    /// runs them on parallel threads, so any two that seeded it and then asserted
    /// on the result raced: one thread's reset could land between the other's
    /// reset and its draw. That was not hypothetical — it failed in CI as
    /// `left: 95, right: 52` from two identically-seeded `sample_top_k_top_p`
    /// calls while passing locally, and was worked around with a mutex every
    /// affected test had to remember to take.
    ///
    /// With the RNG owned by the caller (`SamplerRng`), each test holds its own
    /// stream and the race is gone by construction rather than by discipline.
    /// The mutex being deletable is the clearest evidence that the global was a
    /// real correctness hazard and not merely an aesthetic one.

    #[test]
    fn sample_cpu_applies_presence_and_frequency_penalties() {
        let mut logits = vec![0.0_f32, 4.0, 3.5, 3.0];
        let cfg = SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 20,
            repeat_penalty: 1.0,
            repeat_window: 8,
            presence_penalty: 1.0,
            frequency_penalty: 0.5,
            blocked_tokens: Vec::new(),
        };
        let tok = sample_cpu(&mut logits, &[1, 1, 2], &cfg, &mut SamplerRng::from_seed(7));
        assert_eq!(tok, 3);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - 2.0).abs() < 1e-6);
        assert!((logits[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn sample_cpu_greedy_picks_argmax() {
        // sample_cpu with greedy SamplerConfig should return the
        // argmax of the logits — even when blocked_tokens or
        // repeat_penalty would otherwise mutate the slice.
        let mut logits = vec![1.0_f32, 5.0, 2.0, 7.0, 3.0];
        let cfg = SamplerConfig::greedy();
        let tok = sample_cpu(&mut logits, &[], &cfg, &mut SamplerRng::from_seed(7));
        assert_eq!(tok, 3);
    }

    #[test]
    fn top_k_64_is_seeded_and_excludes_the_tail() {
        let logits: Vec<f32> = (0..96).map(|index| index as f32 * 0.03125).collect();
        let first = sample_top_k_top_p(
            &logits,
            1.0,
            64,
            0.95,
            &mut SamplerRng::from_seed(0x1357_9bdf),
        );
        let second = sample_top_k_top_p(
            &logits,
            1.0,
            64,
            0.95,
            &mut SamplerRng::from_seed(0x1357_9bdf),
        );
        assert_eq!(first, second);
        assert!(first >= 32, "top-k 64 admitted tail token {first}");
    }

    #[test]
    fn sample_cpu_blocks_tokens() {
        // A blocked token should never be the argmax even if it
        // started as the largest logit. The blocker is unconditional
        // (a -INF write) so the next-best token wins.
        let mut logits = vec![1.0_f32, 5.0, 2.0, 7.0, 3.0];
        let mut cfg = SamplerConfig::greedy();
        cfg.blocked_tokens = vec![3];
        let tok = sample_cpu(&mut logits, &[], &cfg, &mut SamplerRng::from_seed(7));
        assert_eq!(tok, 1);
    }

    #[test]
    fn sample_cpu_blocked_tokens_out_of_range_skipped() {
        // Out-of-range token IDs are silently skipped — the GPU path
        // does the same `(tok as usize) < vocab_size` guard.
        let mut logits = vec![1.0_f32, 5.0, 2.0, 7.0, 3.0];
        let mut cfg = SamplerConfig::greedy();
        cfg.blocked_tokens = vec![999, 1234];
        let tok = sample_cpu(&mut logits, &[], &cfg, &mut SamplerRng::from_seed(7));
        assert_eq!(tok, 3); // argmax unchanged
    }

    #[test]
    fn argmax_ignores_nan_logits() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0, f32::NAN]), 1);
        assert_eq!(argmax(&[5.0, f32::NAN, 0.1, 2.0]), 0);
    }

    #[test]
    fn argmax_all_nan_returns_zero_without_panic() {
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
    }

    #[test]
    fn a_seeded_stream_is_reproducible_and_two_streams_are_independent() {
        let mut a = SamplerRng::from_seed(123);
        let mut b = SamplerRng::from_seed(123);
        assert_eq!(a.next_f32(), b.next_f32(), "same seed, same stream");

        // The property the v2 executor actually needs: interleaving draws from
        // one stream must not perturb another. Under the old global this was
        // false by construction, which is why batched decode is greedy-only.
        let mut solo = SamplerRng::from_seed(99);
        let solo_draws: Vec<f32> = (0..4).map(|_| solo.next_f32()).collect();

        let mut interleaved = SamplerRng::from_seed(99);
        let mut other = SamplerRng::from_seed(1234);
        let mut interleaved_draws = Vec::new();
        for _ in 0..4 {
            interleaved_draws.push(interleaved.next_f32());
            let _ = other.next_f32();
        }
        assert_eq!(solo_draws, interleaved_draws);

        // Seed 0 is xorshift32's fixed point; it must be remapped or the stream
        // emits zero forever.
        let mut zero = SamplerRng::from_seed(0);
        assert_ne!(zero.state(), 0);
        assert_ne!(zero.next_f32(), 0.0);
    }

    #[test]
    fn state_round_trips_for_sample_compare() {
        let mut rng = SamplerRng::from_seed(0xABCD);
        let _ = rng.next_f32();
        let saved = rng.state();
        let expected = rng.next_f32();

        let mut restored = SamplerRng::from_seed(1);
        restored.set_state(saved);
        assert_eq!(restored.next_f32(), expected);
    }

    #[test]
    fn attractor_block_below_threshold() {
        // 2 occurrences of token 7 in window=20, threshold=3 → no block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![1, 2, 7, 3, 4, 7, 5];
        apply_special_token_attractor_block(&mut logits, &history, 7, 20, 3);
        assert!(
            logits[7].is_finite(),
            "below threshold should leave logit untouched"
        );
    }

    #[test]
    fn attractor_block_at_threshold() {
        // 3 occurrences of token 5 in last 20 → block fires.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 1, 5, 2, 5];
        apply_special_token_attractor_block(&mut logits, &history, 5, 20, 3);
        assert_eq!(
            logits[5],
            f32::NEG_INFINITY,
            "threshold met should -INF the logit"
        );
    }

    #[test]
    fn attractor_block_window_scoped() {
        // 3 occurrences of token 9, but only 1 in the last 5 tokens (window=5,
        // threshold=3) → no block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![9, 9, 1, 2, 3, 4, 5, 9, 6];
        apply_special_token_attractor_block(&mut logits, &history, 9, 5, 3);
        assert!(logits[9].is_finite(), "older occurrences must not count");
    }

    #[test]
    fn attractor_block_pure_repeat() {
        // Worst case: model emits the same special token 5x in a row. Block
        // must fire.
        let mut logits = vec![0.5f32; 16];
        let history: Vec<u32> = vec![11, 11, 11, 11, 11];
        apply_special_token_attractor_block(&mut logits, &history, 11, 20, 3);
        assert_eq!(logits[11], f32::NEG_INFINITY);
        // Other logits untouched.
        assert!((logits[10] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn attractor_block_oob_token_is_noop() {
        let mut logits = vec![1.0f32; 4];
        let history: Vec<u32> = vec![999, 999, 999];
        // token_id past vocab size — should not panic, leave logits untouched.
        apply_special_token_attractor_block(&mut logits, &history, 999, 20, 3);
        for &v in &logits {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn unclosed_block_below_threshold() {
        // 1 open, 0 closes — depth=1 < threshold=2, no block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 1, 2];
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 20, 2);
        assert!(logits[5].is_finite());
    }

    #[test]
    fn unclosed_block_paired_call_passes() {
        // Single complete call: <tool_call>{}</tool_call> = open + close.
        // Depth ends at 0; a follow-up second open would land at 1,
        // still below threshold=2. Don't block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 1, 2, 6, 5]; // open, body, body, close, open
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 20, 2);
        assert!(
            logits[5].is_finite(),
            "second legit open after a complete call must pass"
        );
    }

    #[test]
    fn unclosed_block_two_stacked_opens_blocks_third() {
        // The exact #111 attractor shape: <tool_call><tool_call>...
        // After two consecutive opens with no close, depth = 2 = threshold,
        // block fires (preventing the third).
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 5];
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 20, 2);
        assert_eq!(logits[5], f32::NEG_INFINITY);
    }

    #[test]
    fn unclosed_block_depth_saturates_at_zero() {
        // Stray close at start of window must not push depth negative
        // and let an attractor through. Window: close, open, open.
        // depth = max(0, -1) + 1 + 1 = 2 → block.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![6, 5, 5];
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 20, 2);
        assert_eq!(logits[5], f32::NEG_INFINITY);
    }

    #[test]
    fn unclosed_block_window_scoped() {
        // 2 unclosed opens earlier in history, but the recent window=3 only
        // sees [body, body, close]. depth = 0, allow.
        let mut logits = vec![1.0f32; 16];
        let history: Vec<u32> = vec![5, 5, 1, 2, 6];
        apply_unclosed_attractor_block(&mut logits, &history, 5, 6, 3, 2);
        assert!(
            logits[5].is_finite(),
            "older unclosed opens must not count once they leave the window"
        );
    }
}

#[cfg(test)]
mod initial_rng_state_tests {
    use super::*;

    /// `"fixed"` must be bit-identical to the constant the call sites used to
    /// inline. If this moves, every reproducible deployment moved with it.
    #[test]
    fn fixed_mode_is_the_historical_constant() {
        // Default config is "fixed"; no env or file needed.
        assert_eq!(initial_rng_state(), 0x13579BDF);
    }

    /// The whole point of `"random"`: two draws differ. Asserted here as a pure
    /// unit property of the mixing, independent of config loading — the mode
    /// switch itself is covered by the config schema test.
    #[test]
    fn the_mix_separates_successive_draws() {
        // Same nanos, successive counter values: the counter must be enough on
        // its own, because `subsec_nanos` is coarser than a nanosecond on many
        // hosts and two admissions can land in one tick.
        let mix = |nanos: u32, n: u32| {
            nanos.wrapping_mul(2654435761).rotate_left(13) ^ n.wrapping_mul(2246822519)
        };
        assert_ne!(mix(1234, 0), mix(1234, 1), "same tick collided");
        assert_ne!(mix(1234, 1), mix(1234, 2), "same tick collided");
    }
}
