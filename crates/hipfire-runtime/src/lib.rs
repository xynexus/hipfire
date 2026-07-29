// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! hipfire-runtime: HFQ model loading and inference on RDNA GPUs.
//!
//! This crate is arch-agnostic. Architecture implementations live in
//! sibling crates (`hipfire-arch-qwen35`, `hipfire-arch-qwen35-vl`,
//! future `hipfire-arch-llama`, etc.) and depend on this crate for
//! shared infrastructure: HFQ/GGUF file readers, the LLaMA-style
//! scratch / KV / sampler primitives, tokenizer, prompt framing, eos
//! filter, loop guard, eviction (TriAttn, CASK), spec-decode primitives
//! (DFlash, DDTree), demand paging (cpu_router, weight_pager), and the
//! [`arch::Architecture`] trait.

pub mod arch;
pub mod bf16_loader;
pub mod calibration;
pub mod cancel;
#[cfg(feature = "deltanet")]
pub mod cask;
pub mod config;
#[cfg(feature = "deltanet")]
pub mod cpu_router;
#[cfg(feature = "deltanet")]
pub mod ddtree;
#[cfg(feature = "deltanet")]
pub mod dflash;
pub mod dispatch;
pub mod env_docs;
pub mod eos_filter;
pub mod ep;
pub mod hfq;
pub mod hfq_compose;
pub mod hfq_modules;
pub mod host_profile;
pub mod kld_eval;
pub mod kv;
pub mod kv_hier;
pub mod layered_kv;
pub mod llama;
pub mod llama_spec;
pub mod load_progress;
pub mod logging;
pub mod loop_guard;
pub mod model_source;
pub mod moe;
pub mod mtp_mirror;
pub mod multi_gpu;
pub mod oq4_arch;
pub mod oq8_arch;
pub mod oq_moe;
pub mod quant;
pub mod safetensors_source;
pub mod sampler;
pub mod sequence_state;
pub mod speed_bench;
pub mod tokenizer;
pub mod tool_call;
pub mod tool_grammar;
pub mod tp_shard;
pub mod transformer;
pub mod transformer_loader;
#[cfg(feature = "deltanet")]
pub mod triattn;
#[cfg(feature = "deltanet")]
pub mod weight_pager;
pub mod weights;

use std::sync::atomic::{AtomicBool, Ordering};

/// Process-global "cancel the current generation" flag.
///
/// The daemon serves one request at a time (single serial worker), so a single
/// process-global boolean is sufficient to signal "stop the in-flight
/// generation". A SIGUSR1 handler (installed via
/// [`install_generation_cancel_handler`]) sets this to `true`; the per-token
/// decode loops poll it at their top-of-loop chokepoint and, when set, break
/// out exactly as a natural end-of-generation would (KV cache and session state
/// stay fully consistent — the pending, unwritten sample is simply dropped).
///
/// The flag is reset to `false` at the start of each generation request (see the
/// daemon's `Generate` handler) and is also cleared by the decode-loop poll
/// itself (via `swap(false)`), so a cancel never leaks into the next request.
pub static GENERATION_CANCEL: AtomicBool = AtomicBool::new(false);

/// Poll-and-clear the generation-cancel flag. Returns `true` exactly once per
/// SIGUSR1 delivery (the flag is atomically reset to `false`), so a decode loop
/// that observes `true` should treat it as a natural stop and break.
#[inline]
pub fn take_generation_cancel() -> bool {
    GENERATION_CANCEL.swap(false, Ordering::Relaxed)
}

/// Clear any pending cancel request. Call at the start of a generation so a
/// stale SIGUSR1 (delivered after the previous request already finished) can't
/// immediately cancel the fresh request.
#[inline]
pub fn reset_generation_cancel() {
    GENERATION_CANCEL.store(false, Ordering::Relaxed);
}

#[cfg(unix)]
extern "C" fn hipfire_generation_cancel_sigusr1(_sig: libc::c_int) {
    // Async-signal-safe: the ONLY thing this handler does is a relaxed atomic
    // store. No allocation, no I/O, no locks — safe to run in signal context.
    GENERATION_CANCEL.store(true, Ordering::Relaxed);
}

/// Install the SIGUSR1 handler that requests cancellation of the in-flight
/// generation. Idempotent; safe to call once at process startup. On non-unix
/// targets this is a no-op (the fleet is Linux).
#[cfg(unix)]
pub fn install_generation_cancel_handler() {
    // SAFETY: `sigaction` with a handler that only performs an async-signal-safe
    // atomic store. SA_RESTART so a signal delivered mid-syscall (e.g. a
    // blocking read on the request channel) transparently restarts rather than
    // failing with EINTR.
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = hipfire_generation_cancel_sigusr1 as *const () as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGUSR1, &sa, std::ptr::null_mut());
    }
}

/// No-op on non-unix targets.
#[cfg(not(unix))]
pub fn install_generation_cancel_handler() {}
