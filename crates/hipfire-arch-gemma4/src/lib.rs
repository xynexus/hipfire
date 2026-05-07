//! hipfire-arch-gemma4: Gemma 4 architecture (text-only + vision tower).
//!
//! Implements the [`hipfire_runtime::arch::Architecture`] trait for the
//! Gemma 4 family (`gemma-4-31B`, `gemma-4-26B-A4B`, `gemma-4-E4B`,
//! `gemma-4-E2B`).
//!
//! Architectural distinctives vs. Qwen3.5:
//!   - Hybrid attention: 5 sliding-window layers (head_dim=256, 16 KV
//!     heads, window=1024) per 1 full-attention layer (head_dim=512,
//!     4 KV heads, K=V shared via `attention_k_eq_v`).
//!   - Proportional partial RoPE on full layers (rotates 64 of 256
//!     pairs, theta=1e6); standard full-rotation RoPE on sliding layers
//!     (theta=1e4).
//!   - Sandwich RMSNorm (input + post-attn + pre-FFN + post-FFN per
//!     layer) plus a learned per-layer scalar `layer_scalar [1]`.
//!   - Final logit softcap `tanh(x / 30) * 30` before sampling.
//!   - Tied LM head (`lm_head` aliases `embed_tokens`).
//!   - `embed_scale = sqrt(hidden_size)` multiplied at every embed lookup.
//!   - SPM-BPE tokenizer (vocab=262144, BOS-prepend, ▁-space prefix).
//!
//! Status (2026-05-07): forward-pass body present but UNTESTED on real
//! Gemma 4 weights. Dispatch helpers (`rope_partial_halved_f32`,
//! `logit_softcap_f32`) and per-arch tokenizer wiring still TODO. See
//! `docs/investigations/2026-05-07-gemma4-arch-intake/arch-report.md`
//! for the kernel-fit checklist and remaining gaps.

pub mod arch;
pub mod gemma4;
pub mod gemma4_vision;

pub use arch::Gemma4;
