// SPDX-License-Identifier: Apache-2.0
// hipfire — Gemma3 speculative-decode target (SpecTarget). See LICENSE / NOTICE.

//! Gemma3 implementation of the arch-generic speculative-decode verifier
//! ([`hipfire_specdecode_dspark::spec::SpecTarget`]).
//!
//! **M1a — per-token baseline.** Gemma3 has its own forward (not
//! `runtime::llama`), and its batched prefill falls back to per-token whenever
//! sliding-window attention is active (gemma3-4b's default). So this baseline
//! drives `verify_block` / `spec_advance` through the SAME per-token stack AR
//! decode uses ([`crate::forward::forward_step_capture`]) — which makes it
//! greedy-equivalent *by construction* (bit-identical to AR) and correct under
//! SWA (the per-token path advances the local-layer rings correctly), but yields
//! NO spec-decode speedup: verifying a `k`-token block costs `k` sequential
//! target forwards, the same as `k` AR steps.
//!
//! This baseline is enough to (a) generate DSpark/DFlash training labels and
//! (b) validate acceptance correctness. The serving *speedup* — a batched forward
//! that verifies the whole block in parallel by wiring the existing
//! `swa_*_batched` primitives at `batch = block_size` — is **M1b**, tracked
//! separately. See docs/plans/2026-07-07-gemma3-4b-dspark-dflash-cask.md.
//!
//! Gemma3 is pure attention (no recurrent state), so `commit_prefix` is a no-op:
//! the accepted-prefix KV a verify wrote is already correct and the rejected tail
//! is overwritten by the next window's verify (which re-anchors `next_pos` to the
//! window position, re-writing the same SWA ring slots).

use crate::arch::Gemma3Backend;
use crate::forward::{embed_token, forward_step_capture, forward_verify_batch};
use hipfire_rdna::{DType, Gpu, GpuTensor};
use hipfire_runtime::llama::HiddenCaptureSink;
use hipfire_runtime::weights::weight_gemv;
use hipfire_specdecode_dspark::spec::{SpecAdvance, SpecScratch, SpecTarget};

impl Gemma3Backend {
    /// Whether the batched verify forward ([`forward_verify_batch`]) can serve a
    /// block of this size: needs ≥2 positions to beat per-token, and KVarN KV has
    /// no batched primitive (strict n=1 fused write).
    fn batched_verify_ok(&self, block_len: usize) -> bool {
        block_len >= 2 && !self.state.kv_cache.quant_kvarn
    }

    /// Embed `block` into an owned `[m, dim]` buffer and run one batched verify
    /// forward at `position`, returning `[m * vocab]` host logits. Optional
    /// `hidden_gpu` (`[m, n_extract, dim]`) captures the extract-layer residuals
    /// on-device. Anchors `state.next_pos` to `position` first.
    fn verify_block_batched_logits(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        hidden_gpu: Option<&GpuTensor>,
    ) -> Result<Vec<f32>, String> {
        let m = block.len();
        let dim = self.config.hidden_size;
        self.state.next_pos = position;
        let x_batch = gpu
            .alloc_tensor(&[m * dim], DType::F32)
            .map_err(|e| format!("gemma3 verify_block_batched x_batch: {e:?}"))?;
        for (i, &t) in block.iter().enumerate() {
            embed_token(
                gpu,
                &self.weights,
                &self.config,
                &x_batch.sub_offset(i * dim, dim),
                t,
            )
            .map_err(|e| format!("gemma3 verify_block_batched embed {i}: {e:?}"))?;
        }
        let extract = self.dflash_extract_layers.clone();
        let ex: &[usize] = if hidden_gpu.is_some() { &extract } else { &[] };
        let logits = forward_verify_batch(
            gpu,
            &self.weights,
            &self.config,
            &mut self.state,
            &x_batch,
            m,
            position,
            ex,
            hidden_gpu,
        )
        .map_err(|e| format!("gemma3 forward_verify_batch: {e:?}"))?;
        let _ = gpu.free_tensor(x_batch);
        Ok(logits)
    }
}

/// Per-position argmax over `[m, vocab]` row-major logits.
fn argmax_rows(logits: &[f32], m: usize, vocab: usize) -> Vec<u32> {
    (0..m)
        .map(|r| hipfire_runtime::sampler::argmax(&logits[r * vocab..(r + 1) * vocab]))
        .collect()
}

/// Gemma3 per-token verify scratch. The baseline path reuses `Gemma3State`'s own
/// B=1 buffers, so there is nothing extra to hold. M1b's batched verify will grow
/// this into a `[block × …]` prefill scratch.
pub struct Gemma3SpecScratch;

impl SpecScratch for Gemma3SpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn free(self: Box<Self>, _gpu: &mut Gpu) {}
}

impl SpecTarget for Gemma3Backend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, _gpu: &mut Gpu) {
        // Pure attention: no recurrent state to zero. Drop the KV eviction offset
        // so the next conversation rotates from absolute 0.
        self.state.kv_cache.compact_offset = 0;
    }

    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        // Per-token baseline: verify reuses the state's own single-token scratch.
        Ok(Box::new(Gemma3SpecScratch))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if reset {
            self.state.kv_cache.compact_offset = 0;
        }
        // Gemma3's forward writes KV at `state.next_pos`; anchor it to the caller's
        // absolute start so the prefill lands at [start_pos, start_pos+tokens.len()).
        self.state.next_pos = start_pos;

        let extract = self.dflash_extract_layers.clone();
        let want_capture = !extract.is_empty() && hidden_out.is_some();

        for &tok in tokens.iter() {
            if abort() {
                self.state.kv_cache.compact_offset = 0;
                return Ok(SpecAdvance::Aborted);
            }
            let mut sink = if want_capture {
                Some(HiddenCaptureSink {
                    extract_layers: &extract,
                    hidden: hidden_out.as_deref_mut().unwrap(),
                    hidden_gpu: None,
                })
            } else {
                None
            };
            forward_step_capture(
                gpu,
                &self.weights,
                &self.config,
                &mut self.state,
                tok,
                sink.as_mut(),
            )
            .map_err(|e| format!("gemma3 spec_advance: {e:?}"))?;
        }
        // The last per-token forward left the final position's logits in state.
        let logits = gpu
            .download_f32(&self.state.logits)
            .map_err(|e| format!("gemma3 spec_advance logits: {e:?}"))?;
        Ok(SpecAdvance::Ready {
            last_argmax: hipfire_runtime::sampler::argmax(&logits),
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        // Batched fast path (M1b): one forward for the whole block, ~2.6× faster
        // than per-token. Used when no host-Vec capture is requested (host capture
        // stays on the per-token path; the on-device capture path is
        // `verify_block_capture_gpu`).
        if hidden_out.is_none() && self.batched_verify_ok(block.len()) {
            let logits = self.verify_block_batched_logits(gpu, block, position, None)?;
            return Ok(argmax_rows(&logits, block.len(), self.config.vocab_size));
        }
        self.state.next_pos = position;
        let extract = self.dflash_extract_layers.clone();
        let want_capture = !extract.is_empty() && hidden_out.is_some();

        let mut picks = Vec::with_capacity(block.len());
        for &tok in block.iter() {
            let mut sink = if want_capture {
                Some(HiddenCaptureSink {
                    extract_layers: &extract,
                    hidden: hidden_out.as_deref_mut().unwrap(),
                    hidden_gpu: None,
                })
            } else {
                None
            };
            forward_step_capture(
                gpu,
                &self.weights,
                &self.config,
                &mut self.state,
                tok,
                sink.as_mut(),
            )
            .map_err(|e| format!("gemma3 verify_block: {e:?}"))?;
            let logits = gpu
                .download_f32(&self.state.logits)
                .map_err(|e| format!("gemma3 verify_block logits: {e:?}"))?;
            // argmax[i] = target's next-token prediction after consuming block[..=i].
            picks.push(hipfire_runtime::sampler::argmax(&logits));
        }
        Ok(picks)
    }

    fn verify_block_logits(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        // Batched fast path (M1b) when no host-Vec capture is requested.
        if hidden_out.is_none() && self.batched_verify_ok(block.len()) {
            return self.verify_block_batched_logits(gpu, block, position, None);
        }
        self.state.next_pos = position;
        let extract = self.dflash_extract_layers.clone();
        let want_capture = !extract.is_empty() && hidden_out.is_some();

        let vocab = self.config.vocab_size;
        let mut out = Vec::with_capacity(block.len() * vocab);
        for &tok in block.iter() {
            let mut sink = if want_capture {
                Some(HiddenCaptureSink {
                    extract_layers: &extract,
                    hidden: hidden_out.as_deref_mut().unwrap(),
                    hidden_gpu: None,
                })
            } else {
                None
            };
            forward_step_capture(
                gpu,
                &self.weights,
                &self.config,
                &mut self.state,
                tok,
                sink.as_mut(),
            )
            .map_err(|e| format!("gemma3 verify_block_logits: {e:?}"))?;
            let logits = gpu
                .download_f32(&self.state.logits)
                .map_err(|e| format!("gemma3 verify_block_logits download: {e:?}"))?;
            out.extend_from_slice(&logits);
        }
        Ok(out)
    }

    /// GPU-resident capture verify (M1b): one batched forward, per-position argmax,
    /// extract-layer residuals written straight into `hidden_gpu`
    /// (`[block × n_extract × dim]`) — the DSpark accepted-prefix on-device reuse
    /// path (no D2H/H2D per window). Returns `(argmax, true)` on success; declines
    /// with `(vec![], false)` for a block too small / KVarN KV, leaving `hidden_gpu`
    /// untouched so the caller re-bootstraps via the host path.
    fn verify_block_capture_gpu(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        _scratch: &mut dyn SpecScratch,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        if !self.batched_verify_ok(block.len()) {
            return Ok((Vec::new(), false));
        }
        let logits = self.verify_block_batched_logits(gpu, block, position, Some(hidden_gpu))?;
        Ok((
            argmax_rows(&logits, block.len(), self.config.vocab_size),
            true,
        ))
    }

    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _accept_len: usize,
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Pure attention: verify's accepted-prefix KV is already correct, and the
        // next window's verify re-anchors `next_pos` and overwrites the rejected
        // tail (same SWA ring slots, since positions are contiguous). Nothing to do.
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token_id
    }

    fn ctx_capacity(&self) -> usize {
        self.state.kv_cache.physical_cap
    }

    fn dflash_extract_layers(&self) -> Option<&[usize]> {
        if self.dflash_extract_layers.is_empty() {
            None
        } else {
            Some(&self.dflash_extract_layers)
        }
    }

    fn set_dflash_extract_layers(&mut self, mut layers: Vec<usize>) {
        layers.sort_unstable();
        layers.dedup();
        self.dflash_extract_layers = layers;
    }

    fn capture_seed_main_hidden(
        &mut self,
        gpu: &mut Gpu,
        seed: u32,
        position: usize,
        layers: &[usize],
    ) -> Result<Vec<f32>, String> {
        // One capture-armed forward at `position`; returns the concat of the
        // residual at `layers` (ascending) — the DSpark `main_hidden`.
        self.state.next_pos = position;
        let mut hidden_out: Vec<f32> = Vec::new();
        let mut sink = HiddenCaptureSink {
            extract_layers: layers,
            hidden: &mut hidden_out,
            hidden_gpu: None,
        };
        forward_step_capture(
            gpu,
            &self.weights,
            &self.config,
            &mut self.state,
            seed,
            Some(&mut sink),
        )
        .map_err(|e| format!("gemma3 capture_seed_main_hidden: {e:?}"))?;
        Ok(hidden_out)
    }

    /// Apply the target's final norm + lm_head to `n` rows of residual hidden
    /// (`hidden_rows`: `[n × hidden]` F32, row-major), returning `n × vocab` host
    /// logits. Mirrors the per-row `rmsnorm(output_norm) + weight_gemv(output)`
    /// tail of `forward_after_x`, reusing the state's single-row scratch.
    fn lm_head_logits(
        &mut self,
        gpu: &mut Gpu,
        hidden_rows: &GpuTensor,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        let dim = self.config.hidden_size;
        let vocab = self.config.vocab_size;
        let eps = self.config.rms_norm_eps;
        let mut out = Vec::with_capacity(n * vocab);
        for r in 0..n {
            gpu.memcpy_dtod_at_auto(&self.state.x.buf, 0, &hidden_rows.buf, r * dim * 4, dim * 4)
                .map_err(|e| format!("gemma3 lm_head_logits copy row {r}: {e:?}"))?;
            gpu.rmsnorm_f32(
                &self.state.x,
                &self.weights.output_norm,
                &self.state.tmp,
                eps,
            )
            .map_err(|e| format!("gemma3 lm_head_logits norm row {r}: {e:?}"))?;
            weight_gemv(
                gpu,
                &self.weights.output,
                &self.state.tmp,
                &self.state.logits,
            )
            .map_err(|e| format!("gemma3 lm_head_logits gemv row {r}: {e:?}"))?;
            let logits = gpu
                .download_f32(&self.state.logits)
                .map_err(|e| format!("gemma3 lm_head_logits download row {r}: {e:?}"))?;
            out.extend_from_slice(&logits);
        }
        Ok(out)
    }
}
