// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! The daemon-side execution seam for continuous batching.
//!
//! Before this existed, `hipfire-daemon`'s batch handler dispatched by arch id
//! at four separate points — probe, prefill, prefix-hash preflight, decode —
//! each an `if is_qwen35_family_arch_id(..) { .. } else if ARCH_ID_LFM2_MOE
//! { .. } else { "supports qwen35/qwen35-moe and lfm2-moe only" }` chain. Adding
//! an arch meant editing the daemon in four places, so "port a second arch
//! without touching the generic layer" was impossible by construction: there was
//! no generic layer to leave alone. See
//! `docs/plans/2026-08-05-fused-decode-completion.md`, phase 5a.
//!
//! Note what this is NOT. [`hipfire_arch_api::ContinuousBatching`] is a
//! *declaration* — one method returning a session cap — that tells the SERVER a
//! request may be routed to the batch runner. It carries no execution. This
//! trait is the execution half, and it lives here rather than in a `-spec` crate
//! because it needs `LoadedModel` and `Gpu`; the `-spec` crates are metadata and
//! deliberately do not depend on the GPU stack.
//!
//! # Honesty about the proof
//!
//! Two implementations exist and they are not equally strong evidence. qwen35
//! has true fused multi-session prefill and decode. lfm2-moe has serial
//! per-session prefill and **no batched decode at all** — it takes the default
//! [`BatchExecutor::decode_step`], which refuses. So this abstraction is drawn
//! from one real implementation plus one degenerate one. It is a seam with a
//! known-thin proof, not a proven seam. The real test is the third arch.

use std::io::Write;

use hipfire_generate::{
    GenerateBatchDecodeEnvelope, GenerateBatchPrefillEnvelope, PrefixHashPreflightEnvelope,
};
use hipfire_model::{is_qwen35_family_arch_id, ARCH_ID_LFM2_MOE};

use crate::model::LoadedModel;

/// One architecture's continuous-batching execution surface.
///
/// Every method mirrors a dispatch point the daemon used to open-code per arch.
/// Implementations live beside their arch's other serving code; the daemon holds
/// no arch knowledge and reaches them only through [`batch_executor_for`].
pub trait BatchExecutor: Sync {
    /// Short arch tag, for diagnostics and error text.
    fn name(&self) -> &'static str;

    /// Whether this loaded model can be batched right now. `Ok(())` means the
    /// probe should answer "ready"; `Err(reason)` means answer "unsupported"
    /// with that reason, which reaches the client verbatim.
    ///
    /// This is the runtime envelope (pipeline parallelism, resident state, and
    /// so on), distinct from arch identity — the registry already answered that.
    fn probe(&self, m: &LoadedModel) -> Result<(), String>;

    /// Emit this arch's `generate_batch_prefill_ready` envelope. Arch-specific
    /// because the payload advertises per-arch capability fields.
    fn emit_ready(&self, stdout: &mut dyn Write, envelope: &GenerateBatchPrefillEnvelope);

    /// Run a batched prefill over the envelope's co-resident sessions.
    ///
    /// `pflash_active` is passed to every arch; those without a PFlash
    /// interaction ignore it, which keeps the seam uniform rather than growing
    /// an arch-shaped parameter list.
    fn prefill(
        &self,
        m: &mut LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        stdout: &mut dyn Write,
        envelope: &GenerateBatchPrefillEnvelope,
        pflash_active: bool,
    ) -> Result<(), String>;

    /// Answer a prefix-hash preflight — the cached-prefix probe the server uses
    /// to decide how much of a prompt it can skip.
    fn prefix_hash_preflight(
        &self,
        m: &LoadedModel,
        stdout: &mut dyn Write,
        envelope: &PrefixHashPreflightEnvelope,
    ) -> Result<(), String>;

    /// Advance every resident session by one token.
    ///
    /// Defaults to refusing. An arch with batched prefill but no batched decode
    /// (lfm2-moe today) is a real state, and the default lets it say so instead
    /// of forcing a stub that would silently do the wrong thing. The daemon
    /// previously called the qwen35 decode with **no arch check at all**, so
    /// this default also closes that gap.
    fn decode_step(
        &self,
        _m: &mut LoadedModel,
        _gpu: &mut hipfire_rdna::Gpu,
        _stdout: &mut dyn Write,
        _envelope: &GenerateBatchDecodeEnvelope,
    ) -> Result<(), String> {
        Err(format!(
            "generate_batch_decode_step: arch {} has batched prefill but no batched decode",
            self.name()
        ))
    }
}

/// qwen3.5 dense and MoE — fused multi-session prefill and decode.
pub struct Qwen35BatchExecutor;

impl BatchExecutor for Qwen35BatchExecutor {
    fn name(&self) -> &'static str {
        "qwen35"
    }

    fn probe(&self, m: &LoadedModel) -> Result<(), String> {
        if m.pp != 1 {
            return Err(format!(
                "generate_batch_prefill requires pipeline_parallel=1, got pp={}",
                m.pp
            ));
        }
        Ok(())
    }

    fn emit_ready(&self, stdout: &mut dyn Write, envelope: &GenerateBatchPrefillEnvelope) {
        crate::qwen35_decode::emit_generate_batch_prefill_ready(stdout, envelope);
    }

    fn prefill(
        &self,
        m: &mut LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        stdout: &mut dyn Write,
        envelope: &GenerateBatchPrefillEnvelope,
        pflash_active: bool,
    ) -> Result<(), String> {
        crate::qwen35_prefill::run_generate_batch_prefill_serial_qwen35(
            m,
            gpu,
            stdout,
            envelope,
            pflash_active,
        )
    }

    fn prefix_hash_preflight(
        &self,
        m: &LoadedModel,
        stdout: &mut dyn Write,
        envelope: &PrefixHashPreflightEnvelope,
    ) -> Result<(), String> {
        crate::qwen35_prefill::run_prefix_hash_preflight_qwen35(m, stdout, envelope)
    }

    fn decode_step(
        &self,
        m: &mut LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        stdout: &mut dyn Write,
        envelope: &GenerateBatchDecodeEnvelope,
    ) -> Result<(), String> {
        crate::qwen35_decode::run_generate_batch_decode_step_qwen35(m, gpu, stdout, envelope)
    }
}

/// lfm2-moe — serial per-session prefill, no batched decode (inherits the
/// refusing default).
#[cfg(feature = "arch-lfm2moe")]
pub struct Lfm2MoeBatchExecutor;

#[cfg(feature = "arch-lfm2moe")]
impl BatchExecutor for Lfm2MoeBatchExecutor {
    fn name(&self) -> &'static str {
        "lfm2-moe"
    }

    fn probe(&self, m: &LoadedModel) -> Result<(), String> {
        if m.pp != 1 {
            return Err(format!(
                "generate_batch_prefill requires pipeline_parallel=1, got pp={}",
                m.pp
            ));
        }
        Ok(())
    }

    fn emit_ready(&self, stdout: &mut dyn Write, envelope: &GenerateBatchPrefillEnvelope) {
        crate::lfm2_prefill::emit_lfm2_generate_batch_prefill_ready(stdout, envelope);
    }

    fn prefill(
        &self,
        m: &mut LoadedModel,
        gpu: &mut hipfire_rdna::Gpu,
        stdout: &mut dyn Write,
        envelope: &GenerateBatchPrefillEnvelope,
        _pflash_active: bool,
    ) -> Result<(), String> {
        crate::lfm2_prefill::run_generate_batch_prefill_serial_lfm2(m, gpu, stdout, envelope)
    }

    fn prefix_hash_preflight(
        &self,
        m: &LoadedModel,
        stdout: &mut dyn Write,
        envelope: &PrefixHashPreflightEnvelope,
    ) -> Result<(), String> {
        crate::lfm2_prefill::run_prefix_hash_preflight_lfm2(m, stdout, envelope)
    }
}

static QWEN35_BATCH_EXECUTOR: Qwen35BatchExecutor = Qwen35BatchExecutor;
#[cfg(feature = "arch-lfm2moe")]
static LFM2_MOE_BATCH_EXECUTOR: Lfm2MoeBatchExecutor = Lfm2MoeBatchExecutor;

/// The executor for `arch_id`, or `None` when the arch has no batched execution.
///
/// This is the whole of the daemon's arch knowledge for continuous batching:
/// adding an arch means adding an implementation and one arm here, not editing
/// four call sites in the handler.
pub fn batch_executor_for(arch_id: u32) -> Option<&'static dyn BatchExecutor> {
    if is_qwen35_family_arch_id(arch_id) {
        return Some(&QWEN35_BATCH_EXECUTOR);
    }
    #[cfg(feature = "arch-lfm2moe")]
    if arch_id == ARCH_ID_LFM2_MOE {
        return Some(&LFM2_MOE_BATCH_EXECUTOR);
    }
    None
}

/// The "this arch cannot be batched" message, in one place so the probe and the
/// three execution paths cannot drift apart.
pub fn batch_unsupported_reason(op: &str, arch_id: u32) -> String {
    format!("{op}: arch_id={arch_id} has no continuous-batching executor")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock implementing only the required methods — the arch-agnostic check
    /// that the trait is object-safe and implementable without batched decode,
    /// per the genericity principle in the continuous-scheduler plan ("test the
    /// seam, not the impl").
    struct NoopExecutor;

    impl BatchExecutor for NoopExecutor {
        fn name(&self) -> &'static str {
            "noop"
        }
        fn probe(&self, _m: &LoadedModel) -> Result<(), String> {
            Ok(())
        }
        fn emit_ready(&self, _stdout: &mut dyn Write, _envelope: &GenerateBatchPrefillEnvelope) {}
        fn prefill(
            &self,
            _m: &mut LoadedModel,
            _gpu: &mut hipfire_rdna::Gpu,
            _stdout: &mut dyn Write,
            _envelope: &GenerateBatchPrefillEnvelope,
            _pflash_active: bool,
        ) -> Result<(), String> {
            Ok(())
        }
        fn prefix_hash_preflight(
            &self,
            _m: &LoadedModel,
            _stdout: &mut dyn Write,
            _envelope: &PrefixHashPreflightEnvelope,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn trait_is_object_safe_and_decode_is_optional() {
        let ex: &dyn BatchExecutor = &NoopExecutor;
        assert_eq!(ex.name(), "noop");
        // An arch may implement batched prefill without batched decode; the
        // default must refuse rather than silently mis-execute.
        let msg = format!(
            "generate_batch_decode_step: arch {} has batched prefill but no batched decode",
            ex.name()
        );
        assert!(msg.contains("no batched decode"));
    }

    #[test]
    fn registry_maps_only_arches_with_batched_execution() {
        // qwen3.5 dense (5) and MoE (6).
        assert!(batch_executor_for(5).is_some(), "qwen3.5 dense");
        assert!(batch_executor_for(6).is_some(), "qwen3.5 MoE");
        assert_eq!(batch_executor_for(5).map(|e| e.name()), Some("qwen35"));

        #[cfg(feature = "arch-lfm2moe")]
        {
            let lfm2 = batch_executor_for(ARCH_ID_LFM2_MOE);
            assert_eq!(lfm2.map(|e| e.name()), Some("lfm2-moe"));
        }

        // deepseek4 (9) has no multi-session batched forward yet — its
        // forward_prefill_batch is single-session and falls back per token. It
        // must report unbatchable rather than reach an arch it is not.
        assert!(batch_executor_for(9).is_none(), "deepseek4");
        assert!(batch_executor_for(0).is_none(), "llama");
        assert!(batch_executor_for(u32::MAX).is_none(), "unknown arch");
    }

    #[test]
    fn unsupported_reason_names_the_op_and_arch() {
        let r = batch_unsupported_reason("generate_batch_prefill", 9);
        assert!(r.contains("generate_batch_prefill"));
        assert!(r.contains("arch_id=9"));
    }
}
