// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Hermeticity for the tiny-gate binaries: pin the model-shape knobs their
//! fixtures depend on, so a gate result does not move with whoever's
//! `~/.hipfire/config.json` happens to be on the box.

/// Neutralise the ambient `qwen35_paged_experts` setting for a gate run.
///
/// `qwen35_paged_experts` defaults OFF but is switched on in real deployments,
/// and it is read from the CONFIG FILE at model build time — no environment
/// variable required. With it on, the tiny MoE fixtures take the paged path,
/// where `MoeParams::routed_experts` is empty by design, so
/// `check_moe_decode_supported` refuses and the gate reports
/// `moe.decode-routed-dtype-unsupported-no-fallback`. That reads as a model
/// failure when it is a config leak.
///
/// Only the CONFIG FILE is overridden. An explicitly exported variable still
/// wins, so deliberately probing the paged path stays possible; what cannot
/// happen is a gate silently measuring a different code path than the one it
/// records baselines for.
///
/// `HIPFIRE_QWEN35_RESIDENCY_MODE` is deliberately NOT touched. It is consulted
/// BEFORE the paged-experts flag and short-circuits it, so pinning it here would
/// make an explicit `HIPFIRE_QWEN35_PAGED_EXPERTS=1` silently ineffective — the
/// exact class of override-that-does-nothing this function exists to prevent.
///
/// Call as the FIRST statement of `main`, before `Gpu::init` or any thread
/// spawn.
///
/// This lives here rather than in one example because it is needed by every
/// binary a tiny gate drives. `tiny_quant_probe` had it and
/// `compare_prefill_hidden_paths` did not, which made the hidden-state arm of
/// `tiny-prefill-gate` fail on both MoE families for a reason the gate then
/// hid — six reported failures, one config leak.
pub fn pin_fixture_environment() {
    // SAFETY: the contract above is "first statement of main", i.e.
    // single-threaded, before any GPU or thread setup.
    if std::env::var_os("HIPFIRE_QWEN35_PAGED_EXPERTS").is_none() {
        unsafe { std::env::set_var("HIPFIRE_QWEN35_PAGED_EXPERTS", "0") };
    }
}
