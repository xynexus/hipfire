# Gemma 4 Phase 1 identity, ingest, and toy models

Date: 2026-07-15. Status: passed.

## Exit-gate evidence

- `hipfire-arch-api` owns stable container id 24. `hipfire-model`, the
  architecture-id documentation, and generated support matrix consume that
  identity; the support row remains honestly `none` until later runtime gates.
- `hipfire-arch-gemma4-spec` registers `gemma4`, `gemma4_text`,
  `gemma4_unified`, and `gemma4_unified_text` as one family. It declares direct
  norm/source-precision ingest, the stacked gate-up/down expert layout, and
  named `dense`, `ple-sharing`, and `dense-moe` toy models.
- The offline registry merges aliases for registrations sharing an id and
  rejects a model-type alias claimed by different ids. Gemma 3 and Qwen3.5 are
  existing identity consumers; Qwen3.5-MoE and Zaya are existing stacked-layout
  consumers.
- The quantizer consults the registry before its compatibility ladder. The
  ladder contains no Gemma 4 model-type literal. Stacked expert splitting is
  selected from the registered source layout plus `num_experts`, so dense and
  MoE variants can share id 24.
- The Gemma 3 norm-offset metadata and tensor transform match only the named
  Gemma 3 text/VL and EmbeddingGemma ids. Id 24 cannot enter either transform.

Targeted gates:

```text
$ cargo test -p hipfire-arch-gemma4-spec
test result: ok. 5 passed; 0 failed

$ cargo test -p hipfire-arch-specs
test result: ok. 2 passed; 0 failed

$ cargo test -p hipfire-quantize registry_resolves_gemma4_identity_and_stacked_expert_layout --bin hipfire-quantize
test result: ok. 1 passed; 0 failed

$ cargo run -p hipfire-cli -- gen-model-support --check
gen-model-support: artifacts are up to date

```

The fixture/conversion gate used seed `0xc0ffee` and the normal CLI path for
each variant:

```text
hipfire-quantize --emit-fixture gemma4_{dense,ple,moe} --output <variant-dir>
hipfire-quantize --input <variant-dir> --output Gemma-4-Tiny-<variant>.bf16.hfq --format bf16
```

All three resolved `Architecture: gemma4 (id=24)`, reported zero mean/max
quantization error, and produced BF16 HFQ files: dense 1.9 MiB, PLE 3.8 MiB,
and dense-plus-MoE 4.6 MiB. The MoE conversion emitted all 39 source tensors,
split both layers' eight stacked experts, and retained each `router.scale`.

## Failed attempt retained

The first MoE conversion exposed an existing generic bug: any tensor ending in
`.scale` was indexed and skipped as an FP8 sidecar even without an FP8 weight
partner. This dropped Gemma 4's learned `router.scale` tensors while the command
still exited successfully. That output was discarded. Sidecar detection now
requires a real I8/F8-E4M3 weight partner, and the full three-variant gate was
rerun from newly generated fixtures.

The first no-GPU boundary run then failed the capability migration ledger with
registered ids `{..., 23, 24, 255}` against expected `{..., 23, 255}`. The
Gemma 4 registration was added to that exact-set completeness assertion; the
targeted crate test and the full no-GPU gate were rerun rather than treating the
new registration as an allowable drift.

The rerun passed the Rust checks, eval-harness check, no-GPU Rust tests,
capability ledger, and fixture round trips, then stopped at the repository-wide
Python lint stage on five unrelated existing findings:

```text
PLW1510 benchmarks/npu_gemm_tuning/iron_ctx_probe/ctx_stability_probe.py:307
PLW2901 benchmarks/npu_gemm_tuning/r70/r70_gen.py:49
PLW2901 benchmarks/npu_gemm_tuning/r71/r71_gen.py:601
PLW2901 benchmarks/npu_gemm_tuning/r71/r71_gen.py:605
PLW2901 benchmarks/npu_gemm_tuning/r81/r81_gen.py:49
Found 5 errors.
```

Those files were unmodified before and remain outside this plan's scope. This
wrapper result is recorded as a repository-level blocker, not represented as a
passed Phase 1 gate.

## Reuse and cleanup ledger

- Existing primitives reused: `ArchRegistry`, `Ingest`, `ToyModel`, the common
  deterministic fixture writer, safetensors ingest, and the BF16 HFQ writer.
- Duplicate removed or retained: model-type aliases and expert source layout
  live in the spec registry instead of new central quantizer family arms. The
  three fixtures remain distinct because each gates different execution
  structure.
- Generic seam added or changed: registered model-type aliases, named toy
  fixtures, and `Ingest::expert_layout`.
- Generic abstraction consumers: Gemma 3/Qwen3.5 consume alias lookup;
  Qwen3.5-MoE, Zaya, and Gemma 4 consume stacked-expert layout; all registered
  toy families consume the named fixture path's default behavior.
- Stale assumption removed: stacked 3-D experts are no longer assumed to use
  Qwen's `.mlp.experts.` spelling, and `.scale` no longer means FP8 sidecar
  without storage evidence.
- Oracle retained: all three source fixtures and their config/manifests remain
  available independently of the HFQ outputs for Phase 2+ loader comparisons.
