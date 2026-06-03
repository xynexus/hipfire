# ParoQ4G128 Productization Audit (2026-06-03)

- repo: `/home/sadara/.hipfire/src`
- branch: `qwen35-native-mtp`
- commit: `fab9d2bc`
- arch target: `gfx1151`
- control: MQ4

## Producer Contract

Astrea exposes the Paro producer-consumer commands:

- `python3 scripts/astrea.py paro-probe`
- `python3 scripts/astrea.py paro-import`
- `python3 scripts/astrea.py paro-oracle`

The native ParoQ4G128 payload is documented as HFQ `quant_type=28` /
`PARO4G128`. Each imported record stores the native Paro/AWQ tensors in this
order:

```text
qweight        int32 [K, M/8]
qzeros         int32 [K/128, M/8]
scales         f16   [K/128, M]
pairs          int16 [8, K]
theta          f16   [8, K/2]
channel_scales f16   [K]
```

The runtime contract is:

```text
x_rot = rotate(x, pairs, theta, channel_scales)
y     = awq_w4_gemv(x_rot, qweight, qzeros, scales)
```

`scripts/paroquant_import.py` is a runtime-enablement bridge, not a quantizer.
It requires an upstream/native Paro checkpoint with complete
`qweight/qzeros/scales/pairs/theta/channel_scales` companions for each module
and writes runtime-loadable HFQ records. `scripts/paroquant_oracle.py` compares
the imported HFQ bytes against the source Paro safetensors and checks the
decode oracle before any quality or perf claim.

## Current Local Source Inventory

No local Paro model artifacts were found under `/home/sadara/Models` or
`~/.hipfire/models`.

The source probe artifact
`benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g128-qwen35-9b-source-probe.json`
was generated from:

```text
/home/sadara/Models/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

It reported `775` tensors and zero complete Paro modules. That source snapshot
cannot satisfy the importer contract and is not a native ParoQuant checkpoint.

Follow-up A3B source probes were generated from:

```text
/home/sadara/Models/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754
/home/sadara/Models/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
```

Their probe artifacts are:

```text
benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g128-qwen35-a3b-source-probe.json
benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g128-qwen36-a3b-source-probe.json
```

Qwen3.5 35B-A3B reported `1811` tensors and zero complete Paro modules. Qwen3.6
35B-A3B reported `1045` tensors and zero complete Paro modules. These A3B
source snapshots also cannot satisfy the importer contract and are not native
ParoQuant checkpoints.

## Runtime Surface Found

- `crates/hipfire-runtime/src/hfq.rs` maps HFQ `quant_type=28` to
  `DType::PARO4G128`.
- `crates/hipfire-runtime/src/llama.rs` has ParoQ4G128 GEMV, residual, and
  SwiGLU dispatch arms.
- `crates/hipfire-arch-qwen35/src/qwen35.rs` has ParoQuant safetensors loading
  for dense and MoE Qwen3.5 paths.
- `crates/hipfire-arch-qwen35/src/qwen35.rs` has ParoQ4G128 routed expert and
  grouped MoE dispatch arms, including gfx1151 grouped MMQ paths.
- No-GPU coverage includes `paro_batched_admit_defaults_on_and_allows_opt_out`,
  `moe_prefill_paro_i8_env_policy_is_gfx1151_default_on_with_opt_out`, and
  `moe_prefill_paro_i8_k8_env_policy_follows_i8_gate_and_allows_opt_out`, all
  covered by the narrow `cargo test -p hipfire-arch-qwen35 --lib moe_prefill`
  subset. There is no artifact-backed Paro admission, oracle, coherence,
  KLD/PPL, or perf row yet.

## Runtime Env Boundary

ParoQ4G128 productization must not silently depend on a research-only env knob.
The current runtime surface has three classes:

- `HIPFIRE_PARO_BATCHED`: default-on admission with `0` as an opt-out. This is
  a productization candidate path, but promotion reports must record the
  effective setting.
- `HIPFIRE_MOE_PARO_I8` and `HIPFIRE_MOE_PARO_I8_K8`: gfx1151 grouped MMQ path
  switches that are default-on unless set to `0`. These can be promotion
  candidates only after artifact-backed coherence, finite-logit/NaN, KLD/PPL,
  and perf rows are recorded with the effective defaults.
- `HIPFIRE_PARO_PREROTATE`, `HIPFIRE_PARO_SMALL_DIRECT`,
  `HIPFIRE_PARO_SWIGLU_FUSED`, `HIPFIRE_PARO_FUSE_RMSNORM`,
  `HIPFIRE_PARO_FA3_FUSED`, `HIPFIRE_PARO_GATE_UP_FUSED`,
  `HIPFIRE_PARO_LA4_FUSED`, `HIPFIRE_PARO_LA2_FUSED`,
  `HIPFIRE_PARO_LA_GATES_MQ4G128`, `HIPFIRE_PARO_PACK1`,
  `HIPFIRE_PARO_PACK2`, `HIPFIRE_PARO_PACK4`,
  `HIPFIRE_PARO_SHARED_PAIRS`, and `HIPFIRE_PARO_FUSED_PACK2` are
  research/probe levers until they have their own artifact-backed evidence.
  They must be kept out of any promoted ParoQ4G128 main-path claim.

## Astrea Package Status

`benchmarks/results/gfx1151-quant-readiness/2026-06-03-paro-q4g128-astrea-bundle-plan.json`
records the current Astrea package contract for the intended canonical
`qwen3.5-9b-paro-mq4.hfq` artifact.

The bundle-plan is not a model writer and does not prove Paro quality. It
records:

- schema: `hipfire.astrea.bundle_plan.v0`
- bundle id: `gfx1151-paro-q4g128-qwen35-9b-package-contract`
- container target: `hfq-package-v0`
- external sidecars: `false`
- sections: `manifest`, `weights`, `transform.paro`, and `evidence.summary`
- `transform.paro.runtime_status`: `deferred_until_loader_and_fused_kernel_exist`
- `transform.paro.runtime_env_boundary`: records productization-candidate env
  defaults, research-only knobs, and promotion-report requirements
- required `weights.source.exists`: `false`

That last point is the current package blocker: the imported Paro HFQ source
does not exist yet because the local Qwen source snapshots are not native
ParoQuant checkpoints. After `paro-import` succeeds, regenerate this bundle plan
with the imported HFQ artifact as the weights source before making any package
or loader promotion claim.

## Decision

ParoQ4G128 has a clear producer-consumer contract and meaningful runtime
substrate, but it is blocked at the producer/artifact stage in this checkout.
Do not promote or run coherence/perf until a native Paro checkpoint is located
or generated for the target fixture, `paro-import` emits a canonical HFQ
artifact, and `paro-oracle` passes for at least one dense or routed module.
After that, run dense and A3B coherence, NaN stability, KLD/PPL, and gfx1151
AR/DFlash perf rows against MQ4 while recording the effective Paro env defaults
and excluding research-only fused knobs from promotion evidence.
