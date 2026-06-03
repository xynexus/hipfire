# MQ8 gfx1151 Runtime Audit (2026-06-03)

- repo: `/home/sadara/.hipfire/src`
- branch: `qwen35-native-mtp`
- commit: `fab9d2bc`
- arch: `gfx1151`
- structured status: `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq8-status.json`
- purpose: decide whether MQ8 should remain a promotion target, or be assigned
  to the permanent runtime-research lane until an artifact and product role
  exist.

## Artifact Inventory

No local MQ8 model artifacts were found under the source model or local hipfire
model roots:

```bash
rg --files /home/sadara/Models /home/sadara/.hipfire/models | rg -i 'mq8|q8f16'
```

The command produced no matches. The readiness candidate root also has no MQ8
candidate artifact. The structured MQ8 status artifact records
`canonical_mq8_artifact_present=false`.

## Producer Surface Found

The producer path exists, but no current run artifact is present:

- `hipfire-quantize --format mq8` and `--format mq8g256` select the MQ8 path
  through `use_mq8g256`.
- The quantizer emits `QuantType::MQ8G256` for 256-aligned tensor rows using
  `quantize_mq8g256`.
- Embeddings and non-256-aligned tensors fall back to Q8F16 rather than MQ8.
- The CLI help is currently minimal, but the entrypoint is
  `cargo run -p hipfire-quantize --bin hipfire-quantize -- --input <model_dir> --output <output.hfq>`.

If the lane is reopened, the first artifact should be generated explicitly
under the readiness candidate root, for example:

```bash
cargo run -p hipfire-quantize --bin hipfire-quantize -- \
  --input /home/sadara/Models/models--Qwen--Qwen3.5-9B/snapshots/<snapshot> \
  --output /home/sadara/Models/hipfire-candidates/gfx1151-readiness/qwen3.5-9b-mq8.hfq \
  --format mq8 --threads 8
```

Do not generate a multi-GB MQ8 artifact just to fill the matrix. Require a
specific role first, such as a high-precision MQ-family oracle, a Q8 replacement
with lower runtime cost, or a targeted MoE expert-compression experiment.

## Runtime Surface Found

- Qwen3.5 qtype `14` maps to `DType::MQ8G256`.
- Dense decode dispatch calls `gpu.gemv_mq8g256_with_rotate`.
- MQ8 does not use the MQ4-style fused rmsnorm+rotate path. The current runtime
  uses split `rmsnorm_f32` plus `rotate_quantize_x_mq8`.
- Prerotated MQ8 GEMV uses the internal `mq_x_q8` and `mq_x_scales` buffers
  populated by `rotate_quantize_x_mq8`.
- Qwen3.5 MoE code includes scalar and indexed MQ8/HFQ8 paths:
  `gemm_gate_up_hfq8g256`,
  `gemv_hfq8g256_residual_sigmoid_scaled_gpu_batched`,
  `gemv_hfq8g256_moe_gate_up_k8_indexed_batched`, and
  `gemv_hfq8g256_moe_down_k8_indexed_batched_expanded`.

Current gfx1151 compiled blobs exist:

```text
/home/sadara/.hipfire/bin/kernels/compiled/gfx1151/attention_hfq8_kv.hsaco 12056
/home/sadara/.hipfire/bin/kernels/compiled/gfx1151/gemv_hfq8g256.hsaco 9088
/home/sadara/.hipfire/bin/kernels/compiled/gfx1151/gemv_mq8g256.hsaco 14848
/home/sadara/.hipfire/bin/kernels/compiled/gfx1151/kv_cache_write_hfq8.hsaco 9768
```

Kernel sources found:

```text
kernels/src/attention_hfq8_kv.hip
kernels/src/gemv_hfq8g256.hip
kernels/src/gemv_mq8g256.hip
kernels/src/kv_cache_write_hfq8.hip
```

Existing MQ8/HFQ8 harness substrate was found:

- `crates/hipfire-runtime/examples/bench_hfq_family.rs` benchmarks the HFQ
  family including `HFQ8-G256`.
- `crates/rdna-compute/examples/test_moe_mq_gfx1151_scalar_jit.rs` JIT-smokes
  the gfx1151 scalar MoE kernels including the HFQ8/MQ8-style routed kernels.

Those are useful runtime-substrate checks, but they are not candidate-model
AR/DFlash perf baselines and they do not supply quality evidence.

The structured MQ8 status artifact records `producer_surface_present=true`,
`runtime_surface_present=true`, `gfx1151_compiled_blobs_present=true`, and
`no_gpu_admission_covered=true`. It also records `product_role_defined=false`,
`example_or_benchmark_harness_present=true`, `quality_evidence_present=false`,
`perf_evidence_present=false`, `active_promotion_backlog=false`, and
`promotion_allowed=false`.

## Byte Tradeoff

MQ8 is not a smaller MQ-family candidate. The structured status records:

```text
MQ4: 136 B/group
MQ6: 200 B/group
MQ8: 258 B/group
MQ8/MQ6: 1.29x
MQ8/MQ4: 1.90x
```

That byte cost means MQ8 needs an explicit role over MQ6 or Q8 before promotion
work restarts. Acceptable examples would be a high-precision MQ-family oracle,
a lower-runtime-cost Q8 replacement proven by perf, or a targeted MoE
expert-compression experiment. None of those roles is established by the
current evidence.

## No-GPU Admission State

- `moe_prefill_quant_matrix_documents_mq2_mq3_mq4_mq6_mq8` keeps MQ8 rejected
  for gfx12 grouped-WMMA admission.
- `moe_prefill_admits_gfx1151_scalar_bringup_families` admits MQ8 only for the
  gfx1151 scalar bring-up lane.
- `moe_prefill_rejects_mixed_routed_family_without_grouped_gemm` rejects a
  mixed MQ8 routed-family configuration on gfx1151.

## Decision

MQ8 is not ready for product or promotion work. It has a producer path, decode
and gfx1151 scalar bring-up substrate, but it has no local candidate artifact,
no KLD/PPL or coherence evidence, no candidate-model perf baseline, and no
demonstrated role over Q8 or MQ6. Keep MQ8 as
`permanent-runtime-research` and remove it from the active promotion backlog.

Runtime work should be limited to maintaining the existing decode and gfx1151
scalar bring-up substrate. Reopen promotion work only with an explicit product
role, a canonical artifact generated under the readiness naming convention,
candidate-model use of the existing MQ8/HFQ8 harness substrate, and full
quality plus gfx1151 perf evidence.
