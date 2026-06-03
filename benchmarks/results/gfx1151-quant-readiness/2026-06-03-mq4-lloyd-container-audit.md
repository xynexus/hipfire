# MQ4-Lloyd Container Inspect Audit

- date: 2026-06-03
- arch: gfx1151
- scope: current 9B MQ4-Lloyd container shape and qtype naming

This audit resolves an observability gap in the readiness matrix: Astrea now
names the current MQ-family HFQ qtypes used by these artifacts instead of
reporting them as unknown records.

Inspect artifacts:

- `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq4-lloyd-astrea-inspect.json`
- `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq3-lloyd-astrea-inspect.json`
- `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq4-control-astrea-inspect.json`
- `benchmarks/results/gfx1151-quant-readiness/2026-06-03-mq6-control-astrea-inspect.json`

## Container Summary

| artifact | file bytes | data end | tensors | tensor names md5 | qtype counts |
|---|---:|---:|---:|---|---|
| `qwen3.5-9b-lloyd-mq4.hfq` | 6056190976 | 6056190976 | 427 | `8c9be4ed228f2fe609445911f215c152` | `F16=153`, `Q8F16=25`, `MQ4G256_LLOYD=249` |
| `qwen3.5-9b-lloyd-mq3.hfq` | 4568261632 | 4568261632 | 427 | `8c9be4ed228f2fe609445911f215c152` | `F16=153`, `Q8F16=25`, `MQ3G256_LLOYD=249` |
| `qwen3.5-9b.mq4` | 5441505280 | 5441505280 | 442 | `5610e763e2ad2ac2b6307e8d9f152945` | `F16=160`, `Q8F16=25`, `MQ4G256=257` |
| `qwen3.5-9b.mq6` | 7486228480 | 7486228480 | 442 | `5610e763e2ad2ac2b6307e8d9f152945` | `F16=160`, `Q8F16=25`, `MQ6G256=257` |

## Interpretation

The current MQ4-Lloyd artifact is not container-blocked by an unknown qtype.
Astrea now maps HFQ qtype `30` to `MQ4G256_LLOYD`, matching the existing
runtime and quantizer contract. The MQ6 control qtype `15` is likewise named
as `MQ6G256`.

The MQ4-Lloyd artifact is internally bounded at the container level:
`data_end` equals the file size, and all tensor records have named qtypes. It
shares the same tensor count and tensor-name hash as the MQ3-Lloyd artifact,
not the MQ4/MQ6 controls. Treat that as Lloyd-producer lineage evidence, not as
an MQ4-Lloyd promotion signal by itself.

This audit does not overturn the current quality decision. The active 9B
MQ4-Lloyd candidate remains coherence-rejected because the
`reason-mq4-lloyd-9b` row emitted the token attractor `!!!!!!!!!!!`. The next
blocker is producer or calibration quality, not HFQ qtype observability.
