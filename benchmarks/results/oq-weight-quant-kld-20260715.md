# Weight-quant KLD: oq8++ / oq4.5++ vs mq4+ (2026-07-15)

qwen3.5-0.8b, ctx=2048, wikitext2 slice (first ~6k tokens, 3 windows, 2039 scored
positions), each scored against a **bf16 reference with fp32 KV**, and every
candidate run **also fp32 KV** — so this isolates the **weight** quant (KV held
lossless). Quantized from `qwen3.5-0.8b-bf16` with the full Hessian
`/srv/hipfire/hessians/qwen3.5-0.8b.hessian.bin` (`--awq --ldlq`).

| weight quant | recipe (from log) | size | PPL | KLD/tok vs bf16 |
|---|---|---:|---:|---:|
| bf16 (ref) | — | 1519 MB | 24.029 | 0 |
| oq8++ | Oq8Plus, OBS int8 + smooth | 787 MB | 24.029 | 3.0e-4 |
| oq4.5++ | OqPlusCompact, tiered OBS int4/int8 + smooth | 566 MB | 24.523 | 4.8e-2 |
| mq4+ (baseline) | — | — | 26.462 | 8.2e-2 |

## Findings

- **oq8++ is essentially lossless** — KLD 3.0e-4, PPL bit-identical to bf16, at
  ~half the size. Use it when quality matters and 8-bit fits.
- **oq4.5++ ≈ halves mq4+'s error** — KLD 0.048 vs 0.082 (~41% lower), PPL 24.52
  vs 26.46, at a smaller size. Clear reason to move off `mq` to `oq` at the ~4-bit
  tier (matches the default-policy directive: oq weights, not mq).
- Both `++` variants really ran LDLQ+AWQ (mixed int4/int8 for oq4.5++).

## Context: weights dominate, not KV

Weight-quant KLD here (0.05–0.08) dwarfs the KV-quant KLD measured separately
(kvarn 4-bit ≈ 9e-4, 8-bit ≈ 1e-5; `docs/todo/kvarn-hot-bitwidth.md`). So the
quality lever is the weight quant — chase oq8++/oq4.5++, not KV bits.

## Reproduce

```bash
Q=target/release/hipfire-quantize; PPX=target/release/examples/perplexity
$Q --input qwen3.5-0.8b-bf16.hfq --output q.oq8++.hfq   --format oq8++   --awq --ldlq --hessian qwen3.5-0.8b.hessian.bin
$Q --input qwen3.5-0.8b-bf16.hfq --output q.oq4.5++.hfq --format oq4.5++ --awq --ldlq --hessian qwen3.5-0.8b.hessian.bin
$PPX qwen3.5-0.8b-bf16.hfq slice.txt --ctx 2048 --kv-mode f32 --dump-ref ref.pkld
$PPX q.oq8++.hfq slice.txt --ctx 2048 --kv-mode f32 --kld-ref ref.pkld
```
