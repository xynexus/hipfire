# TODO: is 4-bit kvarn lossless vs f16? (and can it replace the f16 hot ring)

Status: idea. Owner: KV. Relates to the hier-KV plan.

## The claim vs our reality

The KVarN paper calls **4-bit kvarn lossless relative to f16**. If that held in
our implementation, the **f16 hot ring would be redundant** — we'd use 4-bit
kvarn (unmerged) for the hot tier too and cut the exact tier **4×** instead of
keeping it at f16.

But our own evidence says it does **not** hold. From the adoption-plan
head-to-head (0.8B, ctx=16384): **single-tier 4-bit KVarN = PPL 17.71 / KLD
0.085 vs bf16** (no merge — just the 4-bit quant). An f16 tier is ≈ bf16
(KLD ≈ 0), so our 4-bit kvarn is **measurably lossy vs f16**, KLD ~0.085. That is
precisely why `kv_hier.rs` keeps the hot ring at f16.

So the "4-bit lossless" claim is **unverified/false in our impl**, and the hot
ring cannot become 4-bit kvarn as-is without quality loss.

## Findings (2026-07-14) — codec precision sweep

`cargo run -p hipfire-kvquant --example kvarn_precision_sweep` (offline, on a
realistic K-shaped tile: log-normal per-channel σ + outlier channels/tokens)
sweeps the KVarN codec at 2..8 "bits" vs the f16 round-trip floor:

| bits | rel_rmse | attn_KLD |
|---|---|---|
| f16 | 4.0e-4 | 4.2e-7 (floor) |
| 4 | 1.2e-1 | 6.3e-2 |
| 5 | 6.1e-2 | 1.7e-2 |
| 6 | 3.0e-2 | 2.4e-3 |
| 7 | 1.5e-2 | 9.7e-4 |
| 8 | 7.5e-3 | 3.8e-4 |

Conclusions:
- **4-bit is NOT lossless vs f16.** Its attn-KLD (6.3e-2) matches the observed
  model KLD (~0.085 single-tier), so the synthetic tile is representative — the
  gap is real, not a measurement artifact.
- **Precision-limited, not structural.** Error halves ~2× per bit with no
  plateau — that is *optimal* uniform-quantizer scaling, so Sinkhorn + per-channel
  min/max is already doing its job. There is **no codec bug to fix**; uniform
  4-bit fundamentally cannot be f16-lossless (4-bit ≈ 6% step).
- **More bits help proportionally**, but nothing reasonable reaches the f16 floor
  (8-bit is still ~900× above it; true losslessness needs ~11–12 bits).

## Update (2026-07-14): 8-bit kvarn is now selectable + validated

The 8-bit path **already existed** end to end — `new_gpu_kvarn_capped_filtered`
sizes records by `bits` and asserts `{2,4,8}`, and the GPU quantize/dequant
kernels (`kvarn_quantize_tile`/`kvarn_dequant_tile`, `dispatch/kv.rs`) already
take `bits ∈ {2,4,8}`. Only the *selection* was missing (hardcoded 4). Wired
`KvCache::kvarn_bits_from_env()` (`HIPFIRE_KVARN_BITS`, default 4) at all
construction sites (ModelSlot, run, pflash, perplexity) + a `hipfire eval
--kvarn-bits <2|4|8>` flag.

Validated on nix2 (qwen3.5-9b-mq4): `HIPFIRE_KVARN_BITS=8` builds `K 8b var-norm
records`, coherent, +16 MB K storage vs 4-bit.

**Caveat on the needle test:** a magic-number needle came out one digit off
(`8674309` vs `8675309`) **identically at 4-bit and 8-bit** — and identically in
the f16 hot ring vs a kvarn-quantized block. So that error is **weight-quant
(mq4), not KV quant**; KV bits don't move it. The KV-quant quality delta only
shows up where KV error is the bottleneck (long context, many quantized blocks) —
measure with long-context KLD vs a bf16 ref (`--kv-mode kvarn --kvarn-bits {4,8}`
+ the KLD bridge), not this needle.

## Remaining work

1. **Swap the f16 hot ring for 8-bit kvarn** (`kv_hier.rs`). The container +
   kernels exist; the hot read (`attention_cold_slots` layout-2) would move onto
   the kvarn dequant path. 8-bit gives attn-KLD 3.8e-4 (negligible) at 2× smaller
   than f16 → ~2× hot-tier win. Re-validate exact recall + end-to-end KLD
   (confirm 8-bit ≈ f16 at the model level, not just the tile).
2. **Lloyd-Max / codebook kvarn (per-bit fidelity).** Since it's precision-
   limited, a non-uniform quantizer fit to the balanced-tile distribution
   typically buys ~0.5–1 bit vs uniform min/max at the same storage — the repo
   already has Lloyd variants (`mq4l`). A Lloyd-Max kvarn could make 4-bit
   materially better (closer to today's 5-bit) for the cold tier, and let a
   smaller hot-ring bit-width clear the bar. Measure vs uniform at 4/6/8-bit.

Both are config sweeps once the container/quantizer exist: `new_gpu_kvarn*` takes
`bits`; plumb `HIPFIRE_KVARN_BITS` / `--kv-mode kvarn:N`.

## How to measure

Now testable end-to-end: `hipfire eval MODEL --kv-mode kvarn --ctx N
--kldref <bf16>` for PPL/KLD, and the long-context KLD bridge
(`--battery perplexity --corpus <fixture> --ctx N`). Compare each kvarn bit-width
against an f16/bf16 reference on the same corpus; "lossless" = KLD within noise.
`new_gpu_kvarn*` already takes a `bits` arg (only 4 wired) — plumb
`HIPFIRE_KVARN_BITS` / `--kv-mode kvarn:N` to make the sweep a config change.

## Note

The earlier hier-KV result that the f16 hot ring is "PPL-identical to f32" is the
bar a kvarn hot ring must clear — and the 0.085-KLD single-tier number is the gap
it currently fails by.
