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

## The real work (in order)

1. **Root-cause the losslessness gap.** Why is our 4-bit kvarn ~0.085 KLD off
   f16 when the paper says lossless? Audit our codec vs the paper: Sinkhorn
   variance-normalization iterations/convergence, the FWHT/Hadamard rotation,
   per-channel scale + zp fp16 precision, the 128-token block layout, V handling
   (Q8). Isolate which step costs the fidelity (A/B each on a fixed tile).
2. **If closable at 4-bit** → then 4-bit kvarn ≈ f16 and it *can* replace the
   f16 hot ring (unmerged 4-bit kvarn hot tier; the hot read reuses the cold
   dequant path). 4× hot-tier win; re-validate exact needle recall + coherence.
3. **If not closable at 4-bit** → add a **higher-bit kvarn container** (our codec
   only stores 2/4-bit today — a 4-bit nibble container; `quantize_tile_qmax`
   only *measures* lower precision within the nibble, no 6/8-bit storage). Test
   6-bit (awkward packing) / 8-bit (byte-per-code, new dequant kernel) and find
   the smallest bit-width that is lossless vs f16; use it for the hot ring.

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
