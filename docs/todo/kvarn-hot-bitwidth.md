# TODO: higher-bit kvarn + replace the f16 hot ring

Status: idea. Owner: KV. Relates to the hier-KV plan.

## Question

The KVarN paper calls **4-bit lossless**. Is that true in *our* implementation?
We have never tested it, because our codec only supports **2-bit and 4-bit**
storage:

- `kvarn.rs`: `pack_kvarn_tile_bits` / `kvarn_record_bytes_bits` use a **4-bit
  nibble container** (`cpb = 8/bits`; "bits per code (2 or 4)"), plus a 2-bit
  variant. `quantize_tile_qmax` can *measure* lower precision within the nibble
  (e.g. `qmax=7` ≈ 3-bit fidelity) but does not change storage. There is **no
  6- or 8-bit storage path** (6 doesn't pack into bytes cleanly; 8 = 1 code/byte
  needs a new container + dequant kernel).

So we cannot currently A/B 4-bit kvarn vs 8-bit kvarn to check the "lossless"
claim, nor use a higher-bit kvarn anywhere.

## Idea (from the user)

If a higher-bit kvarn (say 8-bit) is genuinely near-lossless, it could **replace
the f16 hot ring**. Today the hot tier is `f16` (`hot_budget × kv_dim × 2`
B/layer) precisely because it must be near-lossless for the recent exact tokens.
An 8-bit kvarn hot ring would **halve** that (`× 1`), which matters as
`hot_budget` grows for long context.

## Work

1. Add an **8-bit kvarn container** (byte-per-code) + matching dequant kernel
   (the current one is nibble-based). Keep 2/4-bit paths intact.
2. Measure **4-bit vs 8-bit kvarn** PPL + KLD vs bf16 on wikitext + the
   long-context fixtures. Is 4-bit actually lossless, or does 8-bit close a
   measurable KLD gap? (Use the eval: `--kv-mode kvarn` at each bit-width once
   the `bits` arg is plumbed to the CLI, + the long-ctx KLD bridge.)
3. If **8-bit kvarn ≈ f16** (KLD within noise): swap the hot ring from f16 to
   8-bit kvarn in `kv_hier.rs` (the hot read is `attention_cold_slots` layout-2;
   a kvarn hot ring reuses the cold dequant path). Re-validate exact-recall
   coherence (needle retrieval) and the VRAM win.
4. Cross-check the earlier finding that the f16 hot ring is "PPL-identical to
   f32" — the bar an 8-bit kvarn hot ring must clear.

## Note

The `new_gpu_kvarn*` constructors already take a `bits` param (currently only 4
is used); wiring `--kv-mode kvarn:N` or a `HIPFIRE_KVARN_BITS` knob would make
the sweep a config change once the 8-bit container exists.
