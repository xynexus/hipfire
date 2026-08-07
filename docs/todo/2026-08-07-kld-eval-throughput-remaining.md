# KLD/eval throughput — where it landed, and what is left

Status 2026-08-07, after a long optimisation pass. Numbers below are all measured
on halo (gfx1151), qwen3.5-0.8b, one 2048-token chunk, via
`HIPFIRE_KLD_PHASE_TIMING=1` and rocprofv3.

## Where it is

| phase | session start | now | notes |
|---|---|---|---|
| body | ~60 s (per-token) | **0.565 s** | N-heavy 2x8 WMMA |
| head | ~4 s (1.99 s traced) | **0.425 s** | cooperative-decode LUT3 WMMA |
| download | — | ~0.11 s | one per 256-position window |
| scoring (CPU) | 7.30 s | **~0.10 s** | select-not-sort + rayon over windows |
| **chunk** | **~70 s** | **~1.21 s** | |

1-chunk `hipfire eval` end-to-end: 138.1 s -> 16 s. A 128-chunk reference (the
size the convergence sweep settled on) is ~2.5 min, against the ~40 h the
original path implied.

Quality is unchanged throughout: ppl 24.25 / 23.68, mean_kld 0.0413.

## What is left

### Body GEMM — 0.565 s, ~47% of the chunk

~3.28 TFLOP in 0.565 s ≈ **5.8 TFLOPS**, up from 3.3 but still ~20% of peak.
`gemm_bf16_x_bf16_wmma_gfx1151_nheavy`, 2 M-subtiles x 8 N-subtiles, routed at
`m >= 2048 && batch_size >= 16` (`HIPFIRE_BF16_DENSE_NHEAVY=0` opts out).

Untried: deeper M blocking (4x8), K-split across workgroups, or LDS-staging A
as well as X. **Do not re-try wave64 or double-buffering — both measured
negative, see below.**

### M < 2048 shapes still on m128

`dn_out` (1024x2048) and `ffn_down` (1024x3584) fall under the threshold and use
the original m128 kernel. N-heavy measured 1.2-1.4x SLOWER there, so the
threshold is right, but nobody has tried a small-M-tuned variant.

### Head — 0.425 s, ~35%

`gemm_bf16l3_wmma_coop`: the whole wave decodes a row into an LDS A-tile, then
WMMA reads it. 4 kernel launches per chunk, down from 1023 GEMVs. At N=64 it is
FASTER than plain BF16 (0.57-0.68x) — same compute, 1.38x fewer weight bytes.

`gemm_bf16l3_wmma_m128` (per-lane decode) is also in-tree and wins small-M
shapes (1.84x vs coop's 3.92x at M=1024). Nothing selects between them because
the head is the only LUT3 weight in production; worth a shape-aware pick if a
second one appears.

### LUT3 as the only in-memory bf16 form — now viable, not yet done

`Bf16Huff` is always expanded at load even under `HIPFIRE_BF16L3_RESIDENT`,
because Huffman has no in-kernel decoder — so a Huff-stored model banks the
1.507x on-disk win and discards the VRAM one. Transcoding Huff -> LUT3 during
load closes that; `bf16_huff::decode_par` and `bf16_lut3::encode` are both
public already. `.hfa` ingestion feeds the same path (the container is an index
of per-file huff-compressed payloads, NOT 7z, so random access works without a
full restore — `hipfire-coexistence/src/repack.rs`).

This was blocked on batched LUT3 being a prefill regression. With the
cooperative kernel it is not, so the transcode is now payable.

## Measured negatives — do not re-run these

Four ideas were tested and lost. `bench_bf16l3_vs_bf16_gemm` reproduces all of
them; it validates every kernel against a CPU reference BEFORE timing, which is
what caught the mistakes below.

- **Weight re-reads are not the LUT3 GEMM's bottleneck.** Raising its column
  tile 8 -> 32 quartered weight traffic and bought ~10% at N=256, while making
  N=1 worse.
- **Transposing x to [K, N]** so a tile's columns share a cache line made every
  shape 15-30% SLOWER. The [N, K] layout is coalesced ACROSS THE WAVE (adjacent
  lanes own k0 8 apart), which outweighs per-thread contiguity.
- **Double-buffered LDS + BK64** on the body GEMM: ~1% end-to-end, inside
  run-to-run noise, for a double buffer plus a prologue. Reverted.
- **Wave64** on the body GEMM: correct (matches wave32 to the digit) but
  1.2-1.5x SLOWER at every shape that matters. RDNA3/3.5 is wave32-native. The
  "wave64" in the gfx1151 iu4 tuning notes came from int4 WMMA and does NOT
  transfer to bf16.

## Measurement traps

- **The first daemon run after clearing the kernel cache includes JIT
  compilation** — worth ~1.2 s on the body phase. It produced a phantom "1.76x
  regression" twice before being identified. Always take the second run.
- **`bench_bf16l3_vs_bf16_gemm`'s "bf16 m128" baseline column silently becomes
  N-heavy**, because it calls the `gemm_bf16_x_bf16_wmma` wrapper which now
  routes. Take that baseline with `HIPFIRE_BF16_DENSE_NHEAVY=0`.
- **Single-chunk KLD numbers are worthless** — per-chunk mean_kld has a 13.5%
  coefficient of variation, so n=1 is ~26% off converged.
- Profiling the daemon: see the rocprofv3 recipe in `AGENTS.md` (Verification).

## Reference sizing — settled, build at 128 chunks

Measured before regenerating anything. Old 1175-chunk refs were 2.48 GB; the
current encoder bit-packs indices at 18 bits and would land at ~1.93 GB, 64% of
it `top_log_probs` as raw f32. But the chunk count was the real lever, not the
codec.

Candidates are compared on the SAME chunks, so the statistic is paired; per-
chunk KLDs correlate at r=0.78, which shrinks the requirement 2.1x versus an
independent-samples model. 5% resolution needs 16 chunks, 2% needs 95. Real
config gaps are far above that floor — uniform3 vs down7rest1 is 13.6%,
resolvable at n>=3.

**128 chunks: 210 MB measured, ~2.5 min, 95% CI halfwidth 1.7% of the mean.**
9.2x smaller than the old artifact. The format work is moot at that size.

## Open quality question

The batched forward is not numerically identical to per-token. Measured paired
over 16 chunks against one reference: batched 0.052349 vs per-token 0.052109,
**+0.46%, 95% CI [+0.13%, +0.79%]** — the CI excludes zero, so it is a
systematic bias, not noise. It is 3.1% of the config gap being measured and
below a 128-chunk reference's own CI, so it is fine for ranking quant configs.
References built on different forward paths must not be mixed. Root cause
(suspected per-tile vs per-token q8 KV scales) is still unidentified, but the
magnitude is now bounded rather than unknown.

Note also that batched prefill REQUIRES a quantized KV tier — the F4 guard
rejects f32 KV outright — so references are now built with Q8 KV where they
previously used f32.

## Artefact state

- The five `arch_id=0` corrupt references (chunk 0 replicated 1175x, `BUGS.md`)
  are deleted, 23 GB freed across `~/.hipfire/kldrefs` and `/srv/hipfire`.
- qwen3.5-0.8b is the only bf16 base artefact still on this box. 2b/4b/9b/
  35b-a3b are gone; the 4b could be rebuilt from
  `/srv/hipfire/models/models--Qwen--Qwen3.5-4B.hfa` via
  `hipfire-coexistence repack --input <archive.hfa> --output <hf_dir>`, which
  restores an HF source directory that `HfqFile::from_safetensors` loads
  directly.
- The tiny-quant gate has 8 pre-existing failures (`hfq4`/`q8f16`/`mq4`/`mq6`
  across qwen2, gemma3, minimax, qwen3_5, qwen3_5_moe) with drift values
  identical before and after every commit in this pass. While they stand, that
  gate cannot distinguish a new regression from the old ones — worth
  re-recording baselines.
