# MTP on MI300x — Phase 0 results

**Date:** 2026-05-19
**Rental:** DigitalOcean droplet at `129.212.180.71` (alias `mi300`), ~1h wall clock so far, well under the $20 budget
**Hardware:** AMD Instinct MI300X VF / gfx942 / wave64 / sramecc+ xnack-, 192GB HBM3, ROCm 7.0.0 (rocprofv3 1.0.0)
**Branch tested:** `feat/mtp-mi300x` at `391b346d641786d4ec1928da4cd0ad9e83ca3627`

## Headline

| Bench | Tok/s (median of 3, warm) | τ | Reference (gfx1100) |
|---|---|---|---|
| Solo MTP (27B-3.5 mq4-mtp, max=120, --no-chatml, q8 KV, max-n=3, greedy) | **10.82 tok/s** | 2.0517 | 39.7 (deferred memo 2026-05-15) |
| Composition (DFlash B=16 + MTP K=2 chain, same config) | **41.05 tok/s** (warmed) | 10.17 total / 9.17 dflash + 0.00 mtp + 1.00 bonus | 108.9 (mtp_compose) / ~161.7 (DFlash) on gfx1100 |

**Decision-gate outcome: PHASE 1 NEEDED.**
- Solo MTP threshold for skipping Phase 1 was ≥60 tok/s; we measured 10.82 — *below* even the gfx1100 number (39.7), which is the opposite of what we expected from a 1.3 PFLOPS FP16 + HBM3 box.
- Composition does materially better than solo (41.05 vs 10.82, ~3.8×), but the MTP arm contributes **zero accepts** at K=2 — the DFlash drafter alone produces 9.17 τ and MTP never gets a chance to extend.
- Both numbers are below the gfx1100 references for the same harness. Something is fundamentally underperforming on MI300x / ROCm 7.0.0 for this build of `feat/mtp-mi300x` — see "What surprised me" below.

## Reproduction

```bash
# On the droplet
cd /root/hipfire   # feat/mtp-mi300x @ 391b346d
export ROCM_PATH=/opt/rocm; export PATH=/opt/rocm/bin:$PATH

# Solo
./target/release/examples/mtp_only_demo \
  --target /root/.hipfire/models/qwen3.5-27b.mq4-mtp \
  --prompt-file /root/lru_cache_pep8_strict.txt \
  --max 120 --no-chatml --kv-mode q8 --max-n 3 --temp 0

# Composition (requires standalone .mtp split from .mq4-mtp bundle)
./target/release/examples/dflash_mtp_demo \
  --target /root/.hipfire/models/qwen3.5-27b.mq4-mtp \
  --drafter /root/.hipfire/models/qwen35-27b-dflash.mq4 \
  --mtp-head /root/.hipfire/models/qwen3.5-27b.mtp \
  --prompt-file /root/lru_cache_pep8_strict.txt \
  --max 120 --no-chatml --kv-mode q8 --mtp-k 2 --temp 0
```

Prompt: canonical `benchmarks/prompts/lru_cache_pep8_strict.txt`, 232 tokens, source md5
`df5dedc8040ce70ba55080c4548e6024`. The demo's reported `prompt_md5` is
`1e74f17934fe759468dbe1471b732067` because `maybe_normalize_prompt`
mutates the prompt before hashing (the same hash is reported deterministically across all 6 runs, so the prompt is byte-stable).

The bundled `.mq4-mtp` file was split into a standalone `.mtp` because
`dflash_mtp_demo` requires `--mtp-head` explicitly (it does not auto-detect
the bundle the way `mtp_only_demo` does). Splitter:
```python
# Read the 16-byte HFBNDMTP trailer at EOF to find mtp_off=14979312640
# Copy bytes [mtp_off, file_end - 16) → /root/.hipfire/models/qwen3.5-27b.mtp
# Resulting .mtp is 225716224 bytes (~226 MB)
```

## Solo MTP profile (top kernels by total time)

External rocprofv3 invocation:
```
rocprofv3 --kernel-trace --stats -S --summary-units msec \
  -d /tmp/phase0-solo -o trace -f csv -- <mtp_only_demo cmd>
```
Total program duration 23.44s (load + prefill + 13.1s decode @ 9.14 tok/s
in the rocprof'd run; warm runs without rocprof hit 10.82 tok/s).
Decode-only kernel-time budget is the relevant slice.

| Rank | Kernel | Calls | Total (ms) | % | Avg (ms) |
|---|---|---:|---:|---:|---:|
| 1 | `Cijk_Alik_Bljk_HSS_BH_MT256x256x32_MI32x32x8x1_…_WG128_2_1_WGM4` (rocBLAS Tensile HSS, FP16×FP16→F32) | 3712 | 2723.06 | **19.71** | 0.734 |
| 2 | `Cijk_Alik_Bljk_HSS_BH_MT256x256x32_MI32x32x8x1_…_WG128_2_1_WGM18` (same family, different tile mod) | 11194 | 2662.46 | **19.27** | 0.238 |
| 3 | `Cijk_Alik_Bljk_HSS_BH_MT256x256x32_MI32x32x8x1_…_WG64_4_1_WGM6` (same family, smaller workgroup) | 10208 | 2050.91 | **14.84** | 0.201 |
| 4 | `fused_gate_up_hfq4g256_wave64` (native MQ4 fused gate+up GEMV) | 15104 | 850.17 | 6.15 | 0.056 |
| 5 | `fused_rmsnorm_mq_rotate` | 44544 | 802.41 | 5.81 | 0.018 |
| 6 | `gemv_hfq4g256_residual_wave64` | 30208 | 786.38 | 5.69 | 0.026 |
| 7 | `Cijk_Alik_Bljk_HSS_BH_MT256x128x64_MI16x16x16x1_…_WG32_8_1_WGM18` (rocBLAS Tensile, smaller tile) | 3712 | 673.13 | 4.87 | 0.181 |
| 8 | `gemm_gate_up_hfq4g256_fp16` (MQ4 batched GEMM, prefill path) | 3456 | 438.87 | 3.18 | 0.127 |
| 9 | `gemm_hfq4g256_residual_fp16` | 6912 | 437.46 | 3.17 | 0.063 |
| 10 | `fused_qkvza_hfq4g256_wave64` | 11328 | 325.80 | 2.36 | 0.029 |

**Observation — rocBLAS Tensile HSS GEMM is the dominant cost.** Top three
kernels (all variants of the same Tensile `Cijk_Alik_Bljk_HSS_BH_MT256x256x32_MI32x32x8x1` family) account for **53.8%** of GPU time on solo MTP decode, with a fourth Tensile variant adding another 4.87% → **~58.7% of decode is rocBLAS FP16×FP16→F32 GEMM.** These are the dense GEMMs that fall back to rocBLAS because there is no native quantized GEMM/GEMV kernel for the shape (lm_head over vocab=248320 at the MTP head + trunk + the per-cycle verify path).

The kernel-trace coverage is consistent with the deferred-memo finding (lm_head dominates per-step cost on standalone MTP) — but on MI300x it is the **rocBLAS path**, not the native HFQ4G256 lm_head GEMV, that consumes the time. The native MQ4 wave64 kernels do exist and are firing (entries 4, 6, 10) but they are an order of magnitude smaller per call than the rocBLAS Tensile bundle.

Recognized vs unfamiliar kernels:
- Recognized (MTP/trunk/lm_head-adjacent): all `*_hfq4g256_*` entries, `fused_rmsnorm_mq_rotate`, `gemv_hfq4g256_residual_wave64`, `attention_q8_0_kv`, `gated_delta_net_q8`, `repeat_interleave_qk_*`, RoPE entries, KV-cache writes.
- "Hidden-lever" candidates: the four `Cijk_Alik_Bljk_HSS_*` Tensile families combined. The fact that we're hitting **four different Tensile shapes** strongly suggests the path is using rocBLAS for multiple dense GEMMs (likely `lm_head`, the MTP head's dense projections, and some F32→F16 cast-then-GEMM pattern) rather than a unified native dispatch.

## Composition profile (top kernels by total time)

Same invocation pattern, target = `dflash_mtp_demo`. Total program duration
11.81s (prefill 0.284s + 3.0s decode @ 40.61 tok/s in the rocprof'd run).

| Rank | Kernel | Calls | Total (ms) | % | Avg (ms) |
|---|---|---:|---:|---:|---:|
| 1 | `gemm_gate_up_hfq4g256_fp16` | 704 | 1054.13 | **33.28** | 1.497 |
| 2 | `gemm_hfq4g256_residual_fp16` | 1408 | 812.02 | **25.64** | 0.577 |
| 3 | `gemm_qkvza_hfq4g256_fp16` | 528 | 387.45 | **12.23** | 0.734 |
| 4 | `Cijk_Alik_Bljk_HSS_BH_…_WG128_2_1_WGM18` (rocBLAS Tensile) | 528 | 157.61 | 4.98 | 0.299 |
| 5 | `Cijk_Alik_Bljk_HSS_BH_…_WG128_2_1_WGM4` (rocBLAS Tensile) | 198 | 150.07 | 4.74 | 0.758 |
| 6 | `Cijk_Alik_Bljk_HSS_BH_…_WG64_4_1_WGM6` (rocBLAS Tensile) | 596 | 124.02 | 3.92 | 0.208 |
| 7 | `gemm_qkv_hfq4g256_fp16` | 176 | 113.24 | 3.58 | 0.643 |
| 8 | `gated_delta_net_q8` | 1200 | 72.77 | 2.30 | 0.061 |
| 9 | `hfq4g256_dequantize_to_f16` | 533 | 53.98 | 1.70 | 0.101 |
| 10 | `fused_rmsnorm_mq_rotate` | 1664 | 33.62 | 1.06 | 0.020 |

**Observation — composition is dominated by native MQ4 batched GEMMs, not rocBLAS.** Top three kernels (`gemm_gate_up_hfq4g256_fp16`, `gemm_hfq4g256_residual_fp16`, `gemm_qkvza_hfq4g256_fp16`) account for **71.2%** of GPU time — these are the native wave64 batched-prefill kernels that handle the B=18 verify width (DFlash B=16 + MTP K=2 + seed). The rocBLAS Tensile family drops to ranks 4–6 and totals only ~13.6%.

This is encouraging structurally: composition keeps the GPU on native quantized kernels because the batched verify path (B+K positions) hits the `gemm_*_fp16` family rather than the per-token `gemv_*_wave64`/`Cijk_*` path that solo MTP keeps re-entering. The single trunk verify is amortizing kernel-launch overhead the way the master plan predicted.

But absolute throughput is still half of the gfx1100 reference (41 vs ~108–161 tok/s). On a 192GB HBM3 / ~1.3 PFLOPS FP16 box this is the headline anomaly of Phase 0.

## Coherence check (first 50 tokens of each run)

Both bench targets emit byte-identical Python LRU cache completions across all 6 runs (3 solo + 3 composition). No `!!!!!` attractor, no token loops. The composition example finishes with `lru_node = self.tail.prev` while the solo example finishes with `last_node = self.tail.prev` — those are different but legitimate token choices, not a coherence failure.

```
=== solo MTP, run 1/2/3 (byte-identical) ===
         if key in self.cache:
             node = self.cache[key]
             self._remove(node)
             self._add_to_front(node)
             return node.value
         return -1

     def put(self, key: int, value: int) -> None:
         if key in self.cache:
             self._remove(self.cache[key])
         node = ListNode(key, value)
         self.cache[key] = node
         self._add_to_front(node)
         if len(self.cache) > self.capacity:
             last_node = self.tail.prev
```

```
=== composition, run 1/2/3 (byte-identical) ===
         if key in self.cache:
             node = self.cache[key]
             self._remove(node)
             self._add_to_front(node)
             return node.value
         return -1

     def put(self, key: int, value: int) -> None:
         if key in self.cache:
             self._remove(self.cache[key])
         node = ListNode(key, value)
         self.cache[key] = node
         self._add_to_front(node)
         if len(self.cache) > self.capacity:
             lru_node = self.tail.prev
```

Coherence gate would pass on both.

## What surprised me (raw notes for the methodology log)

1. **MI300x is _slower_ than gfx1100 on this branch.** Solo MTP 10.82 vs deferred-memo 39.7. Composition 41.05 vs deferred-memo 108.9 (linear-chain). On any pure-compute reading this is wrong by ~4×.

2. **rocBLAS Tensile is doing a disproportionate share of solo decode (~59%).** Native MQ4 wave64 GEMV kernels (`gemv_hfq4g256_residual_wave64`, `gemv_hfq4g256_wide`) are present and firing but each individual rocBLAS Tensile kernel is 0.2–0.73 ms / call versus 0.018–0.056 ms / call for native MQ4 wave64. Most likely the wave64 paths are firing for the trunk's dense layers but the **MTP head's projections + the lm_head over vocab=248320** are taking the rocBLAS dequantize-then-FP16-GEMM path.

3. **Composition swap-to-native works automatically.** Once you go batched (B=18 verify width), the dispatcher routes to `gemm_*_hfq4g256_fp16`, rocBLAS drops to a sliver, and tok/s ~4× the solo case. This is a strong structural signal that **the lm_head bottleneck is real on MI300x too**, but the fix is the same as on gfx1100: get more work per kernel-launch.

4. **MTP K=2 fanout contributes zero accepts in composition.** `accept_dflash_total=110`, `accept_mtp_total=0` over 12 cycles. With DFlash already at τ=9.17 (out of B=16 candidates), the MTP K=2 extension never wins (drafter already covers the prefix). This matches the master plan's note that "linear-chain composition" is the wrong shape — the MTP arm only earns its keep when the drafter is shorter or the MTP chain is fanned across slots.

5. **ROCm 7.0.0 here, not 7.2.x as the runbook assumed.** This is a DO image difference, not user error. Doesn't change the conclusion but worth noting if a 7.2.x rocBLAS Tensile dispatch is faster — re-test on 7.2.x would be a cheap follow-up before committing to a Phase 1 lm_head rewrite.

6. **Droplet SSH dropped twice during the session** (~30–60s each, no reboot — `uptime` continued monotonically). rsync survived through the second drop (TCP keepalive must be set conservatively on the DO MI300x VFs). No data loss but harness scripts that ssh-loop without retry will misreport "rental dead."

## Recommendation

**Phase 1 lm_head reduction is required**, but reframed by the rocprof:

- The on-paper lm_head BW issue (deferred-memo) is **not the main lever** on MI300x — what matters is that the path is taking rocBLAS Tensile FP16 GEMM instead of a native quantized GEMV. The rocBLAS Tensile kernels are also fine architecturally (CDNA3 MFMA dispatch confirmed via `MI32x32x8x1` and `MI16x16x16x1` in their names), but they're being called in a *per-token-with-dequant* pattern that batches poorly.
- **Cheapest next step (estimated 1 droplet-hour, $5):** test if switching the trunk + MTP head to use the native `gemv_hfq4g256_wide` for the lm_head closes the gap. If yes, the fix is dispatch-side, not lm_head-rewrite-side, and Phase 1 becomes "wire native quantized GEMV into the per-token lm_head path on gfx942" rather than the projected "compressed vocab head retrain."
- **Second-cheapest step:** retest on ROCm 7.2.x to confirm the rocBLAS dispatch is the same shape. ROCm 7.2.x ships different Tensile library tunings.
- **Skip:** the compressed-vocab head (cvs4k/8k/16k) and EAGLE-style retrain. Those would only matter if the lm_head BW were the actual bottleneck; the profile says the bottleneck is *kernel selection*, not weight BW.

Composition's native-MQ4 routing at B=18 is the existence proof that the kernels already exist and are fast enough — we just need to recover them in the per-token path. That's a Phase 1 lever with much shorter half-life than a vocab compression or head retrain.

## Cross-refs

- Runbook this Phase 0 implements: `docs/plans/mtp-mi300x-runbook.md`
- Master plan: `docs/plans/mtp-dflash-composition-master-plan.md`
- Deferred memo with gfx1100 numbers: `project_mtp_native_head_deferred_2026_05_15`
- Prior MI300x rental delivery template: `project_mi300x_rental_2026_05_18_delivery`
- Raw rocprof CSVs alongside this doc:
  - `docs/investigations/2026-05-19-mtp-mi300x-phase0/phase0-solo-kernel-stats.csv`
  - `docs/investigations/2026-05-19-mtp-mi300x-phase0/phase0-compose-kernel-stats.csv`

(Note: the runbook predicted `docs/research/`, but that path is gitignored
in this branch; the project convention `docs/investigations/YYYY-MM-DD-topic/`
is used instead.)

## Rental tear-down

**Left running** for the next session — the droplet is at `129.212.180.71` with `~/hipfire @ feat/mtp-mi300x 391b346d` and both models in `~/.hipfire/models/`. Tear down via DO console when no follow-up is planned. The build cache (`~/hipfire/target/release`) is preserved so the Phase 1 lever test can re-run within minutes.
