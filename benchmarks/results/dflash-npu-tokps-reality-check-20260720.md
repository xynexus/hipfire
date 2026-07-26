# DFlash NPU drafter — tokens/s reality check (task #31)

**Date:** 2026-07-20 · **Machine:** nix1 (gfx1103 Phoenix APU + npu1/aie2) ·
**Branch:** chaingun · **Tool:** `target/release/examples/dflash_spec_demo`
(no runtime seam, no kernel rebuilds).

**Question.** Is the DFlash NPU drafter net-positive on throughput even though its
block wall (111.9 ms) is ~2× the GPU verify "budget" (57 ms)? The budget framing
is the *sufficient* condition for "draft is free"; it is not *necessary* for
"spec decode wins." Settle it from measured pieces, without wiring the NPU into
the runtime (seam #24 deferred).

Model: `~/.hipfire/models/qwen3.5-9b-mq4.hfq` (9B target) +
`~/.hipfire/drafts/Qwen3.5-9B.dflash.f16.hfq` (f16 golden drafter).

Losslessness held throughout: AR and every spec run emit md5 `02e621bd56b5`
(engine prompt) / `d51282bca090` (code prompt); AR == spec per prompt, 3/3
repeats, `2>/dev/null`.

---

## 1. Measured — GPU-only autoregression (`--ar-baseline --draft`)

Engine prompt, `--max 96`, 3 repeats: **17.47 / 17.46 / 17.46 tok/s** →
**median 17.46 tok/s**. Code prompt: 17.37 tok/s.

**This directly measures the 9B single-token forward wall: 1/17.46 = 57.3 ms.**
That is exactly the brief's "57 ms verify budget" — so the budget number is the
*single-token* forward, and it *is* measured (here, now, on gfx1103).

## 2. Measured — GPU spec-decode (GPU DFlash drafter)

Engine prompt, B=16, `--max 96`, 3 repeats: **6.86 / 6.84 / 6.86 tok/s**,
**τ = 2.167**, accept_rate 0.144. Longer `--max 256`: 6.96 tok/s, τ = 2.188.

Block-size sweep (engine, `--max 128`, `HIPFIRE_HOST_TIMING=1`):

| B | decode tok/s | τ | per-cycle wall |
|---|---|---|---|
| 4 | 11.52 | 1.70 | 238 ms |
| 8 | 7.74 | 2.10 | 409 ms |
| 16 | 6.88 | 2.10 | 454 ms |

Code prompt (τ is prompt-sensitive), B=16: **9.55 tok/s at τ = 5.60** —
excellent acceptance, yet AR on the same prompt is **17.37 tok/s**.

**GPU spec-decode is net-NEGATIVE vs AR at every block size AND every τ**
(11.52 best-case < 17.46; even τ=5.60 gives 9.55 < 17.37). This is the load-bearing
finding — see §5.

### Per-cycle breakdown (draft vs verify)

`HIPFIRE_PROFILE=1` over 20 warm cycles (B=16), 239 ms/cycle kernel time,
458 ms/cycle wall. Buckets by call-count signature:

- **GPU draft** = the DFlash 1.05B drafter run **autoregressively B times per
  cycle**. Single-row `gemv_*` (214/cyc) + the 5-layer×16-token GDN fused ops
  (`fused_qkvza`, `fused_qk_l2_norm`, `gated_norm`, ... each exactly 80/cyc =
  5 layers × 16 tokens). ≈ **16 sequential ~1B forwards ≈ 320 ms/cycle**.
- **GPU verify** = one **batched-16** forward of the full 9B (HIP-graph replay).
  Batched `gemm_mw16_residual_wmma` (46/cyc = one down_proj per layer) + the
  batched `gemm_*_wmma/_exact` GEMMs. Profile-summed ≈ **100 ms/cycle** for the
  16-token block.

**Deliverable #3 result:** the 57 ms budget is the *single-token* forward. The
actual **16-token verify block on the 9B is ~100 ms** (~1.75×) — batched-16 adds
compute the single-token bandwidth-bound forward doesn't pay. The verify wall is
measured to be **larger** than the budget, not smaller.

The draft, not the verify, dominates the GPU-spec cycle (~320 of ~450 ms): a
"block drafter" on this iGPU is B *sequential* 1B forwards.

## 3. Projection — NPU variant (arithmetic from measured pieces)

`draft_NPU = 111.9 ms` (measured `--cpu-primitives` multicore block, commit
`8c32a4992`; conservative — context cache #25 would cut steady-state further).
tokens/cycle = τ+1. Two verify anchors: 57 ms (single-token floor / brief budget)
and 100 ms (measured 16-token block).

**Perfect overlap:** step = max(draft_NPU, verify_GPU) = **111.9 ms** (both verify
values ≤ 111.9, so the NPU draft dominates and the result is verify-independent):

| τ (tok/cyc) | NPU-spec overlap |
|---|---|
| 2.10 (3.10) | **27.7 tok/s** |
| 5.60 (6.60) | **59.0 tok/s** |

**No overlap (pessimistic):** step = 111.9 + verify_GPU:

| τ | verify=57 (168.9 ms) | verify=100 (211.9 ms) |
|---|---|---|
| 2.10 | 18.4 tok/s | 14.6 tok/s |
| 5.60 | 39.1 tok/s | 31.1 tok/s |

## 4. Verdict 1 — NPU-draft-overlap vs GPU-only AR (17.46 tok/s)

**YES, net-positive, robustly.**

- Overlap: **27.7 tok/s @ τ2.1 (1.59×)** … **59.0 tok/s @ τ5.6 (3.4×)**. Always wins.
- Pessimistic (no overlap) wins too, except the worst corner (τ=2.1 AND
  verify=100 → 14.6 < 17.46). At τ=5.6 even pessimistic wins (31.1×).

The overlap win is verify-insensitive because draft_NPU (111.9 ms) ≥ verify_GPU.

## 5. Verdict 2 — NPU-spec vs GPU-spec (GPU drafter)

**NPU-spec wins everywhere** (27.7–59.0 overlap, 14.6–39.1 pessimistic) vs GPU-spec
(6.88 @ τ2.1 / 9.55 @ τ5.6). **This REFUTES the aiecost "offload loses at 9B"
claim on this hardware — by refuting its premise.**

The aiecost claim is: *if the GPU drafter already wins on tok/s (draft free under
verify), NPU offload loses.* **On gfx1103 the GPU drafter does NOT win — it loses
to plain AR** (6.88–9.55 < 17.46). The premise is false here because the GPU
"block drafter" is B **sequential** 1B forwards on a weak Phoenix iGPU
(~320 ms/cycle), i.e. the GPU draft is the bottleneck, not a free ride under a
dominant verify.

**Machine-specific caveat (important):** the NPU wins *because the GPU drafter
baseline is bad on the APU*, not because the NPU draft is fast in absolute terms
(111.9 ms is still ~2× the 57 ms single-forward and ~1.1× the 100 ms block
verify). On a fast **discrete** GPU where the drafter is cheap and verify
dominates, the aiecost claim could still hold. The NPU's structural edge here is
that it drafts the whole 16-token block in one shot (111.9 ms) versus the iGPU's
16 sequential forwards (~320 ms) — a 2.9× draft speedup.

**Orthogonal value (excluded from the throughput verdict, per task):** offload
also *frees the GPU* during the draft window for concurrent work (other requests,
batching). Raw tok/s does not capture this; it is a real second axis in favor of
offload beyond the numbers above.

## 6. Summary

| quantity | value |
|---|---|
| GPU-only AR | **17.46 tok/s** (= 57.3 ms/9B-forward) |
| GPU-spec (engine, τ2.1, B16) | 6.88 tok/s |
| GPU-spec (code, τ5.6, B16) | 9.55 tok/s |
| GPU-spec best (B4) | 11.52 tok/s — still < AR |
| GPU verify wall, 16-tok block | **~100 ms** (budget's 57 ms is the single-token forward) |
| GPU draft (B=16) | ~320 ms/cycle (16 sequential 1B forwards) — dominant term |
| NPU-spec overlap | 27.7 (τ2.1) / 59.0 (τ5.6) tok/s |
| NPU-spec pessimistic | 14.6–18.4 (τ2.1) / 31.1–39.1 (τ5.6) tok/s |
| **Verdict 1** (NPU-overlap vs AR) | **WIN, 1.6×–3.4×** |
| **Verdict 2** (NPU-spec vs GPU-spec) | **NPU wins; aiecost "offload loses" premise fails on gfx1103** |

**Bottom line.** At 9B on nix1, NPU-draft-overlap is net-positive vs GPU-only AR,
and NPU-spec beats GPU-spec — but not for the hoped reason. The GPU DFlash
drafter is pathologically slow on the Phoenix iGPU (autoregressive block draft),
so GPU spec-decode never beats AR here. The NPU is competitive because it drafts
the block in parallel. This does not validate NPU offload on a strong discrete
GPU; it validates it *on the APU pairing that actually ships with the NPU.* The
real target to beat is AR (17.46), and only the **overlapped** NPU path clears it
with margin — the pessimistic (serial) path is marginal at low τ. That keeps the
overlap seam (draft block N+1 while GPU verifies block N) as the load-bearing
integration, not just an optimization.
