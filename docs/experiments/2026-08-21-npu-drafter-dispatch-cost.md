# NPU as the spec-decode drafter on halo: measured dispatch cost

**Question.** Phase 1 left dense decode at 15.1 tok/s and ~91% of the DRAM read
ceiling. Spec-decode gets 8.55. Could the NPU run the draft — either to be
faster, or to run concurrently with the GPU verify?

**Box:** halo, Strix Halo gfx1151 + XDNA2 (aie2p), 128 GB UMA. NPU reports
`npu_tops_max: 58`, `h_mhz: 1800`, `mp_npu_mhz: 1267`.

## Drafter geometry (read from the artifact, not assumed)

`Qwen3.8-27B--dflash.oq4+.hfq`, 839 MB:

    num_hidden_layers 5     hidden_size 5120     intermediate_size 17408
    heads 32 / kv 8         head_dim 128         block_size 16
    draft_dtype oq4         group_size 256       vocab 248320

DFlash is a **block** drafter: one forward emits the whole 16-token block, so it
reads its weights ONCE per cycle, not once per drafted token. DFlash2 is the
same shape with `block_size 8`. Per block-forward: ~320M params/layer × 5 layers
× 16 rows ≈ **25.6 GMAC**.

## Measured: NPU W4A8 resident GEMM (`fullk` path, weights already on device)

    M=32  K=2048  N=2048    1282.94 us    104.6 GMAC/s
    M=64  K=2048  N=2048    1533.32 us    175.1 GMAC/s
    M=64  K=2048  N=512      373.57 us    179.6 GMAC/s
    M=64  K=8192  N=2048    4857.69 us    221.0 GMAC/s

Two readings:

- **Peak observed is 221 GMAC/s = 0.44 TOPS, i.e. 0.76% of the 58 TOPS peak.**
- **M is the underutilized axis.** M 32 -> 64 doubles the work for +20% time.
  N and K scale roughly linearly (N 512->2048 costs 4.1x; K 2048->8192 costs
  3.2x). So there is a large fixed per-dispatch cost that only large M amortizes
  — and a drafter at block 16 is exactly the small-M regime.

## The comparison that decides it

The GPU draft's qkv GEMM, from the spec-decode profile:
`gemm_qkvza_hfq4g256_wmma`, 96 calls / 225.4 ms = **2.35 ms per call**. At
N=5120x6144 and M=16 rows that is 503 MMAC per call:

| engine | GMAC/s at drafter shapes | % of its own peak |
|---|---|---|
| GPU (gfx1151, ~56 TOPS) | ~214 | ~0.4% |
| NPU (aie2p, 58 TOPS) | ~221 | ~0.8% |

**They are the same number.** Moving the draft to the NPU is a lateral move on
throughput, not a win. Both engines are dispatch- and small-M-bound at block-16
drafter shapes, not bandwidth- or compute-bound.

(The GPU figure assumes the qkv projection's N=6144. The conclusion survives a
2x error in that assumption — the two engines are within a factor of 2 either
way, and both sit under 1% of peak.)

## So the prize is CONCURRENCY, not speed — and it is not enough alone

Spec-decode at 8.55 tok/s with tau 2.000 is 234 ms/cycle, which the profile
splits as:

    draft     ~64 ms
    verify    ~48 ms
    rollback  ~122 ms   (re-runs a full target prefill per rejection)

Hiding the *entire* draft behind the verify on a concurrent engine removes 64 of
234 ms -> 170 ms/cycle -> **11.8 tok/s. Still below plain decode's 15.1.**

Combined with the KV-tier acceptance fix (tau 2.000 -> 3.375, already diagnosed
in `2026-08-21-compact-batched-prefill-blocker.md`):
3.375 / 0.170 s = **19.9 tok/s, which does clear 15.1.**

**Neither change clears plain decode alone; together they do.** And note
rollback (122 ms) is a bigger term than draft (64 ms) — it is the cheaper target
and needs no NPU at all.

## Blocker for actually trying it

The DFlash NPU body (`hipfire_xdna::dflash_body`) uses the **r14 multicore**
GEMM, not `fullk`. **There are no r14 kernel caches on halo** — `~/.hipfire/npu`
has only `embgemma_*fullk*` and `r6_*`; `/srv/hipfire` has none. The r14 caches
in the phase-0 record were built on nix1 (Phoenix/npu1). Compiling them for
aie2p needs the mlir_aie toolchain, which prior notes record as env-blocked
(placer skew).

So the phase-0 numbers (9B drafter, ~82 ms cached block wall) are **Phoenix and
a different model** and do not transfer to this question.

## Recommended order (cheapest first)

1. **Rollback** — 122 ms/cycle, the largest single term, pure GPU work.
2. **KV-tier acceptance fix** — restores tau 2.000 -> 3.375.
3. **Drafter dispatch fusion on the GPU** — it runs at 0.4% of peak; the win is
   fusing ~60 small sequential M=16 dispatches, and it helps whichever engine
   ultimately runs the draft.
4. **NPU offload** — only worth it for the concurrency, needs an r14 aie2p build
   first, and only pays off after (2).
