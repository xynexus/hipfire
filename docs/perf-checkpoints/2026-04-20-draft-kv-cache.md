# Perf checkpoint — 2026-04-20, post draft-KV-cache

Captured after commit `f328f18` landed the rolling-target_feat cache for
the draft's per-layer K/V projection. This is the baseline for the next
wave of perf work (verify hipGraph, persist-write kernels, launch-overhead
reduction).

**Hardware:** 7900 XTX (gfx1100), 24 GB GDDR6, ROCm 7.2  
**Engine build:** `dflash` branch @ `f328f18`  
**Bench config:** `HIPFIRE_DPM_WARMUP_SECS=10`, no CASK sidecar, no DDTree,
greedy (temp=0). Cold-JIT warmed via the DPM pass.

## DFlash decode (dflash_spec_demo, --ctx 4096)

Short prompt = `def solution(...)` + one-line instruction (~30 tokens).  
Long prompt = the technical Rust+RDNA essay from
`/tmp/sidecar_bench_prompt.txt` (~512 tokens).

| Model | Prompt | tok/s | τ | cycles | accepted | vs AR |
|-------|--------|-------|------|--------|----------|-------|
| 9B    | short  | 187.8 | 3.10 | 29     | 90       | 1.46× |
| 9B    | long   | 112.1 | 1.55 | 60     | 93       | 0.87× |
| 27B   | short  |  47.4 | 4.70 | 23     | 108      | **1.05×** |
| 27B   | long   |  36.9 | 3.55 | 33     | 117      | 0.82× |

## AR baseline (bench_qwen35_mq4, --prefill 16 --warmup 3 --gen 30)

| Model | decode tok/s | effective BW    | ms/tok |
|-------|--------------|-----------------|--------|
| 9B    | 128.9        | 637.9 GiB/s     | 7.52   |
| 27B   |  45.2        | 630.9 GiB/s     | 21.86  |

## Coherence spot-check (27B short, tool-use style prompt)

Prompt: two-sum algorithm Python stub with "Implement this function step
by step." DFlash decode produced a correct, idiomatic dict-based O(n)
implementation with numbered step comments. Output is a valid Python
function. No gibberish, no loop artifacts.

Full transcript of the generated region:
```python
    # Step 1: Create a dictionary to store the numbers we've seen so far.
    seen = {}

    # Step 2: Iterate through the list of numbers.
    for i, num in enumerate(nums):
        # Step 3: Calculate the complement (the number we need to find).
        complement = target - num

        # Step 4: Check if the complement is in the dictionary.
        if complement in seen:
            # Step 5: If the complement is found, return the indices.
            return seen[complement], i

        # Step 6: Add   (truncated at --max 120)
```

Short and long outputs on both models (9B and 27B) spot-checked during
the A/B that landed the KV cache change. All coherent, on-topic, no
degenerate behavior.

## Deltas vs the pre-cache session baseline

| Config     | pre-cache | post-cache | delta |
|------------|-----------|------------|-------|
| 9B short   |   ~182    |   187.8    | +3%   |
| 9B long    |   ~110    |   112.1    | +2%   |
| 27B short  |    38.8   |    47.4    | +22%  |
| **27B long** |  **12.2** |   **36.9** | **3.02×** |

The 27B long-context win is the main signal — the cache specifically
targets L-scaled per-layer K/V GEMMs in the draft's cross-attention
projection. 9B's draft was not L-scaled in this phase (9B was already
near AR), so the cache change is a no-op for it — which is the expected
correctness outcome.

## 27B long-context cycle breakdown (phase-timer, measured pre-commit)

```
              pre-cache        post-cache
verify         52 ms            52 ms
save_from     0.3 ms           0.3 ms
pre_verify       0 ms             0 ms
draft+lmhead  321 ms            72 ms   ← optimized
post_verify   0.8 ms           0.8 ms
             ───────          ───────
total         374 ms           125 ms   ← 3.00×
```

## Next levers (ordered by expected impact × confidence)

1. **hipGraph-capture the verify forward.** `forward_prefill_batch` is
   not graph-captured today (only `forward_scratch` AR path is). 1100
   launches × ~10 µs API overhead ≈ 11 ms wasted minimum; real GPU-idle
   gaps could push that higher. Estimate: verify 52 → 35-40 ms.
2. **hipGraph-capture the draft forward.** Same treatment for the per-
   cycle draft layers. Smaller model, but dense launch topology.
3. **Persist-write for cached→k_cat D2D copies.** Lucebox cites ~9 ms/step
   from this class of fix. In our cycle, n_draft_layers × 4 small
   memcpy_dtod per layer — may add up.
4. **Fuse draft per-layer RMSNorm + GEMM pairs** to cut launches
   further.

Goal is 27B long → ≥ 140 tok/s = 3.1× AR (approximate Lucebox claim of
3.43× on the 3090). Current 36.9 tok/s = 0.82× AR; budget for hitting
3.1× is cycle ≤ 36 ms for 5.55 committed-per-cycle.

## Update — investigation findings (2026-04-20 eve)

**Hypothesis tested and rejected: reducing D2H sync count.**

We profiled the post-cache 27B long cycle and saw 95% of cycle wall
time was spent in `hipMemcpy D2H`:
```
wall=117,741 µs | launch=2,276 µs | dtoh=112,496 µs
kernel execution time (profile): 58 ms/cycle
```

So we built a GPU-side acceptance kernel (`accept_scan`) that runs
on-GPU instead of downloading verify argmax to CPU, bringing the
per-cycle D2H count from 2 to ~1 (the draft argmax still has to come
back for the `drafted` CPU vec used by spec_step reporting).

A/B measured (HIPFIRE_GPU_ACCEPT=1 vs unset):

| Config    | CPU accept | GPU accept | Δ        |
|-----------|-----------|------------|----------|
| 9B short  | 188.2     | 186.8      | -0.7% (noise) |
| 9B long   | 112.2     | 112.3      | flat     |
| 27B short | 47.4      | 47.3       | -0.2% (noise) |
| 27B long  | 36.9      | 36.8       | -0.2% (noise) |

Acceptance was byte-exact (cycles/τ/accepted matched across CPU/GPU
paths on all configs) — so the kernel works. But **total wall time
didn't change.**

**What we learned:** on a serial-dependency workload (draft → verify →
accept), combining two dtohs into one doesn't help because each dtoh
was already mostly waiting for the same critical path of kernels. The
95% dtoh figure wasn't synchronization overhead to eliminate — it was
the GPU kernel execution time *expressed as the CPU's view of waiting
on the queue to drain*.

The code was reverted (correct but zero-benefit) on 2026-04-20. Kernel
source preserved in commit history for reference.

## Revised understanding of the next lever

Kernel execution time IS the bottleneck, not sync structure. The
realistic next levers:

- **Reduce total GPU kernel time** via fusion (fewer larger kernels
  per layer) or faster individual kernels.
- **Multi-stream concurrency** — run independent GPU work in parallel.
  E.g., post-verify tape replay could start while draft_forward of the
  next cycle is already queued. Requires careful stream/event plumbing.
- **Reduce draft model cost specifically** — draft FA attention at L=512
  is a chunk of the 72 ms draft+lmhead phase. A smaller draft or more
  efficient draft attention could move the needle.

hipGraph (original plan) is still worth trying for launch overhead
elimination, but at 2.3 ms/cycle of launch API time, the upside is
smaller than originally estimated.

## Update — fusion attempt (2026-04-20 eve v2)

Tried wiring the target's existing `gemm_qkv_hfq4g256` (3 weights → 3
outputs) and `gemm_gate_up_hfq4g256` (2 → 2) kernels into the draft's
per-layer QKV and FFN-gate/up projections. Expectation: ~10-15 kernel
launches saved per cycle on a 5-layer draft.

Outcome:

- **27B draft is F32** (not MQ4 despite the `.mq4` filename) — fusion
  dispatch fell through to the F32 "separate GEMM" fallback, so the
  fused kernel never ran there. 27B numbers matched baseline because
  it was running the baseline code path.
- **9B draft is MQ4G256** — fusion dispatch fired the fused kernel,
  and output was garbage (τ=0 across all cycles). Tried purging the
  JIT/install kernel caches AND swapping the kernel variant (non-WMMA
  vs WMMA); same breakage both ways.
- **Upside is tiny anyway** — 5-layer drafts × ~3 launches saved per
  layer × ~30 µs = ~0.5 ms per cycle. Not worth deep-debugging the
  MQ4-compatibility issue for that.

Reverted. No code shipped.

**What we learned:** a `.mq4`-named file can contain F32 weights; always
runtime-check `gpu_dtype` when writing dtype-sensitive code. And the
`gemm_qkv_hfq4g256` kernel family works on HFQ4 weights but has an
untested MQ4 compatibility gap we'd need to resolve before any kernel
fusion in the draft.
