# DFlash Phase E — DSpark heads on the NPU

Add the DSpark epilogue (markov + confidence heads) on the drafter's final block
hidden, and match the reference. Gate E: head outputs match `dspark_core`'s
confidence + markov values, and truncation fires at the same positions.

## Weights (the ONLY DSpark sidecar available locally)

`/srv/hipfire/models/qwen3-0.6b.bf16.dspark.hfq` — HFQ arch_id 22, 64 tensors, 872 MB,
all F16 (qt=1). Head tensors:

| tensor | shape | role |
|---|---|---|
| `main_proj.weight` | [1024, 5120] | fc-equivalent: 5 target layers × 1024 → 1024 |
| `main_norm.weight` | [1024] | rmsnorm after main_proj |
| `markov_head.markov_w1.weight` | [151936, 256] | embedding: vocab × rank |
| `markov_head.markov_w2.weight` | [151936, 256] | Linear rank→vocab (stored [vocab,rank]) |
| `confidence_head.proj.weight` | [1, 1280] | 1-row GEMV over `hidden + rank` = 1024+256 |
| `confidence_head.proj.bias` | [1] | qwen3 has a bias (deepseek4 does not) |

Metadata keys: `dspark_block_size`, `dspark_target_layer_ids`, `dspark_markov_rank`
(256), `dspark_noise_token_id`, `dspark_enable_confidence`,
`dspark_confidence_uses_normed`, `norm_eps`.

**NOTE ON PAIRING:** this sidecar is for **qwen3-0.6b** (hidden 1024, vocab 151936),
NOT the Qwen3.5-9B DFlash body validated in Phases A–D (hidden 4096). The heads are
independent modules, so Gate E is validated at THIS sidecar's dims — that is a real
head-kernel validation against the reference. Pairing heads with our 9B body needs a
9B DSpark sidecar, which does not exist locally (the only other DSpark repo,
`models--deepseek-ai--DeepSeek-V4-Flash-DSpark`, has NO safetensors shards downloaded
— 12 MB of config/index only — and is a different architecture). State that
limitation plainly in the results; do not imply an end-to-end 9B DSpark run.

## Exact head math (from `dspark_core.rs::run_heads`, the reference)

```
normed      = rmsnorm(x_head, stage_norm, eps)            # [block, hidden]
logits      = lm_head @ normed                            # [block, vocab]  (target's lm_head)
out_ids[0]  = prev_token
for i in 0..block:                                        # SEQUENTIAL — see below
    emb        = markov_w1[out_ids[i]]                    # [rank] embedding lookup
    if enable_confidence:
        hidden_i = normed[i] if confidence_uses_normed else x_head[i]
        conf[i]  = confidence_proj @ [hidden_i ; emb] + bias   # 1×1280 GEMV -> scalar
    bias_v     = markov_w2 @ emb                          # [vocab]  (rank->vocab GEMV)
    out_ids[i+1] = argmax(logits[i] + bias_v)
truncate where the block's mean confidence < conf_threshold
```

**The markov loop is inherently SEQUENTIAL**: `out_ids[i+1]` depends on the argmax of
slot `i`, so it CANNOT be batched across block positions the way attention heads were.
Expect `block` sequential steps. Per step: an embedding gather (rank 256), a tiny
confidence GEMV (1×1280), a `[151936,256]` markov GEMV (39 M weights) + a vocab argmax.

Cost note: the markov GEMV streams 39 M weights per position (~623 M per block of 16)
— that, not the confidence head, is the head budget. The `lm_head` GEMV belongs to the
TARGET model (the draft owns no vocab head), so it is expected to stay off-NPU;
validate the heads with a supplied/reference `logits` rather than porting lm_head.

## Build order

1. **numpy reference** (`tools/npu/dspark_ref.py`): read the sidecar (HFQ reader:
   see the inline reader in `dflash_body_npu.py` / `dflash_ref.py` for the format —
   32-byte header, JSON metadata, index, then data at `data_offset`), implement the
   loop above exactly, and dump per-slot `conf[]`, `out_ids[]`, and the truncation
   point for a fixed seed input. This is the golden for the NPU work. If feasible,
   cross-check it against the Rust `dspark_core::run_heads` (GPU) for a few slots.
2. **confidence head on NPU** — a 1×1280 GEMV per slot. Tiny; the plan explicitly
   allows host-side, so measure whether it is worth a dispatch at all.
3. **markov head on NPU** — embedding gather + `[151936,256]` GEMV via the proven
   int8 projection path (`oq_gemm_design`, per-row int8 like the body uses) + argmax.
   Validate values vs the numpy reference.
4. **truncation** — mean-confidence threshold; confirm it fires at the same positions
   as the reference.

## Validation (same honest gate as Phases C/D)

Kernels are bf16/int8 → gate on **cosine** vs the reference AND vs an int8/bf16
precision reference; never a fixed absolute vs an f16/f32 golden. For `out_ids` and
the truncation point, require EXACT match (they are argmax/threshold decisions —
report any slot where the argmax differs, since int8 error can flip a near-tie).

## Env / guardrails (same as Phase D)

- Fork toolchain `~/mlir-aie-312/venv312`; ALL NPU loads through the shared
  `CachedXRTRuntime` (`aie.utils._get_default_npu_runtime()`) — a private
  XRTHostRuntime exhausts Phoenix hw-contexts (err=-22).
- `@iron.jit` designs must pass `source_files=[<the .cc>]` so `.cc` edits invalidate
  `~/.npu/cache`. Core tile 64 KB; only 2 inbound / 2 outbound DMA channels per tile
  (ObjectFifo endpoints are static for the whole program) — >2 operands must go
  through memtile.
- AIE traps already hit: runtime `aie::broadcast<bfloat16>(scalar)` miscompiles (use a
  float broadcast); a `noinline` helper around a scalar reduction miscompiles (inline
  it); degree-2 exp poly is too coarse (degree-6).
- graphify before grepping source. Don't loosen tolerances to pass.

## Results (Gate E) — MET

`tools/npu/dspark_ref.py` (f32 golden) and `tools/npu/dspark_heads_npu.py`
(int8 host sim + NPU), seeds 0/6/7, `block_size = 7`.

| tier | markov bias cosine | out_ids | truncation |
|---|---|---|---|
| int8 host sim vs f32 | 0.99999997 | 21/21 exact | match |
| NPU vs f32 | 0.99999997 | 21/21 exact | match |
| NPU vs int8 host sim | max\|delta\| **0.0** | 21/21 | match |

**Zero argmax flips.** The NPU integer GEMM reproduces the host int8 simulation
bit-for-bit, so the residual cosine gap is quantization, not kernel error.
Truncation was swept over thresholds {0.25, 0.35, 0.40, 0.45, 0.50}, exercising
firing positions {1, 2, 3, 7}; every position matched the reference.

### Confidence head: does NOT earn a dispatch

| | latency | confidence cosine vs f32 |
|---|---|---|
| host f32 dot | **0.5 µs** | 1.0 (exact) |
| NPU int8 GEMV | **1640 µs** | 0.99988 (max\|delta\| 1.7e-2) |

~3300× slower *and* less accurate. The `[1, 1280]` proj must zero-pad to
`[32, 1280]` (`m % (4*r) == 0`, `M % m == 0`, `M//m` even) and the activation to
16 columns, so 31/32 × 15/16 of the MACs are padding. Keep it host-side, as the
plan permitted. The NPU path is retained in the harness only as a measurement.

### Markov head cost (the head budget)

**7 dispatches per block** (one per slot — the loop is sequential, `out_ids[i+1]`
needs slot `i`'s argmax) at **25.6–26.8 ms/dispatch** ≈ **180 ms/block**.
`markov_w2` is 38.9 MB int8 and stays NPU-resident (upload 0.05 s, once).

The activation batch must be a multiple of 16 (`n % (2*t) == 0`, t=8), so a
1-row GEMV wastes 15/16 of its MACs. That is the obvious lever, but it does not
rescue the head: this is the single-core `int_matmul`, and 180 ms/block is far
off any useful drafting budget. A multi-core design and/or an on-chip-resident
tiling of `markov_w2` would be the next step if this head is to move.

### Corrections to this plan's earlier text

- Truncation is **not** a mean-confidence threshold. `mtp_step` truncates at the
  **first** slot whose `sigmoid(conf) < conf_threshold`, keeping ≥ 1 slot.
- The sidecar is not "all F16": the head tensors are F16 (qt=1), but the body
  layer weights are Q8F16 (qt=3).
- `dspark_block_size` is **7**, and `confidence_uses_normed` is **true**.
- The container header carries the tensor count, and index records have no
  explicit offset (payloads are sequential from `data_offset`) — the older
  inline reader in `compare_qwen3_embedding_reference.py` does not parse it.

### Scope limitation

This is the **qwen3-0.6b** sidecar (hidden 1024), not the Qwen3.5-9B DFlash body
from Phases A–D. Gate E validates the head kernels at these dims. It is not an
end-to-end 9B DSpark run, and no 9B DSpark sidecar exists locally.

The f32 reference is a line-by-line mirror of `dspark_core::run_heads` +
`mtp_step`, verified by reading the Rust. A **runtime** differential against the
GPU implementation was not run (it needs a GPU harness that uploads synthetic
`x_head`/`logits` into `run_heads`, which does not exist yet) — so the reference
is validated by construction, not by execution against the Rust.
