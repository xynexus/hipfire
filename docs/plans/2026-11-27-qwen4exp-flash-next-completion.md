# Qwen3.8-Flash-Next (arch 26): finishing support — plan

Branch `feat/qwen4exp-batched-prefill`. Scope: the four open gaps from the
status review, **vision excluded**. State box: halo, gfx1151, 128 GB UMA,
`Qwen3.8-Flash-Next-180B--oq4.hfq` (170 GB, experts paged).

Order is by dependency, not by size: (1) unblocks the MTP half of (4) and
makes (2)'s capture and every PPL measurement ~N× cheaper.

## 1. Batched multi-token forward (prefill)

**Now:** `SimpleAr::prefill` loops `decode_step_into` per token: ~5.8 tok/s,
a 2048-token prompt costs ~6 min. Every weight is swept once **per token**.

**Design — layer-major over a chunk of N tokens.** The win is weight reuse,
so only the weight-reading ops need to batch; recurrent and per-token ops can
stay per-row inside the layer at first (they read no weights).

| op | batched form | source |
|---|---|---|
| all dense linears | `weight_gemm(w, x[N,K], y[N,M], N)` | runtime, exists |
| GDN conv + split | `conv1d_silu_split_f32_n` | rdna, exists |
| GDN gates / l2norm / repeat | `*_batched` siblings | rdna, exist |
| GDN recurrence | `gated_delta_net_f32` (`n_tokens` arg) | rdna, exists |
| GDN sigmoid gated norm | `gated_norm_sigmoid_f32(batch=N)` | rdna, exists |
| routed MoE | host group-by-expert, `zaya_gather_rows_f32` → `weight_gemm` ×2 → `zaya_silu_mul_gate_up_f32` → `zaya_scatter_scaled_add_f32` | rdna, exists (zaya) |
| hc read/write, PLE conv, QSA select+attend | per-row loop over views, step 1 | — |
| compact Opus experts (`OqCompactG128/G256`) | **new `weight_gemm` arm** (rotate → `quantize_act_oq8` → `gemm_oq_compact_grouped_wmma`) | gap: falls to per-column GEMV today |

Each expert's weights are read once per chunk instead of once per routed
token (~10× at N=512, k=10, 512 experts).

**Numerics are not bit-identical to per-token**: Opus GEMMs quantise
activations to int8 (W8A8 / W4A8) where the GEMV path does not, and expert
accumulation order changes. Verification is therefore statistical, not a hash:

- fixture: batched-prefill last-position argmax == per-token argmax, and
  max|Δlogit| bounded, on oq4 + oq8 fixtures (add to `qwen4exp-gate.sh`);
- 180B: held-out wikitext2 PPL, batched vs per-token, |ΔPPL| < 0.5%;
- decode after prefill: argmax 1892 on the standard prompt;
- `serve_real` prefill at 8/16/32/64/512 — the slope must bend.

Steps: 1a compact `weight_gemm` arm · 1b `prefill_batch` in `trunk_gpu.rs` ·
1c `SimpleAr::prefill` + all-position logits for `forward_chunk_scored` ·
1d fuse the per-row hc/PLE loops if a trace says they matter.

## 2. Calibration reaches the routed experts (97% of weights)

**Now:** the `weight_gemv` capture tap keys on the weight's device pointer.
Paged experts live in a rotating pager buffer, so no expert is ever named
and `with_imatrix_only(["experts."])` collects nothing.

**Fix:** capture **by name at the call site**. In `moe_forward` (and the
batched MoE), when `gpu.active_capture.is_some()`, call the collector
directly with the canonical expert name for the gate_up input `x` and the
down input `inter`. The name must be what the quantizer looks up — confirm
from a `--hessian` quantize log (`ldlq: skip <name>`) before writing it.
Routed gate_up reads the layer-shared FFN input, so `pooled_donor_candidates`
can then give them LDLQ Hessians as well.

**Also (quantizer, all families):** AWQ on a Hessian that only factorises at
≥10× damping was net-harmful at every alpha tested (PPL 4.04 → 4.11 → 5.06).
Proposed rule: skip AWQ for a tensor whose LDLQ escalated damping. This
touches every family's `++` path, so it needs a separate A/B before landing.

Exit: rebuild `oq4.25++` and beat `oq4` (PPL 4.040) on held-out wikitext2.

## 3. Dense forward ~5× off its bandwidth ceiling

**Now:** ~91 ms/token of dense work where its bytes justify ~17 ms. Suspects,
by inspection of `decode_step_into`, all **host syncs or uploads per layer**:

- `download_f32(topk_idx)` per MoE layer (48 syncs/token);
- `download_f32(count)` per QSA layer;
- `upload_f32(visible)` of `pos+1` entries per QSA layer — O(context) per step;
- `upload_f32` for the seeded wide stream, `pos_buf`, and each n-gram gather.

Method: rocprofv3 kernel trace (AGENTS.md recipe) of decode on the 180B, to
get the GPU-busy fraction before changing anything. Likely fixes: persistent
`visible`/`pos` buffers, device-side expert indexing for resident layers.
Batching (1) already amortises all of these for prefill.

## 4. Eval + MTP wiring (no vision)

- **eval:** `Qwen4ExpBackend` never overrides `kld_forward`, so the daemon's
  `kld_eval` (the multi-arch PPL path) reports "cannot produce logits". It is
  `SimpleAr`, so the blanket `ChunkScoredForward` impl applies; one override
  makes arch 26 scorable through `hipfire eval`. With (1c) the scored walk
  becomes batched too.
- **MTP serving:** 60.9% acceptance is proven (`spec_decode_mtp`), and the
  speed-up needs a 2-token batched verify, i.e. (1). Then wire the head
  into the serving loop behind a setting (schema field, not env), and flip
  `mtp` in `docs/model-support.toml`.
- **docs:** the model-support row still says "NO QUALITY EVIDENCE"; the
  2026-09-24 PPL table exists. Correct it.

## Log

- 2026-11-27: plan written. GPU held by the production daemon, so work
  proceeds on fixtures; 180B measurements are queued for a daemon window.
