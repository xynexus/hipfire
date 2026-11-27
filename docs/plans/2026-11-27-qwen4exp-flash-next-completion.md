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

### MTP serving — design (LANDED 2026-11-27, see log)

The head is a SEPARATE sidecar artifact (`spec_decode_mtp <base> <mtp>`), so it
needs a schema field naming it (an absolute `Path`, `Exists`), loaded by the
factory. A verify step is `prefill_chunk_into` over `[t, draft]` (n=2,
`all_logits`). The cost is rollback when the draft is rejected (~39%):

- re-running the accepted token costs a full trunk step, which takes the win
  from 1.25x to ~1.0x at 61% acceptance — not worth wiring;
- so roll back from a checkpoint AFTER ROW 0, which is exact and free of
  compute: GDN recurrence + conv ring already have snapshot variants
  (`gated_delta_net_f32_batch_seq_opts`, `conv1d_silu_split_f32_n_opts`, used
  by DFlash rollback); the PLE ring runs per row in `ple_tail`, so copy
  `conv_state` after row 0; the QSA caches only need `len` reset.

Expected ~1.25x decode at 61% acceptance (1.28 trunk-equivalents per verify
for 1.61 tokens). Needs its own serve loop beside `run_simple_ar`.

## Log

- 2026-11-27: plan written. GPU held by the production daemon, so work
  proceeds on fixtures; 180B measurements are queued for a daemon window.
- 2026-11-27 (`7a8cd7670`): landed on fixtures, `tests/qwen4exp-gate.sh` PASS.
  - (1) batched prefill. Fixture max|Δlogit| vs per-token, prefill + one
    decode step, single chunk AND chunk=3: bf16 1.5e-3 of scale (BF16 GEMM vs
    GEMV; 1.2e-7 when both arms were f32), oq8 0.7%, oq4@A8 1.0%, mi640
    compact-G128 experts@A8 0.9%. **oq4 at the default small-batch W4A4: 13-15%,
    argmax unchanged** — the shared `weight_gemm` policy (qwen35 production
    prefill runs O/down at A4). Whether qwen4exp should opt out is a 180B PPL
    question: batched vs per-token PPL decides it.
  - (2) expert capture by name (`capture_routed_input`); calibrate example's
    off-by-one feed fixed and now batched.
  - (3) BF16/F16/Q8F16 linears load native (were f32: lm_head 2.54 → 0.68 GB
    per token, ~1.7 GB more across mix_up/router/shared down). Decode upload
    leaks freed (QSA `visible` was O(context²) over a generation).
  - (4) `kld_forward` → batched scorer; fixture NLL matches the walk to 1e-4.
- 2026-11-27 **180B measured** (halo gfx1151, `Qwen3.8-Flash-Next-180B--oq4`,
  experts paged LRU 8 GiB, warm kernels):
  - prefill, batched: 4/8/16/32/64/512 tokens = 1.28/1.84/2.26/3.30/4.46/
    **10.55 s** (512 → 48.5 tok/s). Per-token: 64 tokens 9.34 s; the old 512
    extrapolation was ~88 s. **~8x at 512**, and the slope bends.
  - decode unchanged, 0.17-0.21 s/tok (August 0.18-0.19). Keeping the dense
    linears native did NOT move decode — issue (3)'s bottleneck is elsewhere
    (cold expert page-ins: ~3-6k cold loads per 32-step run). Next: rocprofv3.
  - held-out PPL, `wikitext2-1024s-2048ctx.txt`, 4095 scored tokens:
    per-token walk **4.0411** (reproduces 2026-09-24's 4.0400) in 724 s;
    batched Oq4@A4 (shared default) 4.0868 (+1.13%), @A8 4.0709 (+0.74%),
    **@A16 4.0550 (+0.34%)** at 116-136 s — A16 is no slower, so qwen4exp's
    prefill now requests it (`weight_gemm_oq4_act`). The residual +0.34% is
    the int8 activations of the Oq8 (GDN) and compact-G128 GEMMs.
  - calibration smoke: first run OOM'd — every per-expert imatrix accumulator
    staged a 256-row tile (~80 GB over 49152 accumulators); imatrix-only
    accumulators now stage 16 rows. Rerun: 16384 tokens in 535 s (~31 tok/s,
    was ~3), **23597 / 24576 experts** (96%) carry an imatrix, keyed
    `…experts.E.{gate_up,down}_proj.weight.imatrix`, 0.29 GiB package.
  - `tiny-affected-gate`: quant 12/14 pass; the 2 fails (qwen3_5_moe oq8 KLD
    0.0141 vs 0.00156; qwen3_5_moe_indexed oq8++ 0.000333 vs 0.000446)
    reproduce BIT-IDENTICALLY with origin/master's `weights.rs` — pre-existing.
    tiny-state: inconclusive, no baselines at HIP 7.16 for any family.
- **Next:** rebuild `oq4.25++` from a full calibration (dense Hessians + the
  expert imatrix) and compare with oq4 4.0411 on the same corpus; rocprofv3 a
  decode run; MTP serving per the design above.
- ~~Queued for a GPU window~~ (done above):
  `tiny-affected-gate --require-coverage` (selects 18 families via
  `weights.rs`); 180B `serve_real` prefill at 8/16/32/64/512; decode argmax
  1892 + tok/s before/after the native-weights change; held-out PPL batched vs
  `HIPFIRE_QWEN4EXP_PER_TOKEN_PREFILL=1`, default vs ACT_BITS=8; a calibration
  run confirming per-expert imatrix keys land, then `oq4.25` with expert
  imatrix vs `oq4` (4.040).
- 2026-11-27 **decode profiled** (180B oq4, 16-token prompt + 64 decode
  steps, rocprofv3 + `HIPFIRE_PAGER_TIMING=1`): ~155 ms/token wall, GPU
  kernels only ~45 ms of it (3.57 s total over the run, ~30% busy during
  decode). The rest is COLD EXPERT PAGE-INS: 7.7 s of ~12 s, 525 us each =
  read 173 + **host repack 196** + upload 154. Dense GEMVs are near bandwidth
  (lm_head Q8 3.0 ms for 0.68 GB ≈ 226 GB/s). Expert-cache budget sweep, 160
  steps: 8 GiB 0.17 s/tok (default), **16 GiB 0.13**, 24 0.13, 32 0.15, 48
  0.13 — 16 GiB is the knee (~−24%); past it GTT residency eats the page
  cache the misses are served from. Default NOT changed (multi-model memory is
  the operator's call; should become a schema setting, not the debug env).
  Next levers: drop the per-load repack (pre-repacked module layout, the
  2026-08-31 repacker todo), then prefetch/async page-in.
- 2026-11-27 **MTP wired into serving** (`spec_decode_step`, row-0
  checkpoint rollback, head cache filled over the prompt and on hits without
  running its attention/MoE). Sidecar rebuilt from the local `.hfa`:
  `Qwen3.8-Flash-Next-180B--mtp.oq8.hfq` (2.68 GB; reading the NFS source dir
  instead crawled at ~16 tensors/min through mmap). Fixture
  (`mtp_serve_fixture`, in the gate): teacher-forced all-hit / all-miss /
  synthetic-head arms all match plain decode within the 2-row forward's
  numerics (bf16 0.4%, oq8 0.8%, oq4 1.3%, mi640 compact 2.0% — EQUAL across
  arms). 180B, 256-token wikitext prompt, 128 greedy tokens:
  **acceptance 70.7%, 1.71 tokens/verify, but 7.24 → 5.95 tok/s (−18%)** at
  16 GiB (5.91 → 5.03 at 8 GiB). Cold loads only +8% (32760 → 35522), so the
  loss is the 2-row verify's OWN cost: 2.08 decode steps against the 1.71
  break-even (~30 ms/verify of non-paging overhead). Hence `mtp_mode=auto`
  does NOT attach this family's head; `on` does. Next: profile the n=2
  verify (16-wide GEMM tiles at n=2, per-call scratch alloc, per-row loops,
  host syncs) — under 1.71x it starts paying.
  Plain and MTP greedy outputs first differ at token 7 on the 180B; the
  fixtures attribute such flips to near-ties, but the margin there is NOT yet
  checked on the real model.
