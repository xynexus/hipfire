# Qwen3.6-A3B MQ4 sampler fragility — 2026-05-05

Tracking issue: [#171](https://github.com/Kaden-Schutt/hipfire/issues/171).
Tracking PR: [#167](https://github.com/Kaden-Schutt/hipfire/pull/167).
Hosts: hipx (Strix Halo gfx1151), k9lin (gfx1100), hiptrx (4×R9700).

## TL;DR

User reported that hipfire 3.6-A3B "falls apart" on multi-paragraph
agent prompts. Twelve hypotheses tested across three hosts and the MoE
forward kernel stack. Outcome:

1. **3.5-A3B mq4 master is fine at the existing project default**
   `temp=0.0, RP=1.05, no top_k/min_p` — 5/7 clean wins on a focused
   7-prompt validation matrix.
2. **3.6-A3B mq4 has a quality cliff under MQ4 that no sampler config
   clears** across the matrix — failure mode shifts but does not
   disappear. PR #167's HF-aligned sampler `temp=1.0 + top_k=20 +
   min_p=0.05` cures the structural attractor at the cost of breaking
   3.5-A3B and regressing math/code on 3.6.
3. The bug is **not** in routed-expert weights, GPU topk, atomic-add
   ordering, FP32 dot-product precision in routed-down, or FP32
   dot-product precision in gate_up. Three independent precision
   interventions all fail to cure 3.6, and FP64 in gate_up actively
   makes it worse.
4. The bug is either (A) intrinsic 3.6-A3B fragility under MQ4 — same
   architecture as 3.5 but pushed for higher quality at the cost of
   precision robustness — or (B) somewhere upstream of MoE GEMV
   kernels (DeltaNet drift, RMSNorm, FullAttention, x_residual
   accumulator). Cannot distinguish without a reference forward.

## Recommendation

- KEEP project default `temp=0, RP=1.05, no top_k/min_p`. Do not flip
  to HF-aligned sampler.
- Expose HF-aligned sampler as opt-in (per-request param), not a
  default.
- Document 3.6-A3B as known-fragile at MQ4. Warn user on load when
  pairing 3.6-A3B with non-greedy sampler.

## Files

| file | description |
|------|-------------|
| `INVESTIGATION.md` | Full hypothesis-by-hypothesis log with findings |
| `issue-171-original.md` | Original issue body (12 hypotheses, 11 negative, 1 partial) |
| `issue-171-update.md` | Update with 7-prompt × 5-sampler matrix data |
| `agent_prompt.txt` | The exact agent-style prompt that triggers the failure (1711 bytes, md5 `4d348213eb55e981f7b1bb0195d76015`) |
| `repro_failing.jsonl` | Drop-in daemon-wire reproducer at the failing HF sampler config |
| `expert_absmax_stats.py` | NumPy script for per-expert absmax/median tail-ratio scan |
| `absmax_results.json` | Output: 3.5 vs 3.6 weight magnitude statistics |
| `quant_recon_error.py` | Simulator: MQ4G256 / MQ4G64 / MQ4G256+sidecar / MQ6G256 reconstruction MSE |
| `quant_recon_results.json` | Output: per-scheme reconstruction error on 768 worst-tailed rows |

## Reproducing the failure

The exact prompt is captured byte-for-byte in `agent_prompt.txt`. Per
CLAUDE.md prompt-shape rules, **whitespace and tokenization sensitivity
matters** — copy-pasting from the rendered README will not reproduce
because Markdown collapses whitespace. Use the file directly.

```bash
# build daemon at d67bf02 (or whatever commit you're testing)
cargo build --release --example daemon

# the JSONL has $HOME un-expanded; expand it with envsubst before piping
envsubst < docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/repro_failing.jsonl \
  | ./target/release/examples/daemon
```

Expected on 3.6-A3B mq4 at `temp=1.0, top_k=20, min_p=0.05, top_p=0.95,
thinking=true`: 199 words / 3391 chars of `Wait,Iam reviewing...
rewritewrittenwritebetterwritten... Scripts/\\ Scripts/\\ scripts...`
structural attractor.

To compare against the project default (`temp=0.0, repeat_penalty=1.05,
no top_k/min_p`), edit `repro_failing.jsonl` to set `"temperature":0.0,
"repeat_penalty":1.05` and remove `top_k` + `min_p`. Output should
truncate cleanly mid-commit-list at the end of `max_tokens=800` rather
than attractor — also poor on this prompt for 3.6, but not garbage.

## Reproducing the absmax stats

Both `Qwen/Qwen3.5-35B-A3B` and `Qwen/Qwen3.6-35B-A3B` HF repos must be
present in your HF cache (~140 GB total at bf16). The script reads
safetensors directly — no GPU, no model load. Requires `safetensors`,
`numpy`, `ml_dtypes` (the latter for bf16 — fall back to a manual
u16→f32 reinterpret if missing).

```bash
python3 docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/expert_absmax_stats.py \
    --hf-cache ~/.cache/huggingface/hub \
    --out absmax_results.json
```

Run time ~10-15 min on a 64-core CPU (single-threaded). Verifies
3.6-vs-3.5 ratios are 0.97-0.99× (fractionally LIGHTER tails on 3.6).

## Reproducing the quant reconstruction simulator

```bash
python3 docs/investigations/2026-05-05-qwen36-a3b-mq4-fragility/quant_recon_error.py \
    --snapshot ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<snapshot-hash> \
    --out quant_recon_results.json --max-rows 256
```

Verifies cosine-similarity ≥0.991 across MQ4G256 / MQ4G64 / sidecar /
MQ6G256 schemes — i.e., per-element quantization fidelity is fine
across all candidate variants and is not the cause of the agent-prompt
failure.

## What's NOT here

- Activation dumps from individual layers (would have been ~GB scale; not
  worth persisting until a reference comparison shows where to dig)
- The cherry-picked diagnostic kernels (FP64 down, FP64 gate_up, no-atomic
  scratch+serial reduce) — all reverted from working tree. If a future
  investigation needs them, the `INVESTIGATION.md` describes the algorithm
  fully and they're easy to re-derive against the production source kernels.
