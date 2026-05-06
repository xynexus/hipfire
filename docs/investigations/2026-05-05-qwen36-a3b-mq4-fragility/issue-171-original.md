## Summary

Qwen3.5-MoE-family models (3.5-A3B, 3.6-A3B, 3.5-122B-A10B) at MQ4/MQ6-experts produce token-level structural attractors on multi-paragraph self-referential agent prompts when sampled with the HF-recommended `temperature=1.0, top_p=0.95, top_k=20, min_p=0.05` config. Failure is **not** 3.6-specific (3.5-A3B mq4-mq6exp-port also fails on the same prompt), is **not** an axis the routed-expert weight quantization can repair, and is **not** in the routed-expert MoE GEMV kernels. Greedy decode (`temp=0.0`) also fails on the same prompt → the forward pass produces borderline-coherent logits that wider sampling amplifies into structural loops.

Originally reported by a user running the pi-agent with hipfire 3.6-A3B: "every prompt takes 1-2 min to load then everything falls apart one way or another (max_tokens 65536, max_seq 131072, thinking on, dflash off)."

## Reproduction

Hipx (Strix Halo gfx1151), commit `d67bf02` on `feat/moe-expert-heatmap` (PR #167):

```
load qwen3.6-35b-a3b.mq4, kv_mode=asym3, max_seq=8192
prompt: ~370-token agent-style multi-paragraph "review hipfire repo before PR" prompt
generate temperature=1.0 top_p=0.95 top_k=20 min_p=0.05 thinking=true max_tokens=800
```

Result: 199 words of attractor garbage ("Wait,Iam reviewing... rewritewrittenwritebetterwritten... Scripts/\\ Scripts/\\ scripts..."), self-EOS at ~3391 chars.

Same prompt + sampler on **dense Qwen3.5-4B/9B/27B mq4** produces 529-560 words of verbose-but-coherent meta-thinking. No attractor.

## Operational fix (LANDS NOW)

**Drop default temperature to 0.3 for Qwen3.5-MoE family at MQ4.** At `temp=0.3 + top_k=20 + min_p=0.05`, 3.6-A3B produces a clean coherent 103-word answer with self-EOS. The model can do the prompt; HF's recommended temp=1.0 is too wide for our forward at MQ4.

Where: per-arch SamplerConfig default flip in `crates/hipfire-runtime/examples/daemon.rs` (or in the new SamplerConfig::hf_thinker constructor introduced in PR #167).

Side recommendation: **default `kv_mode=asym3` over `kv_mode=q8`** — with q8, 3.6-A3B at temp=1.0 degenerates to literal letter-soup ("MMEEEKKKAALLCCCC") much faster. asym3 is the safe choice for MoE.

## Investigation log (12 hypotheses, 11 negative, 1 partial)

All probes against `qwen3.6-35b-a3b.mq4` and `qwen3.5-35b-a3b.mq4-mq6exp-port` on hipx with the agent prompt and HF sampler.

### Settled NEGATIVE (cannot be the bug):

1. **Per-expert weight magnitude differs 3.6 vs 3.5** — NumPy absmax/median scan on hiptrx of every routed expert × all 40 layers. Result: 3.6 is **0.97-0.99× of 3.5** on every distributional metric. `gate_up_proj` p99 mean ratio ≈8.9 on both; `down_proj` p99 mean ≈920k-940k on both (with ≈37M max — explained below). 3.6 is fractionally **lighter-tailed**, not heavier.
2. **MQ6-experts cures 3.6** — quantized 3.6-A3B with `--format mq4-mq6exp` (26.75 GB), smoked. 469 tokens of "VLvariant qlArchitecture" attractor. **Critical control: 3.5-A3B mq4-mq6exp-port on the SAME prompt produces 3927 chars of "Wait wait wait" attractor too.** Failure is not 3.6-specific.
3. **GPU topk distribution-sensitive defect (cpu-overwrite gate)** — ported HIPFIRE_MOE_TOPK_CPU_OVERWRITE from `debug/moe-qwen-20260505` into modular crate. Verified firing (greedy gate-on diverges from gate-off on hexagons prompt: 559 vs 488 chars). On agent prompt: gate-on H.out 118 tok / gate-off F.out 271 tok, different surface form, same structural failure. **CPU FP64 topk does not cure**. Path B's `moe_topk_renorm_k8` is correct.
4. **Per-row absmax outlier-aware quant** — simulated MQ4G256 / MQ4G64 / MQ4G256+sidecar{4,16,64} / MQ6G256 reconstruction error on 768 worst-tailed rows of 3.6-A3B `down_proj`. **Cosine sim ≥0.991 across ALL schemes**. Sidecar at sensible byte budgets does not beat MQ6G256 (sidecar4: 1.1× lower MSE, sidecar16: 1.4×, MQ6G256: 17.6×). Per-element quant fidelity is fine.
5. **Atomic-add ordering in routed-down combine** — wrote diagnostic kernel pair (per-(row,krank) scratch + serial reduction). On 3.5: bit-identical to atomic-add path (3927/555 same as base). On 3.6: different attractor (3353/301 vs 3391/199), not cured. **Atomic-add ordering is not the catastrophic bug.**
6. **FP32 dot-product accumulator in routed-down GEMV** — wrote FP64-accumulator drop-in (`gemv_hfq4g256_moe_down_diag_fp64`). Same result pattern as #5: 3.5 bit-identical, 3.6 shifts attractor but not cured.
7. **FP32 dot-product accumulator in gate_up GEMV** — wrote FP64-accumulator drop-in for the gate_up kernel (K=2048, 4× more rounding compounding than down). 3.5 bit-identical to base. **3.6 became catastrophically worse — descended to literal letter-soup gibberish ("hxhhhqhzqdvdhdxdxd...") within ~140 words.** FP64 in gate_up makes 3.6 cliff harder, not better.
8. **Disable shared expert** — different shorter failure (10 words "## 12. ## Summary 756" on 3.6, 129 words CJK reverse-loop on 3.5). Both shared and routed are required.
9. **Disable routed experts** — different 49-word fa0592d-paraphrase soup on 3.6, 477 words CJK "网络/Network" on 3.5. Both required.
10. **HIPFIRE_GRAPH=0 / HIPFIRE_GRAPH_MOE=0 / HIPFIRE_FP16=0 / HIPFIRE_KV_MODE=fp32 env probes** — bit-identical output for all. Either env vars don't toggle on this code path or runtime params override.
11. **Dense 4B/9B/27B Qwen3.5 mq4 same prompt + sampler** — all three produce verbose-but-coherent meta-thinking output. No attractor. **Bug is MoE-specific in our forward path, not MQ4-specific or prompt-specific.**

### PARTIAL (operational fix, not root cause):

12. **Sampler temperature** — temp=0.3 produces clean coherent 103-word answer on 3.6-A3B agent prompt. temp=0.7 partial coherence, temp=1.0 attractor. **But greedy (temp=0.0) ALSO fails** with the same attractor class. So the sampler isn't the source — wider sampling amplifies a forward-pass borderline-coherence problem.

## Side-finding worth keeping in memory

3.5 and 3.6 are bit-invariant under several deep numerical interventions (atomic-add ordering, FP64 down, FP64 gate_up) — but only 3.5 is truly invariant. **3.6 shifts attractor under every change**, indicating its argmax decisions sit close to FP32 rounding boundaries. 3.5's same path is far from those boundaries. The Qwen team appears to have pushed 3.6's quality higher at the cost of precision robustness — common for later iterations of the same architecture.

The earlier finding "down_proj has 37M-tail outliers across both models" was on **raw weights**, not what MQ4G256-FWHT actually quantizes. Group-wise FWHT-rotated MQ4G256 absorbs most outlier energy; per-element cosine similarity to original is ≥0.991 across all candidate schemes including the current one. The raw-weight tail-of-tails finding does not translate to a quant brittleness.

## What we have NOT done that would actually root-cause it

The bug is either (A) intrinsic Qwen3.5-MoE family fragility under MQ4 at temp=1.0 on hard prompts, or (B) somewhere upstream of routed-expert MoE GEMV kernels — LinearAttention/DeltaNet drift across 30 layers, RMSNorm precision, FullAttention output_gate path, or x_residual accumulator across the 40-layer MoE stack. We cannot distinguish (A) vs (B) without a reference comparison:

- **llama.cpp same model + same prompt + same sampler** (~1 hour). If llama.cpp produces coherent output → (B), our forward has a bug. If llama.cpp also degrades → (A), model fragility.
- **CPU reference forward in NumPy/safetensors** matching the quant scheme exactly. Layer-by-layer activation diff vs hipfire forward. ~4-8 hours, definitive — localizes divergence to a specific op in a specific layer.

Both are valid follow-up work for a future investigation; neither is needed for the operational fix.

## Files

- All artifacts at `/tmp/moe-3host-debug/` (gitignored, not committed)
- Per-host investigation notes: `/tmp/moe-3host-debug/INVESTIGATION.md`
- absmax stats: `/tmp/moe-3host-debug/absmax_results.json`
- quant reconstruction prototype: `/tmp/moe-3host-debug/quant_recon_error.py`, results in `quant_recon_results.json`

## Action items

- [ ] Flip default `temperature` to 0.3 for Qwen3.5-MoE family at MQ4 (operational fix that cures user-reported failure)
- [ ] Document `kv_mode=q8` is unsafe for Qwen3.5-MoE — degenerates to letter-soup. Default `asym3`.
- [ ] (Optional, future) llama.cpp reference comparison to determine model-fragility vs forward-bug
- [ ] (Optional, future) CPU reference forward for layer-by-layer activation diff if (B) implicates a forward bug

🤖 Generated with [Claude Code](https://claude.com/claude-code)
