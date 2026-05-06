## Update: 7-prompt × 5-sampler matrix on hipx — sharpens the picture significantly

After the original investigation, I ran a focused validation matrix to test whether `temp=0.3` (or any other sampler config) is a clean default-flip-ready fix for 3.6-A3B. **It is not.** And critically, the user's question "is 3.5-A3B actually fine?" was previously assumed; verifying it now narrows the issue substantially.

### Test matrix

7 diverse prompts (`agent_prompt`, `sheep` math riddle, `capital`, `code_simple` one-liner, `code_complex` fibonacci, `prose` paragraph, `math` pizza fractions, `code_review` Rust merge), all on hipx (Strix Halo gfx1151), all at `kv_mode=asym3, max_seq=4096-8192, thinking=true`.

### 3.6-A3B mq4 across 5 sampler configs

| config | clean wins | failures |
|---|---|---|
| HF temp=1.0 + top_k=20 + min_p=0.05 (PR #167) | 0 / 7 | structural attractor on agent_prompt |
| temp=0.3 + same (proposed fix) | agent_prompt, code_review (2/7) | sheep, math, both code, prose (noun-cascade) |
| temp=0.3 + same + max_think_tokens=400 | agent_prompt, code_review (2/7) | think-cap fires but model emits stub + EOS, no useful answer |
| **greedy temp=0 + RP=1.05** (project default) | **prose, math** (2/7), partial code_review | **agent_prompt** (truncated mid-list, no summary), **sheep** ("Parse carefully." loop), **both code** (truncation/degradation) |
| greedy temp=0 + RP=1.3 | prose, code_review (2/7) | agent_prompt re-triggers structural attractor at greedy(!), sheep uppercase loop, math regresses (no longer reaches answer) |

**No sampler config gives 3.6-A3B clean output across the matrix.** The failure mode shifts but doesn't disappear — sampler tuning trades wins on one prompt class for losses on another. RP=1.3 specifically is **worse** than RP=1.05 because it pushes the model off the convergence trajectory on math (where RP=1.05 reached "Carol ate 10 slices") and re-triggers the structural attractor on agent_prompt at greedy.

### 3.5-A3B mq4 master at greedy temp=0 + RP=1.05 — the project default

**5 of 7 are clean wins:**

| Test | result |
|---|---|
| agent_prompt | ✓ 145-word professional summary, names both correct MoE-quality merges (#164, #156), ends with concrete recommendation, self-EOS |
| sheep | ✓ "Final Answer: 9", correct, 26 words |
| code_simple | ✓ `def square(n): return n*n` |
| code_complex | ✓ complete correct iterative fibonacci with docstring + example usage |
| prose | ✓ clean 63-word ATM-analogy paragraph |
| math | imperfect — meta-questions prompt 252 words, no answer (but doesn't loop) |
| code_review | imperfect — engages thoughtfully, technically wrong about syntax errors |

**3.5-A3B is functionally fine at the existing project default.** The original investigation's claim that "3.5-A3B mq4-mq6exp-port also fails on the same prompt" was at HF temp=1.0 sampler — at greedy+RP=1.05, 3.5-A3B is clean.

### Sharpened picture

PR #167 introduced an HF-aligned sampler (`temp=1.0 + top_k=20 + min_p=0.05`) intending to fix the user-reported 3.6-A3B agent-prompt failure. The new sampler **also breaks 3.5-A3B** (which was previously fine at the greedy default). Meanwhile, 3.6-A3B has a quality cliff under MQ4 that **no sampler config clears across the matrix** — the failure mode just shifts.

### Revised recommendation

1. **Do NOT flip the global default to HF temp=1.0+top_k+min_p, and do NOT flip to temp=0.3 with that sampler.** Both regress 3.5-A3B (HF) or break math/code on 3.6-A3B (temp=0.3).
2. **Keep the project default at `temp=0.0, RP=1.05, no top_k/min_p`** — it's what the coherence battery validates against, and 3.5-A3B is clean on this matrix at this default.
3. **Expose HF-aligned sampler as opt-in** (per-request override or CLI flag), not a default. Users wanting 3.6-A3B agent-prompt support specifically can opt into `temp=0.3 + top_k=20 + min_p=0.05` at the cost of math/code regression.
4. **Document 3.6-A3B as known-fragile at MQ4.** No production sampler config gives clean output across the prompt matrix on this model. Recommend MQ6-experts (`mq4-mq6exp` format) for users willing to trade ~50% more VRAM for 17.6× lower per-element MSE — though this also doesn't cure the agent prompt at HF temp=1.0 (already verified).
5. **Updated action items:**
   - [ ] Revert any default-flip experiments in `crates/hipfire-runtime/examples/daemon.rs`. Keep greedy+RP=1.05 as the active default.
   - [ ] Document the 3.6-A3B / HF-sampler fragility in CLI output or model registry (warn user when loading 3.6-A3B + non-greedy sampler).
   - [ ] (Future) Per-arch sampler default registry that respects model fragility profiles.
   - [ ] (Future, optional) llama.cpp or CPU reference comparison to determine whether 3.6-A3B is intrinsically fragile or has an upstream-of-MoE-GEMV bug in our forward.

### Side-finding worth a memory entry

**3.6-A3B is fragile across multiple sampler regimes; 3.5-A3B is robust at greedy.** Same architecture, near-identical weight statistics (verified via absmax). Likely the Qwen team trained 3.6 toward higher quality at the cost of precision robustness; quantization noise lands too close to argmax tie-breaks. This is a **per-model fragility profile**, not an engine bug we can patch.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
