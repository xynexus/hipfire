# Long-decode NaN bug — agent handoff prompt

Paste the block below into a fresh Claude Code session in this repo.

---

Investigate and fix a pre-existing long-context decode bug in `qwen35::forward_scratch`: at decode positions ≥ ~3265 (confirmed) and possibly lower thresholds with other configs, the function returns 100% NaN logits, causing downstream `argmax` to panic. The bug is **pre-existing on master `4e37618b`** — not introduced by recent F16 cascade work on `perf/dflash-phase1-target-hidden-collapse`. It blocks all end-to-end long-context decode on this codebase, including the asym3-128K push that just lifted the VRAM ceiling to 128K on hiptrx. See memory entry `project_asym3_128k_session_2026_05_16_handoff` for the broader context and `Long-context !!!!!@8K: F32/Q8/asym3 hit; RoPE/position pre-existing` in `recent.md` for the symptom class.

**Work base:** branch off `origin/master` HEAD `4e37618b` into `fix/long-decode-nan`. Do NOT branch off `perf/dflash-phase1-target-hidden-collapse` — the bug exists on master and fixing it on master means the perf branch gets the fix on rebase. Worktree under `~/ClaudeCode/autorocm/hipfire/.worktrees/long-decode-nan/` (or wherever fits your repo layout).

**GPU coordination:** GPU-TCAS wave-1 shipped (memory entry `project_gpu_tcas_implemented_2026_05_16`). Use `gpu-tcas run --devices N -- <cmd>` or `scripts/gpu-lock-tcas.sh`. DON'T use the legacy `scripts/gpu-lock.sh` — leaves locks dangling and blocks parallel agents. Pick a device that won't conflict with other agents on hiptrx (devices 0/1/2/3 may have C-track work in flight — coordinate or pick k9lin's single gfx1100).

**Reproducer (~2 min):**

```bash
cargo build --release --features deltanet --example pflash_niah_bench
./target/release/examples/pflash_niah_bench \
    ~/.hipfire/models/qwen3.5-27b.mq4 \
    benchmarks/longctx/niah/niah_16k.jsonl \
    --maxgen 4 --q8kv
```

Expected: prefill completes at ~465 tok/s (no PFlash) or ~690 tok/s (with `--pflash ~/.hipfire/models/qwen3.5-0.8b.mq4`), then panics at `crates/hipfire-runtime/src/llama.rs:3751:51` (`argmax` on NaN logits). With `--pflash` the panic hits at pos=3265 (compressed prefill end). Without `--pflash` it hits at pos=10881.

**What we already know (verified empirically):**

- Prefill logits ARE clean — no NaN, finite range ~[-10.6, +18.6]. Bug is NOT in the prefill path.
- First decode step (`qwen35::forward_scratch(token, pos=N, ...)` for large N) returns 100% NaN logits (`nan=248320 out of 248320`).
- Reproduces on master `4e37618b` AND every commit on `perf/dflash-phase1-target-hidden-collapse` (B3 `77c9d407`, B2 `4c5fdb42`, HEAD `1dc264ac`) with byte-identical failure mode.
- Reproduces with both `--q8kv` and `--asym3` KV modes (test the latter to confirm if you want).
- Memory ref class `!!!!!@8K`: suspected suspects are RoPE position scaling, KV dequant indexing, or DeltaNet state drift at large `pos`.

**Investigation approach (pick whichever opens the bug fastest):**

1. **Layer-by-layer diag**: instrument `forward_scratch` (`crates/hipfire-arch-qwen35/src/qwen35.rs`) to download intermediate activations after each layer and report NaN counts. Find the first layer whose output goes NaN. The bench-level diag patch the prior session used (lines 540-560 of `crates/hipfire-runtime/examples/pflash_niah_bench.rs`, not checked in but trivial to re-write — `eprintln!` with `logits.iter().filter(|x| x.is_nan()).count()` + finite_min/max) is the pattern to copy at each layer's output.

2. **Bisect master backward** for when the long-decode bug landed. The `Long-context !!!!!@8K` memory entry suggests the bug is "pre-existing", but if it's only a few months old you could find the commit that introduced it and revert / patch precisely. Start with `git log --oneline 4e37618b~50..4e37618b` and `git bisect run` driven by the reproducer above.

3. **Position-threshold scan**: run `pflash_niah_bench` at smaller fixtures (need to generate or use `niah_8k.jsonl` if it exists, otherwise tokenize a custom prompt to ~1K, 2K, 4K, 8K positions) and find the exact threshold where decode first produces NaN. Knowing the threshold narrows the hypothesis space — if it's `pos > 8192`, it's a 13-bit position-encoding issue. If it's `pos > 4096`, it's something else.

4. **DeltaNet state**: this model is hybrid (48 LinearAttention layers + 16 FullAttention layers). DN state has a recurrent structure that could accumulate NaN over many positions. Try running `forward_scratch` at large `pos` but with a FRESH (zero-initialized) DN state vs. a normally-accumulated one. If fresh DN state at pos=10881 works but accumulated doesn't, the bug is in DN state evolution.

**Validation when you have a fix:**

- Reproducer above runs to completion with `--maxgen 64`, producing a coherent answer that matches `expected_answer_substring` in the JSONL fixture (PASS check is in the bench itself, line ~600).
- Re-run at multiple fixtures: `niah_16k.jsonl`, `niah_32k.jsonl`, `niah_64k.jsonl`, `niah_128k.jsonl`. All should PASS.
- Re-run with both `--q8kv` and `--asym3` (the bench default) — fix must work in both KV modes.
- Run with `--pflash ~/.hipfire/models/qwen3.5-0.8b.mq4` too — PFlash compresses to short positions, but the post-compression decode position still hits the same regime; verify both paths work.

**Out of scope:**

- Don't touch `perf/dflash-phase1-target-hidden-collapse` or the C-track branches. Master fix only.
- Don't expand to fixing other long-context issues (e.g. prose τ drift, attractor patterns) — just the NaN-at-large-pos issue.
- Don't refactor `forward_scratch` beyond what the fix requires.
- Don't change RoPE math globally — if RoPE precision is the issue, the fix should be a localized precision lift (e.g. compute angles in F64 then narrow), not a structural rewrite.

**Done criteria:**

- [ ] `pflash_niah_bench ... niah_16k.jsonl --maxgen 64 --q8kv` PASSes
- [ ] same with `niah_32k.jsonl` and `niah_64k.jsonl`  
- [ ] same with `--asym3` (bench default)
- [ ] canonical merge_sort bench from CLAUDE.md is unaffected: `dflash_spec_demo --target qwen3.5-27b.mq4 --draft qwen35-27b-dflash.mq4 --prompt "$(cat benchmarks/prompts/merge_sort_thinking_off.txt)" --max 256 --kv-mode asym3 --no-chatml` produces τ=13.2727 and the expected token sequence (byte-exact)
- [ ] coherence gate clean: `gpu-tcas run --devices 0 -- bash scripts/coherence-gate-dflash.sh` returns "no hard errors"

Begin by reading `project_asym3_128k_session_2026_05_16_handoff` in memory, then running the reproducer to confirm you see the same panic. Report what layer (or what mechanism) first produces NaN before attempting any fix.

---

End of prompt.
