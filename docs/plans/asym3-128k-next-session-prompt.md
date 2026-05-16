# asym3-128K push — post-context-clear starting prompt

Paste the block below into the next Claude Code session in this repo.

---

You're picking up the mid-flight asym3-128K DFlash VRAM push from a prior session. Read `project_asym3_128k_session_2026_05_16_handoff` in memory FIRST — it's the load-bearing handoff doc. After that, glance at `project_dflash_net_loss_on_prose_2026_05_15` (KV-mode prose τ baselines — note the `--chatml` fixture clarification at the top) and `project_gpu_tcas_implemented_2026_05_16` (wave-1 GPU-TCAS shipped; use `scripts/gpu-lock-tcas.sh` or the new `gpu-tcas` wrapper instead of the legacy `gpu-lock.sh`).

**Where things stand (24 GB k9lin → 32 GB hiptrx):**

Primary branch is `perf/dflash-phase1-target-hidden-collapse` (HEAD `1dc264ac`, pushed to origin + hiptrx). 5 perf commits + 3 docs + 1 revert. Lifts ctx ceiling from 24K → **65K on 24 GB k9lin** and to **128K-VRAM-only on 32 GB hiptrx** (see NaN caveat below). Token-byte-exact on the canonical merge_sort code bench at every commit. F16 cascade is innocent on prose τ when measured under the production fixture (`--chatml --max 120`) — re-bisect shows master 3.39 / B3 4.50 / HEAD 3.67, all in the 3.4-4.5 band.

Four C-track subagents shipped substep-1 each, branches pushed to hiptrx (not origin):
- `feat/c1-fused-fc-wk-wv` (24eb84cb) — ready for integration. Chunk-sized `target_hidden_proj` + inverted FC/wk-wv loop. Byte-exact. ~1.3 GB at ctx=128K.
- `feat/f16-attention` (00cff056) — substeps 1+2 of 4 byte-exact. Step 3 subsumed into 2. Step 4 (F16 `attention_dflash` kernel + F16 Q rotation) needs re-spawn — this is where the VRAM win (~0.33 GB) actually lands.
- `feat/c2-asym-kv-cache` (83dd7924) — Q8 K cache (not asym3 — the existing asym3 kernels assert head_dim=256, drafter is 160). Saves 298-598 MB scaling. New `kv_cache_read_q8_0_to_f32` kernel. Substeps 2+3 need re-spawn.
- `feat/c3-asym-target-hidden` (367ec4c7) — storage indirection + asym3 quant/dequant kernel pair (scaffold, not wired). NET NEGATIVE on VRAM until substep 3 retires the F16 intermediate. Latent eviction bug fixed inline. Substeps 2-4 need re-spawn.

**Critical pre-existing bug (out-of-scope but blocks end-to-end 128K):**

`qwen35::forward_scratch` produces 100% NaN logits at decode positions ≥ ~3265 (confirmed via `pflash_niah_bench` at 16K). Prefill is clean; the bug is in the per-token decode forward at long positions. Reproduces on master 4e37618b AND every commit on the perf branch — byte-identical failure. PFlash compression itself works correctly. The 128K hiptrx ceiling is **VRAM-only** — the model can hold the KV cache but can't decode at those positions until this bug is fixed. Matches the `Long-context !!!!!@8K: F32/Q8/asym3 hit; RoPE/position pre-existing` reference in `recent.md`.

**Two coordinator-side gotchas that bit the prior session:**

1. **GPU lock**: the legacy `scripts/gpu-lock.sh` is whole-host advisory. GPU-TCAS (wave-1, shipped) supersedes it with per-device reservations. Use `gpu-tcas run --devices N -- <cmd>` or `scripts/gpu-lock-tcas.sh` (the drop-in shell ABI). Set `HIPFIRE_GPU_LOCK_LEGACY=1` only as a regression-pin escape hatch. Do NOT use bare cargo / dflash_spec_demo / coherence-gate-dflash on the GPU without going through the wrapper.

2. **Bench fixtures**: q8 prose τ measurements MUST use `--chatml --max 120` (the production daemon fixture from the memory entry). `--no-chatml --max 300` produces a different distributional regime — τ drops to ~1.0-1.2 on master because the chat template that biases acceptance isn't applied. If a bench claims a prose-τ regression vs memory, FIRST verify you're using `--chatml --max 120` against the exact byte-identical prompt before declaring the memory wrong.

**Suggested next moves (pick what makes sense):**

1. **Integrate C1 to master** — it's byte-exact, ready, low risk. Open a PR for `perf/dflash-phase1-target-hidden-collapse` (or just C1's commit on its own branch) and land. Then rebase the other C-track branches onto post-C1 master.
2. **Re-spawn agent B for F16-attention substep 4** — biggest unrealized lever from the F16 cascade, ~0.33 GB at 64K + retires the leftover F32 scratches. Plan: `docs/plans/c-track/f16-attention.md` (substep 4 = F16 attention_dflash kernel + F16 Q rotation).
3. **Re-spawn agents C/D** — corrected validation fixture (`--chatml --max 120` for prose, master Roman empire q8 baseline τ ≈ 3.4) is now in the plan docs after the revert. Substeps 2+ for C2 (V→Q8) and C3 (wire asym3 quant on top of F16 intermediate, then drop F16 intermediate) are the next steps.
4. **Investigate the long-decode NaN bug** — orthogonal but blocking real 128K. Bisect older master commits to find when it landed; the bug might be a recent regression with an obvious fix. Tools used last session: `pflash_niah_bench` at 16K NIAH + a small diag patch around `argmax` in the bench (the patch isn't checked in — re-write it as a few `eprintln!` lines on the downloaded logits' NaN count, finite range, position).
5. **Push GPU-TCAS to origin** if not already (wave-1 just shipped — coordinate w/ whoever owns that branch).

If you're short on time, do **(1)** alone — that closes the highest-value, lowest-risk piece of work and unblocks downstream rebases.

Before any GPU work: read the GPU-TCAS section above carefully and use the new wrapper. Don't repeat the lock-dangling failure from last session.

---

End of prompt.
