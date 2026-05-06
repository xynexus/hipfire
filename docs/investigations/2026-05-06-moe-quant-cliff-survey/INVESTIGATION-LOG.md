# Investigation Log

Append-only timestamped log. Significant decisions, findings, and blockers
through Phase 1 and Phase 2 of the MoE quant cliff survey.

---

## 2026-05-06 14:30 UTC — Kickoff under /loop contract

Multi-session contract from Kaden. Drive Phase 1 (survey runner + execution)
and Phase 2 (Super-Expert hypothesis confirmation) end to end; halt for
human review before Phase 3.

**Contract terms:**
- Five models in scope: 3.5-9B, 3.5-27B, 3.5-A3B, 3.6-A3B, 3.5-122B-A10B
- All four GPUs busy when there is parallelizable work; no exceptions
- Pre-register Phase 2 criterion before running ablations
- No autonomous start of Phase 3
- Synthesis docs go to `docs/investigations/2026-05-06-moe-quant-cliff-survey/`,
  raw data to `/tmp/hiptrx-survey/`, code to `survey/moe-quant-cliff-2026-05-06`

**Hardware verification (hiptrx, 2026-05-06):**
- 4× AMD Radeon AI PRO R9700, gfx1201, 32 GB each = 128 GB aggregate VRAM
- Threadripper 9970X, 32 cores, **single NUMA node** (not multi-NUMA as
  contract assumed). `numactl --cpunodebind=0 --membind=0` is the only
  valid binding; node 1 does not exist on this hardware.
- 125 GB system RAM, 119 GB free at survey start

**Branch state:**
- Created `survey/moe-quant-cliff-2026-05-06` from `master` at 262e5f6
- Jinja patch already PR'd as #175 (independent, stacked on
  `feat/moe-expert-heatmap` / PR #167)

**Acknowledged conflict:**
- Contributor `fivetide` posted #171 root-cause findings 2026-05-06
  (~2 hours before this kickoff): MoE router weight at HFQ4 has 152/256
  rows below cos_sim 0.99 vs 0/256 at Q4_K_M. Cure validated by
  rebuilding with Q8 router + everything-else-MQ4: clean self-EOS on
  agent prompt.
- Per the immutable rule "Observations are observations, don't pre-decide
  what the data should show," the survey runs as designed. The data is
  independent of fivetide's evidence. If both converge, that's robust
  triangulation. If the survey reveals SE pathology orthogonal to
  router precision, both fixes apply additively.

**This iteration's deliverables:**
- `01-survey-runner-design.md` (this commit)
- `INVESTIGATION-LOG.md` (this commit, kickoff entry)
- Branch + dir structure ready for runner code authoring next iteration

**Next:** ScheduleWakeup ~25 min, continue with runner skeleton +
calibration corpus + diagnostics modules.

---

## 2026-05-06 14:55 UTC — Codex stop-time audit + remediation

Codex flagged the design + committed diagnostic code as "would produce
invalid or infeasible results." Audit findings:

### Critical infeasibilities

1. **Qwen3.5-122B-A10B not in any local cache.** Verified 2026-05-06:
   - hiptrx HF cache has only `models--Qwen--Qwen3.5-35B-A3B` and
     `models--Qwen--Qwen3.6-35B-A3B`.
   - k9lin has 9B / 27B / A3B / 3.6-A3B / various others, but NOT 122B-A10B.
   - The contract's "122B is in the local cache per inventory" was wrong.
   Action: 122B descoped from primary surveys; documented in
   `01-survey-runner-design.md` as a deferred follow-up.

2. **122B does not fit for D3 even if downloaded.** 244 GB at bf16 vs
   hiptrx 128 GB VRAM + 122 GB RAM = 250 GB total addressable, with
   transformers / Python / activation overhead pushing demand over the
   ceiling. Two viable paths if Phase 2 results warrant: weight-only
   D1/D2/D4 streamed, or int8-load D3 (different from bf16 reference).
   Decision deferred.

3. **transformers + torch NOT installed on hiptrx.** Verified via
   `pip show transformers; pip show torch` over SSH: both packages
   not found. Action: new task #35 to install ROCm-wheel torch +
   transformers before D3 unit-test. Also added trust_remote_code
   note in the design (Qwen3.5/3.6 MoE classes may not be in stock
   transformers).

4. **`benchmarks/calib/blended-32prompts.jsonl` does not exist.** Only
   `calib-1m.txt`, `calib-5m.txt`, and `profiles/` are in
   `benchmarks/calib/`. Action: design updated to derive a 32-prompt
   corpus from the 2026-05-05 7-prompt matrix + 25 sampled prompts
   from `calib-1m.txt`, committed as
   `docs/investigations/2026-05-06-.../calibration_corpus.jsonl`
   in Phase 1A.

### Methodology corrections

5. **FWHT seed mismatch in 2026-05-05 simulation.** The committed
   `quant_recon_error.py` uses `numpy.random.default_rng(0xCAFEBABE)`,
   which produces a DIFFERENT sign table from production
   hipfire-quantize's `gen_fwht_signs(42)` and `gen_fwht_signs(1042)`
   (LCG with `state * 1103515245 + 12345 & 0x7fffffff`, bit
   `(state >> 16) & 1`). Verified via grep at
   `crates/hipfire-quantize/src/main.rs:1530-1531` and at
   `crates/hipfire-quantize/src/main.rs:1906-1907`. Inter-scheme
   relative comparison (MQ6 vs MQ4 vs sidecar) is preserved by the
   simulation, but absolute MSE values do not match production.
   Action: header docstring added to `quant_recon_error.py` flagging
   this; new survey runner uses production seeds in
   `scripts/quant-survey/quant_ops.py` (Phase 1A code).

6. **NRMSE definition was ambiguous.** Original wording
   `sqrt(MSE / var(reference))` could be read several ways. Action:
   D1 spec now defines explicitly as
   `NRMSE = sqrt(MSE) / sqrt(var(reference))` with `MSE = mean((ref - dq)^2)`
   and `var = mean((ref - mean(ref))^2)`. Lower is better; 0 is perfect.
   Mean cosine similarity is also reported per tensor for compatibility
   with 2026-05-05 results.

7. **"37M outliers" was a ratio, not magnitude.** The
   `expert_absmax_stats.py:124` ratio is `absmax / max(median, 1e-9)`
   per row. A row with one extreme value among 1408 typical values
   produces a ratio of millions. Actual absmax magnitudes in transformer
   weights are O(0.1-10). Action: D2 spec now reports BOTH absolute
   absmax and ratio statistics with explicit mean/p50/p99/max for each.
   Outlier classification threshold is on **ratio_p99 z-score**, not
   raw absmax.

### Other corrections

8. **Single-NUMA hiptrx topology.** Contract assumed multi-CCD; verified
   single-node with `numactl --hardware` (only `node 0` present, 32
   cores, 122 GB). Design's parallelization block keeps
   `--cpunodebind=0 --membind=0` for all 4 workers; example bash
   updated to remove invalid `--cpunodebind=1` references.

9. **Round 1 model name list.** Removed `qwen3.5-122b-a10b` from CLI
   choices.

### Remediation status

- 01-survey-runner-design.md: edits applied this iteration. Will commit
  along with this log entry.
- 2026-05-05/quant_recon_error.py: header docstring updated this iteration.
- New tasks: #35 (hiptrx env install), #36 (rsync 9B/27B from k9lin).
- Task #29 (unit-test on 9B) now blocked by #35 + #36.
- Task #31 (122B survey) restated as deferred.

**Next:** ScheduleWakeup ~25 min, continue with runner code authoring
under the corrected design. Order: (1) `quant_ops.py` with production
seeds, (2) `safetensors_reader.py` (lift from 2026-05-05's bf16-aware
reader), (3) `d1_nrmse.py`, `d2_down_proj_max.py`, `d4_fwht.py`
(weight-side, fast to author + test on small tensors), (4) calibration
corpus derivation, (5) `d3_activation.py` + `survey_runner.py` main.
D3 needs the env on hiptrx; if it's still uninstalled when I get to it,
authoring waits + I move on with weight-side surveys on k9lin first.

---
