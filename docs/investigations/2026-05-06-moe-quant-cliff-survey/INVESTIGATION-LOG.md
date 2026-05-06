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
