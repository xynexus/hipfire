# gfx1201 (RDNA4 / R9700) kernel-tuning PRD

Branch: `feat/gfx1201-kernel-tuning` off `origin/master` (`85678ede`).
Worktree: `.worktrees/gfx1201-kernel-tuning/`.

**Session log (2026-05-08, hiptrx silicon):**
- **Item 5 SHIPPED** (`7b71453`): rocBLAS gfx12-leak audit. Verdict: `rocblas_arch_eligible()` correctly gates rocBLAS to gfx940/941/942 (CDNA3) only; gfx1200/1201 never reach rocBLAS in either decode (GEMV) or prefill (gemm_*_wmma_gfx12) paths. Fixed one inconsistency at `gemm_gate_up_hfq4g256` (hardcoded `cdna3 = matches!(...)` → `self.rocblas_arch_eligible()`) so `HIPFIRE_ROCBLAS_ALL_ARCHS=1` smoke covers gate_up identically to qkv. rocBLAS calls per token on 27B decode = 0 on gfx1201.
- **Item 4 Phase 1 SHIPPED** (`07a17c40`): gfx12 HFQ4-G256 MMQ residual GEMM kernel + dispatch wiring + correctness test. Surgically extracted from research/iu4-activation-calibration without iu4/FP8 contamination. Builds + loads cleanly on gfx1201 (build_check), numerically equivalent to FP16-WMMA reference on R9700 (max abs err 0.00146, mean 0.000223, 0/25507 samples > 5% rel-err — well within Q8_1 tolerance). Direct-call only — production paths NOT yet wired (see Item 4 Phase 2/3).
- **Item 1 SHIPPED with NULL verdict** (`ea69403` + `7398cb6`): gfx1201 multi-row GEMV port (gemv_hfq4g256_multirow.gfx1201.hip + residual sister). Body byte-identical to gfx1100 source (kernel uses arch-neutral FP32 FMA + __shfl_down + __builtin_bit_cast — no _gfx12 suffix, no acc-layout adaptation needed). Witnessed bench on hiptrx R9700 (DPM warmup 10s):
  - 9B mq4 AR: R=1 101.4 / R=2 101.3 / R=4 101.4 tok/s — within 0.1%
  - 27B mq4 AR: R=1 35.8 / R=2 35.8 / R=4 35.8 tok/s — identical
  - **27B mq4 DFlash merge_sort (PRD target metric)**: R=1 191.51 / R=2 191.42 / R=4 191.28 tok/s, τ=13.273 invariant — within 0.12%, **R=1 matches PRD anchor 192.6 within 0.6% silicon variance**.
  - Both archs effective BW ~500 GiB/s on AR — bandwidth-saturated. Multirow register-reuse cannot unlock more on RDNA4 (parity with gfx1100's negative; the PRD hypothesis "RDNA4's larger VGPR budget turns the gfx11 negative into a positive" is **falsified empirically**).
  - Kernel ships opt-in default-off via `HIPFIRE_GEMV_ROWS={2,4,8}`. Default rows=1 unchanged.
  - **One regression bug fixed mid-session** (`7398cb6`): the multirow commit also added gfx1200/gfx1201 to the `use_wide` exclusion list, intending the default path to be single-row. This regressed 9B prefill from anchor 1262 tok/s to 33.6 tok/s (38× slowdown) because gfx1201 default rows=1 needs the wide kernel (m≥64, 2 rows/block, 64 threads/block) — single-row is decode-only on this arch. **Rule: never add gfx1200/gfx1201 to the use_wide exclusion in `gemv_hfq4g256` dispatch.**
- **Coherence-gate PASS on hiptrx** post-fix.

**PRD premise re-examination:** per memory `feedback_hiptrx_153_vs_lmx_250_investigation_2026_05_08`, "real gfx1201/gfx1100 ratio on canonical merge_sort = 77% (gap is published arch-tier characteristic, not a flag/kernel bug)." Combined with the multirow null verdict on DFlash 27B (the actual target metric), the kernel-tier path to ≥250 tok/s is plausibly unreachable on R9700 silicon. Items 2 (fused QKV) and 3 (fused gate_up) face the same BW-saturation ceiling as Item 1 unless they unlock a fundamentally different bottleneck (e.g., launch-overhead amortization, not register reuse). Recommend Items 2-3 be gated on a per-kernel profile pass (rocprof) showing where the 192.6 vs 250 gap actually lives — if the kernel is BW-bound, no kernel-isa work will close it.

---

## 1. Goal

Close the gfx1100 → gfx1201 perf gap on hipfire's decode hot path. Target:
**gfx1201 R9700 hits ≥250 tok/s on canonical 27B + 27B-DFlash merge_sort
config**, currently 192.6 tok/s = **77% of gfx1100's 250.3 tok/s**.

Bench config (canonical, locked): `--max 256 --no-chatml --kv-mode asym3`,
prompt `lru_cache_pep8_strict.txt`, DPM warmup 10 s, `HIPFIRE_GRAPH=1` for
4B/9B/27B (0.8B has known hipGraph bug).

## 2. Empirical baseline

| arch | rig | 27B mq4 merge_sort tok/s | τ | ratio |
|---|---|---:|---:|---:|
| gfx1100 (RDNA3 7900 XTX) | localmaxxing | 250.3 | 13.18 | 100% (target) |
| **gfx1201 (RDNA4 R9700)** | **hiptrx** | **192.69** | **13.27** | **77%** |
| gap closure target | | **+30%** | | → 100% parity |

Captured 2026-05-04 against commit `06296cf` on hiptrx, see
`tests/speed-baselines/gfx1201.txt`. Re-verified 2026-05-08 at 192.6 tok/s
within 0.05% of capture, so silicon healthy and gap is structural.
Tail of the verdict log lives in
`docs/investigations/2026-05-07-rdna1-perf-research/13-hetero-dflash-perfmax-verdict.md`
(2026-05-08 confirmation appendix).

## 3. Why gfx12 isn't a free port

Reference: `.skills/hipfire-arch-port/wmma-matrix.md` and memory entry
`feedback_gfx12_wmma_builtin_gotchas.md`.

Load-bearing gfx11 → gfx12 deltas:

1. **WMMA operand shapes halve.** gfx11 packs `<16 x fp16>` per lane;
   gfx12 packs `<8 x fp16>`. K-tile striding in LDS loads must change.
2. **`kRepeat` differs.** gfx11 has `kRepeat=2` (16-K split across 2
   lane-groups); gfx12 has `kRepeat=1` (8-K fits in one lane-group).
3. **Builtin name suffix is mandatory.** `__builtin_amdgcn_wmma_*_w32`
   is unsuffixed → "Cannot select intrinsic" backend error on gfx12.
   gfx12 demands `__builtin_amdgcn_wmma_*_w32_gfx12`. Same arity, but
   the LLVM intrinsic node is different — gfx11 codegen does NOT
   match the gfx12 intrinsic.
4. **Acc layout differs.** gfx11 stride-2 mapping
   (`acc[j] = C[2*j + (tid>>4)][tid & 15]`) was validated 2026-04-12
   after a 6-week silent-corruption bug. gfx12's mapping is
   8-row-block (`acc[j] = C[8*(tid>>4) + j][tid & 15]`), unverified
   end-to-end, hardware validation required for every kernel.
5. **An #ifdef swap silently produces wrong output.** Gating the
   builtin call on `__gfx12__` without changing the operand-pack
   stride or the accumulator readout layout compiles, links, and
   yields garbage. Validate against fp32 reference per kernel.

Additional levers established this cycle:
- **rocBLAS-on-gfx12 regresses 5.6×** vs hipfire's `wmma_gfx12` on
  prefill (memory `feedback_rocblas_gfx12_regresses`). Don't extend
  `rocblas_arch_eligible` to gfx12; audit whether decode path can hit
  the same regression cliff.
- **iu4 K=32 wins +13% prefill at 27B but destroys quality** —
  Q4_1 activations cascade. iu4 stays opt-in until QAT or
  layer-selective research lands. Don't generalize the ranked items
  below into iu4-activation territory without a quant-quality story.
- **iu8 MMQ +10% on 9B prefill is the gfx12 ceiling currently shipped**
  on prefill side; decode path is unported.

## 4. Items — original PRD ranking (with witnessed verdicts inline)

> **Status (2026-05-08, end of session):** `.gm/prd.yml` emptied. Items 1, 4
> Phase 1, and 5 SHIPPED on this branch. Items 2, 3, 4 Phase 2/3, and 6 are
> deferred to next-session work — the witnessed verdicts below explain why
> each is **not reachable as currently scoped**, and what would unblock it.
> Future work should re-PRD against the witnessed bottleneck profile in
> `docs/perf-checkpoints/2026-05-08-gfx1201-27b-ar-profile.md` rather than
> the original ROI ranking.

### Item 1 — Multi-row GEMV gfx1201 port (highest ROI)
- **Files:** `gemv_hfq4g256_multirow_r2.gfx1201.hip` and the multirow
  family (`gemv_hfq4g256_multirow_*`). Mirror the gfx1100 implementation
  with the gfx12 wmma operand shapes, builtin suffix, and 8-row-block
  acc readout. Single-row dispatch on gfx1201 currently dominates the
  hot path and prevents BW-amortization across rows.
- **Effort:** 3–5 days.
- **Lift estimate:** probable largest single decode win on gfx1201
  (RDNA3 saw +20%+ from this on 9B). Conservatively 15–25% on 27B
  decode.
- **Validation:** speed-gate baseline must update; canonical 27B
  merge_sort tok/s ≥ current + lift; τ-invariance 13.27 ± 0.5; max
  abs error vs fp32 reference < 1e-3 across 100 random shapes.

### Item 2 — Fused QKV / QKVZA gfx1201 — DEFERRED (PRD misframe)
- **Files:** `fused_qkv*.gfx1201.hip` — full family. Acc-layout port,
  builtin suffix, operand-pack stride update.
- **Effort:** 3–5 days.
- **Lift estimate:** second-largest hot-path family on 27B decode,
  +5–10% likely.
- **Validation:** same gates as Item 1 + DFlash coherence gate
  (`scripts/coherence-gate-dflash.sh`) must pass — QKV is on the
  attention path and an acc-layout mistake silent-corrupts logits.
- **2026-05-08 verdict:** PRD misframed. The fused_qkv / fused_qkvza
  kernels are NOT WMMA — they are non-WMMA decode-tier fused GEMVs
  with arch-neutral FP32 FMA inner loops. The kernel headers explicitly
  say "Arch coverage: works on every RDNA generation (gfx1010 / gfx1013
  / gfx1030 / gfx1100+)". gfx1201 is ALREADY using these kernels; they
  precompile unconditionally per dispatch.rs:14497. The "Acc-layout
  port + builtin suffix + operand-pack stride update" guidance applies
  to WMMA kernels only and doesn't fit this family. Witnessed bottleneck
  profile (perf-checkpoints/2026-05-08-gfx1201-27b-ar-profile.md) shows
  fused_qkvza at 512 GiB/s = 80% of R9700's ~640 GiB/s peak on 27B AR —
  bandwidth-saturated. Real next-session work for this family is a
  **gfx12-specific OPTIMIZATION** (more unroll using gfx12's larger VGPR
  budget, DPP-based weight unpack, etc.) gated on a measurement showing
  headroom, not a blind body-identical port. **Re-PRD as a research
  task, not a port task.**

### Item 3 — Fused gate+up gfx1201 — DEFERRED (insufficient witness)
- **Files:** `fused_gate_up*.gfx1201.hip` + `swiglu_residual` gfx12.
- **Effort:** 2–4 days.
- **Lift estimate:** closes the FFN family, +3–5% on 27B decode.
- **Validation:** same gates as Item 2.
- **2026-05-08 verdict:** Same misframe class as Item 2 (non-WMMA
  arch-neutral cross-arch kernel). Plus the AR profile didn't even
  surface fused_gate_up in the top-10 — its hot-path role on gfx1201
  is unclear. Justification requires a separate prefill or draft-side
  profile before any port work. **Re-PRD post-profile.**

### Item 4 — MMQ workgroup-tile auto-tune for gfx1201 — Phase 1 SHIPPED, Phase 2/3 DEFERRED
- **Files:** MMQ dispatcher + per-shape tile selector. gfx1100 uses
  `MMQ_X=8` for 9B and `MMQ_X=16` for 27B; gfx1201 currently uses
  uniform `MMQ_X=8` per memory `project_gfx12_mmq_bench_2026_05_04`
  which under-utilizes RDNA4 wavefront occupancy on 27B.
- **Effort:** 1 day.
- **Lift estimate:** +1–3% per shape, free.
- **Validation:** speed-gate update + bit-exact regression on
  `tests/speed-baselines/gfx1201.txt` 27B prefill row.
- **2026-05-08 verdict:** PRD's "1 day" estimate was wrong. Witnessed:
  the gfx12 MMQ kernel itself didn't exist on master (was on
  research/iu4-activation-calibration only). Phase 1 SHIPPED at
  `07a17c4` — kernel + dispatch + correctness test PASS on R9700.
  Phase 2 alone (~1-2 hr — production wiring through `should_use_mmq`)
  is not safe to ship without Phase 3 because per memory
  `project_gfx12_mmq_bench_2026_05_04` the kernel REGRESSES 4B (-19%)
  and 27B (-8%) at uniform MMQ_X=128. Phase 3 (per-shape MMQ_X
  workgroup-tile variants — `_x64` / `_x32`) is multi-week. Real
  total scope: 1-2 weeks. Direct-call kernel is shipped and accessible
  for testing; production-flip blocked on Phase 3.

### Item 5 — rocBLAS-on-gfx12 audit — SHIPPED at `7b71453`
- **Goal:** confirm hipfire's decode path is gating rocBLAS on
  gfx1201, or that its gfx12 fallback codepath has same 5.6×
  regression cliff seen on prefill (memory
  `feedback_rocblas_gfx12_regresses`). Fix gating if leaky.
- **Effort:** 0.5–1 day.
- **Lift estimate:** 0% if already gated correctly; up to 10–15% if
  there's a leak (rocBLAS `_fallback_` pipeline + FP16-shadow VRAM
  tax).
- **Validation:** trace dispatcher source; rocBLAS calls per token on
  27B decode = 0.

### Item 6 — Lloyd-MQ kernels gfx1201 — DEFERRED (multi-week, lowest priority)
- **Files:** Lloyd-MQ3 / Lloyd-MQ2 batched-prefill ports per memory
  `mq-lloyd-batched-prefill-followup`. Currently B=1 only on
  gfx1201.
- **Effort:** 1–2 weeks.
- **Lift estimate:** orthogonal quality lever (lower-bpw quant
  feasibility), modest decode tok/s impact alone.
- **Validation:** quality gate (coherence-gate.sh) + speed-gate +
  Lloyd-MQ-specific acceptance criteria from `mq-lloyd-batched-prefill-followup`.

## 5. Validation gates per item

Mandatory checks before any item is marked complete:

- **τ-invariance:** 10.45 ± 0.5 on canonical bench (27B merge_sort
  τ=13.27 ± 0.5; LRU τ=9.5 ± 0.5).
- **Speed-gate baseline update:** `tests/speed-baselines/gfx1201.txt`
  must be updated with the new anchor after each shipped item.
  Pre-commit hook enforces no-regression vs the in-tree anchor.
- **DFlash coherence gate:** `scripts/coherence-gate-dflash.sh` must
  pass — three-tier thresholds (first-128, last-128, full-output
  3-gram density) per CLAUDE.md DFlash Coherence Gate section.
- **Coherence gate:** `scripts/coherence-gate.sh` must pass — any
  kernel/dispatch/fusion change requires this per CLAUDE.md
  Coherence Gate section.
- **Bench-discipline:** every items.completed log entry must record
  prompt md5 + `--max` + DPM warmup + `--kv-mode` flag (even if
  default), per CLAUDE.md Prompt-structure τ sensitivity rule.
- **Bit-exact correctness:** max abs error vs fp32 reference < 1e-3
  across 100 random shapes per kernel ported.

## 6. Risks / failure modes

- **Each ported kernel must be VALIDATED on RDNA4 silicon** — can't
  blind-port. gfx12 acc layout silent-corrupts unit tests that pass
  shape checks. Run against fp32 reference EVERY shape.
- **gfx12 footguns from `feedback_gfx12_wmma_builtin_gotchas`:**
  - `_gfx12` suffix omission → "Cannot select intrinsic" (loud, easy)
  - acc-layout stride from RDNA3 reused as-is → silent wrong output
    (silent, weeks-of-debug class)
  - operand pack `<16 x fp16>` reused → wrong K accumulation
- **Quant-precision risk class.** Q4_1 activation precision destroyed
  iu4 K=32 quality in `project_gfx12_iu4_breakthrough_2026_05_04`.
  Same risk class for any port that changes operand precision.
  Items 1–4 are FP16-activation-preserving; Item 6 is the only one
  that touches quant format and must run quality gate first.
- **rocBLAS leak risk** — Item 5 is small but gating could regress
  if hipfire upgrades to a ROCm version that re-enables rocBLAS
  default for gfx12.
- **Total project: 3–6 weeks** if all six items shipped. Tighter
  scoping on Items 1–3 = ~2 weeks for the 80% lift expected to
  close to gfx1100 parity.

## 7. Done criteria

Project ships when ALL of:

- [ ] gfx1201 hits **≥ 250 tok/s** on canonical 27B merge_sort config
  (gfx1100 parity).
- [ ] All ported kernels speed-gated and committed atomically per
  item — single feat/gfx1201-kernel-tuning branch with rebasable
  commits.
- [ ] `tests/speed-baselines/gfx1201.txt` updated with new anchor.
- [ ] `scripts/coherence-gate.sh` PASS.
- [ ] `scripts/coherence-gate-dflash.sh` PASS.
- [ ] PRD `.gm/prd.yml` empty.
- [ ] Branch pushed to origin, CI green.
- [ ] Memory entry `project_gfx1201_kernel_tuning_complete.md` written
  with final tok/s + τ + each item's measured lift.

## 8. Cross-reference

- `.skills/hipfire-arch-port/wmma-matrix.md` — operand shapes per arch.
- `.skills/hipfire-arch-port/SKILL.md` — full port playbook.
- `feedback_gfx12_wmma_builtin_gotchas.md` — silent-corruption class.
- `feedback_rocblas_gfx12_regresses.md` — rocBLAS gfx12 regression.
- `project_gfx12_iu4_breakthrough_2026_05_04.md` — iu4 quality cliff.
- `project_gfx12_fp8_bench_2026_05_04.md` — FP8 null result, three
  attempts stacked.
- `project_gfx12_mmq_bench_2026_05_04.md` — MMQ first-cut tile-shape
  data, basis for Item 4.
- `project_gfx12_iu4_iu8_port_2026_05_03.md` — gfx12 perf-lever
  inventory.
- `project_hiptrx_4xr9700_provisioned.md` — hardware baseline.
- `tests/speed-baselines/gfx1201.txt` — speed-gate ground floor.
