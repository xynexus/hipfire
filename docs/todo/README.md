# docs/todo — what is live, what is closed, what is waiting on a decision

Triaged 2026-08-31 across all 41 documents. Nothing had ever been retired from
this directory, so a status line here was as likely to be stale as current —
two of the five closures below were verified by running the thing, and one had
a status line that described behaviour the code no longer has.

**Convention.** A finished document is not deleted; it keeps its write-up and
gains a `> **CLOSED ...**` banner at the top saying what closed it and how that
was verified. Same reasoning as `BUGS.md`: the value of a finished item is
stopping someone re-deriving it, which a deletion loses. Rejections count as
finished work.

## Closed

| doc | closed by |
|---|---|
| `2026-07-20-bf16-moe-expert-split.md` | commit `f6d031b7e`, verified present |
| `2026-07-21-bf16-q8-kv-batched-prefill-gfx1151.md` | root cause was missing BF16 projection arms; guard REMOVED, prefill 58.5 -> 1646.3 tok/s. Its old status line was stale |
| `2026-07-21-embed-per-row-lloyd-codebook.md` | REJECTED by measurement |
| `2026-07-22-hfq-compose-dflash-tria.md` | shipped in `hfq_compose.rs` + `hipfire model compose` |
| `2026-08-25-regen-gfx1103-tiny-state-baselines.md` | baselines recorded; `tiny-state-gate: PASS (18 cell(s))` verified by running it |
| `2026-08-09-deltanet-state-handover.md` | self-declared DONE in its own title; kept as the record of what was removed |
| `free-tensor-provenance.md` | `BufferOrigin` tag added; `dispose` routes Pooled/Direct/NonOwning, and its warning cites this doc |

## Unblocked since they were written

- `2026-08-12-batched-prefill-reconciliation.md` — its ⚠️ BLOCKER (does master's
  grouped-WMMA read resident qt=34/37 experts correctly at offset 0?) is
  **resolved**: the branch's `weight_byte_offset` fix landed on master
  independently, formula for formula. Item (1), the eligibility unlock and the
  actual perf win, is no longer gated. The port itself is still real work and
  wants GPU validation.
- `2026-08-11-122b-paged-serving.md` — partially superseded by
  `2026-08-12-handover-indexed-oq-moe.md` on the decode blocker; read the
  handover first.

## Blocked on a decision only the owner can make

Recorded in `DECISIONS-PENDING.md` rather than left inline, so they are countable.
Each names the fork and what each branch costs. Implementation on these is
deliberately not started.

## Corrected in place — the status line was wrong, the doc is still open

| doc | what had changed |
|---|---|
| `2026-08-20-handover-dflash2-qwen38-27b.md` | its open bug is FIXED and the dense/MoE asymmetry has **inverted** — MoE is now the broken side, so its dense reproducer reproduces nothing |
| `2026-08-12-batched-prefill-reconciliation.md` | its ⚠️ BLOCKER is resolved; the branch's `weight_byte_offset` fix landed on master |
| `2026-08-12-router-bf16-codec-upgrade.md` | the "Lut3 in VRAM" half LANDED; only the Huff->Lut3 recode-at-load remains |
| `2026-08-07-kld-eval-throughput-remaining.md` | the transcode it asks for is written (`storage::transcode_resident`); it needs a call site, not an implementation |
| `2026-08-12-handover-indexed-oq-moe.md` | "defaults ON" vs "reverted" reconciled: ON, with a `% 256` admissibility guard |
| `2026-08-08-quant-benchmark-queue-handoff.md` | its "in flight" jobs are dead and the expensive artifacts are gone — a restart, not a resume |
| `2026-08-26-coresident-training-m8.md` | "unblocked" is true of the code, but neither artifact exists on this box |
| `2026-08-27-oq4-moe-prefill-coverage.md` | its open question is answered: the fixture's K=2 fails `moe_prefill_topk_shape_supported`, so the fix is to the fixture |
| `2026-08-04-lut3-resident-kldrefs.md` | PARKED by its own criterion — largest kldref is 2.31 GiB, 1.38x decides nothing |
| `2026-07-21-tiny-quant-mixed-precision-opus.md` | `TierPlan` is LFM2-gated, so take its own stated fallback |

## LEGACY-TODO.md — passed 2026-09-01

666 lines, 21 sections, now carrying a per-section verdict table. **7 sections
are already DONE** (vision embedding cache, PFlash checkpoint/resume, native
Hessian collector, tiny fixtures + golden tripwire, P2c state-container
unification, DeltaNet state precision, and the Active list's accelerator
inventory), 2 are PARTIAL, 4 are genuinely OPEN, and its "Active" list turned
out to be a legitimate progress tracker rather than drift — past-tense
sub-bullets are landed work recorded under still-open parents.

Six sections remain individually unverified and are named in the banner.

Nothing is struck through there: it is a legacy catch-all whose entries carry
reasoning worth keeping, so each gets a verdict rather than a deletion.

## Live, and startable without any decision

The rest. Where a doc lists "open questions", check whether they are questions
for a person or experiments for a machine — most are the latter and are
answerable by running the thing.
