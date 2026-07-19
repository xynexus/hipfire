# Hybrid speculative decoding — GPU target + NPU DFlash + CPU DDTree/DSpark

Implementation plan for the heterogeneous spec-decode architecture: the target
model verifies on the GPU, DFlash drafts a parallel block on the NPU, and DDTree +
DSpark build and refine a candidate tree on the CPU.

Source design: `hybrid_dflash_ddtree_dspark_design.md` (external). **This plan is
not a restatement of it.** It records what the measured hardware says about that
design, corrects the assumptions that do not survive contact with it, and
sequences the work against what hipfire already has.

Companion docs:
- `docs/npu/dflash-drafter-npu-plan.md` — the NPU drafter build (Phases A–F) and
  all the block-wall measurements this plan depends on.
- `docs/plans/NEXT-STEPS.md` — DeltaNet state policy and the FP16-state follow-on.

## 0. The one number that governs the whole design

The design's §16 pipelining rests on:

```
T_NPU_draft + T_CPU_tree  ≤  T_GPU_verify
```

Measured on nix1 (gfx1103 + npu1), native driver, not projected:

| term | measured |
|---|---|
| NPU DFlash block wall | **726 ms** |
| GPU verify budget, 9B target | 57 ms |
| GPU verify budget, 27B target | 155 ms |
| GPU verify budget, 31B target | 345 ms |

**The inequality fails by 12.7× against a 9B target, before any CPU tree work.**

Attribution of the 726 ms, each term measured with the kernel pinned:

| term | now | after known-available work |
|---|---|---|
| GEMM (weight-bandwidth-bound) | 317 ms | **~32–42 ms** (multi-core W4A8, measured 3.8 → 11.58/15.14 GB/s at real weight scale) |
| attention (single-core today) | 236 ms | ~30 ms if multi-cored (8 kv-heads currently loop on ONE core) |
| host glue (quant, bf16, packing) | 143 ms | tractable, Rust-side |
| primitives (norm/rope/swiglu) | 24 ms | — |

So a realistic post-work NPU block is **~230–250 ms**, and ~100–150 ms if the glue
is also attacked.

### Consequence: the prototype target is 27B/31B, NOT 9B

The verify budget scales with target size while the draft cost does not. This
architecture becomes viable on **27B (155 ms) and comfortable on 31B (345 ms)**,
and is out of reach on 9B. The source design's §17 does not state a target model;
it should say 27B-class or larger.

This inverts the intuition that one prototypes on the small model first. On 9B the
pipeline can never hide the draft, so a 9B prototype would measure a
permanently-negative result and tell us nothing about the architecture.

## 1. What the measurements confirm in the design

**CPU DSpark (§4) is right, with numbers.** The markov head measures **180 ms/block
on the NPU vs 8.7 ms on one CPU thread** — 20×. The reasons match the design's:
the loop is inherently sequential (`out_ids[i+1]` needs slot `i`'s argmax), the CPU
is idle exactly when the drafter runs, and a 1-row GEMV against a 39 MB weight is
the wrong shape for the NPU (activation must be a multiple of 16, so 15/16 of MACs
are padding).

**Candidate-restricted scoring (§10) is exact, verified.** Top-k against full-vocab
argmax: **100% agreement at k = 16…256**, 5 seeds, teacher-forced and free-running,
holding across logit-scale multipliers 1×–8×. Zero argmax flips at any k tested.

**Constants (§17) match the real artifacts.** `r = 256` is the actual
`dspark_markov_rank`; `B = 16` matches the sidecar's `block_size`; the vocab
projection estimate is close (real `markov_w2` is `[151936, 256]` = 38.9 MB int8
vs the design's 38.4 MB).

**B = 16 is independently justified.** The NPU GEMM is weight-bandwidth-bound and
block time is *independent of activation rows* — measured, 4 cores and 16 cores take
the same time at equal weight bytes. Tokens 9–16 are therefore nearly free on the
GEMM, so there is no reason to shrink the block below its trained size.

## 2. What the measurements correct

### 2.1 §10's cost model is wrong in a way that matters

The design costs candidate-restricted scoring at `rK = 32,768` MACs. The gathered
dot does collapse — but **selection is O(V) and dominates**:

| variant | per slot |
|---|---|
| CPU f32 full-vocab GEMV | 10.59 ms (17.2 GB/s, 156 MB streamed) |
| CPU f32 top-k (k=256) | **1.24 ms** |

That is **9×, not the ~300× the MAC count implies**, and k=16 is no faster than
k=256 because selection is the floor.

**Therefore §8.1's `V_DFlash` is load-bearing, not an optimization.** If the
candidate set arrives from DFlash's top-k rather than being selected CPU-side, the
head reduces to a gather plus a 256-wide dot — microseconds. The design should
promote "DFlash supplies the candidate set" from a bullet to a hard requirement,
because without it the CPU-side head has a ~1.2 ms/slot floor that no amount of
vocabulary sharding removes.

### 2.2 CPU DDTree removes a blocker the design does not know about

hipfire's existing DDTree is GPU tree-attention and depends on
`gated_delta_net_q8_tree_batch_seq` — the **only** tree DeltaNet kernel, Q8-only.
Q8 DeltaNet state was deprecated 2026-07-19 (it caused long-decode attractors, a
stochastic-rounding seed that leaked execution history into target numerics, and a
rollback hazard that broke losslessness). **GPU DDTree is therefore currently
unrunnable.**

Moving tree construction to the CPU sidesteps this entirely. That is a stronger
motivation for §3's device split than the design gives, and it should be recorded
as such.

### 2.3 Branch-state ephemerality must be an explicit invariant

As specified the design is safe: the tree is rebuilt per cycle, so DSpark branch
states are ephemeral. **That property is load-bearing and currently implicit.**

Every DeltaNet-state defect found this session came from per-token recurrent state
surviving a spec-decode rollback — `s_ef_residual` (absent from `DeltaNetSnapshot`)
being the live example, and the serial-tape rollback replay being the one that
actually broke losslessness. If DSpark branch states are ever made persistent
across cycles "for reuse", the same class of bug returns silently.

**Write it down as an invariant: branch state is ephemeral per cycle. Any
cross-cycle reuse requires a rollback story first.**

### 2.4 §13's renormalization is correct — hold the line on it

Trimming changes acceptance rate, not the target distribution, *provided* the
acceptance test uses `q'`. This is exactly the property that was broken and
restored in hipfire this session (`6dcddfcd6`); losslessness is now verified with
four drafters and the AR baseline committing byte-identical tokens while differing
in acceptance. Any tree/vocabulary work must re-run that gate, not assume it.

## 3. What already exists

| piece | state |
|---|---|
| DFlash body on NPU | Phases A–E done; native driver at `crates/hipfire-xdna/examples/dflash_body_native.rs` |
| Multi-core W4A8 NPU GEMM | built + validated, artifacts in `~/.hipfire/npu/r14_*` (not yet wired into the body) |
| DSpark heads | validated vs f32 reference; markov top-k CPU path in `tools/npu/dspark_heads_npu.py` (`backend="topk"`) |
| GPU tree verify | `verify_dflash_block_tree` + `TreeVerifyCtx` in `hipfire-arch-qwen35::speculative` |
| DDTree scaffolding | `ddtree_enabled` / `ddtree_budget` / `ddtree_topk` in `dflash_spec_demo` — GPU-side, currently blocked on Q8 |
| Spec-decode losslessness | restored and gated |
| DFlash calibration artifact | `~/.hipfire/drafts/Qwen3.5-9B.dflash.hessian.calib.hfq` (for LDLQ, if int4 quality needs it) |

**Missing and on the critical path:** an oq4 DFlash sidecar (only OQ8 exists), a
host-side stripe packer for the multi-core GEMM, a multi-core NPU attention kernel,
and a 27B/31B DFlash drafter sidecar.

## 4. Phases and gates

Ordering differs from the source design's §18: **the NPU draft must fit the budget
before any tree work is worth doing**, because the tree only adds to the draft side
of an inequality that currently fails.

### Phase 0 — make the NPU draft fit the budget (PREREQUISITE)

Nothing downstream matters until `T_NPU_draft` is inside the verify window.

1. Wire the multi-core W4A8 GEMM into the body (needs the oq4 sidecar + stripe packer).
2. Multi-core the attention kernel (`dflash_attn_all` loops 8 kv-heads on one core).
3. Attack host glue (143 ms, Rust-side).

**Gate 0.** Measured NPU block wall ≤ the 27B verify budget (155 ms) with margin,
on a real 27B-class drafter. Report cold and warm separately. Parity gate unchanged:
full-body cosine > 0.99 vs the f16 golden AND vs the int4/bf16 precision reference.

### Phase 1 — linear integration, no branching

GPU target + NPU DFlash + CPU DSpark on a single path, fixed top-k candidate set
supplied by DFlash. No DDTree.

**Gate 1.** End-to-end tokens/s beats the GPU-only AR baseline on ≥1 real target,
and committed output is byte-identical to `--ar-baseline` at temperature 0 across
≥2 drafters (the losslessness gate).

### Phase 2 — CPU DDTree, forked DSpark

Clone DSpark state at forks, bounded node budget, pruning, packed GPU tree verify
via the existing `verify_dflash_block_tree`.

**Gate 2.** τ improves over Phase 1 at equal or better tokens/s, losslessness gate
still passes, and branch-state ephemerality holds (assert it, do not assume it).

### Phase 3 — learned vocabulary routing

Hand-built shards → learned router → progressive activation, with DFlash top-k as
the escape hatch.

**Gate 3.** Active vocabulary shrinks materially at unchanged committed output.
Note the measured floor: routing cannot beat ~1.2 ms/slot unless the candidate set
comes from DFlash (§2.1).

### Phase 4 — adaptive scheduling

Dynamic tree width/depth against GPU load; per-request latency-vs-throughput policy.

### Phase 5 — joint optimization

Deferred. Requires training infrastructure and a stable Phase 2/3 baseline.

## 5. Risks

**The draft may not fit even after Phase 0.** ~230–250 ms is the realistic
post-work estimate against a 155 ms (27B) budget. If it lands short, the options
are a 31B-class target (345 ms), a smaller drafter, or accepting partial overlap.
This risk is why Phase 0 is a gate and not an optimization pass.

**NPU weight bandwidth is a hard floor.** The npu1 weight path saturates at
~10 GB/s per routing topology (~13 GB/s across two orthogonal routes), and eight
tuning knobs measured null against it. Weight *bytes* are the only remaining lever
— hence int4 — and no dataflow change recovers more.

**Measurement discipline.** Run-to-run variance on identical binaries was 3.4% at
one point, and the verify forward was outright nondeterministic until `6ca303af8`.
Any claim in this plan's phases needs ≥3 repeats. `./tests/coherence-gate-dflash.sh`
compares single runs and structurally cannot detect that class of bug.

**Sizing note.** DeltaNet state is per-sequence and now FP32: ~150 MB/session at
27B, doubled by the spec-decode rollback snapshot. At high concurrency that is the
memory story, not the drafter. See `NEXT-STEPS.md` for the FP16-state follow-on.

## 6. First concrete step

Phase 0, item 1: produce an oq4 DFlash sidecar and the host-side stripe packer, and
re-measure the block wall with the multi-core GEMM wired in. That converts the
largest term (317 ms) into a measured number rather than a projection, and it is
the single change that most moves `T_NPU_draft` toward the budget.
