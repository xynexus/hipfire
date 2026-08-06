# Opus improvements — plans

Companion to `opus-quant.md`. Four scoped plans (P1–P4) for the codec changes
that are genuinely not built, plus one grouped exploration track (E1–E5) for the
ideas that need a measurement before they earn a plan.

Sources of truth this document assumes:

| Thing | File |
|---|---|
| Compact mixed encoder | `crates/hipfire-quantize/src/codecs.rs` (`quantize_oqplus_compact`, L1136) |
| Joint scale/mask search | `crates/hipfire-quantize/src/codecs.rs` (`mixed_clipsearch`, L605) |
| Per-tensor outlier budget | `crates/hipfire-quantize/src/main.rs` (`outliers_per_group_for`, L6516) |
| Tensor-tier allocator | `crates/hipfire-quantize/src/mixed_precision.rs` |
| Load-time expansion | `crates/hipfire-runtime/src/oq8_arch.rs` (L167), `oq_moe.rs` |
| NPU sparse residual | `crates/hipfire-xdna/src/sparse3_mp.rs` |
| QAT primitives | `crates/hipfire-train/src/{oqplus_quant,a4_quant,learn_rotation}.rs` |

## Platform premise

halo (gfx1151, UMA) is **bandwidth**-bound at decode and **compute**-bound at
prefill. That asymmetry sets the priority order:

- A4 alone moves prefill, which is why the SpinQuant learned-rotation work read
  as "prefill-only" — that was a compute effect measured on a bandwidth-bound
  box, not a verdict on rotations at decode.
- Decode only moves when **weight bytes** drop. So the payoff path for A4 here
  is W{2,3}A4 — cut the weight side until the activation side is worth having.
- On a compute-bound part (or the NPU, which is neither UMA nor bandwidth-starved
  the same way) W{2,3,4}A4 pays immediately and no weight-bit reduction is needed
  to justify it.

Every plan below is ranked by that: bytes first, then quality-per-byte, then
compute.

---

## P1 — Codebook residual overlay

**Now:** overlay entry is `(u8 index, i8 value)` = 2 B/outlier;
block = `130 + 2·N_out`; `N_out=3` → 136 B → 4.25 b/w.

**The framing correction.** VQ'ing the *value* to save bits is not the win — at
`N_out=3` a 4-bit code saves 12 bits/group = 0.047 b/w, and block-local packing
rounds 134.5 B up to 135 B, so the headline rate goes 4.25 → 4.21875. Bookkeeping
glitter. The index is the expensive half and it is close to irreducible: a sorted
4-of-256 set carries `log2 C(256,4) ≈ 27.38` bits against 32 bits of raw u8
indices, so even an ideal combinatorial code saves ~4.6 bits/block ≈ 0.018 b/w.
The u8 index format looks crude but is already near its entropy floor, and the
FWHT actively suppresses the spatial structure that delta-coding would need.

So the change is worth making only at **fixed bytes, more outliers**:

```
u8 index + u4 code = 1.5 B/outlier
block = 130 + 1.5·N_out
N_out = 4  →  136 B  →  4.25 b/w
```

Identical byte cost, 33% more promoted positions.

**Steps**

**The selector must change too.** Today the promotion is exact in the integer
domain (`q_final = q8`). Under a codebook it becomes `q_final = q4 + C[c]`, so
every promoted position keeps a residual error unless its Δ is represented
exactly. Four approximate corrections beat three exact ones only when

```
Σ_{i∈S4} (e4,i² − eC,i²)  >  Σ_{i∈S3} (e4,i² − e8,i²)
```

which means the selector must score the **post-codebook** reconstruction. Picking
four positions with the existing W8-gain metric and quantizing their deltas
afterwards is not the same experiment and will understate the format.

**Steps**

1. Offline study first, no format work: for a real `down_proj`, evaluate
   `mixed_overlay_error` at (N_out=3, exact i8) vs (N_out=4, i8 snapped to a
   16-centroid Lloyd–Max codebook fit over that tensor's Δ distribution), with the
   N_out=4 set chosen against the codebook reconstruction per the inequality above.
   If the 4-way-VQ error is not below the 3-way-exact error, **stop here** — the
   idea is dead and cost so far is one example binary.
   Note P2's measurement (below) is a precondition for this study being meaningful:
   under the old selector, extra outliers bought almost nothing past N_out=7, so a
   3-vs-4 comparison run before P2 would have been scored through a selector that
   could not reward the extra correction.
2. If it wins: codebook is per tensor, 16 × i8 = 16 B, stored as a sidecar tensor
   (`<name>.oqvq`), same mechanism the AWQ scale sidecars use.
3. New `QuantType` in `hipfire-quant-format/src/lib.rs` (do **not** overload
   qt=36 — block length is already load-bearing as the N_out oracle). Number
   above the origin-owned range; see the recoding rule in
   `hipfire_quant_format::storage`.
4. Encoder: fork `quantize_oqplus_compact`. Decoder: fork
   `oqplus_compact_to_oq8_combined` — it resolves code→i8 through the codebook and
   emits the *same* dense int8 blocks, so **no kernel, dispatch, or arch change
   at all**.
5. Gate: golden round-trip in `codecs.rs` tests + one KLD run on Qwen3.5-0.8B
   wikitext2 at matched bytes vs qt=36.

**Cost:** encoder + decoder + one qt. No runtime path touched.
**Kill criterion:** step 1 fails, or KLD gain < 3% at matched bytes.

**Better endgame than an arbitrary codebook.** If step 1 clears, prefer a
*structured* 4-bit code — one or two additional signed bitplanes on top of the W4
value — over 16 free centroids. It decodes with shifts instead of a lookup, QAT
can optimise against the exact hardware representation, and it nests into the
existing 2→8-bit progressive Opus ladder instead of bolting a per-tensor table
onto it. That also merges this item with E3 (soft promotion), which is the same
idea approached from the other end.

---

## P2 — Joint scale + mask — **DONE**

**Now:** `mixed_clipsearch` seeds from `symmetric_clipsearch(group, 7.0)` — an
int4-only scale that pretends no position escapes the ±7 clamp — then alternates
twice: mask-at-fixed-scale (`mixed_overlay_indices`), scale-at-fixed-mask
(`refit_mixed_scale`). That is coordinate descent from a biased seed.

**Key observation:** the joint objective is *separable across positions*, and for
any fixed scale the optimal mask is exactly "top-N_out by upgrade gain" — which
`mixed_overlay_indices` already computes. So sweeping the scale with the mask
recomputed inside the loop is not a better heuristic, it is the **exact** joint
minimum over the scale grid:

```
for s in clipsearch_grid(group):
    S    = mixed_overlay_indices(group, s, n_out)
    err  = mixed_overlay_error(group, s, &S, n_out)   // already exists
    keep argmin
```

**Steps**

1. Hoist the candidate grid out of `symmetric_clipsearch` so both callers share it.
2. Rewrite `mixed_clipsearch` as the sweep above (~20 lines, deletes
   `refit_mixed_scale`'s only caller).
3. Extend the existing test `mixed_clipsearch_never_worsens_q4_seeded_overlay_error`
   (`codecs.rs:2881`) to also assert joint ≤ alternating on the same fixtures.
4. Re-run the qt=36 golden battery — **bytes and quant type are unchanged**, so the
   only expected diff is the artifact hash. Update expected hashes in one commit.

**Cost:** one function. Encode time rises by the grid factor; encoding is offline.
Zero runtime change, zero format change, pure quality.
**Kill criterion:** none — the monotonicity test makes it non-regressive by
construction. If KLD does not move, it still costs nothing to keep.

### What shipped

`codecs::mixed_clipsearch` is now the exact grid argmin and is `pub` — the ONE
selector for every mixed Opus packer. It had been duplicated four ways, each at a
different quality tier, which is why this was a root-cause fix rather than a
one-function edit:

| Site | Was | Now |
|---|---|---|
| `codecs::quantize_oqplus_compact` (qt=36) | 2-round alternating | shared joint |
| `codecs::quantize_oqplus_tiered` (int8-stored) | **single-shot**, no refit | shared joint |
| `ldlq::oqplus_tiered_ldlq_pack` | **single-shot**, no refit | shared joint |
| `ldlq::oqplus_compact_ldlq_pack` (the `++` flagship) | **single-shot**, no refit | shared joint |
| `dflash_convert` `*_plain` local copies (4 fns) | 2-round alternating | deleted, routed to shared |

`ldlq::oq4_ldlq_pack` deliberately keeps `symmetric_clipsearch` — it is plain oq4
with no overlay, so the int4-only search is correct there.

Two incidental fixes fell out of the convergence: the `oq4.25++` LDLQ path had
**no scale refit at all** (it was the weakest of the four, not the strongest), and
`dflash_convert`'s gain sort lacked the index tiebreak the others had, so equal
gains ordered non-deterministically under `sort_unstable`.

### Measured (Qwen3.5-0.8B, 6 × {down,gate,o}_proj, 3072 real G256 groups)

Group SSE after FWHT, alternating vs joint:

| N_out | b/w | alternating | joint | reduction | groups whose scale moved |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.125 | 0.9441 | 0.9438 | 0.03% | 1.0% |
| 3 | 4.25 | 0.7895 | 0.7845 | 0.62% | 6.7% |
| 7 | 4.5 | 0.7060 | 0.6334 | **10.3%** | 47.4% |
| 15 | 5.0 | 0.6973 | 0.4827 | **30.8%** | 97.2% |
| 31 | 6.0 | 0.6852 | 0.3332 | **51.4%** | 100% |

**The important column is the alternating one, read downward: 0.7060 → 0.6973 →
0.6852.** Under the old selector, going from 7 to 31 outliers per group — 4× the
overlay bytes — bought 3% error. Promotion was being wasted: the int4-seeded scale
was pinned, so promoting more positions never let the bulk scale shrink, which is
the entire mechanism by which promotion is supposed to pay. Under the joint
selector the same bytes keep paying (0.6334 → 0.4827 → 0.3332).

Consequences worth carrying forward:

- At the shipped `oq4.25++` default (N_out=3) this is a **0.6%** SSE change. Do not
  expect a visible KLD move there; it is real but small.
- The gain is concentrated exactly where the per-layer budget already sends bytes —
  `HIPFIRE_OUTLIERS_BY_LAYER=down_proj:7,...` sits at the knee, and above it the old
  selector was flat.
- **Any prior experiment that concluded "more corrections don't help" was measured
  through a selector that could not reward them.** That verdict should be treated
  as void for P1, P4 and E3, all of which are "spend more on corrections" ideas.

Still open: a KLD run to convert SSE into end-to-end quality, and re-running the
per-layer outlier sweep now that the knee has moved.

---

## P3 — Bitplane / SoA overlay packing

**Reality check that shrinks this.** The extract framed this as fixing
compact-vs-expanded kernel divergence. There is no compact-resident GPU kernel in
this tree — `kernels/src/` has none that reads a qt-36 block, and every GPU route
expands at load. So bitplane packing buys **nothing on GPU** while expand-at-load
stands, and this is not a kernel project.

That absence is **deliberate, not debt**: OpCompact is unimplemented on GPU
because the Opus format has not stabilised, and stabilising it is exactly what
P1/P2/P4 are for. Writing a compact-resident kernel against a layout that P1 is
about to change would be building the expensive half first. Expand-at-load is the
correct holding position until the format settles.

Note P1's index-entropy result also guts the on-disk half of the original
rationale: sorted indices are only ~4.6 bits/block from raw u8, so delta-coding
the index plane is nearly worthless. What survives is (a) the u4 code plane in P1
cannot be interleaved with u8 indices without wasting the nibble, and (b) planar
layout is what a resident kernel — GPU or AIE2P — wants to read.

Where it does pay:

- **Prerequisite for P1.** A u4 code plane cannot be interleaved with u8 indices
  without wasting the nibble. This is now the main reason to do it.
- **NPU.** `sparse3_mp.rs` is the one consumer that executes the overlay sparsely;
  planar indices are what a resident AIE2P kernel wants.
- **On-disk, weakly.** The value plane becomes a narrow distribution the entropy
  coder can use; the index plane, per above, is already near its floor.

**Steps**

1. Change the encoder tail from `[(idx,val)×N]` to `[idx×N][val×N]`, emit indices
   ascending (they come out of a sort already — just don't re-shuffle).
2. Mirror in `oqplus_compact_to_oq8_combined` and
   `oqplus_compact_to_moe_oq8_blocks`. Same block length, so **N_out inference from
   block length is untouched**.
3. New qt (byte layout changed under a fixed length — a silent reinterpretation of
   existing artifacts is the one failure mode that produces plausible garbage).
4. Measure on-disk delta through the existing artifact coder before/after.

**Cost:** bundle it into P1's commit — one qt, both changes. Doing it standalone
is not worth its own format number.
**Kill criterion:** if P1 dies at its step 1 and no compact NPU consumer is queued,
skip — the on-disk gain alone does not justify a format number.

---

## P4 — Cross-group promotion budget

**Now:** N_out is constant per tensor. `HIPFIRE_OUTLIERS_BY_LAYER` already varies
it per tensor by name suffix (`down_proj:7,o_proj:3,default:1`) and
`oq_floor_bpw_for` charges the true rate to the tensor allocator. **The tensor-level
version of this idea is already shipped.** What is missing is per-*group*
allocation inside a tensor.

**Why it is the most expensive of the four.** The container infers N_out from
block length. Per-group variation therefore requires a format change, and the
obvious cheap dodges do not work:

| Option | Verdict |
|---|---|
| Per-group count byte, pad to N_max | **Dead.** You pay N_max per group regardless; padding buys nothing over just setting N_out = N_max. |
| Per-row N_out + row header | Works, but rows within a tensor are far more alike than groups within a row — most of the claimed gain is not there. |
| Variable-length blocks + u32 offset table | The only option that realizes the gain. Costs O(1) block indexing, which the weight pager and every load-time expander rely on. |

So the plan is **measure before touching the format**:

1. Add `crates/hipfire-quantize/examples/opus_group_budget_study.rs`: for a few
   real tensors, compute each group's marginal `mixed_overlay_error` reduction per
   added outlier, then compare total error under (a) uniform N_out, (b) water-filled
   allocation at the same *total* outlier count.
2. Decision gate on (b)/(a): **< 5% error reduction → close the item**, record the
   negative result in `opus-quant.md`, done. The study is ~100 lines and reuses
   `mixed_overlay_error` verbatim.
3. Only if it clears 5%: variable-length blocks + a `[u32; n_groups]` offset prefix,
   new qt, and audit every `block_bytes`-multiplied offset computation — the weight
   pager is the risk, not the codec.
4. Interim consolation prize if the study is borderline: extend
   `outliers_per_group_for` matching from name-suffix to a **per-tensor value
   emitted by the study**, folded into the artifact recipe instead of an env var.
   Same container, finer allocation, no format work.

**Cost:** step 1 is cheap and answers the question. Step 3 is the only genuinely
expensive item in this document.
**Kill criterion:** the 5% gate at step 2.

---

## Grouped exploration track (E1–E5)

These share one harness, so they are one track, not five projects. The harness is
the reason to group them: each is a matched-bytes KLD/PPL comparison on
Qwen3.5-0.8B + wikitext2 with fp32 KV, which is what
`hipfire-train/examples/wa_matrix_sweep.rs` and the `hipfire-eval` batteries
already do. **Step 0 of this track is to make that harness take a codec variant
as a parameter**, so E1–E5 are each a flag rather than a fork.

Run in this order — each result kills or feeds the next.

### E1 — Saliency-weighted upgrade gain

The extract proposes `G_i ∝ σ²_x(i)·(e4² − e8²)`. We have already measured that
family: plain XᵀX-weighted LDLQ ≈ no-calib on held-out data, and GuidedQuant
(end-loss gradients) was the only robust winner. So do **not** build the σ² version.
Build the GuidedQuant-weighted one: `calib_guided.rs` already produces the
saliency, and the gain function is a one-line reweight inside
`mixed_overlay_indices`. Highest expected value in this track because the input
already exists.

### E2 — Intra-block low-rank residual

`HIPFIRE_LOWRANK_R` already does low-rank residual correction at tensor level
(−13% at 2b). The extract's version is the same idea inside a 256-group.
**Predicted dead:** the FWHT exists precisely to decorrelate within the group, so
residual rank structure inside 256 should be flat. One cheap probe: SVD the
per-group Δ matrix for one tensor and read the spectrum. Flat spectrum → close it,
and record why, since this idea recurs.

### E3 — Soft promotion (+1 bitplane instead of full int8)

Promote to W5/W6 rather than W8. Given the W3A4 result (weight bytes are the
decode lever) this is the exploration item most aligned with the platform premise:
a W3 base with a W5 overlay is a smaller artifact than W4+W8 at comparable error.
Needs `opus_lowbit.rs` to grow the W5 grid and a W3 decode GEMV, which is already
an open item. Sequence after P2 so the scale search is honest at W3.

### E4 — Compact-resident GPU kernel

Deferred, deliberately. Today's GPU route is expand-at-load, which is *correct and
fused* — there is no divergence to fix. A compact-resident kernel is only worth
writing if P1/P4 make compact blocks meaningfully smaller than the expanded int8
they become, and if host-RAM pressure (not bandwidth) is the binding constraint.
Revisit after P1 lands; until then the NPU `sparse3_mp.rs` path is the compact
executor and is the better place to prove the scheme.

### E5 — QAT / training-directed quantization

The scaffolding exists: real fp32 GPU autograd, `oqplus_quant.rs` (STE simquant
for oq3/oq4/oq8), `a4_quant.rs`, `learn_rotation.rs`, `qtip_quant.rs`. So this is
a recipe question, not a build question. Order by measured leverage, stopping at
the first that fails to move held-out KLD:

1. **Learned per-group scale** — replaces clip-search with a parameter. Historically
   the largest single QAT win at W4 and the cheapest to wire.
2. **A4 activation clipping penalty** — makes activations *want* to fit the int4
   range, using `a4_quant.rs` in the forward.
3. **W3 base + learned overlay** — the actual platform goal, per the premise
   section: this is what makes A4 pay off at decode on halo.
4. **Learned promotion mask** — relaxed with Gumbel/top-k. Last because it is the
   most fragile and its gain is bounded by whatever P2/E1 already extracted from
   the offline selector.

Prior constraint to respect: light-QAT recovery found W3 loss ~52% recoverable but
KVarN-4 loss non-recoverable — do **not** bundle KV bit reduction into this track,
it will contaminate the attribution.

---

## Suggested order

1. ~~**P2**~~ — **done.** Free, exact, non-regressive, and it turned out to be the
   precondition for measuring the rest: the old selector flattened the payoff of
   every "spend more on corrections" idea past N_out=7.
2. **P4 step 1** — a ~100-line study that either kills the most expensive item here
   or justifies it. Promoted above E1 because P2 moved the knee it measures.
3. **E1** — reuses a saliency signal we already compute.
4. **P1 + P3** as one commit, gated on P1's step-1 study, and preferring the
   structured-bitplane code over a free codebook.
5. **E3 / E5** — the platform-premise work; largest payoff, largest cost.
6. **E2, E4** — probe-to-kill and deferred-until-the-format-stabilises respectively.

Re-run the per-layer outlier sweep (`HIPFIRE_OUTLIERS_BY_LAYER`) before 2–4: its
current values were tuned against the pre-P2 selector.
