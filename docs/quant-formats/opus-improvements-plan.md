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

## P1 — Codebook residual overlay — **step 1 CLEARS, and it deletes the codebook**

**Outcome.** `examples/opus_codebook_residual_study.rs` (Qwen3.5-0.8B, 2048
groups × {down,o,gate,q}_proj, both arms at 136 B/group):

| arm | down | o | gate | q |
|---|---|---|---|---|
| 16-entry Lloyd codebook | +6.12% | +6.31% | +6.25% | +5.41% |
| **raw signed Δ in [-8,7]** | **+6.30%** | **+6.36%** | **+6.30%** | **+6.39%** |

Step 1's gate is cleared: 4 approximate corrections beat 3 exact ones by ~6%
weight SSE, with the 4-set selected against the post-codebook reconstruction as
the plan required.

**But the codebook loses to a raw 4-bit delta on every tensor**, so steps 2–4's
codebook machinery is dead weight. The reason is in the data: the Δ pool holds
only 17–19 distinct values, because Δ = q8 − q4 is bounded by the ±7 clamp and
the FWHT leaves few positions far outside it. A raw i4 covers that core exactly
and clips a rare tail; Lloyd instead spends centroids on the frequent small
deltas and mis-serves the tail (hence q_proj's 5.41%, its pool being the widest
at 19 values and only 91.2% exactly hit).

The raw arm also matches **N=4-exact** to four decimals (e.g. down_proj 0.264185
vs 0.264178), so 4 bits costs essentially nothing against a full i8 here.

**Revised plan.** Drop step 2 (the `<name>.oqvq` sidecar) and the codebook half
of step 4. What remains is one new qt whose overlay entry is `(u8 index, i4
delta)` — 4 entries in the same 6 B the current 3 entries use — plus step 5's
golden round-trip and KLD gate. This *is* the "structured 4-bit code" endgame the
section below prefers, arrived at by measurement rather than taste, and it needs
no per-tensor state at all.

Still ungated: the 3% KLD criterion at matched bytes. Weight SSE has disagreed
with KLD in this exact experiment before (d77fa637a), so the format work should
wait behind that run.

Original plan below.

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

Still open: a KLD run to convert SSE into end-to-end quality.

### Per-layer outlier sweep, re-run (CPU half)

`examples/opus_outlier_budget_study.rs` re-scores commit d77fa637a's "uniform
wins" verdict against the joint selector, on real Qwen3.5-0.8B weights (2048
G256 groups per layer type). Mean per-group SSE:

| layer | share | OLD N=1→31 | NEW N=1→31 |
|---|---:|---|---|
| q_proj | 9.0% | .00047 → .00034 (flat by N=5) | .00047 → .00016 |
| k_proj | 1.1% | .00035 → .00025 | .00035 → .00012 |
| v_proj | 1.1% | .00054 → .00039 | .00054 → .00019 |
| o_proj | 4.5% | .00026 → .00019 | .00026 → .00009 |
| gate_proj | 28.1% | .00047 → .00034 | .00047 → .00017 |
| up_proj | 28.1% | .00018 → .00013 | .00018 → .00006 |
| down_proj | 28.1% | .00019 → .00014 | .00019 → .00007 |

The old selector saturates by N=5 in **every** layer type — the P2 finding
reproduced per-layer. The new one keeps paying out to N=31.

**The original hypothesis was backwards.** d77fa637a spent the budget on
`down_proj` because it is 28% of parameters and consumes SwiGLU output. But the
allocation optimum equalises the *per-group* marginal, and the group count
cancels out of the Lagrange condition entirely — parameter share does not enter
the ranking, only the budget arithmetic. Per group, `down_proj` and `up_proj`
have the LOWEST marginal value of the seven; `v_proj`, `q_proj` and `gate_proj`
have the highest. So `down=7 rest=1` was pointed the wrong way and would lose
under the new selector too.

Greedy water-fill at a matched 4.25 b/w gives:

```
HIPFIRE_OUTLIERS_BY_LAYER=q_proj:5,k_proj:6,v_proj:9,o_proj:3,gate_proj:5,up_proj:1,down_proj:2
```

worth **2.39%** param-weighted SSE over uniform N=3. That is small, and small in
a metric already known to disagree with KLD here. **Verdict: "uniform wins"
survives P2 for the allocation question** — do not re-open it without a reason
better than 2.4% SSE.

What P2 *does* reopen is the neighbouring claim in the same commit: that "N=3 is
a genuine optimum in both directions", evidenced by oq4.5++ (N=7 uniform)
scoring worse KLD than N=3 (0.037291 vs 0.036631) **while spending more bits**.
More bits buying worse quality is not a property of the format; it is the
signature of a selector that could not use them. Under the joint selector N=7
now scores 10.3% better group SSE than N=3 and keeps improving. That comparison
is the experiment worth GPU time, and it is a different one from the sweep this
section re-ran.

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

## P4 — Cross-group promotion budget — **CLOSED (negative)**

**Outcome.** Step 1 ran (`examples/opus_group_budget_study.rs`, Qwen3.5-0.8B,
2048 groups × {down,o,gate,q}_proj) and step 2's 5% gate rejects it — not
narrowly, and not for want of a better allocator.

| arm, at equal bytes | down | o | gate | q |
|---|---|---|---|---|
| free offsets (3 slots/group) | +7.2% | +5.9% | +9.0% | +11.1% |
| u32 offsets (1 slot/group) | −9.0% | −11.9% | −9.3% | −6.9% |

The allocation freedom is real — it is the *address* that is not affordable. A
`[u32; n_groups]` prefix costs 4 B against a 136 B block, which is 2 of the 3
overlay slots; the per-group arm then loses to uniform `N_out=3` outright. The
negative row is a Lagrangian bound, so it holds against any allocator, not just
the study's greedy (the SSE curves are non-convex — the joint selector re-fits
the group scale at each N — so greedy alone could not have closed this).

Break-even is between 2 and 3 bytes of addressing. 2 B caps a tensor at 65535
blocks and 1 B cannot address a block at all, so there is no cheaper index that
both works and wins. Step 3 (variable-length blocks) is therefore not worth
starting, and step 4's consolation prize is already shipped as
`HIPFIRE_OUTLIERS_BY_LAYER`. Recorded in `opus-quant.md` §7.

Original plan below, kept for the reasoning.

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

### E1 — Saliency-weighted upgrade gain — **CLOSED (no-op, provable)**

**Do not build this.** Not because the saliency is wrong, but because the
selector it would reweight cannot see it.

`mixed_overlay_indices` ranks positions INSIDE a group that
`quantize_oqplus_compact` has already passed through `cpu_fwht_256`. The signed
Hadamard has entries of one magnitude (`1/√256`), so for ANY per-input-channel
saliency `s`, the rotated weighting's diagonal is

```
[R·diag(s)·Rᵀ]ᵢᵢ  =  Σⱼ Rᵢⱼ²·sⱼ  =  (1/256)·Σⱼ sⱼ  =  mean(s)
```

— the same number at every position. A per-position reweight of the gain
therefore multiplies every candidate by one constant and cannot reorder them.
Exact, not approximate: `hipfire-primitives` test
`rotation_flattens_any_per_channel_saliency` measures the spread at 0 over the
real transform. And 97% of the mass of `R·diag(s)·Rᵀ` is off-diagonal, so a
diagonal reweight could not capture the true weighted error even if it varied.

The two ways saliency CAN reach this codec are both already shipped: act before
the rotation (per-channel scaling — the `+` AWQ/SmoothQuant pass) or account for
the off-diagonal coupling (`++` LDLQ/OBS error feedback). "Highest expected value
in this track because the input already exists" was wrong for the same reason
the FWHT is worth having: it spreads structure, including the structure a
saliency reweight would need.

Original text kept below.

~~The extract proposes `G_i ∝ σ²_x(i)·(e4² − e8²)`. We have already measured that
family: plain XᵀX-weighted LDLQ ≈ no-calib on held-out data, and GuidedQuant
(end-loss gradients) was the only robust winner. So do **not** build the σ² version.
Build the GuidedQuant-weighted one: `calib_guided.rs` already produces the
saliency, and the gain function is a one-line reweight inside
`mixed_overlay_indices`. Highest expected value in this track because the input
already exists.~~

### E2 — Intra-block low-rank residual — **CLOSED (negative)**

**Probe ran** (`examples/opus_intrablock_rank_study.rs`, Qwen3.5-0.8B, 2048
groups × 4 tensors). The prediction was "flat spectrum → close it". The spectrum
is **near-flat but not flat**, so the prediction's stated reason is wrong and the
item still closes — on economics, not on absence of structure. Record both,
since this is exactly the idea that keeps coming back.

Cumulative energy share of the leading `r` directions of `ΔᵀΔ`, post-FWHT
(white-noise floor is `r/256`):

| tensor | r=1 | r=4 | r=16 | r=32 |
|---|---|---|---|---|
| floor | .0039 | .0156 | .0625 | .1250 |
| down_proj | .0484 | .0810 | .1685 | .2628 |
| o_proj | .0085 | .0325 | .1167 | .2169 |
| gate_proj | .0088 | .0333 | .1226 | .2278 |
| q_proj | .0109 | .0406 | .1382 | .2479 |

down_proj carries 12× the floor at r=1. The FWHT does tighten the spectrum
against a no-rotation control on 3 of 4 tensors, but it does not flatten it.

**Why it closes anyway.** Cheapest sane form is one 256-dim basis shared across
a tensor's groups with `r` f16 coefficients per group — 2r B on a 136 B block.
Best case (perfect projection, no coefficient quantization):

| rank | cost/group | best case | %SSE per byte |
|---|---|---|---|
| r=1 | 2 B | 4.84% | 2.42 |
| r=4 | 8 B | 8.10% | 1.01 |
| r=32 | 64 B | 26.28% | 0.41 |

P1 step 1 gets **+6.3% for zero bytes**, which dominates the r=1 row outright,
and the per-byte column only worsens with rank — the curve never turns
favourable, it just buys more by spending more. E2 is also the only candidate
here that needs a new decode path: the overlay resolves inside the existing
expander, while a shared basis adds a 256-wide matmul per group to every load.

### E2 — Intra-block low-rank residual (original text)

`HIPFIRE_LOWRANK_R` already does low-rank residual correction at tensor level
(−13% at 2b). The extract's version is the same idea inside a 256-group.
**Predicted dead:** the FWHT exists precisely to decorrelate within the group, so
residual rank structure inside 256 should be flat. One cheap probe: SVD the
per-group Δ matrix for one tensor and read the spectrum. Flat spectrum → close it,
and record why, since this idea recurs.

### E3 — Soft promotion — **CLOSED. The soft half is free; the low-base half is dead.**

**Gate ran** (`examples/opus_soft_promotion_study.rs`, Qwen3.5-0.8B, 2048 groups
× 4 tensors, every arm at the shipped 136 B/group). E3 bundles two separable
claims and they land in opposite directions.

SSE vs the shipped W4+W8 arm at equal bytes (negative = worse):

| arm | n_out | down | o | gate | q |
|---|---|---|---|---|---|
| W4+W8 (shipped) | 3 | — | — | — | — |
| W4+W6 | 3 | 0.00% | 0.00% | 0.00% | 0.00% |
| W4+W5 | 3 | −0.00% | −0.00% | −0.00% | −0.00% |
| W3+W8 | 19 | −142.7% | −142.8% | −141.7% | −142.9% |
| W3+W5 | 23 | −122.0% | −121.8% | −120.8% | −121.8% |
| W3+W4 | 25 | −115.5% | −114.5% | −113.4% | −114.6% |
| W2+W5 | 43 | −1055.6% | −1030.8% | −1032.9% | −1047.4% |

**The narrow overlay is free.** W6 and W5 promotions score identically to W8 —
the promoted values simply do not need 8 bits. That corroborates P1 from a
different direction (there the *delta* pool held 17–19 distinct values), and it
is why P1's raw-i4 entry works: narrowing the entry costs nothing in error and
buys a 4th promoted position.

**The low base is dead.** "A W3 base with a W5 overlay is a smaller artifact
than W4+W8 at comparable error" is false at equal bytes — it is not comparable
error. Dropping the base to W3 frees 32 B and buys 16–22 extra promotions, and
still loses by 114–143%; W2 loses by ~1000%. The arithmetic is unsurprising in
hindsight: a bit of base applies to all 256 positions while a promotion applies
to one, so 22 extra promotions cannot pay for 256 positions losing a bit. No
W3 decode GEMV or W5 grid should be built for this reason.

Caveat kept: weight SSE is a proxy, and a lower base also changes the activation
story (W3A4 needed a learned rotation — `opus-quant.md` §7). The study bounds
the weight side only, but the margin is far too large for the proxy to be the
explanation.

### E3 — Soft promotion (original text)

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
2. ~~**P4 step 1**~~ — **done, and it killed the item.** The offset prefix costs
   more than per-group allocation is worth; see P4 above. The most expensive
   item in this document is off the table.
3. ~~**E1**~~ — **closed, no-op.** The overlay selects in the rotated domain,
   where any per-channel saliency is provably constant; see E1 above. Note this
   also constrains E3 and E5: nothing that ranks positions *within* a rotated
   group can use a per-input-channel signal.
4. **P1 + P3** as one commit — P1's step-1 study is **done and clears**, and it
   settled the "structured code vs free codebook" question empirically: raw i4
   Δ wins outright, so there is no codebook and no sidecar to build. Next
   action here is the KLD run at matched bytes, not encoder work.
5. ~~**E3**~~ — **closed.** Its soft-overlay half is free and already exploited by
   P1; its low-base half loses by 114–143% at equal bytes. **E5** — the
   platform-premise work; largest payoff, largest cost, and now the only
   exploration item left standing.
6. ~~**E2**~~ — **probed and killed**; see E2 above. **E4** — still deferred until
   the format stabilises.

Re-run the per-layer outlier sweep (`HIPFIRE_OUTLIERS_BY_LAYER`) before 2–4: its
current values were tuned against the pre-P2 selector.
