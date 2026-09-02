# `lr_u`/`lr_v` low-rank residual sidecars are written by the quantizer and read by one arch

Status: **OPEN**, found 2026-09-03 while scoping a trained low-rank correction.

## The result

`HIPFIRE_LOWRANK_R=<r>` makes the quantizer emit a rank-`r` factorisation of each
tensor's quantization error as `<base>.lr_u.weight` / `<base>.lr_v.weight` f32
sidecars (`hipfire-quantize/src/cli.rs`, ~line 5991). Grepping every consumer:

    crates/hipfire-quantize/src/cli.rs          # the producer
    crates/hipfire-arch-minimax/src/minimax.rs  # the only reader
    crates/hipfire-train/src/qtip_quant.rs      # a doc comment

`hipfire-arch-qwen35` does not mention them. Neither does any other arch. So on
every family except MiniMax the flag inflates the artifact and changes nothing.

## The measurement that shows it

From this session's run log, same body and embed, `HIPFIRE_LOWRANK_R` the only
difference:

| build | size | bpw | evalA kld |
|---|---|---|---|
| C1 `qtip3+greedyOBS+hfq4` | 344.5 MB | 3.66 | 0.246885 |
| C2 `qtip3+greedyOBS+lr32+hfq4` | 428.8 MB | 4.56 | 0.246885 |

**+84.3 MB, +0.90 bpw, and the KLD is identical to six decimal places.** Not
"close" — identical, which is what a tensor that is loaded and then never read
looks like. A 24% larger artifact for no effect.

## Why it matters beyond the wasted bytes

The low-rank residual is recorded elsewhere in the tree as a real lever
(`-13%` KLD at 2-bit). That evidence came from families where it is wired. Anyone
reading that and reaching for `HIPFIRE_LOWRANK_R` on qwen3.5 — as this session
nearly did, as the deployment vehicle for a trained LoRA-style correction — gets
a silent no-op and a bigger file.

It also means **a trained low-rank correction has no serving path on qwen3.5
today.** Training one first would be building against a consumer that does not
exist.

## What the sidecars are actually worth here (measured 2026-09-03)

Before wiring anything, the ceiling was measured offline: dequantize the qtip3
weights, add `lr_u @ lr_v`, and re-measure block-local MSE against the bf16
teacher (`qwen35_norm_recovery` with `HIPFIRE_RECOVER_LR_FOLD=1`, 0 steps).

**The factors are in the FWHT-ROTATED frame.** The quantizer factorises
`E_rot = W_rot - dequant(Q(W_rot))`, so a consumer holding an un-rotated weight
must rotate the correction back before adding it. Getting this wrong is not
subtle but it is silent — adding the correction raw makes the weight *worse*
(cos 0.9866 -> 0.9860, block MSE up ~2.5x). Any dense implementation must get
this right, and it is the first thing to test.

With the basis correct, rank 32 on `Qwen3.5-0.8B-e4--qtip3g`:

| block | no lr | + lr32 | Δ |
|---|---|---|---|
| L1 attn | 9.018e-3 | 8.549e-3 | −5.2% |
| L2 attn | 1.074e-2 | 1.015e-2 | −5.5% |
| L3 attn | 1.150e-2 | 1.067e-2 | −7.2% |
| L2 mlp | 3.652e-5 | 3.470e-5 | −5.0% |
| weight cos (gate_proj) | 0.9877 | 0.9884 | — |

A real ~5% block-MSE reduction, comparable to everything the norm-recovery work
bought — and it is **untrained**, just an SVD of the quantization error.

## But the bits are better spent elsewhere on this model

Rank 32 costs **+81.3 MB, +0.90 bpw** (3.78 -> 4.68). Compare what the same bits
buy in the base format: 3.78 bpw qtip3 (kld 0.189) -> 4.41 bpw `oq4.25++` with a
4-bit embed (kld 0.075) is **+0.63 bpw for a 60% KLD reduction**. A ~5%
block-MSE reduction for +0.90 bpw is not close to competitive.

This is consistent with where the low-rank residual is recorded as a lever: 2-bit,
where the quantization error is large enough that a rank-32 correction is cheap
relative to what it removes. At ~3.8 bpw it is not.

**Recommendation: do not wire dense `lr_u`/`lr_v` for qwen3.5 to chase quality at
4-bit.** Wire it if and when the 2-bit arm becomes live, where the trade inverts.
The interim fix below stands regardless — writing 81 MB nobody reads is a bug
whatever the format is worth.

## Shape of the fix

For a dense linear the correction is `y += (x @ lr_vᵀ) @ lr_uᵀ` — two rank-`r`
GEMMs after the main one. MiniMax is not a clean reference: its implementation is
MoE-expert-specific (per-expert factors packed into blobs with batched pointer
tables). A dense path wants its own small kernel plus an arm in each call site,
which on qwen3.5 means the same eight branch chains that
`docs/todo/2026-09-02-prefill-lowered-dispatch-table.md` is about — more evidence
for making that selector a table that rejects unknown combinations rather than a
chain that silently falls through.

Minimum honest interim fix: **refuse or warn.** If an arch cannot consume the
sidecars, `HIPFIRE_LOWRANK_R` should say so at quantize time instead of writing
84 MB nobody will read.
