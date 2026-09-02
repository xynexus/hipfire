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
