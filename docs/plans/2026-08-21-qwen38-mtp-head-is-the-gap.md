# Qwen3.8-27B spec-decode: we are dropping the model's MTP head

**Status:** root cause identified, not yet implemented. This supersedes the
"Phase 2 is structurally dead" conclusion in
`2026-08-21-qwen38-27b-peak-performance-goal.md`.

## The external datapoint

`github.com/julianmb/q38rocm` benchmarks **the same model on the same silicon**
(Ryzen AI Max+ 395 / 40 CU 8060S / 128 GB LPDDR5X) at a near-identical bit
budget. Only its measured numbers are used here; its prose and supposition are
not relied on.

| | theirs (ROCmFP4_FAST, 4.26 bpw) | ours (oq4.25++, 4.25 bpw) |
|---|---|---|
| plain decode | 14.02 tok/s | **15.1 tok/s** |
| stock Q4_K_M baseline | 12.27 tok/s | — |
| spec-decode | **36.04 tok/s** | 5.06 tok/s |
| acceptance | 71.4% math / 82.6% code / 88.0% JSON | ~36-43% |
| 32K ctx | prefill 245, decode 13.62, MTP 26.85 | — |

Two readings, and they point in opposite directions:

1. **Our plain decode is FASTER at the same bit budget** — 15.1 vs 14.02
   (+7.7%), and both are well above the 12.27 Q4_K_M baseline. Opus W8A8 (4-8 bit
   weights, int8 activations) beating ROCmFP4's W4A16 is consistent: A16 forfeits
   the 2x int8 WMMA rate, and at these shapes activations are not the byte cost.
2. **Their spec-decode is 7x ours**, and the acceptance column says why. They run
   **MTP** — the model's own multi-token-prediction head — at 71-88% acceptance.
   We run an external DFlash2 drafter at ~36-43%.

## What we are throwing away

`Qwen/Qwen3.8-27B` ships a complete MTP head. From the source index:

    mtp.fc.weight                             [5120, 10240]    52.43M
    mtp.layers.0.self_attn.{q,k,v,o}_proj                     104.85M
    mtp.layers.0.mlp.{gate,up,down}_proj                      267.39M
    mtp.layers.0.{input,post_attention}_layernorm
    mtp.layers.0.self_attn.{q,k}_norm
    mtp.norm.weight
    mtp.pre_fc_norm_embedding.weight
    mtp.pre_fc_norm_hidden.weight
                                              MTP total      424.70M

And our artifact's own header records the loss:

    source_total_params  27,781,427,952
    total_params         26,895,998,464
    skipped_params          885,429,488      <- MTP (424.70M) + vision tower

`grep -c mtp` on the .hfq header is **0**. The head is dropped at conversion.

## Why this is the right lever

The MTP head is 424.70M params — **1.6% of the 27B target**, ~226 MB at 4.25
bits, against the DFlash2 drafter's 1.2 GB. So it is both far cheaper per draft
step AND far more accurate, because it was trained jointly with the model it
drafts for.

Roofline for one cycle at n=7 draft steps:

    MTP draft   7 x 226 MB  =  1.58 GB
    verify      1 sweep     = 14.30 GB
    total                     15.88 GB   vs plain decode's 14.30 GB/token

At ~80% acceptance (tau ~5.5) that is 5.5 tokens for 1.11x the bytes of ONE
plain-decode token. Their measured 2.57x over their own base, applied to our
15.1, projects **~38 tok/s**.

## The runtime is already there; the CONVERTER is not

hipfire has the qwen35 MTP stack: `mtp_head.rs` (`Qwen35MtpHead`,
`load_mtp_head`, `load_mtp_head_bundled`, `load_mtp_head_at_offset`),
`mtp_compose.rs`, `mtp_spec.rs`, `mtp_probe.rs`, plus scratch/KV types and a
batched variant. The artifact convention already reserves `+mtp`.

`load_mtp_head` reads a `.mtp` sidecar using **bare** tensor names (`enorm`,
`hnorm`, `wq`, ...). So the missing piece is a name mapping plus the conversion:

    mtp.pre_fc_norm_embedding.weight  -> enorm
    mtp.pre_fc_norm_hidden.weight     -> hnorm
    mtp.fc.weight [5120, 10240]       -> e_proj / h_proj (concat; needs splitting)
    mtp.layers.0.self_attn.*_proj     -> wq / wk / wv / wo
    mtp.layers.0.mlp.*_proj           -> ffn gate / up / down
    mtp.layers.0.*_layernorm          -> attn_norm / ffn_norm
    mtp.layers.0.self_attn.{q,k}_norm -> head norms
    mtp.norm.weight                   -> final norm

Per AGENTS.md this is format conversion, so it belongs in
`hipfire-coexistence`, not the daemon or runtime.

## Open questions before committing to it

- Does `Qwen35MtpHead`'s block match Qwen3.8's MTP layer exactly (GQA 12288/1024
  q/kv, q_norm/k_norm at head_dim 256, o_proj in 6144)? The existing head was
  written against an earlier Qwen MTP.
- `mtp.fc` is a single [5120, 10240] matrix over concat(embed, hidden); hipfire
  models it as separate e_proj/h_proj. A column split at 5120 should be exact —
  verify rather than assume.
- What precision for the head? It is 1.6% of the model, so bf16 costs little and
  protects acceptance; quantizing it to oq4 saves ~600 MB of draft-step traffic.
  Measure both.

---

# CORRECTION: MTP is the WRONG target. DFlash2 already beats it on this model.

Everything above ranks drafters by raw acceptance rate. That is the wrong metric,
and the conclusion it produced ("wire up MTP") is wrong.

The DFlash2 paper (inco.ai/blog/dflash2) reports, **for Qwen3.8-27B specifically**
(its Table 4), mean acceptance LENGTH:

    DFlash 2   4.80
    MTP        4.28
    DSpark     3.62

DFlash2 beats MTP on the exact model in question. And the pass structure differs:
DFlash2 is single-pass — "the entire block, every position, predicted in
parallel", one drafter forward per cycle — whereas MTP runs n sequential passes.
Normalising per byte, which is the metric that matters on a bandwidth-bound box:

    DFlash2   4.80 tokens / 1.2 GB (1 pass)        = 4.0 tokens/GB
    MTP       4.28 tokens / ~1.58 GB (7 x 226 MB)  = 2.7 tokens/GB

DFlash2 wins by ~1.5x per byte. The MTP head's small size is cancelled by needing
to be re-read once per drafted token.

Reported speedup for Qwen3.8-27B at batch 1: **2.7-3.4x autoregressive**. On our
15.1 tok/s AR that is **41-51 tok/s** — above the 36.04 the ROCmFP4 MTP build
reaches. So the ceiling here is higher than the external benchmark, not lower.

## The real gap is overhead, not drafter choice

                        designed        ours
    tau                 4.80            2.0 - 3.0
    speedup vs AR       2.7 - 3.4x      0.33x  (5.06 vs 15.1)

Acceptance is short, but it is the SMALLER problem. At our own measured tau=3.0
the byte roofline — 1.2 GB draft + 14.3 GB verify at the 233 GB/s this kernel
sustains — is 3.0 / 0.0665 s = **~45 tok/s**. We measure 5.06. That is ~9x of
pure overhead, and no amount of acceptance work touches it.

Where it is, from the profile already taken:

- **The draft moves 1.2 GB in 64 ms = 19 GB/s**, against 233 GB/s achievable.
  DFlash2 is single-pass by design, so this is ONE forward running 13x slower
  than its bytes justify — the same dispatch-bound, small-M regime measured in
  `2026-08-21-npu-drafter-dispatch-cost.md` (~214 GMAC/s, 0.4% of peak). That
  document's conclusion stands and is now the main event rather than a footnote.
- Rollback and verify are already addressed on this branch.

## Hypothesis worth checking first (cheap)

The paper puts the selector at 2.0M params and the convolutions at 16.5M = "3% of
drafter", implying a drafter of roughly 550M params ~ 310 MB at oq4+. Our
artifact is **1.2 GB** — a gap about the size of the 248320 x 5120 vocab
embedding (1.27G params). If the drafter artifact carries, uploads and re-reads
an embedding/head the target already holds, that is simultaneously a byte problem
and a dispatch problem, and it would be the first thing to fix.

## Revised order

1. Check the drafter artifact's tensor inventory for a redundant vocab
   embedding / lm_head.
2. Fix the drafter's dispatch efficiency (19 GB/s -> near 233). This is worth
   roughly 9x and is independent of acceptance.
3. Only then chase tau 3.0 -> 4.80.
4. Do NOT convert the MTP head. It loses to DFlash2 per byte on this model.
