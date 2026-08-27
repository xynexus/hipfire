# Spec-decode output is not byte-identical to plain AR decode

**Model:** Qwen3.8-27B--oq4.25++.hfq + `dflash2.oq4+` draft, halo (gfx1151),
greedy, `dflash_spec_demo`.

Under greedy decode, speculative decoding is supposed to be *lossless*: a draft
token is committed only when it equals the target's own argmax, so the emitted
text must match plain autoregressive decode exactly. On this model it does not.

## What was measured

`--ar-baseline` reproduces byte-for-byte across runs, so the reference is
deterministic. Against it, at `--max 256` on a Python prompt:

| config | vs AR |
|---|---|
| `--no-speculate` | IDENTICAL |
| `--no-adaptive-b --block-size 6` | DIFFERS |
| `--no-adaptive-b --block-size 8` | DIFFERS |
| adaptive (default) | DIFFERS |

The divergence point is **B-dependent** — char 697 at B=6, char 401 at B=8 —
so it tracks where cycle boundaries fall, not a fixed context length. This is
why a short run can look clean: at `--max 128` with B=6 the run ends before the
first flip.

## Two hypotheses, both refuted by measurement

**1. "A KVarN block sealed during verify keeps rejected tokens."** KVarN
quantizes K in 128-token blocks with a joint Sinkhorn variance normalization, so
a block sealed mid-speculation could in principle bake in tokens that are later
rejected — and the spec path never re-flushes (`kvarn` appears 3 times in
`speculative.rs`, all construction). The position evidence fit: same prompt and
B=8, `--max 80` (never reaches position 128) is IDENTICAL, `--max 200` DIFFERS.

Refuted: `--kv-mode q8` also diverges, and Q8 has no block tiling, no Sinkhorn
and no records at all. The 128-boundary correlation was coincidence — enough
tokens had simply accumulated to flip an argmax.

**2. "The batched attention takes q8 KV scales per-tile."** This is the
explanation `is_batchable_la`'s own comment offers, hedged as "most likely".

Refuted by reading the kernel: `kv_cache_write_q8_0_batched` derives its scale
from `positions[bid]` over a 32-element block within one head of one token —
exactly the granularity the per-token write uses. Scale granularity is identical
on both paths.

## Actual root cause

**Verify runs the batched forward; AR decode runs the per-token forward; the two
are not numerically equivalent.** This is documented and *deliberately accepted*
in `qwen35/mod.rs` at `is_batchable_la`:

> CAVEAT, deliberately accepted: the batched path is not numerically identical
> to per-token. Typical |delta logit| is ~6e-2 (max 2.4e-1) against ~4e-6 for
> pure reordering, and only 15% of positions keep the same top-256 set. […]
> Anything that needs the two to agree bit-for-bit must pin the path explicitly.

Speculative decoding is precisely something that needs them to agree bit-for-bit.

⚠️ **`--kv-mode f32` looks IDENTICAL, and that result is an artifact.** f32 KV
does not satisfy the batched-verify predicate, so verify silently falls back to
per-token and is trivially equal to AR. The timings give it away: under f32,
spec decode runs **6.79 tok/s against AR's 15.56** — 2.3x *slower*, at tau 3.9.
Under kvarn8 it is 21.35 vs 12.40, genuinely batched, and it diverges. Do not
read the f32 row as evidence that quantized KV causes this.

This also corrects the older note that DFlash "diverges at every KV tier
including the f32 oracle, therefore a verify forward bug" — the f32 tier was
never running the verify path being blamed.

## Where the difference enters

`compare_prefill_hidden_paths --model <27B> --n 512 --kv-mode q8` runs both
forwards in one process against the same ring buffer:

    layer   worst|rel|   at row
        0     1.29e-3      254
        3     5.87e-3       13
       21     8.14e-3      421

    against the fp32-KV reference: batched 4.760e-2, per-token 1.254e-2

Layer 0 already differs, so this is not accumulated drift. The worst layer-0 row
is **254 — the last row of the 256-token chunk**, the row with the most in-chunk
neighbours, and the delta then compounds up the stack. That is the signature of
in-chunk K/V being read one way by the batched path and another by per-token:
within a chunk the batched path attends over its neighbours directly, where the
per-token path reads those same neighbours back through the quantized cache.

The MoE grouped-vs-indexed gate in the same tool reports `worst |rel| 0.000e0`,
so the FFN is not involved.

## Transformers oracle: neither path is defective — PIN the path

Run 2026-08-27. The internal fp32-KV reference can only fix the KV axis; it
cannot arbitrate batched-vs-per-token, because both arms are ours. An external
fp32 HF oracle can.

Done on **Qwen3.5-0.8B bf16**, not the 27B: the 27B is `oq4.25++`, and comparing
a quantized model against a bf16 reference mixes quantization error into the very
quantity being measured. The divergence reproduces in plain bf16 anyway.
`compare_prefill_hidden_paths` now accepts an HF snapshot directory
(`HfqFile::from_safetensors`), so hipfire and the oracle load the *same weights*,
and `HIPFIRE_ORACLE_DUMP=<dir>` exports each arm's per-layer hidden states.
Tokens are the tool's own `1000 + i*7`, n=512.

Alignment was verified, not assumed: hipfire's layer 0 matches HF `hidden_states[1]`
at cos 0.999996, against 0.17 for `hidden_states[0]`.

    layer   cos(batched,HF)   cos(pertoken,HF)   delta
        0     0.999996417       0.999996849      -4.3e-07
        7     0.999979373       0.999980380      -1.0e-06
       22     0.999989880       0.999990868      -9.9e-07

    layers where each arm is closer to the oracle: pertoken 23, batched 1

**Both arms are within ~1e-5 of the fp32 reference.** Per-token is consistently
closer, but by ~1e-6 in cosine — a reproducible sign with a negligible magnitude.
There is no defective batched kernel to hunt here: this is benign numerics.

So the fix is to **pin the path** so verify and AR provably take the same one,
NOT to rewrite the batched GEMM/attention for bit-parity.

### Cross-checked on independent hardware

The arbitration above decides between a one-line path pin and a kernel-parity
project, so it was re-run on a second box rather than trusted from one.

`duat` (RTX 3090, **CUDA** torch 2.13.0+cu130, **transformers 5.15.0**) against
halo (gfx1151, **ROCm** torch 2.12.0a0, **transformers 5.2.0**), same NFS
snapshot `2fc06364…`, same hipfire dump:

    layer 0   halo 0.999996416575   duat 0.999996416569
    layer 22  halo 0.999989880219   duat 0.999989880257
    layer 23  halo 0.935017734096   duat 0.935017733706
    verdict   pertoken 23 / batched 1 on BOTH

Agreement to ~10 significant figures across two GPU vendors and two transformers
majors. So the ~1e-6 per-token edge is a real, reproducible signal rather than
per-box noise — it is just far too small to be worth a kernel rewrite. And the
layer-23 figure reproducing to 9 digits confirms it is a structural convention
difference, not numerical noise on either machine.

Two notes on reading that table:
- Layer 23 reads 0.935 for BOTH arms equally. That is a hipfire-vs-HF export
  convention at the last layer (HF's final `hidden_states` entry has the final
  norm applied), not a path difference, and it cancels in the comparison.
- ⚠️ The first version of this measurement used a 2-D cosine that silently
  returned values >1 (5.18 for two arrays that agree to 1e-6). Those numbers were
  void. The metric now flattens to 1-D and asserts `cos <= 1`, so a broken metric
  cannot quietly emit a table again.

## Hypothesis 3, also refuted: DeltaNet fp16 vs fp32

Raised because it has exactly the right shape — there IS an f32/f16 split running
along precisely the batched/per-token seam. Per-token calls
`gated_delta_net_f32(`; batched calls `gated_delta_net_f16_batch_seq(`; the arm
is chosen by `dn_state.quant`, i.e. by `HIPFIRE_DN_STATE_FP16`. And **every**
spec-decode run in this investigation had that env var set to 1.

`prefill_lowered.rs` even documents the asymmetry:

> FP16 only — the FP32 arm above is batched, and batched vs per-token is
> identical there (no narrowing). f16 narrows once per launch, so per-token
> matters.

So fp16 narrowing once per LAUNCH genuinely makes a batch of n differ from n
single steps, while fp32 has no narrowing and is identical either way.

**Refuted anyway, on both symptoms.** Qwen3.8-27B--oq4.25++, env verified to
have taken effect (`dn_quant=` read back from the feature report, not assumed):

- *Byte divergence*: with DN pinned to FP32 the spec output STILL differs from
  AR, and the divergence lands at the same 595th character as under FP16.
  Stronger still, AR output is byte-identical across FP32/FP16, and so is spec
  output — DN precision changes neither stream.
- *Faithfulness gap*: batched/per-token vs the fp32-KV reference is
  4.760e-2 / 1.254e-2 at DN FP32 and 4.839e-2 / 1.203e-2 at DN FP16. The 3.8x
  gap is fully present at FP32, where DeltaNet is bit-identical between paths.

Keep the mechanism in mind for other models — on one where DeltaNet carries more
of the signal it would bite, and `use_gdn_per_token` exists to neutralise it.
It is simply not what is happening here.

### Side effect worth keeping: fp16 DeltaNet is now the default

Measured while refuting the above, and the reason `deltanet_state_precision`
defaults to fp16 as of 2026-08-27:

| DN state | decode tok/s | tau |
|---|---|---|
| fp32 | 16.78, 14.48 | 3.682 |
| fp16 | 19.14, 18.88 | **4.103** |

Generation is byte-identical to fp32 on code, prose, numbers and JSON prompts,
and tau reproduces EXACTLY across repeats, so the +11.4% is signal rather than
the run-to-run noise that contaminated the raw tok/s. Plain AR decode is flat,
which locates the win: the drafter reads the verify path's hidden states and
agrees with the fp16 ones more often, while the verifier's committed tokens are
unchanged. That is why tau can move without the output moving.

⚠️ An earlier note in this session claimed fp16 was "+11% faster" on the basis of
a single pair of runs (15.80 vs 14.18). That was contention noise — AR decode is
flat. The real and reproducible effect is on tau, not on raw decode rate.

## Still open: the compact 27B batched arm is a separate outlier

Faithfulness to each model's own fp32-KV reference (lower is better):

| model | batched | per-token |
|---|---|---|
| Qwen3.8-27B `oq4.25++` (compact Opus) | 4.760e-2 | 1.254e-2 (**3.8x worse**) |
| qwen3.5-2b bf16 | 2.057e-2 | 1.580e-2 (1.3x) |
| Qwen3.5-0.8B bf16 | 1.453e-2 | 1.471e-2 (tied) |

On bf16 the two arms are comparably faithful; on the compact-Opus 27B the batched
arm is markedly worse. The compact batched arms were only admitted 2026-08-25.
That gap is NOT explained by the oracle result above and is worth its own look —
but it is a faithfulness question, separate from the byte-identity one.

⚠️ Also retired: a recorded claim that the qwen3.5-2b bf16 control "diverges from
layer 0 at |rel| 0.93". Re-measured here at **1.85e-3**, peak 1.36e-2. The 0.93
figure was the documented dump-diffing trap (`.batched.L{i}` vs `.pertoken.L{i}`
are different call sites).

## What a fix requires

**Pin the path.** The oracle above settles the open question: both forwards are
~1e-5 faithful to fp32 HF, so there is no defective kernel to fix, and chasing
bit-parity in the batched GEMM/attention would be work aimed at a ~1e-6 effect.
Verify and AR must simply be made to take the same path.

Note the scope is wider than spec decode — the same caveat means batched prefill
and per-token prefill disagree for *every* dtype, bf16 included.

## Not affected

Draft quality: tau is measured against the verifier's own argmax, so it remains
a valid drafter metric. The +71% tau from driving the draft at its trained block,
and the adaptive-B controller fix, are independent of this and stand.
