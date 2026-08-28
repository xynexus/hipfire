# Opus QAT tier sweep — how much of each width's loss is recoverable?

Track A step 1 of `docs/plans/2026-08-28-opus-qat-and-dflash2-finetune.md`.
halo / gfx1151, Llama-3.2-1B fp32, `crates/hipfire-train/examples/qat_opus_kvarn.rs`.
Run 2026-08-28, ~20–34 min per tier.

## Headline

Frozen Opus-fake-quantized base + trainable LoRA(q/v) + RMSNorm, KL-distilled
against a clean fp32 teacher. **KV clean**, real text, LR 1e-4 with 10-step
warmup then cosine, 120 steps, held-out drawn from the opposite half of the
corpus file:

| tier | held-out KL before | after | **held-out recovered** | in-sample recovered |
|---|---|---|---|---|
| `oq8` (W8) | 0.0012 | 0.0033 | **~0%** (best 0.0012 @ step 0) | 18.1% |
| `oq4` (W4) | 0.2211 | 0.1392 | **37.0%** (best @ step 120) | 65.4% |
| `oq3` (W3) | 1.3768 | 0.4798 | **65.1%** (best 0.4797 @ step 100) | 82.5% |

Recovered share rises monotonically with damage, which is what headroom
predicts. W8 is the control: there is essentially nothing to recover, and the
run drifts very slightly upward rather than down.

**W4 — the deployed tier — had no number before this. It is ~37%, and that is a
floor, not a ceiling:** its best held-out point is the *last* step, with the
curve still descending (0.1419 → 0.1395 → 0.1392) as the cosine reached zero.
The 120-step budget is the binding constraint, not the method.

Weight-only deploy loss per tier, held-out (the LoRA=0 column above), is itself
the more reusable number: W8 0.0012, W4 0.2211, W3 1.3768 nats/tok. W8→W4 costs
~184×; W4→W3 a further ~6.2×.

## Three defects found on the way, all fixed

**1. The example could not load a model.** `DEFAULT_DIR` pointed at a snapshot
holding only a `.gguf`, so `load_llama_fp32` died before touching the GPU. Now
points at `Llama-3.2-1B/snapshots/main` — note the `snapshots/main` pin, the
sibling snapshot dir has no weights — plus a safetensors preflight that names
the reason instead of emitting a bare loader error.

**2. The training set was 64 tokens.** `N_TRAIN * SEQ` = 4 × 16, against 97
trainable tensors, with one fixed batch repeated 120 times. It memorised
instead of generalising: the first run drove in-sample KL 2.2430 → 0.6114
(72.7% "recovered") while **held-out went 2.5152 → 2.9003, 15% worse**. Fixed
by cycling a 32-batch pool (128 sequences) with 32 held-out sequences, both
tokenized from the model's own `tokenizer.json` via `HIPFIRE_QAT_CORPUS`. Costs
one extra teacher precompute and nothing per step.

**3. There was no LR schedule.** A flat `LR=1e-3` diverges, and it is worst
exactly where the model is healthiest. On W8 KV-clean over real text the student
starts essentially undamaged and the optimizer walks it away:

| step | 0 | 20 | 40 | 60 | 80 | 120 |
|---|---|---|---|---|---|---|
| held-out KL, flat 1e-3 | 0.0012 | — | — | — | — | **3.6850** |
| held-out KL, 1e-4 + warmup/cosine | 0.0012 | 0.0012 | 0.0016 | 0.0026 | 0.0023 | 0.0033 |

A ~3000× degradation becomes a ~2.75× drift. `AdamW::set_lr` already existed and
its header already said "supports a schedule" — the example simply never called
it. Same shape as the DSpark drafter's non-convergence.

Reported as `-306125.6% recovered`, which is worth remembering as a shape: a
wildly negative recovered share at a low-damage tier means the optimizer is
diverging, not that QAT does not work.

### Schedule shape vs LR magnitude — the magnitude is the bigger lever

Running peak 1e-3 *with* the same warmup+cosine separates the two. On W8:

| condition | final held-out KL | vs undamaged 0.0012 |
|---|---|---|
| flat 1e-3 | 3.6850 | ~3070× |
| 1e-3 + warmup/cosine | 0.2471 | ~206× |
| 1e-4 + warmup/cosine | 0.0033 | ~2.75× |

So the schedule is worth ~15×, and dropping the peak a further ~75×. The
schedule alone does **not** rescue a too-hot LR. The 1e-3 trace also shows why:
damage is inflicted during the high-LR phase (held-out 0.0010 → 0.3356 by step
20 → 0.6744 by step 40) and the cosine tail only partly repairs it (→ 0.2471).
It never returns to where it started.

Worth stating plainly because the first diagnosis here was "missing LR
schedule", and that is the *smaller* half of the fix.

Running the full tier sweep at both peaks settles the LR choice. A plausible
guess going in was that the damaged tiers, having more headroom, would tolerate
or reward the hotter LR. They do not — 1e-4 wins everywhere:

| tier | held-out recovered, peak 1e-4 | peak 1e-3 |
|---|---|---|
| `oq8` W8 | ~0% (0.0012 → 0.0033) | −20436% (0.0012 → 0.2471) |
| `oq4` W4 | **37.0%** (0.2211 → 0.1392) | −54.5% (0.2211 → 0.3415) |
| `oq3` W3 | **65.1%** (1.3768 → 0.4798) | 36.4% (1.3768 → 0.8752) |

Both 1e-3 traces at W4 and W3 show the same shape as W8: a spike during the
high-LR phase (W4 peaks at 0.8387, W3 at 3.1881, both around step 40) that the
cosine tail only partly walks back. At W4 it never returns below its own
starting point. 1e-4 is the peak to use; there is no tier where 1e-3 pays.

## The default arm measures KV quantization, not the Opus tier

`HIPFIRE_QAT_KVNOISE` defaults on, so the out-of-the-box sweep runs W_tier +
KVarN-4. Its before-KL at W8 was **2.5152** — but W8 weight-only damage is
**0.0012**. Roughly 2.5 nats/tok of that is a floor common to all three tiers
and is not weight damage at all. For reference, that floor is larger than W3's
entire weight-only loss (1.3768).

The default-arm numbers, for the record (synthetic ids, one fixed batch, flat
LR — all three defects above still present, so treat as historical):

| tier | held-out before | after | recovered |
|---|---|---|---|
| `oq8` | 2.5152 | 2.9003 | −15.3% |
| `oq4` | 3.0498 | 2.5124 | +17.6% |
| `oq3` | 3.8128 | 2.8648 | +24.9% |

Consistent with the earlier finding that KVarN-4 loss is largely
non-recoverable and KVarN-8 is the tier to deploy.

## What this does not say

- **Llama-3.2-1B is not a deploy target.** The recoverable share is measured on
  a 1B dense model; it has not been shown to transfer to the models actually
  shipped.
- **Weight-only, A16.** `oqplus_quant` bakes weight error only. Stage A2 below
  measures the activation axis: A8 ≈ A16 is confirmed (so this table is a fair
  W4A8 proxy), but A4 is a materially different regime and none of the numbers
  in *this* section speak to it.
- **120 steps, LoRA(q/v)+norm only.** W4's 37% is what this budget and this
  trainable set reach, not what light QAT can reach.
- The KVarN floor above is inferred across two corpora (the default arm ran on
  synthetic ids, the weight-only arm on real text), so the ~2.5 figure is
  approximate. What is solid is that it is common to all three tiers and cannot
  be weight damage.

## Reproducing

    D=/srv/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/main
    C=benchmarks/calib/calib-1m.txt
    hipfire lock acquire "qat-opus"
    HIPFIRE_QAT_TIER=oq4 HIPFIRE_QAT_KVNOISE=0 HIPFIRE_QAT_CORPUS=$C \
      HIPFIRE_QAT_LR=1e-4 ./target/release/examples/qat_opus_kvarn "$D"
    hipfire lock release      # takes NO label

Each step line prints its own LR and held-out KL, so the schedule is verified by
readback rather than assumed.

## Next

1. ✅ Stage A2 — done, see below. A8 ≈ A16; A4 is a different regime.
2. Raise the step budget at W4; its best point is still the last step.
3. Decide from the above whether light QAT suffices at W4 or the base weights
   must move.

---

# Stage A2 — the W4**A4** path

> ⚠️ **SUPERSEDED IN PART — see Stage A2d.** Every activation-tier number in
> Stage A2 and A2b quantized the **raw channel-basis** activation. The deployed
> W4A4 path rotates the activation with the codec's per-256 FWHT first
> (`quantize_act_oq4.hip` documents its input as "assumed already FWHT-rotated";
> `fused_rmsnorm_mq_rotate_plain.hip` performs that rotation inside RMSNorm).
> Rotating is worth **+6.45 dB** on its own, so these arms describe a grid far
> harsher than anything that ships and the A4 loss figures are pessimistic. The
> weight-tier results (Stage A1, and the compact-grid section) are unaffected —
> `oqplus_simquant` always applied the FWHT. `maybe_quant_act` now quantizes in
> the rotated basis; these arms need re-running.

Plan §"Stage A2". Same day, same model and harness, weight tier pinned at `oq4`
(W4), KV clean, real text, LR 1e-4 warmup+cosine, 120 steps. **Only the
activation width moves**, so the sweep isolates it.

Activations are fake-quantized with `a4_quant::simquant_bits` — per-group
symmetric absmax, `GROUP = 256`, matching `Oq4G256` — applied forward-only (STE)
to the four tensors that feed all seven projections.

| activation | held-out KL before | after | **recovered** | best | wall |
|---|---|---|---|---|---|
| `a16` | 0.2211 | 0.1392 | **37.0%** | @120 | 1964 s |
| `a8` | 0.2202 | 0.1426 | **35.2%** | @120 | 2022 s |
| `a4` | **0.8624** | 0.6637 | **23.0%** | 0.6460 @ step 80 (25.1%) | 2072 s |

## The answer: light QAT does NOT absorb A4

Three independent readings, all pointing the same way:

- **A4 nearly quadruples the deploy loss before any recovery** — 0.2211 → 0.8624
  (3.90×). The activation side is a bigger problem than the W4 weight side it
  sits on top of.
- **It recovers a *smaller* share of that bigger hole** — 23.0% vs 37.0%. More
  damage did not buy more headroom, which is the opposite of what the weight-tier
  sweep showed (W8 ~0% → W4 37% → W3 65%).
- **It turns around.** A4's best held-out is step **80**, not 120; every A16 run
  in this study was still improving at the budget end. So "run it longer" is the
  lever for the weight tiers and is *not* the lever here.

Post-QAT residual is 0.6637 vs A16's 0.1392 — **4.8× worse after recovery**. The
conclusion is that LoRA(q/v)+RMSNorm is the wrong instrument for A4: it cannot
represent what int4-per-token activation quant destroys. That is what the
rotations exist for (plan step A2.3: fixed FWHT vs learned R1), and it is the
next thing to test before concluding the base weights must move.

## A8 ≈ A16 — confirmed, and it retroactively justifies Stage A1

0.2202 vs 0.2211 before recovery (−0.4%, i.e. indistinguishable), 35.2% vs 37.0%
recovered. Stage A1 was measured weight-only on the stated grounds that A8 adds
negligible KLD over A16; that assumption now has a direct measurement behind it
on this harness. **The entire activation cost is the 8→4 step, not the 16→8 one.**

## Notes

- **The A16 leg is a regression test, and it passed exactly.** With
  `HIPFIRE_QAT_ACT` unset the gate compiles to a no-op, and the leg reproduced
  Stage A1's `oq4` run to four decimals at step 0 (0.3029 / 0.2205), step 20
  (0.3262 / 0.1708) and the final (0.2211 → 0.1392, 37.0%). The four new
  insertion points do not perturb the baseline.
- **The host round-trip is cheap — do not write a HIP kernel.** `a4_simquant` is
  host-only, so each quantized tensor costs a `download_f32` + `memcpy_htod`.
  Measured: 2072 s (a4) vs 1964 s (a16), **+5.5%**. Four tensors per block per
  forward at seq 16 is simply not the bottleneck.
- **Do not equate this with "A4 ≈ −3.5 dB".** That figure is activation *SNR*;
  the numbers here are KL. They are consistent in direction, not in units.
- Same caveats as Stage A1 apply: Llama-3.2-1B is not a deploy target, and 120
  steps with LoRA(q/v)+norm is one budget, not the method's ceiling.

## Where the quant is applied, and why there

`linear_forward` (`ops/linear.rs`) is the single funnel all seven projections
pass through, and hooking it is a one-line diff — but it is **wrong**: it also
carries `lm_head`, the MoE router, drafter `in_proj`, and the LoRA A/B legs,
whose B-leg `K = lora_rank` (8–32) is below `GROUP = 256`.

Only **four** tensors feed the seven projections, so they are quantized where
they are produced in `block_forward_inner`:

| tensor | feeds | rows × feat |
|---|---|---|
| `xn1` | q, k, v | seq × h |
| `ctx` | o | seq × q_dim |
| `xn2` | gate, up | seq × h |
| `act` | down | seq × inter |

**The STE is free, structurally.** `linear_backward_x`'s `dx = dy·W` reads only
the (frozen) weight, and `rmsnorm_backward` takes the norm's *input*, never
`xn1`/`xn2`. Writing the quantized value into the same buffer that lands in
`BlockActivations` means `linear_backward_w`'s `dw = dyᵀ·x` uses the same `X` the
forward multiplied — that is the *correct* STE weight gradient, not a bug to fix
by saving an unquantized copy. This is the same mechanism `kv_noise` relies on.

⚠️ **`acts.gate` and `acts.up` must never be quantized.** `swiglu_backward`
recomputes `silu`/`silu'` from those pre-activations, so overwriting them moves
the Jacobian evaluation point. Only `act` (the swiglu *output*) is touched; it is
read solely by `linear_backward_w` for `dwdown`. `ctx` is quantized *after* the
attention-output gate multiply, leaving the `ctx_pre_gate` copy intact.

## Reproducing

    HIPFIRE_QAT_TIER=oq4 HIPFIRE_QAT_KVNOISE=0 HIPFIRE_QAT_ACT=a4 \
      HIPFIRE_QAT_CORPUS=benchmarks/calib/calib-1m.txt HIPFIRE_QAT_LR=1e-4 \
      ./target/release/examples/qat_opus_kvarn "$D"

`HIPFIRE_QAT_ACT` is `a16` (default, no-op) | `a8` | `a4`, panics on anything
else, and is re-read per block so the teacher precompute stays clean.

## Next

1. **Rotations under QAT** (plan A2.3) — score fixed FWHT vs learned R1 on the A4
   arm. This is now the open question, since LoRA alone is ruled out.
   ⚠️ learned rotations are PREFILL-ONLY; state which phase any number belongs to.
2. Only if rotations also fall short, revisit moving the base weights.

---

# Stage A2b — mixed precision: per-group outlier promotion

Same harness and settings (Llama-3.2-1B, KV clean, real text, LR 1e-4
warmup+cosine, 120 steps). Two sweeps, one variable moved at a time.

## Activation policies, weight tier pinned at uniform `oq4`

| activation policy | held-out before | after | **recovered** | best | extra bits/value |
|---|---|---|---|---|---|
| `a16` | 0.2211 | 0.1392 | **37.0%** | @120 | — |
| `a8` | 0.2202 | 0.1426 | **35.2%** | @120 | +4.0 |
| `a4o16` | 0.2803 | 0.1713 | **38.9%** | @120 | +0.25 |
| `a4o8` | 0.3037 | 0.1921 | **36.8%** | 0.1889 @100 | +0.125 |
| `a4,act=a8,ctx=a8` | 0.4580 | 0.3873 | 15.5% | 0.3514 @40 | +2.0 |
| `a4` | 0.8624 | 0.6637 | 23.0% | 0.6460 @80 | 0 |

**Per-group promotion beats per-site promotion, and it is not close.** Promoting
8 of every 256 activations (3.1%, an eighth of a bit) reaches 0.3037; promoting
two entire projections to int8 — 16× the extra bits — only reaches 0.4580. The
damage is not concentrated in particular projections, it is concentrated in a few
outlier channels *within every group*, which a per-group top-N catches and a
per-tensor tier structurally cannot.

**Promotion saturates fast.** 0 → 8 outliers cuts the pre-recovery loss 0.8624 →
0.3037 (−65%); 8 → 16 buys a further −7.7% only. `n_out = 8` is the knee.

**Outliers destroy RECOVERABILITY, not just accuracy — this is the real finding.**
Uniform A4 recovers 23.0% and peaks at step 80; the per-site arm recovers 15.5%
and peaks at step 40. Both `a4o8` (36.8%) and `a4o16` (38.9%) recover at the
**A16 rate** (37.0%) and peak at step 100/120 like the healthy arm. So Stage A2's
conclusion needs narrowing:

> Light QAT cannot absorb A4 **on a uniform grid**. Promote ~8 values per 256 and
> it recovers exactly as well as it does at A16.

A single outlier inflates its group's absmax and coarsens all 256 codes, and that
coarsening is what LoRA cannot compensate. Remove it and the optimizer behaves
normally again — the same mechanism rotations target, which makes the rotation
arm more interesting, not less, since a rotation achieves it with no mask.

## Weight grid: uniform int4 vs the compact grid that actually ships

`oqplus_simquant` models `quantize_oq4g256` — **uniform** int4. The mixed packers
(`oqplus_tiered_ldlq_pack`, compact qt=36) deploy bulk int4 with the top-`w8_frac`
of each group at int8 on **one shared scale**, outliers chosen by int8-upgrade
gain (`err4² − err8²`) via the joint clip search in `codecs::mixed_clipsearch`.
`oq4_mixed_simquant` mirrors that; `oq4.25` is `n_out = 16` (`4 + n_out/64` bits).

At A16:

| weight grid | deploy loss (LoRA=0) | after QAT | **recovered** |
|---|---|---|---|
| `oq4` uniform int4 | 0.2211 | 0.1392 | 37.0% |
| `oq4.25` compact | **0.1367** | **0.0675** | **50.6%** |

⚠️ **Stage A1's W4 numbers describe a harsher grid than deploys.** Real deploy
loss is 0.1367, not 0.2211 (overstated 1.62×), and light QAT recovers 50.6%, not
37.0%. Both remain valid statements about `oq4+`; neither describes `oq4.25`.

Note the symmetry with the activation result: promoting 16/256 **weights** lifts
recoverability 37.0% → 50.6%, exactly as promoting 8–16/256 **activations** lifts
it 23.0% → 37%. Outliers wreck QAT recoverability on *both* operands, and a
per-group top-N fixes both.

Both figures are still conservative: this models `oq4.25+` (clip-search tier).
The shipped `oq4.25++` adds LDLQ error feedback, and per that packer's header the
int8 positions carry ~zero residual, so OBS spends its whole budget on the int4
bulk.

## At A4 the activation grid dominates — fix it first

| weight grid | activation | before | after | recovered |
|---|---|---|---|---|
| `oq4` uniform | `a4` | 0.8624 | 0.6637 | 23.0% (best 0.6460 @80) |
| `oq4.25` compact | `a4` | 0.7753 | 0.6637 | 14.4% (best 0.5715 @40) |

A better weight grid buys −38% at A16 but only **−10% at A4**, and both arms land
on the same final residual. The two traces differ throughout (step 0: 0.7718 vs
0.8273; step 40: 0.5715 vs 0.6748) and only coincide at step 120, so this is
convergence to a shared activation-noise floor, not a tier that failed to apply
— though landing on 0.6637 to four decimals is a striking coincidence given how
much both traces bounce.

**Practical ordering: fix the activation outliers first.** Weight-grid refinement
is second-order until you do.

## Caveats specific to this section

- **`simquant_outlier` uses TWO scales** (a shrunken bulk scale plus a separate
  outlier scale), whereas the deployed weight packer shares one with a wider
  clamp. The `a4o*` rows are therefore an **upper bound** on what promotion can
  buy. That was the right first experiment — if even the upper bound had lost to
  uniform `a8`, promotion would be dead and no kernel needed — but the follow-up
  is whether the cheaper shared-scale form keeps the win.
- **The remaining gap to A8 is an engineering question, not an accuracy one.**
  `a4o16` sits at 1.20× A8's residual for 1/16 of the extra bits. Whether that
  wins depends on the per-group position mask versus `iu4x2`'s second activation
  plane over the same weight tile — which costs nothing in weight traffic.
- Same standing caveats as Stage A1/A2: 1B dense model, 120-step budget,
  LoRA(q/v)+norm only.

---

# Stage A2c — should activation quant be split across the rotation?

The question: outliers are **sparse in the channel basis**, and a rotation
Gaussianizes them but destroys that sparsity. So could a two-pass scheme have
both — capture the sparse outliers in the ORIGINAL basis where they are cheap,
then quantize the now-flat residual in the ROTATED basis where uniform int4 is
near-optimal?

    x ≈ s + r,  s = top-k int8 (channel basis),  r = x − s
    y = s·Wᵀ + Q4(r Rᵀ)·(W Rᵀ)ᵀ

`examples/two_pass_act_probe.rs` answers it host-side from one capture, measured
**end to end through the real weight** (never as reconstruction SNR of the
activation — `a4_quant`'s own tests warn that the raw norm is dominated by the
outliers, which quantize well, so a raw metric rewards a scheme for preserving
them while the bulk is crushed). Llama-3.2-1B, all **four** quantized sites,
SEQ=64 of **real text**, both the dense Hadamard and the codec's own `block_fwht`.

## Answer: no — two-pass never wins, at any operating point

End-to-end output SNR (dB, higher better), mean over 64 (act, weight) pairs:

| n_out | promote alone | + Hadamard | + codec block-FWHT | **2-pass** |
|---|---|---|---|---|
| 4 | 23.01 | 23.36 | **24.24** | 23.82 |
| 8 | 25.14 | 24.32 | 25.18 | 24.56 |
| 16 | **27.38** | 25.64 | 26.44 | 25.65 |
| 32 | **30.06** | 27.51 | 28.28 | 27.40 |

Reference at n_out=8: `a4` uniform 15.68, `a4`+Hadamard 21.20, `a4`+block-FWHT
22.13, `a8` ceiling 39.50.

Two regimes, and two-pass loses in both:

- **n_out = 4** — a rotation *does* help (24.24 vs 23.01). But the winner is
  **single-pass** rotate-then-promote, not two-pass (23.82). When you can afford
  very few promoted values, the rotation carries the outliers the mask misses.
- **n_out ≥ 8** — promotion alone wins outright and the gap widens with k
  (+0.9 dB at 16, +1.8 dB at 32). Adding a rotation actively *hurts*. Once the
  mask is wide enough to catch the outliers, rotating only smears energy the
  promotion was already handling exactly.

**Rotation and promotion are substitutes, not complements.** They attack the same
defect from opposite sides, and past n_out ≈ 4 the sparse mask strictly dominates.

Also worth recording: the codec's own `block_fwht` beats the dense Hadamard at
every k (e.g. 25.18 vs 24.32 at n_out=8), so a rotation study that scores only
`Rotation::hadamard` is scoring the wrong rotation.

## The implementation argument kills it independently

`y = s·Wᵀ + Q4(rRᵀ)·(WRᵀ)ᵀ` needs the weight in **two bases**. Storing both
doubles weight traffic, which is fatal. The only cheap form is `s·Wᵀ` as a gather
of a few UNROTATED weight columns — which requires the outlier positions to be
static enough to fix offline. They are not:

| site | fraction of dynamic top-8 slots a STATIC per-group mask catches |
|---|---|
| `xn1→q_proj` | 42.1% |
| `xn2→gate_proj` | 44.5% |
| `ctx→o_proj` | 49.4% |
| `act→down_proj` | **30.0%** |

Worst at `act→down_proj` — the largest tensor and the most outlier-heavy site.
A fixed offline mask would miss ~70% of the outliers exactly where they matter
most, so the mask has to ride the data and the cheap gather formulation is not
available. (Coverage rises only slowly with k: 28.6% at k=4 → 36.8% at k=32.)

## Method note

The first version of this probe measured only `xn1→q_proj` and `xn2→gate_proj`
on synthetic token ids — i.e. the two sites where the mechanism matters *least*,
on the same uniform-random tokens this repo's QAT loop was already burned by. It
reported the same ranking, but the numbers moved (e.g. promote-alone 26.77 →
25.14) once all four sites and real text were used. Recorded because "the
conclusion did not change" is only reassuring if you check.

## What this does not settle

- Only **fixed** rotations. `learn_rotation`'s Cayley-SGD R1 is untested here,
  and learned rotations are PREFILL-ONLY, so a learned-R1 result would not carry
  to decode regardless.
- SNR is not KL. It ranks schemes cheaply; the QAT arms in Stage A2b are the
  loss-level measurement.
- Nothing here prices the runtime mask against `iu4x2`'s second activation plane,
  which is the actual A8-vs-promotion decision.


---

# Stage A2d — which basis does the deployed W4A4 actually quantize in?

Prompted by a correction: the earlier two-pass probe (Stage A2c) held the
**weight at fp32**, so it scored the rotation on the activation operand only —
while the rotation's main job is making the *weight* quantization work. And it
treated "no rotation" as a baseline, which the serving path cannot express.

## Ground truth

- `kernels/src/quantize_act_oq4.hip` — `X : [B, K] f32 activations` **"(assumed
  already FWHT-rotated / SmoothQuant'd)"**.
- `kernels/src/fused_rmsnorm_mq_rotate_plain.hip` — Phase 2 *is* the FWHT
  rotation, fused into RMSNorm; it emits both `x_rot` and `x_plain`.

**The deployed W4A4 path rotates BOTH operands with the codec's per-256 FWHT,
always.** "Unrotated" is not a configuration.

## Corrected comparison, both operands quantized, in the codec basis

Since F is orthonormal, `<Fx, FW> = <x, W>`, and
`rotate_rows(oqplus_simquant(w), F)` recovers exactly the int4 weight the codec
stores. End-to-end output SNR (dB), 64 (act, weight) pairs, real text:

| scheme | n_out=8 | n_out=16 |
|---|---|---|
| W4A16 — weight quantized, activation exact | **22.00** | 22.00 |
| W4A4 deployed — a4 on the rotated activation | 18.80 | 18.80 |
| **W4A4 + promote in the F basis (deployable)** | **20.12** | **20.53** |
| W4A4 + promote in the channel basis (needs unrotated W) | 20.07 | 20.74 |
| W4A4 2-pass (channel-basis int8 + F-basis a4 residual) | 19.85 | 20.23 |
| W4A8 deployed | 21.98 | 21.98 |

Three conclusions, two of which correct Stage A2c:

1. **W4A16 = 22.00 is the ceiling.** Once the weight is int4, no activation
   scheme can pass it, and W4A8 (21.98) already reaches it. The entire activation
   budget is the 3.2 dB from 18.80 to 22.00.
2. **Promotion works equally well in the rotated basis** — 20.12 vs 20.07 at k=8,
   20.53 vs 20.74 at k=16. It **composes with** the rotation rather than
   substituting for it. Stage A2c's "rotation and promotion are substitutes" was
   an artifact of scoring only the activation operand.
3. **Two-pass is moot, not merely losing.** It trails single-pass F-basis
   promotion (19.85 vs 20.12), but the real point is that there is no longer any
   reason to want the channel basis — which also makes the 30–49% static-mask
   coverage irrelevant, since that only ever mattered for a channel-basis sparse
   term needing unrotated weight columns.

Deployable promotion buys **+1.32 dB** at n_out=8 and **+1.73 dB** at n_out=16 on
top of today's W4A4, closing 41% / 54% of the gap to W4A16 — with no second
weight copy and no change of basis.

## Consequence for the QAT arms

`maybe_quant_act` quantized the raw channel-basis activation, so every A4 figure
in Stage A2/A2b is measured on a grid ~6 dB harsher than deployment. It now
rotates → quantizes → inverse-rotates, which reproduces the deployed error
exactly (`<Fᵀ Q(Fx), Fᵀ Q(FW)> = <Q(Fx), Q(FW)>`) while leaving the tensor in the
channel basis for the rest of the fp32 forward.
`HIPFIRE_QAT_ACT_UNROTATED=1` restores the old behaviour for reproducing the
superseded arms.

**The A2/A2b activation sweep should be re-run before any of its numbers are used
for a deploy decision.** Expected direction: the A4 penalty shrinks substantially
and the recovered-share ordering may change, since the earlier "A4 destroys
recoverability" finding was measured on the harsher grid.
