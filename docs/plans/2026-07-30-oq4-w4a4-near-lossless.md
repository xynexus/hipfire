# oq4++ W4A4 near-lossless — investigation & ranked plan

Date: 2026-07-30. Status: **investigation / design** (no code landed).
Scope: make the shipped `oq4++` weight recipe run at **genuine int4 activations
(A4)** near-losslessly, and decide whether that is worth doing and by which lever.

Sources: `/srv/hipfire/references/Quant/{SURVEY.md, 2511.22316-SingleQuant,
2605.10793-ConQuR, 2412.14363-ResQ, 2310.19102-Atom}`, the linked Wan2.2 W4A4
entry (arXiv 2606.29337), SVDQuant (arXiv 2411.05007, not vendored), and the
in-tree oq4 stack. Prior threads:
`docs/plans/2026-06-23-oqplus-activation-quality.md`,
`docs/plans/2026-07-21-opus-quant-family-completion.md`.

---

## 1. Framing — what "oq4++ w4a4" actually is

In the artifact convention `oq4++` = symmetric-int4 weight + clip/AWQ (`+`) +
Hessian/LDLQ error-feedback (`++`). That **weight** recipe already shipped and is
production-good: `oq4` (LDLQ+AWQ, commit 48a8d07b) at KLD 0.046 (1.7× better than
mq4+), decode 56.7 tok/s, prefill 1610 tok/s (beats mq4+).

**But the shipped path is not W4A4.** It runs the good int4 weights against
**int8/f16 activations**, never int4:

- **Decode** = W4A16: `gemv_oq4_grouped` unpacks the nibble weight inline against
  an **f32** rotated activation. No activation quant at all.
- **Prefill** = W4A8 (int8-WMMA/MMQ, `gemm_oq4_residual_mmq`, q8_1 activation) or
  f16-WMMA (`gemm_oq4_grouped_f16_wmma`, the default coherent path).

So the honest problem statement is: **the int4-*activation* path was retired** — it
hit a list-primes attractor and diverged in batched prefill; the measured int4-act
penalty was **~1.1 PPL over W4A16**. "oq4++ w4a4" = revive int4 activations on top
of the already-good `++` weights, near-losslessly. This is the repo's stated open
frontier ("low-bit *activation* rotation for W4A4").

**Why chase A4 at all (the payoff regime).** int4 activations buy **compute /
tensor-core throughput**, not weight bandwidth. On bandwidth-bound batch-1 **decode**
the weight bytes dominate and A8→A4 barely moves wall-clock (oq4 decode is already
96.7% kernel, weight-bandwidth-bound). The A4 win lands in **compute-bound
prefill / large-batch serving** and in the **diffusion path** (Krea2 DiT / SeFi /
Flux2 — warm DiT steps are ~92% GEMM, compute-bound). Rank levers with that in mind:
A4 is a prefill/batch/diffusion lever, and any quality tool that only pays at decode
is misaimed.

## 2. Current state — the path is parked, not gone

The code map found the true-A4 kernels **still in-tree and parity-validated in
isolation**, reachable via `HIPFIRE_OQ4_BATCHED_PREFILL=1`:

- `quantize_act_oq4.hip` (per-group symmetric int4 + f32 scale, absmax) — present,
  no live default caller.
- `gemm_iu4_i32_wmma[_r1|_lds]`, `gemv_iu4_i32`, `gemm_oq4_grouped_wmma`, fused
  `fused_{qkvza,gate_up}_oq4_wmma/_dp4a`, MoE `gemv_oq4g256_moe_*` — all present.
- `Oq4G256 = 34` is the true-W4A4 quant-type (loader int4-quantizes activations,
  iu4·iu4 GEMM). `OqPlusG256 = 33` is the shipped W4A8.

The code comment at `qwen35/mod.rs:1229` hypothesizes the blocker is a
numerical-parity bug — "the batched rmsnorm+FWHT activation rotation is not
bit-identical to the per-token path" (≈0.63 mean / 3.95 max logit abs-diff on 0.8b
vs ≈0.018 for mq4 W4A16), amplified into flipped greedy argmax by the int4-act step.

**Phase 0 measurement (2026-07-30, gfx1103) REFUTES that hypothesis.** Every oq4
W4A4 building block is **bit-exact batched-vs-per-token** — there is no rotation
bug and no non-batch-invariant oq4 kernel:

| probe | result |
|---|---|
| `quantize_act_oq4` batched vs per-token (`parity_quantize_act_oq4`) | bit-exact, 0/65536 nib mismatch |
| int4-act GEMM `quantize_act_oq4[N]`+`gemm_oq4_grouped_wmma` vs per-row (`parity_oq4_batched_vs_pertoken`) | bit-exact n=5..9, dirty scratch |
| rmsnorm+FWHT rotation: fused-batched vs fused-single vs standalone `rmsnorm_f32`+`rotate_x_mq` (`parity_oq4_rotation_paths`, new) | all three bit-exact |
| `fused_qkvza_oq4_wmma` / `fused_gate_up_oq4_wmma` vs reference | bit-exact, Δ=0 |

So the ≈0.63 e2e divergence is **not** from any oq4 kernel. It comes from the
**non-oq4** pipeline stages that legitimately differ batched-vs-per-token
(attention / flash-attn, RoPE, KV-cache accumulation — mq4 shows ~0.018 there),
**amplified by the int4-activation step's discontinuity and compounded across 32
layers**. This is the *fundamental sensitivity of int4 activations to tiny upstream
perturbations*, exactly the risk flagged below — not a bug to fix. Crucially, the
per-token path is **not a golden oracle**: both batched and per-token int4-act
approximate f16, so bit-parity-with-per-token was the wrong gate. The right gate is
**absolute quality vs bf16** (KLD/ppl). The 06-24 thread already found batched OQ4+
prefill *coherent* on tested prompts with well-conditioned LDLQ+AWQ weights.

**Revised Phase 0 conclusion:** there is nothing to "un-park" at the kernel level.
The question becomes whether batched W4A4 prefill is *coherent/acceptable vs bf16*
despite differing from per-token — and if the int4-act sensitivity degrades quality,
that is precisely what **Lever B (A8 outlier subspace)** stabilizes. Phase 0
therefore feeds directly into the lever work rather than being a prerequisite bug-fix.

### Phase 0b (2026-07-30): the int4-act penalty is not measurable with packaged tools

Attempting to sweep `HIPFIRE_OQ4_PREFILL_ACT_BITS ∈ {16,8,4}` against the bf16
kldref surfaced a **structural gap, which is itself the finding**:

- qwen3.5 KLD/quality scoring (`Qwen35KldForward::forward_chunk_scored`,
  `loading.rs:3201`) is a **per-token loop** over `forward_scratch` at batch=1 → the
  **decode `weight_gemv` path** → `gemv_oq4_grouped` = **W4A16** (f32 activation,
  no `quantize_act_oq4`). `HIPFIRE_OQ4_PREFILL_ACT_BITS` is read **only** in the
  batched `weight_gemm` arm (`weights.rs:1382`), which this path never calls.
- **int4-act (true W4A4) runs only in the batched prefill of serving / `bench_
  qwen35_speed`** — and those paths do **not** score logit quality. So there is
  **no packaged path that both runs int4-act AND measures quality.** `hipfire eval
  --kldref` cannot see int4-act; act 16/8/4 are identical W4A16 there.
- (Also: the current kldref is cross-version — built by 0.2.0, rejected/patched for
  the 0.3.0 daemon — so its absolute KLD is unreliable regardless.)

**Consequence for the plan:** a real int4-act quality number requires a **purpose-
built batched-prefill logit scorer** (e.g. `build_kld_ref_hipfire --kld-graph-prefill`
run on the oq4 model at act4 vs act16 and compared directly — same version, no bf16
needed — *if* its graph-prefill is confirmed to route oq4 through `weight_gemm`; else
a small custom harness). Until then the working estimate for the int4-act penalty is
the survey's **~1.1 PPL** (2026-06-21 sim), consistent with the four papers all
showing uniform-int4 W4A4 is not near-lossless without a lever.

**But the deeper lesson reframes the whole effort (see §4):** the *only* reason to
pay int4-act's quality cost is **throughput** (int4 tensor-core), and int4-act runs
*only* in batched prefill/serving. The shipped default there is already W4A8-MMQ
(int8-act) at batch≥64 — chosen as fastest — which means **int4-act's throughput
advantage over the incumbent is itself unproven** and is the cheap, decisive gate
that must come *before* any quality lever.

## 2c. Correctness check (2026-07-30) — before any speed work

Verify W4A4 is right/coherent before optimizing it.

**Confirmed:**
- **Math (per-projection):** the int4-act W4A4 GEMM realizes **18–23 dB SQNR vs f32**
  (`validate_opus_w4a4_e2e`), and every int4-act kernel is **bit-exact
  batched-vs-per-token** (Phase 0 probes).
- **Full-model stability:** a full 24-layer batched prefill of two prompts completes
  **cleanly (no NaN/explosion, 372 ms)** via the daemon `generate_batch_prefill`
  (state_kinds `attention_kv`+`deltanet_recurrent`, `HIPFIRE_QWEN35_PREFILL_SESSION_
  BATCH=serial`).
- **Coherent output (int4 weights):** `qwen3.5-0.8b.oq4.5++` generates *"The capital
  of France is Paris."* (greedy, clean, no attractor) — via the **W4A16** path
  (single-request prefill is per-token; decode is W4A16).

- **Coherent output (pure oq4, W4 weights):** quantized a fresh **pure `oq4++`**
  (AWQ + LDLQ-where-Hessian-available, `--hessian` = the `.calib.hfq`;
  `qwen3.5-0.8b--oq4++.hfq`) and drove the daemon `generate_batch_prefill` +
  `generate_batch_decode_step` loop. It generates **fully coherent text** — *"Hmm,
  the user is asking about the capital of France. This is a straightforward factual
  question…"* — no gibberish, no attractor. So the pure-int4 **weight** recipe is
  coherent end-to-end.

**The int4-ACTIVATION path is not reachable through serving on this hybrid model —
a routing limitation, established definitively:** even the pure `oq4++` is rejected
by the **fused-dense session-batch kernel** ("unsupported dense/MoE weight dtypes"),
so it is **not** a mixed-model artifact — that fused path simply does not support
`Oq4G256`. Every serving route then falls back to `serial_reference`, whose worker
id is **`worker:arch5:pp1:fp32`** — `pp1` = per-token, `fp32` = f32 activations =
**W4A16**, not int4-act. So oq4's int4-act GEMM (`gemm_oq4_grouped_act_batched`,
which lives in `forward_prefill_chunk`) is **not wired into the daemon/server prefill
for this hybrid model**; serving routes oq4 to per-token W4A16 or the
oq4-incompatible fused kernel.

**int4-ACTIVATION end-to-end — CONFIRMED COHERENT (2026-07-31).** Rather than the
serving path (which routes oq4 to per-token W4A16 or the oq4-incompatible fused
kernel), added a one-line **env-gated int4-act branch** to the per-token decode arm
(`weights.rs` Oq4G256 `weight_gemv`: `HIPFIRE_OQ4_ACT4=1` → `quantize_act_oq4` →
`gemm_oq4_grouped_act_batched` at B=1). This makes `hipfire chat` run **true int4
activations at every oq4 linear across all 24 layers, per-token, end-to-end**, and
directly A/B-comparable to W4A16 (env unset). On the pure `oq4++` artifact, greedy:
- factual: *"…The capital of France is Paris, which has been the seat of government
  since…"* — correct, coherent, no attractor.
- reasoning: *"Average speed is total distance divided by total time… distance = 60
  km, time = 1.5 hours. So average speed = 60 /"* — correct setup (60/1.5 = 40),
  coherent.
W4A16 (env off) is likewise coherent; the two differ only slightly, as expected from
the int4-act perturbation. **So the full-model int4-act path is numerically sound
end-to-end, not just per-projection.**

**Net:** math right (18–23 dB per-projection, kernel bit-exactness, stable full
forward) AND full-model int4-act generation coherent on factual + reasoning. The only
remaining int4-act limitation is a **serving-routing gap** (the batched/fused serving
path doesn't wire oq4's int4-act GEMM), not a correctness or coherence problem. Given
§4.0's roofline (int4-act = memory-capped ~1.6× *prefill-only*), int4-act is *correct
and coherent but of low practical value for LLM serving* — the machinery's real payoff
is diffusion DiT (§8).

## 3. The four levers

Each is evaluated on: mechanism, hipfire mapping (reuse vs build), and an
**evidence-based bound** on how much of the ~1.1 PPL gap it closes. The bounds
reconcile the papers (measured vs *no* rotation / RTN) against hipfire's reality
(the codec **already** applies a per-256 FWHT + AWQ + LDLQ, so most of the 1.1 PPL
is already banked and the *incremental* headroom is smaller).

### Lever A — Learned/closed-form activation rotation (ConQuR ≫ SingleQuant)

**Mechanism.** ConQuR learns an orthogonal `R` that aligns normalized activations to
the nearest inscribed-hypercube corner (`min_R Σ‖Rx̃−z‖²`, `z_j=sign((Rx̃)_j)/√d`),
which reduces to **maximizing the ℓ₁ norm** of rotated activations, solved by
alternating **orthogonal Procrustes** (`C=ZᵀX̃`, `SVD(C)=UΣVᵀ`, `R←UVᵀ`),
online per-mini-batch, quant-aware (activations quantized mid-calibration). Deploy:
`R₁` (dense residual-stream, **folded** into input projections, inverse into
`W_O`/`W_down`) + `R₂,ℓ` (per-head, **folded** into `W_V`/`W_O`) + `R₃/R₄` (online
Hadamards). **R₁/R₂ are weight-merged → zero inference cost.** Cost: 0.42 GPU-h, no
activation corpus stored. SingleQuant's closed-form **ART** Givens angle
(`θ*=atan2(b,a)−π/4`, equalizes a massive-outlier pair) is a cheap init but its
deploy model is an *online* Kron-Hadamard rotation — a step backward from merged.

**hipfire mapping.** Deploy is **100% existing infra**: `rotate.rs::R1Plan`
(`R1=Fᵀ·M`, so the codec FWHT `F` cancels `Fᵀ` and the int4 grid sees learned `M`)
+ `rotation.rs::apply_r2` (per-head merge, already unit-tested) + the FWHT-online
kernels for R₃/R₄ + `a4_quant::{a4_simquant,snr_db}` as the scorer. **New = one
quant-time objective**: corner-Procrustes with online quant-aware accumulation,
slotting into `learn_rotation.rs` next to the existing `kurtosis`/`phase_joint`
objectives and the calibration engine (`LayerStreamEngine`). Optionally ART-init.
**No new kernels.** Build **M**. Caveat: `R1Plan` is arch_id 0/1 (dense-llama) only
today — extending the merge to qwen3.5/GQA/MoE is part of the cost.

**Evidence-based bound.** Papers' headline gains are vs *no* rotation (the full 1.1
PPL cliff). hipfire already has a per-256 FWHT, so the relevant delta is
better-rotation-vs-good-rotation: ConQuR vs QuaRot = **0.44 PPL** (Llama-3-8B
4-4-16); SingleQuant vs SpinQuant-RTN = 0.02 and *loses* to QuaRot-GPTQ on 2-7B.
This matches the repo's own sim (~1 dB / 0.3–0.5 PPL for a learned per-group
rotation; full-dim Hadamard was a **dead end**). The one thing the papers exploit
that the per-256 block FWHT structurally *cannot* is **full-dim outlier spreading** —
a massive-outlier channel can't be dispersed beyond its 256-lane block — and a dense
full-`h` merged `R₁` (exactly what `R1Plan` provides) is the right vehicle for it.
**Bound: ~0.3–0.5 PPL, nearly free at deploy. Will not make W4A4 lossless alone.**

### Lever B — Mixed-precision A8 outlier subspace (ResQ ≫ Atom/MixQ)

**Mechanism.** Keep a small high-variance activation subspace at int8, the rest at
int4. **ResQ**: PCA on activation covariance → the top-`r=d/8` variance directions
form the A8 subspace via an orthogonal `U`; **folded** into adjacent weights offline
(contiguous channel range, **no runtime gather**); a within-subspace random rotation
(hipfire's FWHT) Gaussianizes it. Because `U` is orthogonal the cross-precision
products vanish, so compute = **one int4×int4 GEMM + one int8×int8 GEMM, summed** —
only same-precision kernels, ~14% slower than pure int4. **Atom/MixQ** instead pick
outlier channels by max-abs/square-sum (static, calibration-fixed) and reorder them
contiguous; a scattered set needs a gather that breaks the 128-wide vectorized WMMA
load on RDNA3.

**hipfire mapping.** ResQ fits decisively better: the `U` fold reuses the existing
AWQ scale-fold plumbing (`awq_pre_scale_weights`/`compute_awq_scales`), the two
branches are today's `gemm_oq4_residual_mmq` (int4) + the existing W8A8 iu8 path,
summed — and `roughquant`'s dual-branch (bulk kernel + sparse correction, summed) is
the in-tree template proving the pattern runs on RDNA3. `quantize_oqplus_{tiered,
compact}` already implement "int4 bulk + sparse int8 outliers" with an
**output-error-optimal gain rule** (`e4²−e8²` after FWHT, not raw magnitude) — that
selection idea ports to the activation side; but its **storage trick does not**
(those weight codecs expand to dense int8 on *load* and never see a split compute
path — activations are produced at runtime, so an activation split needs a **real
dual-precision GEMM**). New = the PCA/`U` offline pass + the two-branch launch +
sum; runtime is otherwise reuse. `mixed_precision.rs` has offline 3-tier assignment
but **no runtime mixed-bit A4 dispatch** — that's the build. Build **M**.

**Evidence-based bound.** This is the strongest quality lever. Atom ablation: int8
vs fp16 outliers = **+0.05 PPL** (int8 is enough), and a ~1/32 static int8 channel
set brings W4A4 to **<0.4 PPL of FP16 on 65B**; ResQ's ⅛ subspace is within a few %
of FP16 and 4–33% below SpinQuant. The A4 gap is dominated by a handful of
high-variance channels; keeping ⅛ (ResQ) to 1/32 (Atom) of them at A8 recovers most
of it. **Bound: ~60–90% of the 1.1 PPL gap.** Reframe: since hipfire is *W4A8 today*,
this lever's real payoff is "realize int4-activation **throughput** while keeping
near-A8 quality" — validate as a perf/quality tradeoff, not a quality gain over the
current W4A8. Keeping a subspace at A8 is a middle point between full-A4 and W4A8;
its throughput win over W4A8 is proportional to how small the A8 subspace can be.

### Lever C — SVD low-rank outlier branch (SVDQuant / Nunchaku)

**Mechanism.** Smooth outliers activation→weight (SmoothQuant), then SVD-split the
smoothed weight `Ŵ = L₁L₂ + R`: a rank-`r` (16–32) **f16** branch `L₁L₂` absorbs the
few large singular directions smoothing concentrated there, leaving a residual `R`
that quantizes cleanly to int4. Deploy: `XW ≈ X̂·L₁L₂ + Q₄(X̂)·Q₄(R)`. The crux is
**kernel fusion**: run naively the low-rank branch is a separate f16 GEMM that
re-reads the activation from DRAM (~50% overhead, erasing the 4-bit win); Nunchaku
fuses the down-proj `X̂·L₁` into the activation-quantization kernel (activation read
once) and the up-proj `·L₂` into the 4-bit GEMM's accumulator epilogue → **5–10%
overhead, nearly free.**

**hipfire mapping.** Residual quant = oq4 verbatim (`Q₄(R)` is the existing
LDLQ+AWQ path); new = the offline smooth+SVD-split emitting `L₁/L₂` f16 sidecars (a
`+lowrank` embedded HFQ group) and a fused low-rank add. The prefill MMQ/f16-WMMA
kernels already ship a `..._full_add<ADD=true>` accumulator epilogue — the up-proj
add rides that; the down-proj rides the q8_1 activation-quant pass. Build: **S**
(offline quality probe, zero kernel) → **M** (prefill+diffusion fusion).

**Evidence-based bound & risk.** SVDQuant's near-lossless W4A4 is **diffusion-only,
compute-bound** — no LLM PPL numbers, and the paper explicitly argues LLM decode is
weight-load-bound where W4A16 already wins. **The honest failure mode:** the decode
kernel `gemv_oq4_grouped` doesn't quantize the activation at all, so there is no
quant pass to fuse the down-proj into — on decode the branch is a *separate* rank-r
f16 GEMV with no tensor-core to hide it = **added bandwidth tax on gfx1103/1151**.
**Verdict: right tool for the diffusion path (DiT prefill, compute-bound), wrong tool
for LLM decode.** Reserve it for Krea2/SeFi/Flux2 and LLM prefill; do not put it on
the decode path.

### Lever D — Block-clustered codebooks on the activation side (LO-BCQ)

**Upgraded from "context" — this is the strongest near-lossless-W4A4 datapoint.**
Weight-only codebooks (QTIP trellis, already implemented; AQLM/VPTQ) squeeze weight
bits but do nothing for activation outliers, so they can't enable A4. **LO-BCQ**
(arXiv 2502.05376, "Locally Optimal Block Clustered Quantization for W4A4") extends
codebooks to the **activation** side and is the only surveyed method reporting
genuinely **near-lossless W4A4 for LLMs** (not diffusion, not half-the-gap).

**Mechanism.** Replace the uniform int4 grid (`round(x/scale)` onto evenly spaced
levels) with **per-block non-uniform codebooks**: split each tensor into blocks,
cluster blocks by their statistics, and assign each cluster its own locally-optimal
16-entry codebook (≤16 codebooks total, ~0.19 KB, calibrated offline then frozen);
each scalar becomes a 4-bit index to its nearest entry. Applied to **both** weights
and activations. Training-free PTQ.

**Why it beats uniform int4 on the exact failure this investigation found.** A
codebook lookup is still discontinuous (bin boundaries exist), but a *locally
optimal* codebook places entries at the centroids of the real value density —
**dense where values are dense**. So when a near-boundary value flips to the
adjacent entry, the two entries are close and the reconstructed value barely moves;
a uniform grid whose scale is inflated by an outlier has far-apart levels, so a flip
is a big jump. This is precisely the amplification that produced the ≈0.63 e2e
divergence in Phase 0 (int4 step turning a tiny upstream delta into a large code
jump, compounded over 32 layers). LO-BCQ drives per-element error to near-zero *and*
makes each flip's output jump negligible — attacking the divergence at the
reconstruction-error root, not the discontinuity.

**Numbers.** WikiText-103 W4A4: Llama2-7B 5.06→5.19 (+0.13), Llama2-70B 3.14→3.23
(+0.09), GPT3-8B 7.38→7.48, GPT3-22B 6.54→6.62 — **<0.1 PPL loss**, vs 0.29–0.56 for
Atom/QuaRot at comparable bits. Caveats: this is at **4.56 effective bits** (codebook
+ selector overhead vs uniform 4.06), and ~0.4 bit of the edge is that extra spend;
and it gates on **absolute PPL vs FP16**, never batched-vs-per-token parity —
reconfirming the Phase 0b gate.

**hipfire mapping.** Offline codebook machinery partly exists: **Lloyd-Max MQ
(`mq4l`)** is exactly a per-group non-uniform 1D codebook, and QTIP is a trellis
codebook — so LO-BCQ-style *weight* codebooks are mostly a port. New = (a)
block-**clustering** (multiple codebooks + a per-block selector), (b) an **online,
batch-invariant activation** codebook quantizer (the Phase 0 probes show the current
uniform `quantize_act_oq4` is already batch-exact — a codebook version must preserve
that), and (c) **the GEMM that consumes codebook-indexed activations — the crux.**
hipfire's A4 throughput rests on `iu4·iu4` integer WMMA; a codebook index is not an
int4 integer. Two options, both bad for the throughput that motivates A4: dequantize
activation codes to int8/f16 before the WMMA (→ back to W4A8/W4A16, which already
ships coherent), or build a LUT-based codebook GEMM (AQLM/QuIP# regime — hard to make
fast on GPU; LO-BCQ gives **no kernel analysis**). Same shape as SVDQuant's fusion
risk: a real quality win gated on a hard RDNA-kernel question. Build: **M** (offline,
reusing `mq4l`/QTIP) → **L/research** (fast codebook-activation GEMM).

## 4. Evidence-derived ranking

Phase 0/0b changed the ranking's shape. Two facts dominate:

- **The int4-act kernels are correct and batch-exact** (Phase 0) — there is no bug
  and no "un-parking" to do.
- **Every quality lever erodes the int4 throughput that is A4's only reason to
  exist** (Phase 0b): Lever B keeps a subspace on the int8 GEMM; Lever C adds an f16
  low-rank branch; Lever D breaks `iu4·iu4` WMMA entirely (needs a LUT-GEMM or a
  dequant to W4A8/16). And the shipped batched-prefill default is *already*
  W4A8-MMQ, chosen as fastest — so int4-act may not even be faster than the
  incumbent on gfx1103/1151.

So the gating question is no longer quality — it is **throughput**, and it is cheap
to answer. Revised order:

0. **THROUGHPUT GATE — RUN + CORRECTED, 2026-07-30, gfx1103.** New probe
   `crates/hipfire-rdna/examples/bench_oq4_act_gemms.rs` (kernel-level, since
   `bench_qwen35_speed` runs this hybrid-DeltaNet model's prefill per-token).

   **First read (WRONG conclusion): "int4-act is slowest, NO-GO."** At N=512 the int4
   path (`gemm_oq4_grouped_act_batched`) was 1.4–2× slower than W4A8-MMQ. I wrongly
   inferred int4-act is *inherently* slower on RDNA3.

   **Correction (hardware fact via the AMD matrix calculator, gfx1100):** RDNA3 has
   a native 4-bit WMMA, and it is **2× the MAC rate of int8**:
   `v_wmma_i32_16x16x16_iu4` = 16 exec cycles (2048 ops/WGP/cycle) vs
   `v_wmma_i32_16x16x16_iu8` = 32 cycles (1024 ops/WGP/cycle). And
   `gemm_oq4_grouped_wmma` *does* use it (`__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32`).
   So int4-act has real **2× compute headroom** (plus half the activation bytes).

   **What the bench actually measured — an unoptimized kernel, not a hardware
   ceiling.** Isolating the pure iu4 GEMM (pre-quantized activation, no
   `quantize_act_oq4` launch) at N=512: qkv 0.465 ms vs MMQ 0.282 (**+65%**), gate
   0.881 vs 0.533 (+65%), o/down 0.277 vs 0.220 (+26%). The quantize launch is minor
   (~0.03 ms). So the iu4 GEMM *itself* runs ~1.65× slower than int8-MMQ **despite a
   2× hardware edge** — i.e. it realizes ~⅓ of the achievable, a **~3× optimization
   gap**. Cause: `gemm_oq4_grouped_wmma` is register-tiled with **no LDS staging, no
   K-unroll, no software pipelining** — exactly the optimizations MMQ received to hit
   its shipped numbers. It is memory/occupancy-bound, masking the WMMA advantage.

   **Corrected verdict: the gate is REOPENED, not failed.** int4-act's throughput is
   blocked on kernel work, not hardware.

   **ROOFLINE (measured, 2026-07-30) — the 2× does NOT fully translate; it's memory-
   capped to ~1.6×.** gfx1103: 12 CU / 6 WGP @ 2.7 GHz → iu4 peak 33.2 TOPS, iu8 16.6
   TOPS; **achievable DRAM BW = 71.5 GB/s** (measured, `bench_dram_bw.rs`; 80% of the
   89.6 GB/s LPDDR5-5600 theoretical). Roofline balance AI at 71.5 GB/s: iu4 = 464
   ops/B, iu8 = 232 ops/B. Prefill-GEMM AI (N=512) is ~360–395 ops/B — *between* the
   two — so **an optimal iu4 GEMM is MEMORY-bound while an optimal iu8 GEMM is
   COMPUTE-bound**:

   | shape (N=512) | iu4 roofline (mem) | iu8 roofline (compute) | iu4 win | output % of iu4 traffic |
   |---|---|---|---|---|
   | qkv M=1536 | 59 µs | 97 µs | **1.64×** | 74% |
   | o/down M=1024 | 41 µs | 65 µs | 1.59× | 72% |
   | gate M=3072 | 114 µs | 194 µs | 1.69× | 77% |

   The 2× WMMA edge collapses to **~1.6×** because iu4 hits the memory wall — set by
   the **f32 output write (72–77% of traffic), which is identical across all
   activation precisions**. So: (a) the realizable prefill win is ~1.6×, not 2×;
   (b) it is **prefill-only** — decode (B=1) is weight-bandwidth-bound and the weight
   is int4 in *both* paths, so int4-act gives decode **zero**; (c) the biggest lever
   is precision-independent: an **f16 or fused (no-DRAM-roundtrip) output** would speed
   up the shipped MMQ too *and* push iu4 back toward compute-bound where its full 2×
   lives. **Gate-proper = optimize `gemm_oq4_grouped_wmma`** (LDS-staged tiles +
   K2-unroll + pipelining; fuse the quant; f16/fused output) toward its ~1.6× memory
   roofline and re-bench. Current iu4 is 465 µs = ~8× off its 59 µs roofline (MMQ is
   282 µs = ~3× off its 97 µs), so beating MMQ is very reachable; the ceiling is ~1.6×,
   prefill-only. Standard kernel tuning — `hipfire-kernel-tuning` skill, MMQ history
   (866c55d6/55d9df52/e769832c). Estimated **M**.
### 4.0b — The output-memory lever (explored 2026-07-30): bf16 prefill output

The roofline (§4.0) says iu4's 2× is capped to ~1.6× **only because the f32 output
write is memory-bound**. Exploration finding: the *entire* qwen3.5 prefill activation
stream is **f32** — `x_batch`, `x_rot`, every GEMM `y` (oq4 *and* mq4's
`gemm_hfq4g256_wmma` both write `float* Y`), `silu_mul_f32`, the residual — so the
dominant output term is not oq4-specific.

- **Use bf16, not f16.** Repo note (`2026-07-08-dspark-wmma-training.md`): "plain f16
  dies on *range* (activations)". bf16 = f32 exponent range, 2 bytes; RDNA3 has native
  `v_wmma_f32_16x16x16_bf16`. The iu4 GEMM already accumulates in f32 (`facc`), so a
  bf16 write is just an epilogue cast.
- **Roofline with bf16 output** (qkv N=512): iu4 traffic 0.81+0.27+**1.57** = 2.65 MB
  → AI **607 ops/B** > the 464 balance → **iu4 flips to compute-bound → 48.5 µs → the
  full 2× over iu8 (97 µs) is restored.** Output-only bf16 suffices (output is 74% of
  traffic; the input-side x_rot f32→bf16 is a further ~6% and optional).
- **It also helps the incumbent.** mq4 and the W4A8-MMQ path write f32 output too;
  halving it speeds them in practice (their real kernels run 3× off roofline, i.e.
  not purely compute-bound). So this is a **format-independent prefill win**, valuable
  even if A4 is never pursued — which is why it goes *first*.
- **Scope (contained, not a full stream rewrite):** bf16 GEMM output (f32 accumulate →
  bf16 write) + consumers (attention, SwiGLU, residual) read bf16 + keep the residual
  *accumulation* and final logits in f32. Precision risk low (bf16 = f32 range);
  validate with KLD-vs-f32.
- **Next probe:** add a bf16-output variant to `gemm_oq4_grouped_wmma` (+ MMQ + mq4),
  re-run `bench_oq4_act_gemms`, confirm (a) iu4 → ~compute-bound 2×, (b) MMQ/mq4 also
  speed up. One contained kernel experiment — do this before Phase 0d's full iu4 tiling.

- **PROBE RESULT (2026-07-30, MEASURED — corrects the ordering above).** Built the
  bf16-output iu4 kernel (`gemm_oq4_grouped_wmma_bf16out.hip` + dispatch fn) and
  timed it vs f32-out. **bf16 output barely helps the *current* kernel**: qkv N=512
  −16%, everything else ~0%. Because the current iu4 kernel is **~8× off its roofline**
  (compute/occupancy-bound: `__launch_bounds__(32,8)` = 1 wavefront/block, register-
  tiled, no LDS staging) — so it is **not** memory-bound, and halving an output write
  that is only ~9% of its runtime does nothing. **The roofline f32→bf16 flip to 2× is
  LATENT: it requires the compute optimization FIRST.** Corrected ordering: **Phase 0d
  (optimize iu4 compute → near memory-roofline) comes first; bf16 output is the
  finishing move** that then converts the resulting memory-bound 1.6× into the full 2×.
  The bf16 kernel/dispatch are built and available but not yet wired to production.

1. **Lever A (rotation / ConQuR R₁)** — the only lever that **preserves full int4
   throughput** (weight-merged, zero runtime cost) and is all-reuse (`R1Plan`,
   `apply_r2`). Bounded ~0.3–0.5 PPL, but free — always worth doing if A4 is pursued.
2. **Lever B (mixed-precision A8 / ResQ)** — the primary *quality* lever: ~60–90% of
   the gap, foldable, reuses int4+int8 kernels summed. Costs throughput proportional
   to the A8 subspace size, so sweep ⅛→1/32 for the quality/throughput knee. The
   RDNA-friendly (PCA-folded, gather-free) form of the MixQ paper.
3. **Lever D (block-clustered codebooks / LO-BCQ)** — the **best quality** (<0.1 PPL,
   the only near-lossless-W4A4 LLM result) but gated on the hardest kernel question
   (a fast codebook-indexed-activation GEMM on RDNA; no published kernel). Pursue
   **only if** the throughput gate passes, Lever B proves insufficient, *and* a
   prototype LUT-GEMM clears the int4-WMMA bar. Offline side reuses `mq4l`/QTIP.
4. **Lever C (SVD low-rank)** — reserve for the **diffusion DiT** path where it is the
   right tool (compute-bound, fusible into the `_add` epilogue). For LLMs it's a
   decode-bandwidth tax; the offline quality probe is cheap enough to run
   opportunistically, but do not build the decode fusion.

If A4 is pursued, A and B **compose** (ResQ even prescribes a within-subspace
rotation): target recipe **oq4++ weights + ConQuR R₁/R₂ + an ⅛→1/32 A8 outlier
subspace**, with D held in reserve pending its kernel and C reserved for diffusion.

**The single most likely outcome, stated honestly:** the throughput gate shows
int4-act ≈ or < W4A8-MMQ on these APUs, so **W4A8 stays the LLM production format**
and the near-lossless-W4A4 machinery (LO-BCQ codebooks especially) lands where it
actually pays — the **diffusion DiT** path. The gate run decides this in one bench.

## 5. Costed build order

**Phase 0 — root-cause the divergence (DONE, 2026-07-30).** No oq4 kernel bug; the
divergence is fundamental int4-act sensitivity to non-oq4 batched-vs-per-token
deltas, and per-token is not a golden oracle. New probe
`crates/hipfire-rdna/examples/parity_oq4_rotation_paths.rs`.

**Phase 0b — packaged quality tooling can't see int4-act (DONE, 2026-07-30).** §2b:
KLD scoring is per-token W4A16; int4-act lives only in serving/bench. Working
estimate ~1.1 PPL (sim). A real number needs a batched-prefill logit scorer.

**Phase 0c — THE THROUGHPUT GATE (do next, cheap).** Per §4.0: bench int4-act vs
W4A8-MMQ vs W4A16-WMMA prefill on qwen3.5-0.8b via
`HIPFIRE_OQ4_PREFILL_ACT_BITS ∈ {4,8,16}` + `bench_qwen35_speed --prefill 512`. This
one run decides whether any A4 quality lever is worth building or whether W4A8 stays
production and the effort redirects to diffusion (Lever C/D). Do this before
Phases 1–4.

**Phase 1 — offline lever probe (S, CPU sim, no GPU).** Extend
`a4_quant::{a4_simquant,snr_db}` (+ `rotation_a4_snr_probe`) to score levers A/B/C
by **end-to-end output-SNR** on captured activations, ranking them before kernel
work. Requires one GPU activation-capture pass (daemon calibration) to get real
heavy-tailed activations — the 2026-06-21 Gaussian-from-Hessian sim *overstates*
rotation, so use real captures. This is the "empirical sim probe" deliverable and
de-risks Phases 2–3.

**Phase 2 — rotation (M, mostly quant-time).** ConQuR corner-Procrustes R₁ (+ ART
init) in `learn_rotation.rs` + calibration-engine online loop; deploy via `R1Plan` /
`apply_r2`; extend R1 merge beyond arch_id 0/1 to qwen3.5. Zero new kernels.

**Phase 3 — mixed-precision A8 (M, offline + one kernel).** ResQ PCA `U` fold
(extends AWQ fold) → contiguous A8 subspace → dual-branch int4+int8 GEMM (reuse
`gemm_oq4_residual_mmq` + iu8 path + `roughquant` sum template) → runtime mixed-bit
dispatch (the missing piece in `mixed_precision.rs`). Sweep subspace size ⅛→1/32 for
the quality/throughput knee.

**Phase 4 — SVD low-rank for diffusion (S offline → M prefill).** Offline
smooth+SVD-split quality probe on an LLM and on Krea2 DiT; if it pays, fuse the
rank-r add into the MMQ/WMMA `_add` epilogue for **prefill + DiT** only. Never
decode.

## 6. Eval methodology

Per the house rules (`opqplus-activation-quality` precedent, astrea/hipfire-eval):
KLD-vs-bf16 primary (≥16 chunks; top-K KLD is noisy), ppl ctx=2048 (not 512),
`coherence-gate.sh` on the winner (watch the list-primes attractor that plain oq4
hit). First on qwen3.5-0.8b (has the full recipe front-end + a bf16 ref), then 9B/27B
confirmation. Every A4 recipe compared against **two incumbents**: the current
`oq4++` (W4A16-decode / W4A8-prefill) and `mq4+`. Because A4's motivation is
throughput, **each quality number is paired with a bench** (`bench_qwen35_speed
--prefill 512`) — a lever that recovers quality but loses the int4-act throughput
(e.g. too large an A8 subspace) is a no-op vs shipping W4A8.

## 7. Risks / open questions

- **Phase 0 may not fully close the divergence.** The logit-diff has two sources
  (rotation non-determinism + int4-act nonlinearity amplification). Bit-identical
  rotation should fix the first; if argmax still flips, the int4-act step itself is
  the limit and A4 needs Lever B to be viable at all.
- **The A4 throughput win may be small at hipfire's scales.** On gfx1103/1151 UMA,
  decode is weight-bound (A4 ≈ A8) and prefill batches are modest; confirm the int4
  GEMM is actually faster than the shipped int8-MMQ before investing in quality
  recovery. If not, "oq4++ w4a4" resolves to *keep shipping W4A8* and the real prize
  is the diffusion DiT path (Lever C).
- **Rotation coverage.** `R1Plan` is dense-llama-only; qwen3.5/GQA/MoE merge is
  unbuilt and is real cost in Phase 2.
- **Real-activation sim.** No captured qwen3.5-0.8b activations or current full
  Hessian are on disk (`~/.hipfire/hessians/` has only MiniCPM5); Phase 1 needs a
  capture pass first.

## 8. Decision this produces — REOPENED (2026-07-30)

> **RESOLVED 2026-07-31 → see §9.** The A3 throughput gate was run: the optimized iu4
> GEMM **beats** MMQ (~1.5× full path / ~1.8× GEMM). The "single most likely outcome"
> below (int4-act ≈/< MMQ ⇒ W4A8 stays) is **refuted**. A4 is unlocked.

Earlier draft called this a NO-GO on the strength of the first throughput read; that
was **wrong** (§4.0 correction). RDNA3's iu4 WMMA is 2× the int8 rate, and the
current int4 GEMM simply doesn't realize it (~3× optimization gap: register-tiled, no
LDS/pipeline). So the real state is:

- **int4-act is throughput-viable in principle** on RDNA3 — ~2× compute headroom over
  int8-MMQ, currently unrealized by an unoptimized kernel. The gate now hinges on a
  **kernel-tuning question**, not a hardware verdict.
- **The next concrete step is Phase 0d: optimize `gemm_oq4_grouped_wmma`** (port MMQ's
  LDS-staging / K2-unroll / software-pipelining; fuse the activation quant) and
  re-run `bench_oq4_act_gemms`. Decision rule: if the optimized iu4 GEMM **beats
  MMQ**, A4 is throughput-positive → build the quality levers (A rotation, then B
  ResQ subspace) to close the ~1.1 PPL int4-act gap, and `oq4++ w4a4` becomes a real
  serving format. If, after MMQ-class tuning, it **still can't beat MMQ**, *then* the
  NO-GO stands and low-bit-activation work redirects to diffusion.
- **Meanwhile LLM production stays W4A8** (`oq4++` weights, shipped, KLD 0.046) — that
  is unchanged regardless of how the gate resolves; it is the safe incumbent while the
  iu4 kernel is tuned.
- **Diffusion DiT — profiled 2026-07-31 (Krea2-Turbo, 512², 20 steps).** Confirms the
  compute-bound premise **and** the same kernel-efficiency wall as the LLM:
  - Warm step is **~90% GEMM** (attn 9%, prep 1%) — compute-bound, so low-bit acts
    would pay the **full** 2× (int4) here, *not* memory-capped.
  - But the bf16 GEMM runs at **1.39 TFLOP/s = 8.4% of the ~16.6 TFLOP/s bf16-WMMA
    peak** — the kernel is the bottleneck, not the format (same story as the iu4 GEMM
    at ~⅓ of peak). An efficient bf16 kernel alone is a ~6–10× lever; low-bit acts are
    a 2× multiplier *on top*.
  - **Cold first step = 344 s streaming 22.64 GiB of weights** → quant's clearest
    immediate win is **footprint/cold-load** (q4/q8 shrink the stream), not warm-GEMM
    speed — matching [[krea2-bf16-hotpath-profile]].
  - **Quality side is BLOCKED:** Krea2-Turbo renders a **solid red field** (not the
    prompt) — the [[krea2-dit-noise-bug]] — and no runnable Flux2 `.hfq` is on disk
    (only a calib). So low-bit-activation *quality* validation on the DiT can't proceed
    until a working DiT exists (fix Krea2 or obtain/convert a Flux2 artifact).
  - **Net:** diffusion is the right regime for low-bit acts (compute-bound, full 2×),
    but the binding lever is an efficient DiT GEMM (bf16 first), then low-bit as the
    multiplier; and the quality path needs a working DiT first. Diffusion low-bit-act
    ablation hooks already exist (`HIPFIRE_DIFFUSION_W4A8`, `ABLATE_BITS`).
- **Cross-arch:** also re-run the probe on gfx1201 (RDNA4, medusa); its iu4-vs-iu8
  ratio may differ and it has a discrete GPU (not UMA), changing the bandwidth picture.

## 9. Stream A EXECUTED — A3 THROUGHPUT GATE **PASSED** (2026-07-31, gfx1103)

Phase 0d/§4.0's "optimize `gemm_oq4_grouped_wmma` to MMQ-class and re-bench" was
executed. **The optimized iu4 GEMM beats W4A8-MMQ at every prefill shape.** The
single-most-likely outcome §4 hedged for (int4-act ≈ or < MMQ ⇒ "W4A8 stays") is
**REFUTED**: int4-act is throughput-positive on RDNA3.

**The kernel (new): `kernels/src/gemm_oq4_grouped_wmma_lds.hip`** — a bit-exact,
LDS-staged optimization of `gemm_oq4_grouped_wmma`. It merges the proven tiling of
`gemm_iu4_i32_wmma_lds` (which was ungrouped/int32-out) with the OQ4 per-256-group
f32 rescale. Not a clone of the wave64 template — a **wave32** rewrite matching MMQ +
the §4.0 `_w32` roofline + RDNA2/4 portability. Structure:
- 8 wave32 waves/block computing a **BM=64 × BN=128** block tile (vs the original's
  ONE wave / 16×16 tile with zero reuse — the redundant-DRAM cause of ~8×-off-roofline).
- **WMt=2 × WNt=2 register super-tiling** → 4 independent WMMA accumulator chains,
  breaking the original's single serial `iacc` dependency.
- **BK=64 K-strips in double-buffered LDS** with a register-staged global→reg prefetch
  of the next strip during compute (RDNA has no cp.async).
- Per-group i32→f32 rescale flushed at group boundaries in ascending-g order with the
  identical `(iacc*sw)*sx` arithmetic ⇒ **bit-exact** vs the original.
- Two entry points: `gemm_oq4_grouped_wmma_lds` (f32 out) + `..._bf16out` (bf16 out).
  Dispatch: `gemm_misc.rs` `gemm_oq4_grouped_wmma_lds[_bf16out]` (grid M/64×B/128,
  block 256). Registered `GEMM_OQ4_GROUPED_WMMA_LDS_SRC`.

**Correctness — bit-exact (`parity_gemm_oq4_grouped_wmma_lds`, new).** LDS-f32 vs the
original `gemm_oq4_grouped_wmma`: `max_abs = 0.000000` on all 7 shapes incl. unaligned
M/B (M=1000/B=100, M=17/B=5, M=64/B=129 — bounds clamps verified). Because it is
bit-exact to a path already shown coherent end-to-end (§2c `HIPFIRE_OQ4_ACT4=1`), it
carries **zero incremental quality risk**.

**Throughput (`bench_oq4_act_gemms`, extended). Achievable DRAM BW re-confirmed 69.3
GB/s.** Adversarial-fair note: MMQ's `ensure_q8_1_mmq_x` hard-codes `must_convert=true`,
so MMQ's timed call **re-quantizes the activation every iteration**; the pure-GEMM iu4
number used a pre-quantized activation. So both a GEMM-only and a **full-path**
(quantize + GEMM, matching MMQ) number are reported:

| shape (N=512) | iu4-lds GEMM | iu4-lds FULL (q+gemm) | MMQ (q+gemm) | lds-GEMM vs MMQ | lds-FULL vs MMQ |
|---|---|---|---|---|---|
| qkv  M=1536 | 0.152–0.159 | 0.191 | 0.274–0.280 | **−43…−44%** | **−32%** |
| o/down M=1024 | 0.118 | 0.150 | 0.220–0.221 | **−46…−47%** | **−32%** |
| gate M=3072 | 0.306–0.308 | 0.341 | 0.537–0.542 | **−43%** | **−36%** |

So on the fair full path iu4-lds is **~1.5× MMQ**; GEMM-alone **~1.8×**. Vs the
unoptimized baseline this kernel is **~3×** (qkv N=512 0.470→0.152). Roofline: lds/f32
qkv N=512 = 152 µs vs the 59 µs mem-roofline = **2.6× off** (MMQ is 2.8× off its 97 µs
roofline), i.e. **MMQ-class-or-better efficiency AND 1.8× absolute**. bf16-out now ≈
f32-out (±5%), i.e. the kernel is no longer output-memory-bound — the §4.0b "latent
bf16 flip" is realized as *balanced*, not a further large win, at this tile size; the
remaining ~2.6× to roofline is headroom (larger N-tile / deeper pipeline / K2x32).
N=128 shows small-N run-to-run jitter (short kernels); the decision rests on the clean,
consistent N=512 prefill shapes.

**Independently verified (2026-07-31):** a separate adversarial agent re-ran parity
(all 7 shapes `max_abs=0`, incl. B=129 crossing the BN=128 tile) and the bench 3×
(margins stable cold-run vs hot-run ⇒ no thermal-ordering bias), audited the
bit-exactness/scale-index/double-buffer-sync math line-by-line, and confirmed the
quantize-inclusive fairness of `lds-FULL` vs MMQ. **All three claims CONFIRMED, zero
bugs.** It re-flagged the honest read: "~1.8×" is GEMM-only (MMQ's number carries a
quantize the iu4-GEMM number omits); the fair figure is **~1.5×** (full path). Both beat
MMQ decisively. Untested-but-assert-covered: group=128 / K=2048 (production only uses
group=256), and gfx1201/RDNA4 (portable by construction — identical WMMA builtin — not
yet measured).

**A3 VERDICT: PASS.** Per §8's decision rule, A4 is unlocked (wire int4-act into
batched serving prefill via the LDS kernel) and the quality levers (A rotation, B ResQ)
become worth building to close the ~1.1-PPL int4-act gap. The safe incumbent for LLM
production remains W4A8 until A4 lands + a KLD gate passes; the LDS kernel is also a
drop-in bit-exact ~1.8× speedup for the existing `HIPFIRE_OQ4_ACT4` prefill path.
Portability: iu4 `_w32` WMMA is RDNA3/3.5/4; the kernel JIT-compiles only when
dispatched (gated by `has_wmma_w32`), same as the original — no RDNA2 regression
(RDNA2 has no iu4 WMMA and never dispatches it). gfx1201 re-bench still pending (medusa).

**gfx1201/RDNA4 — matrix-calculator verdict (2026-07-31, no GPU): the A3 win does NOT
transfer with the current kernel.** Per the vendored AMD matrix calculator:
`v_wmma_i32_16x16x16_iu4` and `..._iu8` on gfx1201 are **both 8 cyc / 4096 ops-WGP-cyc =
EQUAL** — RDNA4 sped iu8 up to iu4's rate, so the **2× iu4-over-iu8 edge RDNA3 has (16 vs
32 cyc) is GONE at the 16×16×16 shape** the A3 kernel uses. The iu4 advantage moved to the
**new `v_wmma_i32_16x16x32_iu4`** (K=32, 8 cyc / **8192** ops-WGP-cyc = 2×). Consequence:
`gemm_oq4_grouped_wmma_lds` (which emits `iu4_16x16x16_w32`) would run **≈ MMQ, not 1.5×,
on RDNA4** — the A3 throughput win is **RDNA3/3.5-specific**. To recover it on gfx1201 the
kernel needs a K=32 iu4 variant (new builtin, K-doubled tiling) — an arch-specific port,
tracked for when the effort targets RDNA4. (Bandwidth is separate: medusa's gfx1201 is a
discrete GPU, not UMA, which also shifts the roofline.)

### 9a. Stream A4 — routing map + what shipped vs what the quality gate blocks

**Shipped (bit-exact, safe, default-on).** `gemm_oq4_grouped_act_batched`
(`quant.rs`) — the int4-act prefill wrapper (quantize_act_oq4 + grouped GEMM) — now
routes to `gemm_oq4_grouped_wmma_lds` for `n >= 128`, keeping the original for
decode/small batches. Because the LDS kernel is bit-exact, this is a pure speedup with
**zero numeric change**, so it needs no coherence re-run. It also transitively
accelerates `gemm_oq4_grouped_residual_act_batched` (o/down-proj), which calls the
wrapper internally. Threshold 128 = the benched full-BN-tile-utilization region;
lowering to 32/64 is a follow-up pending a small-N bench.

**The routing map (qwen3.5 `forward_prefill_chunk`).** oq4 prefill is *already mixed
precision*:
- **QKV** (`prefill_chunk.rs:3901`): `n>=64 → gemm_oq4_qkv_mmq` (**W4A8**); `n<64 →
  gemm_oq4_grouped_act_batched` (**W4A4**).
- **gate_up**: same MMQ-vs-int4 split.
- **o / down** (`prefill_chunk.rs:3073,3569,5014,5427`):
  `gemm_oq4_grouped_residual_act_batched` = **W4A4 already** (now LDS-accelerated).

So the shipped default already runs *W4A4 on o/down + W4A8 on qkv/gate_up*. **The
remaining A4 = promote qkv/gate_up from W4A8-MMQ to W4A4-int4-LDS** at `n>=64`. Now
that A3 shows int4-LDS beats MMQ 1.5×, that swap is a **throughput win** — but it is a
**W4A8→W4A4 quality change, NOT bit-exact**, so it must clear the KLD-vs-bf16 gate
before becoming default (it is exactly the ~1.1-PPL int4-act penalty this whole plan
is about; Stream B levers exist to close it if short).

**Two real blockers to a *measured* A4 (both pre-existing, restated precisely):**
1. **No int4-act prefill quality scorer** (§2b): the KLD path is per-token W4A16 and
   never runs batched int4-act. A KLD-vs-bf16 harness that drives `forward_prefill_chunk`
   with qkv/gate_up forced to int4-act must be built before the promotion can be graded.
2. **Daemon serving-routing gap** (§2c): for the qwen3.5 *hybrid* model the daemon
   session-batch path rejects Oq4G256 and falls back to per-token W4A16 serial — so
   `forward_prefill_chunk`'s batched int4-act is not even reached by daemon serving on
   this model. A dense Oq4G256 arch (gemma3/qwen2, which route prefill through
   `weight_gemm`'s act4 arm) *does* reach it and is the cleaner first A4 serving target.

**A4 status: the fast kernel is wired into every reachable int4-act prefill path
(bit-exact); promoting the MMQ projections to int4-act is a one-site-per-projection
change gated on building the batched-prefill KLD harness — that harness + the gate is
the next concrete unit (feeds Stream B).**

### 9b. A4 KLD harness — plumbing mapped; the qkv gate landed; the scorer is a REAL new build (2026-07-31)

A code survey settled how far this is from a running act4-vs-act16 KLD number:

- **DONE — the qkv int4-act gate** (`prefill_chunk.rs:3900`, commit follows §9a): qkv@n>=64
  was the ONLY non-W4A4 site in qwen35 oq4 prefill (gate_up/o/down are already W4A4 via
  `FusedGateUpOq4G256` / `gemm_oq4_grouped_residual_act_batched`). `HIPFIRE_OQ4_PREFILL_
  ACT_BITS=4` now forces true W4A4 qkv even at n>=64. Default unset = W4A8-MMQ (unchanged).
  So a **fully-W4A4 qwen35 prefill is now reachable under one env var.**
- **BLOCKER — no batched-prefill scorer exists.** Every KLD entry (`kld_eval.rs` generic
  L38-64; `qwen35/loading.rs` `forward_chunk_scored` L3183, `Qwen35KldForward` L3217)
  runs a **per-token decode loop** through `forward_scratch` → `weight_gemv*` = **W4A16
  only**. qwen35 never calls `weight_gemm` (so its `HIPFIRE_OQ4_PREFILL_ACT_BITS` handling
  is irrelevant here), and the `KldConfig`/`ScoringMode::Prefill`/`kld_graph_prefill`
  surface is declared but **unconsumed**. No arch does batched KLD scoring. So measuring
  the int4-act penalty needs a **new qwen35 teacher-forced batched-prefill logit scorer**
  (fresh KV+DeltaNet per chunk, lm-head fan-out per scored position) — a real build, not
  a flag flip.
- **Design fork (needs a call):**
  1. *Native scorer* — new `ChunkScoredForward` that drives `forward_prefill_chunk`
     (byte-identical to serving), act4 via the new gate. Cost: also needs a W4A16 qkv/gate_up
     path for the act16 baseline (the fused gate_up kernel is W4A4-only), so more kernel plumbing.
  2. *`weight_gemm` reference-floor* — route the new batched forward's linears through
     `weights::weight_gemm`, then `HIPFIRE_OQ4_PREFILL_ACT_BITS=16` vs `=4` gives both passes
     for free (no new kernels). Cheaper, but a reference floor (weight_gemm kernels ≠ serving's
     native dispatch), so the KLD is indicative, not serving-exact.

  Recommendation: option 2 first (fast, no new kernels) for the go/no-go quality number, then
  option 1 only if it passes and a serving-exact figure is wanted. **Left for the user to steer
  before authoring the scorer** — it's a substantial, quality-sensitive harness, not autonomous-safe.

### 9c. A4 KLD harness BUILT + RUN — the int4-act penalty is measured (2026-07-31)

User chose the **native** (serving-exact) scorer. Built + validated + ran. Scope was ~2×
the survey estimate: `forward_prefill_chunk` has BOTH a dense-session branch and an FA
branch; the dense path was covered (qkv/gate_up/o/down), and the run validated that the
dense path is the one taken (see below).

**What was built** (all env-gated `HIPFIRE_OQ4_PREFILL_ACT_BITS=16`, default-off ⇒ serving
unchanged):
- `gemm_oq4_grouped_residual_f16_batched` (`quant.rs`) — W4A16 residual variant for o/down.
- act16 branches at the 4 dense oq4 prefill sites (`prefill_chunk.rs`: qkv, gate_up unfused
  to 2× f16-WMMA, o, down).
- `perplexity_batched.rs` (new example) — runs ONE `forward_prefill_batch_with_pbs_opts`
  with `per_token_hidden_out`, then fans out the lm-head (per-row `weight_gemv`, decode-
  precision so the KLD isolates the *body*) over the scored window, into the same
  `hipfire_kld` scoring as `perplexity.rs`. Default **q8 KV** (f32 → silent per-token
  fallback).

**Result (gfx1103, qwen3.5-0.8b--oq4++, calib-1m, ctx=512, 503 scored positions):**

| measurement | KLD/tok | PPL | meaning |
|---|---|---|---|
| per-token W4A16 (ref) | — | 13.29 | existing decode-path scorer |
| **batched act16 vs per-token-W4A16** | **0.0174** | 13.64 | **noise floor** (batched-vs-per-token); VALIDATES act16 took effect + dense path covered + `final=true` eligible (Q8 KV, not the fallback) |
| **batched act4 (W4A4) vs act16** | **0.0668** | 14.43 | **the int4-activation prefill penalty** |

So **int4-act costs ≈0.067 KLD / +0.79 PPL over W4A16** — ~4× the 0.017 batched-vs-per-token
noise floor, i.e. a **real, non-trivial degradation** consistent with the survey's ~1.1-PPL
estimate. The validation KLD of 0.017 (vs ~0.63 if a site were missed) is the proof the
measurement is sound.

**ctx=2048 confirmation (2039 scored positions, 4× the data):** noise floor **0.0160**,
int4-act penalty **0.0672 KLD**, PPL 10.67 → 11.30 (**+0.63 PPL**). The penalty is
byte-stable across ctx (0.0668 → 0.0672), so it is **robust, not a small-sample artifact** —
~4.2× the noise floor. Headline number: **int4-act prefill penalty ≈ 0.067 KLD / +0.6 PPL.**

## 9d. Stream B — first lever landed: activation clip-search (2026-07-31)

The int4-act penalty comes from `quantize_act_oq4`: scale = **absmax/7**, so one
outlier in a 256-group inflates the scale and the bulk quantizes coarsely — and hipfire
applies clip/AWQ to *weights* but **nothing on the activation side**. First Stream B lever
= the plan's Lever A "+" (clip) on activations: `quantize_act_oq4_clip.hip` grid-searches a
per-group clip ratio α∈{1.0…0.43} and picks the scale minimizing the group's int4
reconstruction MSE (α=1.0 is in the set, so it can never beat plain absmax). Same output
format ⇒ the GEMM is unchanged; register-cached so the search re-uses the group with **no
extra DRAM traffic** vs plain absmax (throughput-neutral, not separately benched). Gated by
`HIPFIRE_OQ4_ACT_CLIP=1`, default-off.

**Result (A/B on the §9c harness, qwen3.5-0.8b--oq4++, ctx=512):**

| act4 variant vs act16 | KLD/tok | PPL |
|---|---|---|
| plain absmax | 0.0668 | 14.43 (+0.79 vs act16) |
| **+ clip-search** | **0.0586 (−12%)** | **13.92 (+0.28 vs act16)** |

So the activation clip recovers **~12% of the KLD penalty and ~65% of the PPL penalty**,
cheaply and throughput-neutrally — validating that activation-side clip (untested in hipfire
before) helps, exactly as Lever A predicts. **ctx=2048 confirmation (2039 pos):** act4 plain
0.0672 → clip **0.0553 KLD (−18%)**, PPL +0.63 → **+0.33 (−48%)** — the win is robust and
slightly *stronger* at scale. It does NOT close the gap to the 0.016 noise
floor alone (KLD 0.059 still ~3.7× floor), so it composes with the stronger, unbuilt levers:
**ConQuR learned R₁/R₂ rotation** (Lever A proper — needs the quant-time objective +
R1Plan→qwen3.5 extension) and **ResQ ⅛→1/32 A8 outlier subspace** (Lever B, the strongest —
needs the PCA fold + dual int4+int8 GEMM). Recipe trending toward **oq4++ weights + activation
clip + (rotation and/or A8 subspace)**, each now measurable on this harness. Note KLD (top-128
distributional) moved less than PPL (true-token NLL) — the clip helps the true continuation
more than the full tail, so the KLD gate stays the stricter bar.

## 9e. The A4 go/no-go RESOLVES — measure vs the real incumbent, not vs A16 (2026-07-31)

An ideation pass surfaced the decisive correction: the 0.067 penalty is act4 **vs A16**, but
the batched-prefill path never runs all-A16. gate_up/o/down are **already W4A4** (§9b) — qkv
is the *only* non-W4A4 site. So the right baseline is the current mix, measured for free with
the env unset (qkv-W4A8-MMQ + gate_up/o/down-W4A4):

| batched-prefill variant (ctx=2048, 2039 pos, vs A16 ref) | KLD | PPL | vs A16 |
|---|---|---|---|
| A16 (full-precision act) | — | 10.67 | ref |
| **current batched default** (qkv-W4A8 + rest-W4A4) | **0.0585** | 11.07 | +0.40 |
| full W4A4 (act4) | 0.0672 | 11.30 | +0.63 |
| **full W4A4 + clip** | **0.0553** | 10.99 | +0.33 |

**The reframe:** ~87% of the "int4-act penalty" is **already paid** by the shipped path
(gate_up/o/down are W4A4). Promoting qkv W4A8→W4A4 (full A4) adds only **+0.0087 KLD / +0.23
PPL** — near the 0.016 noise floor. And with the clip lever, **full-W4A4+clip (0.0553 KLD /
11.07 PPL) is *better* than today's mix (0.0585 / 11.07) on BOTH metrics** — while also being
**faster** (int4 qkv via the A3 LDS kernel + int4 activation throughput). So:

> **A4 GO/NO-GO → GO on QUALITY.** full W4A4 + activation clip is **better quality than the
> current batched-prefill default** (−0.003 KLD, −0.08 PPL; §9e confirmation below). The
> expensive Stream-B levers (ConQuR rotation, ResQ A8 subspace) are **NOT required** — clip
> alone clears the bar. **Throughput correction (§9f):** clip is NOT free — it ~2× the quantize
> kernel, which offsets the qkv int4-GEMM gain, so W4A4+clip vs the current mix is a **quality
> win at roughly neutral throughput**, not the "faster AND better" first claimed. (Plain W4A4
> without clip IS faster but is a slight quality *regression* vs the mix — 0.073 > 0.067 — so
> clip is what makes the promotion a quality win.) Cheapening the clip (fewer α / closed-form /
> de-dup the redundant qkv quantizes) would restore a clean speed win — a follow-up.

Remaining before flipping the default: house-rule confirmation (≥16 chunks / ctx=2048 done
as 1; coherence-gate on the winner watching the list-primes attractor; 9B/27B confirmation),
a clip-throughput bench (expected neutral), and the daemon serving-routing question (§2c —
whether the hybrid actually reaches this batched path vs per-token W4A16). But the *quality*
verdict is settled: **int4-act + clip ships.**

**≥16-chunk confirmation (2026-07-31, 16 independent ctx=512 windows, 8048 pos):**

| variant (vs A16) | KLD/tok | PPL | per-chunk KLD min/max |
|---|---|---|---|
| current mix (qkv-A8 + rest-A4) | 0.0667 | 24.15 | 0.054 / 0.084 |
| **full W4A4 + clip** | **0.0625** | **23.63** | 0.052 / 0.082 |
| full W4A4 plain | 0.0731 | 24.43 | 0.060 / 0.089 |

**W4A4+clip beats the current shipped mix on BOTH KLD (0.0625<0.0667) and PPL
(23.63<24.15), and per-chunk clip ≤ mix across the whole min/max spread** — the ship claim
holds at house-rule scale, not just as an aggregate. (Absolute KLD is higher than the single
ctx=2048 window's 0.0585 because these 16 windows span harder/more-varied corpus text — PPL
23 vs 10.7; the clip<mix<plain *ordering* is the robust, corpus-independent result.) `perplexity_batched`
now takes `--chunks N` (fresh KV+DeltaNet per chunk, matching the daemon). Confirmation
item 1/4 DONE.

### 9f. Confirmation battery results (2026-07-31)

1. **≥16-chunk KLD — PASS** (above): W4A4+clip 0.0625 beats current mix 0.0667 on KLD + PPL,
   per-chunk clip ≤ mix across the spread.
2. **Coherence-gate — PASS.** `hipfire chat` W4A4+clip (`HIPFIRE_OQ4_ACT4=1 HIPFIRE_OQ4_ACT_CLIP=1`),
   greedy: factual → "…Paris is the capital of…"; reasoning → "60/1.5 = 40, because dividing by
   1.5 is multiply by 2/…" — both correct, no list-primes attractor, no gibberish. (`hipfire chat`
   self-locks the GPU — run it BARE, not under `hipfire lock run`, or it FATALs on a double-lock.)
3. **Clip-throughput bench — clip is NOT neutral (corrects §9d/§9e).** Isolated `quantize_act_oq4`
   timing (`bench_oq4_act_gemms`, `HIPFIRE_OQ4_ACT_CLIP` off vs on), N=512: qkv 0.052→0.114 ms
   (+118%), o/down 0.050→0.103 (+104%), gate 0.050→0.102 (+102%). Register-cache killed the 9×
   DRAM re-read, but the 9-α search's ALU (per-α wave-reduce) still ~2× the quantize kernel.
   Quantize is ~25% of a projection GEMM (and W4A4 qkv runs it 3× redundantly), so full-path
   overhead ~10–30%, offsetting the qkv int4-GEMM gain → **W4A4+clip is a quality win at ~neutral
   throughput, not "faster AND better."** Follow-up: fewer α / closed-form clip / de-dup qkv quantizes.
4. **9B/27B — OWED (artifact-blocked).** No clean qwen3.5-9B/27B `oq4++` on disk (only 0.8b oq4
   + non-3.5 bf16 + confounded 2B/4B triattn.oq4.25); needs a quant pass. Harness is ready.

**Battery verdict: quality LOCKED (16-chunk + coherence both PASS) — ship W4A4+clip as a quality
improvement; throughput ~neutral (clip overhead ≈ qkv gain) pending a clip speed-up; 9B owed.**

### 9g. Clip α-grid tuning — conservative 4-α is strictly better (2026-07-31)

Cheapening the clip (the §9f throughput follow-up). The per-α wave-reduce dominates the kernel,
so fewer α ≈ proportionally cheaper — but α count is quality-fragile:
- **wide 4-α {1.0,0.85,0.70,0.55}: FAILED.** KLD **0.0707 > plain-no-clip 0.0668** — worse than
  no clip, even though α=1.0 is in the set (per-group MSE can't lose). Proof that **per-group MSE
  ≠ end-to-end KLD**: a coarse grid forces some groups onto an over-aggressive clip that trims a
  signal-carrying outlier. The fine search's *resolution in the moderate region* is load-bearing.
- **conservative 4-α {1.0,0.93,0.86,0.79}: WINS.** ctx=512 KLD **0.0589** ≈ 9-α 0.0586; 16-chunk
  **0.0621** ≈ 9-α 0.0625 (both beat mix 0.0667), PPL 23.72. Full quality gain preserved by
  sampling only the moderate-trim region [0.79,1.0] where the real gain lives — and it **avoids
  the aggressive clips entirely**. Throughput: quantize **0.080 ms** vs 9-α's 0.114 (plain 0.052)
  — **~30% less clip overhead at no quality cost.** This is the ship config (committed).

Net: W4A4 + conservative-4-α clip = the same quality win over the current mix, now at ~half the
clip overhead — closer to throughput-neutral (still +54% quantize vs plain; a closed-form or
3-α clip could close the rest, but the knee is here). Lesson: clip quality is fragile to the
α grid; keep α in the moderate region, never let a coarse grid reach aggressive clips.

**A4 decision:** int4-act buys ~1.5× prefill throughput (§9) at ~0.067 KLD / +0.79 PPL.
That is a genuine **throughput-vs-quality tradeoff**, NOT free — so promoting qwen3.5
qkv/gate_up to W4A4-by-default is **not justified on quality alone**; it needs Stream B to
close the gap first. **This is the go/no-go the whole plan was gating on: A3 throughput YES,
A4 quality has a real cost → the recipe becomes "int4-act + a Stream-B lever," not naked
W4A4.** Refinements available: ctx=2048 + ≥16 chunks for a house-rule-grade number; the FA
branch would need its 3 sites edited too if a model routes there (this run confirmed the
0.8b uses the dense path). Then Stream B (ConQuR R₁/R₂; ResQ ⅛→1/32 A8 subspace) measured
on this exact harness.

## 10. Stream C EXECUTED — the VAE is EXONERATED; the grain is a DiT-latent problem (2026-07-31)

The Stream-C premise ("rainbow-speckle **VAE-decode** grain, VAE bug #2 OPEN, in the
shared CPU `wan.decode`, localized to `up_block0_resnets`") **does not survive scrutiny.**
Both the cheap config check (C1) and a golden-reference code audit (C2) exonerate the VAE.

**C1 — config regression RULED OUT.** The embedded `Krea-2-Turbo.source.hfq`
`vae/config.json` is **md5-identical** (11e12ac4a7d2c342bc302f7285c6cfc8) to the golden
krea checkpoint's. Corrected sub-finding: `AutoencoderKLQwenImage` decodes through
`WanImageDecoder` with **RMSNorm** (fixed eps 1e-6), not GroupNorm — the SD
`GroupNorm`/`ResnetBlock2D` path (`vae.rs:845`, `unwrap_or(32)`) is bypassed, so the
"GroupNorm groups=32" mechanism the plan floated is **architecturally impossible** here.

**C2 — decode CODE audited against golden, VAE CORRECT.** Checked the on-disk golden
reference (`~/.venv/.../diffusers/models/autoencoders/autoencoder_kl_qwenimage.py`):
- The #1 suspect (`wan_causal_conv2d` temporal-tap collapse, `vae.rs:1164`) is **the
  CORRECT single-frame convention**: `QwenImageCausalConv3d` zero-pads the temporal axis
  (`F.pad` constant) with `cache_x=None` for a `num_frame=1` still image ⇒ only the last
  tap survives. The hypothesized "sum all taps (first-frame replication)" fix would be a
  **regression** (~3× over-weighting every conv). `wan_rms_norm_nchw` and `conv2d_nchw`
  layouts also match golden.
- **Decisive quantitative exoneration:** a real-image **encode→decode round-trip gives
  MSE = 1e-4** in [-1,1] (`wan_qwen_image_vae_encode_decode_round_trips`). A decoder that
  injected chromatic speckle could not round-trip at 1e-4. (The test's `KNOWN GAP ~MSE
  0.77` comment at `vae.rs:345` is **stale** — that bug is already fixed.)
- The `up_block0_resnets` `|dx|` jump (0.31) that the stage-bisect flagged is present
  **even in provably-clean decodes** (0.148 on the 1e-4 round-trip) and always resolves
  to a smooth `conv_out`. It reflects latent structure at the first-upsample region — it
  is **not** a grain indicator. The stage-bisect localization was a misread.

**Conclusion:** the grain is in the **DiT latent** — the "coherent" Krea-2 render is
low-frequency-coherent but carries high-frequency per-channel content that a faithful
VAE renders as chroma speckle. Stream C's fix target therefore **moves from the VAE to
the DiT / sampling** (connects to the `krea2-dit-noise-bug` thread). **One decisive
GPU experiment still owed** to convert "very likely" → "confirmed": `HIPFIRE_DUMP_LATENT`
on a real Krea-2 render, then feed that exact latent to the CPU decode via
`HIPFIRE_TEST_LATENT` + `HIPFIRE_TEST_SAVE_PNG` — if the grain is already in the latent's
high-freq content, the VAE is confirmed exonerated and the fix is upstream. **Do not
"fix" `wan_causal_conv2d`.** This unblocks nothing on the VAE side and re-points Stream D3
(diffusion quant quality) at needing a grain-free DiT first (still the binding blocker),
while the D1/D2 levers (efficient DiT GEMM; footprint/cold-load quant) are independent of
the grain and can proceed.

**C-confirm render (2026-07-31): attempted, CONFOUNDED — reported honestly, does not
change the conclusion.** Ran `hipfire diffusion txt2img` on `Krea-2-Turbo.source.hfq`
(seed 42, 8 steps, 512², dumping the pre-VAE latent via `HIPFIRE_DUMP_LATENT`). It came
out **degenerate** (a red-dominant field with a strong regular grid, NOT "a red apple on
a wooden table"), because the plain txt2img used the **default `--cfg-scale 7.0`** whereas
Krea-2 **Turbo runs with NO CFG** (`diffusion.rs:418`, preset `krea2-turbo-8plus1`: sigma
0.11, 8+1 refine) — cfg 7 on a distilled model drives the latent off-distribution; 512²
may additionally be OOD if the model is 1024-native (the `sefi-color-grid-artifact`
precedent: turbo models degrade to a grid at low res). So this render is **not a valid
grain test** and no VAE-vs-DiT conclusion is drawn from it. Two observations that are
nonetheless consistent with C2's exoneration: (1) the dumped latent is **low-frequency**
(`|dx|/std = 0.187`; white noise ≈ 1.2–1.4), i.e. NOT the high-freq latent noise a
"grain-in-latent" story predicted — held loosely since the render is degenerate; (2) an
OOD (cfg-7) latent decoding to a grid is garbage-in/garbage-out and does **not** implicate
the VAE, which C2 proved faithful on *in-distribution* latents (MSE 1e-4). **A valid
confirmation render needs the Turbo config (no CFG, likely 1024² via the
`krea2-turbo-8plus1` preset)** — that is the one owed step; but the authoritative Stream C
result is C2's config-independent golden-code match + round-trip, which stands on its own.

**Correct-params retry (2026-07-31): IMPRACTICAL on this box — the render runtime is
CPU-only.** Re-ran at the right Turbo config (cfg 1.0 / no CFG, 1024², seed 42). It ran
~20 min with **GPU utilization 0%** — `Krea-2-Turbo.source.hfq` uses the
`cpu-source-reference` runtime (per `hipfire diffusion inspect`: `runtime:
"cpu-source-reference"`), so `--rocm-device-id 0` does not route the DiT to the GPU. A
1024², 8-step bf16 DiT on one CPU core is ~an hour+; it was killed as a poor trade for a
confirmation of an already-settled result. **A visual confirmation isn't cheaply available
here** — it would need a GPU-runnable Krea-2 `.hfq` (the source-reference artifact runs on
CPU), or a much faster machine. **Bottom line: the VAE-exoneration conclusion rests
entirely on C2's config-independent code proof (golden-source match + MSE-1e-4
round-trip), which needs no render.** The render was only ever belt-and-suspenders.

## 11. Stream D2 — footprint / cold-load quant win (quantified, 2026-07-31)

Measured on disk: full bf16 pipeline `Krea-2-Turbo.source.hfq` = **33.23 GiB**; the
DiT-quantized `Krea-2-Turbo.dit.oq4.25.hfq` = **16.78 GiB** (−50% whole-artifact; VAE +
Qwen3VL text-encoder stay bf16). The DiT weight stream specifically: bf16 **22.64 GiB**
(§8's cold-first-step number) → oq4.25 ≈ **6.0 GiB** = **~3.7× less to stream**. Since
`/srv/hipfire` is a networked mount and the 344 s / 22.64 GiB cold step implies ~67 MB/s
(mount-bound, NOT DRAM-bound), the cold-load time scales ~linearly with bytes → oq4.25
DiT cold-load ≈ **~90 s (~3.7× faster)**. This confirms §8's thesis with numbers: **on
diffusion, quant's clear, immediate, un-blocked win is footprint/cold-load, not warm-GEMM
speed** (warm-GEMM needs D1's efficient DiT kernel first — the bf16 DiT GEMM runs at 8.4%
of peak, a ~6-10× kernel lever that low-bit only multiplies *after* it's fixed). D1
(efficient DiT bf16 GEMM) and D3 (DiT quant quality — blocked on a grain-free DiT per §10)
remain the open diffusion units.

## 12. Stream D1 — LDS staging does NOT transfer to the compute-bound DiT GEMM (2026-07-31)

Built a wave32 LDS-staged bf16 GEMM (`kernels/src/gemm_bf16_tiled_wmma_lds.hip`, dispatch
`gemm_bf16_tiled_wmma_lds`, `parity_gemm_bf16_tiled_wmma_lds`, `bench_dit_bf16_gemm`) — a
simplification of the iu4 LDS kernel (no int4 unpack/scales/rescale; bf16 accumulates
directly in the f32 fragment). **Bit-exact** to `gemm_bf16_tiled_wmma` (max_abs=0 on all
shapes incl. DiT 6144×6144 and unaligned bounds).

**Result (gfx1103, N=2048, vs the current register-tiled 4×4):**

| DiT shape | tiled %peak | LDS %peak | speedup |
|---|---|---|---|
| attn q/o/gate 6144×6144 | 10.2% | 11.5% | 1.13× |
| attn kv GQA 1536×6144 | 16.3% | 11.2% | **0.69× (slower)** |
| ffn gate/up 16384×6144 | 10.4% | 11.5% | 1.10× |
| ffn down 6144×16384 | 9.9% | 11.8% | 1.20× |

**Negative result — the LLM win does NOT transfer.** The iu4 LDS win (~3×) came from LDS
killing *redundant DRAM traffic* (the iu4 GEMM was memory/occupancy-bound). The DiT bf16
GEMM is **firmly compute-bound** (AI 800–1400 ≫ ridge 240, §recon), so a memory-traffic
optimization barely helps: only ~1.1× on big shapes, and it *regresses* small-M GQA (its
larger BM=64/8-wave block under-utilizes vs the 4×4's smaller, more-numerous blocks). The
register-tiled 4×4 (16 independent WMMA chains, 1 wave/block) already has the ILP; the LDS
2×2 (4 chains/wave) does not out-schedule it. **The limiter is WMMA scheduling/occupancy,
not DRAM — and LDS is the wrong lever for it.** The kernel is committed as bit-exact,
reusable infrastructure but is **NOT wired into the DiT routing** (it would regress GQA).

**The real DiT levers, re-ranked by this finding:**
1. **Deep-ILP WMMA scheduling** — the only path to 25–40% of peak, and it's *hard*. wave64
   (your nudge) is the promising angle: its 4-acc-GPRs/tile (vs 8 wave32) lets a wave hold a
   *bigger* super-tile (more independent chains) without spilling — which is exactly what a
   compute-bound WMMA kernel wants. A wave64 4×4-super-tile LDS kernel is the next experiment.
   (LDS at BK=64 caps BN via the 64 KiB budget, so this needs BK=32 or fewer waves.)
2. **Low-bit activations (int4) on the DiT** — on RDNA3 the iu4 WMMA is **2× the bf16 rate**
   (16 vs 32 cyc), so W4A4 is up to a **2× on the WMMA itself** — a *bigger, more accessible*
   lever than out-scheduling the bf16 kernel. But it needs the DiT quant-quality validation,
   which is blocked on a grain-free DiT (§10, §D3).
3. **De-dup the redundant f32→bf16 staging** — `must_convert=true` re-stages identical X for
   all 4 attn projections (§recon); caching it is a free few-% with no kernel work.

Honest conclusion: **D1's naive "port the LDS win" failed — the DiT GEMM is a different
(compute-bound) beast.** A real bf16 win needs the hard wave64 deep-ILP scheduling work; the
more accessible 2× is int4-act, gated on DiT quality. The register-tiled 4×4 is near the
achievable for its structure.

### 12a. The compute headroom is spendable — free-ALU budget = ~16–24 FMAs/WMMA (2026-07-31)

**Correction to §12's ending:** "the register-tiled 4×4 is near the achievable, bf16 is hard"
looks at it backwards. The DiT GEMM at ~10% of peak is COMPUTE-bound-but-stalled → the VALU
is mostly idle, and **that idle compute is spendable** on a heavier, higher-quality quant —
the reframe: *use the extra compute to run QTIP/codebook decode + correction, which normally
is the blocker (a fast codebook-indexed GEMM, plan §4.3 "hardest kernel question"), for free.*

**Measured** (`bench_bf16_lds_freealu.hip` = LDS bf16 GEMM + `extra` throwaway VALU FMAs/WMMA;
attn 6144×6144, N=2048): wall-time is FLAT through **extra=16 FMAs/WMMA (0.99×)**, then rises
(32→1.28×, 64→2.41×). So the free budget is **~16–24 scalar FMAs per WMMA.** Per lane a WMMA
consumes 16 weight elements, so that is **~1–1.5 free ops/weight** — enough to hide:
- a **per-fragment QTIP/codebook decode** (trellis/LUT → weight, ~1–2 ops/weight), or
- a **correction branch** (SVDQuant low-rank add, LDLQ error-feedback, dual-branch sum).

**So the DiT lever is NOT "make bf16 faster" (D1: hard, ~1.1×) — it's "quantize the DiT to a
compute-heavy codebook/corrected low-bit format and hide the decode in the stalls":**
footprint/cold-load win (§8, D2's ~3.7×) + codebook near-lossless quality, at ~zero warm-step
cost because the GEMM was idle anyway. This dissolves Lever D's blocker *for the DiT
specifically* (the LUT-GEMM needn't be fast — just correct — since there's compute to spare),
and makes int4-act's dequant/correction free too. Next: design a codebook-decode / correction
that fits the ~16–24 FMA/WMMA budget, fused into this LDS kernel's WMMA loop; measure footprint
+ quality vs the bf16 baseline. Experiment kernel `bench_bf16_lds_freealu` + the sweep in
`bench_dit_bf16_gemm` are the harness.

### 12b. BUT the budget is FMA-shaped, not gather-shaped — QTIP trellis is NOT free (2026-07-31)

Refining §12a with a realistic decode shape (serial, data-dependent LDS-codebook gather —
the QTIP/Viterbi/LUT pattern) instead of pure register FMAs, the free budget **collapses**:

| extra ops/WMMA | pure FMA (§12a) | serial LDS-gather |
|---|---|---|
| 8 | 1.00× (free) | **1.63×** |
| 16 | 0.99× (free) | 2.19× |
| 32 | 1.28× | 3.35× |

Even 8 gathers/WMMA cost +63%. The "free compute" is real only for **register-arithmetic**
work — the gathers contend for the LDS ports the GEMM's own s_A/s_X staging uses, and a serial
trellis chain exposes LDS latency that a pure-FMA chain doesn't. (Caveat: my gather was
*serial* worst-case; a *parallel* per-weight codebook gather would hide somewhat better, but
LDS-bandwidth contention is intrinsic.)

**Corrected direction — this partially reverses §12a's "QTIP is free":**
- **Arithmetic decode/correction hides (mostly free):** int4→bf16 dequant (shift/convert, no
  gather), a low-rank **FMA** correction (SVDQuant rank-r add in registers), LDLQ error-feedback
  as arithmetic. So **int4-act (2× WMMA) with a cheap arithmetic dequant + a register low-rank
  correction is the DiT low-bit path that fits the budget.**
- **Gather-heavy codebook / QTIP trellis does NOT hide** — it ~doubles the warm step. So LO-BCQ
  codebooks / QTIP LUTs (Lever D) are the WRONG fit for hiding-in-the-GEMM on the DiT, exactly
  opposite to the naive "extra compute corrects QTIP for free."

Net: the compute headroom favors **arithmetic-decodable low-bit (int4 + low-rank/error-feedback
correction)** over LUT/trellis codebooks. The footprint/quant-quality experiment should target
that, not a codebook GEMM. (And it still needs a grain-free DiT for the quality gate, §10/§D3.)

### 12c. The budget is a KERNEL property, and the production kernel has ~8× more (2026-07-31)

§12a/§12b measured the **LDS** kernel (`bench_bf16_lds_freealu`). The same probe on the
**production** kernel — the register-tiled 4×4 that actually runs the DiT
(`gpu_ops.rs` `use_tiled`) — gives a very different, much larger budget
(`kernels/src/bench_bf16_alu_headroom.hip` + `examples/bench_bf16_alu_headroom.rs`,
attn M6144 K6144 N=2048, side FMAs are register-arithmetic = §12b's "real" regime):

| FMAs/WMMA | ms | vs 0 | arithmetic ops per *loaded* weight element |
|---|---|---|---|
| 0 | 97.40 | 1.00× | 0 |
| 4 | 82.31 | **0.85×** | 1 |
| 16 | 82.78 | **0.85×** | 4 |
| 64 | 85.03 | **0.87×** | 16 |
| **128** | **87.23** | **0.90×** | **32** |

**Two findings.**

1. **≥128 FMAs/WMMA is free on the production kernel — 8× the LDS kernel's ~16–24 knee.**
   Per *loaded* weight element (each A-fragment is reused across TILE_NB=4 WMMAs, so decode
   happens once per 4 WMMAs) that is **~32 arithmetic ops/weight-element**, vs ~1–2 on the LDS
   kernel. So **the free budget is a property of the KERNEL STRUCTURE, not of the GPU**: the
   4×4's 1-wave/block, 128-acc-VGPR, zero-LDS shape leaves far more idle issue slots than the
   8-wave LDS kernel. Corollary for design: a low-bit DiT kernel should be built on the
   *register-tiled* shape, where the decode is nearly unbounded, not the LDS shape.

2. **Adding arithmetic makes the production kernel ~15% FASTER** (97.4 → 82.3 ms), and it is
   still 10% faster at 128 FMAs/WMMA. **Control: alu0 re-measured LAST = 96.21 ms vs 97.40 ms
   first (−1.2%) ⇒ order-independent, not a clock-ramp artifact** (a 30-launch warmup precedes
   all timing). This is a **scheduling pathology**: the baseline stalls on dependent global
   fragment loads, and independent VALU work fills those slots (classic software-pipeline
   filler). Two consequences: (a) an arithmetic dequant/correction branch is not merely free
   here, it **pays for itself**; (b) there is an **independent ~15% DiT win** available from
   scheduling/pipelining the existing kernel with *no* format change — cheaper than any of the
   §12 levers and worth taking first.

**Net (consistent with §12b, strengthened):** on the kernel that actually ships, the
arithmetic headroom is ~32 ops/weight-element — int4→bf16 dequant (~2 ops) plus a register
low-rank/error-feedback correction fits with an order of magnitude to spare. §12b's verdict
stands (gather/trellis is NOT free; arithmetic is), and the margin for the arithmetic path is
much larger than §12a suggested. Sequence: (1) take the free ~15% scheduling win, (2) build
int4-weight + arithmetic dequant + register low-rank correction on the register-tiled shape,
(3) gate quality on a grain-free DiT (§10/§D3).
