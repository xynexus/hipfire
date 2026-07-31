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
