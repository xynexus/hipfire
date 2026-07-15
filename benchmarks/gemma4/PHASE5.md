# Gemma 4 Phase 5 dense-31B admission

Date: 2026-07-15. Status: blocked at the frozen base-short BF16 gate.

## Base-short result

Checkpoint: `google/gemma-4-31B` revision
`02e15e4990e8c452f8543fb26beff15b1daf8f3d`, converted losslessly to
`Gemma-4-31B.bf16.hfq`. Prompt: `The capital of France is`.

The corrected upstream capture hooks decoder-layer outputs directly, so layer
59 is the final pre-norm decoder boundary rather than Transformers'
post-final-norm `hidden_states[60]` value. With the proportional-RoPE partner
stride fixed, the initial selected-boundary comparison reported:

- all selected hidden boundaries pass;
- greedy generation matches exactly for 8 tokens;
- final-logit cosine is `0.9997756734978192`;
- final argmax matches (`7001`) and top-5 overlap is `5`;
- final-logit maximum absolute error is `0.5654382705688477`, above the frozen
  `0.5` limit.

Therefore the comparison status is `fail`. Phase 5 does not advance to the SWA
crossing or instruction checkpoint gates.

An expanded capture of all 60 decoder boundaries then showed that the initial
selection had hidden a second failure class. The source-BF16 embedding scale and
result boundary were corrected, and the attention score compensation was moved
after RoPE to match upstream operation order. The best unchanged-threshold run
now reports:

- exact greedy generation and final argmax/top-5 agreement;
- final-logit cosine `0.9997767002770637`;
- final-logit maximum absolute error `0.5618224143981934`;
- the first hidden-state threshold failure at layer 39, followed by failures at
  layers 52, 56, 57, and 58;
- layer-59 normalized RMSE `0.017851585727561726`.

The all-layer trajectory grows gradually through the middle and late stack
rather than jumping at a geometry transition. This remains a frozen-gate
failure.

Evidence on validation host `halo`:

- oracle: `~/.hipfire/evidence/gemma4/base-short-oracle`;
- Hipfire: `~/.hipfire/evidence/gemma4/base-short-hipfire`;
- all-layer oracle: `~/.hipfire/evidence/gemma4/base-short-oracle-all-layers`;
- best all-layer Hipfire:
  `~/.hipfire/evidence/gemma4/base-short-hipfire-all-layers-embed-bf16-rope-order`;
- the superseded post-final-norm oracle capture is retained at
  `~/.hipfire/evidence/gemma4/base-short-oracle-postnorm`.

## Rejected narrowing experiment

Rounding the F32 residual stream through BF16 only at completed decoder-layer
boundaries was tested without changing the comparator or thresholds. It
worsened final-logit maximum absolute error to `0.7078790664672852` and layer-59
NRMSE from `0.017207967824475305` to `0.022919521164880975`. The experiment was
removed; its evidence remains at
`~/.hipfire/evidence/gemma4/base-short-hipfire-layer-round`.

Direct final-head localization also rejected a head-only precision fix. The
best BF16-rounded head variant still had maximum absolute error `0.5625`, while
the existing F32 head reproduced Hipfire's captured logits and `0.5654382705688477`
maximum error exactly. The remaining admission gap is accumulated hidden-state
drift, not final RMSNorm, tied-head loading, softcap, or argmax behavior.

Reproducing every upstream BF16 materialization boundary was also tested after
the all-layer localization. It improved early-layer error (layer-0 normalized
RMSE `0.0021387058560682227`) but amplified late-stack drift and worsened final
logit maximum absolute error to `0.75`. The implementation was removed; its
evidence remains at
`~/.hipfire/evidence/gemma4/base-short-hipfire-all-layers-exact-bf16-v2`.

## Layer-39 operator localization

A benchmark-only operator trace was added at one selected decoder layer. It is
not active in normal serving or in the lowered graph. Layer 39 was chosen
because it is the first layer whose output crosses the frozen hidden-state
threshold. The trace compared the input and output of every major attention,
normalization, residual, and FFN boundary against hooks in the pinned
Transformers oracle.

The normalized RMSE grows continuously rather than jumping at one operator:

- layer input `0.0371752` and input RMSNorm `0.0637810`;
- Q/K/V projections `0.0463999`/`0.0508628`/`0.0626933`;
- normalized Q/K/V `0.0494234`/`0.0520880`/`0.0647346`;
- attention result `0.0563029`, output projection `0.0667375`, and
  post-attention normalization `0.0780552`;
- post-attention residual `0.0440975`, pre-FFN normalization `0.0681140`,
  gate/up projections `0.0517406`/`0.0639714`, and GeGLU `0.0810282`;
- post-FFN normalization `0.0808553` and layer output `0.0490120`.

Same-input checks rule out a defective RMSNorm implementation: each Hipfire
RMSNorm output matches the direct F32 formula to approximately `5e-8` NRMSE,
while applying the upstream BF16 boundary changes the same-input result by only
about `0.0016`--`0.0017` NRMSE. A same-input BF16-staged GeGLU formula matches
the oracle exactly, but differs from Hipfire's F32 GeGLU by only `0.00234256`
NRMSE and `0.00504637` maximum absolute error. Neither local difference can
explain the propagated `0.08`-scale operator error. The trace therefore finds
no discrete layer-39 operator defect; it is consistent with cumulative drift
between the oracle's full-sequence matrix execution and Hipfire's sequential
token/cache execution.

Evidence on `halo`:

- oracle operator trace:
  `~/.hipfire/evidence/gemma4/base-short-oracle-operator-39-attention`;
- Hipfire operator trace:
  `~/.hipfire/evidence/gemma4/base-short-hipfire-operator-39-attention`.

The full-sequence oracle and all frozen thresholds remain unchanged. Switching
to a sequential oracle after observing this result would change the admission
method rather than correct the implementation.

## Rejected normalized-branch rounding

Rounding normalized residual branches through BF16 was tested at both branch
ends and independently. All three variants worsened the retained clean
final-logit maximum absolute error of `0.5618224143981934`:

- attention and FFN branches: `0.6137619018554688`;
- attention branch only: `0.6338434219360352`;
- FFN branch only: `0.6549363136291504`.

The experiments were removed. Their evidence remains at
`~/.hipfire/evidence/gemma4/base-short-hipfire-postnorm-bf16`,
`~/.hipfire/evidence/gemma4/base-short-hipfire-attn-postnorm-bf16`, and
`~/.hipfire/evidence/gemma4/base-short-hipfire-ffn-postnorm-bf16` respectively.

## Rejected batched-prefill hypothesis

The full-sequence-versus-sequential execution hypothesis was tested by porting
the existing causal batched-prefill structure to dense Gemma 4: BF16 batched
linears, batched proportional RoPE, full and local causal attention, KV
materialization, all four normalization boundaries, and final-row capture. A
tiny mixed-geometry F32 fixture, including an SWA ring crossing, matched the
sequential reference exactly at both layer boundaries and final logits.

The first 31B attempt produced non-finite attention because the validation host
loaded an old `rope_partial_halfsplit_batched` blob without a hash after the
`basis_dim` ABI change. Q/K/V projections and norms were finite; attention was
the first non-finite boundary. Rebuilding the kernel with `/opt/rocm/bin/hipcc`
removed every non-finite value. The rebuilt, valid run nevertheless worsened
the frozen comparison:

- final-logit maximum absolute error `0.6142101287841797` versus the retained
  sequential result `0.5618224143981934`;
- final-logit cosine `0.9997564985461272` and normalized RMSE
  `0.023292552002439584`;
- hidden-state failures at layers 39, 40, 52, 53, 56, 57, and 58;
- exact greedy generation, argmax, and top-5 agreement remained unchanged.

The batched serving experiment was removed. The valid evidence remains at
`~/.hipfire/evidence/gemma4/base-short-hipfire-batched-prefill-recompiled`;
the stale-blob NaN run is retained separately and is not admission evidence.

## Rejected BF16 attention candidates

A lightweight layer-0 probe removed full-checkpoint paging from the attention
investigation. `capture_layer0_sdpa.py` loads only decoder layer 0, feeds the
five exact per-position Hipfire `pre_layer` rows, and uses the pinned ROCm
Transformers implementation for the remainder of that layer. Its completed
layer output matches the frozen full-model layer-0 capture bit-for-bit. Evidence
is retained at
`~/.hipfire/evidence/gemma4/base-short-layer0-sdpa-inputs`.

Two benchmark-only attention candidates were then compared with those exact
Q/K/V and SDPA tensors:

- the wave32 BF16 WMMA candidate reached maximum absolute error `0.03125` and
  NRMSE `0.00078965` for the complete five-row sequence; the final-row offset
  form also had maximum error `0.03125` and NRMSE `0.00113709`;
- rocBLAS reproduces the pinned ROCm math-backend decomposition exactly: BF16
  Q/K with F32 accumulation, F32 softmax, F32 P*V over BF16-materialized V,
  and a BF16 output boundary produced `8192/8192` bit-exact values for the final
  layer-0 row.

Exact operator parity did not translate into a whole-model admission result.
The following valid runs all retained exact greedy generation, argmax, and
top-5 agreement, but failed the unchanged `0.5` final-logit limit and hidden
thresholds:

- batched BF16 boundaries plus local WMMA: `0.8274631500244141` maximum error;
- batched linears plus local WMMA: `0.5746259689331055`;
- sequential local WMMA: `0.6481022834777832`;
- sequential exact local math with global attention left on the portable path:
  `0.6000323295593262`;
- sequential exact math for both local and global attention:
  `0.5686526298522949`;
- exact math plus targeted BF16 materialization along only the attention input
  pipeline: `0.6303200721740723`.

The exact-math variants improve layer-0 NRMSE from the retained path's
`0.004286234072177916` to as low as `0.00382647505739898`, but amplify drift
later in the stack. Their serving integrations and the rejected batched prefill
were removed. The standalone parity tools remain as experiment evidence; the
normal inference path remains the retained portable sequential implementation.

## Rejected exact BF16 rocBLAS batched execution

The remaining full-sequence hypothesis was tested with the upstream BF16
materialization contract carried through the complete prompt block. A
same-input layer-0 boundary capture first localized the initial drift:

- the input RMSNorm becomes `5376/5376` bit-exact after BF16 materialization;
- the existing BF16 WMMA Q projection still differs after BF16 materialization
  (`8155/8192` exact, `0.0625` maximum error);
- rocBLAS with BF16 input, F32 accumulation, and a BF16 destination reproduces
  the pinned PyTorch Q projection exactly (`40960/40960` values); using an F32
  destination followed by a separate BF16 cast does not select the same result;
- the strided causal QK/PV wrappers retain the previously established exact
  SDPA result (`8192/8192` values for the final layer-0 row).

A benchmark-only batched path then used the BF16-destination rocBLAS contract
for every Q/K/V/O and FFN linear, BF16 materialization at normalization,
RoPE, attention, activation, residual, and layer-scalar boundaries, and the
exact math-SDPA decomposition for the five-token prompt. The frozen comparison
still failed:

- layer 39: cosine `0.9988711905323525`, normalized RMSE
  `0.04832125303952534`;
- layer 57: cosine `0.9987753804275649`, normalized RMSE
  `0.049580248781128176`;
- final logits: maximum absolute error `0.6519060134887695`, cosine
  `0.9998245681153174`, and normalized RMSE `0.019692281418080104`;
- greedy generation remained exact, argmax remained `7001`, top-5 overlap was
  `5/5`, and every captured value was finite.

Valid evidence is retained at
`~/.hipfire/evidence/gemma4/base-short-hipfire-all-layers-exact-rocblas-batched-5tok`.
The unadmitted batched forward and capture route were removed. The standalone
linear and SDPA parity probes remain because they establish exact operator
contracts, but exact projections and attention alone do not close the
whole-model admission gap.

## Rejected BF16-staged RoPE candidate

The pinned Transformers implementation materializes each pointwise RoPE
product in BF16 before the final BF16 add/subtract. This is a different
precision boundary from the previously tested whole-output rounding. A
standalone HIP parity probe using the exact layer-0 Q/K inputs and RoPE tables
confirmed the distinction:

- the ordinary F32 kernel followed by BF16 rounding matched `35022/40960` Q
  values and `17515/20480` K values;
- a portable diagnostic kernel that rounds each input, trigonometric value,
  product, and final sum to BF16 matched `40960/40960` Q values and
  `20480/20480` K values exactly;
- the generated cosine and sine tables already matched the captured BF16 tables
  exactly (`640/640` values each), so the difference is pointwise staging rather
  than frequency or table generation;
- the diagnostic kernel compiled for `gfx1030`, `gfx1100`, `gfx1151`, and
  `gfx1200`.

Despite exact isolated-operator parity, selecting the staged kernel throughout
the 31B forward worsened the unchanged all-layer comparison:

- final-logit maximum absolute error `0.5677473545074463`, versus the retained
  path's `0.5618224143981934`;
- final-logit cosine `0.9997615861188698` and normalized RMSE
  `0.022828305269326585`;
- hidden-state threshold failures at layers 39, 40, 41, 43, 52, 53, 56, 57,
  and 58, versus layers 39, 52, 56, 57, and 58 on the retained path;
- exact greedy generation, argmax `7001`, top-5 overlap `5/5`, and finite
  captures remained unchanged.

The serving integration was removed. The standalone kernel and parity example
remain only as a diagnostic operator contract. Candidate evidence is retained
at
`~/.hipfire/evidence/gemma4/base-short-hipfire-all-layers-bf16-staged-rope`;
the layer-0 oracle inputs and tables are retained at
`~/.hipfire/evidence/gemma4/base-short-layer0-rope-cos`.

## Same-input decoder-transition sweep

The remaining full-stack drift was separated from individual decoder semantics
with a benchmark-only transition runner. Layer 0 receives the exact captured
post-embedding rows. Every later layer receives the preceding frozen
Transformers decoder boundary. Each layer then runs independently across the
same five prompt positions while building its own real full/SWA KV history.
This preserves each layer's attention geometry and causal history without
propagating error from earlier Hipfire layers.

All 60 transitions were finite, and no individual final-position transition
approached the frozen full-stack hidden-state limits:

- the worst final-position normalized RMSE was `0.005205519351551357` at layer
  58; the next highest values were `0.005173885575744475` at layer 40 and
  `0.005047795244414561` at layer 28;
- layer 39, the first full-stack failure, had same-input final-position NRMSE
  `0.004865753532395866` and cosine `0.9999912831135805`;
- layer 59 had maximum absolute error `0.0066835880279541016`, NRMSE
  `0.00236000046446405`, and cosine `0.9999976753678922` when given the exact
  layer-58 oracle boundary;
- the mean final-position NRMSE was `0.0030814201888346663` for the 50 sliding
  layers and `0.003070946966302261` for the 10 full-attention layers. There is
  no geometry-class discontinuity in this test.

This result rules out a single gross decoder-layer or full-versus-SWA semantic
defect for the frozen prompt. It is consistent with small per-layer numerical
differences accumulating through the 60-layer stack. The transition runner does
not change serving behavior or the frozen admission gate.

Durable evidence on `halo`:

- prepared, hashed transition inputs:
  `~/.hipfire/evidence/gemma4/base-short-layer-transition-inputs`;
- per-layer and per-position report:
  `~/.hipfire/evidence/gemma4/base-short-layer-transition-parity.json`.

The best result remains
`~/.hipfire/evidence/gemma4/base-short-hipfire-all-layers-embed-bf16-rope-order`
at final-logit maximum error `0.5618224143981934`, with exact greedy generation,
argmax, and top-5 agreement throughout. It remains above the frozen `0.5` limit.

## Post-freeze BF16 reference-variability control

`benchmarks/gemma4/control_noise_bf16.py` runs the same pinned Transformers
checkpoint under unpadded/padded and `sdpa`/`eager` BF16 executions. The initial
four-sample control observed an all-pairs final-logit maximum difference of
`0.6875`, hidden-state worst NRMSE `0.055774`, and minimum hidden cosine
`0.998456`; layer 57 was the worst hidden-state hotspot. This is strong evidence
that the retained Hipfire trajectory overlaps a real region of BF16
reduction-order sensitivity rather than a single gross operator defect.

It is not admission evidence. The control was run after the Hipfire result, it
covers one five-token prompt, and the all-pairs maximum may be produced by two
alternate executions rather than by the pinned unpadded SDPA oracle used by the
frozen comparator. Calling it an irreducible noise floor would overstate what the
measurement proves. The tool therefore records both the canonical-oracle
envelope and all six pairwise comparisons, along with output hashes and exact
decision agreement. The Phase-0 `0.5`/`0.045`/`0.999` limits remain unchanged.

The independent `halo` rerun reproduced the committed all-pairwise extrema
exactly. Its canonical unpadded-SDPA envelope was `0.625` final-logit maximum
absolute difference, `0.0518795542` worst hidden NRMSE, and `0.9986535068`
minimum hidden cosine. The broader all-pairwise values were `0.6875`,
`0.0557740958`, and `0.9984564247`. The committed artifact now includes that
canonical/all-pairs split, ROCm/device provenance, sample hashes, decision
summaries, and non-finite counts.

The original base-short verdict is therefore unchanged: final-logit maximum
error is `0.5618224143981934`; hidden thresholds fail at layers 39, 52, 56, 57,
and 58. Phase 5 stops here, so the SWA-crossing, multi-global, reload/sequential,
instruction-checkpoint, and later variant gates are not advanced.
