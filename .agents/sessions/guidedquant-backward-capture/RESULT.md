# RESULT: GuidedQuant backward capture — SCOPE CORRECTED 2026-09-02

**Not started as briefed, because most of it is already built.** The brief says
*"Today's calibration is forward-only"* and *"What is missing is extending
gradient capture to a real target and wiring it into the quantizer's
objective."* Both halves are stale. This is the ninth-of-twenty pattern the
house rules warn about, so the deliverable here is the corrected map.

## What already exists, end to end

| stage | where | state |
|---|---|---|
| fp32 GPU autograd | `hipfire-train/src/ops/` — `linear_backward_x/_w`, `rope_train_bwd`, lora, deltanet, moe, rmsnorm, swiglu, attention | real |
| full model backward | `model_loss_backward`, `model_guided_adjoints` (`hipfire-train/src/model.rs`) | real |
| Fisher weight `w[n] = mean_c (∂ℓ/∂z)²` | `Gpu::calib_row_meansq_f32`, with a GPU parity example (`hipfire-rdna/examples/parity_calib_guided.rs`) | real |
| **weighted Hessian `H̄ = Σ wₙ·xₙxₙᵀ`** | **`CalibCollector::capture_weighted`, in `hipfire-runtime/src/calibration.rs` — documented as "GuidedQuant capture"** | real |
| artifact | `hipfire-train/examples/calib_guided.rs` writes a real `.calib.hfq` of guided Hessians | real |
| cross-tensor magnitude (K-FAC `gamma`) | `calib_gamma.rs` + `GammaAccum`; `model.rs` explains why it is a SEPARATE statistic (guided normalises `w /= mean`, discarding the per-tensor magnitude that says o_proj matters more than k_proj) | real |
| **quantizer consumption** | `HIPFIRE_MIXED_BPW_GAMMA=<gamma.json>` weights mixed-bpw allocation (`hipfire-quantize/src/cli.rs:5923-5992`) | real |
| guided-vs-plain A/B | `cli.rs:7463` — LDLQ with a guided vs a plain training Hessian, scored on held-out guided/plain eval Hessians | real |

The brief's proposed first moves 1 and 2 ("confirm the autograd", "decide where
captured gradients live") are answered: the autograd is real, and gradients live
in `.calib.hfq` as guided Hessians — exactly the "natural sibling" the brief
proposed.

## The gap that IS real, and it is a measurement gap

The existing A/B scores **reconstruction error** on held-out Hessians
(`held-out FISHER eval: guided rel … plain rel …`). The brief's own verification
bar rules that out:

> **Downstream KLD on a held-out corpus, never reconstruction error.** Two codecs
> that reconstruct at per-row cosine 0.99999 differed by ~0.06 KLD downstream —
> reconstruction MSE is not a valid proxy and has misled this project before.

So GuidedQuant is built and is being judged by the metric this project has
already been burned by. **The next session's job is to run the bar, not to write
the feature**: quantize one model twice from `calib_guided` vs a plain
`.calib.hfq`, then `hipfire eval` KLD on a corpus that is NOT the calibration
corpus. That is a day of GPU time, not a research programme, and it is the only
thing that can confirm or retire the held-out finding the brief rests on
("plain XᵀX LDLQ scores about the same as no calibration").

## Real coverage gaps, in priority order

1. **Only `down_proj` is captured.** `down_guided_capture` is called for
   `model.layers.{l}.mlp.down_proj` and nothing else — no q/k/v/o, no gate/up.
   That is a defensible first target (widest fan-in) but it means "GuidedQuant
   on a model" today means "GuidedQuant on one tensor per layer".
2. **It lives in `hipfire-train` EXAMPLES, not the production path.** The
   daemon/`coexistence calibrate` route produces plain Hessians. Per AGENTS.md
   the forward-pass engine belongs in `hipfire_runtime::calibration::layer_stream`
   so both the daemon and the CLI drive one engine — the guided path does not go
   through it, so guided and plain artifacts are not produced by the same code.
3. **`gamma` reaches the quantizer through `HIPFIRE_MIXED_BPW_GAMMA`,** an env
   var. Env is a debug layer here, not a setting: a user-facing knob belongs in
   `hipfire-config`'s schema. As it stands, gamma-weighted allocation cannot be
   reached from `model_overrides`.
4. **`loader.rs:456` warns that the layer-streamed gamma is "not production
   gamma"** — it measures something narrower. Anyone extending gamma should read
   that comment first.

## What did NOT change

The traps in the brief all still stand and are the reason the measurement half
is the hard half:

- **Never calibrate and evaluate on the same corpus.** A "−13.6% from more
  calibration tokens" result was retracted as train-on-test. `reject_eval_corpus`
  guards the obvious case; do not defeat it.
- `DEFAULT_CALIB_SEQ_LEN` is 2048 and matches what KLD references are built at.
- Calibration capture was nondeterministic until a `zaya_value_compose_f32` data
  race was fixed (2026-08-30). Two disagreeing runs ⇒ suspect that class first.
