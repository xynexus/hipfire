# GOAL: Opus QAT, then an in-tree DFlash2 drafter trainer

Two tracks, decided 2026-08-28. **Track A (Opus QAT) first** — its foundations
exist, and it de-risks the training infrastructure Track B also needs.

Start from `origin/master` (5a3f893e5 or later; `4ddf67218` lands Track A's
plumbing). halo / gfx1151 / 128 GB UMA.

---

## Track A — Opus QAT: how much of each tier's loss is recoverable?

The question: under light QAT (frozen fake-quantized base + trainable LoRA),
what share of the deploy loss does each Opus width recover? W3 was previously
measured at ~52% recoverable. W4 (the deployed tier) has no number.

### What already exists — do not rebuild it

- `crates/hipfire-train/src/oqplus_quant.rs` — `oqplus_simquant` (OQ+ W4),
  `oq3_simquant`, `oq8_simquant`. fp32→fp32 round-trips reproducing the
  production codec's *weight* error: FWHT-256 rotate → symmetric clip-searched
  scale → round to signed int → dequant → inverse FWHT. Weight-only on purpose;
  its header records why (A8 vs A16 adds ~0.0005 KLD, W8→W4 is the dominant
  0.15).
- `crates/hipfire-train/examples/qat_opus_kvarn.rs` — the QAT loop: frozen
  fake-quantized weights + trainable LoRA(q/v) + RMSNorm, KL-distilled against a
  clean fp32 teacher, reported on in-sample **and** held-out batches, with the
  KVarN KV path in the student under `HIPFIRE_QAT_KVNOISE` (default on).
  Tier-parametric via `HIPFIRE_QAT_TIER=oq3|oq4|oq8`, default `oq4`.
- `hipfire-train` is real fp32 GPU autograd. `examples/bake_oqplus_sim.rs`,
  `gradcheck_*` exist.

### Two traps that already cost a session — read before running

1. ~~**`DEFAULT_DIR` in the example is stale.**~~ **FIXED 2026-08-28** — it now
   points at `Llama-3.2-1B/snapshots/main`, and a safetensors preflight names the
   reason instead of letting the loader emit a bare "no .safetensors files found"
   before touching the GPU.
2. **Llama-3.2-1B has a revision trap** — same shape as the ZAYA one in
   `AGENTS.local.md`. Two snapshots exist and only `snapshots/main` carries
   `model.safetensors`; `snapshots/<hash>/` is empty of weights, so
   `ls -d snapshots/*/ | head -1` picks the wrong one.

       D=/srv/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/main
       hipfire lock acquire "qat-opus"
       HIPFIRE_QAT_TIER=oq4 ./target/release/examples/qat_opus_kvarn "$D"
       hipfire lock release          # takes NO label

   `DeepSeek-Prover-V2-7B` is the only other local `model_type: llama` with
   safetensors, if a second model is wanted.

### Steps

1. ✅ **DONE 2026-08-28** — results in
   `docs/experiments/2026-08-28-opus-qat-tier-sweep.md`. Held-out recovered
   share, KV clean, real text, LR 1e-4 warmup+cosine: **W8 ~0%** (nothing to
   recover — 0.0012 nats/tok), **W4 37.0%** (0.2211 → 0.1392),
   **W3 65.1%** (1.3768 → 0.4798).
2. ✅ **DONE** — `DEFAULT_DIR` repointed at `Llama-3.2-1B/snapshots/main` plus a
   safetensors preflight that names the reason.
3. Then decide whether light QAT (LoRA only) is enough for W4 or whether the
   base weights need to move.

Three defects had to be fixed before step 1 produced a trustworthy number, all
detailed in the experiment doc: the dead `DEFAULT_DIR`; a 64-token training set
that memorised (72.7% in-sample while held-out got *worse*), fixed with a
32-batch cycled pool over real text via `HIPFIRE_QAT_CORPUS`; and a missing LR
schedule, fixed with 10-step warmup + cosine and `HIPFIRE_QAT_LR`. Peak 1e-4
beats 1e-3 at every tier.

⚠️ **The default arm measures KV quantization, not the Opus tier.**
`HIPFIRE_QAT_KVNOISE` defaults on, and its W8 before-KL is 2.5152 against a
weight-only W8 loss of 0.0012 — ~2.5 nats/tok of common KVarN-4 floor, larger
than W3's entire weight-only loss. Pass `HIPFIRE_QAT_KVNOISE=0` to measure a
weight tier.

### Stage A2 — QAT the Opus **W4A4** path

The tier sweep above is **weight-only**: `oqplus_quant` bakes W4 weight error and
says so, on the grounds that A8 adds ~negligible KLD over A16 (oq8 W8A8 0.00156
vs q8f16 W8A16 0.00101). That reasoning does **not** carry to A4 — the W×A
precision matrix puts A4 at roughly **−3.5 dB**, where A8 ≈ A16. So W4A4 is a
materially harder QAT target than W4A8 and deserves its own stage, not a flag.

Everything needed already exists and has simply never been wired to the QAT loop:

- `src/a4_quant.rs` — `a4_simquant(x, rows, feat)`, `GROUP = 256`, `snr_db(..)`.
  The activation side of the `Oq4G256` W4A4 path. ⚠️ It uses an **absmax** scale,
  deliberately, not the weights' clip-search: online per-token activation quant
  cannot afford a clip search. Do not "improve" it into clip-search — that would
  model a grid the runtime never deploys.
  It models only the int4 round-trip; rotations (R1 residual / R3 KV / R4 down)
  are applied to the activation **upstream**.
- `src/learn_rotation.rs` — SpinQuant Phase 2: learned R1 by Cayley SGD on the
  Stiefel manifold. Read its header before touching the objective: a plain STE on
  `Q(XRᵀ)·Q(WRᵀ)ᵀ` has a near-zero gradient w.r.t. R, because the clean term is
  rotation-invariant and STE zeroes the quant-noise derivative — so the loss
  looks flat exactly where it is not. It minimises a per-element 4th-moment
  (kurtosis) surrogate instead.
- `src/kv_noise.rs` — the worked template for injecting a forward-only sim-noise
  perturbation into a differentiable student.
- Today `a4_quant` is used **only** by `examples/rotation_a4_snr_probe.rs` and
  `examples/learned_r1_probe.rs`. Nothing connects it to `qat_opus_kvarn`.

Steps:

1. ✅ **DONE 2026-08-28** — `HIPFIRE_QAT_ACT=a16|a8|a4`, default `a16` (a true
   no-op: the A16 leg reproduced Stage A1's `oq4` to four decimals). Applied to
   the FOUR tensors that feed all seven projections (`xn1`, `ctx`, `xn2`, `act`)
   in `block_forward_inner`, not to `linear_forward` — that funnel also carries
   `lm_head`, the MoE router and the LoRA B leg whose `K = lora_rank` < GROUP.
2. ✅ **DONE** — results in `docs/experiments/2026-08-28-opus-qat-tier-sweep.md`.
   Held-out recovered, weight tier pinned at `oq4`:
   **A16 37.0%** (0.2211 → 0.1392), **A8 35.2%** (0.2202 → 0.1426),
   **A4 23.0%** (0.8624 → 0.6637).
   **Answer: light QAT does NOT absorb A4.** It nearly quadruples the deploy loss
   (3.90×) and then recovers a *smaller* share of it, leaving a 4.8× worse
   residual. A4's best point is step 80, not 120 — it turns around, so a longer
   budget is not the lever either. A8 ≈ A16 is confirmed, which retroactively
   justifies Stage A1 being weight-only.
3. **NEXT, and now the live question** — add the rotation: A4 is what rotations
   exist *for* (they Gaussianize activations whose measured kurtosis exceeds
   200). Score fixed-FWHT vs learned R1 under QAT, since LoRA alone is ruled out.

⚠️ **Learned rotations are PREFILL-ONLY.** A previously measured result: the
learned M wins on prefill, while plain FWHT is best at decode. So a learned-R1
QAT result does not automatically transfer to the decode path — state which
phase any number belongs to.

### Known weakness — FIXED 2026-08-28, and it was worse than described

The batches were **synthetic random token ids** with `SEQ=16`, `N_TRAIN=4`,
`N_EVAL=4`. Disjoint salts meant no train-on-test, but the real problem was size,
not realism: `N_TRAIN * SEQ` = **64 training tokens** against 97 trainable
tensors. The loop memorised them — in-sample KL 2.2430 → 0.6114 (72.7%
"recovered") while held-out went 2.5152 → **2.9003, 15% worse**.

Fixed by `HIPFIRE_QAT_CORPUS=<text file>`: real text tokenized with the model's
own `tokenizer.json`, a 32-batch cycled train pool (128 seq) and 32 held-out seq
drawn from the **opposite half of the file**. Unset still gives the old synthetic
path, so historical numbers stay reproducible. Related:
`reference_calib_corpus_construction`, and the retracted "budget = −13.6%" result
that came from a calib corpus and a KLD ref being the same file.

---

## Track B — in-tree DFlash2 drafter trainer

**Decision made: build it in Rust in `hipfire-train`**, not as an external
PyTorch fine-tune. Training is inference-shaped work and belongs in-tree per
`AGENTS.md`; only that version can ever be daemon-scheduled.

### Why this is a build, not an extension

- DFlash2 drafters are **imported from HF** (`z-lab/*-DFlash`) via
  `hipfire-quantize`'s `dflash_convert`. Nothing in-tree trains one.
- `examples/tiny_dflash_train.rs` calls itself "the first useful DFlash trainer
  slice" and trains only the bridge `fc.weight` on **synthetic** hidden states,
  emitting a DFlash**1**-shaped config. No conv, no selector, no target-feature
  capture, no body training.
- `src/drafter.rs` is a different model entirely — a block-*scoring* drafter
  (`wk_score`, `pflash_score_backward`, cosine block scores).
- Closest template is `src/dspark_drafter.rs`: a complete un-fused fp32 forward
  **and matching backward** for a 5-layer dense-GQA block drafter, plus
  `dspark_train.rs` / `dspark_loss.rs` / `dspark_labels.rs` / `dspark_export.rs`.
  DSpark ≠ DFlash2, but the shape and the fwd/bwd scaffolding transfer.

### Target architecture to reproduce

From `z-lab/Qwen3.8-27B-DFlash2`'s own `config.json`:

    architectures      DFlash2DraftModel      num_hidden_layers   5
    hidden_size        5120                   intermediate_size   17408
    num_attention_heads 32                    num_key_value_heads 8
    head_dim           128                    vocab_size          248320
    block_size         8                      mask_token_id       248070
    conv_kernel_size   2                      conv_group_size     16
    selector_rank      256                    selector_top_k      16
    target_layer_ids   [5, 19, 33, 47, 61]    num_target_layers   64
    layer_types        5x sliding_attention   sliding_window      2048
    tie_word_embeddings false                 is_causal           false

Note `is_causal: false` and the EAGLE-3-style multi-layer feature fusion over
five target layers — the draft consumes hidden states captured from the 27B at
those depths.

### Why it is worth doing — and the honest ceiling

Driven correctly the current draft already **beats** the published comparable:
τ 3.85 on code against EAGLE-3's 3.21 on Qwen3-30B-A3B/HumanEval. So this is not
fixing a broken drafter; it is pushing past a good one. Set expectations
accordingly and measure before committing to a long training run.

Two facts that shape the work:

- **`use_gdn_per_token` / block size.** The draft is trained at `block_size 8`,
  and the code warns that exceeding the trained block "regresses code by ~30%".
  A retrained drafter at a larger block would legitimise the larger-B regime,
  where τ is highest.
- **The carried `candidate_selector` (rank 256, top_k 16) is NOT applied** — the
  draft path takes a per-position argmax. `HIPFIRE_DFLASH2_SELECTOR=1` applies
  it and measured *worse*, which is suspicious for a trained-in component and
  may indicate the implementation is wrong rather than the selector being bad.
  Worth understanding before training anything, since it is free signal.

### Suggested order

1. **Prove the headroom first.** Capture real 27B features at the five target
   layers and check what τ a well-fit drafter could reach on our prompt mix.
   Cheap next to building a trainer for a win that may not be there.
2. Investigate the unapplied selector (above).
3. Fork `dspark_drafter.rs` for the DFlash2 body fwd/bwd; add conv and selector.
4. Target-feature capture from the 27B (halo has the memory; duat's 3090 at
   24 GB does not).
5. Train, export via the `dflash_convert` metadata shape, measure τ on the mix.

---

## Housekeeping — add a disk-free watcher for `/home/sadara`

**TODO, not yet built.** Add a watcher that keeps at least **20 GB** free on
`/home/sadara` at all times.

Why it matters here: both tracks write large artefacts (fake-quantized student
checkpoints, captured 27B target features across five layers, drafter exports),
and a long unattended download has been running for days. `/home` is a 3.6 T
volume at 77% (833 G free as of 2026-08-28) so there is headroom today — this is
about not discovering the limit mid-training-run.

Shape it should take:

- Threshold 20 GB free, on the `/home` mount (`/dev/nvme0n1p3`), not the repo dir.
- Warn well before the floor, and make the failure *loud and early* rather than a
  half-written artefact — a training run that dies at hour six on ENOSPC costs
  far more than the check.
- Prefer refusing to START a run that cannot fit its expected output over
  aborting midway; that means the estimate lives next to whatever allocates.
- Candidate homes: a preflight in the training examples, or a periodic check in
  the daemon. Do **not** roll a new lock or sentinel-file mechanism for it —
  `AGENTS.md` allows exactly one lock primitive (`hipfire-lock` `flock(2)`).
- Nothing should silently delete anything; report and stop.

## Ground rules that apply to both

- **Bench a prompt MIX, never one prompt.** This repo has been burned
  repeatedly; `dflash_spec_demo.rs` carries three separate warnings about it.
- **τ is deterministic and contention-immune; tok/s is not.** A background
  download has been running for days and moves absolute rates by ~30%. Prefer τ
  and other invariants; if quoting tok/s, use interleaved repeats.
- **`include_str!` stale-binary trap.** Kernel `.hip` sources are embedded at
  Rust build time — editing a kernel without `cargo build` measures the old one.
- **Verify a knob took effect** by reading the value back (the feature report
  prints `dn_quant=`, `kv_mode:`, `block_size override:`), not by assuming the
  env var was honoured. `env -u VAR` to unset; `-u` must precede assignments.
- GPU examples do **not** self-lock — wrap in `hipfire lock acquire/release`.
  Do NOT wrap `hipfire eval` or the `tiny-*` gates; they self-lock and deadlock.
- `./tests/no-gpu-ci.sh` before handing off. Expect to regenerate `docs/env-vars.md`,
  `man/`, and all three `docs/config-schema.*` when touching config or CLI.
- User-facing options go in `hipfire-config` + the schema registry; `HIPFIRE_*`
  env is **debug only**. `deltanet_state_precision` is the recent worked example.
