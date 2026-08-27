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

1. **`DEFAULT_DIR` in the example is stale.** It points at
   `SupraLabs--Supra-50M-Instruct`, whose local snapshot holds only a `.gguf`.
   `load_llama_fp32` wants safetensors, so it dies with "no .safetensors files
   found" *before* touching the GPU. Always pass a model dir.
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

1. Run the sweep `oq8 / oq4 / oq3` on one model and report **held-out** KL
   before vs after recovery, plus recovered share per tier. Held-out is the
   honest number; in-sample is not.
2. Fix `DEFAULT_DIR` to something that actually loads, or make the example fail
   with the reason rather than a bare loader error.
3. Then decide whether light QAT (LoRA only) is enough for W4 or whether the
   base weights need to move.

### Known weakness worth fixing early

The batches are **synthetic random token ids**
(`(t+1)*2654435761 + (s+salt)*40503 % vocab`), with `SEQ=16`, `N_TRAIN=4`,
`N_EVAL=4`. Train/eval use disjoint salts so there is no train-on-test, but a
"recoverable share" measured on uniform-random tokens may not transfer to real
text — quantization damage concentrates on the activations real text actually
produces. Consider swapping in a real corpus slice before trusting the numbers
for a deploy decision. Related: `reference_calib_corpus_construction`, and the
retracted "budget = −13.6%" result that came from a calib corpus and a KLD ref
being the same file.

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
