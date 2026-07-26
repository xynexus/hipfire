# Daemon-resident training & steering — unified forward-hook plan

**Date:** 2026-07-19
**Branch:** chaingun
**Status:** proposed
**Refines:** `docs/plans/2026-07-18-continuous-scheduler-headline.md` P5 step 3
("fold training + steering into the daemon"), and
`docs/plans/2026-06-19-train-as-daemon-op.md` (train-as-daemon-op).

## Goal

Make training and steering first-class, scheduler-arbitrated ops on the **one**
serving daemon — no second daemon, one GPU flock, one arbiter — **without ever
putting a backward pass into the arch seam**.

## Reframe (what this plan corrects)

P5.3 as written says steer needs "capture/apply ops hoisted from the harness"
and calls it "big daemon-side surgery." Measured reality (2026-07-19):

- **Steer is already daemon-resident.** Five ops exist and run real GPU compute:
  `SteerBeginCapture / SteerCapture / SteerFinishCapture / SteerBeginApply /
  SteerClear` (`hipfire-daemon/src/main.rs:6307–6433`), driving the in-forward
  `maybe_steer_block` hook. **But that hook is hand-wired into gemma3's forward
  only** (`hipfire-arch-gemma3/src/forward.rs:656,919`) — not qwen35 (the
  production arch).
- **Drafter training is already a daemon op.** `TrainDrafter`
  (`main.rs:7147`) runs a real backward+optimizer loop
  (`hipfire_train::train_loop`). Scoped to the SSM drafter, file-labels only;
  in-process hidden-state capture is the documented "step 4" gap.
- **The compute is not the gap.** The gaps are: (a) the shared hook is wired per
  arch, gemma3 only; (b) train/steer clients spawn their *own* daemon
  (`DaemonEngine::spawn`) → private GPU flock → invisible to `work_scheduler`;
  (c) the DFlash multi-layer teacher tap doesn't exist yet.

So this is topology + genericity + scheduler wiring — not new forward machinery.

## The four modes reduce to four primitives

Everything we want decomposes into four primitives; **none needs a backward pass
in the arch forward**:

- **TAP** — read-only residual-stream read at block boundaries (arch-uniform f32
  buffer). Already the capture branch of `maybe_steer_block`.
- **HOOK** — in-forward residual edit / adapter-inject at the same site. Already
  the apply branch of `maybe_steer_block`.
- **BWD-OPS** — per-op adjoint library in `hipfire-train/ops` (`gemm_*_train_nt`,
  `sdpa_backward`, `rmsnorm_backward`, `cross_entropy`, …). Arch-agnostic,
  already written.
- **BF16** — dense weight-load mode (`Bf16WeightLoadMode`,
  `HIPFIRE_BF16_WEIGHTS`), already kept for kldref generation.

| Mode | Primitives | Backward in arch? | bf16? | Teacher/base |
|---|---|---|---|---|
| **Steer / abliterate** | TAP + HOOK | no | no | production forward (the served model) |
| **DFlash sidecar** | TAP (multi-layer) + BWD-OPS | no | no | **production forward is the correct teacher** (drafter is verified against it) |
| **LoRA-on-frozen** | HOOK + BWD-OPS + BF16 | no (adapters only) | yes | bf16 base; caveat below |
| **Full-weight** | grad tape per arch | **yes** | yes | — **out of scope** |

`TAP` and `HOOK` are the *same call site* (`maybe_steer_block`), branching on the
active session. That one site is the shared substrate for all three in-scope
modes.

## Design invariants

- **No autodiff in the arch seam.** `Arch::forward` stays inference-only plus one
  block-boundary hook (read-or-write). Gradients live only in `hipfire-train`
  ops and only ever flow through drafters or adapters, never the base weights.
- **Production forward is the teacher for DFlash.** The drafter is verified
  against the production (quantized) model at serve time; distilling that exact
  forward maximizes acceptance. Do **not** use bf16 as the DFlash teacher.
- **bf16 is for LoRA + kldrefs only.** LoRA's base-mismatch caveat (train on
  bf16, serve on quant) is the one rough edge; escape hatch = train the adapter
  against the dequantized production base (load production weights in dense mode
  — no new forward).
- **One daemon, one flock, one arbiter.** Train/steer clients become thin
  clients of the serving daemon; none spawns a private daemon.
- **Offline weight surgery stays offline.** Permanent (weight-baked) abliteration
  is a one-shot weight-column projection — no backward — and per AGENTS.md
  belongs in `hipfire-coexistence`, **not** the daemon. This plan does the
  runtime-hook version only.
- **Test the seam, not the impl.** Every new seam gets one arch-agnostic
  dummy-impl unit test (serving-core `dummy.rs` pattern).

## Phases

Each phase ships something demonstrable, is independently revertible, and is
GPU-validated (nix2/gfx1103 for steer + drafter; halo/gfx1151 for larger runs).

### Phase 0 — Genericize the residual hook (the TAP + HOOK substrate)

The one call site every in-scope mode rides is wired into gemma3 only. Make it an
arch seam.

- Define a block-boundary hook the arch forward calls once per layer:
  `residual_hook(gpu, &mut x, layer_idx)` (name TBD), living in serving-core /
  the arch trait, delegating to `hipfire_steer::maybe_steer_block[_batched]`.
- Replace gemma3's hand-call with the seam; **add qwen35 as the second impl**
  (the two hand-call sites in gemma3 `forward.rs` become one trait call each in
  both archs).
- Keep the fast path free: when no steer/capture/adapter session is active the
  hook is a cheap no-op (it already is — "steering is inactive, the common case
  during normal serving").

**Delivers:** steer capture + apply work on qwen35, not just gemma3; the shared
substrate for Phases 2–3 exists.
**Files:** `hipfire-arch-gemma3/src/forward.rs`, `hipfire-arch-qwen35/src/qwen35/`,
serving-core (seam + dummy test), `hipfire-steer`.
**Exit:** `hipfire-steer/examples/gpu_validate` passes on qwen35;
`coherence-gate-dflash.sh` clean; hook is a measured no-op when inactive.

### Phase 1 — Steer / abliterate under the scheduler (lightest mode, mostly done)

Steer compute is already daemon-resident; wire it into the one arbiter and cut
the private-daemon rivalry.

- Add `ScheduledJob::SteerCapture` to `batch_runner.rs` (preemptible at
  capture-batch boundary, reuse P4 park/resume). **Apply is free** — an active
  apply is session state; ordinary text generation already rides the runner, so
  it steers for free.
- Server routes for capture / derive / apply / clear (derive is
  difference-of-means, host-side — not training).
- Make `hipfire-steer-harness` a **thin client** of the serving daemon; delete
  its `DaemonEngine::spawn`. Same for the abliteration flow (`mode=Ablate`).

**Delivers:** a steer capture and an interactive text stream time-slice on one
daemon; no second daemon is spawned; apply steers live serving.
**Files:** `batch_runner.rs`, new steer route, `hipfire-steer-harness`,
`state.rs`.
**Exit:** trace shows capture + serving interleaving by priority on one daemon;
gpu_validate apply parity holds; abliteration coherent at low strength (known
ceiling ~0.2 per prior evidence).

### Phase 2 — DFlash sidecar from scratch (fixed-arch trainer, production teacher)

The parent contributes only a forward as a labelled-data producer. The trainable
arch is fixed and written once.

- **Multi-layer teacher tap** (the "step 4" gap): extend the TAP branch to
  snapshot hidden states at the drafter's `target_layer_ids` (full states, not
  folded means), plus final logits and next tokens. Emit via the existing
  `PflashLabels` / label op (JSONL + `QEMB` embedding sidecar,
  `hipfire-train/src/labels.rs`). Teacher = **production forward**.
- Add `ScheduledJob::Train` running the fixed-arch drafter trainer
  (`dspark_drafter` forward+backward via BWD-OPS) **one micro-step per lease**,
  yielding at the micro-step boundary (reuse P4 park/resume).
- Make `hipfire-train` a **thin client**; delete its private daemon spawn. Drop
  the generic stand-in `block.rs` LLaMA block in favor of the real arch's tap.

**Delivers:** train a DFlash drafter for a qwen35 parent end-to-end on the
serving daemon, interleaved with live serving; drafter dims threaded from the
parent, embed/lm_head shared (`QEMB`).
**Files:** hook capture branch, `hipfire-daemon` (`TrainDrafter` →
runner-dispatched; label capture step 4), `batch_runner.rs`, `hipfire-train`.
**Exit:** a drafter trains to non-trivial acceptance against a qwen35 teacher
while a serving stream runs; micro-step preemption bounds interactive p99;
no second daemon.

### Phase 3 — LoRA-on-frozen adapters (model adaptation)

Reuse the bf16 forward + HOOK; grads flow only into adapters.

- At HOOK sites, save activations in bf16/dense mode and inject the LoRA delta;
  backprop only through the adapters (BWD-OPS `lora` op), base frozen.
- Serve trained adapters via the existing `LoraLoad / LoraSetScale / LoraUnload`
  ops (apply side already wired).
- Document + expose the base-mismatch escape hatch (train against dequantized
  production base when the adapter will be served on the quantized model).

**Delivers:** fine-tune a LoRA on the serving daemon under the scheduler, then
load and serve it.
**Files:** hook activation-save branch, `hipfire-train` (adapter loop),
`batch_runner.rs` (`ScheduledJob::Train` reuse), LoRA route.
**Exit:** a LoRA trained on-daemon measurably shifts served outputs; base-mismatch
path validated (bf16-trained vs dequant-trained delta on the quantized serve).

### Out of scope — full-weight training

Backprop through the parent arch forward requires a grad tape per arch — the one
case that forces a second forward per model. Explicitly rejected here; DFlash +
LoRA-on-frozen cover trained-from-scratch drafters *and* model adaptation without
it.

## Sequencing rationale

P0 first because TAP/HOOK is the shared floor and today only gemma3 has it. P1
next because steer is the most-done (compute resident, apply free) and proves the
scheduler wiring on the lightest mode. P2 and P3 both stand on P0's hook and
`work_scheduler` wiring from P1; order P2 before P3 only because the multi-layer
tap (P2) is a read extension of the same hook the activation-save (P3) writes
through, so P2 exercises the read path first. Each phase is independently
revertible; welds are cut only where a phase needs them.

## Risks & non-goals

- **Non-goal:** simultaneous multi-model execution — single-HIP-thread daemon
  stands; this is coordinated + preemptible, not parallel (scheduler Phase 7).
- **Risk (P0):** touching every arch's per-block forward. Contained by the
  inactive-session no-op fast path and coherence gate.
- **Risk (P2/P3):** micro-step / capture-batch preemption correctness under the
  scheduler. Reuse P4 park/resume (already GPU-validated byte-identical resume).
- **Caveat (P3):** LoRA base mismatch — named above with an escape hatch.

## Open decisions

1. Hook seam home — arch trait method vs. serving-core free function taking the
   arch's residual buffer. (Lean: trait method, one call per block.)
2. DFlash teacher label transport — reuse `PflashLabels` JSONL+`QEMB` verbatim,
   or stream captured hidden states in-process to the trainer without a file
   round-trip. (Lean: in-process for the daemon-resident path; keep the file
   format for offline capture.)
3. Whether P3 lands at all this cycle, or DFlash (P2) is the stopping point for
   "trained-from-scratch" and LoRA is deferred.
