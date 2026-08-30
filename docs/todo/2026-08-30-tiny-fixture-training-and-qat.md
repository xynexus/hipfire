# TODO: train the tiny fixtures, then add a tiny QAT cell

**Status:** open (proposed 2026-08-30).

Two phases sharing one blocker. Phase 1 makes the tiny-quant numbers mean
something; Phase 2 is the coverage that only becomes possible once they do.

## Motivation: a random-init fixture cannot rank quantizers

`tiny_quant` (`crates/hipfire-eval/src/executor_tinyquant.rs`) scores each
family's quantizer→loader→dequant path by KLD against a per-family anchor, over
a fixture whose weights are **random-init** — every family's tensors come from
its `-spec` crate's `ToyModel::fixture_named(name, seed)`
(`crates/hipfire-quantize/src/fixture.rs`), generated in-process from a seed.

A random-init model has no learned structure, so its output distribution is not
a *quality* signal. KLD-vs-anchor then measures how much a codec **perturbs** an
arbitrary function, not how much it **degrades** a good one — and those two are
not the same ordering. Four things measured in this tree say so:

1. **More fidelity scored worse.** Moving the protected set from `Q8F16` to
   `BF16` (strictly more precision, same everything else) made `nemotron_h`
   `oq8` go 0.00000533 → 0.00001033 — **2× worse** — while the same change made
   `qwen3_5_moe_indexed` `oq8` 0.00194777 → 0.00028148, **6.9× better**. One
   change, opposite signs, no quality interpretation that fits both.
2. **Calibration scores as harmful on some families.** `qwen2` `oq4` 0.00453442
   → `oq4+` 0.03704516 is calibration coming out **8.1× worse**, where `llama`
   has it helping (0.00829894 → 0.00704459). Same machinery. On a trained model
   AWQ/clip-search has a defensible sign; here it does not.
3. **Vacuous cells are invisible.** `zaya`'s `oq4` cell measured nothing for
   months because its whole attention stack was being promoted to `Q8F16` (see
   `BUGS.md`); the numbers looked unremarkable throughout. Nothing in a
   random-init KLD distinguishes "codec is fine" from "codec never ran".
4. **The budget is near the noise.** `zaya`'s calib rows move up to 1.20×
   between two runs of the *same binary*, against a `0.25` relative budget.

The consequence is that the gate today is a strong **crash/coherence** detector
and a weak **quality** detector. It reliably catches non-finite output, a
missing dispatch arm, and a layout regression. It cannot answer "is oq4+ better
than oq4", which is the question the quant work actually asks.

## Phase 1 — trained tiny fixtures

Give each of the 20 fixture families a briefly-trained checkpoint, so its
logits reflect learned structure and KLD regains its usual meaning.

`hipfire-train` is real fp32 GPU autograd (`train_loop.rs`, `optim.rs`,
`model.rs`) and already trains models of this shape, so the training itself is
not new work — the wiring and the artefact handling are.

**Do not train inside the gate.** `tiny-state` asserts bit-exact hashes, and GPU
training is not bit-reproducible (atomics, non-deterministic reduction order).
Train once, store the weights, and have the gate load them.

Storage is the open question. A fixture source is ~7 MB at bf16 (`zaya`), so 20
families is ~140 MB — too much to commit. Options, cheapest first:

- Shrink the fixtures for the trained variant (fewer layers/width) so the set
  fits in-tree. Best if it holds: no new infrastructure, no fetch path.
- Publish a trained-fixture artefact set and fetch on demand, with a checksum
  manifest committed. Note the shared-artefact share is local infrastructure and
  must not be assumed by committed code or docs (`AGENTS.md`).
- Commit a training recipe + seed and regenerate. Rejected unless training is
  made deterministic — otherwise every re-generation silently re-centres every
  baseline.

Keep the random-init fixtures as well. They are the cheap crash tier and cost
nothing to keep; the trained ones are a second tier that gates quality. That
also preserves every currently-recorded baseline instead of invalidating it.

**Done when:** on a trained fixture, `oq8` beats `oq4` and `oq4+` beats `oq4` on
every family, or the exception is explained. That ordering failing today is the
whole problem.

## Phase 2 — a tiny QAT cell

Once a fixture is trainable, QAT coverage follows, and the substrate is already
here: `oqplus_quant.rs`, `qtip_quant.rs`, `a4_quant.rs` (fake-quant forward),
`kv_noise.rs` (KV noise injection), `learn_rotation.rs` (Cayley-SGD rotations).
What is missing is a **test** that any of it still works end to end.

Shape: take a trained fixture, quantize to a lossy target, run a short
recovery-FT with the weights frozen and a STE fake-quant forward, and assert the
recovered KLD beats the un-recovered one by a recorded margin.

Worth covering, in rough priority:

- **W3/W4 recovery-FT** — the main claim (see `docs/` on light QAT recovery:
  W3 loss ~52% recoverable). Nothing tests it.
- **KVarN STE** — recorded as *non*-recoverable at 4-bit, which is a deployment
  decision (ship KVarN-8, not KV4) resting on an untested code path.
- **Learned rotation** — `learn_rotation.rs` has a unit test for the Cayley
  solve, but nothing checks the rotation actually improves a quantized model.

This tier is slower than `tiny-quant` (it trains), so it belongs behind its own
gate script rather than in the `tiny-affected` front tier.

## Open questions

- Do all 20 families train under `hipfire-train`, or only the simple-AR ones?
  Mamba-2, DeltaNet, and MoE routing each need a backward path.
- How few steps still produce a model whose quantization ordering is stable? If
  a few hundred steps suffice the artefact problem shrinks a lot.
- Should the trained fixture replace the anchor too, or stay a separate tier?
