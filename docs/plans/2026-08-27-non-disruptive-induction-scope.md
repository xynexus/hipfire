# Scope: induct a model without interrupting the inference service

Status: scope, 2026-08-27. Goal stated by the user: **run a full induction —
calibration, KLD-ref generation, quantization, QAT — co-resident with live
serving, producing `oq4.25+++` artifacts** (third `+` = QAT).

Evidence base: the co-residency measurements of 2026-08-26
(`docs/experiments/2026-08-26-*.md`). Those established the substrate works —
N models co-reside, the resident swap is free, co-residency costs +0.20% on
inference. This scopes what induction specifically still needs on top.

---

## 1. Where each induction function stands

| function | daemon-reachable | quantum | preemptible | verdict |
|---|---|---|---|---|
| **Calibration** | ✅ `calibrate` | **per layer** | ✅ | **ready** |
| **Evidence capture** (Hessian/imatrix) | ✅ `collect` | per request | ⚠️ unverified | likely ready |
| **KLD-ref generation** | ✅ `kld_eval mode=build_ref` | **none** | ❌ | **needs a quantum** |
| **Quantization** | ❌ not in protocol | — | ❌ | **needs wiring** |
| **QAT — KV path** | ✅ `train_lora` + `HIPFIRE_KVNOISE=1` | per step | ✅ | **ready** |
| **QAT — Opus weights** (`+++`) | ❌ examples only | — | ❌ | **needs wiring** |

### Calibration — already the model to copy

`handlers/calibrate.rs` runs `DaemonCalibration` **in-process** (it reuses the
CLI arg parser to build a `CalibrateCommand`, it does not spawn), and advances
one layer per request:

```rust
sess.session.step(&mut daemon_state.gpu)   // -> Advanced | Paused | LayersComplete
```

That is exactly the quantum shape M8 validated for training. Calibration needs
nothing new to interleave with serving.

### KLD-ref generation — the one that blocks

`kld_eval` reaches `hipfire_runtime::kld_eval::kld_self_score(...)` with a
per-chunk **callback** that emits `{"type":"kld_chunk", …}`. Progress is
reported per chunk, but the call is **monolithic**: it scores every chunk before
returning, with no yield point and no resume. A `build_ref` over a real corpus
therefore occupies the daemon for its full duration — minutes to hours — and
serving stops dead.

This is the single largest gap, and it is *not* an architectural one: the chunk
loop already exists and already has a callback boundary. It needs the loop
inverted into a stepping session (`step()` → `Advanced`/`Paused`/`Complete`)
with resumable chunk-cursor state, mirroring `CalibrationStep`.

### Quantization — not architecturally blocked, just unwired

`hipfire-quantize` **is a library** (`codecs.rs`, `gptq.rs`, `hessian_io.rs`, …)
with GPU behind an optional `gpu` feature, not a bin-only crate. So codecs are
callable in-process. What is missing:

- the daemon has **no dependency** on it and **no `quantize` request type**;
- today's path is the standalone `hipfire-quantize` binary, which per AGENTS.md
  must be coordinated with `hipfire lock` — and that lock is held by the serving
  daemon, so the two cannot run concurrently at all.

**This needs an AGENTS.md boundary ruling before it is built.** The contract says
format conversion belongs in `hipfire-coexistence`, while "if it runs kernels
over model weights it may be scheduled by the daemon". Quantization does *both*:
clip search and Hessian/LDLQ are kernel work over weights, while emitting the
`.hfq` is container writing. My reading is that the split already implied by the
contract is the right one — **the search/solve half becomes daemon-schedulable
work; the container write stays offline** — but that is a judgement call about a
stated invariant and should be confirmed rather than assumed.

### QAT — two different things, only one reachable

**KV-path QAT works today.** `kv_noise.rs` is wired into `block.rs:380` behind
`HIPFIRE_KVNOISE=1`: KVarN-4bit round-trip plus CASK cold-token merge injected
forward-only, gradients flowing to q/v projections as identity — "exactly
QAT-with-STE on the KV path". It rides the `train_lora` quantum (`quantum: 1`),
so it is co-resident-ready now.

**Opus weight QAT — the `+++` — is not reachable.** `oqplus_quant.rs` ("OQ+
(Opus Plus, W4A8) sim-quant for recovery-FT probes") runs the real
`quantize_oq4g256` codec as a differentiable fp32→fp32 round-trip, with
`oq3_simquant`/`oq8_simquant` siblings and `a4_quant.rs` for the activation side.
But **every caller lives in `examples/`** — `qat_w3_kvarn.rs`,
`w3_codec_compare.rs`, `qwen35_mlp_norm_recovery.rs`. Nothing in `block.rs`,
`train_loop.rs`, or `model.rs` calls them, and unlike `kv_noise` there is **no
env gate**. `examples/qat_w3_kvarn.rs` is the closest thing to the target
harness — frozen Oq3-fake-quantized weights, trainable LoRA(q/v)+RMSNorm,
KL-distilled against a clean fp32 teacher, scored in-sample *and* held-out — but
it is a standalone GPU binary that wants `hipfire lock acquire`, which deadlocks
against a serving daemon.

---

## 2. Cross-cutting work

### 2.1 The `+++` token does not exist in the naming spec

AGENTS.md defines the quant token as `<family><bitwidth>[l][+][+]`: first `+` =
clip-search/SmoothQuant/AWQ, second `+` = Hessian/LDLQ error feedback. **There is
no third `+`.** `oq4.25+++` therefore needs:

- the AGENTS.md grammar extended with the QAT meaning;
- provenance in the HFQ metadata, so the claim is checkable from the artifact
  rather than asserted by its filename — a name is not evidence;
- `scripts/check-artifact-names.sh` currently only rejects legacy patterns and
  does not validate the quant grammar, so it will not catch a malformed token
  either way. Worth tightening while the grammar is being changed.

### 2.2 Admission control exists but has no caller — and the quanta are far too long

**Measured 2026-08-27**, see `docs/experiments/2026-08-27-induction-quantum-wcet-nix1.md`:

| quantum | WCET | vs the 200 ms contract |
|---|---|---|
| QAT step (1B fp32) | **15.3 s** | 77× |
| KLD chunk (n_ctx 1024) | **280.8 s** | 1,404× |
| serving baseline | 2.455 s | — |
| serving *between* QAT quanta | 2.884 s (**+17.5%**) | — |

This changes item 3 of §3 below. Giving `kld_eval` a `step()` is **necessary but
not sufficient**: a 280 s chunk is not a quantum in any useful sense, and
`admit_realtime` can only decide whether to *start* one — nothing preempts a
step once running. Both stages need **sub-quantum yielding**, at a layer or
token-block boundary, which is a stronger requirement than exposing a step
function.

The +17.5% steady-state tax is the encouraging half: that is what interleaving
costs *between* quanta, and it is a number a policy could trade against.

⚠️ The KLD figure may be a batching fallback rather than intrinsic cost —
0.274 s/token is decode-speed, and the run logged `KV cache: Q8` despite loading
`kvarn`. `HIPFIRE_KERNEL_TRACE` was not on, so it is unresolved. Settle it before
designing around 280 s.

`admit_realtime` (merged in #350) prices the drain budget including the prefill
term. **Nothing calls it.** For non-disruptive induction it needs to actually
gate: an induction quantum should be admitted only when serving's latency budget
tolerates it, otherwise the "non-disruptive" property is aspirational.

The WCETs above are the input it needed. What they show is that gating alone
cannot deliver the property — the quanta must also shrink. Calibration's
per-layer step is the one induction quantum still unmeasured; it is the most
likely to already be short enough, and worth measuring next.

### 2.3 Memory is the binding constraint, and the ledger lies

From `2026-08-26-load-eviction-semantics-nix1.md`:

- there is **no pressure-driven eviction** anywhere on the daemon path
  (`resident_models` has no policy; `plan_model_residency` is server-side only);
- an over-budget load fails with a clean `hipMalloc` OOM — survivors intact, and
  the retained pool memory is reused rather than leaked;
- **the daemon ledger understates real GTT by ~8.5 GB** (31.25 GB reported vs
  39.47 GB actual at three models), because it counts weights only.

Induction has a large transient working set (fp32 master weights, Hessians,
teacher logits). Any admission decision must size against
`mem_info_gtt_used`, **not** `total_model_weight_bytes` and never `rocm-smi`
(which reports only the 0.2 GB dedicated carveout on these APUs).

### 2.4 The GPU lock is a hard serialiser

Every remaining standalone path — `hipfire-quantize`, the QAT examples — takes
`hipfire lock`, which the serving daemon already holds. Non-disruptive induction
means **no induction step may run outside the daemon process**. That is the real
reason quantization and Opus-QAT must be wired in rather than shelled out.

---

## 3. Proposed order

Sequenced so each step is independently verifiable and the cheap risk-reducers
come first.

1. **Measure per-quantum WCET** for the two ready functions (calibration layer,
   QAT-KV step) while serving. Cheap, needs no new code, and produces the input
   §2.2 requires. Run it on nix1 with the small-model pair.
2. **Run the co-resident QAT-KV + serving test** (the standing TODO,
   `docs/todo/2026-08-26-coresident-training-m8.md`). Proves the interleave end
   to end with a real QAT flavour before any new code lands.
3. **Give `kld_eval` a chunk quantum** — invert the internal loop into a
   resumable stepping session mirroring `CalibrationStep`. Highest value per
   unit of work: it converts the one function that currently *halts serving*
   into an interleavable one.
4. **Wire Opus weight sim-quant into the training loop** behind an env gate
   mirroring `HIPFIRE_KVNOISE`, then expose it as a `train_lora` parameter. This
   is what makes the third `+` mean something.
5. **Settle the quantization boundary** (§1, AGENTS.md ruling), then wire the
   search/solve half as daemon-schedulable work.
6. **Extend the naming grammar and add metadata provenance** for `+++`.
7. **Make `admit_realtime` a real gate** using the WCETs from (1).

Items 1–3 need no invariant changes and can proceed immediately. Items 4–5 touch
stated contracts and want confirmation first.

---

## 4. What is explicitly *not* in scope here

- **Concurrent execution.** Everything measured so far is fast *serial*
  switching — one active model at a time. Induction interleaved with serving
  means alternating quanta, not simultaneous kernels.
- **Module-granular sharing between workloads.** `handlers/train.rs` trains
  hipfire-train's own un-fused `LlamaModel`, not the served adapters, so training
  and serving are two resident workloads rather than two views of one.
- **Eviction/downgrade under pressure**, which needs the HTTP server path.

## 5. Open questions for the user

1. **The AGENTS.md boundary for quantization** (§1) — does the search/solve half
   move into the daemon, or does induction stop at "everything except the
   quantize step" and accept one brief serving pause for that?
2. **Is `+++` strictly Opus-weight QAT**, or does KV-path QAT (which works
   today) also earn it? They are different claims about an artifact and the
   grammar should say which.
3. **What latency degradation is acceptable** while inducting? "+0.20% from
   co-residency" is measured; an actively-running induction quantum will cost
   more, and §2.2's gate needs a number to enforce.
