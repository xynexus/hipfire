# M8 — LoRA training interleaved with interactive decode, on halo (results)

Status: results, 2026-08-26. Host: halo, `gfx1151` Strix Halo, 128 GB UMA
(~120 GB GTT carveout), `HIP_VISIBLE_DEVICES=0` (one device at ordinal 0).
Brief: `2026-08-26-m8-halo-training-interleave.md`.

## Verdict: **PASS**, on the strong form of the criterion

Solo A and solo B came out **bit-identical**, so per the brief's §3.3 the noise
band has ZERO WIDTH and the exit criterion becomes "interleaved is bit-identical
too". It is.

| comparison | result |
|---|---|
| solo A vs solo B (200 steps) | **BIT-IDENTICAL** |
| solo A vs interleaved (200 steps) | **BIT-IDENTICAL** |
| worst-case deviation | **0.0** (exact equality, all 200 steps) |

This is the stricter test, not a widened one: the band was established *before*
the interleaved run, and it is not possible to widen a zero-width band to admit
a failure.

## Decode side verified (brief §3a)

A silently-dead decode stream would have "passed" the loss test by turning the
run back into solo training. Positive probe:

- **200 decode steps interleaved** (not 0, not 1) — one `generate` between every
  pair of training quanta.
- **800 `token` events** (200 requests x `max_tokens: 4`) and **200 `done`**.
- Output non-empty and non-degenerate: `<think>` / `\n` / `Thinking` / ` Process`.

The four tokens are identical across all 200 requests because the decode used
`temperature: 0.0` (greedy) against a fixed prompt — deterministic by
construction, which is what the brief asked for. That is reproducibility, not a
stuck stream; the `done` and per-request `id` counts confirm 200 distinct
completions.

## Residency (brief §6)

| point | `resident_vram_bytes` | workers | `residency_mode` |
|---|---|---|---|
| before load | 0 | 0 | — |
| after load | 19,258,128,446 (19.26 GB) | 1 | **`full`** |
| during a training quantum | 19,258,128,446 | 1 | **`full`** |

`total_resident_bytes` 19,355,735,061. The served model stayed resident across
the training quanta — that is the co-residency fact the experiment needed.

⚠️ **`rocm-smi` cannot substantiate this on halo, and the brief's §6 request for
it is a trap.** `rocm-smi --showmeminfo vram` reports **536,870,912 B** — the
512 MB *dedicated* carveout — flat across every run, because these models live
in the ~120 GB GTT pool that it does not report. Following §6 literally would
produce a flat 512 MB line and the false conclusion that nothing was resident.
Use the daemon's `resource_status` instead.

✅ **§5's "residency mode is chosen silently" is no longer true**: `residency_mode`
is reported per worker by `resource_status` (`"full"` here). No manual logging
needed.

## Exact configuration

- Training base: **tiny `llama` fixture**, `hipfire-quantize --emit-fixture llama
  --seed 4242` — 21 tensors, 3.28M params, fp32. **NOT** an fp32 7B (see
  Limitations).
- Served model: `~/.hipfire/models/Qwen3.5-35B-A3B--oq4.25++.hfq` (19.26 GB,
  arch `qwen3_5_moe`, 40 layers, dim 2048, vocab 248320), `max_seq: 2048`,
  **`dflash_mode: "off"`** (mandatory — see Appendix).
- LoRA: `steps: 200`, `rank: 16`, `alpha: 32.0`, `lr: 5e-3`, `seq: 8`,
  `n_seqs: 3`, `data: "overfit"` (the only implemented source).
- Quantum: **`quantum: 1`** — one training step per `train_lora` request, which
  is what makes the interleave step-granular and the per-step curve capturable.
- Decode: `temperature: 0.0`, `max_tokens: 4`, fixed prompt.
- VRAM budget / headroom: **unset** (`vram_budget_bytes: 0`,
  `vram_headroom_bytes: 0`) — the planner used its defaults.
- Wall clock: solo 200 steps **8.3 s**; interleaved 200 steps + 200 decodes +
  35B load/unload **69.9 s**.

## Raw per-step loss (`per_tok_ce`), 5 per line

Solo A and solo B and interleaved are byte-identical files; the curve is listed
once.

```
  8.413894 7.827491 7.065261 6.713826 6.436330
  6.176356 5.913191 5.704429 5.490046 5.411751
  5.226122 5.038175 4.822204 4.654042 4.462451
  4.313094 4.236345 4.217447 3.996294 3.888727
  3.769297 3.674859 3.603822 3.575516 3.420727
  3.376392 3.330315 3.253726 3.223500 3.202643
  3.123856 3.064865 3.009588 2.952779 2.920433
  2.972927 3.022901 2.951126 2.867376 2.777829
  2.712163 2.659835 2.611037 2.578292 2.531613
  2.509755 2.547092 2.601800 2.563832 2.501143
  2.487246 2.471749 2.440596 2.408884 2.397122
  2.367083 2.327164 2.312814 2.283405 2.266406
  2.278679 2.273813 2.232391 2.212556 2.208672
  2.222024 2.212754 2.217871 2.216717 2.215096
  2.212118 2.162958 2.137714 2.116361 2.105885
  2.114682 2.123541 2.157753 2.191607 2.150860
  2.170574 2.196475 2.146395 2.130678 2.121859
  2.097127 2.112319 2.097573 2.078952 2.089582
  2.080387 2.043418 2.050299 2.053015 2.035450
  2.054165 2.076674 2.068818 2.050099 2.023586
  2.021224 2.003148 2.005529 1.998448 1.984363
  1.970742 1.964607 1.957421 1.963565 1.999163
  2.036967 2.049869 2.014422 2.012700 1.992490
  1.985703 1.974232 1.979765 1.980262 2.044101
  2.013981 2.001065 1.985284 1.975462 1.954477
  1.958294 1.948417 1.929253 1.925651 1.934333
  1.920322 1.911379 1.918666 1.941381 1.983997
  1.938992 1.941324 1.921754 1.954098 1.960360
  1.963645 1.947956 1.908892 1.911790 1.906394
  1.894019 1.906955 1.900631 1.902659 1.888131
  1.890897 1.894017 1.886687 1.888059 1.897918
  1.916539 1.933738 1.913111 1.886366 1.882499
  1.893472 1.888130 1.887708 1.884511 1.886390
  1.897589 1.904176 1.913564 1.917898 1.892105
  1.892222 1.883066 1.875521 1.875820 1.875941
  1.876418 1.873855 1.869326 1.882604 1.894931
  1.865411 1.869683 1.877793 1.868669 1.866631
  1.880058 1.892808 1.898633 1.886240 1.870221
  1.879692 1.867270 1.850405 1.860231 1.860498
  1.866667 1.861732 1.858643 1.874301 1.903175
```

## Limitations — read before citing this as green

1. **The training-side residency claim in §2 was NOT exercised.** The brief's
   argument for halo is that a suspended `LoraTrainSession` holds an fp32 base
   (~28 GB for a 7B). This run's trainer is a 3.28M-param fixture. Co-residency
   with a 19.26 GB served model is real, and the served side is what created
   pressure — but "training as a co-resident *bulk* workload" is untested here.
   Redo with an fp32 7B+ base to close that half.
   *Why:* no complete Llama safetensors dir exists on halo. The 1B snapshot has
   only `original/consolidated.00.pth`; the 8B and 3B snapshots have
   `model.safetensors.index.json` with **no shards**. `hub fetch` with an
   `HF_TOKEN` cleared the initial 401 but then stalled resuming
   `model.safetensors` at 0.12 GB of 2.48 GB. Converting the `.pth` by hand was
   rejected: Meta->HF needs name mapping *and* the rotary permutation, and a
   subtle error there corrupts the base in a way the loss curve would absorb
   rather than expose.
2. **Training and decode are not sharing modules.** `handlers/train.rs` states it
   "trains hipfire-train's own un-fused `LlamaModel`, NOT the served qwen35
   adapters (a follow-on)". So this demonstrates *one executor, alternating
   quanta, both workloads resident* — not two workloads sharing module-granular
   residency of the same weights. The M8 wording admits the former; the stronger
   reading needs the follow-on.
3. **Two curves is what the brief asked for, and here it is enough** only because
   they were bit-identical. Had they differed, two samples would have been a
   crude band and should have been described as such.
4. `HIPFIRE_FORWARD_ORACLE` was **not** used, per the brief's warning that it is
   single-shot and corrupts the run it observes. No loss curve here was read off
   an oracle-enabled run.

## Appendix result — the `dflash_mode` question, captured

**Captured, not inferred.** Two loads of
`Qwen3.5-35B-A3B--oq4.25++.hfq` (lm_head `quant_type=36`, with
`Qwen3.5-35B-A3B--dflash.oq4+.hfq` present in the drafts dir):

- `dflash_mode` **ABSENT** -> `load failed: DFlash draft requested but target
  lm_head quant_type=36 is not supported ... (gfx1151)`. Reproduces the reported
  failure exactly.
- `dflash_mode: "off"` -> loads cleanly.

So the field was **absent** from the failing request, and absence means
`.unwrap_or("auto")` -> auto-pair a sibling drafter **by filename** -> refusal.

⚠️ **Correction to the brief.** It states `hipfire-eval` "does not" forward
`dflash_mode`. It **does** — `hipfire-eval/src/executor_daemon.rs:605,615` send
`dflash_mode: Some(config.dflash.as_str())`, defaulting to `"off"`. The accurate
part of the claim is that eval never consults `HipfireConfig`, which is a
different statement. A default `hipfire-eval` run therefore sends `"off"` and
**cannot** produce this failure.

Clients that actually **omit** the field, and so inherit `"auto"`:

| client | sends `dflash_mode`? |
|---|---|
| `hipfire-eval` (`executor_daemon.rs`) | yes — own default `"off"` |
| `hipfire-cli chat` | yes |
| `hipfire-model` (`load_params_from_config`) | yes |
| **`hipfire-cli bench`** | **no** |
| **`hipfire-daemon-adapter`** | **no** |

Also worth recording: the refusal text says Opus `oq*` (qt=33/34/35/36/38/52) is
"CORRECT now but measured slower than plain decode on this family", gated behind
`HIPFIRE_DFLASH_ALLOW_OPUS=1`. That gate encodes a real measurement, but it
predates the DFlash rollback work landed on
`fix/oq-compact-moe-prefill-silu-rotate` (checkpoint rollback, +31% mean, spec
beating AR on all six prompts) — so the "slower than plain decode" premise is
worth re-measuring before the gate is treated as settled.

I did not investigate `qwen_moe_module_capable_model`'s `lower.contains("a")`
condition: residency came back `full`, which is the expected mode here, so
nothing surprising pointed at it.

## Reproduce

```sh
hipfire-quantize --emit-fixture llama --out DIR --seed 4242
# solo: 200 x {"type":"train_lora","run_id":"soloA","base":DIR,"output":OUT,
#              "quantum":1,"train":{"steps":200,"rank":16,"seq":8,"n_seqs":3,
#              "alpha":32.0,"lr":5e-3},"data":"overfit"}
# interleaved: same, with a {"type":"generate",...} between every quantum,
#              after {"type":"load",...,"params":{"dflash_mode":"off"}}
./target/release/hipfire-daemon < run.jsonl
```

⚠️ `steps` **must be nested under `train`** — `getu` reads only from that object
(`handlers/train.rs:508`). A top-level `"steps": 5` is silently ignored and the
run uses the 200 default, so a run can report a configuration it did not use.
