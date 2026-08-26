# M8 — LoRA training interleaved with interactive decode, on halo

Status: experiment brief, 2026-08-26. Written to be handed to a Claude instance
running **on halo** (`gfx1151`, Strix Halo, 128 GB total / ~120 GB GTT carveout).

You are the halo instance. This document is self-contained: it assumes you have
the repo and no prior conversation context. Read it fully before running
anything — several of the traps below cost the nix1 side hours.

---

## 1. What is being tested

From the v2 daemon plan (`docs/plans/2026-08-09-v2-daemon-module-major-multistream.md`,
§M8):

> *Exit:* a LoRA training step and an interactive decode step interleaved at
> module granularity in one executor, with the loss curve over 200 steps inside
> the run-to-run noise band established by two solo runs.

The claim under test is **not** "training works" and **not** "decode works". It
is that *interleaving them at module granularity does not change the training
result*. The loss curve is the instrument; the noise band is what makes it an
instrument rather than a vibe.

## 2. Why this runs on halo and not nix1

nix1 is size-limited, and the arithmetic is in the plan:

> a suspended `LoraTrainSession` holds the base model in fp32, ~28 GB for a 7B.
> On nix1's 64 GB UMA, one fp32 7B base plus a served MoE plus the cache budget
> does not fit. **Training as a co-resident bulk workload is size-limited on
> this box regardless of scheduling.**

So on nix1 the experiment would either OOM or be shrunk until it no longer
exercises co-residency — and a green result from a model that fits comfortably
is **evidence of nothing**, because nothing is ever evicted or suspended under
pressure. halo's ~120 GB carveout is what makes the pressure real.

**Pin the device**: `HIP_VISIBLE_DEVICES=0`. A live `hipGetDeviceCount` probe on
2026-07-23 confirmed halo has exactly one HIP device at ordinal 0; pinning
ordinal 1 returns `hipErrorNoDevice`.

## 3. Protocol — order matters

The single most important thing in this document: **establish the noise band
before you run the interleaved case.** If you run interleaved first and then go
looking for a band that contains it, you have built a test that cannot fail.

1. **Solo run A** — LoRA training alone, 200 steps. Record loss at every step.
2. **Solo run B** — identical configuration, different seed *only if* the
   training path is seeded independently of the sampler; otherwise identical in
   every respect. Record loss at every step.
3. **Compute the band.** At each step *i*, the band is `[min(A_i, B_i),
   max(A_i, B_i)]` widened by the observed step-to-step variance. Two runs give
   you a crude band — say so in the writeup rather than implying it is tight.
   If A and B are *bit-identical*, that is a finding in itself: it means the
   training path is fully deterministic, the band has zero width, and the exit
   criterion becomes "interleaved is bit-identical too", which is a much
   stronger and much easier test. **Check for this first** — it may save you
   the whole statistical exercise.
4. **Interleaved run** — the same 200 training steps, with an interactive decode
   stream running concurrently against a second resident worker. Record loss at
   every step, *and* record the decode stream's own output.
5. **Verdict.** Pass = every interleaved loss sample lies inside the band. Fail =
   any sample outside it. Report the worst-case deviation either way.

### 3a. The decode side must be verified too

A run where the decode stream silently produced nothing would trivially "pass"
the loss test — the training would just be solo with extra steps. So:

- Capture the decode stream's tokens and assert it produced a non-empty,
  non-degenerate output.
- Record how many decode steps actually interleaved. If the answer is 0 or 1,
  the experiment did not run regardless of what the loss curve says.

This is the same failure the nix1 side hit repeatedly: **equal output never
proves the path ran.** Add a positive probe that fails if the interleaving did
not happen.

## 4. Traps specific to this repo and this host

These are all things that have already bitten someone.

**Do not wrap `hipfire-eval` in `hipfire lock`.** It loads through the daemon,
which acquires the resource lock itself. Wrapping it deadlocks against your own
holder, and the error names *your* label as the blocker:
`resource hip-gpu-0 ... is locked by <your label> ... mode=run`. Run it directly.
Other non-daemon GPU binaries (examples, benches, `hipfire-quantize`) *do* need
`hipfire lock {acquire,release,status}`.

**`pkill -f` will kill your own shell.** `-f` matches the full command line of
the shell running it, so `pkill -f hipfire-daemon` in a wrapper script kills the
script and surfaces as a bare `exit 144` with no output. The `[p]attern` bracket
trick does *not* save you. Either kill a pid you recorded yourself, or use
`scripts/pkill-safe.sh` (same flags and exit codes as pkill, refuses to signal
the caller, any ancestor, or the caller's process group). Note that inside a
Claude Code command shell a `pkill` *function* shadows the PATH symlink, so call
the script **by path** if you want the caller-shell guarantee.

**Artifacts must be on local NVMe, not `/srv`.** `/srv` is an NFS mount from
carbon (172.16.16.254). Any measurement that touches artifact bytes — paging,
cold-load, streaming residency — would be measuring the network instead. Copy
what you need locally first.

**Do not run `hipfire optimize` on a pageable artifact.** The quantizer emits
canonical `Oq4G256`, which is what `WeightPager::register_expert_module` accepts.
`hipfire optimize` converts it to dense `Oq4G256ArchPacked`, which the pager
*refuses*. The failure presents as a paging/residency bug rather than as an
artifact mistake, which is what makes it expensive.

**`HIPFIRE_FORWARD_ORACLE` perturbs the run.** The plan lists an oracle dual-run
as the check for numeric-path collapse in M8. Be aware of how it is actually
implemented: it is **single-shot and it corrupts the run it observes** — it
captures lowered logits, rewinds DeltaNet state, then falls through to the hand
arms, and KVarN's window ring is not idempotent on a second write. An early
version reported `mismatched=0` while the greedy output had degenerated to
`ABrsteqeqeqtual`. It now prints an explicit `PERTURBED` warning. **Use it as a
separate diagnostic run; never read a loss curve off an oracle-enabled run.**

## 5. Residency configuration

Relevant knobs (all in `hipfire-config`, settable per the file → env → CLI
layering):

- `model_residency_mode` — `auto` | `full` | `qwen_moe_modules`. `auto` tries
  `full` and falls back to module-granular residency if it does not fit.
- `scheduler_vram_budget_bytes` / `scheduler_vram_headroom_bytes` — the budget
  the residency planner sizes against. Both are also read by
  `vram_ceiling`, so declaring them makes both sides size against one number.
- `sampler_rng` — `fixed` | `random`. For this experiment set it to **`fixed`**
  on the decode stream so the decode side is reproducible across your three
  runs; `random` seeds each stream from entropy and would make the decode output
  differ run to run for reasons unrelated to interleaving.

Record the values you actually used.

**Corrected 2026-08-26 by the halo run:** this section used to say
`plan_model_residency` chooses the mode silently and told you to log it by hand.
It does not need logging — the daemon's `resource_status` reports
`residency_mode` per worker. (The `reason` field on the plan itself is still
`"admitted"` either way, which is what the original claim was reading.)

## 6. What to bring back

- The three loss curves (A, B, interleaved) as raw per-step numbers, not a plot.
- Whether A and B were bit-identical.
- Interleaved decode step count and a sample of its output.
- The verdict, with the worst-case deviation.
- Exact config: residency mode actually used, VRAM budget/headroom, model paths,
  `sampler_rng`, step count, LoRA rank/target modules.
- Evidence that the served model stayed resident across the training quanta —
  the residency claim is half the point of running on halo.

  ⚠️ **Corrected 2026-08-26: do NOT use `rocm-smi` for this on halo.** This line
  used to ask for `rocm-smi` peak memory, and that is a trap on this host:
  `rocm-smi --showmeminfo vram` reports the 512 MB *dedicated* carveout
  (536,870,912 B), flat across every run, because these models live in the
  ~120 GB GTT pool it cannot see. Following it literally yields a flat line and
  the false conclusion that nothing was resident. Use the daemon's
  `resource_status` (`resident_vram_bytes`, `total_resident_bytes`, per-worker
  `residency_mode`) instead.
- Anything that surprised you. Failed approaches are worth documenting; they
  narrow the search space.

Write it up as `docs/experiments/2026-08-26-m8-halo-training-interleave-results.md`.

---

## Appendix — a 5-minute side task while you are on halo

There is an open question the nix1 side could not settle remotely, and it is
cheap to answer with a shell on the box.

**Background.** A load on halo failed with `DFlash draft requested but target
lm_head quant_type=36 is not supported`, despite nobody asking for a drafter. A
claim was published that `HIPFIRE_DFLASH_MODE=off` "lost to the per-model config
layer" — that claim was **retracted** as an unverified guess. What the code
actually says:

- `env_var_name_for_key` maps `dflash_mode` → `HIPFIRE_DFLASH_MODE`, and `off`
  is a valid `ConfigType::Enum` value, so the env layer parses fine.
- The **daemon never reads that env var**. `dflash_mode` arrives only inside the
  load request params (`hipfire-daemon/src/handlers/lifecycle.rs:225`), and
  `.unwrap_or("auto")` when the field is absent.
- Clients calling `load_params_from_config` forward it — the CLI chat path and
  `hipfire-server`.
- **Corrected 2026-08-26 by the halo run.** An earlier version of this line said
  "`hipfire-eval` does not [forward it]". That is **wrong**, and the grep that
  disproves it was already on screen when it was written:
  `hipfire-eval/src/executor_daemon.rs:605,615` both send
  `dflash_mode: Some(config.dflash.as_str().to_string())`. Eval forwards its own
  `--dflash` (default `Off`, `hipfire-eval/src/config.rs:43`) — so a default eval
  run sends `"off"` and *cannot* cause the reported failure. The accurate half of
  the original claim is only that eval never consults `HipfireConfig`, so
  `HIPFIRE_DFLASH_MODE` is invisible to it.
- The clients that **omit** the field entirely — and therefore hit the
  `.unwrap_or("auto")` default — are **`hipfire-cli bench`** (nothing in
  `commands/bench.rs`) and **`hipfire-daemon-adapter`** (no occurrence at all).
- Anything other than `off` then lets `hipfire-serving-core/src/load.rs:857`
  auto-pair a sibling drafter **by filename** from the drafts directory. halo has
  `Qwen3.5-35B-A3B--dflash.oq4+.hfq` sitting there.

**What to do:** capture the actual load request the daemon received on halo and
report the `dflash_mode` field — present and what value, or absent. That single
fact distinguishes "the env var never reached the request" from "it was never
consulted", and closes the question. Do not reason about it further; capture it.

**Note also** `qwen_moe_module_capable_model` (`hipfire-server/src/routes/chat.rs:773`)
decides residency strategy from a **filename substring**, and one of its
conditions is `lower.contains("a")` — very nearly a tautology for any real path.
If you happen to observe a surprising residency mode on halo, that is a strong
candidate for why. Worth a note either way.
