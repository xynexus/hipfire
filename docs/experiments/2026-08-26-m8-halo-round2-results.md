# M8 round 2 — negative control passes; module-granular residency also bit-identical

Status: results, 2026-08-26. Host: halo (`gfx1151`, 128 GB UMA), run over SSH
from nix1. Round 1: `2026-08-26-m8-halo-training-interleave-results.md`.
Brief + round-2 instructions: `2026-08-26-m8-halo-training-interleave.md`.

Round 1 returned PASS and was honest about what it had not exercised. Round 2
closes the two gaps that were closable on the daemon path, and reports two that
were not — including one where the round-2 instructions themselves were wrong.

## 1. The negative control — round 1's PASS is now falsifiable

Round 1's three curves were bit-identical, which is strong **only if the
comparison could have registered a difference**. Nothing in round 1 showed that
it could, and round 1 had itself found a silent-config-fallback bug (`steps` not
nested under `train`) of exactly the shape that would hide here.

Reusing round 1's own harness with `lr` perturbed 0.005 → 0.0051 (+2%):

| comparison | result |
|---|---|
| soloA vs soloB | **IDENTICAL** (round 1 reproduced) |
| soloA vs interleaved | **IDENTICAL** (round 1 reproduced) |
| soloA vs **negctl (lr +2%)** | **DIFFER** |

The perturbation appears at **step 1** — `8.413893700` vs `8.413988113`,
Δ 9.44e-05 — and grows to Δ 0.023396 by step 200.

So the instrument detects a change four orders of magnitude below the loss
values themselves, at the very first step, while the interleaved run is exactly
zero. **Round 1's verdict stands, and now on evidence that could have refuted
it.**

## 2. Module-granular residency — also bit-identical

M8's exit says *interleaved at module granularity*. Round 1 ran
`residency_mode: "full"`, the weaker reading. Round 2 re-ran the interleave with
the served model loaded as:

```json
{"type":"load","model":".../Qwen3.5-35B-A3B--oq4.25++.hfq",
 "params":{"max_seq":2048,"dflash_mode":"off",
           "residency_mode":"qwen_moe_modules",
           "module_vram_budget_bytes":4294967296}}
```

200 training quanta (`quantum: 1`) interleaved with 200 generates.

| check | result |
|---|---|
| loss vs round-1 solo | **IDENTICAL, all 200 steps** |
| loss vs negative control | differs (instrument still live) |
| `residency_mode` at 3 probes | `qwen_moe_modules` |
| paging active | `paged` appears 42× in the run log |

So interleaving does not perturb training under module-granular residency
either. That satisfies the stronger reading of M8's wording on the *served*
side.

### The cost: ~15× wall clock

| run | residency | wall |
|---|---|---|
| round 1 interleave | `full` | **69.9 s** |
| round 2 interleave | `qwen_moe_modules` (4 GiB cache) | **~18 min** (14:13:48 → 14:31:50) |

Same step count, same work, ~15.5× slower with a 4 GiB expert cache against a
19.26 GB model. That is an M5 paging datapoint, not an M8 result, but it is the
number anyone proposing module-granular residency as a *default* has to answer.

## 3. ⚠️ `resident_vram_bytes` is self-fulfilling — do not cite it as measurement

The probes reported `resident_vram_bytes: 4294967296` — exactly the
`module_vram_budget_bytes` that was sent. That is not a coincidence and not a
measurement:

```rust
let vram_bytes = if residency_mode == "qwen_moe_modules" {
    module_budget.unwrap_or(file_bytes)      // <- reports back the input
} else { file_bytes };                       // <- or the file size
```

`ResidentResourceUsage` is an **accounting ledger built from the load request**,
not an observation of the device. In `full` mode it reports the model's file
size; in module mode it reports the budget you asked for. Round 1's 19.26 GB
figure is therefore also the artifact's file size echoed back, not measured
residency.

This matters because it is the number both rounds used to substantiate
co-residency, and it cannot substantiate anything — it would report the same
value if nothing were resident at all. The honest evidence that paging happened
is behavioural: the 15× slowdown and the 42 `paged` log lines.

**Neither `rocm-smi` (reports the 512 MB dedicated carveout, per round 1) nor
`resource_status` currently measures GTT residency on this host.** Establishing
real residency numbers needs a third instrument that does not exist yet.

## 4. The round-2 instruction about VRAM budgets was wrong

Round-2 item 1 said to set `scheduler_vram_budget_bytes` below what the run
needs and report whether the planner fit, downgraded, or evicted. **That cannot
be done on the path either round used.**

- `plan_model_residency` — the function that evicts, downgrades and refuses —
  is called from exactly one place: `hipfire-server/src/routes/chat.rs`.
- **The daemon never calls it.** Both rounds drove the daemon's stdin JSON
  protocol, which has no residency planner at all.
- The daemon *does* read `HIPFIRE_SCHEDULER_VRAM_BUDGET_BYTES`, but only to size
  the **ballast allocation** (`release_placeholders` / `reacquire_placeholders`)
  — not to make a residency decision.

So round 1's "unlimited budget" was real but is the *second* reason it could not
test residency; the first is that its transport bypasses the planner entirely.
Testing eviction/downgrade requires driving `hipfire-server` over HTTP. Left
undone here, and now stated as a path requirement rather than a config value.

## 5. The varied-prompt probe was mis-designed, and the fix confirms the stream

Round-2 item 4 asked for a different prompt per request so that identical output
would be a *failure* signal. It fired: 200 completions, **2 distinct texts**.

That was the probe's fault, not the stream's. With `max_tokens: 4` against a
thinking model, every prompt emits the same `<think>\nThinking Process` preamble
— four tokens cannot reach the prompt-dependent part, so the probe could not
discriminate no matter how healthy the stream was.

Re-run with `max_tokens: 48` over six prompts: **6 completions, 6 distinct**,
sharing a 94-character prefix and then diverging correctly:

```
shared prefix (94 chars): '...The user is asking for the capital of '
  p0: 'France.\n2.  **Retrieve Knowledge:** ...'
  p1: 'Japan.\n2.   **Retrieve Knowledge:** ...'
  p2: 'Brazil.\n2.  **Retrieve Knowledge:** ...'
  p3: 'Kenya.\n2.   **Retrieve Knowledge:** ...'
  p4: 'Norway.\n2.  **Retrieve Knowledge:** ...'
  p5: 'Peru.\n2.    **Retrieve Knowledge:** ...'
```

The decode stream is live and prompt-sensitive. **Lesson for the brief: a
liveness probe on a thinking model must generate past the reasoning preamble,
or a shared prefix will mask a working stream (and would equally mask a broken
one).**

## What remains open

- **Training-side residency pressure** — still untested. Unchanged from round 1:
  the trainer is a 3.28M fixture, and `emit_fixture(arch, out_dir, seed)` takes
  no shape parameters. Adding them remains the cheapest route to a rerunnable
  pressure test.
- **Eviction / downgrade / refuse** — needs the HTTP server path (§4).
- **A real residency measurement** — needs an instrument that observes GTT,
  since neither existing one does (§3).
- **Module sharing between workloads** — `handlers/train.rs` still trains its own
  un-fused `LlamaModel`, so this is one executor with two resident workloads, not
  two workloads sharing modules.

## Reproducing

Harness and artifacts on halo under `/tmp/claude-1000/m8/`: `negctl.jsonl`
(negative control), `r2mod.jsonl` (module-residency interleave), `probe2.jsonl`
(discriminating decode probe), with `.log` and `.ce` outputs alongside round 1's.

Run detached — the module-residency interleave takes ~18 minutes and an SSH
timeout will otherwise appear to kill it (it does not; the daemon survives, but
the connection carrying its output does not).
