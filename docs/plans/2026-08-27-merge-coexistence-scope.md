# Scope: merge `hipfire-coexistence` back into hipfire

Status: scope, 2026-08-27. Premise stated by the user: *"the idea of keeping
external code requirements out of hipfire core was a failure."* This documents
the evidence for that, and what merging actually costs.

It is not an independent question from
`2026-08-27-non-disruptive-induction-scope.md` — the same boundary is what
blocks daemon-hosted induction. See §3.

---

## 1. The separation did not do what it was for

The stated purpose was to quarantine external-ecosystem code from the inference
core. Measured against the crate as it exists:

**It quarantines no exotic dependencies.** The complete non-hipfire dependency
list is:

```
twox-hash  tokio  serde  serde_json  sha2  base64  blake3  indicatif  chrono  libc
```

Ten ordinary crates, every one of which is either already in the workspace or
utterly unremarkable. **No `safetensors`, no `gguf` crate, no `hf-hub`, no
`tokenizers`, no `reqwest`, no `pyo3`** — none of the external-ecosystem
machinery the boundary was meant to hold back. Whatever import/export code lives
here is hand-rolled against the workspace's own crates.

**The dependency edge runs the wrong way.** `hipfire-coexistence` depends on
**20 hipfire crates** — runtime, model, rdna, quantize, quant-format, steer,
lora-hfq, daemon-adapter, and six arch crates. **Nothing in the workspace depends
on it.** It is a leaf consumer, not an isolation boundary. A boundary that only
one side can see is a build-order constraint, not an architectural one.

**It is not small.** 18 files, ~12,200 lines, including a 3,189-line
`induction/` subtree.

So the separation costs a crate that must be kept in sync with twenty others,
and buys isolation from ten dependencies that were never a threat.

## 2. What is actually in there

| module | what it is | where it belongs after a merge |
|---|---|---|
| `induction/` (3,189 L) | recipe, preflight, two-pass, **orchestrate** | **daemon-schedulable** — see §3 |
| `calibrate.rs` | CLI orchestration around `LayerStreamEngine` | thin CLI over the runtime engine (engine already lives in `hipfire-runtime`) |
| `calibration_audit.rs`, `calibration_compare.rs`, `residual_compare.rs` | artifact comparison | tooling crate / CLI |
| `import_safetensors.rs`, `export_safetensors.rs` | format conversion | **stays offline** — this is the genuine container-translation half |
| `hub_archive.rs`, `repack.rs` | `.hfa` archive handling | stays offline |
| `artifact.rs`, `router_profile.rs` | inspection | tooling crate / CLI |

The honest split is **not** the current crate boundary. It is:

- **forward/backward passes over weights** — calibration, induction
  orchestration, KLD, QAT — which AGENTS.md already says "may be scheduled by
  the daemon";
- **container/format translation** — safetensors import/export, `.hfa` repack —
  which AGENTS.md already says belongs offline.

`hipfire-coexistence` currently contains *both*, which is why the existing
contract reads as contradictory when applied to induction.

## 3. The merge and the induction goal are the same problem

`induction/orchestrate.rs:264`:

```rust
let status = std::process::Command::new(&command[0])
```

**The induction orchestrator shells out to binaries.** That is the root blocker
for non-disruptive induction, not an incidental implementation detail:

- a spawned `hipfire-quantize` takes `hipfire lock`, which the serving daemon
  already holds — so induction and serving cannot overlap *at all*, regardless
  of any quantum work elsewhere;
- a subprocess cannot be preempted at a quantum boundary, cannot be admitted
  against a drain budget, and cannot share the resident model it is inducting
  from;
- every measurement this session established about co-residency (free resident
  swap, +0.20% inference cost, clean OOM) applies to *in-process* workers only.

So "merge coexistence" and "induct without interrupting serving" are one piece of
work: **convert the induction pipeline from an out-of-process `Command::new`
orchestration into in-process, daemon-scheduled quanta.** The crate merge is the
refactor that makes that expressible; it is not a tidy-up.

## 4. Proposed shape

1. **Move `induction/` into a daemon-reachable crate** (`hipfire-runtime`
   alongside `calibration::layer_stream`, or a new `hipfire-induction`). Replace
   `Command::new` dispatch with direct calls — `hipfire-quantize` is already a
   *library* with GPU behind an optional feature, so its codecs are callable
   in-process today.
2. **Add the stage quantum.** Each induction stage exposes
   `step() -> Advanced | Paused | Complete`, mirroring `CalibrationStep`, so the
   daemon can interleave it with serving. Calibration already has this; KLD-ref
   needs it (see the induction scope doc §1); quantize/QAT need it built.
3. **Keep the offline half offline.** safetensors import/export and `.hfa`
   repack stay a separate binary. They are genuinely container translation, they
   need no GPU, and nothing about non-disruptive induction requires them
   in-process.
4. **Fold the rest into CLI/tooling.** `artifact`, `router_profile`, the
   comparison tools — thin CLI over library calls.
5. **Rewrite the AGENTS.md invariant.** The current text draws the line at
   "format conversion, not GPU work" but then names `hipfire-coexistence` as the
   home for a pipeline that is mostly GPU work. The line should be redrawn where
   §2 puts it, and the crate-name references updated.

## 5. Risks and unknowns

- **AGENTS.md is a stated invariant.** Item 5 changes a contract, not just code.
  It needs explicit sign-off, which is why it is listed rather than assumed.
- **`tokio` is in coexistence's dependency list** but the daemon is not
  obviously async in the same way. Moving `induction/` may drag a runtime
  requirement into a crate that does not want one — needs checking before the
  move, not during.
- **Six arch-crate dependencies** (`gemma3`, `gemma4`, `cohere2`, `qwen35`,
  `zaya`, …) suggest induction reaches into arch specifics. Whether that survives
  a move into `hipfire-runtime` without a circular dependency is the main
  structural unknown. **This is the item most likely to force a different
  destination crate**, and it should be resolved by attempting the dependency
  graph on paper before any code moves.
- **~12,200 lines** is not a mechanical move. Sequence it stage by stage
  (induction first, since it unblocks the goal; format conversion last, since it
  is staying put anyway).

## 6. Recommendation

Merge, but **lead with `induction/`** rather than with the crate boundary. That
subtree is the only part blocking the stated goal, it is where the `Command::new`
lives, and moving it in isolation is testable: induction still produces a
byte-identical artifact, but now from inside the daemon process.

The remaining ~9,000 lines can follow at leisure, and the safetensors/`.hfa` half
should stay out regardless — that part of the original separation was correct.
