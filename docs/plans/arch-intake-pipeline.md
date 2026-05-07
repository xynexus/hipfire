# PRD: Architecture Intake Pipeline

**Author:** Kaden / Claude Opus
**Status:** v0 (worked example: gemma4, 2026-05-07)
**Owner:** unassigned

## Why

Adding a new model architecture to hipfire used to be ad-hoc: clone
qwen35.rs, hand-edit, hope it builds. As the project has shipped more
arches (LLaMA, Qwen3, Qwen3.5 dense, Qwen3.5 MoE A3B/A10B/A17B,
Qwen3.5-VL, Gemma 4) the cost of inconsistency between intakes has
gone up:

- Same questions re-investigated for every arch (RoPE convention, GQA
  ratio, norm variant, tokenizer family, special-token IDs).
- Branches go stale because the rebase cost grows quadratically with
  master's pace of kernel / runtime change. The Gemma 4 branch sat
  483 commits behind master before being forward-ported.
- Kernel-fit gaps surface late (e.g. SwiGLU activation variant, sliding-
  window mask, soft-cap), making the build/smoke loop expensive.

A standardized intake pipeline lets contributors and Claude agents
(a) reach the kernel-fit decision quickly, (b) produce branches that
land instead of bit-rotting, and (c) leave behind a report future
intakes can crib from.

The Gemma 4 intake at `docs/investigations/2026-05-07-gemma4-arch-intake/`
is the worked example that this PRD generalizes from.

## Non-goals

- Auto-generating Rust code from HF config.
- Replacing per-arch judgment about kernel design (e.g. is the
  attention-softcap worth a fused kernel or a separate dispatch).
- Performance work — intake produces a coherence-gate-passing forward,
  not a tok/s-tuned one.

## Workflow

```
HF model ID  ──►  inspect_model.py  ──►  arch report skeleton  ──►  rebase / squash-port
     │                   │                       │                        │
     ▼                   ▼                       ▼                        ▼
 vocab/heads/     module class layout,     fill in §1-§4         crates/hipfire-arch-X
 head_dim/RoPE/   weight shape inventory,  identify kernel-fit   + net-new kernels
 chat template    tokenizer family         gaps + effort         + dispatch helpers
                                                                        │
                                                                        ▼
                                                                  build green
                                                                        │
                                                                        ▼
                                                              quantize + smoke run
                                                                        │
                                                                        ▼
                                                              daemon dispatch +
                                                              tokenizer wiring
                                                                        │
                                                                        ▼
                                                              coherence-gate run
                                                                        │
                                                                        ▼
                                                              release notes / PR
```

### Stage 1 — HF inspection (`scripts/arch-intake/inspect_model.py`)

Pre-condition: HF model ID known and accessible (public or token-auth).

Command:
```
python3 scripts/arch-intake/inspect_model.py <hf-model-id> --shapes-only \
    --out /tmp/arch-intake-<name>.json
```

Output: structured JSON with config fields, weight shape inventory,
tokenizer family, special-token IDs, chat template head. The
`--shapes-only` flag uses safetensors metadata directly so the
inspection runs in seconds without param download.

For deeper inspection (named_modules walk needed for unusual subclassed
layers), drop `--shapes-only` and run on hiptrx where the venv has
torch + transformers and the disk has space for weight downloads.

### Stage 2 — Arch-report skeleton

Copy `scripts/arch-intake/arch_report_template.md` to
`docs/investigations/<DATE>-<arch>-arch-intake/arch-report.md`.
Fill in:

- §1.1 (release lineup) from the inspect output and the HF org page.
- §2 (characterization) from the inspect output's `config` block.
- §3 (kernel-fit checklist) by walking the per-decode-step rows and
  asking "does an existing dispatch helper cover this?" — search
  `crates/rdna-compute/src/dispatch.rs` for the relevant step name
  (e.g. `rope`, `attention_flash`, `weight_gemv_swiglu`).

§3 is the **decision document** for the port: net-new kernels go in
`kernels/src/` + `kernels.rs` SRC consts + `dispatch.rs` helpers, and
adapt-existing kernels go into per-arch forward code.

### Stage 3 — Branch / squash-port

Decision tree:

- **Fresh arch (no prior branch):** create `feat/arch-<name>` off
  master HEAD, scaffold from `crates/hipfire-arch-toy/` (template
  designed for this purpose).
- **Stale prior branch (e.g. Gemma 4 was 483 commits behind):**
  preserve the prior branch under its existing name as a rollback
  reference. Create a new `<name>-rebased-<DATE>` branch off master
  HEAD and **squash-port** rather than `git rebase` per-commit. The
  cost of N independent conflicts dwarfs the cost of re-deriving the
  port intentionally on top of the new layout.
  - Per-commit rebase replays N (engine→arch-crate) renames and N
    (master kernel API drift) conflicts.
  - Squash-port replays the conflict ONCE, with full context.

### Stage 4 — Build green

`cargo build --release` (workspace) and `cargo build --release -p
hipfire-arch-<name> --examples` MUST both pass before commit. If a
forward-path kernel signature changed on master, the standard fix is
to comment out the offending arg with a `TODO(<scope>)` annotation
and document the gap in §4 of the arch report — DO NOT silently
revise kernel semantics.

Commit the port as a single `feat(arch-<name>):` commit on the
rebased branch. Use a HEREDOC for the message.

### Stage 5 — Quantize + smoke run

The smoke test in the new arch crate (`<name>_smoke_forward.rs`)
exercises:

- HFQ load
- Config parse round-trip
- Forward pass (single token + N greedy steps)
- Logits-finite assertion

This is the first time real GPU dispatch happens for the arch. Run
on hiptrx (4× R9700 gfx1201). Failures here are usually:

- Kernel signature mismatches (build-stage error, easier to catch).
- Tensor-shape mismatches (load-stage assertion).
- KV-cache sizing (decode-stage segfault or NaN).

### Stage 6 — Daemon dispatch + tokenizer wiring

Once the smoke passes, wire `arch_id == N` into:

- `crates/hipfire-runtime/examples/daemon.rs` — load + dispatch arm.
- `crates/hipfire-runtime/src/tokenizer.rs` — only if the arch uses a
  tokenizer family the runtime tokenizer doesn't yet handle.
- `crates/hipfire-quantize/src/main.rs` — quantize-side arch arm.

### Stage 7 — Coherence-gate run

Add a per-arch profile to `scripts/coherence-gate.sh`. Run the gate:

- Hard-fail tier: panics, zero tokens, timeouts, single-token attractors.
- Soft-flag tier: 3gram density / unique-ratio (see CLAUDE.md "DFlash
  Coherence Gate" for thresholds — same gate runs for AR coherence).

### Stage 8 — Release / PR

Open a PR with:

- The squash-port commit on the rebased branch.
- The arch-report doc (separate commit on a survey/docs branch).
- A 2-3 sentence release-notes line in the version bump commit.
- Link the arch report from `CONTRIBUTING.md` "Adding a new arch"
  section.

## Worked example: Gemma 4

See `docs/investigations/2026-05-07-gemma4-arch-intake/arch-report.md`.

Stages completed in v0 of the pipeline:

| Stage | Status | Notes |
|-------|--------|-------|
| 1. HF inspection | partial | Used WebFetch on `huggingface.co/google` to get the released lineup; full `inspect_model.py` run pending hiptrx access. |
| 2. Arch-report skeleton | done | 7 KB report at `docs/investigations/2026-05-07-gemma4-arch-intake/arch-report.md`. |
| 3. Branch / squash-port | done | `gemma4-rebased-2026-05-07` on `origin`. |
| 4. Build green | done | Full workspace + arch-gemma4 examples build. |
| 5. Quantize + smoke run | blocked | Sliding-window kernel diff + no `.mq4` file — see report §4 gap 1, gap 6. |
| 6. Daemon dispatch + tokenizer | not started | report §4 gaps 3, 4. |
| 7. Coherence-gate run | not started | report §4 gap 7. |
| 8. Release / PR | not yet | will package after stages 5-7. |

## Open questions

- **Should `inspect_model.py` emit a draft Rust `Config` parser?**
  Most of the per-arch `config_from_hfq` body is mechanical
  (`tc.get("hidden_size")?.as_u64()? as usize` for every numeric
  field). A code-gen mode would shorten Stage 3 by an hour or two.
  Risk: code-gen may obscure the arch-specific decisions about
  fallbacks (when is a missing key OK?) that should be conscious.
  Decision deferred until the next intake.

- **Should the arch report live in `docs/architectures/` instead of
  `docs/investigations/`?** Investigations connote one-off research;
  arch ports are durable artifacts. Once the gemma4 intake reaches
  Stage 7-8 the report should be promoted to `docs/architectures/
  gemma4.md` and continuously updated.

## Success criteria

- [ ] A second arch port (next time someone adds a model) follows
      the pipeline end-to-end without modifications.
- [ ] The pipeline survives one master rebase pace shock (~500
      commits) without becoming unusable.
- [ ] Kernel-fit gaps are reliably surfaced before any code is
      written, not after build failures.
