# Deferred tasks

Evicted from the active task list on 2026-04-20 to clear the board for the
DFlash perf stack (target: beat Lucebox 3.43× AR on 27B via the 7900 XTX).

These are real, tracked items — not abandoned. Pick them up when the perf
push lands or when a dependent workflow needs one of them.

## #23 — Get hermes-agent running against hipfire serve (all 4 models)

Hermes binary + config → `hipfire serve :8080`. Per-model config for each
of the 4 sidecars. Run real agentic workflows (not just curl). Multi-turn
via `hermes chat`.

Tied to the ship story for agentic DFlash — once perf is where we want it,
this becomes the demo / validation vehicle. Last state was in-progress
before the Lucebox-gap investigation took over the session.

## #25 — Fix rocBLAS stride bug under physical_cap

Real fix for MI300X prompt-KV corruption. Audit `crates/rdna-compute/src/rocblas.rs`:

1. GEMM leading-dim params — probably still use `max_seq` stride when KV
   buffer is `physical_cap`-sized.
2. `compact_offset` application to write pointers after eviction.
3. FP16 weight shadow cache refresh under reduced K/V size.

Test: remove `HIPFIRE_ROCBLAS_OFF=1` from `/root/hipfire_serve.sh`, rerun
A3B + 27B via `:8080`, expect coherence + no panic.

MI300X-only bug. Doesn't affect the 7900 XTX perf work.

## #21 — Fail-Fast drafting (arXiv 2512.20573)

DEFERRED to later session. Algorithm understood (`draft_conf[i] <
threshold → truncate block`). Blocker: greedy path uses
`gpu.argmax_f32_batched` which returns IDs only, needs a new kernel or
full-logits D2H.

Paper ref: arXiv 2512.20573  
Repo: github.com/ruipeterpan/failfast (`failfast.py`)

Composes with DFlash as a τ-improver on low-confidence cycles. Re-evaluate
value after the current kernel-overhead reduction work lands — Fail-Fast
wins more when cycles are expensive and you want to abort early; our
optimized cycle may change that calculus.
