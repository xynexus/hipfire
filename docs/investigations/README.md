# Investigations

Persistent notes for non-trivial debugging sessions and hypothesis-driven
work. Each investigation gets its own dated subdirectory with the form
`YYYY-MM-DD-short-slug/`.

## Why this exists

Per `CLAUDE.md` (`feedback_perf_bench_tmp_fragile.md`) — never store
investigation artifacts under `/tmp/`. They get wiped on reboot, lost
on `/tmp` sweeps, and cannot be referenced by issues or future
sessions. This folder is the canonical home for:

- `INVESTIGATION.md` running logs (hypothesis × verdict × evidence)
- Linked GitHub issue/PR snapshot bodies (frozen at handoff time)
- Standalone diagnostic scripts (NumPy probes, simulation harnesses)
- Result JSONs (small, structured, version-control-friendly)

## What does NOT belong here

- Quantized weight files, safetensors, or any binary artifact (these
  belong under `/models/`, gitignored, or in HF cache).
- Build outputs, kernel blobs (gitignored under `kernels/compiled/`).
- Activation dumps or other scale-prone byte streams (use a side
  bucket or NAS path; reference it from `INVESTIGATION.md`).
- Per-session daemon logs (transient — only commit a curated excerpt
  if it documents the bug).

## Convention

Each subdirectory should contain:

- `README.md` — TL;DR + what's here + how to reproduce
- `INVESTIGATION.md` — append-only running log
- Optional: `issue-NNN-{original,update}.md` — frozen issue bodies
- Optional: small scripts and JSON results

Every investigation should be linkable from a GitHub issue and (once
closed) referenced from `MEMORY.md` for future sessions to pick up.

## Index

| date | slug | issue | summary |
|------|------|-------|---------|
| 2026-05-05 | [qwen36-a3b-mq4-fragility](2026-05-05-qwen36-a3b-mq4-fragility/) | [#171](https://github.com/Kaden-Schutt/hipfire/issues/171) | 3.6-A3B has a per-model quality cliff under MQ4 that no sampler config clears; 3.5-A3B is fine at greedy default. 12 hypotheses tested, 11 negative. |
