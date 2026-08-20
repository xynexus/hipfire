# AGENTS.md - hipfire repo guidance

This file is the repo-wide contract for agents. Keep it small: put
subsystem-only rules in the nearest nested `AGENTS.md`, keep long procedures in
skills or docs, and use the task prompt for one-off constraints.

For project background, notices, and detailed playbooks, read `README.md` and
the relevant docs under `docs/`.

## Core Invariants

- hipfire is Rust + HIP/ROCm-direct inference and production tooling. Do not
  put Python in production tooling; Python is allowed for experiments,
  benchmarks, diagnostics, and comparison baselines/oracles.
- Do not add Vulkan, wgpu, or a cross-vendor compute backend. The backend is
  HIP/ROCm-direct.
- Treat portability as a design constraint. When touching runtime, dispatch,
  kernels, or quant formats, ask whether the change works on RDNA2, RDNA3, and
  RDNA4.
- Document meaningful experiment results, including failures. Failed approaches
  are useful when they narrow the search space.
- Match numeric precision to the data, not to habit. Do not default to `f32`/
  `f64` for compute or storage when a narrower type (`f16`, `bf16`, or int) loses
  nothing that matters. Coding agents reach for wide floats reflexively; on this
  project wide types cost bandwidth, VRAM, and kernel throughput for no benefit
  when the values don't need the range or mantissa. Decide from the data: what is
  its actual dynamic range, how much error does the downstream consumer tolerate,
  and is the value already an approximation (a calibration statistic, a mean, a
  score) whose intrinsic noise dwarfs the rounding? Reserve `f64` for
  accumulation and genuinely ill-conditioned math; prefer `f16`/`bf16` (or int)
  for stored arrays and hot-path compute unless a precision need is demonstrated.
  Keep exact config/geometry scalars wide; narrow the bulk numeric payloads.
- Use one lock primitive: `hipfire-lock` `flock(2)`. Rust callers use
  `FlockGuard` and the shared path helpers. Shell, script, and external callers
  use `hipfire lock {acquire,release,status}`; `gpu-lock` is only an alias. Do
  not add sentinel files, pidfile liveness locks, `create_dir` mutexes,
  alternate lockfiles, or shell-only mutexes.
- Keep coexistence and compatibility tooling out of the inference binaries. Any
  import/export, format conversion, or interop tool (e.g. safetensors/GGUF
  import, model or LoRA conversion, adapter merge/bundle, external-format export)
  belongs in the `hipfire-coexistence` binary (or another dedicated tooling
  crate), not folded into the daemon, server, or runtime hot path. The inference
  path stays lean and HIP-direct; conversion and compatibility concerns are
  offline tooling.
- The line above is drawn at **format conversion, not at GPU work**. A workload
  that is a forward (or backward) pass over a model — calibration/induction,
  Hessian and imatrix capture, KLD evaluation, training and drafter training — is
  inference-shaped work and may live in the daemon, where it can be scheduled,
  batched, and preempted against serving traffic. What must stay out is
  container/format translation and external-ecosystem interop. Test: if it runs
  kernels over model weights it may be scheduled by the daemon; if it rewrites
  bytes between container formats it belongs in `hipfire-coexistence`.
- Concretely, the layer-stream **calibration/induction engine** — the forward-pass
  evidence producer (`LayerStreamEngine`, `CalibrationSession`, and the
  `DaemonCalibration` daemon wrapper) — lives in
  `hipfire_runtime::calibration::layer_stream`, alongside the rest of
  `hipfire_runtime::calibration`. That is what lets both the daemon
  (`DaemonRequest::Calibrate`, one layer per turn) and the daemon-free
  `hipfire-coexistence calibrate` CLI drive the same engine and produce a
  byte-identical artifact. `hipfire-coexistence` keeps the **offline, zero-GPU**
  half: CLI argument orchestration, the GPU self-lock for the standalone binary,
  corpus/format/storage byte-math, dry-run planning, and artifact
  compare/import/export/interop (plan §1.7: "coexistence keeps index/bytes, zero
  GPU"). Do not depend on `hipfire-coexistence` from the daemon/server/runtime;
  reach for the engine in `hipfire-runtime` instead.

## Branch And Git

- `master` is the default integration branch and the reference for new work.
  The former pre-fork `master` history is preserved as the archival
  `master-prefork` branch; do not base new work on it or merge it wholesale into
  `master` unless the user explicitly requests historical recovery work.
- **`origin` (github.com/xynexus/hipfire) is the only baseline.** `upstream`
  (github.com/Kaden-Schutt/hipfire) is the pre-fork original and is
  **disconnected** — we do not track, fetch, rebase onto, or merge from it. If
  the remote is still configured locally, ignore it. Comparing against it is
  actively misleading: it has 23 crates to our ~99, differs by 1500+ files, and
  lacks whole crates this tree depends on, so a diff against `upstream/master`
  will report an unrelated codebase rather than in-flight work.
- Start feature and fix work from an up-to-date `origin/master` on a descriptive
  topic branch. Prefer reviewed pull requests for integration; commit or push
  directly to `master` only when the user explicitly requests that workflow.
- Before meaningful changes, fetch `origin` and rebase or merge the topic branch
  onto the latest `origin/master` when the worktree state allows it. Check
  `origin/master` — never `upstream` — for competing in-flight work before a
  refactor. Do not rewrite published shared history without explicit approval.
- `git stash` works here, but `git stash pop` is a loaded gun: the stash stack
  holds **pre-existing entries from other branches** (as of 2026-08-20, five —
  `stash@{0}` is a WIP on `feat/npu-flm-reverse-engineering`, and three are from
  the pre-rename `chaingun` era). A bare `pop` takes `stash@{0}`, so if your own
  `push` did not land you restore somebody else's months-old WIP over your tree.
  Pop by explicit ref and check `git stash list` first.
  - This rule used to read "`git stash` is unusable — the untracked `.agents/`
    symlink tree makes it fail". That cause is gone: `.agents/` was once a farm
    of symlinks-to-symlinks, and today `.claude -> .agents` is the single symlink
    in the repo with nothing symlinked underneath. Re-tested 2026-08-20 on a
    faithful clone with a real dirty tree: `stash push -u` and `pop` both
    round-tripped byte-for-byte on git 2.53.0.
  - Prefer a scratch `git worktree` over stashing for anything that needs a clean
    tree — it cannot touch your working copy at all. `scripts/probe_commits.sh`
    is the worked example. Never build tooling on `stash` + `git checkout -f` in
    the main tree: the `-f` discards uncommitted work whenever the stash did not
    cover it, which is how that script used to eat dirty trees.
  - To compare against HEAD without any of this, `git show HEAD:<path>` and
    `git diff <path>` still do the job.
- `git add` cannot stage through the `.claude` symlink — it fails with "beyond a
  symbolic link". Stage agent skills and docs via their real `.agents/...` path.
- Preserve unrelated user changes. When committing or pushing, stage only files
  that belong to the current task and use descriptive messages.

## Verification

- Run `./tests/no-gpu-ci.sh` before handing off workflow-only changes.
- `./tests/tiny-affected-gate.sh --require-coverage` is the automatic GPU
  correctness front tier for covered runtime and quantization changes.
- `./tests/coherence-gate-dflash.sh` remains available as a manual
  DFlash/DDTree diagnostic; it is not an automatic or mandatory gate.
- Model/runtime admission evidence belongs in `hipfire-eval` batteries or
  suites first. Shell gates should remain enforcement wrappers when they still
  provide baseline comparison or hook integration.
- Non-daemon GPU binaries do not self-lock. Coordinate GPU examples, benches,
  and `hipfire-quantize` with `hipfire lock {acquire,release,status}` unless the
  called gate already does so.
- `hipfire-eval` is the exception: it loads through the daemon, which acquires
  the resource lock itself. Do NOT wrap it in `hipfire lock` — it deadlocks
  against the caller's own holder, and the failure names your own label as the
  blocker (`resource hip-gpu-0 ... is locked by <your label> ... mode=run`).
  Run it directly.
- The lock itself is sound: `flock(2)` is kernel state and is released when its
  holder dies, so a crashed job never wedges the lock. The daemon acquires with
  a real `try_lock`, not by reading anything.
- But the holder LINE in `~/.hipfire/locks/hip-gpu-0.lock` is file contents, not
  kernel state, so it outlives its writer — and it is what the "is locked by
  ..." message NAMES. A stale line therefore makes a genuine contention error
  point at a long-dead pid, which is badly misleading when the real holder is
  something else (a wrapper you started, a daemon that outlived its run).
  `hipfire lock kill` now clears such a line when it can prove the writer is
  gone; if you are staring at an error naming a pid that `ps` does not show,
  run it and read the message again.

### Profiling the daemon with rocprofv3

To get a kernel trace of daemon-side work (serving, calibration, KLD), drive
the daemon over its **stdin JSON protocol** so it is the profiler's own child
and exits at EOF:

```sh
cat > /tmp/req.jsonl <<'EOF'
{"type":"load","model":"/path/model.hfq","params":{"max_seq":4096}}
{"type":"kld_eval","mode":"build_ref","corpus":"/path/slice.txt","ref_path":"/tmp/x.kldref.hfq","n_ctx":2048,"max_chunks":1}
{"type":"unload"}
EOF
rocprofv3 --kernel-trace --stats -d /tmp/prof -o run \
  -- ./target/release/hipfire-daemon < /tmp/req.jsonl
```

Results land in `/tmp/prof/run_results.db` (SQLite; the `top_kernels` view has
name/total_calls/total_duration/average/percentage). The three routes that do
NOT work, so nobody re-derives them:

- `rocprofv3 --attach <pid>` fails with `PTRACE_SEIZE ... Operation not
  permitted`: `/proc/sys/kernel/yama/ptrace_scope` is `1`, so only descendants
  of the profiler can be traced.
- Profiling `hipfire eval` does not capture anything — the GPU work happens in
  the daemon it spawns, and rocprofv3 does not follow that child.
- Starting the daemon separately with `--listen` and pointing an eval at it
  fails too: `hipfire eval` refuses to reuse a running daemon (`FATAL: hipfire
  daemon already running`).

Beware `pkill -f hipfire-daemon` in a wrapper script: `-f` matches the full
command line of the shell running it, so it kills its own shell and surfaces as
a bare `exit 144` with no output. Use `pkill -x hipfire-daemon`.

## Artifact Names

Canonical artifact shape:

`<family>[-]<version>-<size[-effective/active]>[-tag1][-tag2...]--[feature1.[feature2.]...]<quant>[.arch].hfq`

A double hyphen `--` separates the human-readable model name from the
machine-readable groups; periods separate groups within the machine section.
Examples:
- LFM2.5-1.2B-Thinking--bf16.hfq
- Qwen3.5-122B-A10B--+mtp.+vl.mq2l.hfq (embedded MTP + VL)
- Gemma-4-8B-E4B-it-heretic-QAT--+dflash.+triattn.oq4++.gfx1151.hfq
- ModelName3.5-20B-it--dflash.oq4+.hfq (DFlash drafter sidecar — carries a quant,
  so it takes the boundary)
- MedGemma-27B-it.triattn.hfq (quant-free role sidecar — plain dotted suffix, see below)

This system allows machine parsing by working backwards:
- last field is always hfq
- a double hyphen `--` marks the boundary: everything left of it is the
  human-readable model name, everything right of it is the machine-readable
  section (features, then quant, then optional arch)
- within the machine section, dots separate fields
- single dashes separate human-readable fields in the model name, aside from
  size and effective/active size
- the `--` boundary applies to any artifact that carries a quant token,
  including role sidecars: a DFlash drafter is
  `ModelName3.5-20B-it--dflash.oq4+.hfq`, because `dflash` and `oq4+` are both
  machine-section groups
- role sidecars that carry NO quant keep a plain dotted role suffix
  (`.triattn.hfq`, `.jinja.`, `.hessian`) — with no machine section there is no
  boundary to mark

Quant tokens use this shape:

`<family><bitwidth>[l][+][+]`

- `mq` / `MQ` is affine Magnum Quant.
- `oq` / `OQ` is symmetric Opus Quant. Do not use `op` for new artifacts.
- `l` after the bitwidth means Lloyd-Max/codebook MQ encoding, for example
  `mq4l`. Do not use `lloyd-mq4` for new artifacts.
- A first `+` means clip-search, SmoothQuant, AWQ, or a comparable
  activation-aware clipping/scaling pass.
- A second `+` means Hessian/LDLQ error feedback.
- Mixed-precision formats include a decimal place in the bitwidth, for example
  `mq4.5+` or `oq4.25++`.
- Examples: `mq4`, `mq4+`, `mq4++`, `mq4l`, `oq4`, `oq4+`, `oq4++`.

Notes:
- `family` and `version` may optionally include a dash. eg. Qwen3.5, Llama-3
- size with optional active/effective parameters. eg. 0.8b, 30B-A3B, 8B-E4B, 50M, 2.5T
- Use `.hfq` for all hipfire container artifacts.
- Use dotted model versions such as `Qwen3.5`. do not use qwen35 for example.
- Put calibration or transform modifiers that are not part of the quant token
  before it. Lloyd is part of the quant token: use `mq3l`, not `lloyd-mq3`.
- Prefix a role group with `+` when the role is EMBEDDED in the artifact rather
  than naming it. `Model--dflash.oq8.hfq` *is* a DFlash sidecar;
  `Model--+dflash.+triattn.oq8.hfq` is a model that *carries* DFlash and
  TriAttention. Without the marker the two are indistinguishable by filename.
  Each embedded feature is its own `+`-marked dot group after the `--` boundary
  and before the quant token, e.g. `Model--+mtp.+vl.mq4.hfq`.
  The `+` here is a prefix on a role; the `+`/`++` in a quant token
  (`oq4+`, `oq4++`) is a suffix on the quant, so the two never collide.
- Use role sidecars when loaded independently. A sidecar carrying its own quant
  takes the boundary and the quant as machine groups
  (`Model--dflash.oq4+.hfq`, `Model--mtp.mq4.hfq`; embedded in a bundle they become `+dflash` / `+mtp`); a quant-free sidecar keeps
  the plain dotted role suffix (`.triattn.hfq`, `.jinja.`, `.hessian`).
- The quant should detail the weight encoding. eg. Lloyd MQ2 uses `--mq2l.hfq`,
  Magnum uses `--mq4.hfq`
- `arch` must start with gfx followed by 3 or 4 numbers. eg. gfx906, gfx1103, gfx1151, gfx1201
- New artifacts use the `--` boundary. The older all-dotted form
  (`Model.mq4.hfq`) stays parseable so existing on-disk artifacts keep loading;
  emit `--` for anything you create and rename dotted names to `--` when you
  touch them.
- When a script, gate, registry, or doc uses an older format, update it to the
  canonical naming convention as part of the fix.
- Remove legacy-name fallback whenever you find it

## Local Routing

When a task primarily targets one of these subtrees, read that subtree's
`AGENTS.md` before editing even if Codex was started from the repo root.

- `.agents/` owns reusable agent skills and workflow instructions.
- `benchmarks/` owns prompt corpora, benchmark scripts, baselines, and results.
- `crates/hipfire-eval/` owns model/runtime evidence batteries and suites.
- `kernels/` and `crates/hipfire-rdna/` own HIP kernel and dispatch mechanics.
- `crates/hipfire-runtime/` owns the inference hot path, model execution, and
  runtime examples.
- `scripts/` owns reusable shell workflows and ad hoc benches outside the formal
  test gate directory.
- `tests/` owns enforcement wrappers and CI/smoke gates.

Add or tighten nested `AGENTS.md` files when a rule only applies to a subtree.

## Local Overlay

`AGENTS.local.md` is gitignored and may be absent. When present, it contains
machine-specific guidance such as hostnames, GPU fleet notes, SSH targets, and
lock pins.

@./AGENTS.local.md

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
