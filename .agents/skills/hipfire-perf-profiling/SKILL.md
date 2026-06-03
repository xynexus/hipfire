---
name: hipfire-perf-profiling
description: Profile hipfire performance on AMD systems before choosing an optimization lever. Use for memory-bandwidth math, rocprofv3 kernel traces/counters, AMDuProf/AMDuProfPcm/AMDuProfSys system telemetry, amd-smi/rocm-smi clock and power checks, fresh-process benchmark hygiene, and deciding whether to hand off to hipfire-kernel-tuning or hipfire-kernel-atlas.
---

# hipfire-perf-profiling

Use this skill when a Hipfire perf result needs profiling evidence before a
kernel, dispatch, quant, or runtime change. The goal is to identify the limit
first: memory bandwidth, launch overhead, kernel occupancy, cache behavior,
clocks/throttling, host overhead, or algorithmic work.

## Tool Roles

- `rocprofv3`: GPU kernel dispatch timeline, HIP/HSA traces, memory-copy traces,
  and GPU PMCs when supported. Use this first for hot-kernel timing.
- `AMDuProfPcm` / `AMDuProfSys`: CPU, data-fabric, memory, power, and system
  telemetry. On APUs, use this to test whether a run is bandwidth, clock, or
  power limited at the platform level.
- `amd-smi` / `rocm-smi`: coarse live GPU clocks, power, utilization, VRAM, and
  throttling context. Use as a low-overhead companion, not as kernel proof.
- `hipfire-kernel-atlas`: use after you have a phase-aware row and want ISA fit,
  dispatch provenance, or "left on table" visualization.
- `hipfire-kernel-tuning`: use after profiling identifies a hot kernel and a
  plausible tuning lever.

Probe availability before picking commands:

```bash
for t in rocprofv3 AMDuProfCLI AMDuProfPcm AMDuProfSys amd-smi rocm-smi; do
  command -v "$t" || true
done
```

On this host, AMDuProf may live under `/opt/AMDuProf_5.3-518/bin/`; use the
absolute path if the tools are not on `PATH`.

Hipfire examples do not all implement a safe `--help`. For example, some
examples treat `--help` as a model path and attempt to load it. Inspect the
example source or use a known command surface before probing a release binary
with `--help`.

For stdin-driven daemon profiling, write an input JSONL file and profile the
daemon directly for `rocprofv3` / `AMDuProfPcm`. For `AMDuProfCLI` and
`AMDuProfSys`, prefer a shell wrapper so the profiled child owns the input
redirection:

```bash
/bin/bash -lc 'target/release/examples/daemon < .codeinsight+research/profiles/<run-id>/daemon-input.jsonl'
```

## AMDuProf CLI Gotchas

AMDuProf command syntax differs by tool. Check help from the installed binary;
do not assume old `CodeXL` or older uProf syntax.

- `AMDuProfCLI` profiles launched programs directly. It does **not** use a
  `-- <program>` separator:
  ```bash
  /opt/AMDuProf_5.3-518/bin/AMDuProfCLI profile \
    -o .codeinsight+research/profiles/<run-id>/uprof-cli \
    --trace gpu --category gputrace \
    /bin/bash -lc '<benchmark command...>'
  ```
  Use discovery before choosing configs or events:
  ```bash
  /opt/AMDuProf_5.3-518/bin/AMDuProfCLI info --list collect-configs
  /opt/AMDuProf_5.3-518/bin/AMDuProfCLI info --list predefined-events
  /opt/AMDuProf_5.3-518/bin/AMDuProfCLI info --list pmu-events
  ```
  `profile` collects and reports in one step; `collect` writes profile data for
  later `report`. GPU trace support is arch-dependent. On the tested gfx1151
  host, `--trace gpu` completed but warned `GPU Tracing is not Supported for
  (gfx1151)` and produced a header-only report, so use `rocprofv3` for GPU
  kernel traces on that arch.

- `AMDuProfPcm` uses a subcommand, options, then `-- <program> [args]`:
  ```bash
  /opt/AMDuProf_5.3-518/bin/AMDuProfPcm profile \
    -m ipc -a -A system \
    -O .codeinsight+research/profiles/<run-id>/uprof-pcm \
    -- <benchmark command...>
  ```
  `profile` writes cumulative and timeseries CSV files. On AMDuProfPcm 5.3,
  `-C` is ignored with `profile`; do not rely on it. Start with a specific
  supported metric such as `-m ipc`. The default metric set can fail if the
  kernel lacks uncore/L3 support, and `-m memory` may be rejected even though
  older examples mention it. Add `--collect-power`, `--collect-pcie`, or
  `--collect-xgmi` only after confirming they are supported on the current CPU
  family/model. Use `-X` for non-root perf-subsystem mode when needed. Use
  `--msr` only when root/MSR access is intentionally available.

- `AMDuProfSys` uses `collect`, `report`, `--config`, or `--metrics` forms and
  can reject a launch if another `AMDuProfSys` instance is active. Do not run
  parallel `AMDuProfSys` sessions. `collect` does not use a `-- <program>`
  separator. A usable two-step shape is:
  ```bash
  /opt/AMDuProf_5.3-518/bin/AMDuProfSys --system-info
  /opt/AMDuProf_5.3-518/bin/AMDuProfSys collect \
    --config core -a -I 100 --force \
    -o .codeinsight+research/profiles/<run-id>/uprof-sys-core \
    /bin/bash -lc '<benchmark command...>'
  /opt/AMDuProf_5.3-518/bin/AMDuProfSys report \
    -i .codeinsight+research/profiles/<run-id>/uprof-sys-core/uprof-sys-core.ses \
    -o .codeinsight+research/profiles/<run-id>/uprof-sys-core-report \
    -f csv -G system --force
  ```
  `--force` is useful for exploratory runs when NMI watchdog is enabled, but
  the report is then marked less reliable. Without `--force`, this tool can
  exit with status 0 while refusing to collect and producing no session. For
  custom metrics, the expression names the config namespace (`core/...` or
  `l3/...`):
  ```bash
  /opt/AMDuProf_5.3-518/bin/AMDuProfSys --metrics --help
  /opt/AMDuProf_5.3-518/bin/AMDuProfSys --metrics \
    'core/ExampleMetric="(0x4300C3)/(0x4300C2)"' \
    -o .codeinsight+research/profiles/<run-id>/uprof-sys \
    /bin/bash -lc '<benchmark command...>'
  ```
  Use `AMDuProfSys --config --help` and `AMDuProfSys collect --help` on the
  target host before writing reusable commands. Short Hipfire smokes may be
  rounded up to the tool's minimum complete-sample interval, so prefer a longer
  run when interpreting the numbers.

## Workflow

1. **Freeze the benchmark shape**
   - Use byte-identical prompts from `benchmarks/prompts/` when possible.
   - Record prompt md5, binary md5, model path, artifact size, arch, commit,
     env, and exact command.
   - Run fresh processes. Do not compare within-session edits as a win.

2. **Do the expected-performance math**
   - For decode GEMV, estimate memory-bound throughput from payload bytes:
     `expected_candidate_tok_s = control_tok_s * control_bytes / candidate_bytes`.
   - Compare observed tok/s to this normalized line before declaring a perf
     regression. A larger format should not be required to match raw MQ4 tok/s.
   - For DFlash/spec rows, normalize by target/draft work and acceptance/tau;
     raw tok/s alone can mislead.

3. **Collect low-overhead system context**
   - Capture clocks/power/utilization during the same run:
     `amd-smi monitor` or `rocm-smi --showclocks --showpower --showuse`.
   - If clocks or power drift, rerun after DPM/thermal state settles.

4. **Collect GPU kernel timing**
   - Start with trace/stats, not counters:
     ```bash
     rocprofv3 --kernel-trace --memory-copy-trace --stats \
       -f csv \
       -d .codeinsight+research/profiles/<run-id>/rocprof \
       -- <benchmark command...>
     ```
   - With `-f csv`, expect files such as `*_kernel_stats.csv`,
     `*_kernel_trace.csv`, `*_memory_copy_stats.csv`,
     `*_memory_copy_trace.csv`, `*_agent_info.csv`, and
     `*_domain_stats.csv` under the output directory.
   - Without `-f csv`, rocprofv3 may default to a rocpd SQLite database under
     `<out>/<host>/<pid>_results.db`. The `sqlite3` CLI is not always installed;
     Python's standard `sqlite3` module is a good fallback. Discover table
     names by prefix because rocpd table names are GUID-suffixed:
     ```bash
     python3 - <<'PY'
     import sqlite3
     from pathlib import Path
     base = Path('.codeinsight+research/profiles/<run-id>/rocprof')
     db = next(
         p
         for host_dir in base.iterdir() if host_dir.is_dir()
         for p in host_dir.iterdir()
         if p.name.endswith('_results.db')
     )
     con = sqlite3.connect(db)
     cur = con.cursor()
     def table(prefix):
         return cur.execute(
             "select name from sqlite_master where type='table' and name like ?",
             (prefix + '%',),
         ).fetchone()[0]
     kd = table('rocpd_kernel_dispatch')
     ks = table('rocpd_info_kernel_symbol')
     for name, calls, total_ns, avg_ns in cur.execute(f'''
         select coalesce(s.display_name, s.kernel_name, '<unknown>'),
                count(*), sum(d.end - d.start), avg(d.end - d.start)
         from `{kd}` d left join `{ks}` s on s.id = d.kernel_id
         group by 1 order by 3 desc limit 20
     '''):
         print(f'{total_ns / 1e6:10.3f} ms {calls:6d} calls {avg_ns / 1e3:9.3f} us {name}')
     PY
     ```
   - `hsa_amd_profiling_get_dispatch_time returned dispatch times where the
     start time is after end time ... Swapping the values` is a known rocprofv3
     warning on this host. Treat it as non-fatal unless the run explicitly sets
     `ROCPROFILER_CI_STRICT_TIMESTAMPS=1`.
   - Summarize the hottest kernels by total time, call count, mean duration,
     and phase if the command exposes prefill/decode or AR/DFlash split.
   - If one kernel dominates, hand off to `hipfire-kernel-tuning`.
   - If the question is ISA fit or quant/hardware utilization, hand off to
     `hipfire-kernel-atlas`.

5. **Collect AMDuProf system telemetry**
   - Prefer `AMDuProfPcm` for quick CPU/system CSV telemetry when a supported
     metric is known:
     ```bash
     /opt/AMDuProf_5.3-518/bin/AMDuProfPcm profile \
       -m ipc -a -A system \
       -O .codeinsight+research/profiles/<run-id>/uprof-pcm \
       -- <benchmark command...>
     ```
     Inspect the generated `report-cumulative.csv` first. If the run is shorter
     than the sample interval, timeseries output may contain headers only.
   - Use `AMDuProfCLI profile --trace gpu` when you need uProf's GPU/HIP/HSA
     trace/report path rather than `rocprofv3` output. Remember: no `--`
     separator for `AMDuProfCLI`, and verify that the report contains real GPU
     rows for the active arch.
   - Use `AMDuProfSys` for deeper CPU/core/L3/data-fabric analysis only after
     checking supported configs on the host. Remember: no `--` separator,
     generate a report from the `.ses`, and avoid parallel `AMDuProfSys`
     sessions.
   - Do not assume AMDuProf GPU metric YAML covers the active RDNA arch; verify
     support. For GPU kernel counters, prefer `rocprofv3` unless AMDuProf
     explicitly supports the target GPU.

6. **Interpret before changing code**
   - If observed decode matches byte-normalized expected tok/s, treat it as
     likely bandwidth-bound and do not block promotion on raw MQ4 parity.
   - If prefill underperforms byte-normalized expected tok/s, inspect kernel
     timing, occupancy/counter data, launch count, and routing.
   - If DFlash underperforms, separate target verify cost, draft cost,
     acceptance/tau, graph capture, and host overhead.
   - If launch count dominates, look for fusion, batching, graph capture, or
     fewer per-token host launches before touching arithmetic.
   - If clocks, power, or thermal state drift, rerun; do not tune code from an
     unstable profile.

## Artifact Discipline

- Raw profiler outputs go under `.codeinsight+research/profiles/` unless the
  user asks for a committed evidence artifact.
- Promotion or release evidence goes under `benchmarks/results/...` and must
  include the exact commands, prompt md5, binary md5, model hashes or sizes,
  arch, tool versions, and a short interpretation.
- Profiler traces are evidence of where time went, not correctness evidence.
  Run the relevant coherence gate for runtime, dispatch, kernel, DFlash, or
  quant changes.

## Handoff Rules

- Use `hipfire-kernel-tuning` when a hot kernel and optimization lever are
  identified.
- Use `hipfire-kernel-atlas` when the next question is ISA fit, dispatch
  provenance, quant format occupancy, or visualizing "left on table".
- Use `hipfire-arch-port` when the bottleneck is missing or wrong arch routing,
  unsupported WMMA/MFMA shape, or a new GPU architecture port.
- Use `astrea` when profiling says the runtime is acceptable and the remaining
  decision is quality, KLD/PPL, calibration, packaging, or promotion policy.

## Good Output

Report:

- command and artifact paths
- tool versions and arch
- expected byte-normalized tok/s versus observed tok/s
- top kernels or platform metrics that explain the gap
- whether the next action is rerun, tune, Atlas, arch-port, or promotion review

Do not report:

- a perf win from one noisy run
- raw MQ4 parity as a requirement for a larger quant format without byte
  normalization
- DFlash speed as correctness evidence
- AMDuProf GPU counter claims without verifying the active GPU arch is supported
