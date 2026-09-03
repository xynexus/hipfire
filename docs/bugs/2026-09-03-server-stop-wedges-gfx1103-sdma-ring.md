# `hipfire stop` wedged the gfx1103 GPU: sdma0 ring timeout → failed ASIC reset

**Status:** OPEN, observed once 2026-09-03. Driver-level, not a hipfire logic
defect — but hipfire's shutdown is the trigger, so it is ours to avoid.

**Host:** nix2 / gfx1103 Phoenix, `amdgpu.cwsr_enable=0` confirmed live in
`/proc/cmdline`, so this is **not** the HIP-719 / CWSR hazard documented in
`docs/plans/gfx1103-lds-hip719-investigation.md`. Different engine (SDMA, not
compute), different trigger (teardown, not an LDS kernel).

## What happened

`hipfire stop` (to free the GPU lock for a diagnostic run) SIGTERMed the
daemon. The DMA engine did not drain:

```
[114249.204853] ring sdma0 timeout, signaled seq=11807, emitted seq=11810
[114249.204873] Starting sdma0 ring reset
[114249.422929] Ring sdma0 reset failed
[114249.422944] GPU reset begin!. Source:  1
```

Three SDMA jobs emitted, none signalled. The ring reset failed, which
escalated to a device reset, which also failed:

```
[114259.180513] MES failed to respond to msg=REMOVE_QUEUE     (x10, ~3.3s apart)
[114285.896634] MODE2 reset
[114292.299894] SMU: No response msg_reg: 1a resp_reg: 0
[114292.302845] Mode2 reset failed!
[114292.308111] ASIC reset failed with error, -62
[114292.310713] GPU Recovery Failed: -62
```

`-62` is `ETIME`. A second recovery attempt began at 114294 and never
completed — a kernel thread was still stuck in
`amdgpu_device_gpu_recover+0x23f` 220 seconds later, emitting hung-task
backtraces. `rocm-smi` reported the device present but with `perf=unknown` and
every clock `N/A`. Recovery required a reboot.

## Sequence, and what is NOT the cause

A daemon started **9 seconds after** the sdma0 timeout blocked in `D` state at
`amddrm_sched_entity_flush` during device init, holding the GPU lock. That
process is the visible symptom and it is easy to misread as the cause. It is
not: the ring timeout at 114249 precedes it, and `dmesg` shows nothing
GPU-related in the preceding 1200 seconds. The load it was attempting
(`qwen3.5-0.8b--oq4++.hfq`, off the `/srv` network mount) never got far enough
to touch a weight — RSS flatlined at 78 MB for four minutes.

`SIGKILL` on the `D`-state process was queued, not dropped: the task left `D`
once the driver released it and the flock came back on its own. `flock(2)` is
kernel state, so the lock never needed manual clearing — consistent with the
lock contract in AGENTS.md.

## Why it matters

Ordinary shutdown is not supposed to be able to brick the device until reboot,
and on this host the driver's own recovery path cannot get it back. Any gate,
bench, or agent loop that stops and restarts the server is exposed. It cost a
reboot mid-session here.

## Not yet known

- **Frequency.** Observed once. The GPU had page-faulted earlier in the same
  28-hour boot (first at 11140s, unrelated workload) and survived those, so a
  degraded-device precondition cannot be ruled out.
- **Whether a drain would prevent it.** The three unsignalled SDMA jobs suggest
  in-flight host↔device copies at SIGTERM. Whether the daemon can be made to
  quiesce SDMA before exiting — and whether that merely moves the timeout into
  the shutdown path — is untested.
- **Whether `/srv` matters.** The wedged run loaded from the network mount, but
  the timeout preceded that process entirely, so there is no evidence either
  way. Naming it only so the next occurrence can be compared.

## Reproduction attempt

None. Deliberately not retried: another attempt costs a reboot, and a sample of
two would not distinguish "shutdown race" from "this device was already sick".
Recorded so the second occurrence has something to match against.
