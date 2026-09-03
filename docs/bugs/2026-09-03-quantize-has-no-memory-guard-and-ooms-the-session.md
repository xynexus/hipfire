# `hipfire-quantize` has no memory guard and OOM-kills the desktop session

Status: **OPEN**, 2026-09-03. Reproduced twice on halo (gfx1151, 128 GB UMA).

## What happens

Quantizing Qwen3.5-4B to `qtip3` with `HIPFIRE_QTIP_COND=greedy` exhausts system
memory and triggers a **global** OOM. The kernel does not kill the quantizer — it
kills the user session:

    tmux: server invoked oom-killer: gfp_mask=0xcc0(GFP_KERNEL), order=0
    Out of memory: Killed process 6839 (wireplumber)
    Out of memory: Killed process 6838 (pipewire)
    Out of memory: Killed process 4861 (dbus-daemon)
    Out of memory: Killed process 4841 (systemd)      <-- the user session manager

with the allocation stack:

    kfd_ioctl_alloc_memory_of_gpu
      amdgpu_amdkfd_gpuvm_alloc_memory_of_gpu
        amdgpu_bo_create -> amdgpu_ttm_tt_populate

The desktop dies, the terminal dies, and the machine needs a reboot. Both of this
session's crashes trace to this.

## Why it hits here specifically

This is a UMA APU: GPU memory and system RAM are **one pool** (512 MB dedicated
VRAM + ~120 GiB GTT). A KFD allocation is backed by the same pages the desktop
runs in, so an over-large GPU request does not fail a `hipMalloc` — it evicts the
session. `HIPFIRE_QTIP_COND=greedy` keeps the OBS residual device-resident, and
that residual scales with tensor size: 2B `qtip3` completes, 4B does not.

## The asymmetry that makes it a bug

The **loader** already has exactly the right guard —
`load_mem_reserve_gib` (default 4), documented as:

> GiB the load check leaves free for the rest of the system. Enough to keep the
> session's supervisor processes alive so a too-large load fails as a refusal
> rather than a reaping.

`hipfire-quantize` has no such check: `grep -rE 'mem_reserve|available_memory|MemAvailable'`
over `crates/hipfire-quantize/src/` returns nothing. The quantizer allocates GPU
memory on the same shared pool with none of the loader's protection, so the
failure mode the loader was explicitly built to prevent is exactly what the
quantizer does.

## Fixes, in order of value

1. **Reserve check before the big allocations**, reusing the loader's setting so
   there is one knob: refuse the run with a clear message naming the tensor and
   the shortfall, rather than reaping the session.
2. **`oom_score_adj`.** A long-running offline job should mark itself the
   preferred OOM victim so a miss kills the job, not the desktop. One write to
   `/proc/self/oom_score_adj` at startup.
3. **Bound the device-resident OBS residual** (or fall back to the host path)
   when the projected footprint exceeds the reserve.

This applies to every offline GPU binary on a UMA part, not only the quantizer —
see also `docs/todo/2026-09-03-quantize-and-qat-are-not-daemon-ops.md`: a
daemon-scheduled quantize would inherit the loader's admission check for free.

## Interim

Run offline GPU jobs with `oom_score_adj=1000`, and avoid
`HIPFIRE_QTIP_COND=greedy` above ~2B on a 128 GB UMA box.
