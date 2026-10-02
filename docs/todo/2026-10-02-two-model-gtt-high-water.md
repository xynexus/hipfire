# TODO: ~53 GiB of GTT above the weights with two models serving a swarm

Status: RESOLVED (2026-10-02) -- ROCr's fragment allocator; see "Resolution".
Date: 2026-10-02.
Models: Qwen3.8-27B--oq4.25++ + Qwen3.6-35B-A3B--oq4.25++ co-resident, gfx1151
(APU: GTT is host RAM, 116 GiB cap of 125 GiB).

## What is fixed (same night)

Unbounded growth over a Corrode CAE swarm turn (24 -> 94..107 GiB, host down to
11-24 GB free) had four causes, all fixed and verified by replaying captured turns:

| cause | fix |
|---|---|
| cold prefill checkpointed every turn boundary (~45 at once) | `46a302380` cap 4 |
| GPU pool cached every freed buffer until device memory ran short | `313b65fb1` 4 GiB cap, >= 256 MiB never cached |
| finished sessions released only at cycle end; with mid-cycle admission a cycle never ends | `313b65fb1` release every step |
| prefix-index eviction released other models' checkpoints to the wrong worker (40 resident vs cap 16) | `b8507d5ac` per-worker cap |

Now: resident sessions = 16 checkpoints per model + live requests, flat across
the run (`qwen35 release` debug line, `hipfire_serving_core::session=debug`).

## What is not explained

Both models loaded, nothing resident: **40.8 GiB** (27B 17.7 + 35B 23.1).
Replaying a captured two-model turn (178 requests, 3 concurrent, outputs capped at
256 tokens), twice back to back on one instance: idle **94.9 GiB, then 93.8 GiB**
-- a plateau, not a leak, but ~53 GiB of state.

Accounted for: 32 checkpoints (16 per model). A cold 13K-token 27B request measured
+1.6 GiB for 1 session + 4 checkpoints, so ~0.4 GiB per checkpoint at that length:
~13 GiB. Pool cache <= 4 GiB (logged ~0.4). Live sessions ~1-2 GiB. That leaves
~35-40 GiB.

Leads, in order:
1. Grow-only scratch: something sized by the largest prompt/batch seen and kept
   per model (prefill activations, GDN chunk state, attention partials -- the old
   tile path's `partials` is n_heads x max_tiles x (2+head_dim) x rows x 4 B, up to
   ~1.6 GiB at 32K context and 256 rows). Not in the pool stats if hipMalloc'd
   directly.
2. A checkpoint is a fork of its source session and keeps the SOURCE'S capacity
   (prompt + that request's max_tokens; `qwen35_fork_session_state`), not its own
   prefix length. With Corrode's 8192-token output cap that is up to 8K tokens of
   empty KV per checkpoint (~200 MiB on the 27B). A checkpoint is never decoded
   into; it could be compacted to its prefix at mint time.
3. The fresh-start run that loaded the 35B mid-way plateaued lower (75.6 GiB) than
   both-loaded-first (94 GiB), so placement/fragmentation may matter.

How to measure: `rocprofv3 --memory-allocation-trace` on `hipfire serve` during a
`replay.py` of a captured turn (see the Corrode memory note on the request-capture
proxy), then sum live allocations by call site at idle.

## Safety meanwhile

The memory watchdog kills the daemon below 16 GB MemAvailable. At the 94 GiB
plateau the host keeps ~28 GB. Lower `HIPFIRE_SERVER_PREFIX_CACHE_MAX` (per model
now) if running heavier than this.

## Measured (2026-10-02, morning)

- `rocprofv3 --memory-allocation-trace` over the same two-model replay: live
  allocations (allocate minus free, by address, incl. VMEM) are **35.5 GiB** once
  both models are loaded and **~42 GiB** at idle after the replay -- while GTT
  reads 40.8 and **95.0 GiB**. So ~53 GiB of GTT is NOT held by any allocation
  the runtime reports as live: it is memory hipfire freed that the system did not
  get back (or allocations outside the traced APIs).
- Not the paged KV's free path: `probe_vmm_free` maps 256 MiB of 128 KiB VMM
  pages and frees the region (one range unmap) -- GTT returns exactly, `Ok(())`.
- Not paged KV at all: with `HIPFIRE_KV_PAGED=0` the same replay climbed to
  105 GiB and the memory watchdog killed the daemon at 15 GB free. Paging uses
  LESS memory.
- Leads now: the HIP/ROCr runtime keeping freed hipMalloc memory (sub-allocator
  or pool caching; try a `hipDeviceGraphMemTrim`/mempool trim or the ROCclr
  memory-pool env knobs and watch GTT after a release), host/pinned allocations
  outside the trace, and fragmentation of many differently sized KV allocations.
- `probe_hipmalloc_free`: ~4 GiB as 170 buffers of 1..48 MiB, freed -- GTT returns
  exactly, three rounds. Plain hipFree is not caching either.
- DRM fdinfo of the daemon at the 94 GiB plateau: `drm-resident-gtt: 95786316 KiB`
  on its own fd -- the memory is the DAEMON's, but the HIP allocation trace only
  accounts for ~42 GiB of it (paged KV ~2 GiB: ~17K live 128 KiB pages). So ~53 GiB
  is held through something that trace does not see. Next: `rocprofv3
  --scratch-memory-trace` (per-queue scratch for spilling kernels is allocated by
  the runtime on demand and kept), and the runtime's internal heaps for small
  allocations (40K live allocations under 1 MiB).
- Checkpoints now map only their sealed prefix (`qwen35_checkpoint_session_state`):
  no measurable change on this replay (93.6 vs 94-95 GiB) because it caps outputs
  at 256 tokens; real Corrode requests carry up to 8K tokens of headroom, so it
  should matter there -- not yet measured.
- `--scratch-memory-trace` over the same replay wrote no scratch records: no
  runtime scratch allocations, so not that.
- Aliased pages free correctly in both orders (`probe_vmm_free` alias rounds: a
  region aliasing another's pages via hipMemRetainAllocationHandle+hipMemMap, freed
  alias-first or source-first -- GTT returns either way).
- Next step needs root: at the plateau, `sudo cat
  /sys/kernel/debug/dri/1/amdgpu_gem_info` lists every buffer object per process
  with its size -- group the daemon's by size and compare against the trace's live
  set to see which BOs nothing in HIP still references.

## Resolution

`sudo cat /sys/kernel/debug/dri/1/amdgpu_vm_info` (amdgpu_gem_info does not list
KFD buffers) at the plateau, against the same right after loading both models:
the daemon's VM held 92.0 GiB of BOs, and the extra over baseline was **24,407 BOs
of exactly 2 MiB (47.7 GiB)** plus 9,774 x 128 KiB live KV pages (1.2 GiB). The
2 MiB blocks are ROCr's fragment allocator: small hipMalloc requests are carved
from 2 MiB blocks, and a block stays allocated while any piece of it lives --
per-request KV/state pieces scattered across blocks pin them all. It is invisible
to the HIP allocation trace (which sees the freed pieces as freed).

`HSA_DISABLE_FRAGMENT_ALLOCATOR=1`, same replay:

| | loaded (both models) | idle after replay | peak |
|---|---|---|---|
| default (fragment allocator on) | 40.8 GiB | 94.2 GiB | 94.3 GiB |
| fragment allocator off | **36.2 GiB** | **42.8 GiB** | **43.6 GiB** |

No errors, same throughput (same requests left at 600 s), cold 8.3K prefill
30.5 s vs 30.2 s. The daemon is now spawned with it set by default
(hipfire-daemon-adapter; an operator value, `=0` included, wins).
