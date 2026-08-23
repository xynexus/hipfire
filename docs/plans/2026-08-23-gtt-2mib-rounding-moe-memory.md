# The 122B does not fit because of two stacked ~1.9x amplifications

**Date:** 2026-08-23 · **Box:** halo, Strix Halo gfx1151, 128 GB UMA, 124 GiB usable

## Summary

Loading `Qwen3.5-122B-A10B--oq4.25++` (63.9 GiB on disk) exhausts memory at
layer 35 of 48. The artifact is half the size of RAM, so "it doesn't fit" was
never obviously true. It doesn't fit because routed MoE experts cost **3.5x
their on-disk size** once resident, from two independent, compounding causes:

| cause | factor | where |
|---|---|---|
| compact -> Oq8 expansion on load | 1.80x | `load_moe_expert`, ours |
| GTT allocation rounded up to 2 MiB | 1.88x | `hipMalloc`, the driver |
| **combined** | **3.5x** | |

Neither is visible in RSS. Both are GTT, and GTT never appears in a process's
RSS -- peak RSS during a 63 GiB load was **2.06 GiB**. Anyone debugging this
with `top` sees nothing.

## Measurements

All from `Qwen3.6-35B-A3B--oq4.25++` (17.93 GiB) unless noted.

Consumption is not a fixed multiple of artifact size -- it is MoE-specific:

| model | artifact | consumed | ratio |
|---|---|---|---|
| Qwen3.8-27B oq4.25++ (dense) | 14.41 GiB | 19.85 GiB | 1.37x |
| qwen3.5-4b bf16 (dense) | 5.29 GiB | 11.27 GiB | 2.13x |
| Qwen3.6-35B-A3B oq4.25++ (MoE) | 17.93 GiB | 63.37 GiB | **3.53x** |

The two dense points fit `consumed ~= 0.94 * artifact + 6.3 GiB`, the constant
being HIP context + KV + scratch. That line predicts 23.2 GiB for the 35B MoE.
It took 63.4.

`HIPFIRE_ALLOC_REPORT=1` attributes the requests:

```
hipMalloc: 33.58 GiB across 42270 allocations, 25 distinct sizes
     20.31 GiB     10240 x      2129920 B     <- expert gate_up
     10.16 GiB     10240 x      1064960 B     <- expert down
      1.89 GiB         1 x   2034237440 B
```

Two gaps to explain: 18 GiB on disk became 33.58 GiB requested, and 33.58 GiB
requested became **63.10 GiB of GTT** (`mem_info_gtt_used`, peak, sampled at
1 Hz alongside the load).

### Cause 1 -- compact routed experts are expanded to Oq8 on load

`hipfire inspect` says the artifact holds `OqPlusCompact 20791 tensors 18.13 GB`
at 4.25 bits/weight. But `load_moe_expert` does:

```rust
Some(OQPLUS_COMPACT_QT) => (
    DType::Oq8G256,
    oqplus_compact_to_moe_oq8_blocks(&data, m, k)?,
)
```

decoding nibbles + sparse overlay into full int8 260-byte blocks -- 8.125
bits/weight. The arithmetic confirms it: 2,129,920 / (2 x 512 x 2048) = 1.0156
bytes/weight = exactly 260/256.

This buys one uniform kernel format. It costs 1.80x on the dominant tensor
class. **Dense compact tensors do NOT expand** -- `oq8_arch_load` returns
`DType::OqCompactG256` for the same quant code -- which is why dense models sit
at 1.37x and MoE at 3.53x.

### Cause 2 -- hipMalloc rounds GTT up to a 2 MiB multiple

Measured with `cargo run -p hipfire-rdna --example gtt_granularity <bytes>`:

| requested | GTT per alloc | ratio |
|---|---|---|
| 1,048,576 (1 MiB exactly) | 1,048,576 | 1.000x |
| **1,064,960** (1 MiB + 16 KiB) | **2,096,103** | **1.968x** |
| 2,097,152 (2 MiB exactly) | 2,097,152 | 1.000x |
| **2,129,920** (2 MiB + 32 KiB) | **4,194,304** | **1.969x** |
| 3,145,728 (3 MiB) | 4,194,304 | 1.333x |
| 4,325,376 (4.125 MiB) | 6,291,456 | 1.455x |
| 5,242,880 (5 MiB) | 6,291,456 | 1.200x |
| 9,437,184 (9 MiB) | 10,485,760 | 1.111x |

The rule is **round up to a multiple of 2 MiB**, not to the next power of two --
5 MiB costs 6, not 8; 9 costs 10, not 16. Consistent with huge-page-backed GTT.

The cruelty is in the constants. Every OQ block carries a scale on top of a
power-of-two payload (260 = 4 + 256, 132 = 4 + 128), so every routed-expert
tensor lands a hair OVER a 2 MiB boundary and pays for the next full granule.
2,129,920 B is 2 MiB + 32 KiB and costs 4 MiB: **1.6% over the line, 97% tax.**

Predicted from the sizes: 10240 x 4 MiB + 10240 x 2 MiB = 60 GiB, plus ~3 GiB
of everything else = 63 GiB. Measured peak GTT: 63.10 GiB.

## Consequences

**Load admission.** A guard added earlier the same day estimated demand from the
artifact's on-disk size. That was wrong by 3.5x on the models that need it: it
admitted the 122B, which then died at layer 35/48. It now prices routed-expert
modules through `estimated_module_resident_bytes`, which applies both factors,
and estimates the 122B at 169.2 GiB against 121.3 available -- refusing it. The
estimate is independently corroborated by where the real load died: 121.3/169.2
= 72% of the way in, and layer 35 of 48 is 73%.

**Expert paging.** The pager charged `range.len` -- what it asked for, not what
the driver took -- so a budget-constrained run could sit at a nominal 8 GiB
budget while holding ~15 GiB of GTT. This is not hypothetical; the code already
carried a note describing exactly that shape ("the pager's own accounting sat
correctly at its 8 GiB budget and the daemon's RSS stayed at 0"). All three
charge sites now use `gtt_alloc_cost`. Note module paging is inherently cheaper
than the per-tensor resident path: one allocation per module means the rounding
is paid once for gate_up + down together (3.19 MB -> 4 MiB, 1.31x) instead of
twice (1.97x each).

## What to do about it

Two independent fixes, and they multiply:

1. **Slab-allocate experts** (fixes cause 2, no numerics risk). One allocation
   per layer holding all experts, addressed via `GpuTensor::sub_offset`, which
   already creates non-owning views. A 128-expert slab of 2,129,920 B tensors is
   260 MiB -- already a 2 MiB multiple, so the waste goes to zero. Expert
   pointers are already indirected through `expert_gate_up_ptrs` tables, so the
   kernels need no change. **This alone makes the 122B loadable** (~110 -> ~58
   GiB).
2. **Keep routed experts compact resident** (fixes cause 1). Needs a compact
   grouped prefill GEMM -- written, `gemm_oq_compact_moe_grouped_wmma`, still
   numerically unvalidated -- AND a compact indexed MoE decode GEMV, which does
   not exist. Note compact alone still pays the rounding tax (gate_up 1.06 MiB
   -> 2 MiB), so it wants fix 1 too.

Both: ~110 GiB -> ~31 GiB for the 122B.

## Reproducing

```sh
HIPFIRE_ALLOC_REPORT=1 hipfire-daemon < load.jsonl     # what was requested
cargo run -p hipfire-rdna --example gtt_granularity 2129920 2000   # what it costs
cat /sys/class/drm/card1/device/mem_info_gtt_used      # the truth; RSS will lie
```

## Trap for the next person

`MemAvailable` tracks GTT almost exactly (fell 64.36 GiB against 63.10 GiB of
GTT), so system memory is a fine proxy. **RSS is not.** It stayed under 4 GiB
throughout. Every tool that reports per-process memory will tell you the daemon
is small while it consumes the machine.

Also: the failure is a `page allocation failure: order:0 ... __GFP_RETRY_MAYFAIL`
in dmesg, NOT an OOM kill. amdgpu tolerates the failure and returns an error, so
nothing gets reaped and `grep -i "killed process"` finds nothing. Look for
`page allocation failure` instead. (An earlier 122B attempt DID trigger the OOM
killer and took out dbus, pipewire and `systemd --user` -- which of the two you
get depends on how much reclaimable cache is around.)
