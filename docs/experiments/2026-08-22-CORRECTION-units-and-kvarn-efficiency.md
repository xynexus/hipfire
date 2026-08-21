# CORRECTION: the GPU is 90% busy, and KVarN attention runs at 6% of peak

This corrects a units error that invalidated a central claim in
`2026-08-22-kvarn-attention-profile.md` and
`2026-08-22-kvarn-regression-is-not-kv.md`.

## What was wrong

Both documents state the profiled run is "99.9% dispatch overhead", "GPU busy
0.08%", and that the attention kernels sit "at the per-dispatch floor". That came
from reading `top_kernels.total_duration` as nanoseconds. **It is microseconds**,
so every total in those files is 1000x larger than labelled: "148.6 ms of GPU
time" is 148.6 SECONDS, and "288 ns per dispatch" is 288 us.

Verified two independent ways:

1. **Unit-free**: `sum(end-start) / (max(end)-min(start))` over all dispatches =
   **89.75% (f32) / 90.00% (kvarn)**. The GPU is ~90% busy, not 0.08%.
2. **Physics**: gate/up moves 44.6 MB of weights, whose floor at the measured
   248.5 GB/s is **179.3 us** — against a measured 209.5. Consistent only if the
   unit is microseconds. As nanoseconds it would be ~850x faster than DRAM allows.

**The run is GPU-bound, not dispatch-bound.** Every "dispatch overhead dominates"
conclusion in those two files is void. What survives unchanged: the per-shape
attribution (six of seven GEMV shapes bit-identical, the whole regression in the
gate/up shape), the ablation table (all null), and the reproducible A/B.

## What the corrected numbers show

Per attention call at 3000-token context:

| | bytes moved | measured | achieved BW | its own DRAM floor |
|---|---|---|---|---|
| `attention_f32` | 24.58 MB | 185 us | **132.8 GB/s (53% of peak)** | 98.9 us |
| `attention_flash_kvarn_tile_batched` | 4.61 MB | 288 us | **16.0 GB/s (6% of peak)** | **18.5 us** |

**KVarN attention is 15.6x off its own bandwidth bound.** f32 attention is within
1.9x of its. That single fact explains the whole puzzle:

- Why KVarN loses despite reading 5.3x fewer bytes — it runs at 6% of peak, so
  the byte saving never surfaces.
- Why six ablations were null — the kernel is neither compute- nor
  bandwidth-bound. It is structurally under-parallelised: `__launch_bounds__(32)`
  (one wave per workgroup), a serial 128-token loop per wave, and ~211 waves per
  call, against a part with 1280 wave slots.

The FFN GEMVs, by contrast, are healthy: gate/up at 209.5 us against a 179.3 us
floor is **86% of peak** (f32) and 78% (kvarn). The unexplained ~7% gate/up
regression is therefore "8 points of DRAM efficiency", not "7% more work" — still
open, but correctly framed.

An extra term fusion could remove, now visible: the two-pass flash writes and
re-reads a `partials` buffer of `max_tiles x (2+head_dim) x n_heads x 4` =
**1.19 MB per call round-trip, 26% of all bytes the kernel moves**, purely to hand
off from the tile kernel to `attention_flash_asym_reduce_batched`.

## Lesson

This is the third units mistake in this session (the first two were `/1e6` vs
`/1e9` on the same field). The reliable checks, both cheap:

- **Ratio-only metrics need no units** — `busy/span` settles utilisation outright.
- **Cross-check one number against physics.** Any kernel that streams a known
  number of bytes has a DRAM floor; if the measurement is far below it, the units
  are wrong, not the hardware.
