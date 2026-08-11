# KVarN — variance-normalized low-bit KV cache

Reimplementation-grade spec. There was no KVarN doc before this one; the method was
described only in scattered plan files.

| Thing | File |
|---|---|
| CPU core + golden oracle | `crates/hipfire-kvquant/src/kvarn.rs` |
| Write-side gather/transpose | `kernels/src/kvarn_gather_k_tiles.hip` |
| Tile quantizer (Sinkhorn + pack) | `kernels/src/kvarn_quantize_tile.hip` |
| Tile dequantizer (read side) | `kernels/src/kvarn_dequant_tile.hip` |
| v1 shadow-cache build | `kernels/src/kvarn_build_kcache.hip` |
| v2 fused flash attention | `kernels/src/attention_flash_kvarn_tile_batched.hip` |
| Routed/batched variant | `kernels/src/attention_kvarn_routed_batched.hip` |
| Cache state + mode selection | `crates/hipfire-runtime/src/kv.rs`, `layered_kv.rs` |
| Rotation | `crates/hipfire-primitives/src/fwht.rs` (shared with Opus Quant) |

Clean-room Rust reimplementation of the KVarN method ([2606.03458]). The published
repo is read only as an algorithm reference: it is MLA-shaped (DeepSeek latent KV,
R=512) and vLLM/Triton-bound. hipfire's FullAttention is **GQA**, so the core here
operates on a generic 2D tile and the KV wiring tiles per `(layer, kv_head)` over
`head_dim` rather than over an MLA latent.

---

## 1. The problem it solves

Naive per-token 4-bit KV quantization fails on keys. Within one token's key vector
the per-channel variance spread is enormous — a handful of channels carry orders of
magnitude more energy than the rest — so a single min/max range per token spends
almost all 16 levels on the loud channels and collapses the quiet ones to one or two
codes. Per-channel scaling instead fixes the channel spread but re-exposes the
per-token spread. Either axis alone leaves the other mis-scaled, and the error lands
exactly where recurrent attention accumulates it.

KVarN scales **both axes at once**: a log-domain Sinkhorn balancing that alternately
normalizes column-std and row-std until the tile has near-uniform variance in every
direction. A single 4-bit affine quantizer then has near-uniform error across the
tile, which is the whole point.

---

## 2. Method

Three steps. Only step 2 is KVarN proper; step 1 is shared with the Opus quant path
and step 3 is a textbook affine quantizer.

### Step 1 — rotation (optional, upstream)

FWHT-rotate the tile for incoherence, exactly as QTIP/Opus do. Deliberately kept
*out* of `kvarn.rs` so callers share the engine's `cpu_fwht_*` / `mq_rotate_x`.
Measured effect: cos-sim **0.995 → 0.999** on the round-trip. Rotation Gaussianizes
the tile and removes the heavy tail that eats 4-bit min/max range.

The runtime rotates **K and Q by the same** orthonormal per-head FWHT-256 (mq signs).
Because the rotation is orthonormal, `(RQ)·(RK)ᵀ = Q·Kᵀ` **exactly** — scores are
preserved with no flash-kernel change and no Q-side un-rotation. K is written to the
cache rotated (both the block records and the recent window derive from the rotated
`fa_k`), so the whole KVarN frame is self-consistent. **V is left un-rotated**, so the
attention output stays in the original basis and `o_proj` is unchanged.

Requires `head_dim == 256` (the FWHT-256 group width). At `head_dim == 128` KVarN runs
un-rotated and takes the ~0.995 cos-sim. `HIPFIRE_KVARN_ROTATE=0` disables it for A/B.

### Step 2 — variance normalization (log-domain Sinkhorn)

`variance_normalize(tile, r_dim, c_dim, iters)` produces
`balanced[r,c] = tile[r,c] / s_col[c] / s_row[r]`.

```
log_s_col[c] = 0 ; log_s_row[r] = 0
repeat iters times:
    # column pass — normalize each column to unit std (down the rows)
    for c: std = stddev over r of (tile · exp(-log_s_row) · exp(-log_s_col))[:,c]
           log_s_col[c] = clamp(log_s_col[c] + ln(clamp(std, 1e-3, 1e3)), -0.3, 10.0)
    # row pass — normalize each row to unit std (across the columns)
    for r: std = stddev over c of the same, recomputed
           log_s_row[r] = clamp(log_s_row[r] + ln(clamp(std, 1e-3, 1e3)), -0.3, 10.0)
    track best-so-far by imbalance()
```

Three details that are not optional:

1. **Work in log space.** Scales compose multiplicatively across passes; accumulating
   logs and exponentiating once avoids the underflow that repeated division produces.
2. **Keep best-so-far, not last.** The alternating projection can overshoot. The loop
   returns the `(s_col, s_row)` that minimized the imbalance metric, not the final
   iterate.
3. **Clamp both the std and the log.** `std ∈ [1e-3, 1e3]`, `log_s ∈ [-0.3, 10.0]`
   (mirrors the reference `_LOG_S_{MIN,MAX}`). A dead channel otherwise drives a scale
   to zero or infinity and poisons the tile.

The objective:

```rust
imbalance = max_c(std_col) / min_c(std_col) + max_r(std_row) / min_r(std_row)
// perfectly balanced tile scores 2.0
```

`SINKHORN_ITERS = 16` on CPU; the GPU kernel uses 10 (`KVARN_SINKHORN_ITERS`) — the
metric is flat past ~10 and the tile is re-read from global memory each pass.

### Step 3 — per-channel affine quantize + scale absorption

Quantize the *balanced* tile per row (= per channel for K), then **absorb the per-row
Sinkhorn scale into the quantizer's scale and zero-point** so the runtime dequant only
needs one extra multiply:

```rust
for r in 0..r_dim:
    lo, hi   = min/max over c of balanced[r, :]
    scale    = max((hi - lo) / qmax, 1e-8)
    q[r,c]   = clamp(round((balanced[r,c] - lo) / scale), 0, qmax)
    scale_abs[r] = scale * s_row[r]      // absorbed
    zp_abs[r]    = lo    * s_row[r]      // absorbed
// s_col is stored as-is
```

Dequant is then, everywhere (CPU oracle, GPU dequant, fused flash):

```
deq[r,c] = (q[r,c] * scale_abs[r] + zp_abs[r]) * s_col[c]
```

`s_row` never appears at runtime. This is why the layout has one per-row pair and one
per-column vector rather than three separate scale planes.

`qmax = 2^bits - 1` — 15 (4-bit), 3 (2-bit), 255 (8-bit). Note this is an **affine
(min/max) quantizer with a zero-point**, unlike the symmetric Opus weight codecs; keys
are not zero-centered per channel and forcing symmetry costs a level.

---

## 3. On-device record layout

One record per tile. Tiles are **channel-major**: K tile = `[head_dim × GROUP]` with
`GROUP = 128` tokens per block, so rows are channels and columns are tokens.

```
[ (r_dim*c_dim) / (8/bits) ]   packed q codes, 8/bits per byte, LSB-first, row-major
[ r_dim * 2 ]                  scale_abs[r_dim]  fp16   (per channel)
[ r_dim * 2 ]                  zp_abs[r_dim]     fp16   (per channel)
[ c_dim * 2 ]                  s_col[c_dim]      fp16   (per token-in-block)
```

```rust
kvarn_record_bytes_bits(r, c, bits) = (r*c).div_ceil(8/bits) + r*2*2 + c*2
```

`bits ∈ {2, 4, 8}`; `bits = 4` is the legacy nibble layout and is byte-identical to it.
Metadata is bit-width-independent, so it is a fixed `2·r_dim + ... ` overhead that
matters more as `bits` drops.

Effective cost, `GROUP = 128`:

| head_dim | bits | record bytes | bits/elem | vs f16 K |
|---|---|---|---|---|
| 256 | 2 | 9 472 | 2.31 | 6.9× |
| 256 | 4 | 17 664 | 4.31 | 3.7× |
| 256 | 8 | 34 048 | 8.31 | 1.9× |
| 128 | 4 | 8 960 | 4.38 | 3.7× |

Metadata overhead is `(4·r_dim + 2·c_dim) / (r_dim·c_dim)` bytes/elem — 0.31 bits/elem
at head_dim 256. **V is not KVarN.** It stays Q8_0 and reuses the asym4 V layout
unchanged, so the headline is a ~4× reduction on the *K* half of the cache.

> **This is a memory lever, not a speed lever.** KVarN 2/4/8 buys KV *capacity*, i.e.
> context length at fixed VRAM. It does not make decode faster on a UMA APU where the
> step is already weight-bandwidth-bound.

### Packing constraints

- `c_dim * bits % 8 == 0` — each row owns whole bytes, so the one-thread-per-row GPU
  pack is race-free without atomics.
- `c_dim <= 256`, `r_dim <= 512` (LDS-sized; gemma4's global layers are head_dim 512).
- `GROUP = 128` is fixed and matches the flash-attention tile width — see §5.

---

## 4. Write path

K arrives from the forward as token-major `[n_tokens × kv_dim]`, `kv_dim = n_kv_heads ·
head_dim`. A record is channel-major over a 128-token block, so there is a transpose:

**`kvarn_gather_k_tiles`** — gather/transpose only, no quantization:

```
T[((b*n_kv_heads + h)*head_dim + ch)*GROUP + tok]
    = K[(b*GROUP + tok)*kv_dim + h*head_dim + ch]
```

Grid `[n_blocks * n_kv_heads]` (one block per output tile), block `[256]`. The caller
slices **complete** blocks only.

**`kvarn_quantize_tile`** — Sinkhorn + per-row affine quantize + pack, one GPU block
per tile, grid `[n_tiles]`, block `[256]`. The tile stays in **global** memory and is
re-read each Sinkhorn pass; only the small per-row/per-column scale arrays live in LDS
(~8 KB). That is deliberate: staging a full tile in LDS hits the gfx1103 large-LDS
hang hazard, and the reduction is register/wave-friendly either way.

GPU uses f32 where the CPU oracle uses f64, so the two are validated by **dequant
cos-sim, not bit-exactness**.

### The recent window

A block can only be quantized once all 128 of its tokens exist. Until then, tokens
live in a per-layer **f32 staging ring** of `GROUP × kv_dim`. On block completion the
ring is flush-quantized into the record cache and reset. So the cache is always
"`n_full_blocks` records + a trailing partial block in f32", and every read path has to
handle both.

---

## 5. Read path

### v1 — shadow cache (superseded, still in tree)

`kvarn_build_kcache` materializes a token-major f16 shadow K cache `[n_tokens ×
kv_dim]` from the records plus the window, transposing back to token-major so an
ordinary f16-K / Q8-V flash kernel can read it. One GPU block per output token.

Correct, and the right first step, but it costs a full `physical_cap × kv_dim` f16
buffer and a per-token pass per layer — which gives back most of the memory KVarN just
saved.

### v2 — fused in-place flash (current)

`attention_flash_kvarn_tile_batched` reads the 4-bit records **in place with inline
dequant**. The enabling coincidence:

> **The flash's 128-token tiles align exactly with the KVarN 128-token blocks.**

So tile selection is a pure index comparison:

```
tile_id <  n_full_blocks  → tokens come from record block tile_id
                            (one [head_dim × GROUP] tile per kv-head, channel-major;
                             deq = (q*scale_abs[d] + zp_abs[d]) * s_col[c])
tile_id == n_full_blocks  → trailing partial block: read the f32 recent window
                            at slot = token - n_full*GROUP
```

Phases B/C/D (softmax + Q8 V accumulation) and the partials layout are **byte-identical**
to `attention_flash_q8_0_tile_batched`, so it shares `attention_flash_asym_reduce_batched`
unchanged. Grid `[n_heads, max_tiles, sub_batch]`, block `[32]`.

This removes the shadow buffer and one kernel launch per layer, and cuts decode K
bandwidth ~4× (4-bit records vs an f16 shadow) — v1 was reading f16 either way.

Head-dim specialization is a template parameter, not a fork: `MAXDPT = max(head_dim)/32`
sizes the per-thread register arrays, the default entry point uses `MAXDPT = 8`
(head_dim ≤ 256 — gemma3/qwen35, byte-identical to before) and `_hd512` uses `MAXDPT = 16`
for gemma4's head_dim-512 global layers. The loop bound `dpt = head_dim/32` is the real
work extent, so head_dim-256 keeps its low register pressure.

---

## 6. Wiring and coverage

`KvQuantMode::Kvarn` (`kv.rs`) sits alongside the older tiers — `Q8`, `Int8`, `Hfq4`,
`Asym{4,3,2}` (Givens-rotated), `Fwht{4,3,2}` (FWHT-rotated, byte-identical storage to
the Asym tiers). The mode is derived from `KvCache`'s exclusive boolean flags by
`kv_quant_mode_from_flags`, which is pure and total so it unit-tests without a GPU.

Cache state (`KvCache`): `quant_kvarn`, `kvarn_bits`, the per-layer recent-window ring,
`kvarn_tiles` (write-side gather scratch, allocated **once per cache** — `GpuTensor` has
no pool-return `Drop`, so a per-call allocation leaks), and `kvarn_shadow` (reserved,
v1's buffer).

| family | status |
|---|---|
| qwen35 | wired, decode + prefill-chunk, FWHT rotation at head_dim 256 |
| gemma3 | wired; requires head_dim ∈ {128, 256} (128 @27B, 256 @4B), rotation at 256 only. Under SWA the KvCache carries only the **global** layers; local layers keep a small f32 ring of the last `swa_window` positions |
| gemma4 | wired incl. head_dim-512 global layers (`_hd512`); the only family declaring `kv = "fp32+kvarn"` in `model-support.toml` |
| zaya | opt-in `HIPFIRE_ZAYA_KVARN=2\|4\|8`; head_dim 128 → no rotation. Prefill (`n=s`) and decode (`n=1`) share one primitive |
| llama | **not** on the kvarn/asym KV path — the global kvarn default falls back to Q8 |

Controls:

```
--kv-mode kvarn                 # hipfire-eval; DEFAULT (with the two-tier hier cache)
--kvarn-bits {2,4,8}            # → HIPFIRE_KVARN_BITS; default 4
--kv kvarn                      # runtime examples (run.rs)
HIPFIRE_KVARN_ROTATE=0          # disable the shared FWHT (A/B); on by default
HIPFIRE_ZAYA_KVARN={2,4,8}      # zaya opt-in
HIPFIRE_KVARN_SIM=1             # perplexity harness: degrade K per group in place
```

Note `--kvarn-bits 2` measures 2-bit **quality** with no storage change if the packer is
left at 4 — codes ≤ 15 fit the same nibble container and dequant kernel. The 2-bit
storage win requires the real sub-nibble packing (`pack_kvarn_tile_bits(qt, 2)`), which
exists; do not conflate the two when reading old probe results.

---

## 7. Results and traps

- **KVarN beats naive per-row 4-bit** on a column-skewed tile — the regression test
  (`balancing_beats_naive_per_row_4bit`) asserts strictly lower MSE. Naive per-row
  cannot see per-column variance spread at all.
- **Sinkhorn collapses imbalance by >2×** on a tile with 128× row-scale spread
  (`sinkhorn_reduces_imbalance`).
- **Round-trip cos-sim: 0.995 un-rotated, 0.999 with the upstream FWHT.** The published
  spec's ≥0.999 gate is measured on the *rotated* tile; the un-rotated core alone does
  not reach it, and that is expected, not a bug.
- **KVarN-4 quality loss is NOT recoverable by light QAT** — measured under STE
  recovery-FT alongside Oq3 weight recovery (which *is* ~52% recoverable). **Deploy
  KVarN-8, not KVarN-4, when quality matters.** This is the single most important
  practical finding here.
- **fp16 metadata drift is real but harmless in aggregate.** On a tile with 128× row
  scale spread, per-element *relative* error explodes on near-zero elements while
  absolute error stays negligible. Validate records with cos-sim (> 0.9999), never
  per-element relative error.
- **Do not stage the tile in LDS** on gfx1103 — large-LDS hang hazard. Global re-read
  per Sinkhorn pass is the tested configuration.
- **Allocate `kvarn_tiles` once per cache.** `GpuTensor` has no pool-return `Drop`.

## 8. Reimplementation checklist

1. `imbalance` + `variance_normalize` (log space, best-so-far, both clamps).
2. Per-row affine quantize on the balanced tile; **absorb `s_row` into `scale_abs`/`zp_abs`**.
3. Record pack/unpack at `bits ∈ {2,4,8}`; assert `c_dim*bits % 8 == 0`.
4. Gather/transpose token-major K → channel-major `[head_dim × GROUP]` tiles.
5. Recent-window f32 ring; flush-quantize on block completion.
6. Read side: align the flash tile width to `GROUP` so tile-id selects record-vs-window
   with no extra indirection. Dequant inline; do not build a shadow cache.
7. Rotate K **and** Q by the same orthonormal FWHT; leave V alone.
8. Validate by cos-sim against the CPU oracle (`kvarn.rs`), not bit-exactness — the GPU
   runs the Sinkhorn in f32.
