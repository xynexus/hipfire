# Opus Quant (OQ) — codec family, shared W{2,4,8}A8 kernel, calibration

Reimplementation-grade spec. Sources of truth:

| Thing | File |
|---|---|
| On-disk byte identity | `crates/hipfire-quant-format/src/lib.rs` |
| Weight codecs (encode/decode) | `crates/hipfire-quantize/src/codecs.rs` |
| FWHT rotation | `crates/hipfire-primitives/src/fwht.rs` |
| LDLQ / OBS error feedback | `crates/hipfire-quantize/src/ldlq.rs` |
| AWQ / SmoothQuant scales | `crates/hipfire-quantize/src/main.rs` (`compute_awq_scales`) |
| Mixed-precision tier assignment | `crates/hipfire-quantize/src/mixed_precision.rs` |
| Unsigned offset-fold reference | `crates/hipfire-quantize/src/opus_lowbit.rs` |
| Shared GEMM | `kernels/src/gemm_opus_tiled_wmma.hip` |
| Activation quantizers | `kernels/src/quantize_act_oq{4,8}.hip` |
| Load-time repack | `crates/hipfire-runtime/src/oq{4,8}_arch.rs` |
| Hessian package reader | `crates/hipfire-quantize/src/hessian_io.rs` |

---

## 1. The one-paragraph version

Opus Quant is a **symmetric**, **per-256-group**, **FWHT-rotated** integer weight
format. One codec shape spans 2/3/4/6/8 bits: rotate a 256-weight group with a
fixed signed Walsh-Hadamard transform, clip-search one f32→f16 scale for the
group, round to a symmetric signed grid, pack. Activations are quantized
**dynamically at runtime** (never stored) into int8 or int4 with a matching
per-group scale, after the same rotation is applied to `x`. Because both sides
land on symmetric signed integers with scales outside the accumulator, a single
`v_wmma_i32_16x16x16_iu8` kernel body serves W8A8 and W4A8 (and, via the
unsigned offset-fold variant, W2A8/W1A8) — the weight bit-width only changes how
the weight fragment is *fetched*.

---

## 2. Rotation: what is stored rotated (asked directly)

**Weights: stored ROTATED on disk.** Every `quantize_oq*g256` calls
`cpu_fwht_256(&mut group, signs1, signs2)` *before* computing the scale and
rounding. The bytes in the `.hfq` are quantized coefficients in the Hadamard
basis. There is no un-rotate at load; `dequant_oq*g256` exists only as a test
oracle and applies the inverse (same transform, signs swapped).

**Activations: NOT stored — rotated and quantized per forward pass.** The
runtime applies the *same* FWHT-256 to `x` before quantizing it (`RotationPlan::FwhtG256`
for `Oq4G256 | Oq8G256`, `crates/hipfire-dispatch/src/types.rs:119`), usually
fused into the preceding rmsnorm or SiLU·mul kernel (`fused_rmsnorm_mq_rotate.hip`,
`fused_silu_mul_mq_rotate.hip`). Because the FWHT is orthonormal,
`⟨R w, R x⟩ = ⟨w, x⟩` — the dot product is identical, but both operands are now
incoherent (outlier energy spread across the group), which is what makes a single
per-group scale survive at 4 and 3 bits.

**AWQ scale: stored, unrotated, as a sidecar.** `--awq` folds `W ← W·s` in the
**plain** basis *before* the FWHT bake-in, and writes `s` as a 1-D f16 tensor
named `<weight_name>.awq_scale`. The runtime divides `x ← x/s` *before* the
rotation kernel (`rotate_x_mq_awq`), so `(W·s)·(x/s) = W·x` cancels. This
ordering is the whole point: FWHT flattens per-channel importance inside a
group, so per-channel scaling has to happen in the unrotated basis or it has
nothing to act on.

The sign tables `signs1`/`signs2` are not stored — they are regenerated from a
fixed LCG seed on both sides (`gen_fwht_signs`, `fwht.rs:46`). Writer and reader
must agree on the seed.

### FWHT-256 (exact)

```
signed_fwht(x[n], signs1, signs2):        # n = 256
    x[i] *= signs1[i]                     # pre-sign
    for stride in 1,2,4,...,n/2:          # in-place butterfly
        for each block of 2*stride:
            a, b = x[i+j], x[i+j+stride]
            x[i+j], x[i+j+stride] = a+b, a-b
    x[i] *= (1/sqrt(n)) * signs2[i]        # 1/16 for n=256, orthonormal
```
Inverse = the same call with `signs1` and `signs2` swapped.

---

## 3. On-disk formats

All Opus types use **group = 256 elements**, scale = **f16 little-endian**, codes
**symmetric two's-complement**, and deliberately drop the asymmetric negative
endpoint (`-8`, `-128`, `-4`, `-32`, `-2`) so `|qmin| == qmax`.

| `QuantType` | id | block bytes | b/w | payload after the 2-byte f16 scale | grid |
|---|---|---|---|---|---|
| `Oq2G256` | 39 | 66 | 2.0625 | 64 B, 4 codes/byte, `(q&3) << 2j` | `[-1, 1]` |
| `Oq3G256` | 38 | 98 | 3.0625 | 8 sub-blocks × 3 u32 **bit-planes** (32 weights each) | `[-3, 3]` |
| `Oq4G256` | 34 | 130 | 4.0625 | 128 B nibbles, `byte = q_even \| (q_odd << 4)` | `[-7, 7]` |
| `Oq6G256` | 40 | 194 | 6.0625 | 192 B, 4 codes per 3 bytes | `[-31, 31]` |
| `Oq8G256` | 35 | 258 | 8.0625 | 256 B int8 | `[-127, 127]` |

Derived / variant ids:

| Type | id | What differs |
|---|---|---|
| `OqPlusG256` | 33 | **Same bytes as `Oq4G256`**; loader nibble-expands to int8 and runs the A8 kernel. W4A8. |
| `OqPlusCompact` | 36 | `[f16][128 nibbles][N_out × (u8 idx, i8 val)]` = `130 + 2·N_out`. int4 bulk + sparse int8 outlier overlay. |
| `Oq4G256ArchPacked` | 37 | `Oq4G256` payload already in the device layout (`hipfire optimize` pre-bakes the repack). |
| `Oq8G256RowPadded` | 43 | Each **row** starts a fresh group sequence (`M·ceil(K/256)` blocks); ragged K. XDNA-native. GPU kernels must reject it. |
| `Oq8Plain` / `Oq4Plain` / `Oq4MixedPlain` | 45/46/47 | Identical byte geometry, **no FWHT on either side**. DFLASH/NPU artifacts. |

Byte-length contract: `tensor_bytes(n) = ceil(n / 256) * block_bytes`, except
`Oq8G256RowPadded` (`matrix_tensor_bytes(rows, cols)`) and the variable-length
`OqPlusCompact`. Single-sourced in `QuantType::block_bytes()`.

### Encoder (all widths, one shape)

```rust
for each 256-element group:
    pad to 256 with zeros (tail groups)
    cpu_fwht_256(group, signs1, signs2)
    scale = symmetric_clipsearch(group, qmax)     // see §5
    write f16(scale)
    for each weight: q = clamp(round(w / scale), -qmax, qmax); pack q
```
`qmax` is 1, 3, 7, 31, 127 for oq2/3/4/6/8. Dequant is `scale · sext_b(code)`
followed by the inverse FWHT.

**Bit-plane detail (oq3).** Per 32-weight sub-block, emit three LE u32:
`p0 |= (u & 1) << i`, `p1 |= ((u>>1) & 1) << i`, `p2 |= ((u>>2) & 1) << i`, where
`u = (q as u8) & 7`. This *is* the kernel layout — the W3A4 GEMM does a Morton
spread from planes to int4 rather than an unpack-then-repack.

**6-bit packing (oq6)**, 4 codes per 3 bytes with `q = code & 0x3f`:
```
b0 = q0 | (q1 << 6)
b1 = (q1 >> 2) | (q2 << 4)
b2 = (q2 >> 4) | (q3 << 2)
```

---

## 4. The shared kernel

### 4.1 Signed path — W4A8 / W8A8, one body

`kernels/src/gemm_opus_tiled_wmma.hip`. gfx1103+ RDNA3, wave32, **zero LDS**.

```
Y[b,m] = Σ_g  sw[m,g] · sx[b,g] · Σ_{k∈g} qw[m,k] · qx[b,k]
```

Buffers:

```
W  : uint8  [M, K]   (W8, int8)   or  [M, K/2]  (W4, packed nibbles)
Ws : f32    [M, K/group]          weight group scales (f16 → f32 at load)
X  : int8   [B, K]                rotated + dynamically quantized activations
Xs : f32    [B, K/group]          activation group scales
Y  : f32    [B, M]                column-major in M (Y[out_col * M + out_row])
```

Requires `K % group == 0`, `group % 16 == 0`. Grid `[ceil(M/(16·MB)), ceil(B/(16·NB))]`,
block `[32]`. Instantiated at `MB×NB` = 2×2 and 2×4.

The *entire* W4/W8 difference is the fragment fetch, and `WBITS` is a macro
parameter so every branch folds at compile time:

```cpp
woff     = (WBITS==8) ? (g*group + kt*16) : (g*group/2 + kt*8);
wstride  = (WBITS==8) ? K : K/2;
a_frag   = (WBITS==8) ? opus_load_i8x16(w_row + woff)      // 16 B → int32x4
                      : opus_unpack_i4x16(w_row + woff);   // 8 B → 16 int8
```

Nibble sign-extension is branch-free: `(nib ^ 0x8) - 8`, K order preserved
(low nibble = even k) so the unpacked fragment is bit-identical to a stored-int8
row. Inner product:

```cpp
iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
           true /*W signed*/, a_frag, true /*X signed*/, x_frag, iacc, false);
```

Scales never enter the WMMA. One **int32 accumulator per group** (fresh each `g`,
giving ILP across the group loop), rescaled into an f32 accumulator at the group
boundary: `facc += (float)iacc * sw * sx`. Fragment layout: `acc[j]` at lane `t`
is `(row = 2*j + (t>>4), col = t & 15)`.

Out-of-range rows/cols clamp their pointer to the last valid row (harmless reads)
and zero their scale, so the tile body stays branch-free; only the store is guarded.

### 4.2 Unsigned offset-fold path — W1/W2/W4/W8, one body

Non-power-of-two-friendly and cheaper to unpack. Stores **unsigned** codes
`u ∈ [0, 2^b-1]` representing `q = u - Z`, `Z = 2^(b-1)`, and folds the zero-point
out of the accumulator:

```
Σ_k (u_k - Z)·x_k  =  ⟨unsigned WMMA⟩  -  Z · Σ_{k∈g} x_k
```

`Σ x` is produced once by the activation quantizer (`quantize_act_oq8_sum`, an
extra `Xsum : int32 [B, K/group]`), so the fold costs **one int32 subtract per
group** and removes all sign extension from the unpack. The WMMA weight operand
is flagged unsigned (`false` for the first signedness flag). This is an *exact
integer identity*, not an approximation — `opus_lowbit.rs` asserts the folded and
signed dots are bit-identical f32 for every width.

Packing is dense LSB-first: code `j` occupies bits `[(j*b)%8 .. +b]` of byte
`j*b/8`. Unpack is pure mask/shift (`opus_unpack_uNx16<WBITS>`), the scalar
analogue of a `v_perm_b32` lane gather. Entry points: `gemm_opus_w{1,2,4,8}a8u_tiled_wmma_{2x2,2x4}`.

Widths 3/5/6/7 use the bit-plane layout instead of dense packing; the fold and
the iu8 core are unchanged.

CPU reference and parity oracle for this path: `opus_lowbit::{quantize_symmetric,
pack_dense, unpack_dense, dot_offset_fold, dot_signed, group_sums_i8}`.

### 4.3 Activation quantizers

`quantize_act_oq8` — grid `[K/group, B]`, block `[32]`, zero LDS:

```
amax  = wave-shuffle absmax over the 256-element group
scale = amax / 127          (1.0 if amax == 0)
Xq[i] = clamp(lround(x[i] * 127/amax), -127, 127)
Xs[b, g] = scale
```

`quantize_act_oq8_sum` additionally reduces `Σ Xq` into `Xsum[b,g]` for the fold
path. `quantize_act_oq4` is identical with `qmax = 7` and nibble packing
(`byte = k_even | k_odd<<4`), feeding the iu4·iu4 W4A4 GEMM.
`quantize_act_int8_per_token` is the coarser per-**row** variant (one scale per
token) used by reference/parity paths.

Input to all of these is **already rotated** (and already `/s` if AWQ is active).

### 4.4 Load-time repack

On-disk blocks interleave scale and payload; kernels want them split.

- **OQ8 family** (`oq8_arch.rs`) → combined `[int8 W  m*k][f32 scales m*ng]`.
  Three on-disk types converge here: qt=35 copies, qt=33 sign-extends nibbles to
  int8, qt=36 expands the bulk then overlays the sparse int8 outliers. The
  forward derives the scale pointer as `sub_offset(m*k, ..)` and dispatches one
  iu8 path.
- **OQ4** (`oq4_arch.rs`) → `[split nibbles m*(k/2)][split f32 scales m*ng][interleaved m*ng*(4+128)]`.
  The prefill GEMM reads the split planes; the decode GEMV reads the interleaved
  `[f32 scale][128 nibbles]` stream for one coalesced fetch. `hipfire optimize`
  can pre-bake this and stamp qt=37 so load becomes a pure upload.

---

## 5. Calibration

Four levers, composable, marked positionally in the artifact name
(`oq4`, `oq4+`, `oq4++`).

### 5.1 Clip-search (always on)

`symmetric_clipsearch(group, qmax)` — the scale is *not* `amax/qmax`. Grid-search
9 clip fractions `[1.0, 0.95, …, 0.6]`, keep the one minimizing plain MSE over the
rotated group:

```rust
for c in [1.0, .95, .9, .85, .8, .75, .7, .65, .6]:
    scale = max(c * amax / qmax, 1e-12)
    err   = Σ (v - clamp(round(v/scale), -qmax, qmax) * scale)²
pick argmin
```

Costs nothing at inference and is unconditional in every `quantize_oq*g256`.

### 5.2 Calibration data capture

Two statistics, both collected by a forward pass over a corpus and written into a
unified `.calib.hfq` (HFQM) package, one entry per dense projection:

| Entry | Shape | Meaning |
|---|---|---|
| `<tensor>.imatrix` | `[K]` | `Σ x²` per input channel — activation energy |
| `<tensor>.hessian` | `[K,K]` | `Σ x·xᵀ` — the GPTQ/OBS Hessian |

Collector: `crates/hipfire-runtime/src/calibration.rs` (`CalibCollector`, armed via
`gpu.active_capture`). Hessians are spooled to disk per layer as the layer streams
in — a 9B model's Hessian set is multi-GB, and a 397B MoE's would be ~196 GB, so
**MoE routed experts are imatrix-only** (`with_imatrix_only(substrings)`); the
quantizer then skips LDLQ for those tensors and falls back to the AWQ-only path.

Compact storage: exact f32 diagonal + bf16 lower strict triangle
(`quant_type = 130`, calibration-only). Reader is mmap + zero-copy
(`hessian_io.rs`), promoting to f64 only at Cholesky time.

```bash
hipfire-coexistence calibrate \
  --model <safetensors-dir> --corpus benchmarks/calib/calib-5m.txt \
  --output ~/.hipfire/calib/<Model>.calib.hfq --kldref \
  --sequence-batch 64 --max-rows 2048 [--expert-capture-target 4096 ...]

hipfire-coexistence artifact audit-calibration --input <...>.calib.hfq
```

> Corpus traps: the sampler reads only a **prefix** of each source, and a sample
> cannot span a blank line — interleave deficit-first or the language mix is a
> lie. An English-only corpus provably starves MoE experts.

### 5.3 `+` — AWQ / SmoothQuant (activation-aware scaling)

`--awq [alpha]`, default `alpha = 0.55`. Per input channel `j`:

```
s[j] = rms_act[j]^α ,  rms_act[j] = sqrt(imatrix[j] / n_tok)
```
computed in log space (`half_alpha * ln(v)`) for dynamic range, then normalized so
`geomean(s) = 1`. `W ← W · s` is folded offline in the **plain** basis before the
FWHT; `s` ships as `<weight>.awq_scale`; the runtime does `x ← x/s` before rotating.

Robustness that matters in practice: clamp `in_sum2` to `[1e-12, 1e30]` before the
log. An `inf` (f32 overflow during collection — observed on a 27B tier-1 imatrix)
makes `mean_log = inf`, then `l - mean_log = NaN`, which survives the output clamp
and NaNs the entire forward.

`--sq-split [frac]` (default 0.01) geo-mean-normalizes the top-`frac` channels by
energy and the bulk **separately**, so outlier energy doesn't skew the bulk's
migration scale. Each group's geomean stays 1, so the cancellation identity holds.

The plain-basis (non-rotated) analogue for the fold GEMM is
`opus_lowbit::quantize_symmetric_clip`: an importance-weighted clip-search
minimizing `Σ_c imp[c]·(w_c − ŵ_c)²` over `n_steps` clip fractions.

Tuning note: `alpha = 0.55` came from an FFN-heavy sweep; it is too aggressive for
Mamba-2 activations (nemotron `oq4+` failed until `--awq-alpha 0.1`).

### 5.4 `++` — LDLQ / OBS error feedback

`--hessian <pkg>` + `oq{2,3,4,8}++`. `oq*_ldlq_pack` in `ldlq.rs`. MSE-optimal
quantization is *not* output-optimal; LDLQ minimizes `‖(W−Ŵ)·√H‖`.

```
1. H ← R H Rᵀ                       # rotate_hessian: row pass then column pass,
                                    # same per-256 FWHT as the weights
2. L with L·Lᵀ = (H_rot + λI)⁻¹     # double Cholesky: llt(H+λI) → solve for H⁻¹ → llt
                                    # λ escalates ×10 up to ×10⁴ on breakdown
3. residual ← FWHT(W)               # weights into the same incoherent domain
4. for each 256-column block, in order:
       per row: scale = symmetric_clipsearch(residual[block], qmax)
                q     = clamp(round(residual/scale), -qmax, qmax)
                emit  = pack(q)                      # the format's own packing
                err_c = (residual_c - q_c·scale) / L[c,c]
       propagate: residual[f] -= err_c · L[f,c]  for every later column f
```

`U = Lᵀ`, so the OBS divisor is `L[c,c]` and the propagation weight is `L[f,c]` —
returning `L` and indexing it transposed avoids a second transpose pass. Output
stays **rotated** (no un-rotate) and is written directly in the target format's
block layout, so only the inner quant range and packing differ between
`oq2/oq3/oq4/oq8_ldlq_pack`. `None` on Cholesky breakdown → caller falls back to
the plain RTN codec. Row loop is `rayon`-parallel; the block loop is sequential by
construction.

AWQ and LDLQ compose: `W·s` is folded first, then LDLQ runs on the smoothed
weights (`ldlq+awq: <name> OBS int4 + smooth`).

### 5.5 Mixed precision (`oq4.25`, `--mix-target-bpw`)

Two independent mechanisms:

**Intra-group (OQ+).** Within one 256-group, promote the top `N_out` positions to
int8 while the bulk stays int4, sharing **one group scale**. Selection is by
*int8-upgrade gain*, not raw magnitude:

```
g_i = (w_i − clamp(round(w_i/s), −7, 7)·s)²  −  (w_i − clamp(round(w_i/s), −127,127)·s)²
```

FWHT flattens per-position activation energy inside the group, so output-error
saliency reduces to the weight-side gain — protect the positions int4 quantizes
*worst*, not the ones that are merely large.

The scale and the promotion set are chosen **jointly**, by `mixed_clipsearch` —
the single selector shared by every mixed packer (both compact and int8-stored,
LDLQ and not). It sweeps a 14-point grid reaching down to `0.35·amax/7`,
recomputing the top-`N_out` set at each candidate and keeping the lowest total
group error. This is an exact minimisation over the grid, not a heuristic: the
error is separable across positions, so for a fixed scale the top-`N_out` gain
sort *is* the optimal set. It matters because the promoted positions escape the
±7 clamp and therefore tolerate a much tighter scale than the int4-only
`symmetric_clipsearch` of §5.1 would ever propose — and the tighter scale is what
makes the bulk more accurate. Measured on Qwen3.5-0.8B, choosing the scale before
the set (the pre-2026-08 behaviour) cost 10% group SSE at `N_out=7` and 31% at
`N_out=15`, and flattened the payoff of additional outliers almost to zero past
`N_out=7`.

`quantize_oqplus_tiered` stores the
result as plain `Oq8G256` (a faithful quality probe at 8 b/w);
`quantize_oqplus_compact` stores it as `130 + 2·N_out` B/group (qt=36). Format
name encodes it: `storage_bits = 4.0625 + N_out/16`, so `oq4.25` ⇒ `N_out = 3`
outliers per group (`w8_frac = N_out/256`, `130 + 2·3 = 136` B/group). `--w8-top <frac>`.

**Inter-tensor.** `mixed_precision.rs` ranks dense linears by imatrix-weighted
output error at the LOW tier and greedily promotes by sensitivity density
(`sensitivity / numel`) until the target average b/w is hit:

```
sensitivity(T) = Σ_c imatrix[c] · Σ_rows (W[r,c] − dequant(quant(W))[r,c])²
```
Tier b/w constants: oq2 = 2.0625, oq4 = 4.0625, oq8 = 8.0625. The greedy keeps
trying smaller tensors after a large one fails to fit, rather than stopping.

Policy: under a 4-bit budget, important tensors go to **OQ8**, not the legacy Q8 —
same Opus iu8 kernels, one dispatch path. oq2 is a low-importance *tail* format,
never standalone.

### 5.6 CLI summary

```bash
hipfire-quantize --input <src> --output <out>.hfq --format oq4++ \
  --hessian ~/.hipfire/calib/<Model>.calib.hfq   # ++ : LDLQ (implies imatrix)
  --awq [0.55] --sq-split [0.01]                 # +  : AWQ / SmoothQuant
  --w8-top 0.015                                 # OQ+ intra-group outliers
  --mix-target-bpw 4.25                          # inter-tensor tiering
  --embed-precision source                       # keep the embed table un-quantized
```

Default format is `oq4.25++`. `oq4+`/`oq8+`/`oq2+` ⇒ AWQ; `oq4++`/`oq8++`/`oq2++`
⇒ AWQ + LDLQ. Legacy `op4`/`op8` spellings normalize to `oq4`/`oq8`; `op` is not
valid for new artifacts.

---

## 6. Reimplementation checklist

1. `signed_fwht` + `gen_fwht_signs` (LCG seed must match writer and reader).
2. `symmetric_clipsearch` — 9-point grid, plain MSE, on the *rotated* group.
3. One encoder per width: rotate → clip-search → f16 scale → round to `±qmax` → pack.
4. Runtime: rotate `x` (fused into rmsnorm / SiLU·mul), then per-group dynamic
   int8 (or int4) quantize with `amax/qmax`, emitting `Xs` (and `Xsum` for the
   unsigned fold).
5. GEMM: per-group int32 WMMA accumulator, rescale by `sw·sx` at the group
   boundary, never inside the WMMA.
6. Calibration is strictly additive — a correct `oq4` (no calibration at all) is
   the floor; `+` and `++` only change the numbers written into the same bytes.

## 7. Known results and traps

- **Do not quantize the embedding table.** It seeds the residual stream
  unnormalized across every layer; an 8-bit embed costs ~40% of the oq4 KLD budget
  to save ~500 MB. `--embed-precision source` is the default for that reason.
- **W×A precision:** A8 ≈ A16; ~6 dB per bit from W3→W4; LDLQ ≈ +1.6 dB; A4 costs
  ~3.5 dB. On held-out data, plain `XᵀX` LDLQ ≈ no calibration — GuidedQuant is
  the only robust winner so far.
- **oq2 alone is dead** as a general format; it is a mixed-precision tail. The
  low-rank residual correction (`HIPFIRE_LOWRANK_R`) is the strongest known lever
  at 2 bits (−13%).
- **oq3 needs more than the FWHT.** The fixed rotation is the floor; W3A4 is only
  viable atop a SpinQuant learned rotation. W3 wins 1.29× over W4 only when
  weight-bandwidth-bound (small batch).
- **gfx1151 int4 is 2.0× int8 at ISA rate** — the W4A4 premise holds; an earlier
  ~1.0× measurement was a bandwidth-bound artifact.
- **Nothing that ranks positions inside a rotated group can use a per-channel
  saliency.** The signed Hadamard's entries all share one magnitude, so for any
  per-input-channel weighting `s`, `[R·diag(s)·Rᵀ]ᵢᵢ = mean(s)` at every
  position — a per-position reweight scales all candidates by one constant and
  cannot reorder them. This is exact (`hipfire-primitives` test
  `rotation_flattens_any_per_channel_saliency`), and 97% of the weighted-error
  mass is off-diagonal besides. It kills the "reweight `mixed_overlay_indices`
  by GuidedQuant saliency" idea outright. Saliency reaches this codec only
  before the rotation (`+`, per-channel AWQ scaling) or through the off-diagonal
  coupling (`++`, LDLQ/OBS) — both already shipped.
- **A per-group outlier budget is dead — the address costs more than the
  allocation is worth.** Letting each 256-group pick its own `N_out` instead of
  a per-tensor constant does help: water-filling at the same total slot count
  cuts weight SSE 5.9–11.1% on Qwen3.5-0.8B. But variable-length blocks are the
  only container that realizes it, and their `[u32; n_groups]` offset prefix
  costs 4 B/group against a 136 B block — which buys back 1 of the 3 overlay
  slots. At equal bytes the best *possible* allocation is 6.9–11.9% **worse**
  than uniform `N_out=3`. That figure is a Lagrangian bound, not a heuristic's
  output, so no smarter allocator recovers it. Break-even sits between 2 and 3
  bytes of addressing overhead, and 2 B caps a tensor at 65535 blocks while 1 B
  cannot address a block at all. Reproduce with
  `examples/opus_group_budget_study.rs`. The *tensor*-level version of the same
  idea already shipped as `HIPFIRE_OUTLIERS_BY_LAYER`.
- `hipfire lock acquire` around any non-daemon GPU binary; `calibrate`
  **self-locks**, so never wrap it in `lock run`.
