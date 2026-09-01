# GGUF import silently scrambles Q4_0 and mis-decodes Q5_K

Status: found and **FIXED** 2026-08-29, master `0c9e3d252`, nix1. Confirmed by
source read plus an arithmetic reproduction, then fixed and pinned by
`dequant_layout_tests` in the same file — five tests written from the upstream
`dequantize_row_*` formulas. Re-introducing either defect fails exactly the two
corresponding tests and leaves Q4_K/Q6_K/Q8_0 passing. Both defects are in
`crates/hipfire-gguf/src/lib.rs` and both are silent — the resulting `.hfq`
loads and generates fluent-looking text.

## Symptom

None. That is the problem. `hipfire-coexistence import gguf` accepts a Q4_0 or
Q5_K GGUF, prints no warning, and writes an artifact whose weights are wrong.
Only a KLD or perplexity run against the source model would show it.

## Bug 1 — Q4_0 nibble order (`lib.rs:305`)

GGML packs element `j` in the **low** nibble of byte `j` and element `j + 16` in
the **high** nibble. The reference decoder (`dequantize_row_q4_0`) is:

```c
for (int j = 0; j < qk/2; ++j) {
    y[i*qk + j       ] = ((qs[j] & 0x0F) - 8) * d;
    y[i*qk + j + qk/2] = ((qs[j] >>   4) - 8) * d;
}
```

`dequant_q4_0` instead writes the pair to adjacent slots:

```rust
let idx = b * block_size + j * 2;      // lib.rs:305
if idx < n { out[idx] = lo as f32 * scale; }
if idx + 1 < n { out[idx + 1] = hi as f32 * scale; }
```

That is a permutation within each 32-element block: **30 of 32 elements land in
the wrong slot** (only elements 0 and 31 are fixed points). No data is lost, so
nothing downstream can notice — the tensor is simply shuffled along its rows
before requantization.

Reproduction, packing a known ramp per the GGML layout and decoding both ways:

```
expected (ggml) : [0,1,2,...,15, 0,1,2,...,15]
hipfire actual  : [0,0,1,1,2,2,...,15,15]
elements moved  : 30 of 32
```

## Bug 2 — Q5_K high-bit selection (`lib.rs:425-426`)

The 5th bit of each weight lives in `qh`, and the reference advances the bit
pair by **two per 64-element group** (`u1 = 1, u2 = 2`, then `u1 <<= 2; u2 <<= 2`
each group). So group *g* uses bits `2g` and `2g+1`.

`dequant_q5_k` uses `group` and `group + 4`:

```rust
let hbit  = ((qh[l] >> group) & 1) as u8;        // lib.rs:425
let hbit2 = ((qh[l] >> (group + 4)) & 1) as u8;  // lib.rs:426
```

| group | half | ggml bit | hipfire bit | |
|---|---|---|---|---|
| 0 | low  | 0 | 0 | ok |
| 0 | high | 1 | 4 | **wrong** |
| 1 | low  | 2 | 1 | **wrong** |
| 1 | high | 3 | 5 | **wrong** |
| 2 | low  | 4 | 2 | **wrong** |
| 2 | high | 5 | 6 | **wrong** |
| 3 | low  | 6 | 3 | **wrong** |
| 3 | high | 7 | 7 | ok |

**6 of 8 sub-blocks read the wrong bit.** Only group 0's low half and group 3's
high half coincide. The high bit contributes 16 to a 0..31 quant value, so every
element whose true and read bits differ is off by `16 * scale` — half the range.

## Why this is credible on its face

The same file gets it right twice. `dequant_q4_k` (`lib.rs:373`) uses
`idx_odd = idx_even + 32`, the correct low/high split, and `dequant_q6_k`
(`lib.rs:444`) matches the reference layout exactly. Only the two decoders above
diverge, so this is a local mistake, not a house convention.

## Reachability

Shipping CLI, no flag required:

`hipfire-coexistence import gguf --input <model.gguf> --output <model.hfq> --format <FMT>`
→ `cli.rs:68` → `import_gguf` (`cli.rs:168`) → `run_gguf_pipeline`
→ `gguf_import.rs:244/249/255/300/356/416` → `gguf_input::tensor_to_f32`
→ the two decoders.

Unsupported GGML types panic explicitly at the dispatcher's `other =>` arm, so
there is no silent fall-through for the formats that are *not* implemented — the
silence is specific to these two that are.

## Why nothing caught it

`crates/hipfire-gguf/src/lib.rs` contains **zero tests** (`grep -c '#\[test\]'`
→ 0), and the crate appears nowhere in `tests/no-gpu-ci.sh`. There is no decode
test pinning any GGML layout against a reference block, so a decoder can be
wrong in a way that is arithmetically silent and nothing will say so.

Both defects are also the *silent* kind by construction: Q4_0's is a bijection
(no value is lost, only moved) and Q5_K's produces in-range values. Neither
trips a bounds check, a NaN guard, or an admission predicate. The import
completes, the artifact loads, and generation is fluent.

## Fix

Q4_0 — match the reference (and the Q4_K sibling):

```rust
let idx = b * block_size + j;
if idx < n { out[idx] = lo as f32 * scale; }
if idx + 16 < n { out[idx + 16] = hi as f32 * scale; }
```

Q5_K — advance the bit pair by two per group:

```rust
let hbit  = ((qh[l] >> (2 * group)) & 1) as u8;
let hbit2 = ((qh[l] >> (2 * group + 1)) & 1) as u8;
```

Worth adding a decode test per format against a hand-packed block; the existing
Q4_K/Q6_K decoders would pass it unchanged and pin the convention.
