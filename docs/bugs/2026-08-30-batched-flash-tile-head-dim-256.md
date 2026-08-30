# Four batched flash tiles are head_dim=256-only, and nothing said so

Status: found 2026-08-30 on master `e4025250f`, nix2. **GUARDED, not fixed** —
the affected kernels now refuse a `head_dim` they cannot compute instead of
corrupting silently. Teaching them the runtime `dpt` is the real fix and needs a
GPU to validate; scope at the end.

Found by running the second never-executed sweep from
`docs/bugs/2026-08-29-hunt-coverage-gaps.md` — "46 `X` / `X_batched` kernel
sibling pairs in `kernels/src`; 'fixed one path, not its sibling' was the most
common confirmed shape in both waves".

## Method

Rather than read 46 pairs, rank them by **commits that touched exactly one
side**. That surfaced `attention_flash_q8_0_tile` immediately: three commits on
the non-batched file that never reached `_batched`, one of them
`5e34d05c5 fix(attention): Q8_0 flash tile — long-decode NaN`.

That commit fixed two defects in the non-batched tile:

1. partials indexed by `n_tiles` (dynamic) where the reduce reads `max_tiles`
   (static);
2. the tile hardcoded 4 dims × 32 threads = 128 dims, so `head_dim=256` left the
   upper half reading uninitialized partials — fixed with
   `n_halves = (head_dim + 127) / 128`.

Defect 1 **did** reach the batched sibling. Defect 2 did not — and is present in
three more kernels.

## The defect

`attention_flash_asym_reduce_batched` derives `dpt = head_dim / 32` at runtime,
with a comment saying it does so "exactly like the tile kernel". Nine batched
flash tiles feed that reduce. The claim holds for five of them:

| kernel | dims per lane | general? |
|---|---|---|
| `attention_flash_kvarn_tile_batched` | `d0 = lane * dpt`, `dpt = head_dim / 32` | yes |
| `attention_flash_asym2_tile_batched` | `half * 128 + tid * 4`, `n_halves = head_dim / 128` | yes |
| `attention_flash_asym4_tile_batched` | same | yes |
| `attention_flash_fwht2_tile_batched` | same | yes |
| `attention_flash_fwht4_tile_batched` | same | yes |
| **`attention_flash_asym3_tile_batched`** | **`d0 = tid * 8`** | **256 only** |
| **`attention_flash_fwht3_tile_batched`** | **`d0 = tid * 8`** | **256 only** |
| **`attention_flash_q8_0_tile_batched`** | **`d0 = tid * 8`** | **256 only** |
| **`attention_flash_f16k_q8v_tile_batched`** | **`d0 = tid * 8`** | **256 only** |

Note the split: the EVEN-bit variants (asym2/4, fwht2/4) got the halves loop; the
ODD ones (asym3, fwht3) did not. Same shape, one more level down.

At `head_dim == 128` a lane with `tid >= 16` gets `d0 >= 128`, so it

- reads `q_head[d0 + i]` past its own head, into the next head's Q (and past the
  buffer entirely for the last head of the last batch row), and
- writes `p[2 + d0 + i]` up to index 257 against a partials stride of
  `2 + head_dim` = 130 — into the **next tile's** slot for the same head.

No error, no NaN necessarily, just wrong attention output. At `head_dim > 256`
the upper dims are never written — defect 2 verbatim, in its batched mirror.

`launch_asym_flash_batched` passes `head_dim` straight through and checks
nothing; neither does any caller in `hipfire-dispatch`.

## Live exposure is narrow, which is why it survived

- `asym3`, `fwht3` and `q8_0` are all in `DEPRECATED_KV_MODES`, so today they are
  reachable only behind `HIPFIRE_KV_ALLOW_DEPRECATED=1`.
- `attention_flash_f16k_q8v_batched_masked` has **no production caller** — only
  `parity_kvarn_flash.rs` and `parity_kvarn_fused_flash.rs`. Those are parity
  harnesses, so at `head_dim=128` they would have reported a divergence caused by
  the harness's own kernel.
- The shipping KV family (kvarn) uses `attention_flash_kvarn_tile_batched`, which
  is correct.

So this is not a live serving corruption today. It is a loaded gun: the guard was
absent, the constraint was undocumented, and the reduce kernel's comment actively
asserted the opposite.

## What was done

- `Gpu::require_head_dim_256` refuses any `head_dim != 256` on the four wrappers,
  naming the kernel and the cause. Refusal, not a clamp: there is no other
  `head_dim` these kernels compute correctly, so every refused call was already
  returning garbage. Same precedent as the cohere2 sliding-window message and the
  AWQ/G128 refusal.
- The reduce kernel's "exactly like the tile kernel" comment is corrected to say
  which tiles do and do not.
- `the_256_only_list_matches_the_kernel_sources` scans all nine tile sources and
  fails if a kernel hardcodes `d0 = tid * 8` without being guarded, or is guarded
  without hardcoding it. That is the drift catcher — the list of laggards is only
  correct until someone edits a `.hip`.

## Scope of the real fix

Give the four tiles `const int dpt = head_dim / 32;`, `d0 = lane * dpt`, and a
`MAXDPT` register-array bound, mirroring `attention_flash_kvarn_tile_batched`
which already ships that pattern. For `head_dim == 256` this is arithmetically
identical (`dpt == 8`), so the risk is not correctness but the loss of the
compile-time `#pragma unroll` — which is exactly what needs a GPU to measure.
Validate with `parity_kvarn_flash` at both 128 and 256, then drop the guard and
the entries from `HEAD_DIM_256_ONLY_TILES`; the test will insist you do both.
