# Four batched flash tiles are head_dim=256-only, and nothing said so

Status: found 2026-08-30 on master `e4025250f`, nix2. **GUARDED, not fixed** —
independently confirmed against a hardware repro in a disconnected lineage (see
below).
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

## Reachability

`head_dim == 128` is not hypothetical. Both of these are real artifacts in the
store:

- `qwen3.5-0.8b--oq4++.hfq` — `q_proj [4096, 1024]`, `k_proj [512, 1024]`, i.e.
  32 heads x 128 and 4 kv-heads x 128. The qwen35 family is the **only** producer
  of `KvTierInputs` today, so it is exactly the family that reaches these tiles.
- `BLS-Mini-Code-1.0--bf16.hfq` — arch 25 (cohere2-moe), `head_dim 128` per
  `hipfire inspect`.

What narrows it is the KV mode, not the model:

- `asym3`, `fwht3` and `q8_0` are all in `DEPRECATED_KV_MODES`, so today they are
  reachable only behind `HIPFIRE_KV_ALLOW_DEPRECATED=1`.
- `attention_flash_f16k_q8v_batched_masked` has **no production caller** — only
  `parity_kvarn_flash.rs` and `parity_kvarn_fused_flash.rs`. Those are parity
  harnesses, so at `head_dim=128` they would have reported a divergence caused by
  the harness's own kernel.
- The shipping KV family (kvarn) uses `attention_flash_kvarn_tile_batched`, which
  is correct.

So a default-configured server does not hit it. A `head_dim=128` model with a
deprecated KV mode and batched prefill does.

## Independent confirmation, and a validated fix, in a lineage we do not merge

`41d597e14` on the **disconnected pre-fork lineage** (it edits
`crates/rdna-compute/`, which does not exist in this tree, and is NOT an ancestor
of `origin/master`) hit this exact defect from the other end:

> Fix the head_dim=128 GPU fault. The batched flash tile + asym reduce kernels
> were hardcoded for head_dim=256 (d0 = tid*8, 8 dims/thread x a 32-lane block =
> 256 dims, ignoring the head_dim arg), so North-Mini-Code (head_dim 128) drove
> threads 16..31 out of bounds -> HIP 700 illegal memory access that wedged the
> stream (presented as a ~27-min "hang"). Parameterize dims-per-thread =
> head_dim/32 (4@128, 8@256); dpt=8 is byte-identical at head_dim=256, so Qwen is
> unaffected.

Three things this tells us, none of which change what we merge:

1. **The symptom is a fault, not just wrong numbers.** The Q read past the last
   head's slice leaves the buffer, so the first observable is HIP 700 wedging the
   stream. This document originally framed it as silent corruption; the partials
   overrun is silent, but the Q read is not.
2. **The diagnosis is confirmed independently**, by someone who reproduced it on
   hardware rather than by reading the source.
3. **The fix scoped below is the one that was measured** — `dpt = head_dim / 32`,
   byte-identical at 256.

Per AGENTS.md `upstream` is disconnected and is not fetched, rebased onto, or
merged. This is cited as **evidence and as a reference implementation**, not as a
change to take. On this tree only the reduce kernel and the kvarn tile carry the
runtime `dpt`; the four tiles here never received it, which is why the reduce
kernel's comment claims a parity that does not exist.

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
