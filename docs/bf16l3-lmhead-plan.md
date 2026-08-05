# Wiring the BF16L3 lm_head — DONE

**Landed.** This file was written as a plan; everything under "Left to do" is
now implemented and measured. Kept because the "Established" notes below are
still the quickest orientation, and because the `K % 256` constraint and the
verification caveat still apply to anyone touching this.

Result, on an artifact with quantized layers and a LUT3 embedding
(`oq4` + `--bf16-codec lut3`), head 379.84 MB packed against 525.47 MB logical:

| | pp512 | tg128 |
|---|---|---|
| `HIPFIRE_BF16L3_RESIDENT` off | 97.43 | 90.74 |
| **on — packed head** | **111.17** | **102.53** |
| | +14.1% | **+13.0%** |

Generation byte-identical between the two.

Cumulative over the whole chain on this path — F32 tied head -> BF16 tied head
-> BF16L3 packed head — **76.07 -> 102.53 tg128, +34.8%**, output unchanged at
every step. llama3.2:1b per-token traffic 1545.5 -> 871.1 MB, i.e. 2.00x FLM's
772.3 MB down to **1.13x**.

`gemv_bf16l3_xf32` itself: 1.917 ms against `gemv_bf16_xf32`'s 3.241 at
128256 x 2048, worst |diff| 2.570e-7, argmax identical.

**Opt-in — a default was tried and reverted.** `HIPFIRE_BF16L3_RESIDENT` must be
set. Defaulting it on for head-shaped tensors took the tiny-quant gate from 8
failures to **58**: `expand_bf16_index` is arch-agnostic, but only the LLaMA
loader was taught to decode a packed embed for the gather, so qwen35, qwen2 and
dots-ocr all panicked with `expected F16/F32/BF16 for embed_tokens.weight, got
qt=49`.

**Prerequisite for the default:** teach every arch loader to decode a packed
tensor. The predicate is already present as `is_head_tensor`; the work is the
per-arch decode arms.

That tail is longer than it looks. Measured by forcing
`HIPFIRE_BF16L3_RESIDENT=1` and running the gate, which is stricter than the
head-only default and so bounds the work:

| state | gate failures |
|---|---|
| residency off (baseline) | 8 |
| forced on, before any arch fix | 58 |
| forced on, after teaching qwen35 | **43** |

One arch bought 15 cells. The remainder are spread across `gemma4`
("unsupported embedding quant type 49"), `zaya` ("zaya gpu: unsupported
quant_type 49") and others, each with its own embedding loader and its own
panic string — there is no shared seam to fix once. `hipfire-runtime::hfq::
decode_bf16_packed` is now `pub` so each arch can use the same decoder, which is
the only part that is shared.

Until every arch is taught, the flag stays opt-in. Enabling it on an untaught
arch fails to load rather than degrading, which is the right failure but not one
to inflict by default.

### Reach

**New artifacts get it for free.** The quantizer's default codec is `huff`, and
`is_gather_shaped` steers gather-shaped tensors to LUT3 anyway, so a stock
`hipfire-quantize` run with no `--bf16-codec` flag produces a `Bf16Lut3`
embedding. Verified on a fresh `oq4` build: `model.embed_tokens.weight`
`Bf16Lut3` 379.74 MB from 525.34 MB.

**Existing artifacts mostly do not.** Only 1 of the 43 registered locally has a
LUT3 head — `Llama-3.2-1B-Instruct-lut3--oq4++`, the one built for it
deliberately. The rest carry Huffman or uncompressed embeddings and are
untouched, so the change is forward-looking with a small blast radius on disk.

On that one artifact:

| | pp512 | tg128 |
|---|---|---|
| default (packed) | 110.10 | **101.45** |
| `HIPFIRE_BF16L3_RESIDENT=0` | 96.70 | 90.05 |

Output byte-identical.

## Established

**The artifact side works.** `hipfire-quantize --input <bf16.hfq> --output
<out.hfq> --format bf16 --bf16-codec lut3` produces a 1797 MB file against a
2.47 GB logical size — 1.37x, LUT3's ratio. Note `hipfire inspect` reports the
tensor as plain `BF16`: `expand_bf16_index` rewrites recoded index entries to
their logical view, so the histogram shows what the tensor *is*, not how it is
stored. File size is the tell.

**The residency mechanism already exists.** `hfq.rs:906` — with
`HIPFIRE_BF16L3_RESIDENT` set, a `Bf16Lut3` tensor is skipped by
`expand_bf16_index`, so its index entry keeps the physical extent and
`quant_type` stays 49. The loader then sees packed bytes without any new
accessor.

**The head has no gather consumer.** `token_embd` (gather) and `output` (GEMV)
are separate GPU buffers, loaded independently. So the head can stay packed
while the embedding table is expanded for lookup — which is what the
`HIPFIRE_BF16L3_RESIDENT` doc means by "a gather-read table ... will fail to
load". That caveat constrains the EMBEDDING, not the head.

## What it took (all landed)

1. **`DType::Bf16L3`.** Cascaded to exactly two exhaustive matches, both of
   which deserved the decision rather than a mechanical fill-in: `DType::size()`
   — BF16L3 is planar with a VARIABLE escape plane, so it has no per-element
   stride and is byte-level like `Raw`; anything computing `n * size()` for it
   is wrong — and `dtype_arch_predicate`, where it is `Always`.
   `dtype_rotation_plan` needed nothing: its `_ => RotationPlan::None` is
   correct here.
2. **Dispatch entry**, mirroring `GemvBf16`: `KernelKey::GemvBf16L3`,
   `(Bf16L3, Plain)` in `for_gemv`, a `launch` arm, and the dtype in
   `register_plain`.
3. **Three loader sites, not one** — each found by it panicking, which is the
   good failure mode:
   * the tied-head branch, which also guards `K % 256 == 0` and decodes
     explicitly when it does not hold;
   * `token_embd`, which must DECODE — the gather reads one arbitrary row and
     the escape plane is only addressable by walking a block. This is what
     `HIPFIRE_BF16L3_RESIDENT`'s "a gather-read table will fail to load" means,
     and it constrains only the gather;
   * `load_f16_tensor` and `load_weight_tensor`, because residency is global:
     norms and layer weights stay packed too. Layer weights are decoded
     deliberately — `gemv_bf16l3_xf32` is batch-1 and there is no BF16L3 GEMM,
     so a packed layer weight would work at decode and break at prefill.

The head is the only tensor that stays packed, because it is the only pure-GEMV
consumer of a large matrix.

## Constraint worth checking first

`gemv_bf16l3_xf32` asserts `K % 256 == 0`. hidden 2048 passes, but a model with
a hidden size that is not a multiple of 256 must fall back to the BF16 path
rather than fail to load — the loader branch should test this, not assume it.

## Payoff

llama3.2:1b per-token traffic 1020.1 -> 871.1 MB, i.e. 1.32x FLM's 772.3 MB
down to **1.13x**, losslessly. Head time 3.24 -> 1.92 ms.

## Verification note

Anything reached through `chat`/`bench` runs in the **`hipfire-daemon` binary**.
`cargo build -p hipfire-runtime` and `--bin hipfire` both leave it stale, and a
stale daemon reports a clean "no difference" — see
`tied-lmhead-f32-expansion.md`. Build the workspace and confirm a diagnostic
fires before trusting a number.
