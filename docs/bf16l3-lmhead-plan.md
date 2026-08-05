# Wiring the BF16L3 lm_head — what is done, what is left

`gemv_bf16l3_xf32` is committed and verified (1.917 ms against
`gemv_bf16_xf32`'s 3.241 at 128256 x 2048, worst |diff| 2.570e-7, argmax
identical). Nothing dispatches to it yet. This is the remaining plumbing, with
the parts already established.

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

## Left to do

1. **`DType::Bf16L3`** in `hipfire-gpu-types`. There is no variant today. Byte
   length is not a simple stride, so anything computing sizes from the dtype
   needs checking.
2. **Dispatch entry**, mirroring the `GemvBf16` work exactly: `KernelKey`
   variant, `(Bf16L3, Plain)` in `for_gemv`, a `launch` arm calling
   `gemv_bf16l3_xf32`, and the dtype in `register_plain`. `dtype_arch_predicate`
   and `dtype_rotation_plan` both need a case.
3. **Loader branch.** In the tied-head `else` of `load_weights_hfq`, alongside
   the existing `quant_type == 16` case: when `quant_type == 49`, upload `data`
   as-is and set `gpu_dtype: DType::Bf16L3`. The bf16 case is the template.
4. **Preflight.** `gemv_dtype_supported` must accept it, and the
   `preflight_dtype_contract` test list needs the variant.

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
