# R18 resident gate/up + GeGLU

R18 fuses the complete scaled `K=768, N=2304` gate/up projection and
EmbeddingGemma GeGLU in one AIE2P program. Weight columns are interleaved by
the output stripe, so every core owns matching gate/up values and applies the
nonlinearity before releasing its tile. The full gate/up intermediate never
leaves the array.

- W4 emits 48 logical GeGLU columns per core stripe.
- W8 emits 24 logical columns in a 32-column physical stripe. The eight padded
  lanes keep vector stores and shim DMA 16-lane aligned.
- W8's nonlinear helpers deliberately use pointer-based `noinline` boundaries;
  passing the live vectors through an inlined helper corrupts registers with
  the scaled W8 MMUL code in the same core program.

Build artifacts stay under `~/.hipfire/npu`:

```sh
bash benchmarks/npu_gemm_tuning/r18/r18_cache.sh w4
bash benchmarks/npu_gemm_tuning/r18/r18_cache.sh w8
```

The hardware verifier also runs the production R16 scaled projection as an
independent CPU-oracle-checked source for the expected gate/up values. Its
reported `fused_ms` times only resident R18 dispatches after input and weights
are uploaded.
