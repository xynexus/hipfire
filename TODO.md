# TODO

## FWHT Residual QJL Transform

- Implement a Johnson-Lindenstrauss / QJL transformation on the residual in the FWHT path. The current FWHT path applies a signed-FWHT rotation to Q/K for attention and leaves the residual stream without a separate QJL transform.

## Quantization Throughput

- Parallelize full-model quantization at the tensor/shard pipeline level. The current path can use Rayon inside tensor work, but large models still walk and emit tens of thousands of tensor records mostly serially. Add a bounded worker pipeline that overlaps safetensor decode, FP8 scale decode, MQ/HF quantization, and HFQ record assembly while preserving deterministic tensor ordering and bounded memory.
- Fix quantization error reporting for MQ-family formats. The current summary can print `Mean quant error: 0.00000000` / `Max quant error: 0.00000000` for MQ4 DeepSeek V4 runs because detailed error accounting only measures selected HFQ paths while still dividing by all quantized params. Either compute MQ/MQ6/MQ-Lloyd dequant error correctly, including inverse FWHT where needed, or report `n/a` with an explicit `not measured for <format>` reason.
