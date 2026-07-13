# R59 — destination-context resident HFP argument ABI

R59 is the first resident-integration ratchet after R58. It deliberately does
not implement attention or FFN compute. It establishes how already-converted
HFP weights can enter the completed R34/R35 contexts without sacrificing the
external feed roof or reordering tensor blocks in a kernel.

The four real version-2 `.rdna2.hfp` inputs are:

- combined QKV, K=768/N=1280;
- attention output, K=768/N=768;
- combined gate/up, K=768/N=2304;
- down, padded K=1280/N=768.

## ABI result

The amdxdna DPU regmap holds at most five data arguments. A synthetic combined
layer request with four weight BOs, a parameter BO, and an output BO exceeded
that limit and caused XRT to reference a missing BO. The real execution graph
uses separate R34 and R35 contexts, but keeping every role separate would still
exceed five arguments once their existing shared I/O BOs are counted.

The admitted production shape is therefore one offline bundle per destination
context:

- R34: QKV + O + immutable attention/norm parameters in one HFP BO;
- R35: gate/up + down + immutable FFN/FWHT/AWQ parameters in one HFP BO.

The existing R34 input/hidden/query/key-value arguments then total five with
the bundle. R35 remains below five with its input/scratch/output arguments.
Role-local HFP block order is unchanged. Only immutable role segments are
placed column-major beside one padded parameter tile by the loader/on-disk
conversion.

`OpusHfpLayout::ResidentContextBundleV1` and
`NpuOpusExecutor::prepack_resident_context_bundle_cached` implement this
generic source/payload-SHA-validated offline container. The header records up
to three source segment lengths plus the logical parameter length.

## Hardware evidence

Three locked AIE2P trials per mode used four-row fanout and the same 1.8-GHz
trace clock as R58:

| mode | median wire GB/s | range | median packed-data GB/s | guards |
|---|---:|---:|---:|---|
| R34 QKV + O + params, separate BO control | 56.121 | 56.012-56.227 | 42.042 | pass |
| R35 gate/up + down + params, separate BO control | 56.053 | 55.941-56.114 | 42.009 | pass |
| R34 bundled weight/parameter BO | 55.884 | 55.789-55.898 | 40.416 | pass |
| R35 bundled weight/parameter BO | 56.018 | 55.922-56.062 | 41.037 | pass |

Both production bundles retain at least 99.48% of R58's 56.173-GB/s feed-only
median and have zero traced receive stalls. Per-column final-tile guards pass for every role;
the parameter kernel checks all 4 KiB. This proves the destination-context
weight ABI and DMA boundary only. It is not resident projection computation,
a complete layer, or model throughput.

The first attempts traced all eight columns and failed AIE packet routing;
using the established four-column trace configuration compiled correctly. A
read-modify-write guard also failed to retain accumulation across helper calls,
so R59 follows R58's final-tile guard convention rather than overstating a
whole-stream checksum.

Durable rows:
`../results/r59-resident-weight-abi-20260713.csv`.

## Run

Run from this directory under `hipfire lock`:

```bash
export PYTHONPATH="/opt/xilinx/xrt/python"
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:/opt/rocm/lib"
export MLIR_AIE_INC="$HOME/.venv/lib/python3.14/site-packages/mlir_aie/include"
export R59_QKV_HFP="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.qkv.oq4.whole-scaled.rdna2.hfp"
export R59_O_HFP="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.o-proj.oq4.whole-scaled.rdna2.hfp"
export R59_GATE_UP_HFP="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.gate-up.oq4.whole-scaled.rdna2.hfp"
export R59_DOWN_HFP="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.down-proj.oq4.whole-scaled.rdna2.hfp"
python run_matrix.py
```

R60 must replace the R34/R35 feed guards with the admitted nibble decode and
projection compute while retaining the current shared activation/output ABI.
