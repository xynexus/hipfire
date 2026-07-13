# R60 — first compute from the R34 bundle and shared input

R60 replaces R59's first QKV feed guard with one exact signed W4A8
`mmul<4,16,16>` on every AIE2P column, then extends that exact mapping to a
complete K=256 reduction for one Opus group. It consumes:

- the admitted R34 `ResidentContextBundleV1` containing QKV, output projection,
  and immutable parameters; and
- the existing 2,949,120-byte R34 shared activation argument without changing
  that argument's layout.

The kernel joins two adjacent 8-wide K fragments for four rows from the first
R34 activation block, loads the first row-major 16x16 OQ4 weight tile directly
from the bundle, decodes signed nibbles in the AIE MMUL operand path, and checks
all 512 produced int32 values against a CPU oracle. `FULL_K_GROUP_STAGE1`
repeats the check after all sixteen K=16 MMULs. The remaining bundle tiles
continue through the same four-row fanout used by R59 so each step measures
whether the compute ratchet preserves the resident-weight DMA boundary.

`THREE_GROUP_SCALED_STAGE2` then consumes the first three existing R34
activation blocks and the first three QKV bundle blocks, performs all 48 MMULs,
and applies the real f32 activation and OQ4 weight-scale tails. Its 512 f32
outputs are checked with the production projection tolerance.

This is intentionally not a complete projection or attention implementation.
It proves the exact shared-input/bundle compatibility and first swizzle/MMUL
only. No kernel performs global tensor-block reordering.

## Hardware evidence

Three locked AIE2P trials per accumulated stage produced:

| stage | median bundle GB/s | range | retention vs R59 R34 bundle | oracle |
|---|---:|---:|---:|---|
| `FIRST_MMUL` | 53.838 | 53.829-54.112 | 96.34% | exact int32 |
| `FULL_K_GROUP_STAGE1` | 54.252 | 53.870-54.635 | 97.08% | exact int32 |
| `THREE_GROUP_SCALED_STAGE2` | 50.912 | 50.605-51.092 | 91.10% | f32 tolerance |

Every stage clears the bandwidth-first 85% gate. Stage 2 consumes three R34
activation blocks, performs 48 real MMULs, and uses the real HFP scale tails;
its median receive-stall fraction is 9.9%, which explains the remaining
consumer-side backpressure. Durable rows are in
`../results/r60-first-shared-input-mmul-20260713.csv`.

Two failures were retained as design evidence. Tracing DMA channel 0 alone
measured the short activation stream and produced an impossible 683-GB/s value;
the admitted harness traces both channels and selects the long bundle stream on
channel 1. A depth-3 activation FIFO did not fit beside the weight double buffer
in tile SRAM; sequential depth-1 activation consumption compiles and passes.

Run under `hipfire lock` from this directory:

```bash
export PYTHONPATH="/opt/xilinx/xrt/python"
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:/opt/rocm/lib"
export MLIR_AIE_INC="$HOME/.venv/lib/python3.14/site-packages/mlir_aie/include"
export R60_R34_BUNDLE_HFP="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.r34-bundled-bo.rdna2.hfp"
python run_matrix.py
```
