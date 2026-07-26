# R62 — producer-native W4 activation control

R62 is an exact A/B against R61. It keeps the R34 QKV/O/parameter bundle,
eight-column/four-row raw-MLIR topology, complete QKV compute, direct padded
row-major output, and CPU oracle unchanged. Only the mutable activation view
changes: the producer supplies the existing whole-scaled W4 8-KiB block layout,
so the AIE cores perform no R34 W8-to-W4 compatibility swizzle.

This is a control for selecting the next shared-input ABI. It does not reorder
immutable weight blocks and does not claim that the current upstream producer
already emits this view.

## Result

Producer-native W4 input retained exact parity but only moved the full
row-major projection from R61's 4.114-ms median to 3.850 ms. Physical output,
QKV-only bundle traversal, and the exact R15 dynamic group loop measured
3.856 ms (3.835-4.221 ms), again with all 327,680 real outputs passing at
`max_abs=3.8147e-6`.

This falsifies the R61 hypothesis that legacy activation compatibility was the
dominant cost. The remaining discrepancy was a measurement-system mismatch:
these are cold, one-command Python `XRTHostRuntime` `npu_time` values, whereas
the sub-millisecond control was a warmed production Rust/C++ wrapper value.
R63 makes the graph and immutable QKV argument identical and compares both
through the production wrapper.

Run under `hipfire lock`:

```bash
export PYTHONPATH="/opt/xilinx/xrt/python"
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:/opt/rocm/lib"
export R62_R34_BUNDLE_HFP="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.r34-bundled-bo.rdna2.hfp"
./build_r62.sh
python run_matrix.py
```
