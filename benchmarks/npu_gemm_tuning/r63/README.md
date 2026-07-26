# R63 — compact offline QKV weight ABI

R63 isolates the resident weight argument size from R62. It keeps the same
producer-native W4 activation tiles, R15 compute functions, dynamic three-group
core loop, physical output stream, M256/K768/N1280 geometry, and complete CPU
oracle. The only change is the immutable weight argument:

- R62 uploads the 3.5 MiB layer bundle and addresses the first 18 of 28 blocks
  in each column.
- R63 uploads the existing 2.25 MiB QKV-only
  `.qkv.oq4.whole-scaled.rdna2.hfp`, with 18 contiguous blocks per column.

The global tensor-block conversion is performed once by the loader/HFP writer,
never by the kernel. The kernel still performs its local signed-nibble decode
and lane swizzle.

Run under the repository hardware lock:

```bash
export PYTHONPATH=/opt/xilinx/xrt/python
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/opt/rocm/lib
export R63_QKV_HFP="$HOME/.hipfire/npu/prepacked/EmbeddingGemma-300M.npu.oq4.layer-0.qkv.oq4.whole-scaled.rdna2.hfp"
./build_r63.sh
python run_matrix.py
```

`build_r63_oldspill.sh` builds the same graph with the pre-`3db7a1497`
accumulator-spill kernel for a controlled binary-revision A/B. Set
`R63_CACHE_DIR`, `R63_KERNEL_VARIANT`, and `R63_RESULT` when recording that
variant so its result does not overwrite the current-kernel row.

`run_wrapper_matrix.py` compares both rebuilt variants and the historical R15
cache through the production Rust/C++ executor. Its warm wrapper result is the
projection timing to use for integration decisions. The cold Python raw-runtime
timing is retained only to validate the complete physical output and must not be
compared directly with the warm production wrapper.

## Result

The complete R63 MLIR is byte-identical to canonical R15 after normalizing the
linked object name. Three fresh production-wrapper processes per variant, each
with two warmups and three timed iterations, produced:

| kernel binary | median wrapper ms | range | logical TOPS at median time |
|---|---:|---:|---:|
| current no-spill | 1.0292 | 1.0011-1.1301 | 0.4890 |
| pre-`3db7a1497` spill | 1.0288 | 1.0059-1.0996 | 0.4892 |
| historical cache | 1.0596 | 0.8912-1.0993 | 0.4750 |

Every run passed all 327,680 real outputs with `max_abs=2e-7`. The current and
old kernel revisions are indistinguishable at this sample size; the no-spill
change is not a 3.7x regression. The final current-kernel cold Python raw
median is 3.964 ms (3.628-3.981 ms); it includes first-command/runtime overhead
and is diagnostic only.

The admitted integration candidate is therefore the current compact QKV
`.rdna2.hfp` plus current whole-scaled production executor. The resident layer
must reuse this cache/BO and its mutable W4 activation contract without an
intermediate global tensor reorder.
