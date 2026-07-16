# R129: batched staged-full-K weight reuse

R129 extends R121's OQ8 `M256 K768 N1280` projection to multiple independent
M256 documents while retaining one immutable 7,987,200-byte weight payload.
Each core stages one 6,336-byte compact activation image per document, acquires
an N32 weight record once, applies it to every staged document, and only then
releases the record.

For B2, each core joins both documents into one 2,048-byte output object and
each shim stream joins four cores into one 8,192-byte object. That keeps the
runtime schedule at R121's 40 objects and one output task per stream instead of
requiring 80 objects and two tasks. Rust restores canonical document-major row
order after readback. B1 continues to use the byte-identical R121 MLIR and
canonical output ABI.

Build B2:

```bash
./benchmarks/npu_gemm_tuning/r129/build_r129.sh 2
```

The same generator has an explicit B4 form for the local-memory/capacity gate:

```bash
./benchmarks/npu_gemm_tuning/r129/build_r129.sh 4
```

Admission requires two distinct documents to match separate R121 M256 hardware
references, fresh and reused context stability, fixed weight traffic, and a
material row-throughput improvement. Compilation alone is not admission.

## Hardware result

B2 and B4 both compile and pass bit-exact parity against distinct, separately
recreated R121 hardware references. Doc0 is unchanged when the last document is
replaced, three fresh contexts pass, and twenty reused commands pass. The weight
payload remains 7,987,200 bytes and makes one DMA pass; activation/output bytes
scale from 1,179,648/2,621,440 at B2 to 2,359,296/5,242,880 at B4.

The gain is real but below the approximately 1.5x target. Three final B2 runs
measure 1.2532x, 1.2735x, and 1.2681x row throughput. Three final B4 runs have a
1.2564x median paired gain. A full tile-local shared-weight implementation
failed parity with eight live accumulators; a four-accumulator N16-sliced form
restored bit-exact output but measured only 1.2596x. The FIFO-reuse topology is
therefore the bounded best result, not evidence for B32/B128.

See `rejected-tile-local-reuse.md` and
`../results/r129-staged-fullk-batched-weight-reuse-20260716.csv`.
