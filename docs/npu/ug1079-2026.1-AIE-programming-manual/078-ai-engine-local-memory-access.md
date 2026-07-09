---
title: "AI Engine Local Memory Access"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-Local-Memory-Access"
toc_id: OXjPtjGQTJ8fK1MzBD9avg
content_id: 20l0EhQgAA5k3dLFaeyugA
---

## AI Engine Local Memory Access

The buffers used in the various kernels are automatically inferred at the graph level as ping-pong buffers located in the AI Engine data memory. The kernel accesses and manages buffers through the address generator units of the processor.

Other components of the device can fill and flush input and output buffers using streams. A DMA handles these streams internally. In order to simplify kernel addressing, you might require the data to be stored in a specific way in the data memory. For instance, a tensor transposition might be needed to simplify matrix multiplication at the kernel level. You can define specific access patterns for this type of memory. Defining tiling parameters enables flexible addressing for local memories.

The following code snippet demonstrates the API using the AI Engine data memory to perform a matrix multiplication where matrix B has been transposed:

```
class MatrixMultiply : public graph {
public:
input_port inA,inB;
output_port outC;

kernel MatMult;

MatrixMultiply() {

	// kernel
	MatMult = kernel::create(ClassicMatMult);
	source(MatMult) = "src/matmult.cpp";
	runtime<ratio>(MatMult) = 0.9;

	// Connect Input A to MatMult Kernel
	connect(inA, MatMult.in[0]);
	dimensions(MatMult.in[0]) = {NColsA,NRowsA};

	// Connect Input B to MatMult Kernel
	connect(inB, MatMult.in[1]);
	dimensions(MatMult.in[1]) = {NRowsB,NColsB};
	/* tiling parameters to transpose matrix */
        write_access(MatMult.in[1]) = adf::tiling(...);

	// Connect MatMult Kernel to  Output
	connect(MatMult.out[0],outC);
	dimensions(MatMult.out[0]) = {NColsC,NRowsC};
};
};
```

AI Engine

```
dimensions(MatMult.in[0]) = {NColsA,NRowsA};
```

You can set the size definition directly within the kernel.
