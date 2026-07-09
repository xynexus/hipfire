---
title: "Buffer Port Connection for Multirate Processing"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Buffer-Port-Connection-for-Multirate-Processing"
toc_id: aZjw1cz0SQctekxCIxBOEg
content_id: 0tQOk_D6IZ4tTCvRdDOUBQ
---

### Buffer Port Connection for Multirate Processing

In some designs, a kernel’s output buffer port can differ in size from the input buffer port of the next kernel. In that case, the connection declaration contains the size of the output port and the size of the input buffer port. See the following example:

```
connect net0 (kernel0.out[0], kernel1.in[0]);
dimensions(kernel0.out[0]) = {128};
dimensions(kernel1.in[0]) = {192};
```

In the above example, `kernel0` writes 128 samples and `kernel1` expects that 192 samples are written to memory. In such scenarios, the AI Engine compiler performs multirate analysis. In this case, the compiler specifies that `kernel1` runs twice while`kernel0` runs three times. You can specify the repetition count for these kernels in the graph manually, as follows:

```
repetition_count(kernel0) = 3;
repetition_count(kernel1) = 2;
```

You can multicast an output buffer port to multiple input buffer ports in the graph (automatic DMA insertion mechanism). You can also perform multirate processing in this specific use case:

```
connect net0 ( kernel0.out[0] , kernel1.in[0] );
connect net1 ( kernel0.out[0] , kernel2.in[0] );
dimensions(kernel0.out[0]) = {128};
dimensions(kernel1.in[0])  = {64};
dimensions(kernel2.in[0])  = {192};
```

In this example, the AI Engine compiler automatically detects the following:

- `kernel0` runs three times
- `kernel1` runs six times
- `kernel2` should run twice

for one graph iteration `graph.run(1)`. You can also specify these repetition counts manually in the graph.

```
connect ( datain , upconv.in[0] );
connect ( upconv.out[0] , singlerate.in[0] );
connect ( singlerate.out[0] , dataout );

// If the connection is buffer based dimensions can be specified
// in the graph
dimensions(upconv.in[0]) = {350};
dimensions(upconv.out[0])  = {490};
dimensions(singlerate.in[0]) = {350};
dimensions(singlerate.out[0])  = {350};

// If the connections are stream based, repetitions count must be specified
repetition_count(upconv) = 5;      // LCM(350,490)/490 = 5
repetition_count(singlerate) = 7;  // LCM(350,490)/350 = 7
```

![lnz1691069024856.png](../assets/091-01-lnz1691069024856-png-d91f11bf34af.png)

*Figure 1. Up-Converter Followed by a Single Rate Kernel*

In the preceding figure, the up-conversion kernel must be run five times. The single rate kernel must be run seven times, meaning the same number of samples are produced and consumed between the two kernels.

In this example the output frame length of the up-converter is larger than the length of the input frame of the single rate kernel. This means that the up-converter writes over the ping and pong buffers (or other more complex schemes) in a single iteration.
