---
title: "Buffer Port-Based Access"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Buffer-Port-Based-Access"
toc_id: dtO69NBEHsxcwDxFOHlwkw
content_id: cTyCPXCiSffVbhnGKfQHrA
---

## Buffer Port-Based Access

Buffer ports provide a way for a kernel to operate on a block of data. Buffer ports operate in a single direction (for example, input or output). The view that a kernel has of incoming blocks of data is called an input buffer. Input buffers are defined by a type. The type of data contained within that buffer needs to be declared before the kernel can operate on it.

The view that a kernel has of outgoing blocks of data is called an output buffer. These are defined by a type. The example below shows a declaration of a kernel named `simple`. The `simple` kernel has an input buffer named `in`, which contains complex integers where both the real and the imaginary parts are each 16-bit wide signed integers. The `simple` kernel also has an output buffer named `out`, which contains 32-bit wide signed integers.

```
void simple(input_buffer<cint16> & in,
           output_buffer<int32> &out);
```

The example below shows input and output buffer port sizes declaration using the adf::extents template parameter.

```
void simple(input_buffer<cint16, adf::extents<INPUT_SAMPLE_SIZE>> & in,
           output_buffer<int32, adf::extents<OUTPUT_SAMPLE_SIZE>> & out);
```

These buffer data structures are inferred by the AI Engine compiler from the data flow graph connections and are declared in the wrapper code implementing the graph control. The kernel functions merely operate on pointers to the buffer data structures that are passed to them as arguments. There is no need to declare these buffer data structures in the data flow graph or kernel program.

When two kernels (k1, k2) communicate through buffers (the output buffer of k1 is connected to the input buffer of k2) the compiler attempts to place them into tiles that can share at least an AI Engine memory module.

- If the two kernels are located on the same tile, the compiler uses a single memory area to communicate because they are not executed simultaneously (see k1 and k2 in tile (8,0) and the single shared memory block in (7,0) in the following figure). Because the execution of multiple kernels within an AI Engine is sequential, access conflicts are avoided when using the same memory area.
- If the two kernels are placed in different tiles sharing an AI Engine memory module, the compiler will infer a ping-pong buffer, allowing the two kernels to write and read at the same time but not to the same memory area (see k1 in tile (10,0), k2 in tile (11,0) and the shared buffer implemented as a ping-pong buffer in (10,0) in the following figure).Figure 1.Same Tile and Memory Sharing Placement Example ![vyo1669822158240.png](../assets/085-01-vyo1669822158240-png-01e7724cf9d2.png)
- If system performance allows, you can enable single buffering by applying the `single_buffer(<port>)` constraint to the kernel ports.
- If the two kernels are placed in distant tiles, the compiler automatically infers a ping-pong buffer at the output of k1, and another one at the input k2. A DMA connects the two ping-pong buffers. The DMA automatically copies the content of the output buffer of k1 onto the input buffer of k2 using a data stream.Figure 2.Distant Tiles Placement Example ![ily1669822708397.png](../assets/085-02-ily1669822708397-png-c6a68ae7e323.png)
- When multiple buffers/streams converge onto a kernel, the various paths can have very different latencies, which can potentially lead to a deadlock. To avoid this type of problem, you can insert a FIFO between the two kernels. The compiler generates the same architecture as for distant tiles, but inserts a FIFO between the stream connection endpoints.
