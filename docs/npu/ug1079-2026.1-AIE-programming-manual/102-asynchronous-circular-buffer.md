---
title: "Asynchronous Circular Buffer"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Asynchronous-Circular-Buffer"
toc_id: BgytBC8_RcKbS1Uh8iNR6w
content_id: tMG1joMcrnhJ32ooSZJZjw
---

#### Asynchronous Circular Buffer

Declare kernel function `k3` with asynchronous circular input one dimension buffer that operates on data type `int32` with buffer size specified in graph and margin size `MARGIN_SIZE` that is named `in0` and output stream that operates on data type `int32` that is named `out0`.

```
void k3(input_async_circular_buffer<int32, adf::extents<adf::inherited_extent>, adf::margin<MARGIN_SIZE>> &in0, output_stream<int32> *out0)
{
    in0.acquire();
    auto in0Iter = aie::begin_circular(in0);
    for (int ind = 0; ind < INPUT_SAMPLE_SIZE + MARGIN_SIZE; ++ind)
    {
        writeincr(out0, *in0Iter++);
    }

    in0.release();
}
```
