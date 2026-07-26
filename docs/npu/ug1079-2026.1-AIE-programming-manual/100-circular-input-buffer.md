---
title: "Circular Input Buffer"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Circular-Input-Buffer"
toc_id: Rmu6m3mE45kgTIJIj4kM~Q
content_id: p5ulcdk_xkeP8wIIas9NZw
---

#### Circular Input Buffer

The following example declares the kernel function `k2`. The example uses an input circular buffer named `in0` that operates on the `int32` data type. th example also defines an output stream named `out0`, which operates on the `int32` data type.

```
void k2(input_circular_buffer<int32, adf::extents<INPUT_SAMPLE_SIZE>, adf::margin<MARGIN_SIZE>> & in0, output_stream<int32> *out0)
{
    auto in0Iter = aie::begin_circular(in0);
    for (int ind = 0; ind < (INPUT_SAMPLE_SIZE + MARGIN_SIZE); ++ind)
    {
        writeincr(out0, *in0Iter++);
    }
}
```
