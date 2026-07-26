---
title: "Circular Output Buffer"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Circular-Output-Buffer"
toc_id: DWXRziSs4yTJOfuRZc7U8Q
content_id: 3v2Lkeh~ykQA2Vr57TE6Dg
---

#### Circular Output Buffer

Declare kernel function `k1` with input stream operates on data type `int32` that is named `in0` and circular `1d` output buffer operates on data type `int32` that is named `out0`.

```
void k1(input_stream<int32> *in0, output_circular_buffer<int32, adf::extents<OUTPUT_SAMPLE_SIZE>> & out0)
{
    auto out0Iter = aie::begin_circular(out0);
    for (int ind = 0; ind < OUTPUT_SAMPLE_SIZE; ++ind)
    {
        *out0Iter++ = readincr(in0);
    }
}
```

**Note:** Important:

AI Engine

This limitation is not applicable to x86 simulation because it only emulates functional aspects of AI Engine tiles and memory. For more details on x86 simulation models, see [Limitations](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment&resourceid=dbf1620212200427.html) in AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)).
