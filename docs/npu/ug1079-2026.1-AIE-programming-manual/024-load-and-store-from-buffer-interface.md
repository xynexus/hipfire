---
title: "Load and Store From Buffer Interface"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Load-and-Store-From-Buffer-Interface"
toc_id: heGRj6UXljIl5mk0JDlXYA
content_id: dwnFWccpSWipP_mYClBKMA
---

#### Load and Store From Buffer Interface

AI Engine APIs provide methods to read from and write to data memory, streaming data ports, and cascade streaming ports used by AI Engine kernels. For additional details on stream APIs, see Streaming Data API. For additional details on buffers, see Input and Output Buffers.

The following example uses the `const` circular iterator to read an `aie::vector<cint16,8>` vector. Similarly, `readincr_v<8>(cin)` is used to read a sample of int16 data from the `cin` stream. `writeincr(cas_out, v)` is used to write data to a cascade stream output.

```
void func(input_buffer<cint16> &din,
          input_stream<int16> *cin,
          output_cascade<cacc48> *cas_out) {
  auto cirIter=aie::cbegin_vector_circular<8,512>(din.data());
  aie::vector<cint16,8> data=*cirIter++;
  aie::vector<int16,8> coef=readincr_v<8>(cin);
  aie::accum<cacc48,4> v;
  …
  writeincr(cas_out, v);
}
```
