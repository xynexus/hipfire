---
title: "Moving the Current Read/Write Position Forward"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Moving-the-Current-Read/Write-Position-Forward"
toc_id: Lqvs7YaGExzRKjmlnrVkDg
content_id: G1vv2c6dDynexPjBHaOP6g
---

### Moving the Current Read/Write Position Forward

In the following description, `input_buffer`<TYPE> stands for any of the allowed input buffer port data types. Likewise, `output_buffer`<TYPE> stands for any of the allowed output buffer port data types.

| Purpose | Input Buffer Port Type | Output Buffer Port Type |
| --- | --- | --- |
| To advance the current read/write position | Option 1, Using an iterator:void simple(input_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pIn = aie::begin(in); TYPE data = *pIn++; ... Option 2, Using the data() API:void simple_1(input_buffer<TYPE> & in, output_buffer<TYPE> & out) { const TYPE* pIn=in.data(); TYPE data = pIn[index++]; ... | Option 1, Using an iterator:void simple(input_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pOut= aie::begin(out); TYPE data; *pOut++ = data; ... Option 2, Using the data() API:void simple_1(input_buffer<TYPE> & in, output_buffer<TYPE> & out) { const TYPE* pOut =(TYPE*)out.data(); pOut[index++] = data; ... |
| To advance the current read/write position by four times the underlying buffer port type. | #define VECTOR_SIZE 4 void simple(input_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pIn = aie::begin_vector<VECTOR_SIZE>(in); v4TYPE data = *pIn++; ... | #define VECTOR_SIZE 4 void simple(input_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pOut = aie::begin_vector<VECTOR_SIZE>(out); v4TYPE data; *pOut++ = data; ... |
| To advance the current read/write position by eight times the underlying buffer port type. | #define VECTOR_SIZE 8 void simple(input_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pIn = aie::begin_vector<VECTOR_SIZE>(in); v8TYPE data = *pIn++; ... | #define VECTOR_SIZE 8 void simple(input_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pOut = aie::begin_vector<VECTOR_SIZE>(out); v8TYPE data; *pOut++ = data; ... |
