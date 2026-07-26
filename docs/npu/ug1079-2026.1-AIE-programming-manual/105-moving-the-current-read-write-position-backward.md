---
title: "Moving the Current Read/Write Position Backward"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Moving-the-Current-Read/Write-Position-Backward"
toc_id: ha40JuRRu9tSDItGFPeIUA
content_id: G9meyfJUQSwdU~ouNB_g~Q
---

### Moving the Current Read/Write Position Backward

In the following description, `input_buffer`<TYPE> and `input_circular_buffer`<TYPE> stands for any of the allowed input buffer port data types. Likewise, `output_buffer`<TYPE> and `output_circular_buffer`<TYPE> stands for any of the allowed output buffer port data types.

| Purpose | Input Buffer Port Type | Output Buffer Port Type |
| --- | --- | --- |
| To decrease the current read/write position. | void simple(input_circular_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pIn = aie::begin_random_circular(in); ... TYPE data = *pIn--; ... | void simple(input_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pOut = aie::begin_random_circular(out); TYPE data; ... *pOut-- = data; ... |
| To decrement the current read/write position by four times the underlying buffer port type. | #define VECTOR_SIZE 4 void simple(input_circular_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pIn = aie:: begin_vector_random_circular <VECTOR_SIZE>(in); ... v4TYPE data = *pIn--; ... | #define VECTOR_SIZE 4 void simple(input_buffer<TYPE> & in, output_circular_buffer<TYPE> & out) { auto pOut = aie:: begin_vector_random_circular <VECTOR_SIZE>(out); v4TYPE data; ... *pOut-- = data; ... |
| To decrement the current read/write position by eight times the underlying buffer port type. | #define VECTOR_SIZE 8 void simple(input_circular_buffer<TYPE> & in, output_buffer<TYPE> & out) { auto pIn = aie:: begin_vector_random_circular <VECTOR_SIZE>(in); ... v8TYPE data = *pIn--; ... | #define VECTOR_SIZE 8 void simple(input_buffer<TYPE> & in, output_circular_buffer<TYPE> & out) { auto pOut = aie:: begin_vector_random_circular <VECTOR_SIZE>(out); v8TYPE data; ... *pOut-- = data; ... |
