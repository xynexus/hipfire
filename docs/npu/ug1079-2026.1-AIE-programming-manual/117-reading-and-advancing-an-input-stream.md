---
title: "Reading and Advancing an Input Stream"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Reading-and-Advancing-an-Input-Stream"
toc_id: j4XXG8f5OnC0OQxDSxA41w
content_id: NsDVPKIM4GqTo9IKp9zabw
---

### Reading and Advancing an Input Stream

##### AI Engine Operations

The following operations read data from the given input stream and advance the stream on the AI Engine. The AI Engine has two input stream ports. The compiler automatically assigns physical ports and includes the assignment in the stream data structure.

Data values from the stream can be read one at a time or as a vector. In the latter case, unless all values are present, the stream operation stalls. The data groupings are based on the underlying stream operations. These include single-cycle, 32-bit operations and four-cycle, 128-bit wide operations. The cascade connection reads all accumulator values in parallel.

```
//Scalar operations
//#include<aie_adf.hpp> or #include<adf.h>
int32 readincr(input_stream<int32> *w);
uint32 readincr(input_stream<uint32> *w);
cint16 readincr(input_stream<cint16> *w);
float readincr(input_stream<float> *w);
cfloat readincr(input_stream<cfloat> *w);

//AIE API Operations to read vector data which supports more vector lanes
//#include<aie_adf.hpp>
aie::vector<int8,16> readincr_v<16>(input_stream<int8> *w);
aie::vector<uint8,16> readincr_v<16>(input_stream<uint8> *w);
aie::vector<int16,8> readincr_v<8>(input_stream<int16> *w);
aie::vector<cint16,4> readincr_v<4>(input_stream<cint16> *w);
aie::vector<int32,4> readincr_v<4>(input_stream<int32> *w);
aie::vector<cint32,2> readincr_v<2>(input_stream<cint32> *w);
aie::vector<float,4> readincr_v<4>(input_stream<float> *w);

template<typename T,int N>
aie::accum<T,N> readincr_v<N>(input_cascade<T> *w);
template<typename T,int N>
aie::vector<T,N> readincr_v<N>(input_cascade<T> *w);


//ADF API Operations to read vector data which supports limited vector lanes
//#include<adf.h>
v8acc48 readincr_v8(input_cascade<acc48>* str);
v4acc80 readincr_v4(input_cascade<acc80>* str);
v8accfloat readincr_v8(input_cascade<accfloat>* str);
v4cacc48 readincr_v4(input_cascade<cacc48>* str);
v2cacc80 readincr_v2(input_cascade<cacc80>* str);
v4caccfloat readincr_v4(input_cascade<caccfloat>* str);
v32int8 readincr_v32(input_cascade<int8>* str);
v32uint8 readincr_v32(input_cascade<uint8>* str);
v16int16 readincr_v16(input_cascade<int16>* str);
v8cint16 readincr_v8(input_cascade<cint16>* str);
v8int32 readincr_v8(input_cascade<int32>* str);
v4cint32 readincr_v4(input_cascade<cint32>* str);
v8float readincr_v8(input_cascade<float>* str);
v4cfloat readincr_v4(input_cascade<cfloat>* str);
```

For the supported data types and lanes in AI Engine API `readincr_v<N>`, see Stream Data Types.

**Note:** `readincr`

`TLAST`

```
int32 readincr(input_stream<int32> *w, bool &tlast);
```
