---
title: "Writing and Advancing an Output Stream"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Writing-and-Advancing-an-Output-Stream"
toc_id: moQhBmEk78zLVz1Mq3QIxg
content_id: bh~GRDRgggLVggNUaBc1oQ
---

### Writing and Advancing an Output Stream

##### AI Engine Operations

The following operations write data to the given output stream and advance the stream on the AI Engine. The AI Engine has two output stream ports. The AI Enginecompiler automatically assigns physical ports and includes the assignment in the stream data structure.

You can write data values to the output stream individually or as a vector. In the latter case, until all values are written, the stream operation stalls. The data groupings are based on the underlying stream operations. These include single cycle, 32-bit stream operation, and 4 cycle, 128-bit wide stream operation. The cascade connection writes all values in parallel.

```
//Scalar operations
//#include<aie_adf.hpp> or #include<adf.h>
void writeincr(output_stream<int32> *w, int32 v);
void writeincr(output_stream<int64> *w, int64 v);
void writeincr(output_stream<uint32> *w, uint32 v);
void writeincr(output_stream<cint16> *w, cint16 v);
void writeincr(output_stream<cint32> *w, cint32 v);
void writeincr(output_stream<float> *w, float v);
void writeincr(output_stream<cfloat> *w, cfloat v);

//AIE API Operations to read vector data which supports more vector lanes
//#include<aie_adf.hpp>
void writeincr(output_stream<int8> *w, aie::vector<int8,16> &v);
void writeincr(output_stream<uint8> *w, aie::vector<uint8,16> &v);
void writeincr(output_stream<int16> *w, aie::vector<int16,8> &v);
void writeincr(output_stream<cint16> *w, aie::vector<cint16,4> &v);
void writeincr(output_stream<int32> *w, aie::vector<int32,4> &v);
void writeincr(output_stream<cint32> *w, aie::vector<cint32,2> &v);
void writeincr(output_stream<float> *w, aie::vector<float,4> &v);

template<typename T,int N>
void writeincr(output_cascade<T> *w, aie::accum<T,N> &v);
template<typename T,int N>
void writeincr(output_cascade<T> *w, aie::vector<T,N> &v);


//ADF API Operations to read vector data which supports limited vector lanes
//#include<adf.h>
void writeincr(output_cascade<acc48>* str,v8acc48 value);
void writeincr(output_cascade<acc80>* str,v4acc80 value);
void writeincr(output_cascade<cacc48>* str,v4cacc48 value);
void writeincr(output_cascade<cacc80>* str,v2cacc80 value);
void writeincr(output_cascade<accfloat>* str,v8accfloat value);
void writeincr(output_cascade<caccfloat>* str,v4caccfloat value);
void writeincr(output_cascade<int8>* str,v32int8 value);
void writeincr(output_cascade<uint8>* str,v32uint8 value);
void writeincr(output_cascade<int16>* str,v16int16 value);
void writeincr(output_cascade<cint16>* str,v8cint16 value);
void writeincr(output_cascade<int32>* str,v8int32 value);
void writeincr(output_cascade<cint32>* str,v4cint32 value);
void writeincr(output_cascade<float>* str,v8float value);
void writeincr(output_cascade<cfloat>* str,v4cfloat value);
```

For the supported data types and lanes in AI Engine API `writeincr`, see Stream Data Types.

**Note:** `writeincr`

`TLAST`

```
void writeincr(output_stream<int32> *w, int32 value, bool tlast);
```
