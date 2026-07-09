---
title: "Stream Data Types"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Stream-Data-Types"
toc_id: OWiaJvH0zbdg3jJxLsII3w
content_id: x6pqjazIppe3dvSICboDzQ
---

### Stream Data Types

You can read or write each data type in the table from the AI Engine as scalars or in vector groups. However, valid groupings are subject to certain restrictions. These depend on the bus data width supported on the AI Engine to programmable logic interface ports or through the stream-switch network. The valid combinations for AI Engine kernels are vector bundles totaling up to 32-bits or 128-bits. The accumulator data types specify cascade-stream connections between adjacent AI Engines. Its valid groupings are based on the 384-bit wide cascade channel between two processors.

Youcan use input and output Cascade data types in the context of AI Engine APIs or ADF APIs. AMD recommends the use of AI Engine APIs because it supports a larger range of lanes. To use AI Engine APIs, include `#include <aie_adf.hpp>` in the kernel source code.

ADF APIs support a limited number of lanes. To use ADF APIs, include `#include <adf.h>` in the kernel source code. Use ADF APIs for advanced kernel programming using intrinsic calls.

| Input Stream Types | Output Stream Types |
| --- | --- |
| input_stream<int8> | output_stream<int8> |
| input_stream<int16> | output_stream<int16> |
| input_stream<int32> | output_stream<int32> |
| input_stream<int64> | output_stream<int64> |
| input_stream<uint8> | output_stream<uint8> |
| input_stream<uint16> | output_stream<uint16> |
| input_stream<uint32> | output_stream<uint32> |
| input_stream<uint64> | output_stream<uint64> |
| input_stream<cint16> | output_stream<cint16> |
| input_stream<cint32> | output_stream<cint32> |
| input_stream<float> | output_stream<float> |
| input_stream<cfloat> | output_stream<cfloat> |

| Input Cascade Types | Output Cascade Types | Lanes in ADF API (adf.h) | Lanes in AIE API (aie_adf.hpp) |
| --- | --- | --- | --- |
| input_cascade<acc48> | output_cascade<acc48> | 8 | 8/16/32/64/128 |
| input_cascade<cacc48> | output_cascade<cacc48> | 4 | 4/8/16/32/64 |
| input_cascade<acc80> | output_cascade<acc80> | 4 | 4/8/16/32/64 |
| input_cascade<cacc80> | output_cascade<cacc80> | 2 | 2/4/8/16/32 |
| input_cascade<accfloat> | output_cascade<accfloat> | 8 | 4/8/16/32 |
| input_cascade<caccfloat> | output_cascade<caccfloat> | 4 | 2/4/8/16 |
| input_cascade<int8> | output_cascade<int8> | 32 | 16/32/64/128 |
| input_cascade<uint8> | output_cascade<uint8> | 32 | 16/32/64/128 |
| input_cascade<int16> | output_cascade<int16> | 16 | 8/16/32/64 |
| input_cascade<int32> | output_cascade<int32> | 8 | 4/8/16/32 |
| input_cascade<cint16> | output_cascade<cint16> | 8 | 4/8/16/32 |
| input_cascade<cint32> | output_cascade<cint32> | 4 | 2/4/8/16 |
| input_cascade<cfloat> | output_cascade<cfloat> | 4 | 2/4/8/16 |
| input_cascade<float> | output_cascade<float> | 8 | 4/8/16/32 |
