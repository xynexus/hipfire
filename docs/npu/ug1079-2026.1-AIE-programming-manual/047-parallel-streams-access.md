---
title: "Parallel Streams Access"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Parallel-Streams-Access"
toc_id: 3HVjI0E9DGIwusvkbCULgQ
content_id: HXKR6lKM7Isim3lEeeIHWA
---

## Parallel Streams Access

The AI Engine can use two streaming inputs or two streaming outputs in parallel.

`aie_stream_resource_in`

`aie_stream_resource_out`

`aie_stream_resource_in::a`

`aie_stream_resource_in::b`

```
void vect_mul(input_stream<int8>* __restrict data1, input_stream<int8>* __restrict data2,
      output_stream<int8>* __restrict out){
  while(true)
  chess_prepare_for_pipelining
  chess_loop_range(8,)
  {
    aie::vector<int8,16> va_int8=readincr_v<16,aie_stream_resource_in::a>(data1);
    aie::vector<int8,16> vb_int8=readincr_v<16,aie_stream_resource_in::b>(data2);
    auto vc=aie::mul(va_int8,vb_int8);

    // Avoid the write instruction to occur at the same cycle as the readincr of data1
    writeincr<aie_stream_resource_out::a>(out,vc.to_vector<int8>(0));

    va_int8=readincr_v<16,aie_stream_resource_in::a>(data1);
    vb_int8=readincr_v<16,aie_stream_resource_in::b>(data2);
    vc=aie::mul(va_int8,vb_int8);

    // This writeincr can be scheduled as soon as possible
    // as there is the __restrict keyword in function signature
    writeincr(out,vc.to_vector<int8>(0));
  }
}
```

Similarly, you can use `aie_stream_resource_out::a` and `aie_stream_resource_out::b` to denote two parallel output streams.

**Recommended:** For example, AMD does not recommend the following stream operations:

```
//input_stream<int32> * __restrict sin
......
int32 a = readincr(sin); //access stream with none annotation
int32 b = readincr(sin);
v4int32 c_vect = readincr_v<4, aie_stream_resource_in::a>(sin); //access stream with different annotation
```

Instead, use the following type of code:

```
//input_stream<int32> * __restrict sin
......
int32 a = readincr<aie_stream_resource_in::a>(sin); //access stream with annotation
int32 b = readincr<aie_stream_resource_in::a>(sin);
v4int32 c_vect = readincr_v<4, aie_stream_resource_in::a>(sin); //access stream with the same annotation
```
