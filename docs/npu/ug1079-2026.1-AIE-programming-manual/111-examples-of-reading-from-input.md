---
title: "Examples of Reading From Input"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Examples-of-Reading-From-Input"
toc_id: YxKmikE6BL3v6hI8KG6myA
content_id: ESNmiipjTksVEtwx4DcxBw
---

### Examples of Reading From Input

Window read operations need to convert to input buffer port iterator reads. Vector-based operations need to use a vector iterator. Iterator incrementing/decrementing corresponds to`window_readincr()` and `window_writeincr()` operations.

Window-based input example:

```
input_window_cint16 * restrict inputw_l;
...
window_incr_v8(inputw_r,3);
v8cint16 vdata;
window_readincr(inputw_l, vdata);
window_read(inputw_l, vdata);
window_decr_v8(inputw_l, 1);
```

Buffer port based input:

```
adf::input_circular_buffer<cint16, adf::extents<adf::inherited_extent>, adf::margin<MARGIN_SIZE>> & __restrict inputw_l;
auto inputw_lItr = aie::begin_vector_random_circular<8>(inputw_l);

inputw_lItr += 3;

v8cint16 vdata;
vdata = *inputw_lItr++;
vdata = *inputw_lItr;
inputw_lItr--;
```

Window-based output example:

```
output_window_cint16 * __restrict outputw;
const unsigned output_samples = GET_NUM_SAMPLES(outputw);
acc0 = mac4_sym_ct(acc0,lbuff,14,0x6420,2,rbuff,0,5,coe,8,0x0000,1);
window_writeincr(outputw, srs(acc0,shift));
```

Buffer port-based output:

```
output_circular_buffer<cint16> & __restrict outputw;
const unsigned output_samples = outputw.size();
auto outputwItr = aie::begin_vector_random_circular<8>(outputw);

acc0 = mac4_sym_ct(acc0,lbuff,14,0x6420,2,rbuff,0,5,coe,8,0x0000,1);
aie::vector<cint16, 4> v1 = srs(acc0,shift);
*outputwItr++ = v1;
```
