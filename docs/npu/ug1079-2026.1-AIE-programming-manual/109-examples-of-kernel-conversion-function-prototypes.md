---
title: "Examples of Kernel Conversion Function Prototypes"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Examples-of-Kernel-Conversion-Function-Prototypes"
toc_id: K~Z33Uu_RawA6ZnSAFB2Jg
content_id: S74ul1jjJ_SzMuS9Ck7UPA
---

### Examples of Kernel Conversion Function Prototypes

Examples of kernel conversions follows:

##### Example 1

Window-based function prototype:

```
void k1(input_window_cint16 * __restrict inputw_l, input_window_cint16 *  __restrict inputw_r, output_window_cint16 * __restrict output);
```

One dimension circular buffer port function prototype:

```
void k1_buffer_port(input_circular_buffer <cint16, adf::extents<adf::inherited_extent>, adf::margin<MARGIN_SIZE> > & __restrict inputw_l, input_circular_buffer <cint16, adf::extents<adf::inherited_extent>, adf::margin<MARGIN_SIZE> > & __restrict inputw_r, output_circular_buffer <cint16, adf::extents<adf::inherited_extent>> & __restrict output);
```

##### Example 2

Window-based function prototype:

```
void k2(input_window_cint16 * __restrict input_cb0, input_window_cint16 * __restrict input_cb1, input_window_cint16 * __restrict input_cb2, input_window_cint16 * __restrict input_cb3, input_window_cint16 * __restrict input_cb4, output_window_cint16 * __restrict output_cb);
```

One dimension buffer port function prototype:

```
void k2_buffer_port(input_buffer<cint16> & __restrict input_cb0, input_buffer<cint16> & __restrict input_cb1, input_buffer<cint16> & __restrict input_cb2, input_buffer<cint16> & __restrict input_cb3, input_buffer<cint16> & __restrict input_cb4, output_buffer<cint16> & __restrict output_cb);
```

**Note:** `dimensions()`
