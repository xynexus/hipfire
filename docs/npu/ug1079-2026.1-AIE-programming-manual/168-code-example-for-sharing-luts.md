---
title: "Code Example for Sharing LUTs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Code-Example-for-Sharing-LUTs"
toc_id: ~MHdvTtoZy2~MI_CP~3e~w
content_id: DQ6Glj9P95vw6ZtVtH0onQ
---

### Code Example for Sharing LUTs

```
// in kernels.cpp
// --------------

// Define LUT array
const int32 lut_array[8] = {1,2,3,4,5,6,7,8};

void func1(output_stream_int32 *out) {
  // ... make use of lut_array ...
}

void func2(output_stream_int32 *out) {
  // ... make use of lut_array ...
}

void func3(output_stream_int32 *out) {
  // ... make use of lut_array ...
}

// In graph definition graph.h
// ---------------------------

lut_array_master = adf::parameter::array(lut_array); // will be in kernel0
lut_array_slave1 = adf::parameter::array(lut_array);
lut_array_slave2 = adf::parameter::array(lut_array);

// Constraints to place slave kernels in the neighboring tiles of kernel0
adf::location<kernel>(kernel0) = tile(25,0);
adf::location<kernel>(kernel1) = tile(25,1);
adf::location<kernel>(kernel2) = tile(26,0);

// Constraint to place the master parameter in the same tile as the kernel0
adf::location<adf::parameter>(lut_array_master) = adf::location<kernel>(kernel0);

// use new graph construct for shared LUT parameters
adf::share_parameter(lut_array_master, {lut_array_slave1, lut_array_slave2});

adf::connect<>(lut_array_master, kernel0);
adf::connect<>(lut_array_slave1, kernel1);
adf::connect<>(lut_array_slave2, kernel2);
```
