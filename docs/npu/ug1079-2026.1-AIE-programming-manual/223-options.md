---
title: "Options"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Options"
toc_id: lL_~noGkv8Z4zSvu48AYSw
content_id: Uy_03sYd9JYeBQr9UKNo5A
---

#### Options

There are rich sets of MAC intrinsic with additional operations like pre-adding, pre-subtraction, and conjugation. The naming convention for the vector MAC intrinsics is as follows. Optional characteristics are shown in [] and mandatory ones in {}.

```
[l]{mac|msc|mul|negmul}{2|4|8|16}[_abs|_max|_min|_maxdiff][_conj][{_sym|_antisym}[_ct|_uct]][_c|_cc|_cn|_nc]
```

Every operation will either be a multiplication, initializing an accumulator, or a MAC operation which accumulates to a running accumulator of 2, 4, 8, or 16 lanes.

- **`l`:** Denotes that an accumulator with 80-bit lanes is used for the operation.
- **`sym` and `antisym`:** Indicates the use of pre-adding and pre-subtraction respectively.
- **`max`, `min`, and `maxdiff`:** Indicates the pre-selection of lanes in the `xbuff` based on the maximum, minimum, or maximum difference value.
- **`abs`:** Indicates the pre-computation of the absolute value in the `xbuff`.
- **`ct`:** Used for partial pre-adding and pre-subtraction (separate selection for the data input from X for the final column).
- **`uct`:** Used for unit center optimization for certain types of FIR filters. Refer to the AI Engine Intrinsics User Guide ([UG1078](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_intrinsics/intrinsics/index.html)) for more information.
- **`n` and `c`:** Indicates that the complex conjugate will be used for one of the input buffers with complex values:`c` Only the complex input buffer is conjugated. `cn` Complex conjugate of X (or XY if using pre-adding) buffer. `nc` Complex conjugate of Z buffer. `cc` Complex conjugate of both X (or XY if using pre-adding) and Z buffers.
- **`conj`:** Indicates that the complex conjugate of Z is used when multiplying the data input from Y.
