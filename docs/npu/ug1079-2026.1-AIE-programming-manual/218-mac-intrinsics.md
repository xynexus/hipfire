---
title: "MAC Intrinsics"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/MAC-Intrinsics"
toc_id: cCYKJWSsprj0G7A7QFZThA
content_id: ReYXVkXkaMsUzV3xZbrKSA
---

### MAC Intrinsics

MAC intrinsics perform vector multiply and accumulate operations between data from two buffers (X and Z). Other parameters and options provide flexibility (data selection within the vectors, number of output lanes) and optional features (different input data sizes, pre-adding, etc.). There is an additional input buffer (Y). Buffer Y values can be pre-added with those from the X buffer before the multiplication occurs. The intrinsic result adds to an accumulator.

The parameters of the intrinsics allow flexible data selection from different input buffers for each lane and column, all following the same pattern of parameters. A starting point in the buffer is given by the (x/y/z) start parameter. This parameter selects the first element for the first row as well as the first column. To allow flexibility for each lane, (x/y/z) offsets provide an offset value for each lane that adds to the starting point. Finally, the (x/y/z) step parameter defines the step in data selection between each column based on the previous position. It is worth noting that when the ystep is not specified in the intrinsic it is the symmetric of the xstep.

Main permute granularity for x/y and z buffers is 32 bits and 16 bits, respectively. Permute treats complex numbers as a single entity (for example, it considers `cint16` as 32 bits during permutation). Parameter `zstart` must be a compile time constant. 8-bit and 16-bit permute granularity in x/y and 8-bit permute granularity in z have certain limitations. The end of this section addresses these limitations. The following sections covers the different data widths and explains the result of the MAC intrinsic on these data widths.
