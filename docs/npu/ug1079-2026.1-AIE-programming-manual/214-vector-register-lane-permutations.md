---
title: "Vector Register Lane Permutations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vector-Register-Lane-Permutations"
toc_id: BGuTdoPR0WnQpESoPpvG~A
content_id: Ie4H0C2Bs78Mh0SiP3Sgdg
---

## Vector Register Lane Permutations

The AI Engine fixed-point vector units datapath consists of the following three separate and largely independently usable paths:

- Main MAC datapath
- Shift-round-saturate path
- Upshift path

The main multiplication path performs the following:

1. Reads values from vector registers
2. Permutes the values in a user-controllable fashion
3. Performs optional pre-adding
4. Multiplies
5. After some post-adding accumulates them to the previous value of the accumulator register

The main datapath stores to the accumulator. During this, the shift-round-saturate path reads from the accumulator registers and stores to the vector registers or the data memory. In parallel to the main datapath runs the upshift path. It does not perform any multiplications but simply reads vectors, upshifts them and feeds the result into the accumulators.

See the Versal Adaptive SoC AI Engine Architecture Manual ([AM009](https://docs.amd.com/go/en-US/am009-versal-ai-engine)) for details on the fixed-point and floating-point data paths.

See the AI Engine Intrinsics User Guide ([UG1078](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_intrinsics/intrinsics/index.html)) for details on the intrinsic functions that can be used to exercise these data paths.

The following figure shows the basic functionality of MAC data path. The path consists of vector multiply and accumulate operations between data from the X and Z buffers. Other parameters and options support flexible data selection within the vectors and number of output lanes. Optional features support different input data sizes and pre-adding. There is an additional input buffer, the Y buffer, whose values can be pre-added with those from the X buffer before the multiplication occurs. The result from the intrinsic is added to an accumulator.

![hja1610155668510.png](../assets/214-01-hja1610155668510-png-b0a825ae1c21.png)

*Figure 1. Functional Overview of the MAC Data Path*

The operation can be described using lanes and columns. The number of lanes corresponds to the number of output values that the intrinsic call generates. The number of columns equals the number of multiplications per output lane. Each multiplication result is added together. For example:

```
acc0 += z00*(x00+y00) + z01*(x01+y01) + z02*(x02+y02) + z03*(x03+y03)
acc1 += z10*(x10+y10) + z11*(x11+y11) + z12*(x12+y12) + z13*(x13+y13)
acc2 += z20*(x20+y20) + z21*(x21+y21) + z22*(x22+y22) + z23*(x23+y23)
acc3 += z30*(x30+y30) + z31*(x31+y31) + z32*(x32+y32) + z33*(x33+y33)
```

This example generates four outputs. Therefore there are four lanes and four columns for each of the outputs with pre-addition from the X and Y buffers.

Intrinsics parameters allow for flexible data selection from the different input buffers for each lane and column, all following the same pattern of parameters. The following section introduces the data selection (or data permute) schemes with detailed examples that include `shuffle` and `select` intrinsics. The following sections discuss the `mac` intrinsic and its variants.
