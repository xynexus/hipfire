---
title: "Pointer Aliasing"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Pointer-Aliasing"
toc_id: QZO~F8dkjiRBME~tOsrk_w
content_id: y9xffDu47sf~g1Jou5u0Cw
---

## Pointer Aliasing

Pointer aliasing occurs when different pointer names access the same memory location. The strict aliasing rule in C/C++ means that pointers are assumed not to alias if they point to fundamentally different types. Aliasing introduces strong constraints on program execution order. The following shows the aliasing of `p` and `q`.

![xdh1593461230536.png](../assets/283-01-xdh1593461230536-png-07c2d9bf563f.png)

*Figure 1. Pointer Aliasing*

The following is an example of pointer aliasing, in which both the pointers `p` and `q` point to the same address. The assembly language code produced by the compiler is shown in the middle column, and the operations and clock cycles are shown on the right.

![xkh1593463517325.png](../assets/283-02-xkh1593463517325-png-cd5eff85294c.png)

*Figure 2. Aliasing Code Example*

By adding the restrict keyword into this code example, the compiler can optimize the resulting assembly language to increase parallelization of the operations in hardware. The following example shows that using the restrict keyword to prevent aliasing uses fewer clock cycles to complete the same operation.

![ans1593463701117.png](../assets/283-03-ans1593463701117-png-af40c5c34894.png)

*Figure 3. Use of Restrict Keyword to Avoid Aliasing*

#### Memory Dependencies

Memory dependencies in the code can limit the kinds of optimizations attempted by the compiler. For example in the following code, `xyz` and pointers `p` and `q` can be unrelated. However, within the function code both pointer `p` and pointer `q` point to same global variable `xyz`. The compiler must guarantee the correct execution under both these conditions. Due to these kinds of memory dependencies the compiler needs to be conservative and limit optimizations.

![rpx1594833782121.png](../assets/283-04-rpx1594833782121-png-92b21915861d.png)

*Figure 4. Unrelated Pointers*
