---
title: "Floating-Point Operations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Floating-Point-Operations"
toc_id: mD8zMFvtZ1rlbZ0pJiPxGA
content_id: 3N8uyJ692ax1LvErO9bq9g
---

## Floating-Point Operations

The scalar unit floating-point hardware support includes square root, inverse square root, inverse, absolute value, minimum, and maximum. It supports other floating-point operations through emulation. The `softfloat` library must be linked in for test benches and kernel code using emulation. You must use the single precision float version for math library functions, (for example, use `expf()` instead of `exp()`).

The AI Engine vector unit provides eight lanes of single-precision floating-point multiplication and accumulation. The unit reuses the vector register files and permute network of the fixed-point data path. In general, only one vector instruction per cycle can run in fixed-point or floating-point.

Floating-point MACs have two-cycle latency. Using two accumulators in a ping-pong manner helps performance by allowing the compiler to schedule a MAC on each clock cycle.

```
acc0 = fpmac( acc0, abuff, 1, 0x0, bbuff, 0, 0x76543210 );
acc1 = fpmac( acc1, abuff, 9, 0x0, bbuff, 0, 0x76543210 );
```

There are no divide scalar or vector intrinsic functions at this time. You can implement vector division using an inverse and a multiply, as shown in the following example.

```
invpi = upd_elem(invpi, 0, inv(pi));
acc = fpmul(concat(acc, undef_v8float()), 0, 0x76543210, invpi, 0, 0);
```

You can implement similar patterns for vector operations such as `sqrt`, `invsqrt`, and `sincos`.
