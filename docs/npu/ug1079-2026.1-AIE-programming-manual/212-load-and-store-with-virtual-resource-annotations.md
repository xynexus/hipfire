---
title: "Load and Store with Virtual Resource Annotations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Load-and-Store-with-Virtual-Resource-Annotations"
toc_id: g0fhULinVOO0r34tttLKmQ
content_id: QBC_15LM85RVS13MUU0oNg
---

#### Load and Store with Virtual Resource Annotations

AI Engine can perform several vector load or store operations per cycle. However, in order for the load or store operations to be executed in parallel, they must target different memory banks. In general, the compiler tries to schedule many memory accesses in the same cycle when possible, but there are some exceptions. Memory accesses coming from the same pointer are scheduled on different cycles. If the compiler schedules the operations on multiple variables or pointers in the same cycle, memory bank conflicts can occur.

To avoid concurrent access to a memory with multiple variables or pointers, the compiler provides the following `aie_dm_resource` annotations to annotate different virtual resources. Accesses that use types that are associated with the same virtual resource are not scheduled to access the resource in the same cycle.

```
__aie_dm_resource_a
__aie_dm_resource_b
__aie_dm_resource_c
__aie_dm_resource_d
__aie_dm_resource_stack
```

For example, the following code annotates two arrays to the same `__aie_dm_resource_a` that guides the compiler to not access them in the same instruction.

```
v8int32 va[32];
v8int32 vb[32];
v8int32 __aie_dm_resource_a* restrict p_va = (v8int32 __aie_dm_resource_a*)va;
v8int32 __aie_dm_resource_a* restrict p_vb = (v8int32 __aie_dm_resource_a*)vb;
//access va, vb by p_va, p_vb
v8int32 vc;
vc=p_va[i]+p_vb[i];
```

The following code annotates an array and a window buffer to the same `__aie_dm_resource_a` that guides the compiler to not access them in the same instruction.

```
void func(input_window_int32 * __restrict wa, ......
  v8int32 coeff[32];
  input_window_int32 sa;
  v8int32 __aie_dm_resource_a* restrict p_coeff = (v8int32 __aie_dm_resource_a*)coeff;
  input_window_int32 __aie_dm_resource_a* restrict p_wa = (input_window_int32 __aie_dm_resource_a*)&sa;
  window_copy(p_wa,wa);
  v8int32 va;
  va=window_readincr_v8(p_wa);//access wa by p_wa
```
