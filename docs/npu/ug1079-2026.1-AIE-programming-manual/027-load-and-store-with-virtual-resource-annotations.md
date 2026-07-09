---
title: "Load and Store with Virtual Resource Annotations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Load-and-Store-with-Virtual-Resource-Annotations"
toc_id: B2as4QlPrgAfmA2HVySVSg
content_id: sTaz9ByCuOBEtCzl9plj4g
---

#### Load and Store with Virtual Resource Annotations

The AI Engine can perform several vector load or store operations per cycle. Load and store operations can run in parallel only if they target different memory banks. In general, the compiler tries to schedule many memory accesses in the same cycle when possible but there are some exceptions. Memory accesses coming from the same pointer are scheduled on different cycles. If the compiler schedules the operations on multiple variables or pointers in the same cycle, memory bank conflicts can occur.

To prevent concurrent access to memory with multiple variables or pointers, most memory access functions in the AI Engine API use an enum value from `aie_dm_resource`. See the following code example:

```
enum class aie_dm_resource {
  none,
  a,
  b,
  c,
  d,
  stack
};
```

Also, the compiler provides the following `aie_dm_resource` annotations to annotate different virtual resources. Accesses using types that are associated with the same virtual resource do not access the resource at the same cycle.

```
__aie_dm_resource_a
__aie_dm_resource_b
__aie_dm_resource_c
__aie_dm_resource_d
__aie_dm_resource_stack
```

The following example shows how to annotate memory access. It binds individual access to virtual resources and controls accessing memories at the same cycle.

```
int __aie_dm_resource_a *A;
int *B;

aie::vector<int,8> v1 = aie::load_v<8>(A);

/* Following access can be scheduled on the same cycle
 * as the access to A since B is not annotated.
 */
aie::vector<int,8> v2 = aie::load_v<8>(B);

/* Following specific access to B
 * is annotated with the same virtual resource as A
 * so they cannot be scheduled on the same cycle.
 */
aie::vector<int,8> v3 = aie::load_v<8, aie_dm_resource::a>(B);

/* vector iterator of B
 * annotated with the same virtual resource as A
 * so they cannot be scheduled on the same cycle.
 */
auto it = aie::begin_vector<8, aie_dm_resource::a>(B);
aie::vector<int,8> v4 = *(++it);
```

For example, this code annotates two arrays with the same `__aie_dm_resource_a`. This tells the compiler not to access them in the same instruction. It shows two ways to load vectors, using `aie::load_v` and iterators.

```
aie::vector<int32,8> va[32];
aie::vector<int32,8> vb[32];
int32 __aie_dm_resource_a* __restrict p_va = (int32 __aie_dm_resource_a*)va;
int32 __aie_dm_resource_a* __restrict p_vb = (int32 __aie_dm_resource_a*)vb;
auto it_b=aie::begin_vector<8>(p_vb);

// access va, vb by p_va, it_b
aie::vector<int32,8> vc;
vc=aie::load_v<8>(p_va)+*it_b;
p_va+=8;
++it_b;
```

```
void kernel_top(input_buffer<int32> & __restrict data1,
                input_buffer<int32>& __restrict data2, ...) {
  auto w_data1 = (int32 __aie_dm_resource_a* __restrict)data1.data();
  auto w_data2 = (int32 __aie_dm_resource_b* __restrict)data2.data();
  auto pv = aie::begin_vector<8>(w_data1);
  auto pv2 = aie::begin_vector<8>(w_data2);
  auto va = *pv++;
  auto vb = *pv2++;
  ...
}
```

`__aie_dm_resource_a`

```
alignas(aie::vector_decl_align) static int32 coeff[256]={...};
void func(input_buffer<int32> & __restrict wa, ...... )
{
aie::vector<int32,8> v_coeff=aie::load_v<8>((int32 __aie_dm_resource_a*)coeff);
int32 __aie_dm_resource_a* __restrict p_wa = (int32 __aie_dm_resource_a*)wa.data();
auto waIter = aie::begin_vector<8>(p_wa);
aie::vector<int32,8> va;
va = *waIter;
...
}
```
