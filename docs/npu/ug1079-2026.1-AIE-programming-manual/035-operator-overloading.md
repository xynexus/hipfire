---
title: "Operator Overloading"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Operator-Overloading"
toc_id: eqanU2qETCijeeiUAEP4xw
content_id: XynT_0FuqBzb0j9uk5HL4A
---

## Operator Overloading

The AI Engine API provides operator overloading for many operations. Use this feature by including the `aie_api/operators.hpp` header and the `aie::operators` namespace.

The included operators are as follows:

- **Operator `+`:** Addition operator. It is equivalent to `aie::add`.
- **Operator `+=`:** Addition assignment operator.
- **Operator `-`:** Negation operator. It is equivalent to `aie::neg`.
- **Operator `-`:** Subtraction operator. It is equivalent to `aie::sub`.
- **Operator `-=`:** Subtraction assignment operator.
- **Operator `!=`:** Not equal to comparison operator. It is equivalent to `aie::neq`.
- **Operator `<`:** Less than comparison operator. It is equivalent to `aie::lt`.
- **Operator `<=`:** Less than or equal comparison operator. It is equivalent to `aie::le`.
- **Operator `==`:** Equal to comparison operator. It is equivalent to `aie::eq`.
- **Operator `>`:** Greater than comparison operator. It is equivalent to `aie::gt`.
- **Operator `>=`:** Greater than or equal comparison operator. It is equivalent to `aie::ge`.
- **Operator `<<`:** Bitwise left shift operator. It is equivalent to `aie::upshift`.
- **Operator `>>`:** Bitwise right shift operator. It is equivalent to `aie::downshift`.
- **Operator `&`:** Bitwise AND operation. It is equivalent to `aie::bit_and`.
- **Operator `^`:** Bitwise XOR operation. It is equivalent to `aie::bit_xor`.
- **Operator `|`:** Bitwise OR operation. It is equivalent to `aie::bit_or`.
- **Operator `~`:** Bitwise NEG operation. It is equivalent to `aie::bit_not`.

An example code of using operators is as follows.

```
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include <aie_api/operators.hpp>
using namespace aie::operators;

aie::vector<int32,8> va,vb;
aie::vector<int32,8> vadd=va+vb;
aie::vector<int32,8> vsub=va-vb;

// negation of each element
aie::vector<int32,8> vneg=-vb;
vadd+=vneg;
aie::print(va,true,"va=");
aie::print(vadd,true,"vadd=");

// compare each element of the vector correspondingly
auto msk_neq=(vadd!=va);

// mask bit equals 1 if vector element not equal
aie::print(msk_neq,true,"msk_neq=");

// bit not
auto vnot_va=~va;

// bit xor
auto vones=va ^ vnot_va;
aie::print(vones,true,"vones=");

// choose vadd if mask bit equals 0
auto vout=aie::select(vadd,vones,msk_neq);

aie::print(vout,true,"vout=");
```

One possible output of the previous code is as follows.

```
va=9 10 11 12 13 14 15 16
vadd=9 10 11 12 13 14 15 16
msk_neq=0 0 0 0 0 0 0 0
vones=-1 -1 -1 -1 -1 -1 -1 -1
vout=9 10 11 12 13 14 15 16
```
