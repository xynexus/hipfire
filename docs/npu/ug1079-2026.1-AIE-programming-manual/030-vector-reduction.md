---
title: "Vector Reduction"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vector-Reduction"
toc_id: qLu2g4OlPALuR579xxDZeg
content_id: MAn1rTNIlL5MgE6Nbg5oPA
---

## Vector Reduction

The AI Engine API supports vector reduce operations on `aie::vector`.

- **`aie::reduce_add`:** Returns sum of the elements in the input vector.
- **`aie::reduce_add_v`:** Returns the sums of the elements in the input vectors. The sum of each input vector is stored in an element of the output vector.
- **`aie::reduce_max`:** Returns the element from the input vector with the largest value.
- **`aie::reduce_min`:** Returns the element from the input vector with the smallest value.
- **`aie::reduce_mul`:** Returns multiplication of the elements in the input vector.

Vector reduction examples are as follows.

```
aie::vector<int16,8> iv,iv2,iv3,iv4;
int16 iv_add=aie::reduce_add(iv);
aie::vector<int16,8> iv_add_v4=aie::reduce_add_v(iv,iv2,iv3,iv4);
int16 iv_max=aie::reduce_max(iv);
int16 iv_min=aie::reduce_min(iv);
int16 iv_mul=aie::reduce_mul(iv);
```
