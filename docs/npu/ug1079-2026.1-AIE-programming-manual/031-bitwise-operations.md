---
title: "Bitwise Operations"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Bitwise-Operations"
toc_id: rR59Hu1pjpV4sTxBAGlSMA
content_id: 3JiOWcPaDJIk2yHv0qXd4g
---

## Bitwise Operations

The following AI Engine APIs allow bitwise AND, OR, XOR, and NOT operations on a scalar value and all the elements of the input vector with same type, or on two input vectors with same type and size:

- `aie::bit_and`
- `aie::bit_or`
- `aie::bit_xor`

The following AI Engine API allows bitwise NOT operations on the elements of the input vector:

- `aie::bit_not`

**Note:** `aie::bit_*`

AI Engine

```
aie::vector<int16,8> iv,iv2;

// 0xf bit AND with each component of iv
aie::vector<int16,8> iv_bit_and=aie::bit_and((int16)0xf,iv);
aie::vector<int16,8> iv_bit_or=aie::bit_or((int16)0xf,iv);
aie::vector<int16,8> iv_bit_xor=aie::bit_xor((int16)0xf,iv);

// bit AND on each component of iv and iv2
aie::vector<int16,8> iv_bit_and2=aie::bit_and(iv,iv2);
aie::vector<int16,8> iv_bit_or2=aie::bit_or(iv,iv2);
aie::vector<int16,8> iv_bit_xor2=aie::bit_xor(iv,iv2);
aie::vector<int16,8> iv_not=aie::bit_not(iv);
```

The following AI Engine APIs shift all components of a vector right or left by specified number of bits:

- `aie::downshift`: Shifts each component of a vector arithmetically (signed extended) right by specific bits. `aie::vector<int16,8> iv_down=aie::downshift(iv,2);`

  ```
  aie::vector<int16,8> iv_down=aie::downshift(iv,2);
  ```
- `aie::upshift`: Shifts each component left by specific bits. Filled by 0. `aie::vector<int16,8> iv_up=aie::upshift(iv,2);`

  ```
  aie::vector<int16,8> iv_up=aie::upshift(iv,2);
  ```
- `aie::logical_downshift`: Shifts each component of a vector logically (zero extended) right by specific bits.`aie::vector<int16,8> iv_logic_down=aie::logical_downshift(iv,2);`

  ```
  aie::vector<int16,8> iv_logic_down=aie::logical_downshift(iv,2);
  ```
