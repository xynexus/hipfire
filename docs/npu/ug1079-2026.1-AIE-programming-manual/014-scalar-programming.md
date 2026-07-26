---
title: "Scalar Programming"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Scalar-Programming"
toc_id: JHpWmmix~IkkFXIb0jkAaQ
content_id: jAGpsbkq5VIbn4s6VHci8A
---

## Scalar Programming

The compiler and scalar unit enable the programmer to use standard C data types. The following table shows standard C data types with their precisions. All types, except float and double, support signed and unsigned prefixes.

| Data Type | Precision | Comment |
| --- | --- | --- |
| char | 8-bit signed |  |
| short | 16-bit signed |  |
| int | 32-bit signed | Native support |
| long | 64-bit signed |  |
| float | 32-bit | Emulated. Scalar processor does not contain a floating point unit (FPU). |
| double | 64-bit | Emulated. Scalar processor does not contain a floating point unit (FPU). |

**Note:** Important:

AI Engine

**Note:** AI Engine

complex

`<complex.h>`

AI Engine
