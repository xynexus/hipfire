---
title: "AI Engine Memory"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-Memory"
toc_id: _COHxJXG1VCkCnSjaMmHlw
content_id: deJFipUN0GOboG8EuKlSEQ
---

## AI Engine Memory

Each AI Engine has 16 KB of program memory, which allows storing 1024 instructions of 128 bits each. AI Engine instructions are 128 bits (maximum) wide and support multiple instruction formats, as well as variable length instructions to reduce the program memory size. Many instructions outside of the optimized inner loop can use the shorter formats.

Each AI Engine tile has eight data memory banks. Each memory bank (single bank) is a 256 word x 128-bit single-port memory (for a total of 32 KB). Each AI Engine can access three of the memories from neighboring tiles plus its own data memory for a total of 128 kB. The stack is a subset of the data memory. The default value for stack size and heap size is 1 KB. The compiler can automatically compute and adjust Heap Size when optimization level is larger than zero (`xlopt`>=1 for the AI Engine compiler).

Stack size and heap size can be changed using compiler options or constraints in the source code. Refer to the AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)) for more information about stack and heap size usage.

In a logical representation, the 128 KB memory can be seen as either a single continuous 128 KB block or as four separate 32 KB blocks. Each 32 KB block can further be split into four odd and four even banks. One even bank and one odd bank are interleaved to comprise a double bank. AI Engines on the edges of the AI Engine array have fewer neighbors and correspondingly less memory available.

Each memory port operates in 256-bit/128-bit vector register mode or 32-bit/16-bit/8-bit scalar register mode. An even and odd pairing of the memory banks creates the 256-bit port. The 8-bit and 16-bit stores are implemented as read-modify-write instructions. Concurrent operation of all three ports is supported if each port is accesses a different bank.

Data stored in memory is in little endian format.

**Recommended:** AMD

Each AI Engine has a DMA controller that is divided into two separate modules:

- S2MM to store stream data to memory (32-bit data)
- MM2S to write the contents of the memory to a stream (32-bit data)

Each S2MM and MM2S has two independent data channels.
