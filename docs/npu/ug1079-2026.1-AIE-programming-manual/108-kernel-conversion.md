---
title: "Kernel Conversion"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Kernel-Conversion"
toc_id: Mil~WQEe9YlNyiLJONS_iA
content_id: 6yme5018lzj_EE7DFwkwSw
---

## Kernel Conversion

When migrating from window based kernels to buffer port based kernels, you need to determine the buffer port addressing type for each kernel. Window data types are implemented as circular buffers; however, buffer port types are implemented as linear buffers unless declared as circular buffer port types. Criteria selecting addressing mode are listed below.

Criteria to select circular address mode are as follows:

- Both kernels must be in the same tile, and they must communicate with each other directly.
- A margin size greater than 0 is used.
- Kernel needs to iterate over data cyclically.

Criteria to select linear address mode are as follows:

- Kernel location is not guaranteed to be within the same tile.
- The kernel does not require any margin.
- Extra margin copy does impact design performance.

For an explanation of the differences between linear and circular buffer ports, see Linear and Circular Addressing of Buffer Ports.
