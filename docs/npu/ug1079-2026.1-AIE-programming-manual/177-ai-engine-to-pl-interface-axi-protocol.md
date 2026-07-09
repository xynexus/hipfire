---
title: "AI Engine to PL Interface AXI Protocol"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-to-PL-Interface-AXI-Protocol"
toc_id: tIXavRcscvi5soL1AFCd4Q
content_id: ggmyYd95V5dSR1L2Q~wH4Q
---

## AI Engine to PL Interface AXI Protocol

AI Engine to PL AXI4-Stream interfaces support a subset of the AXI4-Stream protocol (per the [AMBA 4 AXI4-Stream Protocol Specification](https://developer.arm.com/documentation/ihi0051/a/)). Internally, AI Engine to PL AXI4-Stream interfaces are 64-bit channels physically. And 128-bit occupies two adjacent physical channels. This imposes additional requirements on sending data timely between AI Engine and PL via AXI4-Stream interfaces. This section focuses on `TLAST` and `TKEEP` requirements of the interfaces to send data without stall.

`TLAST` is required for a 64-bit stream between the AI Engine and PL if single 32-bit words are sent. AI Engine to PL 32-bit stream interfaces are automatically internally up-sized to 64-bit interfaces by the AI Engine compiler. When sending 32-bit stream data (to or from the PL from the AI Engine), single 32-bit words without `TLAST` are held in the interface until a second 32-bit word arrives to complete a 64-bit up-sizing. The solution is to assert `TLAST` for the single 32-bit data. The data is pushed into the AI Engine without stall.

When using 64-bit and 128-bit interfaces, it is valid to send data without `TLAST`. Then, even number of data are sent without stall. `TLAST` can be used depending on the need. When `TLAST` is asserted, `TKEEP` can be used together with `TLAST` to send arbitrate number of 32-bit data for 64-bit and 128-bit interfaces. `TKEEP` must be set correctly, either -1 (all bits are 1) or partial 32-bit words are enabled, for example 0x0F. The lower parts of the data must be asserted if only partial data is valid.

`TLAST`

`TKEEP`

AI Engine

AXI4-Stream

| Interface Bit Width | TLAST | TKEEP | Note |
| --- | --- | --- | --- |
| 32-bit | 1 | -1 | 32-bit word to send without stall. |
| 64-bit | 1 | 0x0F | Lowest 32-bit word to send without stall. |
| 128-bit | 1 | 0x000F | Lowest 32-bit word to send without stall. |
| 128-bit | 1 | 0x00FF | Lowest two 32-bit word to send without stall. |
| 128-bit | 1 | 0x0FFF | Lowest three 32-bit word to send without stall. |

**Note:** AI Engine

`TLAST=1`

`TKEEP=0xF0`

AI Engine

`TLAST`

AI Engine

`TLAST`

AI Engine

```
writeincr(out,value,true);//"true" to assert TLAST
```
