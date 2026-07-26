---
title: "AI Engine-to-PL Rate Matching"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/AI-Engine-to-PL-Rate-Matching"
toc_id: Yu1kbtIw5yvHkJ2BlloIxw
content_id: R5H5s6rUNQkuJoI_Z9keOg
---

### AI Engine-to-PL Rate Matching

The AI Engine runs at 1 GHz (or more, depending on the speed grade). Each interface channel can read or write with a 64-bit data width per cycle. In contrast, a PL kernel can run at 500 MHz (half the frequency of the AI Engine), while consuming a larger bit width. Rate matching balances throughput between the producer and the consumer. It ensures that neither of the processes creates a bottleneck with respect to the total performance. The following equation shows the rate matching for each channel:

AI Engine

AI Engine

The following table shows a PL rate matching example. The example is a 32-bit channel written to each cycle by the AI Engine at 1 GHz for -1L speed grade devices. As shown, the PL IP must consume twice the data at half the frequency, or four times the data at one quarter of the frequency.

| AI Engine | PL |  |  |
| --- | --- | --- | --- |
| Frequency | Data per Cycle | Frequency | Data per Cycle |
| 1 GHz | 32-bit | 500 MHz | 64-bit |
|  |  | 250 MHz | 128-bit |

Because the Vitis compiler (`v++`) understands the need to match frequency and adjust data-path width, it automatically:

- Extracts the port width from the PL kernel
- Reads the frequency from the clock specification
- Introduces an upsizer/downsizer to temporarily store the data exchanged between the AI Engine and the PL regions to manage the rate match.

To avoid deadlocks, ensure that if multiple channels are read or written between the AI Engine and the PL, the data rate per cycle is concurrently achieved on all channels. For example, if one channel requires 32 bits, and the second 64 bits, the AI Engine code must ensure that both channels are written adequately to avoid back pressure or starvation on the channel. Additionally, to avoid deadlock, writing/reading from the AI Engine and reading/writing in the PL code must follow the same chronological order.

The number of interfaces used in the graph function definition for the PL defines the number of AXI4-Stream interfaces. Each argument results in the creation of a separate stream.
