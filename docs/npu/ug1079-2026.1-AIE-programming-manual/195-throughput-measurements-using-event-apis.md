---
title: "Throughput Measurements Using Event APIs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Throughput-Measurements-Using-Event-APIs"
toc_id: BkTfc_J9tnjG5X95cqIlig
content_id: MNro0iFlvC8Y5Yh1CdPa3w
---

### Throughput Measurements Using Event APIs

This method is more precise than the previous one. You use the events to automatically count the clock ticks in between the start of the transfers to the time the interface is idle. So, you do not have to precisely position the `start_profiling` and the `stop_profiling` instruction. You only have to set them before the graph run and input data transfer for the start, and after the graph ends for the end.

Knowing the amount of data that has been transferred allows you to calculate the average throughput of the system more easily.
