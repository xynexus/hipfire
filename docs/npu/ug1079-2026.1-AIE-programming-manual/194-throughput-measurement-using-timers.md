---
title: "Throughput Measurement Using Timers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Throughput-Measurement-Using-Timers"
toc_id: LJIG6LdLL9nk~M1YxAug5w
content_id: I9tx7p3LrEdFtcSuf0kd1Q
---

### Throughput Measurement Using Timers

Throughput measurement using timers is one of the easiest way to measure the throughput at the AI Engine Array interface coarsely. The goal is to measure the time interval in between the input data transfer start and end.

- Open the device.
- Load the `xclbin` (bitstream for the PL and executables for the AIE array).
- Declare the source and destination buffers.
- Run the graph.
- Launch the graph.
- Launch input transfers.
- Launch output transfers.
- Wait for the end of input transfers.
- Wait for the end of output transfers.
- Wait for the end of the graph.
- Free all buffers.

When using timers to evaluate interface throughput, the reset of the timer should be as close as possible to the beginning of the transfer. Also, the timer stop should be as close as the effective end of the transfer. The best ordering of these actions depends on the throughput that you want to estimate.

For input transfers, timer fences should be just before input transfer launch and just after the wait of the end of the input transfers. If graph processing is very fast, you can launch output transfers before input ones.

- Launch the graph.
- Launch output transfers.
- Reset timer.
- Launch input transfers.
- Wait for the end of input transfers.
- Stop timer.
- Wait for the end of output transfers.

- Launch the graph.
- Launch output transfers.
- Reset timer.
- Launch input transfers.
- Wait for the end of output transfers.
- Stop timer.
- Wait for the end of output transfers (should be safe to skip this stage).

Once you have the time interval, knowing the amount of data allows you to compute directly the throughput of your system.
