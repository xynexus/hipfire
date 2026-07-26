---
title: "Data Movement Between AI Engines"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Data-Movement-Between-AI-Engines"
toc_id: rqqS15dckdyXYcxMRJ8PRA
content_id: 1klOKj0sgBGxE9BEZt_30Q
---

## Data Movement Between AI Engines

Generally, there are two methods to transfer data between kernels: buffer or stream.

#### Buffer

When using buffer, you can use ping-pong buffers or a single buffer. The AI Engine tools handle buffer synchronization between the kernels. Designers need to decide the buffer size and optionally buffer location between kernels when partitioning the application. If you need an overlap between different data buffers, AI Engine tools let you set a buffer margin. This margin automatically copies the overlapping data.

#### Stream

When using streams, data movement uses two input and two output stream ports, plus one dedicated cascade stream input port and one output port. Stream ports can provide 32-bit per cycle or, 128-bit per four cycles on each port. Stream interfaces are bidirectional and can read or write neighboring or non-neighboring AI Engines by stream ports. However, cascade stream ports are unidirectional and only provide a one-way access between the neighboring AI Engines.
