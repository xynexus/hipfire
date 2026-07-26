---
title: "Streaming Data API"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Streaming-Data-API"
toc_id: MJn3J2fvIxGdPiq8OQdwLw
content_id: x5GUnZMiXyiPzLAyQTdvkw
---

# Streaming Data API

Data flow graph kernels operate on data streams that are infinitely long sequences of typed values. These data streams can be broken into separate blocks and these blocks are processed by a kernel. Kernels consume input blocks of data and produce output blocks of data. Kernels can also access the data streams in a sample-by-sample fashion. This chapter describes the data access API in these two cases.

**Note:** AI Engine

vector

- `aie::vector<uint8,16>`
- `aie::vector<uint8,32>`
- `aie::vector<uint8,64>`
- `aie::vector<uint8,128>`

scalar

- `unsigned char(uint8)`
- `unsigned short(uint16)`
- `unsigned int(uint32)`
- `unsigned long long(uint64)`
