---
title: "Stream-Based Access"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Stream-Based-Access"
toc_id: ML2wtwDlpMHTMm8uQLq~VQ
content_id: ga9LEcToCWZpvtP8BTpIAQ
---

### Stream-Based Access

With a stream-based access model, the kernels receive an input stream or an output stream of typed data as an argument. Each access to these streams is synchronized. That is, reads stall if the data is not available in the stream and writes stall if the stream is unable to accept new data.

AI Engine

`id=0`

`1`

`id=0`

`1`

AI Engine

AI Engine

Explicit Packet Switching

```
public:
  input_plio din;
  output_plio dout;
  adf::kernel k0,k1;
...
connect <stream> (din.out[0], k1.in[0]);
connect <stream> (k1.out[0], k2.in[0]);
connect <stream> (k2.out[0], dout.in[0]);
```

AI Engine

AI Engine

AI Engine

```
connect <cascade> (k1.out[1], k2.in[1]);
```

The AI Engine compiler automatically infers stream data structures from data flow graph connections. The structures are automatically declared in the wrapper code implementing the graph control. Kernel functions operate on pointers to stream data structures. These pointers are passed to the functions as arguments. There is no need to declare these stream data structures in data flow graph or kernel program.

##### Stream Connection for Multi-Rate Processing

`pktstream`

```
// constraint to specify samples per iteration for stream/pktstream ports to support multirate connections
constraint<uint32_t> samples_per_iteration(adf::port<adf::input>& p);
constraint<uint32_t> samples_per_iteration(adf::port<adf::output>& p);
```

`constraint`

`samples_per_iteration`

**Note:** `adf::samples_per_iteration (>0)`
