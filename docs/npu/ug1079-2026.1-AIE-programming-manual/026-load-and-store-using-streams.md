---
title: "Load and Store Using Streams"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Load-and-Store-Using-Streams"
toc_id: 58fTdP2P8kGTYlL3E8jSmw
content_id: eIDLg5YuXaYxblnO8Hi5cQ
---

#### Load and Store Using Streams

You can load from or store vector data in streams, as shown in the following example.

```
void func(input_stream<int32> *s0, …){
	for(…){
		int32 data0=readincr(s0); //32 bits load
		aie::vector<int32,4> data1=readincr_v<4>(s0); //128 bits load
		…
	}
}
```
