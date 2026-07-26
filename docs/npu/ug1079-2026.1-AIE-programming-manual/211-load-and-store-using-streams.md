---
title: "Load and Store Using Streams"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Load-and-Store-Using-Streams"
toc_id: XOihZ5Ja5CTnk~2AU2hCIQ
content_id: iPMNJUsAnKBqrZ~2Ifvxjw
---

#### Load and Store Using Streams

You can load from or store vector data in streams as shown in the following example.

```
void func(input_stream_int32 *s0, input_stream_int32 *s1, …){
	for(…){
		data0=readincr(s0);
		data1=readincr(s1);
		…
	}
}
```

For more information about streaming data APIs, see Streaming Data API.
