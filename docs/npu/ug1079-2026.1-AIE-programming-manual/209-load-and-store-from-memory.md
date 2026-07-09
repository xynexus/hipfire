---
title: "Load and Store From Memory"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Load-and-Store-From-Memory"
toc_id: d9eYNVggFyk22L_FX5A~aw
content_id: v2ubj0sqDudpn0Mm_IbMmA
---

#### Load and Store From Memory

AI Engine APIs let you read and write data from data memory, streaming data ports, and cascade streaming ports used by AI Engine kernels. For additional details on streaming APIs, see Streaming Data API. In the following example, the window `readincr` (`window_readincr_v8(din)`) API reads a window of complex int16 data into the data vector. Similarly, `readincr_v8(cin)` reads a sample of int16 data from the `cin` stream. `writeincr (cas_out, v)` writes data to a cascade stream output.

```
void func(input_window_cint16 *din,
			input_stream_int16 *cin,
			output_cascade<cacc48> *cas_out){
	v8cint16 data=window_readincr_v8(din);
	v8int16 coef=readincr_v8(cin);
	v4cacc48 v;
	…
	writeincr(cas_out, v);
}
```
