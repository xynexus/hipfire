---
title: "Free-Running AI Engine Kernel"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Free-Running-AI-Engine-Kernel"
toc_id: 89VAgoF89F4IOASz_7RHUQ
content_id: njP_HzajmbJhBWPB8~4w_w
---

## Free-Running AI Engine Kernel

The AI Engine kernel can be made to run continuously by using `graph::run(-1)`. This means the kernel restarts automatically after the last iteration is complete.

**Note:** `graph::run()`

AI Engine

`mygraph.run(3); mygraph.run();`

However, it requires input buffers and output buffers to be ready before it can start. Thus, there is a small overhead between kernel execution iterations. This section describes a method to construct a type of kernel that has zero overhead and runs forever. It is called a free-running AI Engine kernel.

Free-running kernels can only have streaming interfaces. Loops with infinite iterations can be inside the kernel. See the following example:

```
void free_running_aie(input_stream<int32> *in, output_stream<int32> *out) {
	while(true){//for(;;) is acceptable for C++
		int32 tmp=readincr(in);
		chess_separator_scheduler();//make sure stream is flushed
		writeincr(out,tmp+1);
		chess_separator_scheduler();//make sure stream is flushed
	}
}
```

The free-running kernel must have its own graph defined. This graph must have no other non-free-running kernels, because the graph never stops and non-free-running kernels lose control after being started. The graph containing the free-running kernel must be a top-level graph that can be connected to other graphs, or it can be connected to PLIO or GMIO. The following code sample shows a sample connection between the free-running graph and other graphs.

```
passingGraph mygraph;
freeGraph mygraph_free;
connect<> net0(mygraph.out1,mygraph_free.in);
connect<> net1(mygraph_free.out,mygraph.in2);
```

![yke1606893765249.png](../assets/058-01-yke1606893765249-png-9a10d892762a.png)

*Figure 1. Free-Running Graph Connection*

The free-running graph can be started using `mygraph_free.run(-1)` or automatically started after loading.
