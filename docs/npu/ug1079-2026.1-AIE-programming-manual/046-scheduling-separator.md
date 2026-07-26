---
title: "Scheduling Separator"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Scheduling-Separator"
toc_id: fn4bFZJ3kaomqrq5N1UPyA
content_id: LZI5ARWqldjyrlp_OIAg_g
---

## Scheduling Separator

The AI Engine compiler can reorder data movements and expressions in a basic block. If the compiler does not see a data dependence, or you want to insert a sequential point, use pragma `chess_separator_scheduler()` to separate code, as follows:

```
func(...){
  //code block 1
  chess_separator_scheduler();
  //code block 2
}
```

The `chess_separator_scheduler(N)` pragma is another form of the `chess_separator_scheduler()` pragma, where `N` indicates additional N cycles between two blocks. `N` can be positive or negative. With a negative offset, you can allow a partial overlap (up to absolute N cycles) between two blocks.

For example, the compiler does not have knowledge about dependence between the different streams of the kernel. The compiler can schedule different stream reads or writes in the same cycle. If a stream read or write stalls the kernel, it depends on an external source to provide or consume data for the kernel. In the following example code, the stream write (to `out`) and read (from `receive_back`) can be scheduled in the same cycle.

```
void producer(output_stream<int32> *out, input_stream<int32> *receive_back){
  int32 data;
  ...
  writeincr(out,data); //schedule in the same cycle
  readincr(receive_back); //schedule in the same cycle
  ...
}
```

This kernel stalls if there is no data read from stream `receive_back`. Thus, there is no data sent to stream `out`. If an external source must receive data from `producer` before sending data to `receive_back`, the kernel stalls and cannot be recovered. To schedule the stream operations in different cycles, the `chess_separator_scheduler(N)` can be added, as follows:

```
void producer(output_stream<int32> *out, input_stream<int32> *receive_back){
  int32 data;
  ...
  writeincr(out,data);
  //Make sure read occurs after write and data is sent out before a possible stall
  chess_separator_scheduler(1);
  readincr(receive_back);
  ...
}
```
