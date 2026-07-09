---
title: "Loops"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Loops"
toc_id: xpWshew5QCx_yN7qw62ONA
content_id: yqrRa7KuyrYf8MLYi6TunQ
---

## Loops

The AI Engine has a zero-overhead loop structure that does not incur any branch control overhead for comparison and branching. This design reduces the inner loop cycle count. Pipelining allows the compiler to add pre-amble and post-amble so that the instruction pipeline is always full during loop execution. With a pipelined loop, you can start a new iteration before the previous one ends to achieve higher throughput.

The following figure shows the assembly code of a zero-overhead loop. The following are shown in order in different slots:

- Two vector loads
- One vector store
- One scalar instruction
- Two data moves
- One vector instruction

![sem1610658284502.png](../assets/041-01-sem1610658284502-png-eb9dec85bb29.png)

*Figure 1. Assembly Code of Zero-Overhead Loop*

The following pragmas tell the compiler to pipeline the loop and indicate that the loop always runs at least three times.

```
for (int i=0; i<N; i+=2)
   chess_prepare_for_pipelining
   chess_loop_range(3,)
```

`chess_loop_range(<minimum>, <maximum>)` tells the compiler that the loop runs at least `<minimum>` times and at most `<maximum>` times. Both values are non-negative constant expressions, or can be omitted.

When omitted, `<minimum>` defaults to 0, and `<maximum>` defaults to the maximum preset in the compiler. While `<maximum>` is not relevant for the pipeline implementation, `<minimum>` guides the pipeline implementation.

The `<minimum>` value specifies the minimum number of loop iterations each time the loop runs. The software pipeline is then tuned to allow at least that many iterations to execute in parallel if possible. The software pipeline also determines that checking the boundaries for the loop is not necessary before the `<minimum>` number of iterations are executed.

You do not need the loop range pragma if the loop range is a compile-time constant. In general, the AI Engine compiler reports the theoretical number best suited for optimum pipelining of an algorithm. If the range specification is not optimal, the compiler issues a warning and suggests the optimal range. Towards that end, you can initially set the `<minimum>` to one `[chess_loop_range(1,)]` and observe the theoretical best suited `<minimum>` being reported by the compiler.

```
Warning in "matmul_vec16.cc", line 10: (loop #39)
further loop software pipelining (to 4 cycles) is feasible with `chess_prepare_for_pipelining'
but requires a minimum loop count of 3
... consider annotating the loop with `chess_loop_range(3,)' if applicable,
... or remove the current `chess_loop_range(1,)` pragma
```

At this point, you can choose to update the `<minimum>` number to the reported optimum.

This second part of the pipeline implementation can be a reason for potential deadlocks in the AI Engine kernels if the actual `<minimum>` number of iterations is not reached. For this reason, you must ensure that the number of iterations is always at least the number specified in the `chess_loop_range` directive.

Loop carried dependencies impact the vectorization of code. If you cannot remove an inner loop dependency, step out one level and manually unroll the loop. This creates multiple inner loop copies running in parallel.

Try to avoid sequential load operations to fill a vector register completely before use. Interleave loads with `aie::sliding_mul` functions, where the MAC and loads can be done in the same cycle.

```
buff.insert(3,readincr_v<4>(sig_in));
acc = aie::sliding_mul<4,8>(coe,0,buff,4);
writeincr(cascadeout,acc);
```

In certain use cases loop rotation, which rotates the instructions inside the loop, can be beneficial. Instead of loading data into a vector at the start of a loop, load a block for the first iteration before the loop. Then, for the next iteration, load the next block near the end of the loop. Loading the next block near the end of the loop adds additional instructions but shortens the dependency length of the loop which helps to achieve an ideal loop with a potentially lower loop range.

```
// Load starting data for first iteration
aie::vector<cint16,16> buff = delay_line;

for (unsigned int i = 0; i < LSIZE; ++i)
chess_prepare_for_pipelining
chess_loop_range(4,)
{
  //template <unsigned Lanes, unsigned Points, int CoeffStep = 1, int DataStepX = 1, int DataStepY = DataStepX, AccumElemBaseType AccumTag = accauto, VectorOrOp VecCoeff = void, VectorOrOp VecData = void>
  //auto sliding_mul (const VecCoeff &coeff, unsigned coeff_start, const VecData &data, unsigned data_start)
  buff.insert(2,readincr_v<4>(sig_in));
  acc = aie::sliding_mul<4,8>(coe,0,buff,0);
  writeincr(cascadeout,acc);

  buff.insert(3,readincr_v<4>(sig_in));
  acc = aie::sliding_mul<4,8>(coe,0,buff,4);
  writeincr(cascadeout,acc);

  buff.insert(0,readincr_v<4>(sig_in));
  acc = aie::sliding_mul<4,8>(coe,0,buff,8);
  writeincr(cascadeout,acc);

  buff.insert(1,readincr_v<4>(sig_in));
  acc = aie::sliding_mul<4,8>(coe,0,buff,12);
  writeincr(cascadeout,acc);
}
```
