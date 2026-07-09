---
title: "Software Pipelining of Loops"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Software-Pipelining-of-Loops"
toc_id: gvox_UgqeoCETbUDts8vIA
content_id: 25IeexO195totV~YCtYriw
---

## Software Pipelining of Loops

This section discusses software pipelining of loops, an important concept that enables the AI Engine to concurrently execute different parts of a program. For example, a loop that takes nine cycles per iteration is shown in the following figure. The figure illustrates sequential execution through full overlap pipelining.

![uww1569603231026.png](../assets/043-01-uww1569603231026-png-f4abb1a819c5.png)

*Figure 1. Pipelining Example*

Counting the cycles through each of the examples shows that to fully execute three loop iterations, the following is required:

- The sequential execution requires 27 cycles
- The partially overlapped pipeline requires 13 cycles
- The fully pipelined loop requires 11 cycles

From a performance perspective, it is therefore desirable to have a fully overlapping pipeline. However, this is not always possible, because resource constraints, as well as inter-iteration loop dependencies can prevent a full overlap (see the following figure).

![olq1606891959456.png](../assets/043-02-olq1606891959456-png-5577b9d12107.png)

*Figure 2. Dependencies in Pipelining*

In this example, the program performs load A (2 x 256-bit) in cycle 2, load B (2 x 256-bit) in cycle 3, and in cycle 6 and 7 it is executes operations on loop variable A. The remaining instructions of this iteration are of no importance with respect to the loop performance analysis.

Cycles 2 and 3 of this loop iteration execute 4 x 256-bit load operations. The four required loads execute in two cycles because AI Engines can perform only two loads per cycle. This is called a resource constraint. If the loop containing this iteration is supposed to be pipelined, this constraint limits the overlap to no less than two cycles.

Similarly, code dependencies between iterations shown in cycle 6 and 7 can prevent additional overlap. In this case, the next iteration of the loop requires the value of A to be updated before it can be used by the loop, limiting the overlap.

The AI Engine compiler reports on each loop in the following form.

**Note:** Work/aie/core_ID/core_ID.log

`-v`

```
HW do-loop #397 in "testbench.cc", line 132: (loop #16) :
Critical cycle of length 2 : b67 -> b68 -> b67
Minimum length due to resources: 2
Scheduling HW do-loop #397
(algo 1a)       -> # cycles: 9
(modulo)        -> # cycles: 2 ok (required budget ratio: 1)
(resume algo)     -> after folding:  2 (folded over 4 iterations)
  -> HW do-loop #397 in "testbench.cc", line 132: (loop #16) : 2 cycles
NOTICE: loop #397 contains folded negative edges
NOTICE: postamble created
Removing chess_separator blocks (all)
```

In the previously shown AI Engine compiler report, the `Critical cycle of length` provides feedback on code dependencies. The `Minimum length due to resources` indicates the minimum overlap required due to resource constraints. The `algo 1a` line states the total amount of cycles for a single iteration. Given these numbers, there are a maximum of five iterations active at a time creating the pipeline.

The AI Engine compiler reports these five overlapping iterations (the current iteration plus four folded iterations) in the `resume algo` line. In addition, the compiler states the initiation interval (II), the number of cycles a single iteration has to execute before the following iteration is started. In this example, the II is two.

In general, it is sufficient to provide the directive `chess_prepare_for_pipelining` to instruct the compiler to attempt software pipelining. When the number of loop iterations is a compile time constant, the chess compiler creates the optimum software pipeline.

For a dynamic loop range (defined by a variable start/end), the compiler requires additional information to create an effective pipeline loop structure. The directive `chess_loop_range(<minimum>, <maximum>)` performs this.

**Note:** If the number of cycles in the loop exceeds 64 cycles, pipelining can be disabled by the compiler for that loop. In this case, the compiler reports the following message.

```
(algo 1a)       -> # cycles: 167 (exceeds -k 64) -> no folding: 167
  -> HW do-loop #511 in "xxxx", line 794: (loop #8): 167 cycles
```

AI Engine

```
--Xchess="main:backend.mist2.maxfoldk=200"
```

**Note:** Important:

Vitis

`-verbose`

`v++ -c --mode aie --aie.verbose`

![xbi1695662492513.png](../assets/043-03-xbi1695662492513-png-9464b7f5eddb.png)

*Figure 3. Enabling Verbose Mode in Debug Reports*

You can generate a modulo scheduling report for modulo scheduled loops by specifying the option `-Xchess=main:backend.mist2.xargs=-ggraph` for the AI Engine compiler. This report is available for software pipelined loop with the name *_modulo.rpt in Work/aie/core_ID/Release/chesswork/<mangled_function_name>/*.rpt, where * is the block name. The modulo scheduling report also contains the information about register live ranges for register files. Live ranges can be useful for finding inefficiencies in register assignment and can be improved by using `chess_storage`.

After completing compilation and linking, you can open the compile log for an individual kernel in the Vitis IDE. For more information, see the AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)).

Related reference

Loops
