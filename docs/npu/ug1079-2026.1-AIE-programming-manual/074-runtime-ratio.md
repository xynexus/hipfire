---
title: "Runtime Ratio"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Runtime-Ratio"
toc_id: gj_VwuCLpzUmUJpw7kB81A
content_id: np3tVav6bPAJn4x~KeDqQA
---

### Runtime Ratio

The runtime ratio is a user-specified constraint that allows the AI Engine tools the flexibility to put multiple AI Engine kernels into a single AI Engine, if their total runtime ratio is less than one. The runtime ratio of a kernel is calculated using the following equation.

The cycle budget is the number of cycles allowed to run one invocation of the kernel, which depends on the system throughput requirement.

You can estimate cycles for one run of the kernel in the initial design stage. For example, if the kernel contains a loop that can be well pipelined, and each cycle is capable of handling that amount of data, then the cycles for one run of the kernel can be estimated by the following.

**Note:** synchronization of synchronous buffers + function initialization

You can also profile the cycles in the `aiesimulator`. This applies when vectorized code is available.

If multiple AI Engine kernels are in a single AI Engine, they run sequentially. Each runs one time per `graph::run` iteration, unless multi-rate processing is used. The following points explain what this means:

- If the AI Engine runtime percentage (specified by the runtime constraint) is allocated for the kernel in each iteration of `graph::run` (or on an average basis, depending on the system requirement), the kernel performance requirement can be met.
- For a single iteration of `graph::run`, the kernel takes no more percentage than that specified by the runtime constraint. Otherwise, it can affect the performance of other kernels in the same AI Engine.
- Even if multiple kernels have a summarized runtime ratio less than one, they are not necessarily put into a single AI Engine. The mapping of an AI Engine kernel into an AI Engine is also affected by hardware resources. For example, there must be enough program memory to allow the kernels to be in the same AI Engine. Also, stream interfaces must be available to allow all the kernels to be in the same AI Engine.
- You can save resources by placing multiple kernels in the same AI Engine. For example, the buffers between the kernels in the same AI Engine are single buffers instead of ping-pong buffers.
- Increasing the runtime ratio of a kernel does not necessarily increase the performance of the kernel or graph. This is because performance is affected by the data availability to the kernel and the data throughput in and out of the graph. A pessimistically high runtime ratio setting might result in inefficient resource utilization.
- Low runtime ratio does not necessarily limit the performance of the kernel to the specified percentage of the AI Engine. For example, the kernel can run immediately when all the data is available if there is only one kernel in the AI Engine. This true regardless of the runtime ratio set.
- You cannot place kernels from different top-level graphs in the same AI Engine. The graph API must control each graph independently.
- Set the runtime ratio as accurately as possible. It affects both the AI Engine used and the data communication routes between kernels. It might also affect other design flows, for example, the power estimation.
