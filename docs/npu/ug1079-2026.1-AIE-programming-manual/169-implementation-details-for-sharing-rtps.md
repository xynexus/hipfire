---
title: "Implementation Details for Sharing RTPs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Implementation-Details-for-Sharing-RTPs"
toc_id: Kl6w6TU5bIM0Vbe2frq53A
content_id: sv8p_GU5KensN5TYFYnV0A
---

### Implementation Details for Sharing RTPs

- Only supports asynchronous input RTPs from the host, marked as "read-only."
- Synchronous RTPs, in/out RTPs, or output RTPs are not supported.
- The RTP memory in the master core is shared across neighboring slave cores.
- No locks (such as semaphore locks) are explicitly required from the user; the tool automatically removes locks when sharing is enabled.
- In the graph code, the `initial_value` constraint and update API can only be applied to the master RTP port.
- You must manage updates explicitly and ensure a valid initial value before running the graph, preventing garbage reads during initialization.
- Usage: `adf::share_parameter(master_rtp_port, {vector of slave_rtp_ports})` to define shared RTPs.
- Checks performed to ensure correct usage include the following:Verify placement of master and slave RTP buffers to ensure shared memory access between neighboring cores. An error occurs if sync RTPs, output RTPs, or in/out RTPs are used. Validate that master and slave RTPs have matching array sizes and data types.

  - Verify placement of master and slave RTP buffers to ensure shared memory access between neighboring cores.
  - An error occurs if sync RTPs, output RTPs, or in/out RTPs are used.
  - Validate that master and slave RTPs have matching array sizes and data types.
