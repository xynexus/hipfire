# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""aiecost — predict AIE2/XDNA NPU kernel latency and overheads ahead of time.

Implements docs/npu/aie2-cost-model-plan.md. Best-estimate, not a simulator:
the goal is to rank candidate schedules and name the limiter, converging over
predict -> measure -> refit iterations.

Offline-first: nothing here reaches the network.
"""

__all__ = ["device", "env"]
