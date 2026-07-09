---
title: "Linear and Circular Addressing of Buffer Ports"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Linear-and-Circular-Addressing-of-Buffer-Ports"
toc_id: L5YVOgE0MZbdQmeLyjpvSg
content_id: QckMyf58aYRYYd_KPpVGQQ
---

### Linear and Circular Addressing of Buffer Ports

Buffer ports can be addressed in a linear or circular manner. In linear addressing mode, data is addressed linearly and has no wrap around. In circular addressing mode, data is addressed circularly and has wrap around. Kernels with margin in the same tile are recommended to use circular buffer ports. Port connections with non-zero margin require copying the margin data at the consumer port after the consumer kernel returns. However, if the producer and consumer are on the same core and you use circular buffer ports, then margin copy can be avoided, which improves performance. Data in a regular buffer port can be accessed with a pointer obtained from the `data()` API, or using linear or circular iterators. Linear iterators have lower overhead than circular iterators.

**Note:** - Circular addressing mode is supported only in one dimension buffer ports.
- The iterator to access the data in a circular buffer port must be a circular iterator.
- Circular buffer port can only be declared in the function prototype.
