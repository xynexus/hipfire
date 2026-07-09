---
title: "Implementation Details for Sharing LUTs"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Implementation-Details-for-Sharing-LUTs"
toc_id: bRSOZLyL9yxb7agFF85ipg
content_id: ACEn5jGbUhrTusxzWzmctA
---

### Implementation Details for Sharing LUTs

- A master kernel is designated to own the L1 memory location for the LUT.
- Unlike standard kernel designs, slave kernels—which reside on neighboring tiles—can directly access the L1 memory of the master core, reducing duplication of data storage.
- Usage: `adf::share_parameter(master_lut_parameter, {vector of slave_parameters} )` to define the shared parameters explicitly.
- Checks performed to ensure correct usage include the following:Master core has its neighboring slave kernels which can access its memory. Confirm that the same LUT array is used across the master and slave parameters Only constant LUTs are shareable. Attempting to share non-constant LUTs results in an error.

  - Master core has its neighboring slave kernels which can access its memory.
  - Confirm that the same LUT array is used across the master and slave parameters
  - Only constant LUTs are shareable. Attempting to share non-constant LUTs results in an error.
