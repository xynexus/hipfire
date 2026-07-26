---
title: "Scalar Iterators"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Scalar-Iterators"
toc_id: K4QNEb_dthlyHJk26Jj1yw
content_id: 18F_D2iPRhgXSWv4ZvoZlA
---

#### Scalar Iterators

- **Iterator `begin`:** A bidirectional scalar iterator used to return an iterator pointing to the first sample of the container.
- **Iterator `cbegin`:** A bidirectional scalar iterator that points to the first `const` content. `cbegin` cannot be used to modify the contents it points to.
- **Iterator `begin_circular`:** Same as iterator begin but is used for circular buffer port.
- **Iterator `cbegin_circular`:** Same as iterator `begin_circular` but points to `const` content.
- **Iterator `begin_random_circular`:** Same as `begin_circular` iterator and can be used to access samples at an arbitrary offset position relative to the sample they point to.
- **Iterator `cbegin_random_circular`:** Same as `begin_random_circular` iterator but cannot be used to modify the contents it points to.
