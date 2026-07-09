---
title: "Vector Iterators"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vector-Iterators"
toc_id: vi7mY6USlGJThXZqow4dFg
content_id: EviI6qLZQp5D_PBQ9mRQBQ
---

#### Vector Iterators

- **Iterator `begin_vector`:** A bidirectional vector iterator used to return an iterator pointing to the first vector sample of the container.
- **Iterator `cbegin_vector`:** A bidirectional vector iterator that points to `const` content. You cannot use this iterator to modify the content it points to.
- **Iterator `begin_vector_circular`:** Same as iterator `begin_vector` but is used for circular buffer port.
- **Iterator `cbegin_vector_circular`:** Same as iterator `begin_vector_circular` but points to ‘const’ content.
- **Iterator `begin_vector_random_circular`:** Same as iterator `begin_vector_circular` and can be used to access samples at an arbitrary offset position relative to the sample they point to. The arbitrary offset is in unit of configured vector size.
- **Iterator `cbegin_vector_random_circular`:** Same as iterator `begin_vector_random_circular` but cannot be used to modify the contents it points to.

| Iterator API | Description |
| --- | --- |
| auto begin_vector | Random scalar iterator; not supported for circular buffer ports. |
| auto cbegin_vector | Same as begin_vector but only for read access. |
| auto begin_vector_circular | Forward scalar circular iterator. |
| auto cbegin_vector_circular | Same as begin_vector_circular but only for read access. |
| auto begin_vector_random_circular | Random scalar circular iterator. |
| auto cbegin_vector_random_circular | Same as begin_vector_random_circular but only for read access. |
