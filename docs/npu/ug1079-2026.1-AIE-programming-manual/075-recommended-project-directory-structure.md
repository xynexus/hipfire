---
title: "Recommended Project Directory Structure"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Recommended-Project-Directory-Structure"
toc_id: hApEDCuRt274A_~ibeQ19g
content_id: xOOPs0OU1ksMw0sr2hCN3A
---

## Recommended Project Directory Structure

Use the following directory structure and coding practices to organize your AI Engine projects to provide clarity and reuse:

- Include all adaptive data flow (ADF) graph class definitions (all ADF graphs derived from graph class `adf::graph`), in a header file.Multiple ADF graph definitions can be in the same header file. Include the class header file in the `main` application file where the actual top-level graph is declared in the file scope.

  Multiple ADF graph definitions can be in the same header file. Include the class header file in the `main` application file where the actual top-level graph is declared in the file scope.
- Ensure there are no dependencies on the order that the header files are included. All header files must be self-contained and include all the other header files that they need.
- Ensure there are no file-scoped variables or data-structure definitions in the graph header files. Declare definitions (including `static`) in a separate source file. Ensure the source file can be identified in the `header` property of the kernel where they are referenced (see Lookup Tables).
- Only one source file is allowed to be specified as kernel source via `adf::source`. When sub-functions are defined as a library in separate source files (for example, `util.hpp` and `util.cpp`), the workaround is as follows:Include all the library source files in a header file (for example, `lib.hpp`):`#pragma once #include <util.hpp> #include <util.cpp>` Include the header file `lib.hpp` in the kernel source file.

  1. Include all the library source files in a header file (for example, `lib.hpp`):`#pragma once #include <util.hpp> #include <util.cpp>`

    ```
    #pragma once
    #include <util.hpp>
    #include <util.cpp>
    ```
  2. Include the header file `lib.hpp` in the kernel source file.

Related tasks

Creating a Data Flow Graph (Including Kernels)

Related reference

Lookup Tables
