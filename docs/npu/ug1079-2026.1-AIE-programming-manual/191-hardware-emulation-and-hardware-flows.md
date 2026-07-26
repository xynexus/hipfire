---
title: "Hardware Emulation and Hardware Flows"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Hardware-Emulation-and-Hardware-Flows"
toc_id: PrlTdCdQ6gjSmvQR0Tye3w
content_id: ~aZvkX4~8TPn36rYeuQcKg
---

### Hardware Emulation and Hardware Flows

You can use `input_gmio` and `output_gmio` not only with the AI Engine simulator. They also work in hardware emulation and full hardware flows. To enable in hardware emulation and hardware flows, add the following code to graph.cpp.

```
#if !defined(__AIESIM__) && !defined(__X86SIM__)
    #include "adf/adf_api/XRTConfig.h"
    #include "experimental/xrt_kernel.h"
    // Create XRT device handle for ADF API

    char* xclbinFilename = argv[1];
    auto dhdl = xrtDeviceOpen(0);//device index=0
    xrtDeviceLoadXclbinFile(dhdl,xclbinFilename);
    xuid_t uuid;
    xrtDeviceGetXclbinUUID(dhdl, uuid);

    adf::registerXRT(dhdl, uuid);
#endif
```

Using the guard macro __AIESIM__ and __X86SIM__, the same version of graph.cpp can work for the AI Engine simulator, x86simulator, hardware emulation, and hardware flows. Place the preceding code before calling the graph or the GMIO ADF APIs. At the end of the program, close the device using the `xrtDeviceClose()` API.

```
#if !defined(__AIESIM__) && !defined(__X86SIM__)
    xrtDeviceClose(dhdl);
#endif
```

To compile the code for hardware flow, see [Programming the PS Host Application](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment&resourceid=ykt1590616160037.html) in AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)).
