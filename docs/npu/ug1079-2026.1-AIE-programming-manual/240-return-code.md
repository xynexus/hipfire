---
title: "Return Code"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Return-Code"
toc_id: 8Ri1c~t60j6ky0vD1ppFVQ
content_id: wd2Vy7cs0YbNmWznPWJhYA
---

## Return Code

ADF APIs have defined return codes to indicate success or different kinds of failures in the `adf` namespace.

```
enum return_code
  {
    ok = 0,
    user_error,
    aie_driver_error,
    xrt_error,
    internal_error,
    unsupported
  };
```

The following defines the different return codes:

- **ok:** success
- **user error:** Possible invalid argument or use of the API in an unsupported way
- **aie_driver_error:** If the AI Engine driver returns errors, the graph API returns this error code.
- **xrt_error:** If XRT returns errors, the graph API returns this error code.
- **internal error:** This means something is wrong with the tool; contact AMD support.
- **unsupported:** Unsupported feature or unsupported scenario.
