---
title: "Rounding and Saturation Modes"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Rounding-and-Saturation-Modes"
toc_id: 0kMYiXoktYwFk38vd08pmg
content_id: ASGG03DL6dzJ3tuTl~QxlQ
---

### Rounding and Saturation Modes

The `aie::set_rounding()` and `aie::set_saturation()` APIs are used to set the rounding and saturation modes of the `to_vector` operation.

The `aie::rounding_mode` includes the following:

- **`floor`:** Truncate LSB, always round down (towards negative infinity). Default.
- **`ceil`:** Always round up (towards positive infinity).
- **`positive_inf`:** Rounds halfway towards positive infinity.
- **`negative_inf`:** Rounds halfway towards negative infinity.
- **`symmetric_inf`:** Rounds halfway towards infinity (away from zero).
- **`symmetric_zero`:** Rounds halfway towards zero (away from infinity).
- **`conv_even`:** Rounds halfway towards nearest even number.
- **`conv_odd`:** Rounds halfway towards nearest odd number.

The `aie::saturation_mode` includes the following:

- **`none`:** No saturation is performed and the value is truncated on the MSB side. Default.
- **`saturate`:** Controls saturation. Saturation rounds an n-bit signed value in the range [- ( 2(n-1) ) : + 2(n-1) - 1 ]. For example if n=8, the range would be [-128:127].
- **`symmetric`:** Controls symmetric saturation. Symmetric saturation rounds an n-bit signed value in the range [- ( 2(n-1) -1 ) : + 2(n-1) - 1 ]. For example if n=8, the range would be [-127:127].

The rounding and saturation settings are applied on the AI Engine tile. The `aie::get_rounding` and `aie::get_saturation` methods can be used to get the current rounding and saturation modes.

```
aie::set_rounding(aie::rounding_mode::ceil);
aie::set_saturation(aie::saturation_mode::saturate);
aie::rounding_mode current_rnd=aie::get_rounding();
aie::saturation_mode current_sat=aie::get_saturation();
...
aie::set_rounding(aie::rounding_mode::floor);
aie::set_saturation(aie::saturation_mode::none);
```

**Note:** AI Engine
