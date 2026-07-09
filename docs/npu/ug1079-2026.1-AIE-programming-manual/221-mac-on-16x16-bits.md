---
title: "MAC on 16x16 bits"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/MAC-on-16x16-bits"
toc_id: h1BUKvXojl~n_hZSwcKhWw
content_id: 2I2Oly6ARmu6hsYo9MBfMw
---

#### MAC on 16x16 bits

An example of MAC with int16 `X` buffer and int16 `Z` buffer is as follows. Note that the permute granularity for `X` buffer is 32 bits. The `start` and `step` parameters are always in terms of data type granularity. To get to a 16-bit index, you need to multiply them by 2.

The `xoffsets` parameter comes as a pair. The first hex value is an absolute 32 bits offset and picks up 2 x 16 bits values (index, index+1) in the even row. The second hex value offsets from the first hex value plus 1, using a 32-bit offset. It selects two 16-bit values from the odd row. So the hex value `0x24` in `xoffsets` selects index 8, 9 for even row and index 14, 15 for odd row from `xbuff`:

```
even: 2 * 4 -> get indices [8, 9]
odd: 2 * ( 2 + 4 + 1 ) -> get indices [14, 15]
```

Similarly, the hex value `0x00` in `xoffsets` selects index 0, 1 for even row and index 2, 3 for odd row from `xbuff`.

There is another `xsquare` parameter to perform 16 bits granularity twiddling after the main permute. It gives additional contribution to the index in a 2 by 2 matrix recurring across the 8x4 matrix compute given by MUL8 in int16 x int16 mode.

For example, `xsquare` value `0x2103` (from lower to higher hex value) puts index `3, 0` in the even row and index `1, 2` in the odd row. How the `xsquare` parameter works can be seen in the center of the following figure.

![hwb1606892732289.png](../assets/221-01-hwb1606892732289-png-26e02934ff7e.png)

*Figure 1. MAC8 on int16 x int16 Type*

The following figure is an example of `mac16` intrinsic of int16 and int16.

![qtb1611096574633.png](../assets/221-02-qtb1611096574633-png-3c1a959274a5.png)

*Figure 2. MAC16 on int16 x int16 Type*
