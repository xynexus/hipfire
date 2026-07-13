# R68 — overlapping padded joined staging

R68 retains R67's fast joined consumer order but reduces projection output
task count roughly threefold. Each core emits one padded 24-token x 32-column
BF16 object per slice. DMA places the four core-row objects three records apart:
the padding record at the end of one core object is deliberately overwritten by
the next core's first real record. A 37th 8-KiB record per role safely absorbs
the final padded write. This is mutable DMA placement; immutable HFP order and
local OQ4 nibble/lane handling are unchanged.

## Result

All 327,680 BF16 projection values pass bit-for-bit. Every preseeded cos/sin and
parameter byte is preserved, and all padding records remain zero. Three fresh
warmed projection processes measure 0.465281, 0.494605, and 1.067005 ms
(median 0.494605 ms), with one high outlier. Maximum projection core text is
9,280 bytes.

The 37-record joined pack consumer preserves Q cosine 0.99999121/max error
0.0078125, K cosine 0.99999156/max error 0.0078125, and bit-exact V. Three
fresh 100-command runs measure 0.3435, 0.3579, and 0.3627 ms (median 0.3579
ms). Maximum pack core text is 8,784 bytes.

Sequential medians total about 0.8525 ms before attention. This is 24% faster
than R67's 1.1182 ms and 42% faster than R65+R66's approximately 1.48 ms.
R69 should import one shared 1,517,568-byte stage BO into both NPU contexts,
preseed its position/parameter tails once, and measure the actual projection ->
pack chain without a host copy.

Durable rows:

- `../results/r68-w4-overlap-joined-stage-20260713.csv`
- `../results/r68-overlap-joined-stage-to-qkv-20260713.csv`
