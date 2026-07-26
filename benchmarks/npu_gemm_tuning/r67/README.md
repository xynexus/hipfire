# R67 — joined mutable QKV staging

R67 changes only the mutable projection-to-pack boundary. The compact W4
`.rdna2.hfp`, R15 compute, local nibble/lane handling, and canonical Q/K/V
pack math remain unchanged.

The projection emits 8-token x 32-dimension BF16 tiles. Four core rows join
into one DMA object, and each role reserves 36 8-KiB records: 32 records for
M256 plus four padding records. Cos/sin occupies the second 4 KiB of every
record; one 2-KiB parameter tail follows the five roles. This makes the four
records needed by each R28 pack row contiguous, so one joined input activates
all four core pairs concurrently.

## Result

Three locked fresh projection processes pass all 327,680 BF16 values
bit-for-bit, preserve every preseeded cos/sin/parameter byte, and leave all
padding values zero. Projection median is 0.751200 ms (0.725232-1.152343 ms),
with one high outlier. Maximum projection core text is 12,224 bytes.

The joined pack consumer preserves the established oracle: Q cosine
0.99999121/max 0.0078125, K cosine 0.99999156/max 0.0078125, and bit-exact V.
Three fresh 100-command processes measure 0.3517, 0.3670, and 0.3687 ms
(median 0.3670 ms), recovering R28 performance. Maximum pack core text is
8,784 bytes.

Sequential medians total about 1.1182 ms before attention. This improves on
R65+R66's roughly 1.48 ms but is not yet admitted for resident integration:
the projection uses approximately 360 small output DMA tasks. R68 should emit
one padded 24-token x 32-dimension object per slice and deliberately overlap
padding records, cutting producer task count about threefold while retaining
the same joined consumer order.

Durable rows:

- `../results/r67-w4-joined-stage-20260713.csv`
- `../results/r67-joined-stage-to-qkv-20260713.csv`
