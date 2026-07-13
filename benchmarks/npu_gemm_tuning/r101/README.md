# R101: direct-X row-state attention boundary

R101 keeps the R44 direct architectural-X output and appends the pre-FFN
inverse RMS already computed by the attention tail to each canonical token row.
Each row is 1,664 bytes: 1,536 bytes of BF16 X followed by one 128-byte state
record whose first four bytes hold the F32 inverse. Tensor columns and token
order are unchanged; only mutable output DMA stride and metadata placement
change.

The row state lets the following native-W4 FFN apply pre-FFN normalization
inline while R100 consumes the same unmodified X payload for the residual tail.
The R15 rounding/saturation settings remain numerical controls. The separately
added kernel parameter remains the platform-issue workaround; neither is an
LDS-avoidance requirement.

## Result

The literal row-state relay is rejected. Scattering sixteen inverse values
into 128-byte records grows the odd attention cores to 16,444 bytes. Moving
the scatter into shim DMA fits at 16,380 bytes, but the hardware oracle reads
misaddressed state (including negative inverse values) and collapses layer-0
cosine to 0.50248530. Moving the metadata object onto the even normalized-X
output channel also fits (16,268 bytes), but that extra object reaches the
independent four-second command timeout and produces invalid state.

None of these failures changes the platform-workaround kernel parameter.
R15's `rounding=floor` and `saturation=none` numerical settings also remain
required. The failures are respectively program
capacity, DMA addressing, and output-channel scheduling results. R101 is not
selected by the reusable runtime.
