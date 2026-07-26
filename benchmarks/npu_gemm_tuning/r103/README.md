# R103: row-state-strided R100 tail

R103 keeps R100's compute unchanged and changes only the split-X input DMA
stride from 1,536 to 1,664 bytes. It reads the canonical 1,536-byte BF16 X
payload from each R101 row and skips the 128-byte inverse-state tail. No tensor
block reorder or nibble layout change occurs.
