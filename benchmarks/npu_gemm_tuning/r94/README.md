# R94: vector inverse-AWQ FFN preparation

R94 preserves R93's canonical-BF16 to R25 activation byte contract while
testing the program-capacity lever needed for first-stage fusion. The loader
packs the same F32 AWQ/sign arrays already carried by R25 weight records. AIE2P
then uses BF16-to-F32 vector conversion, vector division, and vector sign
multiplication. This removes scalar soft-float helpers and makes the preparation
ABI directly reusable by the first gate/up stage without any tensor-block
reorder.

The existing kernel parameter remains the platform correctness workaround.
Tile-memory use is an independent capacity/performance choice.

```bash
./build_r94.sh
```
