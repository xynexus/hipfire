---
title: "Vectorization"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vectorization"
toc_id: ykLkTTRkNSonrgmmq82Elg
content_id: CLA5sFpiaw1E1K9QOPx_eg
---

### Vectorization

For a complicated vector processing algorithm, start with a scalar version. This is useful as a golden reference for verifying accuracy. The scalar version for matrix multiplication is shown as follows.

```
void matmul_scalar(input_window_int16* matA,
      input_window_int16* matB,
      output_window_int16* matC){ //A[M,N], B[N,1], C[M,1]. M=64, N=16
    for(int i=0; i<M; i++){
        int temp = 0 ;
        for(int j=0; j<N; j++){
            temp += window_read(matA)*window_readincr(matB) ;
            window_incr(matA,64); //Jump of 64 elements to access the next element of the same row
        }
        window_writeincr(matC,(int16_t)(temp>>15)) ;
        window_incr(matA,1); //Jump of one element for moving to the next row.
    }
}
```

Note that in the previous code stores `matA` in the column base and `matB` is a circular buffer to the kernel. It can be read continuously by `window_readincr` for computing different rows of output because it will loop back to the start of the buffer.

There are total 64 outputs (M=64), and each output needs 16 (N=16) multiplications. When choosing MAC intrinsics to do vector processing, for the data type int16 * int16, select lane 4, 8, 16 to do the equation. The following figure illustrates these.

![tcg1606885499457.png](../assets/230-01-tcg1606885499457-png-c319dc1f4e34.png)

*Figure 1. Lane Selection*

The main difference between 4, 8, and 16 lanes MAC is data consumption. If you assume that the data is stored by column, then 16 lanes MAC might be the best option, because only two parts of continuous data needs to be loaded for the MAC operation, – a0 to a15 and a64 to a79. a0 to a15 are 256 bits, which allows one load to load the value into vector register.

To allow two loads to occur at the same cycle, a0 to a15 and a64 to a79 are required to be in separate data banks. The data needs to be divided column by column into two separate buffers to the kernel. That is to say:

- a0 to a63 are in the first buffer,
- a64 to a127 are in the second buffer,
- a128 to a191 are in the first buffer again, and so on.

By vectorization, the matrix multiplication can have a loop with 64/16=4 iterations and each iteration of the loop contains eight MAC operations. Every iteration of the loop produces 16 output data. The following figure illustrates this.

![beu1606886002537.png](../assets/230-02-beu1606886002537-png-943d1b7fcef9.png)

*Figure 2. Vectorization*

The mac16() intrinsic function to be used has the following interface.

```
v16acc48 mac16( v16acc48 acc,
    v32int16 xbuff,
    int xstart,
    unsigned int xoffsets,
    unsigned int xoffsets_hi,
    unsigned int xsquare,
    v16int16 zbuff,
    int zstart,
    unsigned int zoffsets,
    unsigned int zoffsets_hi,
    int zstep
)
```

The buffers contain parameters (start, offsets, square, and step) to compute the indexing into the buffers (vector registers). For details about the lane addressing scheme with these parameters, see MAC Intrinsics.

The following section covers coding with MAC intrinsics.
