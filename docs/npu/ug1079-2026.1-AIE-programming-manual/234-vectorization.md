---
title: "Vectorization"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vectorization"
toc_id: 4P5REeIpdTvKIAVFgPI7hA
content_id: ~0A5qTIHP~NMVclPkPEj7w
---

### Vectorization

The following shows the scalar reference code for this matrix multiplication example.

**Note:** The data is stored in columns.

```
void matmul_mat8_scalar(input_window_int16* matA,
		input_window_int16* matB,
		output_window_int16* matC){

	for(int i=0; i<M; i++){//M=64
		for(int j=0;j<L;j++){//L=2
			int temp = 0 ;
			for(int k=0; k<N; k++){//N=8
				temp += window_read(matA)*window_readincr(matB);//B is circular buffer, size N*L
				window_incr(matA,64); //Jump of 64 elements to access the next element of the same row
			}
			window_write(matC,(int16_t)(temp>>15)) ;
			window_incr(matC,64); //Jump to the next column
		}
		window_incr(matA,1); //Jump of one element for moving to the next row.
		window_incr(matC,1); //Jump to the next row
	}
}
```

As the previous example shows, the `mac16` intrinsic is the best choice for computing 16 lanes together This because 16 int16 from a column can be loaded at once. To compute 16 output data in a column, four `mac16` operations are needed. The same data in vector "a" is used twice to compute the data for two output columns. Thus, two columns of data can be loaded and two `mac16` used for accumulations to the two output columns. These two loads and two MACs are repeated four times to get the results of two output columns.

The following pseudo-code shows this method.

```
C_[0:15,0] = A_[0:15,0:1]*B_[0:1,0]
C_[0:15,1] = A_[0:15,0:1]*B_[0:1,1]

C_[0:15,0]+= A_[0:15,2:3]*B_[2:3,0]
C_[0:15,1]+= A_[0:15,2:3]*B_[2:3,1]

C_[0:15,0]+= A_[0:15,4:5]*B_[4:5,0]
C_[0:15,1]+= A_[0:15,4:5]*B_[4:5,1]

C_[0:15,0]+= A_[0:15,6:7]*B_[6:7,0]
C_[0:15,1]+= A_[0:15,6:7]*B_[6:7,1]
```

In the previous code, each "*" denotes a MAC operation. `C_[0:15,0]` and `C_[0:15,1]` denote two output columns that are accumulated separately. `A_[0:15,0:1]` denotes the column 0 and 1, and each column has 16 elements. `B_[0:1,0]` denotes column 0 with 2 elements. There will be a loop for the code in the real vectorized code because there are 64 output rows. The `mac16` intrinsic function to be used has the following interface.

```
v16acc48 mac16	(	v16acc48 	acc,
	v64int16 	xbuff,
	int 	xstart,
	unsigned int 	xoffsets,
	unsigned int 	xoffsets_hi,
	unsigned int 	xsquare,
	v16int16 	zbuff,
	int 	zstart,
	unsigned int 	zoffsets,
	unsigned int 	zoffsets_hi,
	int 	zstep
)
```

The buffers contain parameters (start, offsets, square, and step) to compute the indexing into buffers (vector registers). For details about the lane addressing scheme with these parameters, see MAC Intrinsics.

Note that the `mac16` intrinsic function prototype is different with the one introduced in the previous matrix vector multiplication example. The `xbuff` is `v64int16` which allows two sets of data to be stored and used in an interleaved way.

Coding with MAC intrinsics can be seen in the following section.
