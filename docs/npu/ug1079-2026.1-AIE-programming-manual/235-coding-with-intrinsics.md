---
title: "Coding with Intrinsics"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Coding-with-Intrinsics"
toc_id: cj~xF5E5PCu0AbE6JrAaQQ
content_id: lw1qRXFjQaFgxpzVCU3XyA
---

### Coding with Intrinsics

After analyzing how the function maps into the AI Engine vector processor, review the vectorized code.

```
void matmul_mat8(input_window_int16* matA,
		input_window_int16* matB,
		output_window_int16* matC){

	v16int16 buf_matB = window_read_v16(matB);

	v64int16 buf_matA = undef_v64int16();
	buf_matA=upd_w(buf_matA,0,window_read_v16(matA));
	window_incr(matA,64);
	buf_matA=upd_w(buf_matA,1,window_read_v16(matA));
	window_incr(matA,64);

	for (unsigned int i=0;i<M/16;i++)  //M=64, Each iteration computes 16 outputs
	chess_prepare_for_pipelining
	chess_loop_range(4,)
	{
		v16acc48 acc0=null_v16acc48();//For first output column
		v16acc48 acc1=null_v16acc48();//For second output column

		acc0 = mac16(acc0,buf_matA,0,0x73727170,0x77767574,0x3120,buf_matB,0,0x0,0x0,1);
		buf_matA=upd_w(buf_matA,2,window_read_v16(matA));
		window_incr(matA,64);
		acc1 = mac16(acc1,buf_matA,0,0x73727170,0x77767574,0x3120,buf_matB,8,0x0,0x0,1);
		buf_matA=upd_w(buf_matA,3,window_read_v16(matA));
		window_incr(matA,64);

		acc0 = mac16(acc0,buf_matA,32,0x73727170,0x77767574,0x3120,buf_matB,2,0x0,0x0,1);
		buf_matA=upd_w(buf_matA,0,window_read_v16(matA));
		window_incr(matA,64);
		acc1 = mac16(acc1,buf_matA,32,0x73727170,0x77767574,0x3120,buf_matB,10,0x0,0x0,1);
		buf_matA=upd_w(buf_matA,1,window_read_v16(matA));
		window_incr(matA,64);

		acc0 = mac16(acc0,buf_matA,0,0x73727170,0x77767574,0x3120,buf_matB,4,0x0,0x0,1);
		buf_matA=upd_w(buf_matA,2,window_read_v16(matA));
		window_incr(matA,64);
		acc1 = mac16(acc1,buf_matA,0,0x73727170,0x77767574,0x3120,buf_matB,12,0x0,0x0,1);
		buf_matA=upd_w(buf_matA,3,window_read_v16(matA));
		window_incr(matA,80);//point to next 16 rows

		acc0 = mac16(acc0,buf_matA,32,0x73727170,0x77767574,0x3120,buf_matB,6,0x0,0x0,1);
		window_write(matC,srs(acc0,15));
		window_incr(matC,64);
		buf_matA=upd_w(buf_matA,0,window_read_v16(matA));
		window_incr(matA,64);
		acc1 = mac16(acc1,buf_matA,32,0x73727170,0x77767574,0x3120,buf_matB,14,0x0,0x0,1);
		window_write(matC,srs(acc1,15));
		window_incr(matC,80);//point to next 16 rows
		buf_matA=upd_w(buf_matA,1,window_read_v16(matA));
		window_incr(matA,64);
	}
}
```

In the previous code, `buf_matB` is for matrix B and it is loaded outside the loop. `buf_matA` is for matrix A and two sets of A are stored in lower and higher parts. When `mac16` has the value "0" for `xstart`, the lower part of `buf_matA` is used. When `mac16` has the value "32" for `xstart`, the higher part of `buf_matA` is used. `acc0` and `acc1` are the accumulated values for two output columns.

Note that `buf_matA` is preloaded before the loop. In the loop, the loads with window buffer pointer increment, MAC operations and the stores are interleaved. To understand how the `mac16()` intrinsic works, refer to MAC Intrinsics. The assembled code for the loop is as shown in following figure.

![vwu1611096394838.png](../assets/235-01-vwu1611096394838-png-debd151c542c.png)

*Figure 1. Assembly Code for the Loop*

From the previously assembled code, you can see there is a MAC operation and a load operation in every cycle of the loop. Wide registers `wr0`, `wr1`, `wr2`, and `wr3` are used for `buf_matA`. Accumulator registers `bm0` and `bm1` are used for the two accumulated results.

Keys to pipelining the loop are as follows:

- Preload the data into vector registers before the loop start.
- Interleave data loads, MAC operations, data stores in the loop body.
- Use wide input data vector register (`v64int16` in the example) to make data load and MAC operation perform on different parts of the vector register.
- Use multiple accumulator registers and reuse input data for multiple outputs.
- Data load and buffer pointer increment come in pairs. This applies for data store and buffer pointer increments as well.
