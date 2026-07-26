---
title: "Coding with Intrinsics"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Coding-with-Intrinsics"
toc_id: uojDmrlYn8RWhMErXAcw7g
content_id: 7OJELFuIeHzqIdVBqSGAgw
---

### Coding with Intrinsics

After analyzing how the function maps into the AI Engine vector processor, review the first version of the vectorized code.

```
inline void mac16_sub(input_window_int16* matA, v16int16 &buf_matB, v16acc48 &acc, int i){
	v32int16 buf_matA = undef_v32int16(); // holds 32 elements of matA
	buf_matA=upd_w(buf_matA, 0, window_read_v16(matA));
	window_incr(matA,64);
	buf_matA = upd_w(buf_matA, 1, window_read_v16(matA));
	window_incr(matA,64);
	acc = mac16(acc,buf_matA,0,0x73727170,0x77767574,0x3120,buf_matB,i,0x0,0x0,1);
}

void matmul_vec16(input_window_int16*  matA,
		input_window_int16* matB,
		output_window_int16* matC){

	v16int16 buf_matB = window_read_v16(matB); // holds 16 elements of matB
	v16acc48 acc = null_v16acc48(); // holds acc value of Row * column dot product

	for (unsigned int i=0;i<M/16;i++)  //M=64, Each iteration computes 16 outputs
	{
		acc=null_v16acc48();
		for(int j=0;j<16;j+=2){
			mac16_sub(matA,buf_matB,acc,j);
		}
		window_writeincr(matC,srs(acc,15));
		window_incr(matA,16);
	}
}
```

In the main function `matmul_vec16`, the loop produces 16 output data per iteration. In the outer loop body, there is an inner loop with eight iterations. In each iteration of the inner loop, an inline function `mac16_sub` is called. In the inline function, there is a `mac16` operation, with two loads of data for the MAC operation.

Inside `mac16_sub()`, `buf_matA` is declared as local variable and `buf_matB` and `acc` are declared as local variables in the main function. They are passed between functions by reference (or pointer). This ensures that only one identical vector exists for each variable. The function has one parameter that is used in the `mac16()` intrinsic as follows and this specific intrinsic (i=0) has been introduced in MAC Intrinsics.

```
acc = mac16(acc,buf_matA,0,0x73727170,0x77767574,0x3120,buf_matB,i,0x0,0x0,1);
```

At the end of each iteration of the loop, window pointer for the data is incremented by 16 (that is 16 rows for the matrix).

**Note:** `inline`

`inline __attribute__((always_inline))`

`__attribute__ ((noinline)) void func(...)`

The compiled code for the kernel can be found in the disassembly view in the debug perspective of the AMD Vitis™ IDE. Note that a graph is needed for compiling the kernel with AI Engine tools. For more understanding about the assembly code in disassembly view, refer to Using Vitis Unified IDE and Reports. For additional details on graph coding and Vitis IDE usage, refer to the AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)).

![iwc1606888923232.png](../assets/231-01-iwc1606888923232-png-6ec7e6329fde.png)

*Figure 1. Assembly Code for the Loop*

**Note:** This degradation of performance is caused by unbalanced window pointer increment at the end of the loop. This can be resolved by pairing the last increment with the last MAC operation. The optimized code is as follows.

```
inline void mac16_sub(input_window_int16* matA, v16int16 &buf_matB, v16acc48 &acc, int i,int incr_num){
	v32int16 buf_matA = undef_v32int16(); // holds 32 elements of matA
	buf_matA=upd_w(buf_matA, 0, window_read_v16(matA));
	window_incr(matA,64);
	buf_matA = upd_w(buf_matA, 1, window_read_v16(matA));
	window_incr(matA,incr_num);
	acc = 	mac16(acc,buf_matA,0,0x73727170,0x77767574,0x3120,buf_matB,i,0x0,0x0,1);
}

void matmul_vec16(input_window_int16*  matA,
		input_window_int16* matB,
		output_window_int16* matC){

	v16int16 buf_matB = window_read_v16(matB); // holds 16 elements of matB
	v16acc48 acc = null_v16acc48(); // holds acc value of Row * column dot product

	for (unsigned int i=0;i<M/16;i++)  //M=64, Each iteration computes 16 outputs
	{
		acc=null_v16acc48();
		for(int j=0;j<16;j+=2){
			int incr_num=(j==14)?80:64;
			mac16_sub(matA,buf_matB,acc,j,incr_num);
		}
		window_writeincr(matC,srs(acc,15));
	}
}
```

The function `mac16_sub` has a new parameter `incr_num`. This parameter is for the pointer increment, which is different for the last function call in the inner loop. This increment value of `80` ensures that the next 16 rows are selected during the next iteration of the outer loop. Now the assembled code for the loop is as shown in following figure.

![mag1606887255039.png](../assets/231-02-mag1606887255039-png-4727f1449870.png)

*Figure 2. Optimized Assembly Code for the Loop*

An iteration of the loop requires 16 cycles. This means that the compute bound for this kernel is 16*4=64 cycles per invocation. As seen in the previous section, the theoretical limit is 32 cycles per invocation. That is eight cycles for an iteration of the loop, which means that eight MAC operations must be compacted into eight cycles.

Depending on system performance needs, you can split the input data column by column into two window buffers: `matA_0` and `matA_1`. The data of the two windows is first read into two v16int16 vectors and concatenated into one v32int16 vector for use in the `mac16` intrinsic. The code for the kernel is as follows.

```
inline void mac16_sub_loads(input_window_int16* matA_0, input_window_int16* matA_1, v16int16 &buf_matB, v16acc48 &acc, int i, int incr_num){
	v16int16 buf_matA0 = window_read_v16(matA_0);
	window_incr(matA_0,incr_num);
	v16int16 buf_matA1 = window_read_v16(matA_1);
	window_incr(matA_1,incr_num);
	acc = 	mac16(acc,concat(buf_matA0,buf_matA1),0,0x73727170,0x77767574,0x3120,buf_matB,i,0x0,0x0,1);
}

void matmul_vec16(input_window_int16* __restrict matA_0,
		input_window_int16* __restrict matA_1,
		input_window_int16* __restrict matB,
		output_window_int16* __restrict matC){
	v16int16 buf_matB = window_read_v16(matB);
	for (unsigned int i=0;i<M/16;i++)  //M=64, Each iteration computes 16 outputs
	chess_prepare_for_pipelining
	{
		v16acc48 acc=null_v16acc48();
		for(int j=0;j<16;j+=2){
			int incr_num=(j==14)?80:64;
			mac16_sub_loads(matA_0,matA_1,buf_matB,acc,j,incr_num);
		}
		window_writeincr(matC,srs(acc,15));
	}
}
```

The code defines and concatenates two `v16int16` vectors, `buf_matA0` and `buf_matA1`, for the `mac16` intrinsic. Also note that `chess_prepare_for_pipelining` is added for the loop and `__restrict` keyword for the window interfaces. This ensures that the loop is pipelined and window operations can be optimized.

**Note:** Important:

`__restrict`

Using the Restrict Keyword in AI Engine Kernels

The assembly code for the version of two window loads in a cycle is as follows.

![fag1606887407846.png](../assets/231-03-fag1606887407846-png-6286e3f2475a.png)

*Figure 3. Assembly Code for Two Window Loads a Cycle*
