---
title: "Coding with Intrinsics"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Coding-with-Intrinsics"
toc_id: FG0OumYiGZfcBEznslRc_A
content_id: Nj7epKF6uZDeItVk1UTstQ
---

### Coding with Intrinsics

The four kernels in the 1 Gsps implementation can have different sets of coefficients and cascade streams between them. The following figure shows this implementation.

![djm1606888612523.png](../assets/238-01-djm1606888612523-png-584eba6bd8f5.png)

*Figure 1. Four Kernels with Split Coefficient and Cascade Stream*

Input data flows from stream to these four kernels. However, the second kernel discards the first eight input data. The third kernel discards the first 16 input data. Similarly, the fourth kernel discards the first 24 input data.

The code for the first kernel is as follows.

```
#include <adf.h>
#include "fir_32tap.h"
// buffer to keep state
static v16cint16 delay_line;

void fir_32tap_core0(
	input_stream_cint16 * sig_in,
	output_cascade_cacc48 * cascadeout)
{
	const cint16_t * __restrict coeff = eq_coef0;
	const v8cint16 *coef_  =  (v8cint16 const*)coeff;
	const v8cint16 coe = *coef_;

	v16cint16 buff = delay_line;
	v4cacc48 acc;
	const unsigned LSIZE = (samples/4/4); // assuming samples is integer power of 2 and greater than 16

	for (unsigned int i = 0; i < LSIZE; ++i)
	chess_prepare_for_pipelining
	chess_loop_range(4,)
	{
		acc  = mul4(buff, 0 , 0x3210, 1,  coe, 0, 0x0000, 1);
		acc  = mac4(acc, buff, 2 , 0x3210, 1,  coe, 2, 0x0000, 1);
		buff = upd_v(buff, 2, readincr_v4(sig_in));
		acc  = mac4(acc, buff, 4 , 0x3210, 1,  coe, 4, 0x0000, 1);
		acc  = mac4(acc, buff, 6 , 0x3210, 1,  coe, 6, 0x0000, 1);
		writeincr(cascadeout,acc);

		acc  = mul4(buff, 4 , 0x3210, 1,  coe, 0, 0x0000, 1);
		acc  = mac4(acc, buff, 6 , 0x3210, 1,  coe, 2, 0x0000, 1);
		buff = upd_v(buff, 3, readincr_v4(sig_in));
		acc  = mac4(acc, buff, 8 , 0x3210, 1,  coe, 4, 0x0000, 1);
		acc  = mac4(acc, buff, 10, 0x3210, 1,  coe, 6, 0x0000, 1);
		writeincr(cascadeout,acc);

		acc  = mul4(buff, 8  , 0x3210, 1,  coe, 0, 0x0000, 1);
		acc  = mac4(acc, buff, 10 , 0x3210, 1,  coe, 2, 0x0000, 1);
		buff = upd_v(buff, 0, readincr_v4(sig_in));
		acc  = mac4(acc, buff, 12 , 0x3210, 1,  coe, 4, 0x0000, 1);
		acc  = mac4(acc, buff, 14 , 0x3210, 1,  coe, 6, 0x0000, 1);
		writeincr(cascadeout,acc);

		acc  = mul4(buff, 12 , 0x3210, 1,  coe, 0, 0x0000, 1);
		acc  = mac4(acc, buff, 14 , 0x3210, 1,  coe, 2, 0x0000, 1);
		buff = upd_v(buff, 1, readincr_v4(sig_in));
		acc  = mac4(acc, buff, 0  , 0x3210, 1,  coe, 4, 0x0000, 1);
		acc  = mac4(acc, buff, 2  , 0x3210, 1,  coe, 6, 0x0000, 1);
		writeincr(cascadeout,acc);
    }
    delay_line = buff;
}

void fir_32tap_core0_init()
{
	// Drop samples if not first block
	int const Delay = 0;
	for (int i = 0; i < Delay; ++i)
	{
		get_ss(0);
	}

};
```

The function, `fir_32tap_core0_init`, is the initialization function for the AI Engine kernel, `fir_32tap_core0`, which is executed one time at the kernel start. The purpose of this initialization function is to discard the unnecessary samples to align the input stream.

Similarly, the function, `fir_32tap_core1_init`, is going to be the initialization function for the AI Engine kernel, `fir_32tap_core1`, in the following codes. Same applies for the initialization functions, `fir_32tap_core2_init` and `fir_32tap_core3_init`.

The second kernel code is as follows.

```
#include <adf.h>
#include "fir_32tap.h"
// buffer to keep state
static v16cint16 delay_line;

void fir_32tap_core1(
	input_stream_cint16 * sig_in,
	input_cascade_cacc48 * cascadein,
	output_cascade_cacc48 * cascadeout)
{
    const cint16_t * __restrict coeff = eq_coef1;
    const v8cint16 *coef_  =  (v8cint16 const*)coeff;
    const v8cint16 coe = *coef_;

    v16cint16 buff = delay_line;
    v4cacc48 acc;
    const unsigned LSIZE = (samples/4/4); // assuming samples is integer power of 2 and greater than 16

    for (unsigned int i = 0; i < LSIZE; ++i)
    chess_prepare_for_pipelining
    chess_loop_range(4,)
    {
        acc = readincr_v4(cascadein);
        acc  = mac4(acc, buff, 0 , 0x3210, 1,  coe, 0, 0x0000, 1);
        acc  = mac4(acc, buff, 2 , 0x3210, 1,  coe, 2, 0x0000, 1);
        buff = upd_v(buff, 2, readincr_v4(sig_in));
        acc  = mac4(acc, buff, 4 , 0x3210, 1,  coe, 4, 0x0000, 1);
        acc  = mac4(acc, buff, 6 , 0x3210, 1,  coe, 6, 0x0000, 1);
        writeincr(cascadeout,acc);

        acc = readincr_v4(cascadein);
        acc  = mac4(acc, buff, 4 , 0x3210, 1,  coe, 0, 0x0000, 1);
        acc  = mac4(acc, buff, 6 , 0x3210, 1,  coe, 2, 0x0000, 1);
        buff = upd_v(buff, 3, readincr_v4(sig_in));
        acc  = mac4(acc, buff, 8 , 0x3210, 1,  coe, 4, 0x0000, 1);
        acc  = mac4(acc, buff, 10, 0x3210, 1,  coe, 6, 0x0000, 1);
        writeincr(cascadeout,acc);

        acc = readincr_v4(cascadein);
        acc  = mac4(acc, buff, 8  , 0x3210, 1,  coe, 0, 0x0000, 1);
        acc  = mac4(acc, buff, 10 , 0x3210, 1,  coe, 2, 0x0000, 1);
        buff = upd_v(buff, 0, readincr_v4(sig_in));
        acc  = mac4(acc, buff, 12 , 0x3210, 1,  coe, 4, 0x0000, 1);
        acc  = mac4(acc, buff, 14 , 0x3210, 1,  coe, 6, 0x0000, 1);
        writeincr(cascadeout,acc);

        acc = readincr_v4(cascadein);
        acc  = mac4(acc, buff, 12 , 0x3210, 1,  coe, 0, 0x0000, 1);
        acc  = mac4(acc, buff, 14 , 0x3210, 1,  coe, 2, 0x0000, 1);
        buff = upd_v(buff, 1, readincr_v4(sig_in));
        acc  = mac4(acc, buff, 0  , 0x3210, 1,  coe, 4, 0x0000, 1);
        acc  = mac4(acc, buff, 2  , 0x3210, 1,  coe, 6, 0x0000, 1);
        writeincr(cascadeout,acc);
    }
    delay_line = buff;
}

void fir_32tap_core1_init()
{
	// Drop samples if not first block
    int const Delay = 8;
    for (int i = 0; i < Delay; ++i)
    {
        get_ss(0);
    }
};
```

The third kernel is similar to the second one. The last kernel is as follows.

```
#include <adf.h>
#include "fir_32tap.h"
// buffer to keep state
static v16cint16 delay_line;

void fir_32tap_core3(
	input_stream_cint16 * sig_in,
	input_cascade_cacc48 * cascadein,
	output_stream_cint16 * data_out)
{
	const cint16_t * __restrict coeff = eq_coef3;
	const v8cint16 *coef_  =  (v8cint16 const*)coeff;
	const v8cint16 coe = *coef_;

	v16cint16 buff = delay_line;

	v4cacc48 acc;

	set_rnd(rnd_pos_inf);
	set_sat();
	const unsigned LSIZE = (samples/4/4); // assuming samples is integer power of 2 and greater than 16

	for (unsigned int i = 0; i < LSIZE; ++i)
	chess_prepare_for_pipelining
	chess_loop_range(4,)
	{
		acc = readincr_v4(cascadein);
		acc  = mac4(acc, buff, 0 , 0x3210, 1,  coe, 0, 0x0000, 1);
		acc  = mac4(acc, buff, 2 , 0x3210, 1,  coe, 2, 0x0000, 1);
		buff = upd_v(buff, 2, readincr_v4(sig_in));
		acc  = mac4(acc, buff, 4 , 0x3210, 1,  coe, 4, 0x0000, 1);
		acc  = mac4(acc, buff, 6 , 0x3210, 1,  coe, 6, 0x0000, 1);
		writeincr(data_out,srs(acc,shift));

		acc = readincr_v4(cascadein);
		acc  = mac4(acc, buff, 4 , 0x3210, 1,  coe, 0, 0x0000, 1);
		acc  = mac4(acc, buff, 6 , 0x3210, 1,  coe, 2, 0x0000, 1);
		buff = upd_v(buff, 3, readincr_v4(sig_in));
		acc  = mac4(acc, buff, 8 , 0x3210, 1,  coe, 4, 0x0000, 1);
		acc  = mac4(acc, buff, 10, 0x3210, 1,  coe, 6, 0x0000, 1);
		writeincr(data_out,srs(acc,shift));

		acc = readincr_v4(cascadein);
		acc  = mac4(acc, buff, 8  , 0x3210, 1,  coe, 0, 0x0000, 1);
		acc  = mac4(acc, buff, 10 , 0x3210, 1,  coe, 2, 0x0000, 1);
		buff = upd_v(buff, 0, readincr_v4(sig_in));
		acc  = mac4(acc, buff, 12 , 0x3210, 1,  coe, 4, 0x0000, 1);
		acc  = mac4(acc, buff, 14 , 0x3210, 1,  coe, 6, 0x0000, 1);
		writeincr(data_out,srs(acc,shift));

		acc = readincr_v4(cascadein);
		acc  = mac4(acc, buff, 12 , 0x3210, 1,  coe, 0, 0x0000, 1);
		acc  = mac4(acc, buff, 14 , 0x3210, 1,  coe, 2, 0x0000, 1);
		buff = upd_v(buff, 1, readincr_v4(sig_in));
		acc  = mac4(acc, buff, 0  , 0x3210, 1,  coe, 4, 0x0000, 1);
		acc  = mac4(acc, buff, 2  , 0x3210, 1,  coe, 6, 0x0000, 1);
		writeincr(data_out,srs(acc,shift));
	}
	delay_line = buff;
}

void fir_32tap_core3_init()
{
	// Drop samples if not first block
	int const Delay = 24;
	for (int i = 0; i < Delay; ++i)
	{
		get_ss(0);
	}
};
```

The graph code is as follows.

```
#include <adf.h>
#include "kernels.h"
using namespace adf;
class firGraph : public graph {
	public:
	kernel k0,k1,k2,k3;
	input_port in0123;
	output_port out;
	firGraph()
	{
		k0 = kernel::create(fir_32tap_core0);
		runtime<ratio>(k0) = 0.9;
		source(k0) = "fir_32tap_core0.cpp";
		connect<stream> n0(in0123,k0.in[0]);

		k1 = kernel::create(fir_32tap_core1);
		runtime<ratio>(k1) = 0.9;
		source(k1) = "fir_32tap_core1.cpp";
		connect<stream> n1(in0123,k1.in[0]);
		connect<cascade> (k0.out[0],k1.in[1]);

		k2 = kernel::create(fir_32tap_core2);
		runtime<ratio>(k2) = 0.9;
		source(k2) = "fir_32tap_core2.cpp";
		connect<stream> n2(in0123,k2.in[0]);
		connect<cascade> (k1.out[0],k2.in[1]);

		k3 = kernel::create(fir_32tap_core3);
		runtime<ratio>(k3) = 0.9;
		source(k3) = "fir_32tap_core3.cpp";
		connect<stream> n3(in0123,k3.in[0]);
		connect<cascade> (k2.out[0],k3.in[1]);
		connect<stream> (k3.out[0],out);

		initialization_function(k0) = "fir_32tap_core0_init";
		initialization_function(k1) = "fir_32tap_core1_init";
		initialization_function(k2) = "fir_32tap_core2_init";
		initialization_function(k3) = "fir_32tap_core3_init";
	};
};
```

The kernels connected through cascade streams must operate synchronously. Conflicts in cascade streams can stall the kernels. Loops in the kernels must have input data available to run smoothly. Hence it is important that the input stream arrives at the appropriate time for each kernel. The input stream stall (if any) can be resolved by adding a large enough FIFO to the net connecting to the AI Engine kernels.

For example:

```
fifo_depth(n0)=175;
fifo_depth(n1)=150;
fifo_depth(n2)=125;
fifo_depth(n3)=100;
```

**Note:** The previous example specifies different FIFO depths to prevent auto FIFO merge which can occur when a common FIFO depth is used for all nets.

To save FIFO resources, set individual FIFO depths by looking at when the event `CORE_INSTREAM_WIDE` occurs for each kernel. The earlier the event occurs, the deeper the FIFO needs to be. For example:

```
fifo_depth(n0)=45;
fifo_depth(n1)=33;
fifo_depth(n2)=23;
fifo_depth(n3)=10;
```

For additional details about coding on graph, refer to the AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)).
