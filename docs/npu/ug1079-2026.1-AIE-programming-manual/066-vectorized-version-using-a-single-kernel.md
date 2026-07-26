---
title: "Vectorized Version using a Single Kernel"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vectorized-Version-using-a-Single-Kernel"
toc_id: 3o3P6sFyWPEEhpyUWsn~yA
content_id: U6TtRjq2M86GPLYeMK49Sg
---

### Vectorized Version using a Single Kernel

The AI Engine naturally supports multiple lanes of MAC operations. For FIR application variations, use the group of `aie::sliding_mul*` classes and functions introduced in Multiple Lanes Multiplication - sliding_mul.

`aie::sliding_mul`

`aie::sliding_mac`

`Lanes=8`

`Points=8`

`acc = aie::sliding_mac<8,8>(acc,coe[1],0,buff,8);`

```
Lane 0: acc[0]=acc[0]+coe[1][0]*buff[8]+coe[1][1]*buff[9]+...+coe[1][7]*buff[15];
Lane 1: acc[1]=acc[1]+coe[1][0]*buff[9]+coe[1][1]*buff[10]+...+coe[1][7]*buff[16];
...
Lane 7: acc[7]=acc[7]+coe[1][0]*buff[15]+coe[1][1]*buff[16]+...+coe[1][7]*buff[22];
```

Notice that the data `buff` starts from different indexes in different lanes. It requires more than eight samples (from `buff[8]` to `buff[22]`) to be ready before execution.

Because it has 32 taps, the FIR requires one `aie::sliding_mul<8,8>` operation and three `aie::sliding_mac<8,8>` operations to calculate eight lanes of output. The stream port updates the data buffer using `buff.insert`.

The vectorized kernel code is as follows:

```
//keep margin data between different executions of graph
static aie::vector<cint16,32> delay_line;

alignas(aie::vector_decl_align) static cint16 eq_coef[32]={{1,2},{3,4},...};

__attribute__((noinline)) void fir_32tap_vector(input_stream<cint16> * sig_in, output_stream<cint16> * sig_out){
  const int LSIZE=(SAMPLES/32);
  aie::accum<cacc48,8> acc;
  const aie::vector<cint16,8> coe[4] = {aie::load_v<8>(eq_coef),aie::load_v<8>(eq_coef+8),aie::load_v<8>(eq_coef+16),aie::load_v<8>(eq_coef+24)};
  aie::vector<cint16,32> buff=delay_line;
  for(int i=0;i<LSIZE;i++){
    //1st 8 samples
    acc = aie::sliding_mul<8,8>(coe[0],0,buff,0);
    acc = aie::sliding_mac<8,8>(acc,coe[1],0,buff,8);
    buff.insert(0,readincr_v<4>(sig_in));
    buff.insert(1,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<8,8>(acc,coe[2],0,buff,16);
    acc = aie::sliding_mac<8,8>(acc,coe[3],0,buff,24);
    writeincr(sig_out,acc.to_vector<cint16>(SHIFT));

    //2nd 8 samples
    acc = aie::sliding_mul<8,8>(coe[0],0,buff,8);
    acc = aie::sliding_mac<8,8>(acc,coe[1],0,buff,16);
    buff.insert(2,readincr_v<4>(sig_in));
    buff.insert(3,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<8,8>(acc,coe[2],0,buff,24);
    acc = aie::sliding_mac<8,8>(acc,coe[3],0,buff,0);
    writeincr(sig_out,acc.to_vector<cint16>(SHIFT));

    //3rd 8 samples
    acc = aie::sliding_mul<8,8>(coe[0],0,buff,16);
    acc = aie::sliding_mac<8,8>(acc,coe[1],0,buff,24);
    buff.insert(4,readincr_v<4>(sig_in));
    buff.insert(5,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<8,8>(acc,coe[2],0,buff,0);
    acc = aie::sliding_mac<8,8>(acc,coe[3],0,buff,8);
    writeincr(sig_out,acc.to_vector<cint16>(SHIFT));

    //4th 8 samples
    acc = aie::sliding_mul<8,8>(coe[0],0,buff,24);
    acc = aie::sliding_mac<8,8>(acc,coe[1],0,buff,0);
    buff.insert(6,readincr_v<4>(sig_in));
    buff.insert(7,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<8,8>(acc,coe[2],0,buff,8);
    acc = aie::sliding_mac<8,8>(acc,coe[3],0,buff,16);
    writeincr(sig_out,acc.to_vector<cint16>(SHIFT));
  }
  delay_line=buff;
}
void fir_32tap_vector_init()
{
  //initialize data
  for (int i=0;i<8;i++){
    aie::vector<int16,8> tmp=get_wss(0);
    delay_line.insert(i,tmp.cast_to<cint16>());
  }
};
```

**Note:** - `alignas(aie::vector_decl_align)` can be used to ensure data is aligned for vector load and store.
- Each iteration of the main loop computes multiple samples, reducing the loop count.
- Data update, calculation and data write are interleaved in the code. Determining which portion of data buffer `buff` to read is controlled using `data_start` of `aie::sliding_mul`.
- For more information about supported data types and lane numbers for `aie::sliding_mul`, see AI Engine API User Guide ([UG1529](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_api/aie_api/doc/index.html)).

You need to identify the II of the main loop. To locate the II of the loop, do the following:

1. Add the `-v` option to the AI Engine compiler to output a verbose report of kernel compilation.
2. Open the kernel compilation log, for example, `Work/aie/<COL_ROW>/<COL_ROW>.log`.
3. In the log, search keywords, such as `do-loop`, to find the initiation interval of the loop.See the following example result: `HW do-loop #3651 in "/wrk/xcohdnobkup3/brucey/AIE_test_cases/ug1079/fir_32tap/fir_32tap_asym_1kernel/aie/fir_32tap_vector.cc", line 21: (loop #3) : critical cycle of length 121 : b428 -> b95 -> b96 -> b97 -> b98 -> b102 -> b115 -> b119 -> b120 -> b121 -> b122 -> b48 -> b49 -> b65 -> b73 -> b74 -> b75 -> b92 -> b99 -> b103 -> b104 -> b105 -> b106 -> b116 -> b123 -> b127 -> b128 -> b129 -> b130 -> b448 -> b215 -> b216 -> b217 -> b218 -> b50 -> b51 -> b175 -> b176 -> b177 -> b178 -> b188 -> b195 -> b199 -> b200 -> b201 -> b202 -> b212 -> b219 -> b223 -> b224 -> b225 -> b226 -> b236 -> b241 -> b67 -> b79 -> b80 -> b81 -> b93 -> b124 -> b131 -> b132 -> b133 -> b134 -> b52 -> b53 -> b155 -> b156 -> b157 -> b158 -> b165 -> b172 -> b179 -> b180 -> b181 -> b182 -> b419 -> b69 -> b85 -> b86 -> b87 -> b498 -> b205 -> b206 -> b213 -> b220 -> b227 -> b228 -> b229 -> b230 -> b445 -> b111 -> b112 -> b113 -> b114 -> b118 -> b125 -> b135 -> b136 -> b137 -> b138 -> b54 -> b55 -> b159 -> b160 -> b161 -> b162 -> b166 -> b173 -> b183 -> b184 -> b185 -> b186 -> b190 -> b221 -> b231 -> b232 -> b233 -> b234 -> b428 minimum length due to resources: 128 scheduling HW do-loop #3651 (algo 2) -> # cycles: 173 (exceeds -k 110) -> no folding: 173 -> HW do-loop #3651 in "/proj/xbuilds/SWIP/2025.1_0427_1909/installs/lin64/2025.1/Vitis/aietools/include/adf/stream/me/stream_utils.h", line 258: (loop #3) : 173 cycles` where: The II of the loop is 173. This means it takes 173/32~=5.4 cycles to produce a sample. The message `(exceeds -k 110) -> no folding` indicates that the scheduler is not attempting software pipelining because the loop cycle count exceeds a limit.

  See the following example result:

  ```
  HW do-loop #3651 in "/wrk/xcohdnobkup3/brucey/AIE_test_cases/ug1079/fir_32tap/fir_32tap_asym_1kernel/aie/fir_32tap_vector.cc", line 21: (loop #3) :
  critical cycle of length 121 : b428 -> b95 -> b96 -> b97 -> b98 -> b102 -> b115 -> b119 -> b120 -> b121 -> b122 -> b48 -> b49 -> b65 -> b73 -> b74 -> b75 -> b92 -> b99 -> b103 -> b104 -> b105 -> b106 -> b116 -> b123 -> b127 -> b128 -> b129 -> b130 -> b448 -> b215 -> b216 -> b217 -> b218 -> b50 -> b51 -> b175 -> b176 -> b177 -> b178 -> b188 -> b195 -> b199 -> b200 -> b201 -> b202 -> b212 -> b219 -> b223 -> b224 -> b225 -> b226 -> b236 -> b241 -> b67 -> b79 -> b80 -> b81 -> b93 -> b124 -> b131 -> b132 -> b133 -> b134 -> b52 -> b53 -> b155 -> b156 -> b157 -> b158 -> b165 -> b172 -> b179 -> b180 -> b181 -> b182 -> b419 -> b69 -> b85 -> b86 -> b87 -> b498 -> b205 -> b206 -> b213 -> b220 -> b227 -> b228 -> b229 -> b230 -> b445 -> b111 -> b112 -> b113 -> b114 -> b118 -> b125 -> b135 -> b136 -> b137 -> b138 -> b54 -> b55 -> b159 -> b160 -> b161 -> b162 -> b166 -> b173 -> b183 -> b184 -> b185 -> b186 -> b190 -> b221 -> b231 -> b232 -> b233 -> b234 -> b428
  minimum length due to resources: 128
  scheduling HW do-loop #3651
  (algo 2)	-> # cycles: 173  (exceeds -k 110)    -> no folding: 173
    -> HW do-loop #3651 in "/proj/xbuilds/SWIP/2025.1_0427_1909/installs/lin64/2025.1/Vitis/aietools/include/adf/stream/me/stream_utils.h", line 258: (loop #3) : 173 cycles
  ```

  where:

  - The II of the loop is 173. This means it takes 173/32~=5.4 cycles to produce a sample.
  - The message `(exceeds -k 110) -> no folding` indicates that the scheduler is not attempting software pipelining because the loop cycle count exceeds a limit.
4. To override the loop cycle limit, add a user constraint, such as `--Xchess="fir_32tap_vector:backend.mist2.maxfoldk=200"` to the AI Engine compiler. The example result is then as follows: `(resume algo) -> after folding: 156 (folded over 1 iterations) -> HW do-loop #3651 in "/proj/xbuilds/SWIP/2025.1_0427_1909/installs/lin64/2025.1/Vitis/aietools/include/adf/stream/me/stream_utils.h", line 258: (loop #3) : 156 cycles` where, the software requires approximately 156/32=4.9 cycles to produce a sample. Note: The exact number of cycles might fluctuate slightly based on the specific compiler settings and the version of the compiler being used. However, the analysis techniques described in this section remain relevant and applicable regardless of these variations.

  The example result is then as follows:

  ```
  (resume algo)	  -> after folding: 156  (folded over 1 iterations)
    -> HW do-loop #3651 in "/proj/xbuilds/SWIP/2025.1_0427_1909/installs/lin64/2025.1/Vitis/aietools/include/adf/stream/me/stream_utils.h", line 258: (loop #3) : 156 cycles
  ```

  where, the software requires approximately 156/32=4.9 cycles to produce a sample.
