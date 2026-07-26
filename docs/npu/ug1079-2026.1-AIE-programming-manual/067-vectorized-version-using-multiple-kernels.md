---
title: "Vectorized Version using Multiple Kernels"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vectorized-Version-using-Multiple-Kernels"
toc_id: GyHwmEs7skmGGWQO3~~uiQ
content_id: x1KCNeyBO8bX2wHnsk577g
---

### Vectorized Version using Multiple Kernels

With one input data per cycle, as estimated in Design Analysis, you must run four kernels in parallel to get the best throughput. One such solution is to divide 32 coefficients into four kernels, where data is broadcast to four kernels. The partial accumulation results from kernels can be cascaded to produce result in the last kernel. The following figure shows the implementation.

![djm1606888612523.png](../assets/067-01-djm1606888612523-png-584eba6bd8f5.png)

*Figure 1. Data Broadcast to four Kernels*

**Note:** Subsequent kernels discard part of the initial data. For example, the second kernel discards the first eight inputs.

Four lanes and eight points are chosen for `aie::sliding_mul`. Data reads and writes are interleaved with computation.

The first kernel code is as follows:

```
alignas(aie::vector_decl_align) static cint16 eq_coef0[8]={{1,2},{3,4},...};

//For storing data between graph iterations
static aie::vector<cint16,16> delay_line;
__attribute__((noinline)) void fir_32tap_core0(input_stream<cint16> * sig_in,
      output_cascade<cacc48> * cascadeout){
  const cint16_t * restrict coeff = eq_coef0;
  const aie::vector<cint16,8> coe = aie::load_v<8>(coeff);

  aie::vector<cint16,16> buff = delay_line;
  aie::accum<cacc48,4> acc;
  const unsigned LSIZE = (SAMPLES/4/4); // assuming samples is integer power of 2 and greater than 16

  main_loop:for (unsigned int i = 0; i < LSIZE; ++i)
  chess_prepare_for_pipelining
  {
    //8 MAC produce 4 partial output
    buff.insert(2,readincr_v<4>(sig_in));
    acc = aie::sliding_mul<4,8>(coe,0,buff,0);
    writeincr(cascadeout,acc);

    //8 MAC produce 4 partial output
    buff.insert(3,readincr_v<4>(sig_in));
    acc = aie::sliding_mul<4,8>(coe,0,buff,4);
    writeincr(cascadeout,acc);

    buff.insert(0,readincr_v<4>(sig_in));
    acc = aie::sliding_mul<4,8>(coe,0,buff,8);
    writeincr(cascadeout,acc);

    buff.insert(1,readincr_v<4>(sig_in));
    acc = aie::sliding_mul<4,8>(coe,0,buff,12);
    writeincr(cascadeout,acc);
  }
  delay_line = buff;
}
void fir_32tap_core0_init(){
  // Drop samples if not first block
  int const Delay = 0;
  for (int i = 0; i < Delay; ++i){
    get_ss(0);
  }
  //initialize data
  for (int i=0;i<8;i++){
    int tmp=get_ss(0);
    delay_line.set(*(cint16*)&tmp,i);
  }
};
```

**Note:** - `__attribute__((noinline))` is optional to keep function hierarchy.
- `chess_prepare_for_pipelining` is optional as the tools can do automatic pipelining.
- Each `aie::sliding_mul<4,8>` is multiplying four lanes eight points MAC and the partial result is sent through the cascade chain to the next kernel.
- Data `buff` is read starting from `data_start` parameter of `aie::sliding_mul`. The kernel code goes back to the beginning when it reaches the end in a circular fashion.

`Work/aie/<COL_ROW>/<COL_ROW>.log`

`-v`

`do-loop`

```
(resume algo) -> after folding: 16 (folded over 1 iterations)
-> HW do-loop #128 in ".../Vitis/<VERSION>/aietools/include/adf/stream/me/stream_utils.h", line 1192: (loop #3) : 16 cycles
```

**Note:** Tip:

The kernel code above takes roughly `16 (cycles) / 16 (partial results) = 1 cycle` to produce a partial output.

```
alignas(aie::vector_decl_align) static cint16 eq_coef2[8]={{17,18},{19,20},...};

//For storing data between graph iterations
alignas(aie::vector_decl_align) static aie::vector<cint16,16> delay_line;

__attribute__((noinline)) void fir_32tap_core1(input_stream<cint16> * sig_in, input_cascade<cacc48> * cascadein,
      output_cascade<cacc48> * cascadeout){
  const aie::vector<cint16,8> coe = aie::load_v<8>(eq_coef1);
  aie::vector<cint16,16> buff = delay_line;
  aie::accum<cacc48,4> acc;
  const unsigned LSIZE = (SAMPLES/4/4); // assuming samples is integer power of 2 and greater than 16

  for (unsigned int i = 0; i < LSIZE; ++i)
  chess_prepare_for_pipelining
  {
    //8 MAC produce 4 partial output
    acc = readincr_v4(cascadein);
    buff.insert(2,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,0);
    writeincr(cascadeout,acc);

    acc = readincr_v4(cascadein);
    buff.insert(3,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,4);
    writeincr(cascadeout,acc);

    acc = readincr_v4(cascadein);
    buff.insert(0,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,8);
    writeincr(cascadeout,acc);

    acc = readincr_v4(cascadein);
    buff.insert(1,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,12);
    writeincr(cascadeout,acc);
  }
  delay_line = buff;
}
void fir_32tap_core1_init()
{
  // Drop samples if not first block
  int const Delay = 8;
  for (int i = 0; i < Delay; ++i){
    get_ss(0);
  }
  //initialize data
  for (int i=0;i<8;i++){
    int tmp=get_ss(0);
    delay_line.set(*(cint16*)&tmp,i);
  }
};
```

```
alignas(aie::vector_decl_align) static cint16 eq_coef2[8]={{33,34},{35,36},...};

//For storing data between graph iterations
alignas(aie::vector_decl_align) static aie::vector<cint16,16> delay_line;
__attribute__((noinline)) void fir_32tap_core2(input_stream<cint16> * sig_in, input_cascade<cacc48> * cascadein,
      output_cascade<cacc48> * cascadeout){
  const aie::vector<cint16,8> coe = aie::load_v<8>(eq_coef2);
  aie::vector<cint16,16> buff = delay_line;
  aie::accum<cacc48,4> acc;
  const unsigned LSIZE = (SAMPLES/4/4); // assuming samples is integer power of 2 and greater than 16

  for (unsigned int i = 0; i < LSIZE; ++i)
  chess_prepare_for_pipelining
  {
    //8 MAC produce 4 partial output
    acc = readincr_v4(cascadein);
    buff.insert(2,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,0);
    writeincr(cascadeout,acc);

    acc = readincr_v4(cascadein);
    buff.insert(3,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,4);
    writeincr(cascadeout,acc);

    acc = readincr_v4(cascadein);
    buff.insert(0,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,8);
    writeincr(cascadeout,acc);

    acc = readincr_v4(cascadein);
    buff.insert(1,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,12);
    writeincr(cascadeout,acc);
  }
  delay_line = buff;
}
void fir_32tap_core2_init(){
  // Drop samples if not first block
  int const Delay = 16;
  for (int i = 0; i < Delay; ++i)
  {
    get_ss(0);
  }
  //initialize data
  for (int i=0;i<8;i++){
    int tmp=get_ss(0);
    delay_line.set(*(cint16*)&tmp,i);
  }
};
```

```
alignas(aie::vector_decl_align) static cint16 eq_coef3[8]={{49,50},{51,52},...};

//For storing data between graph iterations
alignas(aie::vector_decl_align) static aie::vector<cint16,16> delay_line;
__attribute__((noinline)) void fir_32tap_core3(input_stream<cint16> * sig_in, input_cascade<cacc48> * cascadein,
      output_stream<cint16> * data_out){
  const aie::vector<cint16,8> coe = aie::load_v<8>(eq_coef3);
  aie::vector<cint16,16> buff = delay_line;
  aie::accum<cacc48,4> acc;
  const unsigned LSIZE = (SAMPLES/4/4); // assuming samples is integer power of 2 and greater than 16

  for (unsigned int i = 0; i < LSIZE; ++i)
  chess_prepare_for_pipelining
  {
    //8 MAC produce 4 output
    acc = readincr_v4(cascadein);
    buff.insert(2,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,0);
    writeincr(data_out,acc.to_vector<cint16>(SHIFT));

    acc = readincr_v4(cascadein);
    buff.insert(3,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,4);
    writeincr(data_out,acc.to_vector<cint16>(SHIFT));

    acc = readincr_v4(cascadein);
    buff.insert(0,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,8);
    writeincr(data_out,acc.to_vector<cint16>(SHIFT));

    acc = readincr_v4(cascadein);
    buff.insert(1,readincr_v<4>(sig_in));
    acc = aie::sliding_mac<4,8>(acc,coe,0,buff,12);
    writeincr(data_out,acc.to_vector<cint16>(SHIFT));
  }
  delay_line = buff;
}
void fir_32tap_core3_init()
{
  // Drop samples if not first block
  int const Delay = 24;
  for (int i = 0; i < Delay; ++i){
    get_ss(0);
  }
  //initialize data
  for (int i=0;i<8;i++){
    int tmp=get_ss(0);
    delay_line.set(*(cint16*)&tmp,i);
  }
};
```

The last kernel writes results to the output stream using `acc.to_vector<cint16>(SHIFT)`.

Each kernel takes one cycle to produce a partial output. When they are working simultaneously, the system performance is one cycle to produce one output, which meets the design goal.

See the AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)) for more information about the following:

- Graph construction
- Stream broadcast
- DMA FIFO insertion
- Profiling in simulation and hardware
- Design stall and deadlock analysis
