---
title: "Scalar Golden Reference"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Scalar-Golden-Reference"
toc_id: TmWXhgY008E3IUpc_0_4uQ
content_id: fnuV81TIqwtWkReVHgAFQw
---

### Scalar Golden Reference

The AI Engine contains a scalar processor. The processor implement scalar math operations, non-linear functions, and other general purpose operations. Sometimes it can be helpful to have a golden scalar reference version of code.

**Note:** The scalar version of code takes much more time to run in simulation and hardware compared to the vectorized version.

```
static cint16 eq_coef[32]={{1,2},{3,4},...};

//keep margin data between different invocations of the graph
static cint16 delay_line[32];

__attribute__((noinline)) void fir_32tap_scalar(input_stream<cint16> * sig_in,
      output_stream<cint16> * sig_out){

  for(int i=0;i<SAMPLES;i++){
    cycle_num[0]=tile.cycles();//cycle counter of the AI Engine tile
    cint64 sum={0,0};//larger data to mimic accumulator
    for(int j=0;j<32;j++){
      //auto integer promotion to prevent overflow
      sum.real+=delay_line[j].real*eq_coef[j].real-delay_line[j].imag*eq_coef[j].imag;
      sum.imag+=delay_line[j].real*eq_coef[j].imag+delay_line[j].imag*eq_coef[j].real;
    }
    sum=sum>>SHIFT;
    //produce one sample per loop iteration
    writeincr(sig_out,{(int16)sum.real,(int16)sum.imag});

    for(int j=0;j<32;j++){
      if(j==31){
        delay_line[j]=readincr(sig_in);
      }else{
        delay_line[j]=delay_line[j+1];
      }
    }
    cycle_num[1]=tile.cycles();//cycle counter of the AI Engine tile
    printf("cycle start=%d, cycle end=%d, total cycles=%d\n",cycle_num[0],cycle_num[1],(cycle_num[1]-cycle_num[0]));

  }

}
void fir_32tap_scalar_init()
{
  //initialize data
  for (int i=0;i<32;i++){
    int tmp=get_ss(0);
    delay_line[i]=*(cint16*)&tmp;
  }
};
```

**Note:** - Function `fir_32tap_scalar_init` is used as an initialization function for the kernel, which is only called one time after `graph.run()`.
- The scalar processor does not support rounding and saturation modes. You can implement these via standard C operations, like `shift`.
- The tile counter profiles the main loop of code.

From the profiling result, you can see that each sample (one sample per iteration) takes 2804 cycles. You can view the information under the Profile section in the Vitis IDE if you enable the option `--profile` during AI Engine simulation.

**Note:** The profiled cycle can vary when there are different compiler options, location constraints, etc. The profiled cycle can also vary between versions. But the concept of design analysis and performance analysis introduced here still applies.

AI Engine Simulation-Based Performance Analysis

Performance Analysis of AI Engine Graph Application on Hardware

AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment))
