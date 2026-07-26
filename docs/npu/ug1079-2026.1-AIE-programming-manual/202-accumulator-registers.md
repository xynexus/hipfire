---
title: "Accumulator Registers"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Accumulator-Registers"
toc_id: 5611fBsRB4YsvMqgjoH_Og
content_id: DK5qjQkAdrd7I1I_vJWthA
---

## Accumulator Registers

The accumulation registers are 384 bits wide and can be viewed as eight vector lanes of 48 bits each. The idea is to have 32-bit multiplication results and accumulate over those results without bit overflows. The 16 guard bits allow up to 216 accumulations. The accumulator registers store the output of fixed-point vector MAC and MUL intrinsic functions. The following table shows the set of accumulator registers and how smaller registers combine to form large registers.

| 384-bit | 768-bit |
| --- | --- |
| aml0 | bm0 |
| amh0 |  |
| aml1 | bm1 |
| amh1 |  |
| aml2 | bm2 |
| amh2 |  |
| aml3 | bm3 |
| amh3 |  |

The letters 'am' prefix the accumulator registers. Two of the registers are aliased to form a 768-bit register which uses the prefix 'bm'.

The shift-round-saturate `srs()` intrinsic moves a value from an accumulator register to a vector register with any required shifting and rounding.

```
aie::vector<int32,8> res = srs(acc, 8); // shift right 8 bits, from accumulator register to vector register
```

The upshift `ups()` intrinsic moves a value from an vector register to an accumulator register with upshifting:

```
aie::accum<acc48,8> acc = ups(v, 8); //shift left 8 bits, from vector register to accumulator register
```

The `set_rnd()` and `set_sat()` instrinsics set the rounding and saturation mode of the accumulation result. The `clr_rnd()` and `clr_sat()` intrinsics clear the rounding and saturation mode, that is, truncate the accumulation result.

Note that shifting, rounding, or saturation modes are only active when operations use the shift-round-saturate data path. Some intrinsics only use the vector pre-adder operations, where there is no shifting, rounding, or saturation mode for configuration. Such operations are adds, subs, abs, vector compares, or vector selections/shuffles. It is possible to choose MAC intrinsics instead to do subtraction with shifting, rounding, or saturation mode configuration. The following code performs subtraction between `va` and `vb` with `mul` instead of `sub` intrinsics.

```
v16cint16 va, vb;
int32 zbuff[8]={1,1,1,1,1,1,1,1};
aie::vector<int32,8> coeff=*(v8int32*)zbuff;
aie::accum<acc48,8> acc = mul8_antisym(va, 0, 0x76543210, vb, 0, false, coeff, 0 , 0x76543210);
aie::vector<int32,8> res = srs(acc,0);
```

Floating-point intrinsic functions do not have separate accumulation registers and instead return their results in a vector register. You can use the following streaming data APIs can be used to read and write floating-point accumulator data from or to the cascade stream.

```
aie::vector<float,8> readincr_v<8>(input_cascade<accfloat> * str);
aie::vector<cfloat,4> readincr_v4(input_cascade<caccfloat> * str);
void writeincr(output_cascade<accfloat>* str, v8float value);
void writeincr(output_cascade<caccfloat>* str, v4cfloat value);
```

For more information about the window and streaming data APIs, refer to Input and Output Buffers

The data size in memory is aligned to the next power of 2 (here 64b for `acc48`), hence it is best to use `sizeof` to determine the position on the elements. The following code is an example to print accumulator vector registers.

```
aie::accum<acc48,8> vacc; //cascade value
const int SIZE_ACC48=sizeof(aie::accum<acc48,8>)/8;
for(int i=0;i<8;i++){//8 number
  int8 *p=(int8*)&vacc+SIZE_ACC48*i;//point to start of each acc48
  printf("acc value[%d]=0x",i);
  for(int j=5;j>=0;j--){//print each acc48 from higher byte to lower byte
    printf("%02x",*(p+j));
  }
  printf("\n");
}
```

The output is as follows.

```
acc value[0]=0x000000000000
acc value[1]=0x000000000001
acc value[2]=0x000000000002
acc value[3]=0x000000000003
acc value[4]=0x000000000004
acc value[5]=0x000000000005
acc value[6]=0x000000000006
acc value[7]=0x000000000007
```
