---
title: "Data Shuffling Kernel"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Data-Shuffling-Kernel"
toc_id: rh0psaL8tK8MUSqhwvICAw
content_id: QidLlpE~6tTS5rsiZuibnA
---

### Data Shuffling Kernel

`aie::mmul` accepts row-major format vector data for shaping matrix multiplication. Therefore it might require data shuffling in PL or AI Engine with raw data for performance. This section assumes that the original data is row-major format for whole matrices. It shuffles the data to match the shape `4*16*8` used in the matrix multiplication.

`4*16`

```
//element matrix size
const int M=4;
const int N=16;

//Total matrix sizes
const int rowA=64;
const int colA=64;

void shuffle_4x16(input_buffer<int8> & __restrict matA, output_buffer<int8> & __restrict matAout){

    const int sizeA=M*N;
    auto pV=aie::begin_vector<16>((int8*)matA.data());
    auto pOut=aie::begin_vector<sizeA>((int8*)matAout.data());

    aie::vector<int8,sizeA> mm;
    for(int i=0;i<rowA/M;i++){
      for(int j=0;j<colA/N;j++){
        for(int k=0;k<M;k++){
          mm.insert(k,*pV);
          pV=pV+4;
        }
        *pOut++=mm;
        pV=pV-15;
      }
      pV=pV+12;
    }
}
```

`16*8`

```
//element matrix size
const int M=16;
const int N=8;

//Total matrix sizes
const int rowA=64;
const int colA=64;

void shuffle_16x8(input_buffer<int8> & __restrict matA, output_buffer<int8> & __restrict matAout){

  const int sizeA=M*N;
  auto pV=aie::begin_vector<16>((int8*)matA.data());
  auto pOut=aie::begin_vector<16>((int8*)matAout.data());

  aie::vector<int8,16> sv1,sv2;
  for(int i=0;i<rowA/M;i++){
    for(int j=0;j<colA/N/2;j++){
      for(int k=0;k<M/2;k++){
          sv1=*pV;
          pV=pV+4;
          sv2=*pV;
          pV=pV+4;
          auto mm=aie::interleave_zip(sv1,sv2,8);
          *pOut=mm.first;
          pOut+=8;
          *pOut=mm.second;
          pOut-=7;
      }
      pOut+=8;
      pV-=63;
    }
    pV+=60;
  }
}
```

`4*8`

```
//element matrix size
const int M=4;
const int N=8;

//Total matrix sizes
const int rowA=64;
const int colA=64;

void shuffle_4x8(input_buffer<int8> & __restrict matA, output_buffer<int8> & __restrict matAout){
  const int sizeA=M*N;
  auto pV=aie::begin_vector<sizeA>((int8*)matA.data());
  auto pOut=aie::begin_vector<sizeA>((int8*)matAout.data());

  aie::vector<int8,sizeA> mm1,mm2,mm3,mm4;
  for(int i=0;i<rowA/M;i++){
    for(int j=0;j<colA/N/4;j++){
      mm1=*pV++;
      mm2=*pV++;
      mm3=*pV++;
      mm4=*pV++;
      auto mm12=aie::interleave_zip(mm1,mm2,8);
      auto mm34=aie::interleave_zip(mm3,mm4,8);
      auto mm1234_low=aie::interleave_zip(mm12.first,mm34.first,16);
      auto mm1234_high=aie::interleave_zip(mm12.second,mm34.second,16);
      *pOut=mm1234_low.first;
      pOut=pOut+2;
      *pOut=mm1234_low.second;
      pOut=pOut+2;
      *pOut=mm1234_high.first;
      pOut=pOut+2;
      *pOut=mm1234_high.second;
      pOut=pOut-5;
    }
    pOut=pOut+6;
  }
}
```
