---
title: "Vectorized Matrix Multiplication"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Vectorized-Matrix-Multiplication"
toc_id: y1~~SJR4P6tpB2RGyUpNCQ
content_id: WKTzKzeQ0FZJAwSBTi7bEw
---

### Vectorized Matrix Multiplication

`(64 * 64) x (64 * 64)`

`int8 x int8`

`4*16*8`

```
const int SHIFT=10;

//For element mmul
const int M=4;
const int K=16;
const int N=8;

//Total matrix sizes
const int rowA=64;
const int colA=64;
const int colB=64;

//mmul numbers
const int num_rowA=rowA/M;
const int num_colA=colA/K;
const int num_colB=colB/N;

void matrix_mul(input_buffer<int8> & __restrict matA, input_buffer<int8> & __restrict matB, output_buffer<int8> & __restrict matC){
  using MMUL = aie::mmul<M, K, N, int8, int8>;

  const int8* __restrict pA=(int8*)matA.data();
  const int8* __restrict pB=(int8*)matB.data();
  int8* __restrict pC=(int8*)matC.data();
  //For profiling only
  unsigned cycle_num[2];
  aie::tile tile=aie::tile::current();
  cycle_num[0]=tile.cycles();//cycle counter of the AI Engine tile

  for (unsigned i = 0; i < num_rowA; i++) { //for output row number of element matrix
    for (unsigned j = 0; j < num_colB; j++) { //for output col number of element matrix
      const int8 * __restrict pA1 = pA + ( i * num_colA + 0) * MMUL::size_A;
      const int8 * __restrict pB1 = pB + ( 0 * num_colB + j) * MMUL::size_B;

      aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
      aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;

      MMUL C00; C00.mul(A0, B0);

      for (unsigned k = 0; k < num_colA-1; k++) {
        A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
        B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;
        C00.mac(A0, B0);
      }

      aie::store_v(pC, C00.template to_vector<int8>(SHIFT)); pC += MMUL::size_C;
    }
  }
  //For profiling only
  cycle_num[1]=tile.cycles();//cycle counter of the AI Engine tile
  printf("start=%d,end=%d,total=%d\n",cycle_num[0],cycle_num[1],cycle_num[1]-cycle_num[0]);

}
```

`int8*int8`

`int8*int8`

**Note:** The exact number of cycles can fluctuate slightly based on the specific compiler settings and the version of the tool being used. However, the analysis techniques described in this section remain relevant and applicable regardless of these variations.
