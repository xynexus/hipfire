---
title: "Examples"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Examples"
toc_id: Ey2EuirZxHrQeM~F9I~vlw
content_id: N1T8E5W0sxl_dS39i2a5bQ
---

#### Examples

###### Example of begin Iterator

```
#include "aie_api/aie.hpp"
void simple(adf::input_buffer<cint16, adf::extents<BUFFER_SIZE>> & in, adf::output_buffer<cint16, adf::extents<BUFFER_SIZE>> & out)
{
    // Use scalar iterator to traverse data
    auto pIn  = aie::begin(in);
    auto pOut = aie::begin(out);

    // For loop to go through all data from input_buffer via iterator
    for (unsigned i=0; i<(BUFFER_SIZE); i++)
    {
        *pOut++ = *pIn++;
    }
}
```

###### Example of begin_vector Iterator

```
#define VECTOR_SIZE 8
void simple(adf::input_buffer<cint16, adf::extents<BUFFER_SIZE>> & in, adf::output_buffer<cint16, adf::extents<BUFFER_SIZE>> & out)
{
    // Use vectoriterator to traverse data
    auto pIn  = aie::begin_vector<VECTOR_SIZE>(in);
    auto pOut = aie::begin_vector<VECTOR_SIZE>(out);

    // For loop to go through all data from input_buffer via iterator
    for (unsigned i=0; i<(BUFFER_SIZE/VECTOR_SIZE); i++)
    {
        *pOut++ = *pIn++;
    }
}
```

###### Example of cbegin Iterator

```
void simple(adf::input_buffer<cint16, adf::extents<BUFFER_SIZE>> & in, adf::output_buffer<cint16, adf::extents<BUFFER_SIZE>> & out)
{
    // Use scalar iterator to traverse data
    auto pIn  = aie::cbegin(in);
    auto pOut = aie::begin(out);

    // For loop to go through all data from input_buffer via iterator
    for (unsigned i=0; i<(BUFFER_SIZE); i++)
    {
        *pOut++ = *pIn++;
    }
}
```

**Note:** `cbegin`

AI Engine

`pOut`

`cbegin`

###### Example of cbegin_vector Iterator

```
#define VECTOR_SIZE 8
void simple(adf::input_buffer<cint16, adf::extents<BUFFER_SIZE>> & in, adf::output_buffer<cint16, adf::extents<BUFFER_SIZE>> & out)
{
    // Use vector iterator to traverse data
    auto pIn  = aie::cbegin_vector<VECTOR_SIZE>(in);
    auto pOut = aie::begin_vector<VECTOR_SIZE>(out);

    // For loop to go through all data from input_buffer via vector iterator
    // The buffer contains  (BUFFER_SIZE/VECTOR_SIZE) vectors
    for (unsigned i=0; i<(BUFFER_SIZE/VECTOR_SIZE); i++)
    {
        *pOut++ = *pIn++;
    }
}
```

###### Example of begin_random_circular Iterator

```
void simple(adf::input_circular_buffer<cint16, adf::extents<BUFFER_SIZE>> & in, adf::output_circular_buffer<cint16, adf::extents<BUFFER_SIZE>> & out)
{
    // Use scalar iterator to traverse data
    auto pIn  = aie::begin_random_circular(in);
    auto pOut = aie::begin_random_circular(out);

    // Position the pointer at the middle of the buffer
    pIn += BUFFER_SIZE/2;
    // Copies the second half, then the first half of the buffer onto the output
    for (unsigned i=0; i<(BUFFER_SIZE); i++)
    {
        *pOut++ = *pIn++;
    }
}
```

###### Example of begin_vector_random_circular Iterator

```
#define VECTOR_SIZE 8
void simple(adf::input_circular_buffer<cint16, adf::extents<BUFFER_SIZE>> & in, adf::output_circular_buffer<cint16, adf::extents<BUFFER_SIZE>> & out)
{
    // Use vector iterator to traverse data
    auto pIn  = aie::begin_vector_random_circular<VECTOR_SIZE>(in);
    auto pOut = aie::begin_vector_random_circular<VECTOR_SIZE>(out);

    // Position the pointer at the end of the buffer
    pIn += BUFFER_SIZE/VECTOR_SIZE;
    // Copies the input buffer onto the output buffer
    for (unsigned i=0; i<(BUFFER_SIZE/VECTOR_SIZE); i++)
    {
        *pOut++ = *pIn++;
    }
}
```

- Circular buffer ports must use circular iterators. Linear buffer ports can use linear iterators or circular iterators.Warning: If two kernels are connected by buffers (output from the first kernel, and input to the second kernel), and mapped to the same tile, use either linear or circular addressing for both kernels. If the buffers are not configured for linear or circular addressing, x86 Simulation gives correct simulation output while AI Engine Simulation is incorrect. This difference is due to optimizations in memory organization that are performed in hardware implementation.
- Use vector iterators to access `VECTOR_SIZE` samples for each iteration. Where `VECTOR_SIZE` is 4 (128 bits), 8 (256 bits), 16 (512 bits), and 32 (1024 bits) for this specific example where the data-type is cint16 (32 bits).
- Use random iterators when iterators need to be moved more than one step at a time in either direction.

###### Reading and Writing Data

There are several ways of reading and writing data to a one dimensional buffer port.

- Using a raw pointer. Do not use this mechanism with circular buffer ports.`void simple(adf::input_buffer<int32> & in, adf::output_buffer<int32> & out) { int32 * pin = in.data(); int32 * pout = out.data(); for (int i = 0; i < BUFFER_SIZE; i++) { *pout++ = *pin++; } ... }`

  ```
  void simple(adf::input_buffer<int32> & in, adf::output_buffer<int32> & out) {
    int32 * pin = in.data();
    int32 * pout = out.data();
    for (int i = 0; i < BUFFER_SIZE; i++) {
      *pout++ = *pin++;
    }
    ...
  }
  ```
- Using a scalar iterator.`void simple(adf::input_buffer<int32> & in, adf::output_buffer<int32> & out) { auto pin = aie:begin(in); auto pout = aie:begin(out); for (int i = 0; i < BUFFER_SIZE; i++) { *pout++ = *pin++; } ... }`

  ```
  void simple(adf::input_buffer<int32> & in, adf::output_buffer<int32> & out) {
    auto pin = aie:begin(in);
    auto pout = aie:begin(out);
    for (int i = 0; i < BUFFER_SIZE; i++) {
      *pout++ = *pin++;
    }
    ...
  }
  ```
- Using a vector iterator.`void simple(adf::input_buffer<int32> & in, adf::output_buffer<int32> & out) { auto pin = aie::begin_vector<VECTOR_SIZE>(in); auto pout = aie::begin_vector<VECTOR_SIZE>(out); for (int i = 0; i < BUFFER_SIZE/VECTOR_SIZE; i++) { *pout++ = *pin++; } ... }`

  ```
  void simple(adf::input_buffer<int32> & in, adf::output_buffer<int32> & out) {
    auto pin = aie::begin_vector<VECTOR_SIZE>(in);
    auto pout = aie::begin_vector<VECTOR_SIZE>(out);
    for (int i = 0; i < BUFFER_SIZE/VECTOR_SIZE; i++) {
      *pout++ = *pin++;
    }
    ...
  }
  ```

###### Using Input and Output Buffer as Intermediate Storage

After acquiring an input or output buffer but before releasing it, the buffer is owned by the kernel. The kernel can be responsible to read or write to the buffer by pointer or iterator without conflicting the data. The following code shows an example of an asynchronous output buffer being used for temporary storage between iterations:

```
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
const int BUFFER_SIZE=1024;
const int VECTOR_SIZE=16;
const int TOTAL_N=2;
static int iteration=0;
__attribute__ ((noinline)) void accumulation(adf::input_buffer<int32,extents<BUFFER_SIZE>> & __restrict in1,
		adf::input_buffer<int32,extents<BUFFER_SIZE>> & __restrict in2,
		adf::output_async_buffer<int32,extents<BUFFER_SIZE>> & __restrict out
){
	auto pin1 = aie::begin_vector<VECTOR_SIZE>(in1);
	auto pin2 = aie::begin_vector<VECTOR_SIZE>(in2);

	if(iteration==0){
		out.acquire();
                //must be done after lock acquisition
		auto pout = aie::begin_vector<VECTOR_SIZE>(out);
		for (int i = 0; i < BUFFER_SIZE/VECTOR_SIZE; i++) {
			*pout++ = aie::add(*pin1++, *pin2++);
		}
		iteration++;
	}else{
		auto pout=aie::begin_vector<VECTOR_SIZE>(out);//lock acquired
		for (int i = 0; i < BUFFER_SIZE/VECTOR_SIZE; i++) {
			auto tmp = aie::add(*pin1++, *pin2++);
			auto tmp2=*pout;
			*pout++ = aie::add(tmp2, tmp);
		}
		iteration++;
	}
	if(iteration==TOTAL_N){
		iteration=0;
		out.release();
	}
}
```
