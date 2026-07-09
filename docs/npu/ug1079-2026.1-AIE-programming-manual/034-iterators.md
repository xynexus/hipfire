---
title: "Iterators"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Iterators"
toc_id: 6y9yvFYt3hAJvgo9s7W5iQ
content_id: QzWxQE1rzIE6mMm8JSgBKw
---

## Iterators

The AI Engine API provides iterators to access AI Engine memory. The AI Engine API provides iterators to iterate over both scalar and vector data types. In addition, the API also provides circular iterators. The following table lists the different types of iterators provided by the AI Engine API.

| Iterator | Description |
| --- | --- |
| aie::begin | Returns a scalar iterator. |
| aie::cbegin | Returns a scalar iterator with read-only access. |
| aie::begin_circular | Returns a scalar circular iterator. You can specify the circular buffer size. |
| aie::cbegin_circular | Returns a scalar iterator with read-only access. |
| aie::begin_random_circular | Returns a scalar circulator iterator with random access. |
| aie::cbegin_random_circular | Returns a scalar circulator iterator with random, read-only access. |
| aie::begin_vector | Returns a vector iterator. You can specify the size of the vector. |
| aie::begin_restrict_vector | Same as aie::begin_vector, but the return iterator is considered restrict. |
| aie::cbegin_vector | Returns a vector iterator with read-only access. |
| aie::cbegin_restrict_vector | Same as aie::cbegin_vector, but the return iterator is considered restrict. |
| aie::begin_restrict_vector | Returns a vector iterator that is a restrict type. |
| aie::cbegin_restrict_vector | Returns a vector iterator that is a restrict type with read-only access. |
| aie::begin_vector_circular | Returns a circular vector iterator. You can specify the elementary vector size and total circular buffer size. |
| aie::cbegin_vector_circular | Returns a circular vector iterator with read-only access. |
| aie::begin_vector_random_circular | Returns a circular iterator for the array described by the given address and size. |
| aie::cbegin_vector_random_circular | Returns a circular iterator for the array described by the given address and size, with read-only access. |

Iterators fall into two categories: forward iterators and random access iterators. All iterators support the following:

- dereferencing `*`
- operator `++`
- operator `==`
- operator `!=`

Random access iterator supports more operators. For example, `aie::begin_vector` and `aie::begin_vector_random_circular` also support the following:

- operator `--`
- operator `+`
- operator `-`
- operator `+=`
- operator `-=`

For more information about the type of each iterator, see [Memory](https://www.xilinx.com/htmldocs/xilinx2024_2/aiengine_api/aie_api/doc/group__group__memory.html) in the AI Engine API User Guide ([UG1529](https://www.xilinx.com/htmldocs/xilinx2025_2/aiengine_api/aie_api/doc/index.html)).

```
alignas(aie::vector_decl_align) int32 init_value[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
auto iIter = aie::begin(init_value);
for(int i=0;i<16;i++, iIter++){
  *iIter=*iIter+1;
}
```

```
alignas(aie::vector_decl_align) int32 init_value[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

// create vector iterator
// first value is [1,2,3,4]
auto pv=aie::begin_vector<4>(init_value);

// create const circular vector iterator
// first value is [1,2,3,4]
// total 16 elements
auto pv_c=aie::cbegin_vector_circular<4,16>(init_value);
for(;pv!=init_value+16;pv++){
  aie::vector<int32,4> buff=*pv;
  aie::print(*pv,true,"pv:");
}
for(int i=0;i<5;i++){

  // go back to start
  // if reach the end of the circular buffer
  aie::vector<int32,4> buff=*pv_c++;
}

// point to circular buffer size 16.
auto p=aie::begin_circular<16>(init_value);

for(int i=0;i<N;i++){

  // return back to init_value+0
  // if reaches init_value+16
  *p++=i;
}
```

```
__attribute__ ((noinline)) void pass_through(input_buffer<int> & __restrict in,
    output_buffer<int> & __restrict out
    ){
  auto inIter1=aie::cbegin_restrict_vector<8>(in);
  auto outIter=aie::begin_restrict_vector<8>(out);
  for(int i=0;i<32;i++) {
    *outIter++=*inIter++;
  }
}
```
