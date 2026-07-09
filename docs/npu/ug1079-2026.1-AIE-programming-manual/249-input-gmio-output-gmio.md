---
title: "input_gmio/output_gmio"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/input_gmio/output_gmio"
toc_id: Z8j6JsgPN~TwN_BL4AFTcw
content_id: ai76iDCrwXFEyEvu4yzf2g
---

### input_gmio/output_gmio

This class represents the global memory (DDR) resource management and data transfer between AI Engine and global memory (DDR). The `input_gmio` object manages data transfer from global memory (DDR) to AI Engine or read from global memory (DDR) operation. The `output_gmio` object manages data transfer from AI Engine to global memory or write to global memory (DDR) operation.

##### Base Class Member Functions

```
static void* malloc(size_t size);
```

`malloc`

`size`

`nullptr`

```
static void free(void* address);
```

`free`

`GMIO::malloc`

```
return_code wait();
```

`wait`

`GMIO`

AI Engine

**Note:** `input_gmio`

`output_gmio`

`malloc()`

`free()`

`wait()`

##### input_gmio Member Functions

```
create(std::string logical_name, size_t burst_length, size_t bandwidth);
```

The above port specification connects DDR memory to AI Engine kernels. `logical_name` is the name of the port. The `burst_length` is the length of the DDR memory burst transaction. This can be 64, 128 or 256 bytes. The `bandwidth` is the average expected throughput in MB/s.

```
create(size_t burst_length, size_t bandwidth);
```

AI Engine

`burst_length`

`bandwidth`

```
return_code gm2aie_nb(const void* address, size_t transaction_size);
```

`gm2aie_nb`

AI Engine

`GMIO::malloc`

```
return_code gm2aie(void* address, size_t transaction_size);
```

The `gm2aie` method is a blocking version of `gm2aie_nb`. It blocks until the AI Engine–DDR read transaction completes.

##### output_gmio Member Functions

```
create(std::string logical_name, size_t burst_length, size_t bandwidth);
```

AI Engine

`logical_name`

`burst_length`

`bandwidth`

```
create(size_t burst_length, size_t bandwidth);
```

AI Engine

`burst_length`

`bandwidth`

```
return_code aie2gm_nb(void* address, size_t transaction_size);
```

`aie2gm_nb`

AI Engine

`GMIO::malloc`

```
return_code aie2gm(void* address, size_t transaction_size);
```

The `aie2gm` method is a blocking version of `aie2gm_nb`. It blocks until the AI Engine–DDR write transaction completes.

**Note:** `GMIO::malloc()`

`GMIO::free()`
