---
title: "Asynchronous Buffer Port Access"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Asynchronous-Buffer-Port-Access"
toc_id: StFTRTI4cjvM6lCIlPHqng
content_id: 2vdkmUxaZ0erIobzWo~2BA
---

### Asynchronous Buffer Port Access

In some situations, if you are not consuming a buffer port worth of data on every invocation of a kernel, or if you are not producing a buffer port worth of data on every invocation, then you can control the buffer synchronization by declaring the kernel port using `async` to declare the async buffer port in kernel function prototype. The following example illustrates that the kernel `simple` uses these buffer ports:

- **`ifm`:** Synchronous input buffer port.
- **`wts`:** Asynchronous input buffer port.
- **`ofm`:** Asynchronous output buffer port.

The following declaration informs the compiler to omit synchronization of the buffer named `wts` upon entry to the kernel. You must use the buffer port synchronization member function shown inside the kernel code before accessing the buffer port using read/write iterators/references, as shown below.

```
void simple(adf::input_buffer<uint8>& ifm, adf::input_async_buffer<uint8>& wts, adf::output_async_buffer<uint8>& ofm)
{
    ...
    wts.acquire(); // acquire lock unconditionally inside the kernel
    if (<somecondition>) {
        ofm.acquire(); // acquire output buffer conditionally
    }
    ... // do some computation
    wts.release(); // release input buffer port inside the kernel
    if (<somecondition>) {
        ofm.release(); // release output buffer port conditionally
    }
    ...
};
```

The `acquire()` member function of the buffer object `wts` performs the appropriate synchronization and initialization to ensure that the buffer port object is available for read or write. This function keeps track of the appropriate buffer pointers and locks to be acquired internally, even if the buffer port is shared across AI Engine processors and can be double buffered. This function can be called unconditionally or conditionally under dynamic control and is potentially a blocking operation. It is your responsibility to ensure that the corresponding `release()` member function is executed sometime later (possibly even in a subsequent kernel call) to release the lock associated with that buffer object. Incorrect synchronization can lead to a deadlock in your code.

**Note:** Important:

`acquire()`

In the following example, the kernel located in tile 1 requests a lock acquisition (write access) three times per each run. The kernel located in tile 2 requests a lock acquisition (read access) twice per each run.

![ksd1690982834888.png](../assets/092-01-ksd1690982834888-png-71cd5ad9cfc9.png)

*Figure 1. Lock Mechanism for Asynchronous Ping-pong Buffer Access*

The lock acquisition and release is a kernel-only process. The `main` function is not taking care of the buffer synchronization; buffer synchronization is the user responsibility. Kernel in tile 1 requests three times the access to the ping pong buffer and tile 2 only twice. To balance the number of accesses, run tile 1 twice, and run tile 2 three times per iteration.

As seen in the figure, the lock acquisition occurs alternatively on the ping then pong buffer. The buffer choice is automatic. No user action is required at this stage.

Lock acquisition has a minimum latency of seven clock cycles during which the kernel is stalled. If the buffer unavailable for acquisition, the kernel stalls for a longer time (indicated in red in the figure) until the buffer is available. Depending on the application, there can be time intervals when the ping and/or the pong buffer is not locked.

For an asynchronous buffer port, the `acquire` and `release` APIs explicitly acquire and release the buffer port of the kernel. You can release the asynchronous output buffer anytime inside the kernel using the `release` API, regardless of how many samples the kernel writes to the buffer. After the port is released, the asynchronous output buffer can be acquired by its consumer kernel or it can be transferred by DMA to its destination, such as PLIO.

Consider a system with one producer AI Engine kernel and one consumer AI Engine kernel, communicating via asynchronous buffers. Initially, there are two empty buffers between the producer and the consumer.

From the producer's perspective:

Each time the producer wants to write data to a buffer, it must first call the `acquire` API. When acquired, the producer owns the buffer. The producer can read from or write to the buffer as required. After finishing the operation—either in the same iteration or later—it must call the `release` API to release the buffer. Once released, the buffer becomes available to the consumer, increasing the count of full buffers. If both buffers are full, any subsequent `acquire` call by the producer blocks until an empty buffer becomes available.

From the consumer's perspective:

The consumer must also call the `acquire` API before accessing a buffer. After acquiring, it owns the buffer and can read from or write to it. Once finished, it calls the `release` API to release the buffer, making it available for the producer again and increasing the count of empty buffers. If both buffers are empty, the consumer stalls when trying to acquire a buffer, until one becomes full.

In this system, PLIO or GMIO can also act as producers or consumers. DMA manages data exchange between PLIO/GMIO and the AI Engine. DMA handles buffer availability transparently. Data can only be sent or received when the corresponding buffer is ready (that is, empty for writing or full for reading).
