---
title: "Programming Model for AI Engine–DDR Memory Connection"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Programming-Model-for-AI-Engine-DDR-Memory-Connection"
toc_id: 0zCVhArh9RTskJMY081ZmA
content_id: FexA3LHKQtLa00DuvETAag
---

### Programming Model for AI Engine–DDR Memory Connection

You can use the `input_gmio`/`output_gmio` port attribute to initiate AI Engine–DDR memory read and write transactions in the PS program. This enables data transfer between an AI Engine and the DDR controller through APIs written in the PS program. The following example shows how to use GMIO APIs to send data to an AI Engine for processing. It also shows how to retrieve the processed data back to the DDR through the PS program.

graph.h

```
class mygraph: public adf::graph
{
private:
  adf::kernel k_m;

public:
  adf::output_gmio gmioOut;
  adf::input_gmio gmioIn;
  mygraph()
  {
    k_m = adf::kernel::create(weighted_sum_with_margin);
    gmioOut = adf::output_gmio::create("gmioOut", 64, 1000);
    gmioIn = adf::input_gmio::create("gmioIn", 64, 1000);

    adf::connect(gmioIn.out[0], k_m.in[0]);
    adf::connect(k_m.out[0], gmioOut.in[0]);

    dimensions(k_m.in[0]) = {256};
    dimensions(k_m.out[0]) = {256};

    adf::source(k_m) = "weighted_sum.cc";
    adf::runtime<adf::ratio>(k_m)= 0.9;
  };
};
```

graph.cpp

```
myGraph gr;
int main(int argc, char ** argv)
{
    const int BLOCK_SIZE=256;
    int32 *inputArray=(int32*)GMIO::malloc(BLOCK_SIZE*sizeof(int32));
    int32 *outputArray=(int32*)GMIO::malloc(BLOCK_SIZE*sizeof(int32));

    // provide input data to AI Engine in inputArray
    for (int i=0; i<BLOCK_SIZE; i++) {
        inputArray[i] = i;
    }

    gr.init();

    gr.gmioIn.gm2aie_nb(inputArray, BLOCK_SIZE*sizeof(int32));
    gr.gmioOut.aie2gm_nb(outputArray, BLOCK_SIZE*sizeof(int32));

    gr.run(8);

    gr.gmioOut.wait();

    // can start to access output data from AI Engine in outputArray
	...

    GMIO::free(inputArray);
    GMIO::free(outputArray);
    gr.end();
}
```

This example declares two I/O objects:

- `gmioIn` represents the DDR memory space to be read by the AI Engine
- `gmioOut` represents the DDR memory space to be written by the AI Engine

The constructor specifies the following:

- Logical name of the GMIO
- Burst length (that can be 64, 128, or 256 bytes) of the memory-mapped AXI4 transaction
- Required bandwidth (in MB/s)

```
gmioOut = adf::output_gmio::create("gmioOut", 64, 1000);
gmioIn  = adf::input_gmio::create("gmioIn", 64, 1000);
```

The application graph (`myGraph`) has an input port (`myGraph::gmioIn`) connecting to the processing kernels. The kernels produce data to the output port (`myGraph::gmioOut`) producing the processed data from the kernels. The following code connects the input port of the graph and connects to the output port of the graph.

```
adf::connect(gmioIn.out[0], k_m.in[0]);
adf::connect(k_m.out[0], gmioOut.in[0]);
dimensions(k_m.in[0]) = {256};
dimensions(k_m.out[0]) = {256};
```

Inside the main function, `GMIO::malloc`allocates two 256-element int32 arrays. The `inputArray` points to the memory space to be read by the AI Engine and the `outputArray` points to the memory space to be written by the AI Engine. In Linux, the virtual address passed to `GMIO::gm2aie_nb`, `GMIO::aie2gm_nb`, `GMIO::gm2aie` and `GMIO::aie2gm` must be allocated by `GMIO::malloc`. After the input data is allocated, it can be initialized.

```
const int BLOCK_SIZE=256;
int32 *inputArray=(int32*)GMIO::malloc(BLOCK_SIZE*sizeof(int32));
int32 *outputArray=(int32*)GMIO::malloc(BLOCK_SIZE*sizeof(int32));
```

`gr.gmioIn.gm2aie_nb()`

AXI4

AI Engine

`gr.gmioIn.gm2aie_nb()`

`GMIO::malloc`

`gr.gmioOut.aie2gm_nb()`

AXI4

AI Engine

`gr.gmioOut.gm2aie_nb()`

`gr.gmioOut.aie2gm_nb()`

`gr.gmioIn.gm2aie()`

`gr.gmioOut.aie2gm()`

AXI4

`gr.gmioIn.gm2aie_nb()`

`gr.gmioOut.aie2gm_nb()`

```
gr.gmioIn.gm2aie_nb(inputArray, BLOCK_SIZE*sizeof(int32));
gr.gmioOut.aie2gm_nb(outputArray, BLOCK_SIZE*sizeof(int32));
```

`gr.run(8)` is also a non-blocking call to run the graph for eight iterations. Use `gr.gmioOut.wait()` to synchronize between the PS and AI Engine for DDR memory read/write access. This blocks PS execution until the GMIO transaction is complete. In this example, `gr.gmioOut.wait()` is called to wait for the output data to be written to `outputArray` DDR memory space.

**Note:** The memory is non-cachable for GMIO in Linux.

After that, the PS program can access the data. When PS processing completes, release the memory space allocated by `GMIO::malloc` using `GMIO::free`.

```
GMIO::free(inputArray);
GMIO::free(outputArray);
```

You can use the `input_gmio`/`output_gmio` APIs in various ways to control for read/write access. These APIs also support synchronization between the AI Engine, PS, and DDR memory. You can call `input_gmio::gm2aie`, `output_gmio::aie2gm`, `input_gmio::gm2aie_nb`, or `output_gmio::aie2gm_nb` multiple times. This lets you associate different memory spaces with the same `input_gmio` or `output_gmio` object during different graph execution phases. Different `input_gmio`/`output_gmio` objects can be associated with the same memory space for in-place AI Engine–DDR read/write access. Blocking versions of `input_gmio::gm2aie` and `output_gmio::aie2gm` APIs themselves are synchronization point for data transportation and kernel execution. Calling `input_gmio::gm2aie` (or `output_gmio::aie2gm`) is equivalent to calling `input_gmio::gm2aie_nb` (or `output_gmio::aie2gm_nb`) followed immediately by `output_gmio::wait`. The following example shows the combination of the aforementioned use cases.

```
myGraph gr;

 int main(int argc, char ** argv)
{

    const int BLOCK_SIZE=256;
    // dynamically allocate memory spaces for in-place AI Engine read/write access
    int32* inoutArray=(int32*)GMIO::malloc(BLOCK_SIZE*sizeof(int32));
    gr.init();

    for (int k=0; k<4; k++)
    {
        // provide input data to AI Engine in inoutArray
        for(int i=0;i<BLOCK_SIZE;i++){
            inoutArray[i]=i;
        }

        gr.run(8);
        for (int i=0; i<8; i++)
        {
            gr.gmioIn.gm2aie(inoutArray+i*32, 32*sizeof(int32)); //blocking call to ensure transaction data is read from DDR to AI Engine
            gr.gmioOut.aie2gm_nb(inoutArray+i*32, 32*sizeof(int32));
        }
        gr.gmioOut.wait();

        // can start to access output data from AI Engine in inoutArray
	//	...
    }
    GMIO::free(inoutArray);
    gr.end();
    return 0;
}
```

The previous example shows two GMIO objects `gmioIn` and `gmioOut` using the same memory space allocated by `inoutArray` for in-place read and write access.

Without knowing data flow dependency among the kernels inside the graph, and to ensure write-after-read for the `inoutArray` memory space, the blocking version `gr.gmioIn.gm2aie()` is called to ensure transaction data is copied from DDR memory to AI Engine local memory before issuing a write transaction to the same memory space in `gr.gmioOut.aie2gm_nb()`.

```
gr.gmioIn.gm2aie(inoutArray+i*32, 32*sizeof(int32)); //blocking call to ensure transaction data is read from DDR to AI Engine
gr.gmioOut.aie2gm_nb(inoutArray+i*32, 32*sizeof(int32));
```

`gr.gmioOut.wait()` ensures that is migrated to DDR memory. After migration is complete, the PS can access output data for post-processing.

The for loop divides graph execution into four phases, `for (int k=0; k<4; k++)`. `inoutArray` can be re-initialized in the `for` loop with different data to be processed in different phases.
