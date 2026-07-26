---
title: "2D Matrix Transpose Using Tiling Parameters"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/2D-Matrix-Transpose-Using-Tiling-Parameters"
toc_id: m4ZIW5lo9ouNXo08DDoG6w
content_id: rsVVRzdY1Il69ru_FVzo~w
---

### 2D Matrix Transpose Using Tiling Parameters

You can use tiling parameters to handle a matrix transpose. A matrix transpose creates a new matrix wherein the rows correspond to the columns of the original matrix.

DMAs can only generate addresses for 32-bit aligned data. This means transpose results are exact for 32-bit (`int32, uint32, cint32, cint16, float, cfloat`) or higher data widths.

The transpose result is only partial and requires further steps for 16-bit, 8-bit, and 4-bit data.

For 16-bit data, matrix elements have to be taken 2x1 when performing the transpose. For 8-bit data, the elements must be taken 4x1. For data widths lower than 32 bits, matrix transpose can be a two step process. The first partial transpose uses the tiling parameters on the memory tile data. The next step is the final transpose of the partial results in the kernel.

In the following code, the kernels are passthrough functions that copy the input data into the output without any modification. The DMA handles the transpose through the read access tiling parameter setting:

```
class Transpose : public adf::graph
{
public:
    adf::kernel k32_1,k16_1,k8_1;
    adf::kernel k32_2,k16_2,k8_2;

    adf::input_plio plin32,plin16,plin8;
    adf::output_plio plout32,plout16,plout8;

    Transpose()
    {

        // Kernel Creation
        k32_1 = adf::kernel::create(Passthrough<int32,Nrows,Ncolumns>);
        k16_1 = adf::kernel::create(Passthrough<int16,Nrows,Ncolumns>);
        k8_1 = adf::kernel::create(Passthrough<int8,Nrows,Ncolumns>);
        k32_2 = adf::kernel::create(Passthrough<int32,Nrows,Ncolumns>);
        k16_2 = adf::kernel::create(Passthrough<int16,Nrows,Ncolumns>);
        k8_2 = adf::kernel::create(Passthrough<int8,Nrows,Ncolumns>);


        adf::source(k32_1) = "Passthrough.cpp";
        adf::runtime<ratio>(k32_1) = 0.9;
        adf::source(k32_2) = "Passthrough.cpp";
        adf::runtime<ratio>(k32_2) = 0.9;
...

        // PLIO Creation
        plin32 = adf::input_plio::create("input64_32", adf::plio_64_bits,
                  "data/Input_32.csv", 625);
        plout32 = adf::output_plio::create("output64_32", adf::plio_64_bits,
                  "data/Output_32.csv", 625);
...

        // Connections

        adf::connect (plin32.out[0],k32_1.in[0]);
        adf::connect(k32_1.out[0],k32_2.in[0]);
        adf::connect(k32_2.out[0],plout32.in[0]);
...

        // Tiling Parameters
        read_access(k32_1.out[0]) =
            adf::tiling({.buffer_dimension={Ncolumns,Nrows},
                  .tiling_dimension={Ncolumns,Nrows},.offset={0,0}});
        write_access(k32_2.in[0]) =
            adf::tiling({.buffer_dimension={Ncolumns,Nrows},
                  .tiling_dimension={1,1},.offset={0,0},
                  .tile_traversal = {{.dimension=1,.stride=1,.wrap=Nrows},
                        {.dimension=0, .stride=1, .wrap=Ncolumns}}});

        read_access(k16_1.out[0]) =
         adf::tiling({.buffer_dimension={Ncolumns,Nrows},.tiling_dimension={Ncolumns,Nrows},.offset={0,0}});
        write_access(k16_2.in[0]) =
            adf::tiling({.buffer_dimension={Ncolumns,Nrows},
                  .tiling_dimension={2,1},.offset={0,0},
                  .tile_traversal = {{.dimension=1,.stride=1,.wrap=Nrows},
                        {.dimension=0, .stride=2, .wrap=Ncolumns/2}}});

        read_access(k8_1.out[0]) =
            adf::tiling({.buffer_dimension={Ncolumns,Nrows},
                  .tiling_dimension={Ncolumns,Nrows},.offset={0,0}});
        write_access(k8_2.in[0]) =
            adf::tiling({.buffer_dimension={Ncolumns,Nrows},
                  .tiling_dimension={4,1},.offset={0,0},
                  .tile_traversal = {{.dimension=1,.stride=1,.wrap=Nrows},
                        {.dimension=0, .stride=4, .wrap=Ncolumns/4}}});


    };
};
```

`.tiling_dimension={2,1}`

AI Engine

```
Failed in converting tiling parameters to buffer descriptors for port T.mtx16.out[0]: In dimension 0, stepsize 8 and element size 16 (in bytes) cannot be adjusted to 32 bit word address. In dimension 1, stepsize 1 and element size 16 (in bytes) cannot be adjusted to 32 bit word address.
```

A step size of 1 with a 16-bit data width generates 16-bit aligned addresses, which are unsupported.

The following figure provides a visualization of the effect of this transpose function with 32-bit, 16-bit and 8-bit data:

![zsc1696942126965.png](../assets/082-01-zsc1696942126965-png-2a20ffa6f302.png)

*Figure 1. 32-bit, 16-bit and 8-bit Results For An 8x8 Matrix*

This example transposes all matrices in various ways depending on data bitwidth. The color coding helps understand how the data are shuffled around. The effect on the 32-bit data is clearly a transpose function. 16-bit and 8-bit data undergo a partial transpose. Multiple data on each row (contiguous on the column dimension) are kept contiguous on the column dimension. This can be used if you have multiple matrices that are interleaved in the column dimension. For example, for 16-bit data you can have two matrices (or 4, 6, 8, …) where element (0,0) of the first matrix is followed by the exact same element of the second matrix, and so on for element (0,1).

You can transpose all matrices at once using tiling parameters if each concatenated element's bit width is a multiple of 32 bits.
