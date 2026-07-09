---
title: "Programming Model Features"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Programming-Model-Features"
toc_id: iouc95PrJaaticX3n1yZVw
content_id: VRSYGtjW9N6YQLvcSUnQTA
---

## Programming Model Features

The AI Engine programming model has the following features.

- All I/O objects (`input_plio`/`output_plio` and`input_gmio`/`output_gmio`) are public members of graph class. Looping through I/O objects is supported. This makes I/O objects construction easier and less error prone. For example:`class mygraph : public graph { public: // Declare N kernels. kernel k[N]; // Declare N input_gmio(s) and output_gmio(s) input_gmio gmioIn[N]; output_gmio gmioOut[N]; mygraph() { // Create multiple input_gmio(s)/output_gmio(s) and connections for (int i=0; i<N; i++) { gmioIn[i] = input_gmio::create("gmioIn" + std::to_string(i), 64, 1); gmioOut[i] = output_gmio::create("gmioOut" + std::to_string(i), 64, 1); connect(gmio[i].out[0], k[i].in[0]); connect(k[i].out[0], gmioOut[i].in[0]); dimensions(k[i].in[0]) = {FRAME_LENGTH}; dimensions(k[i].out[0]) = {FRAME_LENGTH}; } } };` Note: While not required, you can provide a logical name when creating the `input_plio`, `output_plio`, `input_gmio` and `output_gmio` objects. If you do not provide this, the AI Engine compiler automatically sets a unique logical name.

  ```
  class mygraph : public graph
  {
  public:
      // Declare N kernels.
      kernel k[N];
      // Declare N input_gmio(s) and output_gmio(s)
      input_gmio gmioIn[N];
      output_gmio gmioOut[N];

      mygraph()
      {
          // Create multiple input_gmio(s)/output_gmio(s) and connections
          for (int i=0; i<N; i++)
          {
              gmioIn[i]  = input_gmio::create("gmioIn" + std::to_string(i), 64, 1);
              gmioOut[i] = output_gmio::create("gmioOut" + std::to_string(i), 64, 1);
              connect(gmio[i].out[0], k[i].in[0]);
              connect(k[i].out[0], gmioOut[i].in[0]);

              dimensions(k[i].in[0]) = {FRAME_LENGTH};
              dimensions(k[i].out[0]) = {FRAME_LENGTH};
          }
      }
  };
  ```
- The programming model provides ease of use for connections within the design. With the `input_plio`, `output_plio`, `input_gmio` and `output_gmio` objects in the local scope, `connect` calls are not required to provide a unique connection name.
- The programming model supports the direction property of `input_plio`, `output_plio`, `input_gmio` and `output_gmio` objects. Object construction determines the direction property.

An example of the programming model that addresses these previously listed features is as follows.

```
/* A graph with both PLIO and GMIO*/
class mygraph : graph
{
public:

    /* Classes support input_gmio/output_gmio and input_plio/output_plio */
    input_gmio gm_in;
    output_gmio gm_out;

    input_plio pl_in;
    output_plio pl_out;

    kernel k1, k2;
    mygraph()
    {
        k1 = kernel::create(...);
        k2 = kernel::create(...);

        /* create() API for PLIO and GMIO objects support same arguments specified as in global scope */
        gm_in = input_gmio::create("GMIO_In0",/*const std::string& logical_name*/
                                    64,       /*size_t burst_length*/
                                    1         /*size_t bandwidth*/);

        gm_out = output_gmio::create("GMIO_Out0", 64, 1);

        pl_in = input_plio::create("PLIO_In0",       /* std::string logical_name */
                                    plio_32_bits,    /* enum plio_type plio_width */
                                    "data/input.txt",/* std::string data_file */
                                    250.0            /* double frequency */ );

        pl_out = output_plio::create("PLIO_Out0", plio_32_bits, "data/output.txt", 250.0);

        /* Each input_gmio/output_gmio and input_plio/output_plio supports 1 port per direction */
        connect(gm_in.out[0], k1.in[0]);
        connect(k1.out[0], gm_out.in[0]);

        connect(pl_in.out[0], k2.in[0]);
        connect(k2.out[0], pl_out.in[0]);

        location<GMIO>(gm_in) = shim(col);
        location<GMIO>(gm_out) = shim(col, ch_num); /* not supported */

        location<PLIO>(pl_in) = shim(col);
        location<PLIO>(pl_out) = shim(col, ch_num);
    }

};
```

The previous code snippet demonstrates:

- `input_plio`, `output_plio`, `input_gmio`, and `output_gmio` are public objects within the graph.
- `input_plio`, and `output_plio` classes are inherited from the `PLIO` class.
- `input_gmio`, and `output_gmio` classes are inherited from the `GMIO` class.
- `input_plio`, `output_plio`, `input_gmio`, and `output_gmio` object constructions contain the direction property.
- `connect` call does not require unique name.
- Location constraints can be applied to the `input_plio`, `output_plio`, `input_gmio`, and `output_gmio` objects directly with the exception of GMIO channel constraints which are not supported in this release.Note: GMIO shim tile location constraint with DMA channel is not supported. Note: Host application calling profiling APIs need reference object’s public members, `mygraph.gm_in`, `mygraph.gm_out`, `mygraph.pl_in`, `mygraph.pl_out`. For additional details about profiling APIs, see [Performance Analysis of AI Engine Graph Application during Simulation](https://docs.amd.com/access/sources/dita/topic?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment&resourceid=bvm1601048694615.html) in AI Engine Tools and Flows User Guide ([UG1076](https://docs.amd.com/access/sources/dita/map?Doc_Version=2025.2%20English&url=ug1076-ai-engine-environment)).An example of a profile call from programming model is as follows. `event::handle handle0 = event::start_profiling(mygraph.gm_in, event::io_stream_start_to_bytes_transferred_cycles, sizeIn*sizeof(cint16));`

**Note:** Important:

`input_plio`

`output_plio`

`input_gmio`

`output_gmio`

```
ERROR: [aiecompiler 77-4551] The logical name DataIn1 of node i6 conflicts with the logical/qualified name of node i0.
```
