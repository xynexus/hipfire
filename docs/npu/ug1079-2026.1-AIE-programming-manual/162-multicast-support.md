---
title: "Multicast Support"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Multicast-Support"
toc_id: s4wwgWl9zZNeq9Ld_VBlZg
content_id: ibTqixzQtFJLuU7rQzRz5w
---

## Multicast Support

The graph supports various multicast scenarios. These include:

- from a buffer to multiple buffers
- from stream to multiple streams
- from `input_plio` to multiple buffers, etc.

This section lists the supported types of multicast from a single source to multiple destinations. For additional details on`input_plio`/`output_plio`, and `input_gmio`/`output_gmio`, see Graph Programming Model.

| Scenario # | Source | Destination 1 | Destination 2 | Support |
| --- | --- | --- | --- | --- |
| 1 | AI Engine Buffer | AI Engine Buffer | AI Engine Buffer | Supported |
| 2 | AI Engine Buffer | AI Engine Buffer | AI Engine Stream | Supported |
| 3 | AI Engine Buffer | AI Engine Buffer | output_plio/output_gmio | Supported |
| 4 | AI Engine Buffer | AI Engine Stream | AI Engine Stream | Supported |
| 5 | AI Engine Buffer | AI Engine Stream | output_plio/output_gmio | Supported |
| 6 | AI Engine Buffer | output_plio/output_gmio | output_plio/output_gmio | Supported |
| 7 | AI Engine Stream | AI Engine Buffer | AI Engine Buffer | Supported |
| 8 | AI Engine Stream | AI Engine Buffer | AI Engine Stream | Supported |
| 9 | AI Engine Stream | AI Engine Buffer | output_plio/output_gmio | Supported |
| 10 | AI Engine Stream | AI Engine Stream | AI Engine Stream | Supported |
| 11 | AI Engine Stream | AI Engine Stream | output_plio/output_gmio | Supported |
| 12 | AI Engine Stream | output_plio/output_gmio | output_plio/output_gmio | Supported |
| 13 | input_plio/input_gmio | AI Engine Buffer | AI Engine Buffer | Supported |
| 14 | input_plio/input_gmio | AI Engine Buffer | AI Engine Stream | Not Supported |
| 15 | input_plio/input_gmio | AI Engine Buffer | output_plio/output_gmio | Not Supported |
| 16 | input_plio/input_gmio | AI Engine Stream | AI Engine Stream | Supported |
| 17 | input_plio/input_gmio | AI Engine Stream | output_plio/output_gmio | Not Supported |
| 18 | input_plio/input_gmio | output_plio/output_gmio | output_plio/output_gmio | Not Supported |

**Note:** - All source and destination buffers in the multicast connections are required to have the same size to stay in a single rate environment.
- If all sources and destinations do not have the same size, the compiler automatically switches to multirate processing. The compiler determines the number of times the kernels need to be executed per iteration.
- The tool realizes buffer multicast by adding DMA to source and destination buffers.
- Each connection between the source and destination is blocking. Any destination blocks the multicast if it is not ready to accept data.
- This section does not cover RTP and packet switching.
- If the multicast type is supported, you can use any number of destinations, as long as they fit within the hardware.

When multiple streams connect to the same source, data is sent to all destination ports simultaneously when all destinations are ready to receive. This can cause stream stall or design hang if the FIFO depth of the stream connections are not deep enough.

The following multicast example shows scenario number 10 from above table. Source and both destinations are stream.

This `graph.h` code snippet defines two sub-graphs `_graph0` and `_graph1` within the top graph (`top_graph`). This ensures that sending data to all destination ports at the same time.

```
class _graph0: public adf::graph {
private:
    adf::kernel kr;
public:
    adf::port<input> instream;
    adf::port<output> outstream;
    _graph0() {
        kr = adf::kernel::create(compute0);
        adf::runtime<ratio>(kr) = 0.9;
        adf::source(kr) = "compute0.cc";
        adf::connect<adf::stream> n0(instream, kr.in[0]);
        adf::connect<adf::stream> n1(kr.out[0], outstream);
    }
};

class _graph1: public adf::graph {
private:
    adf::kernel kr;
public:
    adf::port<input> instream;
    adf::port<output> outstream;
    _graph1() {
        kr = adf::kernel::create(compute1);
        adf::runtime<ratio>(kr) = 0.9;
        adf::source(kr) = "compute1.cc";
        adf::connect<adf::stream> n0(instream, kr.in[0]);
        adf::connect<adf::stream> n1(kr.out[0], outstream);
    }
};

class top_graph: public adf::graph {
private:

public:
    _graph0 g0;
    _graph1 g1;
    adf::input_plio  instream;
    adf::output_plio outstream0;
    adf::output_plio outstream1;
    top_graph()
    {
        instream   = adf::input_plio::create("aie_brodcast_0_S_AXIS",
                         adf::plio_32_bits,
                         "data/input.txt");
        outstream0 = adf::output_plio::create("aie_graph0_outstream",
                         adf::plio_32_bits,
                         "data/output0.txt");
        outstream1 = adf::output_plio::create("aie_graph1_outstream",
                         adf::plio_32_bits,
                         "data/output1.txt");

        adf::connect<adf::stream> n0(instream.out[0], g0.instream);
        adf::connect<adf::stream> n1(instream.out[0], g1.instream);
        adf::connect<adf::stream> n2(g0.outstream, outstream0.in[0]);
        adf::connect<adf::stream> n3(g1.outstream, outstream1.in[0]);
    }
};
```

In this `graph.cpp` code snippet, the graph calls are invoked from the top graph so all sub-graphs are receiving the same data at the same time.

```
top_graph top_g;

#if defined  (__AIESIM__) || defined(__X86SIM__)
int main () {
        top_g.init();
        top_g.run(3);
        top_g.wait();
        top_g.end();
        return 0;
}
#endif
```
