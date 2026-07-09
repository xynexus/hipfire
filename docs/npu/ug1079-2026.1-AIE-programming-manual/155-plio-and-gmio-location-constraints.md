---
title: "PLIO and GMIO Location Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/PLIO-and-GMIO-Location-Constraints"
toc_id: Lebpf592YbXnzNkmsT4xHA
content_id: WcpqgDhGYTP6GqQeoO1EwQ
---

### PLIO and GMIO Location Constraints

In AI Engine graphs, when defining external stream connections that cross the AI Engine to the programmable logic (PL) boundary via PLIOs or to the NoC (PL) via GMIOs, it is beneficial to specify both the interface tile column and the exact channel within the interface tile. This specification can impact throughput and latency, optimizing the overall performance.

When setting these constraints, it is essential to select columns in the AI Engine that have direct connections to the PLIO or GMIO shim interface. This ensures optimal functionality and data flow through the AI Engine interfaces.

By default, the AI Engine compiler automatically assigns a channel in the PLIO or GMIO shim interface for data routing. However, you can explicitly specify a channel for further control using the following constraints:

```
location_constraint shim(int column)
location_constraint shim(int column,int channel)
```

The available channel ranges for GMIO and PLIO differ based on the data direction:

|  | Input Channel Range | Output Channel Range |
| --- | --- | --- |
| GMIO | 0, 1 | 0, 1 |
| PLIO | 0 through 7 | 0 through 5 |

Following is an example of how to constrain the location of kernels and interfaces within an AI Engine graph using C++:

```
template <int COL,int ROW>
class SimpleGMIOGraph : public adf::graph
{

private:
public:
    adf::kernel k1,k2;

    adf::input_gmio gmin;
    adf::output_gmio gmout;
    adf::input_plio plin;
    adf::output_plio plout;

    SimpleGMIOGraph()
    {

        k1 = adf::kernel::create(passthrough_buffer<1024>);
        adf::source(k1) = "passthrough.cpp";
        adf::runtime<ratio>(k1) = 0.9;
        adf::location<adf::kernel>(k1) = adf::tile(COL, ROW);

        k2 = adf::kernel::create(passthrough_buffer<1024>);
        adf::source(k2) = "passthrough.cpp";
        adf::runtime<ratio>(k2) = 0.9;
        adf::location<adf::kernel>(k2) = adf::tile(COL+1, ROW);

        gmin = adf::input_gmio::create("gmin", 64, 4000);
        gmout = adf::output_gmio::create("gmout", 64, 4000);
        adf::location<adf::GMIO>(gmin) = adf::shim(COL);
        adf::location<adf::GMIO>(gmout) = adf::shim(COL);

        plin = adf::input_plio::create("plin", adf::plio_64_bits, "data/Input_64.txt", 625);
        plout = adf::output_plio::create("plout", adf::plio_64_bits,"data/Output.txt, 625);
        adf::location<adf::PLIO>(plin) = adf::shim(COL+1);
        adf::location<adf::PLIO>(plout) = adf::shim(COL+1);

        // Connections
        adf::connect(gmin.out[0], k1.in[0]);
        adf::connect(k1.out[0],gmout.in[0]);
        adf::connect(plin.out[0], k2.in[0]);
        adf::connect(k2.out[0], plout.in[0]);
       };
};


```
