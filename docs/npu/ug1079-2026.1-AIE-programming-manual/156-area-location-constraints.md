---
title: "Area Location Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Area-Location-Constraints"
toc_id: yqfZSAN3e~h2G6LaLDgneQ
content_id: PADOEl_AxNEZB8m9AgBMQg
---

### Area Location Constraints

Area location constraints direct the compiler to contain nodes to a custom location in the array. Properties to specify on an area group are described in the following table.

| Property | Description |
| --- | --- |
| group | Specify the collection of group. Each group can be: tile-type: Specify the tile-type for the group. Supported tile-types are aie_tile, shim_tile, or memory_tile. column_min: Column index for lower left corner of the group. row_min: Row index for lower left corner of the group. column_max: Column index for upper right corner of the group. row_max: Row index for upper right corner of the group. |
| contain_routing | A boolean value that when specified true ensures all routing, including nets between nodes contained in the nodeGroup, is contained within the area group. Default: false. |
| exclusive_routing | A boolean value that when specified true ensures all routing, excluding nets between nodes from the nodeGroup, is excluded from the area group. Default: false. |
| exclusive_placement | A boolean value that when specified true prevents all nodes not included in the nodeGroup from being placed within the area group bounding box. Default: false. |

The following examples show how an area location constraint can be applied in a graph file.

```
class testGraph1: public adf::graph {

    private:
        adf::kernel first;
        adf::kernel second;
    public:
        testGraph1()  {
            first = adf::kernel::create(simple1);
            second = adf::kernel::create(simple2);
            adf::connect(first.out[0], second.in[0]);
            adf::source(first) = "src/kernels/kernels.cc";
            adf::source(second) = "src/kernels/kernels.cc";
            adf::runtime<adf::ratio>(first) = 0.1;
            adf::runtime<adf::ratio>(second) = 0.1;

            // Create area group with some valid ranges.
            adf::location<adf::graph>(*this) = adf::area_group({{adf::aie_tile, 0, 0, 1, 7}, {adf::shim_tile, 0, 0, 1, 0}});
        }

};
```

```
class testGraph2: public adf::graph {

    private:
        adf::kernel first;
        adf::kernel second;
    public:
        testGraph2()  {
            first = adf::kernel::create(simple1);
            second = adf::kernel::create(simple2);
            adf::connect(first.out[0], second.in[0]);
            adf::source(first) = "src/kernels/kernels.cc";
            adf::source(second) = "src/kernels/kernels.cc";
            adf::runtime<adf::ratio>(first) = 0.1;
            adf::runtime<adf::ratio>(second) = 0.1;

            // Explicitly specify contain_routing, exclusive_routing, exclusive_placement.
            adf::location<adf::graph>(*this) = adf::area_group({{adf::aie_tile, 0, 0, 1, 7}, {adf::shim_tile, 0, 0, 1, 0}}, true, false, true);
        }

};
```

The following table clarifies whether the nodes and nets can access the resources of the area group given various combinations of the flags. The use cases presented in the table as columns are illustrated in the figure that follows. Two important things to consider when applying the rules from the following table.

- Broadcast nets that are driven from one node to several destination nodes, are considered as individual point-to-point nets
- Any FIFOs on a net adhere to the same conditions as the net. For example, if a net is fully contained in an area group (both driver and receiver are contained in the area group) and the contain_routing flag is used, then the following table indicates that the net routing must be fully contained in the area group. Similarly, any FIFOs on that net, must also be placed within the boundary of the area group.

| contain_routing | exclusive_routing | exclusive_placement | Placement of Nodes Contained in the Area Group (1) | Placement of Nodes External to the Area Group (2) | Routes between Nodes Fully Contained in the Area group (3) | Routes between Nodes Spanning the Area Group (4) | Routes between Nodes Entirely External to the Area Group (5) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | False | False | Must | May | May | May | May |
| False | False | True | Must | Must Not | May | May | May |
| False | True | False | Must | May | May | May | Must Not |
| False | True | True | Must | Must Not | May | May | Must Not |
| True | False | False | Must | May | Must | May | May |
| True | False | True | Must | Must Not | Must | May | May |
| True | True | False | Must | May | Must | May | Must Not |
| True | True | True | Must | Must Not | Must | May | Must Not |

The following figure shows an illustration of the use cases.

![aui1631213811621.png](../assets/156-01-aui1631213811621-png-aa271e3df9fd.png)

*Figure 1. Use Cases*
