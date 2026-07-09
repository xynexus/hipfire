---
title: "FIFO Location Constraints"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/FIFO-Location-Constraints"
toc_id: cos_RSBvNPkRitRgi61dbA
content_id: m1Y2oaaC3tV_dJVj_Ip~xA
---

### FIFO Location Constraints

The AI Engine compiler tries to allocate FIFOs in the most efficient manner possible. However, you might want to explicitly control their placement in memory, as shown in the following example. This constraint is useful to preserve the placement of FIFO resources between runs of the AI Engine compiler.

Note the following considerations for FIFO constraints:

- If you use FIFO constraints, constrain the entire depth of the FIFO. It is not possible to constrain a portion of the FIFO and leave the rest for the compiler to add.
- If you add FIFO constraints to branching nets, add the constraint to each point-to-point net. To share stream switch FIFOs or DMA FIFOs before the branch, duplicate the FIFO type and location on each point-to-point net.
- You can use the constraint without a location to specify the desired type of FIFO without specifying a location or depth.

The following example shows using a FIFO constraint in a graph file.

```
two_node_graph() {
  loop0 = adf::kernel::create(loopback_stream);
  loop1 = adf::kernel::create(loopback_stream);
  loop2 = adf::kernel::create(loopback_stream);

  adf::source(loop0) = "loopback_stream.cc";
  adf::source(loop1) = "loopback_stream.cc";
  adf::source(loop2) = "loopback_stream.cc";

  adf::connect net0 (in0, loop0.in[0]);
  adf::connect net1 (loop0.out[0], loop1.in[0]);
  adf::connect net2 (loop1.out[0], loop2.in[0]);
  adf::connect net3 (loop2.out[0], out0);

  adf::runtime(loop0) = 0.9;
  adf::runtime(loop1) = 0.9;
  adf::runtime(loop2) = 0.9;

  adf::fifo_depth(net1) = 32;
  adf::fifo_depth(net2) = 48;

  adf::location< adf::fifo >(net1) = {adf::ss_fifo(adf::shim_tile, 16 , 0, 1), adf::ss_fifo(adf::shim_tile,17,0,0)};
  adf::location< adf::fifo >(net2) = adf::dma_fifo(adf::aie_tile, 8, 0, 0x0000, 48);
};
```

The second example shows that you can add a FIFO constraint to a constraints file.

```
{
  "PortConstraints": {
    "fifo_locations_records": {
      "dma_fifos": {
        "r1": {
          "tile_type": "core",
          "row": 0,
          "column": 0,
          "size": 16,
          "offset": 8,
          "bankId": 2
        },
        "r2": {
          "tile_type": "core",
          "row": 0,
          "column": 1,
          "size": 16,
          "offset": 9
        },
        "r4": {
          "tile_type": "mem",
          "row": 2,
          "column": 4,
          "size": 16,
          "offset": 6,
          "bankId": 2
        }
      },
      "stream_fifos": {
        "r3": {
          "tile_type": "shim",
          "row": 1,
          "column": 3,
          "channel": 1
        }
      }
    },
    "mygraph.k2.in[0]": {
      "fifo_locations": ["r1", "r2", "r3"]
    },
    "mygraph.k4.in[0]": {
      "fifo_locations": ["r1", "r2", "r4"]
    }
  }
}
```
