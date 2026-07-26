---
title: "Tiling Parameters Examples"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Tiling-Parameters-Examples"
toc_id: X5IdvIOuvEu6spzgdBHmIg
content_id: r0k~t0cv79PJFj3t1NRlMA
---

### Tiling Parameters Examples

This section uses a graph that contains two directly-connected kernels. The graph shows how data is written to or read from the AI Engine memory by tiling parameters. The graph code is as follows:

```
class mygraph: public graph {
  public:
    kernel first,second;
    input_plio in;
    output_plio out;

    mygraph() {
      first=kernel::create(producer);
      source(first) = "producer.cc";
      second=kernel::create(consumer);
      source(second) = "consumer.cc";

      in=input_plio::create("Datain0", plio_64_bits, "data/input.txt");
      out=output_plio::create("Dataout0", plio_64_bits, "data/output.txt");

    runtime<ratio>(first)=0.9;
    runtime<ratio>(second)=0.9;
    connect(in.out[0], first.in[0]);
    connect(first.out[0], second.in[0]);
    connect(second.out[0], out.in[0]);
    dimensions(first.in[0])={256};
    dimensions(first.out[0])={256};
    dimensions(second.in[0])={16,16};
    dimensions(second.out[0])={16,16};

    read_access(first.out[0]) = tiling({.buffer_dimension={256}, .tiling_dimension={256}, .offset={0} });
    write_access(second.in[0]) = tiling({.buffer_dimension={256}, .tiling_dimension={256}, .offset={0} });
  }
};
```

Following are examples of different `tiling` configurations. The examples show the following:

- how source data is transferred to destination
- how data is stored in row basis in memory
- how the configurations can be applied on `write_access` or `read_access` by default.

In the graphics Source represents data organization within the first output buffer, and Destination represents data organization within the second input buffer.

##### 1D Linear Transferring From Source to Destination

1D Linear transferring from source to destination using three equivalent lines of code (any one applies):

```
tiling({.buffer_dimension={256}, .tiling_dimension={256}, .offset={0} });
tiling({.buffer_dimension={256}, .tiling_dimension={256}, .offset={0},
      .tile_traversal = {{.dimension=0, .stride=256, .wrap=1}} });
tiling({.buffer_dimension={256}, .tiling_dimension={1}, .offset={0},
      .tile_traversal = {{.dimension=0, .stride=1, .wrap=256}} });
```

![iuh1669144085679.png](../assets/081-01-iuh1669144085679-png-ca0594cc7e64.png)

*Figure 1. Example of 1D Linear Transferring From Source to Destination*

##### 2D Linear Transferring from Source to Destination

2D Linear transferring from source to destination using two equivalent lines of code (any one applies):

```
tiling({.buffer_dimension={128,2}, .tiling_dimension={128,2},
      .offset={0,0} });
tiling({.buffer_dimension={128,2}, .tiling_dimension={128,1}, .offset={0,0},
      .tile_traversal = {{.dimension=0, .stride=128, .wrap=1},
            {.dimension=1, .stride=1, .wrap=2}} });
```

*Figure 2. 2D Linear*

![zwr1669144444964.png](../assets/081-02-zwr1669144444964-png-ef496af52ede.png)

##### 2D Matrix Transferring to 4x2 Sub Matrices

2D Matrix transferring to transfer 4x2 sub matrices. The numbers show how the source stores and transfers data to the destination with one such tiling parameter:

```
read_access(first.out[0]) = tiling({.buffer_dimension={16,4},
         .tiling_dimension={4,2}, .offset={0,0},
         .tile_traversal = {{.dimension=0, .stride=4, .wrap=4},
               {.dimension=1, .stride=2, .wrap=2}} });
```

![xyk1669144666021.png](../assets/081-03-xyk1669144666021-png-3ebdad8dfab7.png)

*Figure 3. Example of 2D Matrix Transferring to 4x2 Sub Matrices*
