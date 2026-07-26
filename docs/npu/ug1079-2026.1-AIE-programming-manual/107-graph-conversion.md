---
title: "Graph Conversion"
source_url: "https://docs.amd.com/r/en-US/ug1079-ai-engine-kernel-coding/Graph-Conversion"
toc_id: wm7G6VRJSbVEBCeBtD5l~g
content_id: cMfV_1VZ~kBN3~1JisoIfQ
---

## Graph Conversion

Connect statements with buffer port to buffer port do not require template parameters specifying the size. Use `dimensions()` API in graph to specify number of samples (not bytes) for all input/output buffer ports. You can specify buffer port size (number of samples, not bytes) using kernel function prototypes when the `dimensions()` APIs in graph are not used.

#### Window-Based Connect Example

```
connect<window<EQ24_INPUT_SAMPLES*4, EQ24_INPUT_MARGIN*4> > (in.out[0], eq24.in[0]);
connect<window<HB27_2D_INPUT_SAMPLES*4, HB_2D_INPUT_MARGIN*4> > (eq24.out[0], hb27.in[0]) ;
connect<window<HB27_2D_INPUT_SAMPLES*4, HB_2D_INPUT_MARGIN*4> > (eq24.out[1], hb27.in[1]) ;
connect<window<CE_INPUT_SAMPLES*4, CE_INPUT_MARGIN*4> > (hb27.out[0], ce.in[0]) ;
```

#### Buffer Port-Based Connect and Dimensions Example

```
connect(in.out[0], eq24.in[0]) ;
connect(eq24.out[0], hb27.in[0]) ;
connect(eq24.out[1], hb27.in[1]) ;
connect(hb27.out[0], ce.in[0]);
adf::dimensions(eq24.in[0]) = { EQ24_INPUT_SAMPLES };
adf::dimensions(hb27.in[0]) = { HB27_2D_INPUT_SAMPLES };
adf::dimensions(hb27.in[1]) = { HB27_2D_INPUT_SAMPLES };
adf::dimensions(eq24.out[0]) = { EQ24_OUTPUT_SAMPLES };
adf::dimensions(eq24.out[1]) = { EQ24_OUTPUT_SAMPLES };
adf::dimensions(hb27.out[0]) = { HB27_2D_OUTPUT_SAMPLES };
adf::dimensions(ce.in[0]) = { CE_INPUT_SAMPLES };
```

**Note:** Margin size can be specified only in kernel function prototypes. Kernel Conversion section contains kernel function prototype with margin examples. The example above illustrates only graph code updates.
