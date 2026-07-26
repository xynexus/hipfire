# R109: in-place next-layer preparation

R109 is R47 with its R34 activation output shifted by 884,736 bytes. The shared
argument contains the compensated completed state first and the persistent R34
activation/weight-scale records second. R109 reads the completed prefix and
writes only the dynamic activation fields in the suffix, allowing R108 to
consume both through one five-argument DPU ABI.

The graph uses one in-place dma-buf argument. An intermediate three-argument
form imported the same dma-buf separately as source and destination and was
rejected by amdxdna with `EALREADY`. The in-place graph reads and writes
disjoint regions and passes the complete layer oracle with R108.
