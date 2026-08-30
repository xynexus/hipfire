Yes. That is the correct reframing.

A codebook over (\Delta) is not especially valuable as a **compression mechanism by itself**, because the positional address dominates the sparse record. Its useful interpretation is:

[
\text{less precision per correction}
\quad\longrightarrow\quad
\text{more corrected positions at the same block size}
]

For the current compact block:

[
130\text{ B base} + 3(1\text{ B index}+1\text{ B value})
=136\text{ B}
]

With four-bit codebook values:

[
130\text{ B base}
+4(1\text{ B index})
+4(0.5\text{ B code})
=136\text{ B}
]

So the meaningful comparison is:

* **3 exact int8 replacements**
* versus **4 codebook-approximated replacements**
* at the same (136)-byte block, or (4.25) bits/weight.

That is a plausible trade. Calling the codebook a way to reduce the model from 4.25 to roughly 4.20 bits is mostly bookkeeping glitter.

One numerical correction: reducing each value from eight bits to four saves

[
\frac{4}{256}=0.015625\text{ b/w}
]

**per outlier**. At three outliers, the theoretical saving is:

[
3\times 0.015625=0.046875\text{ b/w}
]

although block-local byte packing reduces the practical saving. Three four-bit codes occupy two bytes unless you allow cross-block packing, so the physical block would shrink from 136 to 135 bytes:

[
\frac{135\times 8}{256}=4.21875\text{ b/w}
]

Still small, and likely not worth a separate format.

There are two caveats to the 3-to-4 exchange.

First, the existing replacement is exact in the integer domain:

[
q_i^{\text{final}} = q_i^{(8)}
]

With a codebook, it becomes:

[
q_i^{\text{final}} = q_i^{(4)} + C[c_i]
]

Unless every useful (\Delta_i=q_i^{(8)}-q_i^{(4)}) is represented exactly, each promoted position retains some residual error. Therefore four codebook corrections beat three exact corrections only when:

[
\sum_{i\in S_4}
\left(e_{4,i}^2-e_{C,i}^2\right)

>

\sum_{i\in S_3}
\left(e_{4,i}^2-e_{8,i}^2\right)
]

where (S_4) is the best four positions under the codebook and (S_3) is the best three exact promotions. The selector must score the actual codebook reconstruction, not choose four positions using the existing W8-gain metric and quantize their deltas afterward.

Second, the index really is close to irreducible at this sparsity if positions are approximately uniform after the FWHT. A sorted set of four positions among 256 contains:

[
\log_2 {256\choose4}\approx27.38\text{ bits}
]

Four raw u8 indices consume 32 bits. So even an ideal combinatorial encoding saves only about 4.6 bits per block:

[
\frac{4.6}{256}\approx0.018\text{ b/w}
]

That means the u8 index format looks crude, but it is actually reasonably close to the entropy floor. Delta coding only becomes substantially better if promoted positions have spatial structure, which the FWHT may actively suppress.

The strongest version is therefore not “VQ the existing three residuals.” It is:

1. Use a fixed or layer-level 16-entry integer-(\Delta) codebook.
2. Jointly optimize scale, selected positions, and codebook entries.
3. Select four positions according to their **post-codebook reduction in output error**.
4. Compare the resulting four approximate corrections against three exact W8 replacements.
5. Reject the idea unless it wins KLD or Hessian-weighted reconstruction at identical 136-byte storage.

And this points toward an even better bitplane interpretation. Instead of an arbitrary 16-entry residual codebook, the four-bit code could represent a structured refinement such as one or more additional signed bitplanes. Then the correction is cheap to decode, training can optimize against the exact hardware representation, and the format naturally nests into your 2-to-8-bit progressive scheme.

So yes: **codebook residuals only become interesting when reframed as increased correction density, not reduced headline bit rate.** Your 3-exact versus 4-approximate comparison is the right experiment.
