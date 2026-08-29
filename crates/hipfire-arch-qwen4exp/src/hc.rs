// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Gated residual (hyper-connections) — CPU reference.
//!
//! This family carries a 4-wide residual stream, and every block reads and writes
//! it twice. There are no `input_layernorm` / `post_attention_layernorm` tensors in
//! the checkpoint at all: `hc_norm` inside these blocks replaces them, and the
//! model-level mixer's `hc_norm` is the only pre-head norm.
//!
//! This is the ORACLE, written before the kernel deliberately. The adversarial
//! review of the earlier MoE work turned on exactly this: an end-to-end comparison
//! where both arms run the same code cannot fail, so a GPU kernel needs an
//! independent implementation to be differenced against, not another run of itself.
//!
//! Three details are silent-wrong if guessed, all taken from the reference
//! (`Qwen4ExpTextGatedResidual`, and `Qwen3_5RMSNorm` which it inherits):
//!
//! * **The norm applies `(1.0 + weight)`, not `weight`.** The parameter is
//!   zero-initialised, so a loader that multiplies by `weight` alone collapses the
//!   residual toward zero — and one that *also* bakes the `+1` in at convert time
//!   double-counts it.
//! * **The norm is GROUPED**: each of the 4 streams is normalised independently
//!   over `hidden`, not jointly over `4 * hidden`.
//! * **The stream collapse is a MEAN, not a sum**, and the mix is per-channel with
//!   a sigmoid — there is no cross-stream 4x4 matrix anywhere in this family.

/// Grouped RMSNorm: normalise each `group` -wide slice independently, then scale by
/// `1.0 + weight`.
///
/// `weight` spans the whole vector, not one group — the checkpoint's `hc_norm` is
/// `[hc_count * hidden]`.
pub fn grouped_rmsnorm(x: &[f32], weight: &[f32], group: usize, eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        weight.len(),
        "norm weight must span the whole vector"
    );
    assert!(
        group > 0 && x.len() % group == 0,
        "group must divide the vector"
    );
    let mut out = vec![0.0f32; x.len()];
    for (g, chunk) in x.chunks(group).enumerate() {
        let mean_sq = chunk.iter().map(|v| v * v).sum::<f32>() / group as f32;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        let base = g * group;
        for i in 0..group {
            // `(1.0 + w)`, matching Qwen3_5RMSNorm — the parameter is zero-init.
            out[base + i] = chunk[i] * scale * (1.0 + weight[base + i]);
        }
    }
    out
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Row-major `[out, in]` matrix times a vector, as `nn.Linear` stores it.
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(w.len(), out_dim * in_dim);
    assert_eq!(x.len(), in_dim);
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            row.iter().zip(x).map(|(a, b)| a * b).sum()
        })
        .collect()
}

/// One gated-residual block's weights, in checkpoint layout.
pub struct GatedResidual<'a> {
    /// `[hc_count * hidden]`
    pub hc_norm: &'a [f32],
    /// `[lowrank, hc_count * hidden]`
    pub mix_down: &'a [f32],
    /// `[hc_count * hidden, lowrank]`
    pub mix_up: &'a [f32],
    /// `[hc_count, hc_count * hidden]`, or `None` for the model-level mixer, which
    /// collapses the streams without injecting a block output.
    pub block_inject: Option<&'a [f32]>,
    pub hc_count: usize,
    pub hidden: usize,
    pub lowrank: usize,
    pub eps: f32,
}

/// What a block needs from the residual before running its transform.
pub struct Read {
    /// `[hidden]` — the collapsed transform input.
    pub mixed_input: Vec<f32>,
    /// `[hc_count * hidden]` — the normalised streams, reused by the write side.
    pub normed: Vec<f32>,
    /// `[hc_count]` per-branch write gates in `(0, 2)`, absent for the mixer.
    pub inject: Option<Vec<f32>>,
}

impl GatedResidual<'_> {
    fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// The read side: normalise, build the per-channel mix, collapse to one vector.
    pub fn read(&self, streams: &[f32]) -> Read {
        assert_eq!(streams.len(), self.width());
        let normed = grouped_rmsnorm(streams, self.hc_norm, self.hidden, self.eps);

        // Low-rank gate: down -> /hc_count -> silu -> up -> sigmoid. The division
        // is INSIDE the silu, before the expand.
        let mut t = matvec(&self.mix_down[..], &normed, self.lowrank, self.width());
        for v in t.iter_mut() {
            *v = silu(*v / self.hc_count as f32);
        }
        let mut mix = matvec(&self.mix_up[..], &t, self.width(), self.lowrank);
        for v in mix.iter_mut() {
            *v = sigmoid(*v);
        }

        // MEAN over streams of the per-channel-gated normed streams.
        let mut mixed_input = vec![0.0f32; self.hidden];
        for s in 0..self.hc_count {
            let base = s * self.hidden;
            for d in 0..self.hidden {
                mixed_input[d] += mix[base + d] * normed[base + d];
            }
        }
        for v in mixed_input.iter_mut() {
            *v /= self.hc_count as f32;
        }

        let inject = self.block_inject.map(|bi| {
            matvec(bi, &normed, self.hc_count, self.width())
                .into_iter()
                .map(|v| 2.0 * sigmoid(v / self.hc_count as f32))
                .collect()
        });

        Read {
            mixed_input,
            normed,
            inject,
        }
    }

    /// The write side: add the block's output back into each stream, scaled by that
    /// stream's gate. Operates on the RAW streams, not the normalised ones — the
    /// normalisation is for computing the gates, not for the residual itself.
    pub fn write(&self, streams: &mut [f32], block_out: &[f32], inject: &[f32]) {
        assert_eq!(streams.len(), self.width());
        assert_eq!(block_out.len(), self.hidden);
        assert_eq!(inject.len(), self.hc_count);
        for s in 0..self.hc_count {
            let base = s * self.hidden;
            for d in 0..self.hidden {
                streams[base + d] += inject[s] * block_out[d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: usize = 8;
    const C: usize = 4;
    const R: usize = 3;
    const EPS: f32 = 1e-6;

    fn seeded(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    fn gr<'a>(
        norm: &'a [f32],
        down: &'a [f32],
        up: &'a [f32],
        inject: Option<&'a [f32]>,
    ) -> GatedResidual<'a> {
        GatedResidual {
            hc_norm: norm,
            mix_down: down,
            mix_up: up,
            block_inject: inject,
            hc_count: C,
            hidden: H,
            lowrank: R,
            eps: EPS,
        }
    }

    /// A zero norm weight must be the IDENTITY scale, because the reference applies
    /// `1.0 + weight` and the parameter is zero-initialised. Getting this wrong
    /// collapses the residual toward zero.
    #[test]
    fn zero_norm_weight_is_unit_scale() {
        let x = vec![3.0f32; 4];
        let w = vec![0.0f32; 4];
        let out = grouped_rmsnorm(&x, &w, 4, EPS);
        // rms of [3,3,3,3] is 3, so each element normalises to ~1.0.
        for v in &out {
            assert!((v - 1.0).abs() < 1e-4, "got {v}, expected ~1.0");
        }
        // And a weight of -1.0 must zero it, confirming the offset is really +1.
        let out = grouped_rmsnorm(&x, &vec![-1.0f32; 4], 4, EPS);
        assert!(out.iter().all(|v| v.abs() < 1e-6));
    }

    /// Each stream normalises INDEPENDENTLY. Scaling one stream must not move any
    /// other — the check that catches a flat norm over the full width.
    #[test]
    fn grouping_isolates_streams() {
        let mut x = seeded(C * H, 7);
        let w = vec![0.0f32; C * H];
        let a = grouped_rmsnorm(&x, &w, H, EPS);
        for v in x[0..H].iter_mut() {
            *v *= 10.0;
        }
        let b = grouped_rmsnorm(&x, &w, H, EPS);
        assert_eq!(a[H..], b[H..], "streams 1..3 must be untouched");
        // Scaling a whole group is a no-op under RMS normalisation.
        for d in 0..H {
            assert!((a[d] - b[d]).abs() < 1e-4, "stream 0 is scale-invariant");
        }
    }

    /// The collapse is a MEAN, not a sum. With the gate forced to 1 everywhere,
    /// the result must be the average of the streams — a sum would be 4x larger.
    #[test]
    fn collapse_is_a_mean_not_a_sum() {
        // A zero `up` weight makes sigmoid(0) = 0.5 uniformly, which is a known
        // gate value, so the expected output is 0.5 * mean(normed).
        let norm = vec![0.0f32; C * H];
        let down = seeded(R * C * H, 11);
        let up = vec![0.0f32; C * H * R];
        let g = gr(&norm, &down, &up, None);
        let streams = seeded(C * H, 3);
        let r = g.read(&streams);
        for d in 0..H {
            let mean: f32 = (0..C).map(|s| r.normed[s * H + d]).sum::<f32>() / C as f32;
            assert!(
                (r.mixed_input[d] - 0.5 * mean).abs() < 1e-5,
                "d={d}: {} vs {}",
                r.mixed_input[d],
                0.5 * mean
            );
        }
    }

    /// The mix is PER-CHANNEL: changing `up` for one channel of one stream must
    /// move only that channel. A per-stream scalar gate could not do this.
    #[test]
    fn mix_is_per_channel_not_per_stream() {
        let norm = vec![0.0f32; C * H];
        let down = seeded(R * C * H, 5);
        let mut up = seeded(C * H * R, 9);
        let streams = seeded(C * H, 2);
        let before = gr(&norm, &down, &up, None).read(&streams).mixed_input;
        // Perturb the `up` row feeding stream 2, channel 3.
        let row = 2 * H + 3;
        for k in 0..R {
            up[row * R + k] += 5.0;
        }
        let after = gr(&norm, &down, &up, None).read(&streams).mixed_input;
        for d in 0..H {
            if d == 3 {
                assert!((after[d] - before[d]).abs() > 1e-6, "channel 3 must move");
            } else {
                assert!((after[d] - before[d]).abs() < 1e-6, "channel {d} must not");
            }
        }
    }

    /// Inject weights are `2 * sigmoid(.)`, so they live in `(0, 2)` — a plain
    /// sigmoid would cap at 1 and halve every residual write.
    #[test]
    fn inject_weights_span_zero_to_two() {
        let norm = vec![0.0f32; C * H];
        let down = seeded(R * C * H, 4);
        let up = seeded(C * H * R, 6);
        let zero_bi = vec![0.0f32; C * C * H];
        let g = gr(&norm, &down, &up, Some(&zero_bi));
        let inj = g.read(&seeded(C * H, 8)).inject.unwrap();
        assert_eq!(inj.len(), C);
        // A zero weight gives 2 * sigmoid(0) = 1.0 exactly.
        for v in &inj {
            assert!((v - 1.0).abs() < 1e-6, "got {v}");
        }
        // A large positive weight approaches 2, not 1.
        let big: Vec<f32> = vec![50.0; C * C * H];
        let g = gr(&norm, &down, &up, Some(&big));
        for v in g.read(&vec![1.0f32; C * H]).inject.unwrap() {
            assert!(v > 1.9, "must approach 2.0, got {v}");
        }
    }

    /// The model-level mixer has no `block_inject_weight`, and must still collapse.
    #[test]
    fn mixer_without_inject_still_collapses() {
        let norm = vec![0.0f32; C * H];
        let down = seeded(R * C * H, 12);
        let up = seeded(C * H * R, 13);
        let r = gr(&norm, &down, &up, None).read(&seeded(C * H, 14));
        assert!(r.inject.is_none());
        assert_eq!(r.mixed_input.len(), H);
    }

    /// The write side adds into the RAW streams, per-stream-scaled. A gate of zero
    /// must leave a stream untouched; a gate of one adds the block output verbatim.
    #[test]
    fn write_scales_each_stream_by_its_own_gate() {
        let norm = vec![0.0f32; C * H];
        let down = seeded(R * C * H, 15);
        let up = seeded(C * H * R, 16);
        let g = gr(&norm, &down, &up, None);
        let mut streams = vec![1.0f32; C * H];
        let block_out = vec![2.0f32; H];
        g.write(&mut streams, &block_out, &[0.0, 1.0, 0.5, 2.0]);
        for d in 0..H {
            assert_eq!(streams[0 * H + d], 1.0, "gate 0 leaves the stream alone");
            assert_eq!(streams[1 * H + d], 3.0, "gate 1 adds the output verbatim");
            assert_eq!(streams[2 * H + d], 2.0);
            assert_eq!(streams[3 * H + d], 5.0);
        }
    }
}

/// Gated RMSNorm with a **sigmoid** output gate — the Gated DeltaNet delta.
///
/// Per-head RMSNorm over `head_dim`, scaled by a PLAIN (ones-initialised) weight,
/// then gated. Qwen3.5/3.8 gate with silu here; qwen4_exp sets
/// `output_gate_type: "sigmoid"` and the reference feeds it through
/// `ACT2FN[self.activation](gate)`.
///
/// Note the weight convention differs from [`grouped_rmsnorm`] above, which
/// carries a `+1`. Both norms exist in this model and they are not interchangeable.
pub fn gated_rmsnorm_sigmoid(
    x: &[f32],
    z: &[f32],
    weight: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    gated_rmsnorm(x, z, weight, n_heads, head_dim, eps, true)
}

/// As above, with the gate activation selected by the config.
///
/// `output_gate_type` decides it, and the reference falls back to `hidden_act`
/// (silu) when the key is ABSENT. The shipped checkpoint sets `sigmoid`, so the
/// two differ on any config that omits the key — do not hardcode either.
pub fn gated_rmsnorm(
    x: &[f32],
    z: &[f32],
    weight: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    sigmoid: bool,
) -> Vec<f32> {
    assert_eq!(x.len(), n_heads * head_dim);
    assert_eq!(z.len(), x.len());
    assert_eq!(weight.len(), head_dim, "weight spans one head");
    let mut out = vec![0.0f32; x.len()];
    for h in 0..n_heads {
        let xh = &x[h * head_dim..(h + 1) * head_dim];
        let zh = &z[h * head_dim..(h + 1) * head_dim];
        let inv = 1.0 / (xh.iter().map(|v| v * v).sum::<f32>() / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            // PLAIN weight, no `+ 1`.
            let normed = xh[i] * inv * weight[i];
            let gate = zh[i] / (1.0 + (-zh[i]).exp()); // silu
            out[h * head_dim + i] = if sigmoid {
                normed / (1.0 + (-zh[i]).exp())
            } else {
                normed * gate
            };
        }
    }
    out
}

#[cfg(test)]
mod gated_sigmoid_tests {
    use super::*;

    /// The gate must be sigmoid, not silu. At z = 0 they differ maximally in
    /// relative terms: sigmoid(0) = 0.5 while silu(0) = 0.
    #[test]
    fn gate_is_sigmoid_not_silu() {
        let (nh, hd) = (2usize, 4usize);
        let x = vec![1.0f32; nh * hd];
        let z = vec![0.0f32; nh * hd];
        let w = vec![1.0f32; hd];
        let out = gated_rmsnorm_sigmoid(&x, &z, &w, nh, hd, 1e-6);
        // rms of all-ones is 1, so normed = 1; gate = sigmoid(0) = 0.5.
        for v in &out {
            assert!((v - 0.5).abs() < 1e-5, "got {v}; silu would give 0.0");
        }
    }

    /// The weight is applied PLAIN. A zero weight must zero the output — the
    /// opposite of the `1 + weight` norm in the same model.
    #[test]
    fn weight_is_plain_not_offset_by_one() {
        let (nh, hd) = (1usize, 4usize);
        let x = vec![2.0f32; hd];
        let z = vec![10.0f32; hd];
        let out = gated_rmsnorm_sigmoid(&x, &z, &vec![0.0f32; hd], nh, hd, 1e-6);
        assert!(
            out.iter().all(|v| v.abs() < 1e-9),
            "a zero weight must zero the output here: {out:?}"
        );
    }

    /// Heads normalise independently.
    #[test]
    fn heads_normalise_independently() {
        let (nh, hd) = (2usize, 4usize);
        let mut x = vec![1.0f32; nh * hd];
        for v in x[hd..].iter_mut() {
            *v = 50.0;
        }
        let z = vec![100.0f32; nh * hd]; // gate ~1
        let w = vec![1.0f32; hd];
        let out = gated_rmsnorm_sigmoid(&x, &z, &w, nh, hd, 1e-6);
        // Both heads are internally uniform, so both normalise to ~1 despite the
        // 50x magnitude difference.
        for v in &out {
            assert!((v - 1.0).abs() < 1e-3, "got {v}");
        }
    }
}
