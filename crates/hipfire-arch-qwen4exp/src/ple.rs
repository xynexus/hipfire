// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! The PLE short convolution — CPU reference.
//!
//! The n-gram block ends with a depthwise causal convolution over the widened
//! residual, and it is **dilated by `ngram_size`**. That is the detail that makes
//! it a separate kernel from every other short conv in this tree: the Gated
//! DeltaNet mixer in the SAME model uses an undilated kernel-4 conv with a 3-slot
//! ring, so a shared path would silently apply the wrong taps.
//!
//! With `kernel = 4` and `dilation = 3` the state is `(4 - 1) * 3 = 9` positions
//! deep, but only three of those slots are ever read — `t-9`, `t-6`, `t-3` —
//! alongside the current input. The reference notes it cannot use the standard
//! conv path for exactly this reason.
//!
//! Tap layout, for output position `t`:
//!
//! ```text
//! y[t] = sum_k w[k] * x[t - (kernel - 1 - k) * dilation]
//!      = w[0]*x[t-9] + w[1]*x[t-6] + w[2]*x[t-3] + w[3]*x[t]
//! ```
//!
//! then SiLU. `w[kernel - 1]` multiplies the CURRENT input — reversing the kernel
//! is the obvious mistake and produces a plausible, wrong result.

/// One decode step of the dilated depthwise conv, followed by SiLU.
///
/// `state` is `[channels, state_len]` holding the last `state_len` positions
/// per channel, **oldest first**, and is advanced in place. `x` is the current
/// position. Returns `[channels]`.
pub fn dilated_conv_silu_step(
    state: &mut [f32],
    x: &[f32],
    weight: &[f32],
    channels: usize,
    state_len: usize,
    kernel: usize,
    dilation: usize,
) -> Vec<f32> {
    assert_eq!(state.len(), channels * state_len);
    assert_eq!(x.len(), channels);
    assert_eq!(weight.len(), channels * kernel);
    assert_eq!(
        state_len,
        (kernel - 1) * dilation,
        "state depth must be (kernel - 1) * dilation"
    );

    let mut out = vec![0.0f32; channels];
    for c in 0..channels {
        let st = &state[c * state_len..(c + 1) * state_len];
        let w = &weight[c * kernel..(c + 1) * kernel];
        let mut acc = 0.0f32;
        for k in 0..kernel {
            let back = (kernel - 1 - k) * dilation;
            // `back == 0` is the current input; anything else indexes the ring,
            // where the newest position sits at `state_len - 1`.
            let v = if back == 0 {
                x[c]
            } else {
                st[state_len - back]
            };
            acc += w[k] * v;
        }
        out[c] = acc / (1.0 + (-acc).exp());
    }

    // Advance: drop the oldest position, append the current one.
    for c in 0..channels {
        let base = c * state_len;
        state.copy_within(base + 1..base + state_len, base);
        state[base + state_len - 1] = x[c];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const C: usize = 3;
    const K: usize = 4;
    const D: usize = 3;
    const S: usize = (K - 1) * D; // 9

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    /// Only three of the nine state slots are ever read: t-9, t-6, t-3. Perturbing
    /// any OTHER slot must not change the output — the check that catches an
    /// undilated conv, which would read t-1, t-2, t-3 instead.
    #[test]
    fn only_the_dilated_taps_are_read() {
        let w = vec![1.0f32; C * K];
        let x = vec![0.0f32; C];
        let base = vec![0.0f32; C * S];
        let mut s0 = base.clone();
        let want = dilated_conv_silu_step(&mut s0, &x, &w, C, S, K, D);

        for slot in 0..S {
            let mut s = base.clone();
            s[slot] = 10.0; // channel 0 only
            let got = dilated_conv_silu_step(&mut s, &x, &w, C, S, K, D);
            // state_len - back for back in {9, 6, 3} => slots 0, 3, 6.
            let is_tap = slot == 0 || slot == 3 || slot == 6;
            if is_tap {
                assert!(got[0] != want[0], "slot {slot} is a tap and must matter");
            } else {
                assert_eq!(got[0], want[0], "slot {slot} is not a tap");
            }
        }
    }

    /// `w[kernel - 1]` multiplies the CURRENT input. A reversed kernel would put
    /// `w[0]` there, which this pins.
    #[test]
    fn last_weight_multiplies_the_current_input() {
        let mut w = vec![0.0f32; C * K];
        for c in 0..C {
            w[c * K + (K - 1)] = 2.0; // only the current-input tap is live
        }
        let x = vec![3.0f32, -1.0, 0.5];
        let mut s = vec![7.0f32; C * S]; // history is loud but must be ignored
        let out = dilated_conv_silu_step(&mut s, &x, &w, C, S, K, D);
        for c in 0..C {
            assert!((out[c] - silu(2.0 * x[c])).abs() < 1e-6, "channel {c}");
        }
    }

    /// Channels are independent — it is a DEPTHWISE conv, not a full one.
    #[test]
    fn channels_do_not_mix() {
        let mut w = vec![0.0f32; C * K];
        w[0 * K + (K - 1)] = 1.0;
        w[1 * K + (K - 1)] = 1.0;
        w[2 * K + (K - 1)] = 1.0;
        let mut s = vec![0.0f32; C * S];
        let a = dilated_conv_silu_step(&mut s, &[1.0, 0.0, 0.0], &w, C, S, K, D);
        assert!(a[0] > 0.0 && a[1].abs() < 1e-9 && a[2].abs() < 1e-9);
    }

    /// The ring advances by one position per step, oldest out and current in.
    #[test]
    fn state_advances_by_one_position() {
        let w = vec![0.0f32; C * K];
        let mut s: Vec<f32> = (0..C * S).map(|i| i as f32).collect();
        let before = s.clone();
        dilated_conv_silu_step(&mut s, &[99.0, 98.0, 97.0], &w, C, S, K, D);
        for c in 0..C {
            let b = c * S;
            // Everything shifted left by one within the channel...
            assert_eq!(&s[b..b + S - 1], &before[b + 1..b + S], "channel {c} shift");
            // ...and the current input landed in the newest slot.
            assert_eq!(s[b + S - 1], [99.0, 98.0, 97.0][c]);
        }
    }

    /// A full run of steps must reproduce a direct convolution over the same
    /// sequence — the end-to-end check that the ring bookkeeping is right.
    #[test]
    fn stepping_matches_a_direct_convolution() {
        let w: Vec<f32> = (0..C * K).map(|i| (i as f32 * 0.37).sin()).collect();
        let seq: Vec<Vec<f32>> = (0..12)
            .map(|t| (0..C).map(|c| ((t * C + c) as f32 * 0.11).cos()).collect())
            .collect();

        let mut state = vec![0.0f32; C * S];
        let mut stepped = Vec::new();
        for x in &seq {
            stepped.push(dilated_conv_silu_step(&mut state, x, &w, C, S, K, D));
        }

        // Direct: zero-padded history before t = 0.
        for (t, got) in stepped.iter().enumerate() {
            for c in 0..C {
                let mut acc = 0.0f32;
                for k in 0..K {
                    let back = (K - 1 - k) * D;
                    let v = if back <= t { seq[t - back][c] } else { 0.0 };
                    acc += w[c * K + k] * v;
                }
                assert!((got[c] - silu(acc)).abs() < 1e-6, "t={t} c={c}");
            }
        }
    }
}

/// The whole PLE block for one token: n-gram value, per-stream gate, dilated conv.
///
/// The block injects hashed n-gram features into every hyper-connection stream. A
/// shared VALUE is projected from the token's concatenated n-gram embedding, and
/// one KEY per stream; the normalised stream activations gate that value, and a
/// dilated depthwise convolution then adds local lexical context.
///
/// Two details are easy to get wrong and are both load-bearing:
///
/// * The gate is a **signed square root** of the key·query dot product, clamped
///   away from zero before the root — not a plain dot product and not a softmax.
/// * The convolution reads the **normalised** gated value, while the residual adds
///   the **un-normalised** one. Feeding one to both is a plausible reading that
///   changes the output.
///
/// The caller owns `conv_state` (`hc_count * hidden * state_len()`), which carries
/// the dilated tap history across tokens.
pub struct PleLayer<'a> {
    /// `[hc_count * hidden, embed_dim]`
    pub key_proj: &'a [f32],
    /// `[hidden, embed_dim]`
    pub value_proj: &'a [f32],
    /// `[hc_count * hidden]`, all three grouped by `hidden` with the `1 + w` convention.
    pub norm_key: &'a [f32],
    pub norm_query: &'a [f32],
    pub norm_conv: &'a [f32],
    /// `[hc_count * hidden, kernel]`, depthwise.
    pub conv_weight: &'a [f32],
    pub hc_count: usize,
    pub hidden: usize,
    pub embed_dim: usize,
    pub kernel: usize,
    /// `ngram_size` — see the module docs on why this conv is dilated.
    pub dilation: usize,
    pub eps: f32,
}

impl PleLayer<'_> {
    pub fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    pub fn state_len(&self) -> usize {
        (self.kernel - 1) * self.dilation
    }

    /// One token. `ngram_embed` is the concatenated per-head n-gram embedding
    /// (`embed_dim`), `hidden_wide` the layer's residual streams (`width()`).
    pub fn step(
        &self,
        hidden_wide: &[f32],
        ngram_embed: &[f32],
        conv_state: &mut [f32],
    ) -> Vec<f32> {
        assert_eq!(hidden_wide.len(), self.width());
        assert_eq!(ngram_embed.len(), self.embed_dim);

        let mv = |w: &[f32], x: &[f32], o: usize, i: usize| -> Vec<f32> {
            (0..o)
                .map(|r| (0..i).map(|c| w[r * i + c] * x[c]).sum())
                .collect()
        };
        let key = crate::hc::grouped_rmsnorm(
            &mv(self.key_proj, ngram_embed, self.width(), self.embed_dim),
            self.norm_key,
            self.hidden,
            self.eps,
        );
        let value = mv(self.value_proj, ngram_embed, self.hidden, self.embed_dim);
        let query = crate::hc::grouped_rmsnorm(hidden_wide, self.norm_query, self.hidden, self.eps);

        let scale = 1.0f32 / (self.hidden as f32).sqrt();
        let mut gated = vec![0.0f32; self.width()];
        for s in 0..self.hc_count {
            let b = s * self.hidden;
            let dot: f32 = (0..self.hidden)
                .map(|d| key[b + d] * query[b + d])
                .sum::<f32>()
                * scale;
            // Signed sqrt, clamped away from zero before rooting.
            let g = dot.abs().max(1e-6).sqrt() * dot.signum();
            let gate = 1.0 / (1.0 + (-g).exp());
            for d in 0..self.hidden {
                gated[b + d] = gate * value[d];
            }
        }

        let normed = crate::hc::grouped_rmsnorm(&gated, self.norm_conv, self.hidden, self.eps);
        let conv = dilated_conv_silu_step(
            conv_state,
            &normed,
            self.conv_weight,
            self.width(),
            self.state_len(),
            self.kernel,
            self.dilation,
        );
        // Residual adds the UN-normalised gated value.
        gated.iter().zip(&conv).map(|(a, b)| a + b).collect()
    }
}
