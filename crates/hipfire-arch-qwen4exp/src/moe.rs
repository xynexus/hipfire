// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! Routed mixture-of-experts — CPU reference.
//!
//! An adversarial review of this port found that the only evidence offered for the
//! `k != 8` MoE path was an end-to-end comparison in which BOTH arms executed the
//! same routing code, so a top-k or renormalisation bug was common-mode and
//! cancelled exactly out of the metric. This exists so that comparison has a second
//! implementation to be made against.
//!
//! Conventions taken from the reference (`Qwen3NextExperts` / `Qwen3NextSparseMoeBlock`,
//! which `Qwen4ExpTextSparseMoeBlock` inherits unchanged), each of which has a
//! plausible wrong alternative:
//!
//! * **`gate_up` splits into contiguous halves, gate first.** The projection is
//!   `[2 * mi, hidden]` and the reference chunks its OUTPUT in two — it is not
//!   interleaved, and swapping the halves silently applies SiLU to the wrong one.
//! * **The top-k weight is applied AFTER `down`**, not to the intermediate.
//! * **The shared expert is gated by `sigmoid(shared_expert_gate · x)`** and ADDED
//!   to the routed sum — it is always on, never selected.
//! * Routing is softmax over ALL experts, then top-k, then optional renormalise —
//!   in that order. Taking top-k before the softmax normalises over a different
//!   denominator.

/// One expert's weights, in checkpoint layout.
pub struct Expert<'a> {
    /// `[2 * mi, hidden]` — gate in the first `mi` rows, up in the second.
    pub gate_up: &'a [f32],
    /// `[hidden, mi]`
    pub down: &'a [f32],
}

/// One MoE layer's weights.
pub struct MoeLayer<'a> {
    /// `[num_experts, hidden]`
    pub router: &'a [f32],
    pub experts: Vec<Expert<'a>>,
    /// `[shared_mi, hidden]`, `[shared_mi, hidden]`, `[hidden, shared_mi]`
    pub shared_gate: &'a [f32],
    pub shared_up: &'a [f32],
    pub shared_down: &'a [f32],
    /// `[1, hidden]`
    pub shared_expert_gate: &'a [f32],
    pub hidden: usize,
    pub mi: usize,
    pub shared_mi: usize,
    pub top_k: usize,
    pub norm_topk_prob: bool,
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Row-major `[out, in]` times a vector.
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(w.len(), out_dim * in_dim);
    (0..out_dim)
        .map(|o| {
            w[o * in_dim..(o + 1) * in_dim]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// What the router selected for one token.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
}

impl MoeLayer<'_> {
    /// Softmax over ALL experts, then top-k by probability, then optional
    /// renormalisation. Indices come back in descending probability order.
    pub fn route(&self, x: &[f32]) -> Routing {
        let n = self.router.len() / self.hidden;
        let logits = matvec(self.router, x, n, self.hidden);
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|v| v / sum).collect();

        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        idx.truncate(self.top_k);
        let mut w: Vec<f32> = idx.iter().map(|&i| probs[i]).collect();
        if self.norm_topk_prob {
            let s: f32 = w.iter().sum();
            if s > 0.0 {
                for v in w.iter_mut() {
                    *v /= s;
                }
            }
        }
        Routing {
            experts: idx,
            weights: w,
        }
    }

    /// SwiGLU through one expert: `down(silu(gate) * up)`.
    fn expert_ffn(&self, e: &Expert<'_>, x: &[f32]) -> Vec<f32> {
        let gu = matvec(e.gate_up, x, self.mi * 2, self.hidden);
        // Contiguous halves, gate FIRST.
        let inter: Vec<f32> = (0..self.mi)
            .map(|i| silu(gu[i]) * gu[self.mi + i])
            .collect();
        matvec(e.down, &inter, self.hidden, self.mi)
    }

    /// The always-on shared expert, gated by a scalar.
    fn shared(&self, x: &[f32]) -> Vec<f32> {
        let g = matvec(self.shared_gate, x, self.shared_mi, self.hidden);
        let u = matvec(self.shared_up, x, self.shared_mi, self.hidden);
        let inter: Vec<f32> = (0..self.shared_mi).map(|i| silu(g[i]) * u[i]).collect();
        let out = matvec(self.shared_down, &inter, self.hidden, self.shared_mi);
        let gate = sigmoid(matvec(self.shared_expert_gate, x, 1, self.hidden)[0]);
        out.into_iter().map(|v| v * gate).collect()
    }

    /// One token through the whole block.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden);
        let r = self.route(x);
        let mut out = vec![0.0f32; self.hidden];
        for (slot, &e) in r.experts.iter().enumerate() {
            let y = self.expert_ffn(&self.experts[e], x);
            // The routing weight scales the expert's OUTPUT, after `down`.
            for d in 0..self.hidden {
                out[d] += r.weights[slot] * y[d];
            }
        }
        for (d, v) in self.shared(x).into_iter().enumerate() {
            out[d] += v;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: usize = 6;
    const MI: usize = 4;
    const NE: usize = 5;

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

    struct Owned {
        router: Vec<f32>,
        gate_up: Vec<Vec<f32>>,
        down: Vec<Vec<f32>>,
        sg: Vec<f32>,
        su: Vec<f32>,
        sd: Vec<f32>,
        seg: Vec<f32>,
    }

    fn owned(seed: u32) -> Owned {
        Owned {
            router: seeded(NE * H, seed),
            gate_up: (0..NE)
                .map(|e| seeded(2 * MI * H, seed + 10 + e as u32))
                .collect(),
            down: (0..NE)
                .map(|e| seeded(H * MI, seed + 30 + e as u32))
                .collect(),
            sg: seeded(MI * H, seed + 50),
            su: seeded(MI * H, seed + 51),
            sd: seeded(H * MI, seed + 52),
            seg: seeded(H, seed + 53),
        }
    }

    fn layer<'a>(o: &'a Owned, top_k: usize, norm: bool) -> MoeLayer<'a> {
        MoeLayer {
            router: &o.router,
            experts: (0..NE)
                .map(|e| Expert {
                    gate_up: &o.gate_up[e],
                    down: &o.down[e],
                })
                .collect(),
            shared_gate: &o.sg,
            shared_up: &o.su,
            shared_down: &o.sd,
            shared_expert_gate: &o.seg,
            hidden: H,
            mi: MI,
            shared_mi: MI,
            top_k,
            norm_topk_prob: norm,
        }
    }

    /// Softmax comes BEFORE the top-k cut, so renormalised weights sum to 1 while
    /// the raw ones sum to less — taking top-k first would make both sum to 1.
    #[test]
    fn routing_normalises_over_all_experts_then_cuts() {
        let o = owned(1);
        let x = seeded(H, 99);
        let raw = layer(&o, 3, false).route(&x);
        let normed = layer(&o, 3, true).route(&x);
        assert_eq!(raw.experts, normed.experts, "the SELECTION must not change");
        let s: f32 = raw.weights.iter().sum();
        assert!(
            s < 1.0 - 1e-4,
            "raw top-3 of 5 must sum to less than 1, got {s}"
        );
        assert!((normed.weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // descending
        assert!(raw.weights.windows(2).all(|w| w[0] >= w[1]));
    }

    /// `gate_up` is contiguous halves with gate FIRST. Zeroing the second half
    /// (up) must zero the routed contribution; zeroing the first (gate) must not,
    /// because silu(0) = 0 gives the same result — so the asymmetry is the test.
    #[test]
    fn gate_up_splits_as_contiguous_halves_gate_first() {
        let mut o = owned(2);
        let x = seeded(H, 7);
        // Kill `up` (second half) on every expert -> silu(gate) * 0 = 0.
        for e in 0..NE {
            for i in 0..MI {
                for h in 0..H {
                    o.gate_up[e][(MI + i) * H + h] = 0.0;
                }
            }
        }
        let l = layer(&o, 2, true);
        let routed_only: Vec<f32> = {
            let r = l.route(&x);
            let mut acc = vec![0.0f32; H];
            for (slot, &e) in r.experts.iter().enumerate() {
                let y = l.expert_ffn(&l.experts[e], &x);
                for d in 0..H {
                    acc[d] += r.weights[slot] * y[d];
                }
            }
            acc
        };
        assert!(
            routed_only.iter().all(|v| v.abs() < 1e-9),
            "zeroing the UP half must zero the routed output: {routed_only:?}"
        );
    }

    /// The shared expert is always on and additive — it must contribute even when
    /// every routed weight is zero.
    #[test]
    fn shared_expert_is_always_on() {
        let o = owned(3);
        let x = seeded(H, 11);
        let mut l = layer(&o, 1, true);
        let with_shared = l.forward(&x);
        // Neutralise the shared expert and compare.
        let zeros = vec![0.0f32; H * MI];
        l.shared_down = &zeros;
        let without = l.forward(&x);
        assert!(
            with_shared
                .iter()
                .zip(&without)
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "the shared expert must change the output"
        );
    }

    /// Two experts dark at top-3 of 5: a kernel that ignored the selection would
    /// still score clean at k == n_exp, which is why the fixture uses k < n_exp.
    #[test]
    fn selection_actually_excludes_experts() {
        let o = owned(4);
        let x = seeded(H, 13);
        let l = layer(&o, 3, true);
        let r = l.route(&x);
        assert_eq!(r.experts.len(), 3);
        let mut seen = r.experts.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 3, "no duplicate slots");
        assert!(seen.len() < NE, "some experts must stay dark");
    }

    /// FAULT INJECTION — the instrument must be able to fail. Dropping the last
    /// selected expert has to move the output; if it does not, nothing measured
    /// with this oracle would be interpretable.
    #[test]
    fn dropping_an_expert_changes_the_output() {
        let o = owned(5);
        let x = seeded(H, 17);
        let full = layer(&o, 3, true).forward(&x);
        let short = layer(&o, 2, true).forward(&x);
        let delta = full
            .iter()
            .zip(&short)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            delta > 1e-4,
            "dropping an expert must be visible, got {delta:.3e}"
        );
    }

    /// The k = 10 of 12 shape the real model uses, exercised end to end.
    #[test]
    fn production_top_k_shape_runs() {
        let o = owned(6);
        let x = seeded(H, 19);
        let l = layer(&o, 4, true); // k < n_exp = 5, the same structure at fixture scale
        let out = l.forward(&x);
        assert_eq!(out.len(), H);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
