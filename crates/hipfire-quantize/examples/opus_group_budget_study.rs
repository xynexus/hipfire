// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors
// hipfire — see LICENSE and NOTICE in the project root.

//! P4 step 1 — is a per-GROUP outlier budget worth a format change?
//!
//! `opus_outlier_budget_study` answers the per-TENSOR version of this question,
//! and the answer already shipped as `HIPFIRE_OUTLIERS_BY_LAYER`. This one asks
//! the strictly harder question the plan gates the format work on: inside a
//! single tensor, does water-filling `N_out` across its 256-groups beat giving
//! every group the same `N_out`, at the SAME total outlier count?
//!
//! Cost of saying yes: the container infers `N_out` from block length, so
//! per-group variation needs variable-length blocks plus a `[u32; n_groups]`
//! offset prefix — it costs O(1) block indexing, which the weight pager and
//! every load-time expander rely on. Hence: measure first.
//!
//! **Kill criterion (plan P4 step 2): < 5% SSE reduction closes the item.**
//!
//! Three arms per tensor, all bit-matched against uniform `N_UNIFORM`:
//!
//! 1. **free offsets** — same total slots, offset table not charged. An upper
//!    bound on the idea, and the arm that says whether allocation freedom is
//!    worth anything at all.
//! 2. **u32 offsets** — same BYTES, so the offset prefix is paid for out of the
//!    slot budget. This is the arm the decision rests on.
//! 3. **Lagrangian bound** — the best any allocator could do in arm 2. The SSE
//!    curves turn out to be non-convex, so greedy water-filling is an
//!    approximation and a greedy miss could not close the item on its own.
//!
//! A break-even sweep over the offset width then says where the line is, so a
//! "no" comes with the addressing budget that would turn it into a "yes".
//!
//! Same caveat as the tensor-level study, and it is not a small one: weight SSE
//! is a proxy for KLD and the two have demonstrably disagreed in this exact
//! experiment (d77fa637a: N=7 scored worse KLD than N=3 while spending more
//! bits). A pass here is a reason to run KLD, never a substitute for it.
//!
//!   cargo run --release -p hipfire-quantize --example opus_group_budget_study \
//!     -- <model.safetensors> [max_groups_per_tensor]

use hipfire_quantize::codecs::{mixed_clipsearch, mixed_overlay_error};
use hipfire_quantize::{cpu_fwht_256, gen_fwht_signs};
use std::collections::BinaryHeap;

/// `codecs::MIXED_CLIP_GRID`, which is private. Inlined so the `n_out = 0` point
/// is searched over the SAME scale grid as every `n_out ≥ 1` point — otherwise
/// the first marginal measures a grid change instead of an overlay slot.
/// (`symmetric_clipsearch` bottoms out at 0.6 and would do exactly that.)
const CLIP_GRID: [f32; 14] = [
    1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35,
];

/// Overlay slots per group in the uniform arm — the shipped `oq4.25++` point
/// (`130 + 2·3` = 136 B/group).
const N_UNIFORM: usize = 3;
/// Water-filling may not push a single group past this.
const NMAX: usize = 16;
/// Bytes of the `[u32; n_groups]` offset prefix, per group. Variable-length
/// blocks are the only P4 option that realises the gain, and this is its price.
const OFFSET_BYTES: usize = 4;
/// Bulk W4 payload of a 256-group, before any overlay slots.
const BULK_BYTES: usize = 130;

/// Total overlay slots the variable-length container can afford across
/// `n_groups`, at the same byte count as uniform `N_UNIFORM`.
///
/// `n_groups·(130 + 2·N_UNIFORM)  ==  n_groups·(130 + OFFSET_BYTES) + 2·ΣN`
fn budget_at(n_groups: usize, offset_bytes: usize) -> usize {
    let uniform_bytes = n_groups * (BULK_BYTES + 2 * N_UNIFORM);
    let fixed = n_groups * (BULK_BYTES + offset_bytes);
    uniform_bytes.saturating_sub(fixed) / 2
}

fn charged_budget(n_groups: usize) -> usize {
    budget_at(n_groups, OFFSET_BYTES)
}

/// Tensor name suffixes to sample. One tensor per suffix, taken from the middle
/// of the stack — early and final layers are atypical.
const SUFFIXES: [&str; 4] = ["down_proj", "o_proj", "gate_proj", "q_proj"];

/// SSE of a group at `n_out` overlay slots, under the joint selector.
/// `n_out = 0` is the pure-int4 group, which `mixed_clipsearch` cannot express.
fn group_sse(group: &[f32; 256], n_out: usize) -> f64 {
    if n_out == 0 {
        let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        return CLIP_GRID
            .iter()
            .map(|&clip| {
                let scale = (clip * amax / 7.0).max(1e-12);
                let inv = 1.0 / scale;
                group
                    .iter()
                    .map(|&v| {
                        let d = v - (v * inv).round().clamp(-7.0, 7.0) * scale;
                        (d * d) as f64
                    })
                    .sum::<f64>()
            })
            .fold(f64::INFINITY, f64::min);
    }
    let (scale, indices) = mixed_clipsearch(group, n_out);
    mixed_overlay_error(group, scale, &indices, n_out) as f64
}

/// f64 ordered by `total_cmp`, so marginal gains can go in a `BinaryHeap`.
#[derive(PartialEq)]
struct Key(f64);
impl Eq for Key {}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Spend `budget` unit steps across the per-group SSE curves, always taking the
/// largest remaining marginal. Returns the per-group allocation.
///
/// A heap, not a global top-`budget` sort: the sort is only equivalent when
/// every curve is convex in `n`, and these curves are re-fitting the group scale
/// at each `n`, so convexity is empirical rather than guaranteed. The heap
/// respects the prefix constraint either way (a group's step `n+1` only becomes
/// available once step `n` is taken).
fn water_fill(curves: &[Vec<f64>], budget: usize) -> Vec<usize> {
    let mut alloc = vec![0usize; curves.len()];
    let mut heap: BinaryHeap<(Key, usize)> = curves
        .iter()
        .enumerate()
        .map(|(g, c)| (Key(c[0] - c[1]), g))
        .collect();
    for _ in 0..budget {
        let Some((_, g)) = heap.pop() else { break };
        alloc[g] += 1;
        if alloc[g] < NMAX {
            heap.push((Key(curves[g][alloc[g]] - curves[g][alloc[g] + 1]), g));
        }
    }
    alloc
}

/// Best SSE any allocation of `budget` slots could reach — a true bound, not a
/// heuristic's output.
///
/// Greedy is only exact on convex curves and these are not convex, so a greedy
/// result that misses the gate cannot by itself close the item: the shortfall
/// might be the allocator's, not the idea's. The Lagrangian dual settles it. For
/// any `λ ≥ 0`,
///
/// ```text
/// L(λ) = Σ_g min_n ( sse_g(n) + λ·n )  −  λ·budget
/// ```
///
/// lower-bounds the minimum achievable SSE at that budget, whatever the curve
/// shapes. Maximising `L` over a λ sweep gives the tightest such bound, so
/// `uniform − max_λ L(λ)` is an UPPER bound on the gain available. If even that
/// misses 5%, the item is dead on the merits.
fn dual_bound(curves: &[Vec<f64>], budget: usize) -> f64 {
    let mut best = f64::NEG_INFINITY;
    // λ is an SSE-per-slot exchange rate; the useful range is set by the largest
    // single-slot gain any group offers.
    let lambda_max = curves
        .iter()
        .map(|c| c[0] - c[NMAX])
        .fold(0.0f64, f64::max)
        .max(1e-12);
    for step in 0..=512 {
        let lambda = lambda_max * step as f64 / 512.0;
        let sum: f64 = curves
            .iter()
            .map(|c| {
                (0..=NMAX)
                    .map(|n| c[n] + lambda * n as f64)
                    .fold(f64::INFINITY, f64::min)
            })
            .sum();
        best = best.max(sum - lambda * budget as f64);
    }
    best
}

/// Count curves whose marginal gain rises as `n` grows — where that happens the
/// greedy is an approximation, so the study has to say how often it happens.
fn nonconvex(curves: &[Vec<f64>]) -> usize {
    curves
        .iter()
        .filter(|c| (1..NMAX).any(|n| (c[n] - c[n + 1]) > (c[n - 1] - c[n]) + 1e-12))
        .count()
}

fn selfcheck() {
    // Two groups, one steeply improving and one flat: every unit must land on
    // the steep curve, and the flat one must be left at zero.
    let steep: Vec<f64> = (0..=NMAX).map(|n| 10.0 - n as f64).collect();
    let flat: Vec<f64> = (0..=NMAX).map(|_| 1.0).collect();
    assert_eq!(water_fill(&[steep.clone(), flat.clone()], 4), vec![4, 0]);
    // Budget beyond one curve's cap spills into the other rather than exceeding NMAX.
    let alloc = water_fill(&[steep, flat], NMAX + 3);
    assert_eq!(alloc[0], NMAX);
    assert_eq!(alloc[1], 3);
    // A straight line has zero curvature, so it must not be reported as non-convex.
    let line: Vec<f64> = (0..=NMAX).map(|n| 10.0 - n as f64).collect();
    assert_eq!(nonconvex(&[line]), 0);

    // The bound must never exceed what a real allocation achieves — on a convex
    // pair, where greedy IS optimal, it must also be tight.
    let convex: Vec<f64> = (0..=NMAX).map(|n| 1.0 / (1.0 + n as f64)).collect();
    let curves = vec![convex.clone(), convex];
    let budget = 6;
    let achieved: f64 = curves
        .iter()
        .zip(&water_fill(&curves, budget))
        .map(|(c, &n)| c[n])
        .sum();
    let bound = dual_bound(&curves, budget);
    assert!(
        bound <= achieved + 1e-9,
        "bound {bound} exceeded {achieved}"
    );
    assert!(
        achieved - bound < 0.02,
        "bound {bound} too loose against {achieved}"
    );
}

fn main() {
    selfcheck();

    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1).cloned() else {
        eprintln!(
            "usage: opus_group_budget_study <model.safetensors> [max_groups_per_tensor]\n\
             \n\
             Any BF16 safetensors checkpoint works; the recorded result used\n\
             Qwen3.5-0.8B. No default path — the model store is machine-local."
        );
        std::process::exit(2);
    };
    let cap: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2048);

    let file = std::fs::File::open(&path).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let hlen = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&mmap[8..8 + hlen]).expect("parse header");
    let base = 8 + hlen;
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);

    println!("model: {path}");
    println!("cap:   {cap} groups per tensor");
    println!("arms:  uniform N={N_UNIFORM} vs water-filled at the same total slots\n");

    let mut verdicts: Vec<(String, f64)> = Vec::new();
    let mut sampled: Vec<(String, f64, Vec<Vec<f64>>)> = Vec::new();
    for suffix in SUFFIXES {
        let mut names: Vec<&String> = header
            .as_object()
            .unwrap()
            .keys()
            .filter(|n| n.contains(suffix) && n.ends_with(".weight"))
            .filter(|n| header[*n]["dtype"].as_str() == Some("BF16"))
            .collect();
        if names.is_empty() {
            continue;
        }
        names.sort();
        let name = names[names.len() / 2];

        let off: Vec<usize> = header[name]["data_offsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let bytes = &mmap[base + off[0]..base + off[1]];
        let mut groups: Vec<[f32; 256]> = Vec::new();
        for chunk in bytes.chunks_exact(2).collect::<Vec<_>>().chunks(256) {
            if chunk.len() < 256 || groups.len() >= cap {
                break;
            }
            let mut g = [0.0f32; 256];
            for (i, c) in chunk.iter().enumerate() {
                g[i] = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
            cpu_fwht_256(&mut g, &signs1, &signs2);
            groups.push(g);
        }
        if groups.is_empty() {
            continue;
        }

        // curves[g][n] = SSE of group g at n overlay slots, n = 0..=NMAX.
        let curves: Vec<Vec<f64>> = groups
            .iter()
            .map(|g| (0..=NMAX).map(|n| group_sse(g, n)).collect())
            .collect();

        let uniform: f64 = curves.iter().map(|c| c[N_UNIFORM]).sum();
        let alloc = water_fill(&curves, N_UNIFORM * curves.len());
        let filled: f64 = curves.iter().zip(&alloc).map(|(c, &n)| c[n]).sum();
        let reduction = 100.0 * (uniform - filled) / uniform;

        // The arm that decides the item: same BYTES, not same slots.
        let charged_alloc = water_fill(&curves, charged_budget(curves.len()));
        let charged: f64 = curves.iter().zip(&charged_alloc).map(|(c, &n)| c[n]).sum();
        let charged_reduction = 100.0 * (uniform - charged) / uniform;
        let bound = dual_bound(&curves, charged_budget(curves.len()));
        let bound_reduction = 100.0 * (uniform - bound) / uniform;
        verdicts.push((suffix.to_string(), bound_reduction));

        let mut hist = [0usize; NMAX + 1];
        for &n in &alloc {
            hist[n] += 1;
        }
        let spent: usize = alloc.iter().sum();
        println!("== {name} ==");
        println!(
            "  {} groups   uniform SSE {uniform:.6}   non-convex curves: {} / {}",
            curves.len(),
            nonconvex(&curves),
            curves.len()
        );
        println!(
            "  free offsets  ({spent} slots): {filled:.6}  {reduction:+.2}%   \
             [upper bound — offset table not charged]"
        );
        println!(
            "  u32 offsets   ({} slots): {charged:.6}  {charged_reduction:+.2}%   \
             [the decision arm — same bytes as uniform]",
            charged_alloc.iter().sum::<usize>()
        );
        println!(
            "  u32 offsets, BEST POSSIBLE: {bound:.6}  {bound_reduction:+.2}%   \
             [Lagrangian bound — no allocator can beat this]"
        );
        let occupied: Vec<String> = hist
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(n, c)| format!("N={n}:{c}"))
            .collect();
        println!("  allocation: {}\n", occupied.join("  "));
        sampled.push((suffix.to_string(), uniform, curves));
    }

    // What addressing cost would the idea survive? Everything above says "not
    // u32"; this says where the line actually is.
    println!("== break-even on the offset prefix (Lagrangian bound vs uniform) ==");
    println!("  offset  slots/group   {}", {
        let heads: Vec<String> = sampled.iter().map(|(s, _, _)| format!("{s:>11}")).collect();
        heads.join("")
    });
    for offset_bytes in 0..=2 * N_UNIFORM {
        let cells: Vec<String> = sampled
            .iter()
            .map(|(_, uniform, curves)| {
                let bound = dual_bound(curves, budget_at(curves.len(), offset_bytes));
                format!("{:>10.2}%", 100.0 * (uniform - bound) / uniform)
            })
            .collect();
        println!(
            "  {offset_bytes} B     {:>4.1}         {}",
            budget_at(1_000_000, offset_bytes) as f64 / 1_000_000.0,
            cells.join("")
        );
    }
    println!(
        "  (u32 = 4 B is what variable-length blocks need for O(1) indexing; \n\
         2 B caps a tensor at 65535 blocks, 1 B cannot address a block at all.)\n"
    );

    println!("== verdict (plan P4 step 2: < 5% closes the item) ==");
    println!("  best achievable gain at equal bytes, per tensor:");
    let best_gain = verdicts
        .iter()
        .fold(f64::NEG_INFINITY, |m, (_, r)| m.max(*r));
    for (suffix, reduction) in &verdicts {
        println!("  {suffix:<11} {reduction:+.2}%");
    }
    println!(
        "\n  best tensor: {best_gain:+.2}%  →  {}",
        if best_gain < 5.0 {
            "CLOSE the item — record the negative result in opus-quant.md"
        } else {
            "clears the gate — but confirm with a real KLD run before touching the format"
        }
    );
    println!(
        "\nNOTE: charging the offset prefix costs {} of the {N_UNIFORM} slots per \n\
         group — {OFFSET_BYTES} B of offsets buys back only {} overlay entries, so the \n\
         decision arm water-fills a much smaller budget than the free-offset arm.",
        N_UNIFORM - charged_budget(1_000_000) / 1_000_000,
        charged_budget(1_000_000) / 1_000_000,
    );
    println!(
        "NOTE: the SSE curves are non-convex (the joint selector re-fits the \n\
         group scale at each N, and an extra slot can unlock a tighter one), so \n\
         greedy water-filling is an approximation and its number alone could \n\
         never close the item. The verdict is taken from the Lagrangian bound, \n\
         which no allocator can beat, so a miss there is a miss on the merits."
    );
}
