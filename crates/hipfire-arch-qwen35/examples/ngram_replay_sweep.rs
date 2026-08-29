// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ngram_replay_sweep: CPU-only acceptance oracle for n-gram spec-decode.
//!
//! Replays a recorded token stream through a multi-order n-gram table and
//! asks, at every position, what the table *would* have drafted. Because the
//! stream is its own ground truth, a correct prediction is exactly a token the
//! target would have accepted — so this measures acceptance without a verify
//! pass, a drafter, or a GPU.
//!
//! What it answers (the four table-design questions):
//!   1. mixed order  — per-order precision for n ∈ {2,3,4,5}
//!   2. multi-token  — chained spine length, broken out by the order that hit
//!   3. admission    — precision bucketed by the gram's count at hit time
//!   4. budget       — share of hits retained if only the top-K grams are kept
//!
//! Caveat: corpus text is a proxy for a decode stream. It captures prompt-echo
//! repetition (the dominant PLD win) but not the model's own output-side
//! repetition. Treat the numbers as a lower bound on a real serving stream.
//!
//! Usage:
//!   ngram_replay_sweep --tokenizer <any-qwen.hfq> --corpus <path> \
//!       [--field text] [--rs-dir crates/] [--max-tokens 2000000] \
//!       [--orders 5,4,3,2] [--min-count 2] [--max-spine 8]

use std::collections::HashMap;
use std::path::Path;

use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::hfq::HfqFile;

/// One context -> next-token histogram, plus the age bookkeeping the real
/// design needs (last_seen drives MRU eviction; hits_served sizes the budget).
#[derive(Default)]
struct Entry {
    /// (token, count), kept tiny — most contexts have 1-3 distinct followers.
    counts: Vec<(u32, u32)>,
    /// Stream position where this gram was last observed.
    last_seen: u32,
    /// How many correct drafts this gram served over the whole replay.
    hits_served: u32,
    /// Oldest / newest token of the context, for the block-occupancy report.
    first_tok: u32,
    last_tok: u32,
}

impl Entry {
    #[inline]
    fn observe(&mut self, next: u32, pos: u32) {
        self.last_seen = pos;
        match self.counts.iter_mut().find(|(t, _)| *t == next) {
            Some((_, c)) => *c = c.saturating_add(1),
            None => self.counts.push((next, 1)),
        }
    }

    #[inline]
    fn best(&self) -> Option<(u32, u32)> {
        self.counts.iter().copied().max_by_key(|(_, c)| *c)
    }

    #[inline]
    fn total(&self) -> u32 {
        self.counts.iter().map(|(_, c)| *c).sum()
    }
}

/// splitmix64 over (order, tokens) — the same fingerprint-only keying the
/// on-disk table would use. No exact key stored: a collision yields a bad
/// draft, which the target rejects, so correctness never depends on it.
#[inline]
fn fingerprint(order: u8, toks: &[u32]) -> u64 {
    let mut h = 0x9e3779b97f4a7c15u64 ^ (order as u64).wrapping_mul(0xff51afd7ed558ccd);
    for &t in toks {
        h ^= t as u64;
        h = h.wrapping_mul(0xbf58476d1ce4e5b9);
        h ^= h >> 31;
        h = h.wrapping_mul(0x94d049bb133111eb);
        h ^= h >> 29;
    }
    h
}

#[derive(Default, Clone)]
struct OrderStats {
    proposals: u64,
    correct: u64,
}

fn bucket_count(c: u32) -> usize {
    match c {
        0..=1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        _ => 5,
    }
}
const COUNT_BUCKETS: [&str; 6] = ["1", "2", "3-4", "5-8", "9-16", "17+"];

fn bucket_age(a: u32) -> usize {
    match a {
        0..=63 => 0,
        64..=511 => 1,
        512..=4095 => 2,
        4096..=32767 => 3,
        _ => 4,
    }
}
const AGE_BUCKETS: [&str; 5] = ["<64", "64-511", "512-4k", "4k-32k", "32k+"];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut tokenizer_path = String::new();
    let mut corpus = String::new();
    let mut field = "text".to_string();
    let mut rs_dir = String::new();
    let mut max_tokens = 2_000_000usize;
    let mut orders: Vec<u8> = vec![5, 4, 3, 2];
    let mut min_count = 2u32;
    let mut max_spine = 8usize;
    // Continuation gate: after the first drafted token, only keep extending the
    // chain while the winning order is >= this. Order is the confidence signal
    // — a chain that falls back to a bigram is drifting, and every further
    // token costs verify width for near-nothing.
    let mut chain_floor = 0u8;
    // Split the stream: the first `prime_frac` is observe-only (stands in for a
    // cold table built offline), the remainder is what we score. `freeze` stops
    // the scored half from feeding the table, isolating cold-table value from
    // the hot/self-warming effect.
    let mut prime_frac = 0.0f64;
    let mut freeze = false;
    // Block-occupancy report: model a fixed file of `block_bytes` blocks and
    // report fill skew under first-token vs last-token keying.
    let mut block_report = false;
    let mut file_bytes: u64 = 1 << 30;
    let block_bytes: u64 = 4096;
    let rec_bytes: u64 = 24;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tokenizer" => {
                tokenizer_path = args[i + 1].clone();
                i += 2;
            }
            "--corpus" => {
                corpus = args[i + 1].clone();
                i += 2;
            }
            "--field" => {
                field = args[i + 1].clone();
                i += 2;
            }
            "--rs-dir" => {
                rs_dir = args[i + 1].clone();
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--orders" => {
                orders = args[i + 1]
                    .split(',')
                    .map(|s| s.trim().parse::<u8>().expect("--orders ints"))
                    .collect();
                orders.sort_by(|a, b| b.cmp(a)); // longest first
                i += 2;
            }
            "--min-count" => {
                min_count = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--prime-frac" => {
                prime_frac = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--freeze" => {
                freeze = true;
                i += 1;
            }
            "--block-report" => {
                block_report = true;
                i += 1;
            }
            "--file-mb" => {
                file_bytes = args[i + 1].parse::<u64>().unwrap() << 20;
                i += 2;
            }
            "--self-check" => {
                i += 1;
            }
            "--chain-floor" => {
                chain_floor = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--max-spine" => {
                max_spine = args[i + 1].parse().unwrap();
                i += 2;
            }
            other => panic!("unknown arg {other:?}"),
        }
    }
    if args.iter().any(|a| a == "--self-check") {
        self_check();
        return;
    }
    assert!(
        !tokenizer_path.is_empty(),
        "--tokenizer <qwen.hfq> required"
    );

    // ── tokenizer: index-only open, no tensor pages touched, no GPU ──
    let hfq = HfqFile::open_index_only(Path::new(&tokenizer_path)).expect("open hfq index");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer from hfq");

    // ── corpus -> one flat token stream ──
    let mut text = String::new();
    if !rs_dir.is_empty() {
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        collect_rs(Path::new(&rs_dir), &mut files);
        files.sort();
        for f in files {
            if let Ok(s) = std::fs::read_to_string(&f) {
                text.push_str(&s);
                text.push('\n');
            }
            if text.len() > max_tokens * 8 {
                break;
            }
        }
    } else {
        let raw = std::fs::read_to_string(&corpus).expect("read corpus");
        if corpus.ends_with(".jsonl") {
            for line in raw.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let v: serde_json::Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(s) = v.get(&field).and_then(|x| x.as_str()) {
                    text.push_str(s);
                    text.push('\n');
                }
                if text.len() > max_tokens * 8 {
                    break;
                }
            }
        } else {
            text = raw;
        }
    }

    let mut tokens = tok.encode(&text);
    tokens.truncate(max_tokens);
    let n = tokens.len();
    assert!(n > 1000, "need a real stream, got {n} tokens");
    eprintln!(
        "[replay] {n} tokens | orders={orders:?} min_count={min_count} max_spine={max_spine}"
    );

    let mut table: HashMap<u64, Entry> = HashMap::new();
    let mut per_order: HashMap<u8, OrderStats> = HashMap::new();
    let mut by_count = vec![OrderStats::default(); COUNT_BUCKETS.len()];
    let mut by_age = vec![OrderStats::default(); AGE_BUCKETS.len()];

    // Mixed-order policy: hit rate + accepted spine length.
    let mut mixed_steps = 0u64; // positions where any order proposed
    let mut mixed_accepted = 0u64; // total tokens accepted across those steps
    let mut mixed_drafted = 0u64; // total tokens drafted across those steps
    let mut spine_hist = vec![0u64; max_spine + 1];
    let mut spine_by_order: HashMap<u8, (u64, u64)> = HashMap::new(); // order -> (steps, accepted)
    let mut steps_total = 0u64;

    let max_order = *orders.iter().max().unwrap() as usize;

    let eval_start = (n as f64 * prime_frac) as usize;
    if prime_frac > 0.0 {
        eprintln!(
            "[replay] prime on [0,{eval_start}) observe-only; score [{eval_start},{n}) freeze={freeze}"
        );
    }
    for i in 0..n {
        let scoring = i >= eval_start;
        if scoring {
            steps_total += 1;
        }

        // ── predict phase: what would the table draft at position i? ──
        if i >= max_order && scoring {
            // Per-order precision (independent arms, for the mixed-order question).
            for &ord in &orders {
                let o = ord as usize;
                let key = fingerprint(ord, &tokens[i - o..i]);
                if let Some(e) = table.get(&key) {
                    if let Some((pred, cnt)) = e.best() {
                        if cnt >= min_count {
                            let correct = pred == tokens[i];
                            let s = per_order.entry(ord).or_default();
                            s.proposals += 1;
                            s.correct += correct as u64;

                            let cb = bucket_count(cnt);
                            by_count[cb].proposals += 1;
                            by_count[cb].correct += correct as u64;

                            let ab = bucket_age(i as u32 - e.last_seen);
                            by_age[ab].proposals += 1;
                            by_age[ab].correct += correct as u64;
                        }
                    }
                }
            }

            // Mixed-order chained spine: highest order that clears min_count
            // wins the first token, then re-probe at full order for the next.
            let mut spine: Vec<u32> = Vec::with_capacity(max_spine);
            let mut first_order: Option<u8> = None;
            let mut ctx: Vec<u32> = tokens[i - max_order..i].to_vec();
            for step in 0..max_spine {
                let mut hit: Option<(u8, u32)> = None;
                for &ord in &orders {
                    if step > 0 && ord < chain_floor {
                        continue;
                    }
                    let o = ord as usize;
                    if ctx.len() < o {
                        continue;
                    }
                    let key = fingerprint(ord, &ctx[ctx.len() - o..]);
                    if let Some(e) = table.get(&key) {
                        if let Some((pred, cnt)) = e.best() {
                            if cnt >= min_count {
                                hit = Some((ord, pred));
                                break; // orders is longest-first
                            }
                        }
                    }
                }
                match hit {
                    Some((ord, pred)) => {
                        first_order.get_or_insert(ord);
                        spine.push(pred);
                        ctx.push(pred);
                    }
                    None => break,
                }
            }

            if let Some(ord) = first_order {
                // Accepted length = leading correct prefix (spec-decode stops
                // at the first reject).
                let mut acc = 0usize;
                for (k, &p) in spine.iter().enumerate() {
                    if i + k < n && p == tokens[i + k] {
                        acc += 1;
                    } else {
                        break;
                    }
                }
                mixed_steps += 1;
                mixed_accepted += acc as u64;
                mixed_drafted += spine.len() as u64;
                spine_hist[acc.min(max_spine)] += 1;
                let e = spine_by_order.entry(ord).or_insert((0, 0));
                e.0 += 1;
                e.1 += acc as u64;

                // Credit the serving gram so the budget curve is hit-weighted.
                if acc > 0 {
                    let o = first_order.unwrap() as usize;
                    let key = fingerprint(first_order.unwrap(), &tokens[i - o..i]);
                    if let Some(en) = table.get_mut(&key) {
                        en.hits_served += acc as u32;
                    }
                }
            }
        }

        // ── observe phase: fold position i into the table ──
        if scoring && freeze {
            continue;
        }
        for &ord in &orders {
            let o = ord as usize;
            if i >= o {
                let key = fingerprint(ord, &tokens[i - o..i]);
                {
                    let e = table.entry(key).or_default();
                    e.first_tok = tokens[i - o];
                    e.last_tok = tokens[i - 1];
                    e.observe(tokens[i], i as u32);
                }
            }
        }
    }

    // ── report ──
    println!("\n=== stream ===");
    println!("tokens: {n}   distinct grams: {}", table.len());

    println!("\n=== per-order precision (independent arms) ===");
    println!(
        "{:>6} {:>12} {:>10} {:>10}",
        "order", "proposals", "cover%", "precision%"
    );
    let mut ords: Vec<u8> = per_order.keys().copied().collect();
    ords.sort_by(|a, b| b.cmp(a));
    for ord in ords {
        let s = &per_order[&ord];
        println!(
            "{:>6} {:>12} {:>9.1}% {:>9.1}%",
            ord,
            s.proposals,
            100.0 * s.proposals as f64 / steps_total as f64,
            100.0 * s.correct as f64 / s.proposals.max(1) as f64,
        );
    }

    println!("\n=== mixed-order policy (longest-first, chained) ===");
    println!(
        "coverage: {:.1}%  ({mixed_steps} of {steps_total} steps proposed)",
        100.0 * mixed_steps as f64 / steps_total as f64
    );
    println!(
        "mean accepted tokens per proposing step: {:.2}",
        mixed_accepted as f64 / mixed_steps.max(1) as f64
    );
    println!(
        "mean accepted tokens per decode step:    {:.2}",
        mixed_accepted as f64 / steps_total as f64
    );
    println!(
        "mean drafted tokens per proposing step:  {:.2}",
        mixed_drafted as f64 / mixed_steps.max(1) as f64
    );
    println!(
        "verify efficiency (accepted/drafted):    {:.1}%",
        100.0 * mixed_accepted as f64 / mixed_drafted.max(1) as f64
    );
    println!("\naccepted-length histogram:");
    for (k, &c) in spine_hist.iter().enumerate() {
        if c > 0 {
            println!(
                "  {k:>2} tok: {c:>10}  {:>5.1}%",
                100.0 * c as f64 / mixed_steps.max(1) as f64
            );
        }
    }

    println!("\n=== multi-token yield by winning order ===");
    println!("{:>6} {:>12} {:>16}", "order", "steps", "mean accepted");
    let mut so: Vec<u8> = spine_by_order.keys().copied().collect();
    so.sort_by(|a, b| b.cmp(a));
    for ord in so {
        let (steps, acc) = spine_by_order[&ord];
        println!(
            "{:>6} {:>12} {:>16.2}",
            ord,
            steps,
            acc as f64 / steps.max(1) as f64
        );
    }

    println!("\n=== admission: precision by gram count at hit time ===");
    println!("{:>8} {:>12} {:>12}", "count", "proposals", "precision%");
    for (b, name) in COUNT_BUCKETS.iter().enumerate() {
        let s = &by_count[b];
        if s.proposals > 0 {
            println!(
                "{:>8} {:>12} {:>11.1}%",
                name,
                s.proposals,
                100.0 * s.correct as f64 / s.proposals as f64
            );
        }
    }

    println!("\n=== MRU: precision by gram age (positions since last seen) ===");
    println!("{:>8} {:>12} {:>12}", "age", "proposals", "precision%");
    for (b, name) in AGE_BUCKETS.iter().enumerate() {
        let s = &by_age[b];
        if s.proposals > 0 {
            println!(
                "{:>8} {:>12} {:>11.1}%",
                name,
                s.proposals,
                100.0 * s.correct as f64 / s.proposals as f64
            );
        }
    }

    println!("\n=== budget: hits retained keeping only top-K grams by count ===");
    let mut ranked: Vec<(u32, u32)> = table.values().map(|e| (e.total(), e.hits_served)).collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0));
    let total_hits: u64 = ranked.iter().map(|(_, h)| *h as u64).sum();
    println!(
        "{:>12} {:>10} {:>14}",
        "top-K grams", "% of table", "% of hits kept"
    );
    let mut cum = 0u64;
    let mut next_mark = 1000usize;
    for (idx, (_, h)) in ranked.iter().enumerate() {
        cum += *h as u64;
        if idx + 1 == next_mark || idx + 1 == ranked.len() {
            println!(
                "{:>12} {:>9.1}% {:>13.1}%",
                idx + 1,
                100.0 * (idx + 1) as f64 / ranked.len() as f64,
                100.0 * cum as f64 / total_hits.max(1) as f64
            );
            next_mark = next_mark.saturating_mul(10);
        }
    }

    if block_report {
        let n_blocks = file_bytes / block_bytes;
        let per_block = (block_bytes - 8) / rec_bytes; // 8B block header
        println!("\n=== block occupancy ===");
        println!(
            "file={} MB  block={} B  record={} B  -> {} blocks x {} records = {} slots",
            file_bytes >> 20,
            block_bytes,
            rec_bytes,
            n_blocks,
            per_block,
            n_blocks * per_block
        );
        for (label, keyed_last) in [("first-token keyed", false), ("last-token keyed", true)] {
            let mut per_key: HashMap<u32, u64> = HashMap::new();
            for e in table.values() {
                let k = if keyed_last { e.last_tok } else { e.first_tok };
                *per_key.entry(k).or_insert(0) += 1;
            }
            let total: u64 = per_key.values().sum();
            let mut counts: Vec<u64> = per_key.values().copied().collect();
            counts.sort_unstable_by(|a, b| b.cmp(a));
            // One block per key: how much spills?
            let fits: u64 = counts.iter().map(|c| (*c).min(per_block)).sum();
            let spill = total - fits;
            // Multi-block: blocks needed if a hot key may claim several.
            let need: u64 = counts.iter().map(|c| c.div_ceil(per_block)).sum();
            let occupied_keys = counts.len() as u64;
            println!("\n  -- {label} --");
            println!("    distinct keys in use : {occupied_keys}");
            println!(
                "    grams per key        : max={} p50={} p99={}",
                counts[0],
                counts[counts.len() / 2],
                counts[counts.len() / 100]
            );
            println!(
                "    1 block/key          : {:.1}% of grams fit, {spill} spill",
                100.0 * fits as f64 / total as f64
            );
            println!(
                "    blocks needed (multi): {need} of {n_blocks} ({:.1}% of file)",
                100.0 * need as f64 / n_blocks as f64
            );
            println!(
                "    mean fill            : {:.1}% ({:.0} of {per_block} records/block)",
                100.0 * total as f64 / (need * per_block) as f64,
                total as f64 / need as f64
            );
        }
    }

    // ── PLD baseline arm: the existing matcher, same stream ──
    let pld = hipfire_arch_qwen35::speculative::PldMatcher::default();
    let mut pld_steps = 0u64;
    let mut pld_acc = 0u64;
    // Sampled — PLD's lookup is an O(context) rescan, so a full sweep is O(n^2).
    let stride = (n / 5_000).max(1);
    // Real decode contexts are 4-32k, not the whole corpus; cap the rescan to
    // a realistic window so this arm measures context-local PLD, not an
    // unbounded corpus scan.
    const PLD_WINDOW: usize = 8192;
    let mut sampled = 0u64;
    for i in (256..n).step_by(stride) {
        sampled += 1;
        let lo = i.saturating_sub(PLD_WINDOW);
        if let Some(m) = pld.lookup(&tokens[lo..i]) {
            let mut acc = 0usize;
            for (k, &p) in m.tokens.iter().enumerate() {
                if i + k < n && p == tokens[i + k] {
                    acc += 1;
                } else {
                    break;
                }
            }
            pld_steps += 1;
            pld_acc += acc as u64;
        }
    }
    println!("\n=== PLD baseline (existing PldMatcher, {sampled} sampled steps) ===");
    println!(
        "coverage: {:.1}%   mean accepted per proposing step: {:.2}   per decode step: {:.2}",
        100.0 * pld_steps as f64 / sampled.max(1) as f64,
        pld_acc as f64 / pld_steps.max(1) as f64,
        pld_acc as f64 / sampled.max(1) as f64
    );
}

fn collect_rs(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let rd = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_rs(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The whole report rests on the oracle being right, so check it on a stream
/// whose answer is known by construction: a fixed 32-token cycle repeated. After
/// the first cycle every order >= 2 has seen each context exactly once, so a
/// table with min_count=1 must predict every subsequent token correctly and the
/// chain must run to max_spine with 100% acceptance.
fn self_check() {
    let period = 32usize;
    let reps = 50usize;
    let tokens: Vec<u32> = (0..period * reps).map(|i| (i % period) as u32).collect();
    let n = tokens.len();
    let orders: [u8; 4] = [5, 4, 3, 2];
    let max_order = 5usize;
    let max_spine = 8usize;

    let mut table: HashMap<u64, Entry> = HashMap::new();
    let mut checked = 0u64;
    let mut accepted_full = 0u64;

    for i in 0..n {
        // Only judge once the table has seen two full cycles, so every context
        // is populated and the cycle is unambiguous.
        if i >= 2 * period + max_order {
            let mut ctx: Vec<u32> = tokens[i - max_order..i].to_vec();
            let mut acc = 0usize;
            for _ in 0..max_spine {
                let mut hit = None;
                for &ord in &orders {
                    let o = ord as usize;
                    let key = fingerprint(ord, &ctx[ctx.len() - o..]);
                    if let Some(e) = table.get(&key) {
                        if let Some((pred, _)) = e.best() {
                            hit = Some(pred);
                            break;
                        }
                    }
                }
                match hit {
                    Some(p) => {
                        if i + acc < n && p == tokens[i + acc] {
                            acc += 1;
                            ctx.push(p);
                        } else {
                            break;
                        }
                    }
                    None => break,
                }
            }
            if i + max_spine < n {
                checked += 1;
                accepted_full += (acc == max_spine) as u64;
            }
        }
        for &ord in &orders {
            let o = ord as usize;
            if i >= o {
                let key = fingerprint(ord, &tokens[i - o..i]);
                {
                    let e = table.entry(key).or_default();
                    e.first_tok = tokens[i - o];
                    e.last_tok = tokens[i - 1];
                    e.observe(tokens[i], i as u32);
                }
            }
        }
    }

    assert!(
        checked > 1000,
        "self-check needs a real sample, got {checked}"
    );
    assert_eq!(
        accepted_full, checked,
        "oracle broken: on a perfectly periodic stream every chain must accept \
         all {max_spine} tokens, got {accepted_full}/{checked}"
    );

    // A stream with no structure must produce no useful chain: with a random
    // stream over a large vocab, contexts essentially never repeat.
    let mut rng = 0x12345678u64;
    let rand: Vec<u32> = (0..20000)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % 100_000) as u32
        })
        .collect();
    let mut t2: HashMap<u64, Entry> = HashMap::new();
    let mut correct = 0u64;
    let mut props = 0u64;
    for i in 0..rand.len() {
        if i >= max_order {
            let key = fingerprint(5, &rand[i - 5..i]);
            if let Some(e) = t2.get(&key) {
                if let Some((pred, _)) = e.best() {
                    props += 1;
                    correct += (pred == rand[i]) as u64;
                }
            }
        }
        for &ord in &orders {
            let o = ord as usize;
            if i >= o {
                let key = fingerprint(ord, &rand[i - o..i]);
                t2.entry(key).or_default().observe(rand[i], i as u32);
            }
        }
    }
    assert!(
        props < 50,
        "oracle broken: a random stream must almost never produce a 5-gram \
         proposal, got {props} (correct={correct}) — suggests fingerprint collisions"
    );

    println!("self-check OK");
    println!("  periodic stream: {accepted_full}/{checked} chains accepted all {max_spine} tokens");
    println!("  random stream:   {props} spurious 5-gram proposals in 20000 tokens");
}
