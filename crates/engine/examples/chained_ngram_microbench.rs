//! Microbench: ChainedNgramCache vs NgramCache prediction accuracy on
//! real DFlash-emitted token streams. Pure CPU, no GPU/HIP dependency.
//!
//! Methodology: load a token stream (one int per line), train each
//! cache on the first half, then walk the second half measuring how
//! often each cache correctly predicts the next token from its rolling
//! context tail.
//!
//! This isolates "does higher-order chained lookup beat bigram lookup
//! on real outputs?" before committing to engine plumbing.
//!
//! Usage: chained_ngram_microbench <tokens.txt> [n=4]

use engine::speculative::{ChainedNgramCache, NgramCache};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("path to tokens.txt");
    let n: usize = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let content = std::fs::read_to_string(path).expect("read tokens");
    let tokens: Vec<u32> = content
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|s| s.parse().ok())
        .collect();
    println!("loaded {} tokens from {}", tokens.len(), path);
    println!("training cache on first half ({}), evaluating on second half ({}), n={}",
             tokens.len() / 2, tokens.len() - tokens.len() / 2, n);

    let split = tokens.len() / 2;
    let train = &tokens[..split];
    let test  = &tokens[split..];

    // Train both caches.
    let mut bigram = NgramCache::new(1); // min_count=1: any seen transition is a candidate
    bigram.observe_many(train);
    let mut chained = ChainedNgramCache::new(n, 1 << 20); // 1M entries
    chained.observe_many(train);

    println!("bigram: {} keys", bigram.bigram.len());
    println!("chained: used={} cap={} ({:.2}% occupancy)",
             chained.used(), chained.capacity(), chained.occupancy() * 100.0);

    // Walk test set, predict each position, compare to actual.
    let mut bigram_hits = 0usize;
    let mut bigram_attempts = 0usize;
    let mut chained_hits = 0usize;
    let mut chained_attempts = 0usize;
    let mut chain_lengths: Vec<usize> = Vec::new();
    let mut chain_hit_lengths: Vec<usize> = Vec::new();

    let want = n - 1;
    for i in 1..test.len() {
        let actual = test[i];

        // Bigram prediction: needs (a, b) = test[i-2], test[i-1]
        if i >= 2 {
            bigram_attempts += 1;
            if let Some((pred, _)) = bigram.predict(test[i - 2], test[i - 1]) {
                if pred == actual { bigram_hits += 1; }
            }
        }

        // Chained prediction: needs last (n-1) tokens
        if i >= want {
            chained_attempts += 1;
            let key = &test[i - want..i];
            if let Some(pred) = chained.get(key) {
                if pred == actual { chained_hits += 1; }
            }
        }

        // At every n-th position, predict a CHAIN starting from i and
        // see how many sequential tokens match.
        if i % 16 == 0 && i >= want && i + 16 < test.len() {
            let chain = chained.predict_chain(&test[i - want..i], 16);
            chain_lengths.push(chain.len());
            let mut hits = 0;
            for (j, &c) in chain.iter().enumerate() {
                if i + j < test.len() && c == test[i + j] {
                    hits += 1;
                } else {
                    break; // chain hit-rate measured up to first miss
                }
            }
            chain_hit_lengths.push(hits);
        }
    }

    println!("\n=== prediction accuracy ===");
    let b_acc = bigram_hits as f32 / bigram_attempts.max(1) as f32;
    let c_acc = chained_hits as f32 / chained_attempts.max(1) as f32;
    println!("  bigram:   {}/{} = {:.3} ({:.1}%)", bigram_hits, bigram_attempts, b_acc, b_acc * 100.0);
    println!("  chained:  {}/{} = {:.3} ({:.1}%)", chained_hits, chained_attempts, c_acc, c_acc * 100.0);
    println!("  Δ:        {:+.1}%", (c_acc - b_acc) * 100.0);

    println!("\n=== chained-mode chain prediction ===");
    if !chain_lengths.is_empty() {
        let mean_chain: f32 = chain_lengths.iter().sum::<usize>() as f32 / chain_lengths.len() as f32;
        let mean_hits: f32 = chain_hit_lengths.iter().sum::<usize>() as f32 / chain_hit_lengths.len() as f32;
        let total_chain_tokens: usize = chain_lengths.iter().sum();
        let total_chain_hits: usize = chain_hit_lengths.iter().sum();
        let chain_acc = total_chain_hits as f32 / total_chain_tokens.max(1) as f32;
        println!("  samples: {}  mean chain len: {:.2}  mean hit len: {:.2}",
                 chain_lengths.len(), mean_chain, mean_hits);
        println!("  total chain tok-acc: {}/{} = {:.3} ({:.1}%)",
                 total_chain_hits, total_chain_tokens, chain_acc, chain_acc * 100.0);
        // Histogram
        let mut hits_hist = [0usize; 17];
        for &h in &chain_hit_lengths {
            hits_hist[h.min(16)] += 1;
        }
        println!("  hit-length histogram (up-to-first-miss):");
        for (k, &v) in hits_hist.iter().enumerate() {
            if v > 0 {
                println!("    {:2} hits → {:4} samples", k, v);
            }
        }
    }
}
