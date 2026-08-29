// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Drive the real [`NgramSpec`] over a recorded token stream and report what it
//! would have accepted. Same acceptance oracle as
//! `hipfire-arch-qwen35/examples/ngram_replay_sweep.rs` — the stream is its own
//! ground truth — but exercising the shipped crate rather than a model of it,
//! so the two are expected to agree.
//!
//!   replay_real --tokenizer <qwen.hfq> [--rs-dir crates] [--corpus f.jsonl]
//!               [--max-tokens N] [--cold /path/store.hng] [--blocks N]

use std::path::Path;

use hipfire_model::tokenizer::Tokenizer;
use hipfire_runtime::hfq::HfqFile;
use hipfire_specdecode_ngram::{NgramConfig, NgramSpec, Tier, WriteTarget};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut tokenizer = String::new();
    let mut rs_dir = String::new();
    let mut corpus = String::new();
    let mut field = "text".to_string();
    let mut max_tokens = 1_000_000usize;
    let mut user: Option<String> = None;
    let mut topic: Option<String> = None;
    let mut base: Option<String> = None;
    let mut topic_writable = false;
    let mut blocks = 262_144usize;
    let mut chain_floor = 8u8;
    let mut merge_every = 0usize;
    let mut hot_cap = 1usize << 20;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tokenizer" => {
                tokenizer = args[i + 1].clone();
                i += 2;
            }
            "--rs-dir" => {
                rs_dir = args[i + 1].clone();
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
            "--max-tokens" => {
                max_tokens = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--user" => {
                user = Some(args[i + 1].clone());
                i += 2;
            }
            "--topic" => {
                topic = Some(args[i + 1].clone());
                i += 2;
            }
            "--base" => {
                base = Some(args[i + 1].clone());
                i += 2;
            }
            "--topic-writable" => {
                topic_writable = true;
                i += 1;
            }
            "--blocks" => {
                blocks = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--chain-floor" => {
                chain_floor = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--hot-cap" => {
                hot_cap = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--merge-every" => {
                merge_every = args[i + 1].parse().unwrap();
                i += 2;
            }
            o => panic!("unknown arg {o:?}"),
        }
    }

    let hfq = HfqFile::open_index_only(Path::new(&tokenizer)).expect("open hfq");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");

    let mut text = String::new();
    if !rs_dir.is_empty() {
        let mut files = Vec::new();
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
        for line in raw.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(s) = v.get(&field).and_then(|x| x.as_str()) {
                    text.push_str(s);
                    text.push('\n');
                }
            }
            if text.len() > max_tokens * 8 {
                break;
            }
        }
    }

    let mut tokens = tok.encode(&text);
    tokens.truncate(max_tokens);
    let n = tokens.len();
    assert!(n > 1000, "need a real stream, got {n}");

    let write_target = if user.is_some() {
        WriteTarget::User
    } else if topic.is_some() && topic_writable {
        WriteTarget::Topic
    } else {
        WriteTarget::None
    };
    let mut ng = NgramSpec::new(NgramConfig {
        chain_floor,
        hot_capacity: hot_cap,
        write_target,
        ..Default::default()
    });
    if let Some(p) = &user {
        ng.attach_user(Path::new(p), 200_000, blocks)
            .expect("attach user");
        eprintln!(
            "[replay] user store {p} ({blocks} blocks = {} MB)",
            blocks * 4096 / 1_048_576
        );
    }
    if let Some(p) = &topic {
        ng.attach_topic(Path::new(p), 200_000, blocks, topic_writable)
            .expect("attach topic");
        eprintln!("[replay] topic store {p} (writable={topic_writable})");
    }
    if let Some(p) = &base {
        ng.attach_base(Path::new(p)).expect("attach base");
        eprintln!("[replay] base store {p} (read-only)");
    }
    eprintln!("[replay] {n} tokens, chain_floor={chain_floor}, hot_cap={hot_cap}");

    let t0 = std::time::Instant::now();
    for i in 0..n {
        let acc = match ng.draft() {
            Some(spine) => {
                let mut a = 0usize;
                for (k, &p) in spine.iter().enumerate() {
                    if i + k < n && p == tokens[i + k] {
                        a += 1;
                    } else {
                        break;
                    }
                }
                a
            }
            None => 0,
        };
        ng.record_acceptance(acc);
        ng.observe(&tokens[i..i + 1]);
        if merge_every > 0 && i > 0 && i % merge_every == 0 {
            ng.merge().expect("merge");
        }
    }
    let el = t0.elapsed();

    ng.merge().expect("final merge");
    let s = ng.stats();
    println!(
        "\n=== NgramSpec over {n} tokens ({:.1}s, {:.0} tok/s CPU) ===",
        el.as_secs_f64(),
        n as f64 / el.as_secs_f64()
    );
    println!("coverage                : {:.1}%", 100.0 * s.coverage());
    println!("accepted / decode step  : {:.2}", s.accepted_per_step());
    println!(
        "drafted  / proposing    : {:.2}",
        s.drafted as f64 / s.steps_proposed.max(1) as f64
    );
    println!(
        "verify efficiency       : {:.1}%",
        100.0 * s.verify_efficiency()
    );
    println!("hot table entries       : {}", ng.hot_len());
    println!(
        "\n--- tier attribution (first hit wins, so each row is that tier's marginal value) ---"
    );
    println!(
        "{:<7} {:>12} {:>12} {:>10} {:>12} {:>10}",
        "tier", "lookups", "hits", "drafted", "accepted", "marginal"
    );
    for t in Tier::ALL {
        let li = s.lookups_by_tier[t.idx()];
        let hi = s.hits_by_tier[t.idx()];
        if li == 0 && s.drafted_in(t) == 0 {
            continue;
        }
        println!(
            "{:<7} {:>12} {:>12} {:>10} {:>12} {:>9.2}%",
            t.name(),
            li,
            hi,
            s.drafted_in(t),
            s.accepted_in(t),
            100.0 * s.marginal_share(t)
        );
    }
    for t in [Tier::User, Tier::Topic, Tier::Base] {
        if let Some(c) = ng.store(t) {
            let (recs, blks) = c.occupancy();
            println!(
                "{:<7} occupancy: {recs} records in {blks}/{} blocks{}",
                t.name(),
                c.n_blocks(),
                if c.is_read_only() { " (read-only)" } else { "" }
            );
        }
    }
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
