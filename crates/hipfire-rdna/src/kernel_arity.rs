// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Cross-check every raw-`params` kernel launch against the arity its `.hip`
//! source actually declares.
//!
//! HIP takes the argument count from the code object's metadata, **not** from
//! the length of the `kernelParams` array. A dispatch that passes too few
//! therefore makes `hipModuleLaunchKernel` read past the `Vec` and copy whatever
//! heap word follows into the missing kernarg; one that passes too many silently
//! binds the wrong values to the trailing parameters. Neither is a compile error
//! and neither needs a GPU to detect, which is what this test is for.
//!
//! It exists because that class has bitten twice:
//!   - `attention_dflash_wmma_f32` passed 10 to an 11-parameter kernel, so
//!     `is_causal` came from adjacent heap bytes — a nondeterministic causal mask
//!     on an attention that must be bidirectional.
//!   - `gemv_mq4g256` passed 7 to a 5-parameter kernel, binding `M` and `K` from
//!     the low halves of two sign POINTERS.
//! Both were "a change updated one call site and not its sibling".
//!
//! **Coverage is partial and deliberately so.** A function that names more than
//! one kernel, or builds more than one params vec, cannot be attributed
//! statically — the kernel is sometimes chosen at runtime — so those are SKIPPED,
//! not guessed at. The test asserts a floor on how many sites it actually checked
//! so the coverage cannot quietly rot to zero while the test keeps passing.

#![cfg(test)]

use std::collections::{HashMap, HashSet};

/// Strip `//` and `/* */` so commas inside comments never reach the counter.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let (mut out, mut i) = (String::with_capacity(src.len()), 0);
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// `name -> declared parameter count`, over every `__global__` in kernels/src.
/// A name may legitimately appear more than once (per-arch variants), so the
/// value is a set and a dispatch matching ANY of them is accepted.
fn kernel_arity() -> HashMap<String, HashSet<usize>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels/src");
    let mut out: HashMap<String, HashSet<usize>> = HashMap::new();
    let entries = std::fs::read_dir(dir).expect("kernels/src must be readable");
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|s| s.to_str()) != Some("hip") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let src = strip_comments(&raw);
        let mut from = 0;
        while let Some(g) = src[from..].find("__global__") {
            let after = from + g + "__global__".len();
            from = after;
            // `__global__ void NAME(` — tolerate attributes between the tokens.
            let Some(vpos) = src[after..].find(" void ") else {
                continue;
            };
            let ident_start = after + vpos + " void ".len();
            let Some(paren) = src[ident_start..].find('(') else {
                continue;
            };
            let name = src[ident_start..ident_start + paren].trim();
            if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                continue;
            }
            // Balance the parameter list, then count top-level commas.
            let (mut depth, mut j) = (1i32, ident_start + paren + 1);
            let start = j;
            let bytes = src.as_bytes();
            while depth > 0 && j < bytes.len() {
                match bytes[j] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            let params = src[start..j.saturating_sub(1)].trim();
            let count = if params.is_empty() {
                0
            } else {
                let mut n = 1usize;
                let mut d = 0i32;
                for c in params.chars() {
                    match c {
                        '(' | '<' | '[' => d += 1,
                        ')' | '>' | ']' => d -= 1,
                        ',' if d == 0 => n += 1,
                        _ => {}
                    }
                }
                n
            };
            out.entry(name.to_string()).or_default().insert(count);
        }
    }
    out
}

fn rust_sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_sources(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

fn quoted_after(line: &str, marker: &str) -> Option<String> {
    let rest = &line[line.find(marker)? + marker.len()..];
    let a = rest.find('"')? + 1;
    let b = rest[a..].find('"')? + a;
    Some(rest[a..b].to_string())
}

fn kernel_name_on(line: &str) -> Option<String> {
    quoted_after(line, "ensure_kernel(").or_else(|| quoted_after(line, "self.functions["))
}

fn is_fn_start(line: &str) -> bool {
    let indent = line.len() - line.trim_start().len();
    if indent > 8 {
        return false;
    }
    let t = line.trim_start();
    t.starts_with("fn ")
        || t.starts_with("pub fn ")
        || t.starts_with("pub(crate) fn ")
        || t.starts_with("unsafe fn ")
        || t.starts_with("pub unsafe fn ")
}

const PARAMS_DECL: &str = "params: Vec<*mut c_void>";

/// Count top-level comma-separated entries in `text`, ignoring nesting.
fn top_level_entries(text: &str) -> usize {
    let t = text.trim();
    if t.is_empty() {
        return 0;
    }
    let (mut n, mut depth) = (1usize, 0i32);
    for c in t.chars() {
        match c {
            '(' | '[' | '<' | '{' => depth += 1,
            ')' | ']' | '>' | '}' => depth -= 1,
            ',' if depth == 0 => n += 1,
            _ => {}
        }
    }
    // A trailing comma leaves an empty final entry.
    if t.ends_with(',') { n - 1 } else { n }
}

/// `self.launch_kernargs("NAME", ..., &kernargs![...])` — the kernel name is the
/// call's own first argument, so attribution here is exact rather than heuristic.
/// This is the majority path (478 sites) and is where a `bits`/`row_offset`
/// regression actually slipped through once.
fn scan_kernargs(text: &str, arity: &HashMap<String, HashSet<usize>>, file: &str,
                 checked: &mut usize, mismatches: &mut Vec<String>) {
    let mut from = 0usize;
    while let Some(rel) = text[from..].find("launch_kernargs(") {
        let call = from + rel;
        from = call + "launch_kernargs(".len();
        let rest = &text[from..];
        // The kernel name must be this call's own FIRST argument as a string
        // LITERAL. Many sites pass a variable (`tile_func_name`) chosen at
        // runtime; those are skipped, never guessed at — reaching for the next
        // quote anywhere downstream attributes a distant literal to this call.
        let head = rest.trim_start();
        if !head.starts_with('"') {
            continue;
        }
        let q1 = rest.len() - head.len();
        let Some(q2rel) = rest[q1 + 1..].find('"') else { continue };
        let name = &rest[q1 + 1..q1 + 1 + q2rel];
        let Some(karg) = rest.find("kernargs![") else { continue };
        // Only accept a kernargs! that belongs to THIS call, not a later one.
        if let Some(next_call) = rest.find("launch_kernargs(") {
            if next_call < karg {
                continue;
            }
        }
        let open = from + karg + "kernargs![".len();
        let (mut depth, mut j) = (1i32, open);
        let b = text.as_bytes();
        while depth > 0 && j < b.len() {
            match b[j] {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            j += 1;
        }
        let Some(declared) = arity.get(name) else { continue };
        let passed = top_level_entries(&text[open..j.saturating_sub(1)]);
        *checked += 1;
        if !declared.contains(&passed) {
            let line = text[..open].matches('\n').count() + 1;
            let mut want: Vec<_> = declared.iter().copied().collect();
            want.sort_unstable();
            mismatches.push(format!(
                "{file}:{line} launches `{name}` with {passed} args; the kernel declares {want:?}"
            ));
        }
    }
}

#[test]
fn raw_kernel_launches_match_their_hip_arity() {
    let arity = kernel_arity();
    assert!(
        arity.len() > 300,
        "parsed only {} kernels — the .hip scanner is broken, not the dispatches",
        arity.len()
    );

    let mut files = Vec::new();
    rust_sources(
        std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src")),
        &mut files,
    );
    files.sort();

    let (mut checked, mut mismatches) = (0usize, Vec::new());
    for path in files {
        // Skip this file: its own doc comments contain example launch snippets.
        if path.file_name().and_then(|s| s.to_str()) == Some("kernel_arity.rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let fname = path.file_name().unwrap().to_string_lossy().to_string();
        scan_kernargs(&text, &arity, &fname, &mut checked, &mut mismatches);
        let lines: Vec<&str> = text.lines().collect();

        // Function spans, so a kernel name never leaks across a `fn` boundary.
        let mut bounds: Vec<usize> = (0..lines.len()).filter(|&i| is_fn_start(lines[i])).collect();
        bounds.push(lines.len());
        let mut ambiguous = vec![false; lines.len()];
        for w in bounds.windows(2) {
            let (a, b) = (w[0], w[1]);
            let names: HashSet<String> = lines[a..b].iter().filter_map(|l| kernel_name_on(l)).collect();
            let vecs = lines[a..b].iter().filter(|l| l.contains(PARAMS_DECL)).count();
            // More than one kernel, or more than one params vec: which feeds which
            // is not statically decidable (the kernel may be picked at runtime).
            if names.len() > 1 || vecs > 1 {
                ambiguous[a..b].iter_mut().for_each(|f| *f = true);
            }
        }

        let mut current: Option<String> = None;
        for (i, line) in lines.iter().enumerate() {
            if is_fn_start(line) {
                current = None;
            }
            if let Some(n) = kernel_name_on(line) {
                current = Some(n);
            }
            if !line.contains(PARAMS_DECL) || ambiguous[i] {
                continue;
            }
            let Some(name) = current.as_ref().filter(|n| arity.contains_key(*n)) else {
                continue;
            };
            let mut passed = 0usize;
            for l in &lines[i..] {
                passed += l.matches("as *mut c_void,").count();
                if l.contains("];") {
                    break;
                }
            }
            checked += 1;
            let declared = &arity[name];
            if !declared.contains(&passed) {
                let mut want: Vec<_> = declared.iter().copied().collect();
                want.sort_unstable();
                mismatches.push(format!(
                    "{}:{} launches `{name}` with {passed} args; the kernel declares {want:?}",
                    path.file_name().unwrap().to_string_lossy(),
                    i + 1
                ));
            }
        }
    }

    // A silent drop to zero checked sites would make this test pass while
    // measuring nothing — the exact failure mode tests/no-gpu-ci.sh documents.
    assert!(
        checked >= 500,
        "only {checked} launch sites were attributable (expected >=500); the \
         scanner or the dispatch style changed, so this test is no longer covering \
         what it claims"
    );
    assert!(
        mismatches.is_empty(),
        "kernel argument-count mismatches ({} site(s)):\n  {}",
        mismatches.len(),
        mismatches.join("\n  ")
    );
}
