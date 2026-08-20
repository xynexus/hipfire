// Find overlapping tensor byte ranges in an HFQ, mirroring the compose check
// (sort by data_offset, walk a cursor, flag any entry starting before it).
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("usage: overlap <file.hfq>");
    let hfq = HfqFile::open(Path::new(&path)).expect("open");
    let mut v: Vec<(u64, u64, String, u8)> = hfq
        .tensors()
        .iter()
        .map(|t| {
            (
                t.data_offset as u64,
                t.data_size as u64,
                t.name.clone(),
                t.quant_type,
            )
        })
        .collect();
    v.sort_by_key(|e| e.0);
    println!("tensors: {}", v.len());
    let mut cursor = 0u64;
    let mut prev = String::from("<start>");
    let mut bad = 0usize;
    for (off, len, name, qt) in &v {
        if *off < cursor {
            bad += 1;
            if bad <= 20 {
                println!(
                    "OVERLAP: {name} (qt {qt}) starts {off} but previous ({prev}) ends {cursor}  [back {} bytes]",
                    cursor - off
                );
            }
        }
        cursor = cursor.max(off + len);
        prev = name.clone();
    }
    println!("overlapping entries: {bad}");
    // Also report duplicate names, a common cause of a doubled index entry.
    let mut names: Vec<&String> = v.iter().map(|e| &e.2).collect();
    names.sort();
    let total = names.len();
    names.dedup();
    println!("duplicate names: {}", total - names.len());
}
