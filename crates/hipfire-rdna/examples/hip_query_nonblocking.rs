// Proves `hipEventQuery` / `hipStreamQuery` are wired and actually NON-BLOCKING.
//
// A hard prerequisite in docs/plans/2026-08-09-v2-daemon-module-major-multistream.md
// §1.6: without a non-blocking query the residency prefetch cannot poll, and
// `event_synchronize` would serialise the very pipeline the prefetch exists to
// overlap.
//
// The test that matters is not "the call returns Ok" — a blocking call would
// also do that, and would pass a test that only checked completion. It is
// "`Ok(false)` is observed while work is outstanding", which only a non-blocking
// query can produce.
//
// Non-daemon GPU binary: coordinate with `hipfire lock` (AGENTS.md).
//
//   cargo run --release -p hipfire-rdna --example hip_query_nonblocking

use hipfire_rdna::Gpu;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    let hip = &gpu.hip;
    let mut failures: Vec<String> = Vec::new();

    // ── an idle stream/event must report complete ────────────────────────────
    let stream = hip.stream_create()?;
    let event = hip.event_create()?;
    hip.event_record(&event, Some(&stream))?;
    hip.stream_synchronize(&stream)?;

    match hip.event_query(&event) {
        Ok(true) => println!("idle event_query      -> true  (complete, as expected)"),
        Ok(false) => failures.push("event_query said NOT ready after a synchronize".into()),
        Err(e) => failures.push(format!("event_query errored on a completed event: {e}")),
    }
    match hip.stream_query(&stream) {
        Ok(true) => println!("idle stream_query     -> true  (drained, as expected)"),
        Ok(false) => failures.push("stream_query said NOT ready after a synchronize".into()),
        Err(e) => failures.push(format!("stream_query errored on a drained stream: {e}")),
    }

    // ── the real test: outstanding work must be observable as NOT ready ──────
    // Enough D2D traffic to stay in flight past the query. Sized so the copy
    // takes long enough to observe, not so long the example is slow.
    const BYTES: usize = 256 * 1024 * 1024;
    let src = hip.malloc(BYTES)?;
    let dst = hip.malloc(BYTES)?;

    let mut saw_pending = false;
    let mut polls_total = 0u64;
    for attempt in 0..8 {
        for _ in 0..4 {
            hip.memcpy_dtod_async_at(&dst, 0, &src, 0, BYTES, &stream)?;
        }
        hip.event_record(&event, Some(&stream))?;

        // Poll. A blocking implementation returns only once, already true.
        let mut polls = 0u64;
        loop {
            match hip.event_query(&event) {
                Ok(true) => break,
                Ok(false) => {
                    saw_pending = true;
                    polls += 1;
                    if polls > 50_000_000 {
                        failures.push("event_query never completed".into());
                        break;
                    }
                }
                Err(e) => {
                    failures.push(format!("event_query errored while polling: {e}"));
                    break;
                }
            }
        }
        polls_total += polls;
        if saw_pending {
            println!(
                "busy  event_query     -> false x{polls} then true  (attempt {}, NON-BLOCKING confirmed)",
                attempt + 1
            );
            break;
        }
    }
    hip.stream_synchronize(&stream)?;

    if !saw_pending {
        failures.push(format!(
            "never observed Ok(false) in {polls_total} polls across 8 attempts — the query \
             returned only after completion, which is what a BLOCKING call looks like. Either \
             the copies finished faster than one query round trip, or the binding is wrong."
        ));
    }

    let _ = hip.free(src);
    let _ = hip.free(dst);
    let _ = hip.event_destroy(event);
    let _ = hip.stream_destroy(stream);
    let _ = &mut gpu;

    if failures.is_empty() {
        println!("\nPASS: hipEventQuery / hipStreamQuery are wired and non-blocking");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAIL: {f}");
        }
        std::process::exit(1);
    }
}
