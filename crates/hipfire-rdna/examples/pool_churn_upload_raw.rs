// M1a exit measurement for docs/plans/2026-08-09-v2-daemon-module-major-multistream.md.
//
// `Gpu::upload_raw` allocates straight from HIP while `Gpu::free_tensor` frees
// into `GpuPool`. For a load-once/free-at-unload caller that asymmetry is
// invisible. For anything that CHURNS — the weight pager cycling expert modules
// through a bounded VRAM budget — every cold load allocates fresh while every
// eviction accumulates in a free-list the next load cannot reach, so VRAM grows
// in proportion to paging traffic.
//
// This drives both paths over the same alloc/free cycle and reports the pool
// counters plus real VRAM, so the difference is measured rather than argued.
//
// Non-daemon GPU binary: coordinate with `hipfire lock` (AGENTS.md).
//
//   cargo run --release -p hipfire-rdna --example pool_churn_upload_raw

use hipfire_rdna::Gpu;

/// One routed expert of a 35B-A3B-class MQ4 artifact.
const MODULE_BYTES: usize = 1_667_235;
const CYCLES: usize = 4_000;

fn vram_in_use(gpu: &Gpu) -> u64 {
    match gpu.hip.get_vram_info() {
        Ok((free, total)) => total.saturating_sub(free) as u64,
        Err(_) => 0,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut gpu = Gpu::init()?;
    let payload = vec![0u8; MODULE_BYTES];
    let mut failures = Vec::new();

    // ── pooled: what the pager transports now use ────────────────────────────
    // Settle first — the very first cycles legitimately allocate, and the claim
    // is about steady state, not about never calling HIP at all.
    for _ in 0..16 {
        let t = gpu.upload_raw_pooled(&payload, &[MODULE_BYTES])?;
        gpu.free_tensor(t)?;
    }
    let (new_before, reused_before) = gpu.pool_counters();
    let vram_before = vram_in_use(&gpu);
    for _ in 0..CYCLES {
        let t = gpu.upload_raw_pooled(&payload, &[MODULE_BYTES])?;
        gpu.free_tensor(t)?;
    }
    let (new_after, reused_after) = gpu.pool_counters();
    let vram_after = vram_in_use(&gpu);

    let pooled_new = new_after - new_before;
    let pooled_reused = reused_after - reused_before;
    let pooled_growth = vram_after.saturating_sub(vram_before);
    println!("pooled  ({CYCLES} cycles x {MODULE_BYTES} B):");
    println!("  pool total_new    += {pooled_new}");
    println!("  pool total_reused += {pooled_reused}");
    println!("  vram in use: {vram_before} -> {vram_after} (+{pooled_growth} B)");

    if pooled_new != 0 {
        failures.push(format!(
            "pooled path called HIP {pooled_new} times in steady state; \
             alloc and free are not meeting in the pool"
        ));
    }
    if pooled_reused != CYCLES {
        failures.push(format!(
            "pooled path reused {pooled_reused} of {CYCLES} buffers"
        ));
    }
    // One module's worth of slack: the driver's own bookkeeping moves a little.
    if pooled_growth > MODULE_BYTES as u64 {
        failures.push(format!(
            "pooled path grew VRAM by {pooled_growth} B over {CYCLES} cycles"
        ));
    }

    // ── unpooled: the pre-fix behaviour, for contrast ────────────────────────
    // Deliberately a shorter run. Each cycle strands a buffer in the pool's
    // free-list that no `upload_raw` can reach, so this leaks by construction
    // and there is no reason to leak 4,000 modules to prove it.
    const CONTRAST_CYCLES: usize = 200;
    let vram_before_raw = vram_in_use(&gpu);
    for _ in 0..CONTRAST_CYCLES {
        let t = gpu.upload_raw(&payload, &[MODULE_BYTES])?;
        gpu.free_tensor(t)?;
    }
    let vram_after_raw = vram_in_use(&gpu);
    let raw_growth = vram_after_raw.saturating_sub(vram_before_raw);
    println!("\nunpooled ({CONTRAST_CYCLES} cycles, pre-fix behaviour):");
    println!("  vram in use: {vram_before_raw} -> {vram_after_raw} (+{raw_growth} B)");
    println!(
        "  = {:.1} MiB stranded, {:.1} modules' worth",
        raw_growth as f64 / (1024.0 * 1024.0),
        raw_growth as f64 / MODULE_BYTES as f64
    );

    println!();
    if failures.is_empty() {
        println!("M1a PASS: pooled churn is allocation-free and VRAM-flat");
        Ok(())
    } else {
        for f in &failures {
            println!("FAIL: {f}");
        }
        Err("M1a measurement failed".into())
    }
}
