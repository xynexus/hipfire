// hipfire example clippy sweep: examples are GPU probes/benches, not reusable APIs.
// SPDX-License-Identifier: Apache-2.0
// hipfire — measure slab-backed expert residency against one allocation per module.
//
// `gtt_alloc_cost`'s >2 MiB regime is a FLAT 2 MiB granule, so the waste is bounded
// per ALLOCATION and amortises as `2 MiB / (N * size)`. That is arithmetic; this
// spends real GTT to check it, because the whole grouped-allocation plan for
// Qwen3.8-Flash-Next rests on it.
//
// A Qwen3.8-Flash-Next routed expert at 4-bit is gate_up (Oq4G256 over K=hidden
// 2560) + down (Oq4G128 over K=mi 640) = 2,560,000 B, which lands at 2.441 MiB —
// a hair over the 2 MiB line, so it pays for 4 MiB. The model holds 512 experts x
// 49 MoE layers = 25,088 of them.
//
//   cargo run --release -p hipfire-rdna --example gtt_slab_vs_permodule [modules]
//
// Prints the measured per-module cost of each strategy and what it extrapolates to
// across the full table. Exit code is nonzero if slabbing does not actually win, so
// this can gate the change rather than merely inform it.

fn gtt_used() -> u64 {
    std::fs::read_to_string("/sys/class/drm/card1/device/mem_info_gtt_used")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0)
}

/// One routed expert at 4-bit, for this model's geometry.
const MODULE: usize = 2_560_000;
/// 512 experts x (48 text + 1 MTP) MoE layers.
const TABLE: usize = 512 * 49;
/// Experts per slab. N=4 is where the tax flattens; larger slabs do not improve it
/// and only coarsen the eviction unit.
const PER_SLAB: usize = 4;

fn main() {
    let modules: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256);
    let mut gpu = hipfire_rdna::Gpu::init().expect("gpu");
    let _warm = gpu
        .alloc_tensor(&[1024], hipfire_rdna::DType::F32)
        .expect("warm");

    // ── one allocation per module (what the pager does today) ────────────────
    let before = gtt_used();
    let mut per_module = Vec::with_capacity(modules);
    for _ in 0..modules {
        per_module.push(
            gpu.alloc_tensor(&[MODULE / 4], hipfire_rdna::DType::F32)
                .expect("alloc module"),
        );
    }
    let single = (gtt_used() - before) as f64 / modules as f64;
    drop(per_module);

    // ── one allocation per PER_SLAB modules, sub-allocated ───────────────────
    // Only the allocation is grouped. A real implementation hands out non-owning
    // views into the slab (`BufferOrigin::NonOwning`), so the residency unit stays
    // one expert and the eviction granularity does not change — this measures the
    // allocation half, which is where the tax lives.
    let n_slabs = modules.div_ceil(PER_SLAB);
    let before = gtt_used();
    let mut slabs = Vec::with_capacity(n_slabs);
    for _ in 0..n_slabs {
        slabs.push(
            gpu.alloc_tensor(&[MODULE * PER_SLAB / 4], hipfire_rdna::DType::F32)
                .expect("alloc slab"),
        );
    }
    let slabbed = (gtt_used() - before) as f64 / (n_slabs * PER_SLAB) as f64;
    drop(slabs);

    let gib = |b: f64| b / (1u64 << 30) as f64;
    println!(
        "routed expert module = {MODULE} B ({:.3} MiB)",
        MODULE as f64 / (1 << 20) as f64
    );
    println!("measured over {modules} modules, {PER_SLAB} per slab:\n");
    println!(
        "  per-module alloc : {single:>10.0} B/module  ({:.3}x)",
        single / MODULE as f64
    );
    println!(
        "  slab-backed      : {slabbed:>10.0} B/module  ({:.3}x)",
        slabbed / MODULE as f64
    );
    println!(
        "\nextrapolated across {TABLE} modules (512 experts x 49 layers):\n  \
         raw        {:>7.1} GB\n  per-module {:>7.1} GB\n  slab       {:>7.1} GB\n  saving     {:>7.1} GB",
        gib(TABLE as f64 * MODULE as f64) * 1.073741824,
        gib(TABLE as f64 * single) * 1.073741824,
        gib(TABLE as f64 * slabbed) * 1.073741824,
        gib(TABLE as f64 * (single - slabbed)) * 1.073741824,
    );

    // The claim under test is that slabbing is materially cheaper, not merely not
    // worse — anything under a few percent means the premise is wrong.
    let win = (single - slabbed) / single;
    println!("\nslab saves {:.1}% of expert GTT", win * 100.0);
    if win < 0.05 {
        eprintln!("gtt_slab_vs_permodule: FAILED — slabbing must win materially, got {win:.3}");
        std::process::exit(1);
    }
    println!("gtt_slab_vs_permodule: OK");
}
