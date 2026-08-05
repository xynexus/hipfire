# Instrument audit: `handlers/diag.rs`

A written pass over every instrument in the daemon's diagnostic handlers,
against the failure class in `~/measurement-integrity-goal.md`: **a name that
promises one measurement while the body performs another**.

Method: read each handler's body against the label its output carries, not
against its function name or doc comment. Both of those were accurate for
`bench_prefill`; the *output* was what lied.

| instrument | output label | what it does | verdict |
|---|---|---|---|
| `diag()` | `diag` | reads VRAM, HIP version, loaded-model arch and reports them | **honest** — reports state, claims no measurement |
| `bench_prefill()` | `pp<N> t/s` | qwen3.5/3.6: real `forward_prefill_batch`. Every other arch: per-token decode/warm-pass loop | **was lying; now discloses** (`37a12508b`) |
| `profile()` | kernel profile | precompiles a HARDCODED matrix, then reports the compiled kernel set | **suspect — see below** |

## `profile()` reports a set it created itself

```rust
for kv in &["q8"] {
    for wq in &["hfq4", "hfq6", "q8"] {
        for hd in &[128usize, 256] {
            let _ = daemon_state.gpu.precompile_qwen35(wq, kv, *hd);
```

Then it reports `gpu.profile()`. So the kernel list is substantially a function
of this loop, not of what the machine or the loaded model actually uses.

Two concrete gaps:

* **`kv = ["q8"]` only.** The shipped default is `kvarn4`
  (`~/.hipfire/config.json`), so the KV format most models run is never
  precompiled and its kernels will be absent or under-represented.
* **`wq` covers `hfq4`/`hfq6`/`q8`** — no `oq4`, `mq4`, `mq3`, `qtip3`,
  `mfp4`, or the Lloyd variants, several of which are what production artifacts
  use.

This is not the same severity as `bench_prefill` — nothing here reports a
throughput number that means something else. But anyone reading `profile` output
as "the kernels this system uses" is reading a hardcoded sweep, and the sweep
does not include the default KV format. Worth either widening to the formats
`model-support.toml` admits, or renaming what it emits so the sweep is visible.

**Not verified:** whether `gpu.profile()` reports only compiled kernels or
enumerates a static registry. If the latter, the precompile loop is
inconsequential and this entry downgrades to cosmetic. That check is one read of
`Gpu::profile` and was not done here — flagged rather than assumed.

## Other hunt sites, already covered

| site | finding | state |
|---|---|---|
| `tiny-quant-gate` cells | `minimax/kld:mq4` = exactly 0.000000; `qwen3_5_moe` mq6 and mq4 bit-identical across different bit widths | 3 vacuous cells, **awaiting decision** (fix or delete) |
| `verify_la_core_vs_kernels`, `verify_deltanet_vs_kernel` | hardcoded 2 heads / h=64 / `n_k == n_v` — passed at a size no model runs, GQA never exercised | **fixed** — geometry from argv, real shapes in `tests/linear-attn-verify.sh` |
| `hipfire inspect` quant histogram | reported the logical view; a 379 MB compressed tensor showed as uncompressed, "saved 55.73 KB" against a real 145 MB | **fixed** — lossless-storage section, per-tensor annotation, JSON fields |
| `gamma_hybrid` verdict | printed "predicts text" on synthetic tokens after its own preamble said the loss was not meaningful | **fixed** — verdict branches on real vs synthetic |

## Standing rule this pass confirms

Every instance found this session was invisible from the function name and the
doc comment, and visible only from the body. `bench_prefill`'s comment said
"warm-pass" in plain English three lines above the code that printed
`pp512 t/s`. **Audit the output label against the body, not against the
documentation** — the documentation was right every time and did not help.
