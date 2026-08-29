# Executable plans for the three spec-decode bugs still open

Master `0c9e3d252`, branch `fix/bug-hunt-2026-08-29`. Each plan below was produced
by reading the code and then **adversarially checked** by a second pass against the
same source.

All three are `implementable-with-gpu-validation`: the structure can be written and
compile-checked here, but the numeric behaviour needs hardware and an artifact this
box does not have. They are recorded rather than half-applied — each is a silent
corruption path where a fix that *looks* right is worse than the documented bug.

**Read the "why the obvious fix is wrong" notes first.** In all three cases the
approach suggested by the original bug report is wrong.


---

## MTP hidden-row / position desync

**Verdict:** `implementable-with-gpu-validation`

### What the change is

`spec_step_dflash_mtp_tree` keeps `accept_dflash + 1` hidden rows while committing (and advancing `position` by) `accept_dflash + accept_mtp + 1` tokens. Because `draft_scratch.target_hidden` is indexed by ABSOLUTE position (`dst_row_offset = position`) and `target_hidden_abs_positions` is a parallel list that `draft_forward`'s `positions_k` is built from, every MTP-accepted cycle leaves a permanent hole: after the first one, row `i` of `target_hidden` is no longer described by `positions_k[i]`, so the drafter RoPEs its context rows with the wrong phases (not merely "short by accept_mtp"). The naive `rows_to_keep = accept_dflash + accept_mtp + 1` is wrong because the block scatter copies a contiguous run from the verify block, and slot `accept_dflash + 1` is the NEXT (rejected) dflash candidate, not the accepted MTP child, which lives at `b + (s-1)*mtp_k + k`. Fix: gather by slot. `spec_step_ddtree_batched` already does the equivalent — but on the HOST vec (`speculative.rs:12312-12319`, `src_row = accepted_node_indices[i] + 1`), not on the GPU tensor, so it cannot be called; the reusable piece is the ring-wrap math in `scatter_hidden_block_to_interleaved`, which I generalize into a slot-indexed sibling and have the block form delegate to. The committed slot list is already computed: `[0] ++ accepted_slots`. `accept_mtp` is at most 1 today (MTP slots are leaves in the greedy walk), but the fix is agnostic to that.

### Why the obvious fix is wrong

The likely-wrong fix is exactly the one in the report's first sentence: `rows_to_keep = accept_dflash + accept_mtp + 1` left on the CONTIGUOUS `scatter_hidden_block_to_interleaved`. That copies tree slot `accept_dflash + 1`, which is the next dflash candidate — the token the target just REJECTED — into the committed position. It never crashes, never trips an assert, and the row is a plausible-looking hidden state, so the drafter is fed a wrong-branch context and acceptance quietly degrades; it would look like "the fix didn't help much". Worst variant: at `accept_dflash == b-1` the copied slot is `b`, i.e. the first MTP child of dflash slot 1, whose true position is `position+2` — a row from a completely different position. Second risk: fixing the row count but not the `target_hidden_abs_positions` pushes (or vice versa) reproduces the same desync in mirror image — that is why the fix drives both from one `keep_slots` and asserts against `tree_positions`. Third: "fixing" the linear composer at mtp_compose.rs:574 by symmetry would break a correct path. Fourth: changing the meaning of `n_rows` in `scatter_hidden_block_to_interleaved` instead of adding a sibling silently changes five other call sites including production serving prompt-seed. Fifth (if someone prefers the alternative below): capturing hidden from the step-9 replay `forward_prefill_batch` by passing `Some(hidden_rb)` works only on the branches that actually replay — the tape branch with `accept_mtp == 0` does not prefill, so a uniform `block_size = n_replay` there reads the wrong ring slots, silently.

### Validation

WITHOUT a GPU: (1) `cargo check -p hipfire-arch-qwen35 --all-targets` and `cargo check -p hipfire-runtime --examples` — the new helper, the `tree_positions` re-borrow after `TreeVerifyCtx` is consumed, and the `assert_eq!` format args are all compile-checked; (2) `cargo clippy -p hipfire-arch-qwen35`; (3) `./tests/no-gpu-ci.sh`; (4) `graphify update .` after editing, per CLAUDE.md. The real no-GPU check is the invariant itself: the new `assert_eq!(tree_positions[slot], position + row + co)` reads the position array the verify actually consumed, so any wrong slot list becomes a loud panic instead of a degraded drafter. I deliberately did NOT add a CPU unit test that mirrors the tree layout — it would duplicate the construction at mtp_compose.rs:1078-1096 and keep passing if that construction changed, which is the failure mode that matters. NEEDS a GPU (everything numeric): the scatter is `hipMemcpyDtoD` over `hidden_rb.layer_bufs`, and the rows it moves only exist after a real `verify_dflash_block_tree` forward. Run `dflash_mtp_tree_demo` (trunk + `--dflash` sidecar + MTP head) under `hipfire lock acquire` — non-daemon GPU examples do not self-lock — on halo (gfx1151) for a 27B-class artifact, or nix1 for something small. Two runs matter: (a) a prompt/`--mtp-k` where the demo's end-of-run `accept_mtp_total` is 0 — an A/A control that must be byte-identical before and after, since the gather reduces to the old block copy; (b) a prompt where `accept_mtp_total > 0` — before/after `tau_dflash`, tokens/s and eyeballed coherence. Expect the fix to raise acceptance on (b): today those cycles poison the drafter's context permanently. Also confirm the new assert never fires. `./tests/tiny-affected-gate.sh --require-coverage` does not cover this path (no caller outside the example), so the demo IS the gate.

### Smallest safe first step

Not needed — the fix is implementable. But if you want to land it in two steps: step one is edit 1 (the helper, behavior-preserving for all five existing callers) plus ONLY the `assert_eq!(tree_positions[slot], abs)` loop bolted onto the current `rows_to_keep = accept_dflash + 1`. That makes the silent wrong answer loud — it panics on the first cycle with `accept_mtp > 0` — and proves the bug is live on real hardware before any behavior changes. Step two is the gather itself.

### Other call sites touched

- crates/hipfire-arch-qwen35/src/speculative.rs:8289 — spec_step_dflash's scatter. Unaffected: block form keeps its signature and delegates; its committed rows are contiguous by construction (linear verify).
- crates/hipfire-arch-qwen35/src/mtp_compose.rs:575 — spec_step_dflash_mtp (LINEAR composer). Checked and NOT buggy: its verify block is the linear chain [seed, dflash..., mtp...] (line 492-496), so committed slots are the contiguous prefix and its `rows_to_keep = accept_len + 1` (line 574) is already right. Do not 'fix' it.
- crates/hipfire-serving-core/src/generate.rs:733 — prompt-seed scatter (dst_row_offset 0, n_rows = prompt_len). Unaffected by delegation.
- crates/hipfire-runtime/examples/dflash_mtp_tree_demo.rs:355, dflash_mtp_demo.rs:349, dflash_spec_demo.rs:1786 — prompt-seed scatters. Unaffected.
- crates/hipfire-runtime/examples/dflash_mtp_tree_demo.rs:414 — the ONLY caller of spec_step_dflash_mtp_tree. It advances `position += result.committed.len() - 1` (line 443-444), which is what the fix aligns the row count to. Blast radius of the mtp_compose edit is this example only.
- crates/hipfire-arch-qwen35/src/lib.rs:69 — re-export of spec_step_dflash_mtp_tree/MtpComposeTreeState/MtpComposeTreeResult. No signature change, no update needed. Nothing in the daemon/server/serving-core reaches this path.
- Consumers of the data the fix repairs: crates/hipfire-runtime/src/dflash.rs:2447-2462 (`uploaded_target_hidden_rows` delta-upload tracker — only consulted on the `target_hidden = Some(host)` path, which this composer never takes, but keep it in lockstep) and dflash.rs `draft_ctx_cached_rows` (k_ctx/v_ctx projection cache keyed on row index — it is the thing that silently caches mis-positioned rows today).
- crates/hipfire-arch-qwen35/src/speculative.rs:12312-12319 — spec_step_ddtree_batched's host-side slot gather (`src_row = accepted_node_indices[i] + 1`). This is the 'equivalent' the report refers to; it is NOT reusable (operates on the `target_hidden_host` Vec<f32>, not the GPU tensor), it is only the precedent. spec_step_ddtree (11695-11702) instead re-verifies the committed prefix so its rows come out contiguous.

### Edits (4)

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — Adds the slot-indexed gather the tree path needs (lines 7183-7224). The block form keeps its exact signature and semantics for its five existing callers and delegates, so the ring-wrap/start_slot math is not duplicated. `Gpu`, `GpuTensor`, `HipResult`, `HiddenStateRingBuffer` are all already in scope in this file; no new imports. The per-call `Vec` is <=B elements per cycle (and one prompt-length alloc at seed time) — noise against the ~80 memcpy enqueues it wraps.

```
ANCHOR:
pub fn scatter_hidden_block_to_interleaved(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    dst: &GpuTensor,
    dst_row_offset: usize,
    block_size: usize,
    n_rows: usize,
) -> HipResult<()> {
    assert!(
        n_rows <= block_size,
        "scatter: n_rows {n_rows} > block_size {block_size}"
    );
    let num_extract = hidden_rb.extract_layers.len();
    let hidden = hidden_rb.hidden_dim;
    let max_pos = hidden_rb.max_positions;
    let head = hidden_rb.head;
    let written = hidden_rb.written;
    assert!(
        block_size <= written,
        "scatter: block_size {block_size} > written {written}"
    );
    let row_bytes = hidden * 4;
    let start_slot = (head + max_pos - block_size) % max_pos;

    for r in 0..n_rows {
        let slot = (start_slot + r) % max_pos;
        let dst_row = dst_row_offset + r;
        let dst_row_base_bytes = dst_row * num_extract * row_bytes;
        for ext in 0..num_extract {
            let src_offset_bytes = slot * row_bytes;
            let dst_offset_bytes = dst_row_base_bytes + ext * row_bytes;
            gpu.hip.memcpy_dtod_at(
                &dst.buf,
                dst_offset_bytes,
                &hidden_rb.layer_bufs[ext].buf,
                src_offset_bytes,
                row_bytes,
            )?;
        }
    }
    Ok(())
}

REPLACEMENT:
pub fn scatter_hidden_block_to_interleaved(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    dst: &GpuTensor,
    dst_row_offset: usize,
    block_size: usize,
    n_rows: usize,
) -> HipResult<()> {
    assert!(
        n_rows <= block_size,
        "scatter: n_rows {n_rows} > block_size {block_size}"
    );
    let slots: Vec<usize> = (0..n_rows).collect();
    scatter_hidden_slots_to_interleaved(gpu, hidden_rb, dst, dst_row_offset, block_size, &slots)
}

/// Slot-indexed form of [`scatter_hidden_block_to_interleaved`], for TREE
/// verifies where the committed path is NOT a contiguous prefix of the
/// verified block.
///
/// `slots[i]` indexes the just-written `block_size`-row window (i.e. the tree
/// linearization slot) and lands at dst row `dst_row_offset + i`. Passing
/// `slots = [0, 1, ..., n-1]` is exactly the block form, which is why that one
/// now delegates here — the ring-wrap math lives in one place.
pub fn scatter_hidden_slots_to_interleaved(
    gpu: &Gpu,
    hidden_rb: &HiddenStateRingBuffer,
    dst: &GpuTensor,
    dst_row_offset: usize,
    block_size: usize,
    slots: &[usize],
) -> HipResult<()> {
    let num_extract = hidden_rb.extract_layers.len();
    let hidden = hidden_rb.hidden_dim;
    let max_pos = hidden_rb.max_positions;
    let head = hidden_rb.head;
    let written = hidden_rb.written;
    assert!(
        block_size <= written,
        "scatter: block_size {block_size} > written {written}"
    );
    let row_bytes = hidden * 4;
    let start_slot = (head + max_pos - block_size) % max_pos;

    for (dst_i, &src) in slots.iter().enumerate() {
        assert!(
            src < block_size,
            "scatter: slot {src} >= block_size {block_size}"
        );
        let slot = (start_slot + src) % max_pos;
        let dst_row = dst_row_offset + dst_i;
        let dst_row_base_bytes = dst_row * num_extract * row_bytes;
        for ext in 0..num_extract {
            let src_offset_bytes = slot * row_bytes;
            let dst_offset_bytes = dst_row_base_bytes + ext * row_bytes;
            gpu.hip.memcpy_dtod_at(
                &dst.buf,
                dst_offset_bytes,
                &hidden_rb.layer_bufs[ext].buf,
                src_offset_bytes,
                row_bytes,
            )?;
        }
    }
    Ok(())
}
```

**`crates/hipfire-arch-qwen35/src/mtp_compose.rs`** — The actual fix (lines 1265-1290). `accepted_slots` (line 1211) already holds the committed non-root slots in path order, so `[0] ++ accepted_slots` is the gather list — no new arithmetic to get wrong, and it stays correct if the walk ever accepts more than one MTP level. `tree_positions` (built at lines 1078-1096) is readable here: its only borrow was through the `TreeVerifyCtx` consumed by `verify_dflash_block_tree` at line 1190. `co` is already bound at line ~855.

```
ANCHOR:
    // ── 8. Append accepted-prefix target hidden rows to draft_scratch ────
    //
    // verify wrote n_total rows of target hidden into hidden_rb in linearization
    // order. The committed-path slots are [0, accepted_slots[0], accepted_slots[1], ...].
    // For dflash chain accepts these are contiguous: 0, 1, 2, ..., accept_dflash.
    // The drafter will only consume up to position+accept_dflash+1 next cycle,
    // so we copy that many rows as a contiguous prefix.
    //
    // MTP-accepted rows are valid target hiddens too, but they correspond to
    // tree positions BEYOND the dflash chain — we just keep them as additional
    // context. Same scatter pattern as spec_step_dflash_mtp's accept_len+1.
    let rows_to_keep = accept_dflash + 1; // committed dflash prefix only (drafter doesn't see MTP rows yet)
    speculative::scatter_hidden_block_to_interleaved(
        gpu,
        hidden_rb,
        &draft_scratch.target_hidden,
        position,
        n_total,
        rows_to_keep,
    )?;
    draft_scratch.uploaded_target_hidden_rows = position + rows_to_keep;
    for p in 0..rows_to_keep {
        draft_scratch
            .target_hidden_abs_positions
            .push(position as i32 + p as i32 + co);
    }

REPLACEMENT:
    // ── 8. Append committed target hidden rows to draft_scratch ──────────
    //
    // verify wrote n_total rows into hidden_rb in TREE-LINEARIZATION order.
    // The committed path visits slots `[0] ++ accepted_slots`. The dflash
    // prefix is contiguous (0..=accept_dflash), but an accepted MTP slot sits
    // at `b + (s-1)*mtp_k + k` — NOT at `accept_dflash + 1`, which is the next
    // (REJECTED) dflash candidate. So gather BY SLOT; a contiguous block copy
    // of `accept_dflash + accept_mtp + 1` rows would silently pick up a
    // rejected token's hidden.
    //
    // The row count has to equal the caller's `position` advance
    // (`committed.len() - 1`): `target_hidden` is indexed by absolute position
    // and `target_hidden_abs_positions` is the parallel array `positions_k` is
    // built from. Keeping only `accept_dflash + 1` left a permanent hole, so
    // after the first MTP accept `positions_k[i]` no longer described row `i`
    // and the drafter RoPE'd its context with the wrong phases.
    let keep_slots: Vec<usize> = std::iter::once(0usize)
        .chain(accepted_slots.iter().copied())
        .collect();
    let rows_to_keep = keep_slots.len(); // = accept_dflash + accept_mtp + 1
    speculative::scatter_hidden_slots_to_interleaved(
        gpu,
        hidden_rb,
        &draft_scratch.target_hidden,
        position,
        n_total,
        &keep_slots,
    )?;
    draft_scratch.uploaded_target_hidden_rows = position + rows_to_keep;
    for (row, &slot) in keep_slots.iter().enumerate() {
        let abs = position as i32 + row as i32 + co;
        // The gather is only correct if the kept slot really carries the
        // absolute position of committed[row]. `tree_positions` is the array
        // the verify itself consumed, so this catches any later change to the
        // linearization or to the greedy walk instead of silently feeding the
        // drafter a mis-positioned row.
        assert_eq!(
            tree_positions[slot], abs,
            "mtp_tree: kept slot {slot} is at tree position {} but committed row {row} is at {abs}",
            tree_positions[slot],
        );
        draft_scratch.target_hidden_abs_positions.push(abs);
    }
```

**`crates/hipfire-arch-qwen35/src/mtp_compose.rs`** — Line 1298. The trunk replay length and the drafter row count are the same quantity; the bug was two independent expressions for it. Binding one from the other makes a future edit that changes one change both.

```
ANCHOR:
    let n_replay = accept_dflash + accept_mtp + 1; // committed up to (but not including) bonus

REPLACEMENT:
    // Identical count to the hidden rows kept in step 8 — they describe the
    // same committed tokens. Deriving it instead of recomputing it is what
    // stops the two from drifting apart again (that drift WAS the bug).
    let n_replay = rows_to_keep; // = accept_dflash + accept_mtp + 1
```

**`crates/hipfire-runtime/examples/dflash_mtp_tree_demo.rs`** — OPTIONAL, demo-only (line 409). Pre-existing latent issue that the fix makes reachable sooner: `DflashScratch::new_with_mq(..., ctx_capacity, ...)` (line 241) sizes `target_hidden` at `[ctx_capacity * ne * h]` while the loop guard uses `max_seq_total = ctx_capacity + max_tokens*(b+k)/b + n_verify + 16`. Failure mode is a loud assert, not corruption, so skip it if you want the minimal diff.

```
ANCHOR:
        if position + n_verify >= max_seq_total {
            eprintln!("hit max_seq {}; stopping", max_seq_total);
            break;
        }

REPLACEMENT:
        // `target_hidden` / `k_ctx_cached` are allocated for `ctx_capacity`
        // rows and indexed by ABSOLUTE position, and `max_seq_total` is larger
        // than `ctx_capacity`. Now that every committed row is scattered
        // (rows written == position), the draft-scratch ceiling is the real
        // one — without this the run ends in a memcpy_dtod_at bounds assert
        // or draft_forward's `ctx_len > scratch max`.
        if position + n_verify >= max_seq_total || position + n_verify >= draft_scratch.max_ctx_len
        {
            eprintln!(
                "hit ceiling (max_seq {} / draft ctx {}); stopping",
                max_seq_total, draft_scratch.max_ctx_len
            );
            break;
        }
```


---

## Gemma3 rejected draft K/V in the SWA ring

**Verdict:** `implementable-with-gpu-validation`

### What the change is

Gemma3's `SpecTarget` verify writes every block position into the LOCAL layers' modular SWA ring (`swa_ring_write_batched_f32`, slot = `pos % window`), and `commit_prefix` (crates/hipfire-arch-gemma3/src/spec_impl.rs:298) does nothing to undo the rejected tail. The no-op's stated justification is false for a modular ring: a rejected draft at position `p` occupies slot `p % win`, which aliases position `p - win` — a position that is still inside a later forward's visible span (`swa_visibility_stage_batched` reads `ring[pos_i % win]` for every `pos_i < start_pos`), so once context passes `sliding_window` the rejected K/V is silently attended to. The fix mirrors deepseek4's `spec_decode::swa_rewind` but lands where the trait already asks for it: `Gemma3SpecScratch` (today a unit struct) grows a `[n_local_layers, n_kv*head_dim, cols]` F32 K and V snapshot allocated in `new_spec_scratch`; all three verify entry points snapshot the ring slots `[position, position+m)` before the forward; `commit_prefix` restores the columns for the rejected tail `[position+accept_len+1, position+block.len())`. The whole change is one file (`crates/hipfire-arch-gemma3/src/spec_impl.rs`) plus one line in `tests/no-gpu-ci.sh`; no new deps, no `Gemma3State` field, no `forward.rs` change. Two differences from deepseek4 are deliberate: it restores the WHOLE rejected tail (deepseek4 skips the first stale slot on the assumption the next verify rewrites it), and it is default-ON rather than env-gated, with a cheap pre-wrap skip (`position + m <= window`) so short-context requests pay nothing. gemma4 has NO `impl SpecTarget` (only llama and gemma3 do), so despite sharing the same SWA staging primitives in `crates/hipfire-arch-gemma4/src/forward.rs:704,715` it has no reject path and therefore does not have this gap today — same for cohere2, whose use is calibration-only.

### Why the obvious fix is wrong

Four concrete ways a plausible version breaks:

1. **Copying deepseek4's `lo = accepted_len + 1` verbatim.** deepseek4's `accepted_len` is `accepted_tokens.len()`, which INCLUDES the corrected token at the divergence position, so `accepted_len + 1` is "first stale slot, plus one, because the next verify rewrites that one". Gemma3's `accept_len` excludes the bonus (the trait doc pins it: full accept is `accept_len == block.len() - 1`). Transplanting `+1` leaves exactly one rejected draft's K/V in the ring — the rarest, hardest-to-see corruption, one aliased position per window. The unit test above fires on precisely this.

2. **Restoring `[accept_len, ...)` instead of `[accept_len + 1, ...)`.** Off by one the other way: it reverts the LAST ACCEPTED draft's K/V to a `pos - window` value, corrupting the committed prefix — a much worse, always-on wrongness that a short-prompt smoke would still miss.

3. **Snapshotting from inside `forward.rs` next to the `swa_ring_write_batched_f32` call.** That is where the bug report points, but the ring write happens per layer inside the layer loop, after that layer's staging has already consumed the pre-chunk ring — a snapshot there is still correct, but it has to be threaded through `forward_verify_batch` AND `forward_after_x_capture` (the per-token arm at forward.rs:592, which also writes the ring), and it puts spec-decode state into the plain AR forward that prefill and decode also call. Doing it at the `SpecTarget` boundary covers both arms with one call and leaves `forward.rs` untouched.

4. **Flattening the ring wrong for n_kv > 1.** deepseek4 has `num_key_value_heads == 1`, so its `rows = cfg.num_key_value_heads * cfg.head_dim` never exercised the multi-head case. Gemma3-4b has n_kv=4 (head_dim 256) and 27b has n_kv=16 (head_dim 128). The flattening is still valid — the ring is `[n_kv, head_dim, win]` contiguous, i.e. `[n_kv*head_dim, win]` row-major, which is why one `strided_copy_2d` per run covers all heads — but a "fix" that loops per kv head with `sub_offset(kvh * head_dim * win, ...)` and then passes `rows = head_dim` while keeping the snapshot stride at `n_kv*head_dim*cols` will silently interleave heads. Symptom: heads 1..n_kv attend to another head's K/V post-wrap, which reads as mild quality loss, not a crash.

Also: do not "simplify" the pre-wrap skip to `position < win`. The condition must be `position + m <= win` — a block that straddles the wrap point (position 1022, m 5, win 1024) does alias.

### Validation

**Without a GPU (covers the arithmetic, which is where the fix can be subtly wrong):**
- `cargo test -p hipfire-arch-gemma3 --lib swa_ring` — the four `ring_slot_runs` cases plus `restore_leaves_every_readable_slot_at_its_own_position`, which sweeps `base` over three ring wraps × every accept length and asserts the invariant the fix exists for: every position still readable out of the ring carries its own K/V. It fails if the restore is dropped, if the restore range is shifted either direction, or if the pre-wrap skip is widened to `position < win`. Added to `tests/no-gpu-ci.sh`.
- `./tests/no-gpu-ci.sh` — its `cargo check --workspace --examples -D warnings` catches the signature churn (`_scratch` -> `scratch`) and, importantly, that `crates/hipfire-arch-gemma3/examples/spec_parity.rs` still builds against the widened `Gemma3SpecScratch`.

**Needs a GPU (nothing above touches `strided_copy_2d`, VRAM, or the real ring):**
- The offsets/strides handed to `strided_copy_2d` are only checked on device. A transposed snapshot lane or a wrong `cols` stride is invisible to the CPU test and shows up as corrupted attention.
- **The A/B that actually proves it: an AR-vs-spec losslessness run at a context past `sliding_window` with a forced partial accept.** `crates/hipfire-arch-gemma3/examples/spec_parity.rs` is the right vehicle but CANNOT see this bug today — its prompt is `"<start_of_turn>user\nIn one sentence, what is a CT scan?..."`, ~20 tokens, while gemma3-4b's `sliding_window` is 1024 and `Gemma3State::new_with_max_seq` only enables SWA at all when `cfg.sliding_window < max_seq`. Pre-wrap the ring is a plain array and the bug is provably absent, so the smoke passes either way. Extend it (or add a sibling example) to: (a) take a prompt longer than `cfg.sliding_window`; (b) AR-decode a reference continuation of N tokens; (c) reset, re-prefill, then loop `new_spec_scratch` -> `verify_block(block, position)` -> `commit_prefix(block, accept_len, position)` with a deliberately wrong draft tail so `accept_len < block.len() - 1` every window; (d) assert the emitted tokens are token-identical to the AR reference for at least ~3× `sliding_window` of decode. Without the fix that diverges after the first post-wrap partial accept; with it, it must be identical. Run under `hipfire lock acquire --label gemma3-swa-rewind` (a non-daemon GPU example, so it does not self-lock).
- Anything with `sliding_window > 1024` cannot be tested at all: `swa_visibility_stage_batched` rejects it (dispatch/attention.rs:5597, the window is the thread-block dimension).
- Perf: post-wrap the fix adds `2 × n_local_layers` `strided_copy_2d` launches per window (one or two runs each) for the snapshot, plus the same on a partial accept — ~56 tiny launches per window on gemma3-4b, ~104 on 27b. Expect ≲2% of a verify forward; measure it on the same run rather than trusting that estimate. Pre-wrap the cost is exactly zero.

### Other call sites touched

- crates/hipfire-specdecode-dspark/src/dspark_core.rs:1210 / 1228 / 1265 — the ONLY production driver: `new_spec_scratch(gpu, n_verify)` -> `verify_block_capture_gpu` -> `accept_greedy_prefix` -> `commit_prefix(gpu, &verify_tokens, accept_len, position, scratch)` -> `scratch.free(gpu)`. No change needed: the scratch's lifetime already brackets the snapshot/restore pair exactly, and nothing between verify and commit_prefix touches the target's GPU state.
- crates/hipfire-arch-gemma3/examples/spec_parity.rs:143,145,210,286,287,302 — calls `new_spec_scratch` + `verify_block` / `verify_block_logits` / `verify_block_capture_gpu` but never `commit_prefix` and never `scratch.free`. Still compiles and still passes (its ~20-token prompt is far below sliding_window=1024, so `snapshot_swa_ring` short-circuits). The scratch it leaks now leaks two pooled tensors instead of nothing — pre-existing shape, example-only. This is the file to extend for GPU validation (see validation).
- crates/hipfire-arch-gemma3/examples/dspark_labels_gemma3.rs — the other SpecTarget consumer (label generation for a DSpark/DFlash drafter). It benefits directly: corrupted target KV there silently poisons training labels rather than one response.
- crates/hipfire-arch-llama/src/spec_impl.rs:312 `commit_prefix` — stays a correct no-op. LlamaBackend has no SWA ring at all (the grep for `swa_ring_write_batched_f32` hits only deepseek4, gemma3, gemma4 and cohere2), its K/V is the absolute-position-indexed `hipfire_runtime::llama::KvCache`. Do NOT copy this change there.
- crates/hipfire-arch-gemma4/src/forward.rs:639,652,704,715 — gemma4 uses the SAME `swa_visibility_stage_batched` / `swa_ring_write_batched_f32` pair, but there is no `impl SpecTarget for Gemma4*` anywhere in the tree (`grep -rn 'impl SpecTarget'` returns only hipfire-arch-llama, hipfire-arch-gemma3, and a test double in hipfire-specdecode-dspark/src/spec.rs:1030). With no verify/reject path, gemma4 does NOT have this gap today. It inherits it the moment it gets a SpecTarget — port this change with it.
- crates/hipfire-arch-cohere2/src/calibration_stream.rs:1063,1073,1099,1109 — same primitives, calibration-only (a forward pass with no speculative reject). Unaffected.
- crates/hipfire-arch-deepseek4/src/spec_decode.rs:324,387 (`swa_rewind::snapshot` / `restore`, gated by `HIPFIRE_DEEPSEEK4_SPEC_KV_REWIND=1`) — left untouched. Its `slot_subranges` is the arithmetic this fix mirrors; de-duplicating the two into hipfire-rdna is a follow-up, not part of this change.
- crates/hipfire-rdna/src/dispatch/mod.rs:4397 `strided_copy_2d` and kernels/src/strided_copy_2d.hip — reused as-is (`dst[dst_off + r*dst_stride + c] = src[src_off + r*src_stride + c]`, offsets are i32 but the index math casts to size_t before multiplying; largest offset here is n_local*rows*cols ≈ 1e6, far inside i32).
- crates/hipfire-serving-core, crates/hipfire-daemon — do NOT reference SpecTarget at all today, so gemma3 spec-decode is currently reachable only through hipfire-specdecode-dspark and the gemma3 examples. That bounds the blast radius now, and means the fix lands before the daemon wiring rather than after.

### Edits (11)

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — The module doc is the load-bearing statement of the wrong invariant; leaving it would re-justify removing the fix.

```
ANCHOR:
//! Gemma3 is pure attention (no recurrent state), so `commit_prefix` is a no-op:
//! the accepted-prefix KV a verify wrote is already correct and the rejected tail
//! is overwritten by the next window's verify (which re-anchors `next_pos` to the
//! window position, re-writing the same SWA ring slots).

REPLACEMENT:
//! Gemma3 is pure attention (no recurrent state), but `commit_prefix` is NOT a
//! no-op. The GLOBAL layers store K/V in the absolute-position-indexed
//! `kv_cache`, where a rejected draft's row is genuinely overwritten before it
//! can be read. The LOCAL (sliding-window) layers do not: they keep K/V in a
//! MODULAR ring — `swa_ring_write_batched_f32` stores position `p` at slot
//! `p % window`, and `swa_visibility_stage_batched` reads the pre-chunk window
//! back through the same map. A rejected draft at `p` therefore aliases position
//! `p - window`, which is still inside a later forward's visible span, so once
//! the context passes `sliding_window` the rejected K/V is silently attended to.
//! `verify_block*` snapshots those ring slots into the scratch and
//! `commit_prefix` restores the rejected tail — the same treatment deepseek4
//! applies in `spec_decode::swa_rewind`.
```

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — The two helpers the fix needs. `ring_slot_runs` is the pure, GPU-free arithmetic (mirrors deepseek4's unit-tested `slot_subranges`); `swa_ring_copy` is the only GPU code added, shared by snapshot and restore so the two can never disagree on layout.

```
ANCHOR:
/// Per-position argmax over `[m, vocab]` row-major logits.
fn argmax_rows(logits: &[f32], m: usize, vocab: usize) -> Vec<u32> {
    (0..m)
        .map(|r| hipfire_runtime::sampler::argmax(&logits[r * vocab..(r + 1) * vocab]))
        .collect()
}

REPLACEMENT:
/// Per-position argmax over `[m, vocab]` row-major logits.
fn argmax_rows(logits: &[f32], m: usize, vocab: usize) -> Vec<u32> {
    (0..m)
        .map(|r| hipfire_runtime::sampler::argmax(&logits[r * vocab..(r + 1) * vocab]))
        .collect()
}

/// Split the absolute positions `[p_start, p_end)` into `(ring_slot,
/// snapshot_col, len)` runs. Position `p` lives at ring slot `p % win` — the map
/// `swa_ring_write_batched_f32` writes and `swa_visibility_stage_batched` reads
/// — and at snapshot column `p - snap_base`. The ring wraps at `win`, so this
/// yields 1..=2 contiguous runs.
///
/// Mirror of `hipfire_arch_deepseek4::spec_decode::swa_rewind::slot_subranges`,
/// duplicated rather than shared because an arch crate must not depend on
/// another arch crate. If a third SWA arch needs it, hoist both into
/// `hipfire-rdna`, which owns the ring kernels that define the slot map.
///
/// Requires `win > 0` and `p_end - p_start <= win`; both callers guarantee it.
fn ring_slot_runs(
    snap_base: usize,
    p_start: usize,
    p_end: usize,
    win: usize,
) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut p = p_start;
    while p < p_end {
        let len = (win - p % win).min(p_end - p);
        out.push((p % win, p - snap_base, len));
        p += len;
    }
    out
}

/// Copy the SWA ring columns for absolute positions `[p_start, p_end)` between
/// every local layer's ring and its snapshot lane in `sc`. `save = true` copies
/// ring -> snapshot; `false` restores snapshot -> ring.
///
/// Each ring is `[n_kv_heads, head_dim, win]` contiguous, i.e. one `[n_kv *
/// head_dim, win]` row-major block, so a single `strided_copy_2d` per run covers
/// every kv head at once — the same flattening `swa_rewind::snapshot` uses on
/// deepseek4 (where `n_kv_heads == 1` made it invisible).
#[allow(clippy::too_many_arguments)]
fn swa_ring_copy(
    gpu: &mut Gpu,
    state: &Gemma3State,
    sc: &Gemma3SpecScratch,
    rows: usize,
    snap_base: usize,
    p_start: usize,
    p_end: usize,
    save: bool,
) -> Result<(), String> {
    let (Some(snap_k), Some(snap_v)) = (sc.snap_k.as_ref(), sc.snap_v.as_ref()) else {
        return Ok(());
    };
    let win = state.swa_window;
    let cols = sc.cols;
    let runs = ring_slot_runs(snap_base, p_start, p_end, win);
    let mut lane = 0usize; // snapshot lane index, local layers ascending
    for (l, (rk, rv)) in state.swa_k.iter().zip(state.swa_v.iter()).enumerate() {
        // Global l
```

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — The scratch is the designed home for verify->commit_prefix state (the trait's CONTRACT paragraph says so), and dspark_core creates/frees it exactly around the verify+commit pair, so its lifetime already matches the snapshot's.

```
ANCHOR:
/// Gemma3 per-token verify scratch. The baseline path reuses `Gemma3State`'s own
/// B=1 buffers, so there is nothing extra to hold. M1b's batched verify will grow
/// this into a `[block × …]` prefill scratch.
pub struct Gemma3SpecScratch;

impl SpecScratch for Gemma3SpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn free(self: Box<Self>, _gpu: &mut Gpu) {}
}

REPLACEMENT:
/// Gemma3 verify scratch. The verify forward itself reuses `Gemma3State`'s own
/// buffers (per-token arm) or pooled per-call scratch (batched arm), so the only
/// thing this carries is the SWA-ring rewind snapshot the
/// [`SpecTarget::verify_block`] contract demands.
///
/// Why gemma3 needs one where llama does not: llama's K/V lives in the
/// absolute-position-indexed `llama::KvCache`, so a rejected draft's row is
/// overwritten before anything reads it. Gemma3's LOCAL layers keep K/V in a
/// MODULAR ring instead (`swa_ring_write_batched_f32` writes slot `pos % win`;
/// `swa_visibility_stage_batched` reads it back by the same map), so a rejected
/// draft at `p` aliases position `p - win`, which a later forward still reads.
/// Global layers stay on `kv_cache` and need no rewind — the same scoping as
/// deepseek4's `swa_rewind`.
#[derive(Default)]
pub struct Gemma3SpecScratch {
    /// `[n_local_layers, n_kv * head_dim, cols]` F32, local layers ascending.
    /// `None` when SWA is off (there is no ring).
    snap_k: Option<GpuTensor>,
    snap_v: Option<GpuTensor>,
    /// Snapshot column stride (= the largest block this scratch can rewind).
    cols: usize,
    /// First absolute position the snapshot holds; `None` when the last verify
    /// saved nothing (SWA off, or pre-wrap where no aliasing is possible).
    base: Option<usize>,
}

impl SpecScratch for Gemma3SpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn free(self: Box<Self>, gpu: &mut Gpu) {
        for t in [self.snap_k, self.snap_v].into_iter().flatten() {
            let _ = gpu.free_tensor(t);
        }
    }
}
```

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — Sizes the snapshot to the caller's block. `alloc_tensor` goes through `self.pool.alloc` (crates/hipfire-rdna/src/dispatch/mod.rs:2136), so two pooled allocs per window is noise next to the ~9 `alloc_owned` that `forward_verify_batch` already does per call. The snap_v failure arm frees snap_k because `GpuTensor` has no Drop.

```
ANCHOR:
    fn new_spec_scratch(
        &mut self,
        _gpu: &mut Gpu,
        _block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        // Per-token baseline: verify reuses the state's own single-token scratch.
        Ok(Box::new(Gemma3SpecScratch))
    }

REPLACEMENT:
    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        // Verify compute reuses the state's own buffers; the scratch exists only
        // to hold the local layers' SWA-ring snapshot (see `Gemma3SpecScratch`).
        let win = self.state.swa_window;
        let n_local = self.state.swa_k.iter().filter(|r| r.is_some()).count();
        let cols = block_size.min(win);
        if win == 0 || n_local == 0 || cols == 0 {
            return Ok(Box::new(Gemma3SpecScratch::default()));
        }
        let rows = self.config.num_key_value_heads * self.config.head_dim;
        let elems = n_local * rows * cols;
        let snap_k = gpu
            .alloc_tensor(&[elems], DType::F32)
            .map_err(|e| format!("Gemma3SpecScratch snap_k: {e:?}"))?;
        let snap_v = match gpu.alloc_tensor(&[elems], DType::F32) {
            Ok(t) => t,
            Err(e) => {
                let _ = gpu.free_tensor(snap_k);
                return Err(format!("Gemma3SpecScratch snap_v: {e:?}"));
            }
        };
        Ok(Box::new(Gemma3SpecScratch {
            snap_k: Some(snap_k),
            snap_v: Some(snap_v),
            cols,
            base: None,
        }))
    }
```

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — One snapshot entry point shared by all three verify methods, so the per-token and batched arms cannot drift. The `m > sc.cols` arm makes an impossible-on-real-gemma3 config (window smaller than the spec block) loud instead of silently unrewindable.

```
ANCHOR:
        .map_err(|e| format!("gemma3 forward_verify_batch: {e:?}"))?;
        let _ = gpu.free_tensor(x_batch);
        Ok(logits)
    }
}

REPLACEMENT:
        .map_err(|e| format!("gemma3 forward_verify_batch: {e:?}"))?;
        let _ = gpu.free_tensor(x_batch);
        Ok(logits)
    }

    /// Save the SWA ring slots this verify is about to overwrite into `scratch`,
    /// so [`SpecTarget::commit_prefix`] can put the rejected tail back. Both
    /// verify arms write the ring — the per-token one through
    /// `forward_step_capture` (forward.rs:592) and the batched one through
    /// `forward_verify_batch` (forward.rs:1255) — so this runs for both.
    ///
    /// No-op when SWA is off and, the common case, while the ring has not yet
    /// wrapped: with `position + m <= window` every slot equals its own position,
    /// so a rejected draft's slot is strictly above every position a later
    /// forward can read and can never alias one.
    fn snapshot_swa_ring(
        &mut self,
        gpu: &mut Gpu,
        position: usize,
        m: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        let sc = scratch
            .as_any_mut()
            .downcast_mut::<Gemma3SpecScratch>()
            .ok_or("gemma3 verify: scratch is not Gemma3SpecScratch")?;
        sc.base = None;
        let win = self.state.swa_window;
        if win == 0 || sc.snap_k.is_none() || m == 0 || position + m <= win {
            return Ok(());
        }
        if m > sc.cols {
            return Err(format!(
                "gemma3 verify: block {m} exceeds the SWA rewind snapshot ({} cols, \
                 sliding_window {win}); new_spec_scratch was sized for a smaller block",
                sc.cols
            ));
        }
        let rows = self.config.num_key_value_heads * self.config.head_dim;
        swa_ring_copy(gpu, &self.state, sc, rows, position, position, position + m, true)?;
        sc.base = Some(position);
        Ok(())
    }
}
```

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — verify_block at line 180; the per-token arm advances the ring one slot per token exactly as the batched arm does, so it needs the same snapshot.

```
ANCHOR:
        _scratch: &mut dyn SpecScratch,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        // Batched fast path (M1b): one forward for the whole block, ~2.6× faster

REPLACEMENT:
        scratch: &mut dyn SpecScratch,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        // CONTRACT: snapshot what `commit_prefix` has to rewind BEFORE the forward
        // advances it. For gemma3 that is the local layers' SWA ring — written by
        // BOTH arms below.
        self.snapshot_swa_ring(gpu, position, block.len(), scratch)?;
        // Batched fast path (M1b): one forward for the whole block, ~2.6× faster
```

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — verify_block_logits at line 229 — the temp>0 / logits consumer of the same forward, identically ring-advancing.

```
ANCHOR:
        _scratch: &mut dyn SpecScratch,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        // Batched fast path (M1b) when no host-Vec capture is requested.

REPLACEMENT:
        scratch: &mut dyn SpecScratch,
        mut hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        // Same CONTRACT as `verify_block`: snapshot the SWA ring before the
        // forward advances it.
        self.snapshot_swa_ring(gpu, position, block.len(), scratch)?;
        // Batched fast path (M1b) when no host-Vec capture is requested.
```

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — verify_block_capture_gpu at line 280 — the path dspark_core.rs:1228 actually calls in production.

```
ANCHOR:
        _scratch: &mut dyn SpecScratch,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        if !self.batched_verify_ok(block.len()) {
            return Ok((Vec::new(), false));
        }

REPLACEMENT:
        scratch: &mut dyn SpecScratch,
        hidden_gpu: &GpuTensor,
    ) -> Result<(Vec<u32>, bool), String> {
        // Same CONTRACT as `verify_block`. Runs before the decline check on
        // purpose: it also clears any snapshot a previous verify left on this
        // scratch, so a declined call cannot make `commit_prefix` restore a stale
        // window. On the decline path no forward runs, so the restore it enables
        // would write the ring back byte-identical anyway.
        self.snapshot_swa_ring(gpu, position, block.len(), scratch)?;
        if !self.batched_verify_ok(block.len()) {
            return Ok((Vec::new(), false));
        }
```

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — The actual fix. `accept_len` is accepted DRAFTS only (the trait doc says full accept is `accept_len == block.len() - 1`), so the committed span is `accept_len + 1` slots and the stale span starts at `position + accept_len + 1`. `base.take()` makes a second commit_prefix on the same scratch a no-op.

```
ANCHOR:
    fn commit_prefix(
        &mut self,
        _gpu: &mut Gpu,
        _block: &[u32],
        _accept_len: usize,
        _position: usize,
        _scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // Pure attention: verify's accepted-prefix KV is already correct, and the
        // next window's verify re-anchors `next_pos` and overwrites the rejected
        // tail (same SWA ring slots, since positions are contiguous). Nothing to do.
        Ok(())
    }

REPLACEMENT:
    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        // No recurrent state to rewind — but the LOCAL layers' K/V lives in a
        // MODULAR ring and the verify wrote all `block.len()` positions into it.
        // The committed prefix is `block[..accept_len + 1]`; the rejected tail
        // [position + accept_len + 1, position + block.len()) sits in ring slots
        // that alias positions a later forward still reads, so put those slots'
        // pre-verify contents back. Global layers are absolute-position indexed
        // and need nothing.
        //
        // Unlike deepseek4's `swa_rewind::restore`, this restores the WHOLE
        // rejected tail instead of skipping its first slot on the assumption that
        // the next window's verify rewrites it: nothing guarantees the next call
        // is a verify at `position + accept_len + 1` (the request can finish,
        // abort, or fall back to AR decode), and one extra column is free.
        let sc = scratch
            .as_any_mut()
            .downcast_mut::<Gemma3SpecScratch>()
            .ok_or("gemma3 commit_prefix: scratch is not Gemma3SpecScratch")?;
        let Some(base) = sc.base.take() else {
            return Ok(()); // SWA off, or pre-wrap: nothing was saved
        };
        if base != position {
            return Err(format!(
                "gemma3 commit_prefix: position {position} does not match the \
                 snapshot base {base}"
            ));
        }
        let stale_start = position + accept_len + 1;
        let stale_end = position + block.len();
        if stale_start >= stale_end {
            return Ok(()); // full accept: every written position is committed
        }
        let rows = self.config.num_key_value_heads * self.config.head_dim;
        swa_ring_copy(gpu, &self.state, sc, rows, base, stale_start, stale_end, false)
    }
```

**`crates/hipfire-arch-gemma3/src/spec_impl.rs`** — The one runnable check. `restore_leaves_every_readable_slot_at_its_own_position` fails if the restore is dropped, and also fails if the restore start is shifted by one (deepseek4's skip-the-first-slot form applied to gemma3's accept_len convention) — which is the most likely wrong version of this fix.

```
ANCHOR:
            let logits = gpu
                .download_f32(&self.state.logits)
                .map_err(|e| format!("gemma3 lm_head_logits download row {r}: {e:?}"))?;
            out.extend_from_slice(&logits);
        }
        Ok(out)
    }
}

REPLACEMENT:
            let logits = gpu
                .download_f32(&self.state.logits)
                .map_err(|e| format!("gemma3 lm_head_logits download row {r}: {e:?}"))?;
            out.extend_from_slice(&logits);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod swa_ring_tests {
    use super::ring_slot_runs;

    #[test]
    fn runs_no_wrap() {
        assert_eq!(ring_slot_runs(100, 100, 103, 128), vec![(100, 0, 3)]);
    }

    #[test]
    fn runs_wrap() {
        // positions 126..130 in a 128-slot ring: slots 126,127,0,1 -> cols 0..4.
        assert_eq!(
            ring_slot_runs(126, 126, 130, 128),
            vec![(126, 0, 2), (0, 2, 2)]
        );
    }

    #[test]
    fn runs_subset_across_wrap() {
        // Restore-only slice of the same snapshot: cols [2,4) -> slots 0,1.
        assert_eq!(ring_slot_runs(126, 128, 130, 128), vec![(0, 2, 2)]);
    }

    #[test]
    fn runs_empty_when_nothing_rejected() {
        assert!(ring_slot_runs(100, 102, 102, 128).is_empty());
    }

    /// The invariant the fix exists for, on a CPU model of one ring row: after a
    /// verify that wrote `M` positions and a partial accept of `accept` drafts,
    /// EVERY position a later forward can still read out of the ring must carry
    /// its own K/V, never a rejected draft's. Sweeps `base` across three wraps so
    /// both the pre-wrap (snapshot skipped) and post-wrap regimes are covered.
    #[test]
    fn restore_leaves_every_readable_slot_at_its_own_position() {
        const WIN: usize = 8;
        const M: usize = 5; // seed + 4 drafts
        for base in 0..3 * WIN {
            for accept in 0..M {
                let committed_end = base + accept + 1; // first uncommitted position
                // Pre-verify ring: AR history [0, base). Store p+1 so an unwritten
                // slot (0) is distinguishable from position 0.
                let mut ring = vec![0usize; WIN];
                for p in 0..base {
                    ring[p % WIN] = p + 1;
                }
                // Snapshot, skipped pre-wrap exactly as `snapshot_swa_ring` does.
                let wrapped = base + M > WIN;
                let mut snap = vec![0usize; M];
                if wrapped {
                    for &(slot, col, len) in &ring_slot_runs(base, base, base + M, WIN) {
                        snap[col..col + len].copy_from_slice(&ring[slot..slot + len]);
                    }
                }
                // Verify writes all M block positions into the ri
```

**`tests/no-gpu-ci.sh`** — no-gpu-ci.sh runs `cargo check --workspace --examples` but no gemma3 unit tests, so the new test would never run. The filter matches the `swa_ring_tests` module path.

```
ANCHOR:
cargo test -p hipfire-arch-llama --lib caps

REPLACEMENT:
cargo test -p hipfire-arch-llama --lib caps
# gemma3 spec-decode SWA ring rewind: pure slot arithmetic, no GPU.
cargo test -p hipfire-arch-gemma3 --lib swa_ring
```


---

## DDTree GDN tape replay

**Verdict:** `implementable-with-gpu-validation`

### What the change is

The fix is a capture counter, not predicate mirroring. `forward_prefill_chunk` (crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs:2385) is the single site that writes GDN tape rows — it is reachable only from `forward_prefill_batch*`, it already increments `BATCHED_PREFILL_ROWS` for exactly this "did the batched path run" question, and it receives both the tape and `tape_offset`. Add `GdnTape::captured_rows`, stamp it there (`tape_offset + n`), zero it in the one branch that writes no rows (`forward_prefill_batch_with_pbs_opts`'s `if !eligible` per-token fallback, prefill_batch.rs:6482), and restamp it on hipGraph replay in `verify_dflash_block_inner` (speculative.rs:6891) from a new per-B `verify_graph_gdn_tape` set that mirrors the existing `verify_graph_lmhead_argmax`. Then replace the seven bare `gdn_tape.replay_gdn(...)` calls in the three DDTree steps with one helper that replays when `captured_rows >= n_steps` and otherwise re-runs `forward_prefill_batch` over the committed tokens — the exact `FullPrefill` fallback `spec_step_dflash` already uses at speculative.rs:10861. Mirroring `spec_step_dflash`'s predicate is the wrong fix and I can show why: eligibility is n-dependent (`n >= MIN_BATCH`, `n <= 32` decouple, mod.rs:2428/2300) and each DDTree verify uses a different n (tape_block, big_n, main chain, branch chain), the tree verify adds a `kv_asym2_tree` term the DFlash predicate does not have, and — decisively — `forward_prefill_batch_single_chunk_captured_opts` (prefill_batch.rs:6055) calls `forward_prefill_chunk` WITHOUT consulting `prefill_batch_pbs_eligible` at all, so on the verify-graph path the tape IS written even when the predicate says it is not. The predicate is wrong in both directions; the counter is a record of what actually happened. The same counter also fixes two sibling bugs of the same class for free: mtp_spec.rs:2727 gates on the base predicate without the `kv_f32` term the forward adds, and mtp_compose.rs:503/1161 gates only on MoE-ness — both already have correct full-prefill fallbacks, so each is a one-line condition change.

### Why the obvious fix is wrong

The obvious wrong fix is the one the bug report offers as an option: give the three DDTree steps `prefill_batch_pbs_eligible(...) && kv_batched_capable` the way spec_step_dflash does. It is wrong in BOTH directions and would look like it worked. (1) Eligibility is n-dependent — `n >= MIN_BATCH` (mod.rs:2428, MIN_BATCH=2) and the `n <= 32` verify-decouple term (mod.rs:2300) — and every DDTree verify uses a different n: tape_block.len() in spec_step_ddtree, big_n then accept_len+1 in the batched step, 1+main_path.len() then 1+chain.len() in Path C. One predicate evaluated once with `b` describes none of them. (2) The tree verify additionally needs the `kv_asym2_tree` term (prefill_batch.rs:6470) that the DFlash predicate does not carry. (3) Decisively, `forward_prefill_batch_single_chunk_captured_opts` (prefill_batch.rs:6055) calls forward_prefill_chunk WITHOUT consulting the predicate at all — so on the verify-graph path the tape IS written when the predicate says it is not, and the "fix" would silently disable the tape fast path (the documented 40 ms -> 79 ms rollback regression) on exactly the models where it works today, while the metric that would catch it is tok/s, not correctness.

Second wrong version: putting the `captured_rows` check inside `replay_gdn` instead of the helper. Five env-gated diagnostics replay tapes filled by `forward_scratch_capture_gdn_tape` and `repair_qkvza_from_captured_x_single_row`, neither of which goes through forward_prefill_chunk; they would all start returning Err the moment anyone sets HIPFIRE_DFLASH_ROLLBACK_SERIAL_REPLAY or the qkvza-repair flag.

Third: clearing `captured_rows` on ENTRY to forward_prefill_batch_with_pbs_opts rather than only in the `!eligible` branch. Correct, but every hipGraph replay cycle then reads 0 and takes FullPrefill — a large, silent throughput regression with no correctness signal.

Fourth: making the miss a hard error instead of falling back. DDTree then stops working entirely on fp32-KV / unbatchable targets where it runs today (wrongly). Loud is better than silent, but the FullPrefill fallback already exists three functions away and costs one call.

Fifth: forgetting `verify_graph_gdn_tape.clear()` in verify_graph_destroy_all. After a model unload the stale B-set claims a freshly captured tape-less graph writes the tape — reintroducing the exact bug through the graph path.

Sixth (compile-level): the Path B edit (~12271) passes `&mut target` where an earlier `let kv = &mut target.kv_cache;` (~12163) is in scope. NLL should have ended that borrow at its last use inside the layer loop, and the line immediately above already does `&mut target.dn_state`, but this is the one hunk most likely to need a borrow shuffle. Do not "fix" it by cloning or by moving the restore.

### Validation

WITHOUT a GPU:
- `cargo test -p hipfire-arch-qwen35 gdn_tape_replay_needs_rows_the_forward_actually_wrote` — the new pure predicate test. It fails if anyone re-widens the gate to "tape is Some".
- `cargo check -p hipfire-arch-qwen35 -p hipfire-rdna` catches the three struct-literal sites (GdnTape::new_for_config, GdnTapeShards::new, the Gpu ctor), the move-vs-borrow of `gdn_tape` across the graph arms in verify_dflash_block_inner, and the Path B borrow risk above. This is the load-bearing no-GPU check — most of the risk in this change is borrow/initialisation, not arithmetic.
- `./tests/no-gpu-ci.sh` before handoff.
- Static invariant a reviewer can check by grep, no hardware: `captured_rows` is written in exactly three places — forward_prefill_chunk (set to tape_offset+n), the `!eligible` branch of forward_prefill_batch_with_pbs_opts (set to 0), and the graph-replay arm of verify_dflash_block_inner (set from verify_graph_has_gdn_tape). If a fourth write ever appears, or if forward_prefill_batch_with_pbs_opts grows a second early return that skips both, the invariant is broken.

NEEDS a GPU, and why:
- The corruption itself only exists on device: it is stale F32 innovation rows replayed through conv1d/GDN kernels into live DeltaNet state. Repro on halo (gfx1151) or nix1: DDTree (Path C is the cleanest, since HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH defaults off so it always takes the eligibility-gated route) on a hybrid Qwen3.5 target forced to a declining tier — `--kv-mode f32`, or HIPFIRE_PREFILL_BATCHED=0 with n>32. Confirm the decline with HIPFIRE_DEBUG_PREFILL_ELIGIBLE=1 (`[prefill-eligible] final=false`). Before the fix: garbage/attractor output. After: the new `ddtree rollback: full-prefill re-run, verify captured no GDN tape` line in the kernel_trace fallback report, and coherent output.
- The no-regression half also needs a GPU and is the more important measurement: on a KVarN or Q8 target where the tape DOES get captured, rollback_parity must still report a non-zero `replay_gdn_tape=` and tok/s must be unchanged. This is what proves the hipGraph stamp works — graph capture and replay only happen on device, so there is no host-side way to test that the replay arm restamps correctly. If tok/s drops and `replay_full_prefill=` climbs, the graph-side stamp is not firing.
- `./tests/tiny-affected-gate.sh --require-coverage` is the automatic front tier for this (runtime + spec-decode change). Run `hipfire-eval` directly, NOT under `hipfire lock` — it loads through the daemon, which takes the lock itself.

### Other call sites touched

- crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs:2385 forward_prefill_chunk — pub(crate), 6 callers. Stamping there is seen by ALL of them: prefill_batch.rs:6700 (the chunk loop under forward_prefill_batch_with_pbs_opts), prefill_batch.rs:6055 (forward_prefill_batch_single_chunk_captured_opts, the hipGraph entry point), prefill_batch.rs:6131 (forward_prefill_batch_banded, gdn_tape=None), ep.rs:327 (EP driver, gdn_tape=None), ep.rs:2466 (PP bands, per-shard tape).
- crates/hipfire-arch-qwen35/src/speculative.rs:5192 GdnTape::replay_gdn — DELIBERATELY UNCHANGED. Adding the check inside it would break five env-gated diagnostics whose tapes are filled by writers that never touch forward_prefill_chunk: forward_scratch_capture_gdn_tape (speculative.rs:8481, 9279, 10224, 10727 -> qwen35/mod.rs:1083, a per-token row-by-row writer) and repair_qkvza_from_captured_x_single_row (speculative.rs:9708). Those tapes would report captured_rows == 0 forever.
- crates/hipfire-arch-qwen35/src/speculative.rs:8446 prefix_tape.replay_gdn — filled by verify_dflash_block_with_graph_policy(VerifyGraphPolicy::Disabled), i.e. via forward_prefill_batch_with_pbs_opts, so it is stamped/cleared correctly. Left unchanged (it is already inside a `gdn_tape_opt.is_some()` guard that implies spec_step_dflash's own working predicate).
- crates/hipfire-arch-qwen35/src/speculative.rs:7343 spec_step_dflash — NOT changed. Its `verify_populates_tape` (7994) plus `dflash_use_gdn_tape_replay` (6574) already gate correctly; it is conservative (it also declines on the graph path, which does write the tape) but never wrong. Churning it risks the one working path.
- crates/hipfire-arch-qwen35/src/mtp_spec.rs:3611 tape_captured = trunk_gdn_tape_shards.is_some() — the SAME allocation-is-not-capture bug for the multi-GPU PP path, feeding GdnTapeShards::replay_gdn_multi (speculative.rs:3534), which is a separate implementation and does not route through replay_gdn. Out of scope here: the shards are captured through ep.rs:2466 -> forward_prefill_chunk, so each shard's captured_rows IS stamped by this change and a follow-up can gate on `shards.shards[0].captured_rows >= advance` once someone can test PP.
- crates/hipfire-detect/src/rollback.rs:32/53 — parses `replay_gdn_tape=` / `replay_full_prefill=` counters out of the rollback_parity line. Threading the helper's returned SpecRollbackReplayKind into the three DDTree return values keeps those counters truthful; without it a full-prefill cycle still reports gdn_tape.
- crates/hipfire-rdna/src/dispatch/mod.rs:2473 invalidate_graph_state — delegates to verify_graph_destroy_all, so the new set is cleared on model unload and on KV-mode switch (invalidate_for_kv_mode_switch) with no extra edit. Update the doc-comment field list at 2469 if you want it exhaustive.
- crates/hipfire-arch-qwen35/src/mtp_probe.rs:264 passes gdn_tape=None — unaffected.
- crates/hipfire-arch-qwen35/src/pflash.rs:860 — pflash passes max_layer with no tape; the unconditional stamp is a harmless field write there.

### Edits (27)

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — The marker. No existing field distinguishes "verify wrote these rows" from "verify declined and last cycle's rows are still here".

```
ANCHOR:
    pub base_position: usize,
}

REPLACEMENT:
    pub base_position: usize,
    /// How many tape rows the LAST forward handed this tape actually wrote.
    ///
    /// Stamped by `qwen35::prefill_chunk::forward_prefill_chunk` — the only
    /// code that writes tape rows — zeroed by
    /// `forward_prefill_batch_with_pbs_opts` when it declines the batched path
    /// and drops to the tape-less per-token loop, and restamped by
    /// `verify_dflash_block_inner` on a hipGraph replay, where the captured
    /// memcpy nodes write the tape with no Rust code running.
    ///
    /// This is a RECORD of what the forward did. `base_position` is not: it is
    /// set unconditionally at the top of `verify_dflash_block_inner`, BEFORE
    /// the batched/per-token decision, so it is non-zero on a tape nobody
    /// wrote. Neither is a caller-side copy of `prefill_batch_pbs_eligible`:
    /// the captured-graph entry point bypasses that predicate entirely.
    pub captured_rows: usize,
}
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — `GdnTape::new_for_config` struct literal (~2610) must initialise the new field; a fresh tape has captured nothing.

```
ANCHOR:
            q8_requant_frame_layers: n_la_layers,
            base_position: 0,

REPLACEMENT:
            q8_requant_frame_layers: n_la_layers,
            base_position: 0,
            captured_rows: 0,
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — `GdnTapeShards::new` builds `GdnTape` by literal (~2946), the only other construction site; without this the crate does not compile.

```
ANCHOR:
                q8_requant_frame_layers: n_la_total,
                base_position: 0,

REPLACEMENT:
                q8_requant_frame_layers: n_la_total,
                base_position: 0,
                captured_rows: 0,
```

**`crates/hipfire-arch-qwen35/src/qwen35/prefill_chunk.rs`** — The one site where tape rows are written. Stamping beside the existing execution probe keeps the two facts in one place; `tape_offset + n` is correct across the multi-chunk loop because `tape_offset == chunk_start`.

```
ANCHOR:
    let n = tokens.len();
    BATCHED_PREFILL_ROWS.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);

REPLACEMENT:
    let n = tokens.len();
    BATCHED_PREFILL_ROWS.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
    // Same question BATCHED_PREFILL_ROWS answers, asked per-tape: this function
    // is the only writer of GDN tape rows, and it is reachable only from
    // `forward_prefill_batch*`. Rollback callers read this to tell a tape this
    // cycle filled apart from one a declined forward left holding last cycle's
    // rows — replaying the latter walks live DeltaNet state off stale data.
    if let Some(tape) = gdn_tape.as_deref_mut() {
        tape.captured_rows = tape_offset + n;
    }
```

**`crates/hipfire-arch-qwen35/src/qwen35/prefill_batch.rs`** — The only exit from this function that leaves the tape untouched. Clearing here (not on entry) is what lets a hipGraph replay keep its stamp.

```
ANCHOR:
    if !eligible {
        // Flush the recorded declines on the path that actually refuses.

REPLACEMENT:
    if !eligible {
        // The per-token loop below writes NO tape rows. Clear the marker so a
        // caller that replays `gdn_tape` after this call sees "not captured"
        // instead of the rows some earlier cycle left in it. This is the branch
        // this comment block already calls out as "leaving any passed tape
        // stale" — now it says so in the tape itself.
        if let Some(tape) = gdn_tape.as_deref_mut() {
            tape.captured_rows = 0;
        }
        // Flush the recorded declines on the path that actually refuses.
```

**`crates/hipfire-rdna/src/dispatch/mod.rs`** — Graph replay bypasses `forward_prefill_chunk`, so the stamp has to come from per-B capture metadata. Exact twin of the field above it.

```
ANCHOR:
    pub verify_graph_lmhead_argmax: HashSet<usize>,

REPLACEMENT:
    pub verify_graph_lmhead_argmax: HashSet<usize>,
    /// Subset of `verify_graph_cache` whose captured region also writes GDN
    /// tape rows. A graph replay re-executes those memcpy nodes with no Rust
    /// code running, so this is the only way a caller can tell whether the
    /// graph it just launched filled the tape it was handed.
    pub verify_graph_gdn_tape: HashSet<usize>,
```

**`crates/hipfire-rdna/src/dispatch/mod.rs`** — Gpu struct literal (~1075) must initialise the new set.

```
ANCHOR:
            verify_graph_lmhead_argmax: HashSet::new(),

REPLACEMENT:
            verify_graph_lmhead_argmax: HashSet::new(),
            verify_graph_gdn_tape: HashSet::new(),
```

**`crates/hipfire-rdna/src/dispatch/mod.rs`** — has/mark pair, same shape and same bind_thread comment convention as the lmhead pair directly above.

```
ANCHOR:
    pub fn verify_mark_graph_lmhead_argmax(&mut self, b: usize) {
        // bind_thread: skip — pure state update
        self.verify_graph_lmhead_argmax.insert(b);
    }

REPLACEMENT:
    pub fn verify_mark_graph_lmhead_argmax(&mut self, b: usize) {
        // bind_thread: skip — pure state update
        self.verify_graph_lmhead_argmax.insert(b);
    }

    pub fn verify_graph_has_gdn_tape(&self, b: usize) -> bool {
        // bind_thread: skip — pure state query
        self.verify_graph_gdn_tape.contains(&b)
    }

    pub fn verify_mark_graph_gdn_tape(&mut self, b: usize) {
        // bind_thread: skip — pure state update
        self.verify_graph_gdn_tape.insert(b);
    }
```

**`crates/hipfire-rdna/src/dispatch/mod.rs`** — `verify_graph_destroy_all` is the only eviction site (drain at 1398). Missing this leaves a stale B claiming a freshly captured graph writes the tape after a model unload.

```
ANCHOR:
        self.verify_graph_lmhead_argmax.clear();
        self.verify_warmed_up.clear();

REPLACEMENT:
        self.verify_graph_lmhead_argmax.clear();
        self.verify_graph_gdn_tape.clear();
        self.verify_warmed_up.clear();
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — `gdn_tape` is moved into the warmup/capture/direct arms, so tape-ness must be read before the branch.

```
ANCHOR:
    let mut graph_includes_lmhead_argmax = false;

    let batch_result = if verify_graph_ok {

REPLACEMENT:
    let mut graph_includes_lmhead_argmax = false;
    // Captured before the arms below move `gdn_tape`: the graph capture arm
    // needs to record whether THIS capture included tape writes.
    let tape_present = gdn_tape.is_some();

    let batch_result = if verify_graph_ok {
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — Closes the one hole in the counter: graph replay writes the tape without executing the stamping code. Without this, graph-replay cycles fall back to FullPrefill every time (correct, ~2x slower rollback).

```
ANCHOR:
            // Replay path: kernels read pbs.tokens/pbs.positions/dn_state/
            // kv_cache contents that were freshly updated above + upstream.
            gpu.verify_graph_launch(b)?;
            Ok(())

REPLACEMENT:
            // Replay path: kernels read pbs.tokens/pbs.positions/dn_state/
            // kv_cache contents that were freshly updated above + upstream.
            gpu.verify_graph_launch(b)?;
            // No Rust ran, so `forward_prefill_chunk` did not stamp the tape.
            // The replayed graph writes tape rows iff it was CAPTURED with a
            // tape; graphs are keyed only by `b`, so ask the per-B metadata
            // rather than assuming this caller always passes one.
            if let Some(tape) = gdn_tape.as_deref_mut() {
                tape.captured_rows = if gpu.verify_graph_has_gdn_tape(b) { b } else { 0 };
            }
            Ok(())
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — Records that the graph now cached for this B contains tape-writing nodes; read back by the replay arm above.

```
ANCHOR:
                gpu.end_verify_graph_capture()?;
                if capture_lmhead_argmax {
                    gpu.verify_mark_graph_lmhead_argmax(b);
                    graph_includes_lmhead_argmax = true;
                }

REPLACEMENT:
                gpu.end_verify_graph_capture()?;
                if capture_lmhead_argmax {
                    gpu.verify_mark_graph_lmhead_argmax(b);
                    graph_includes_lmhead_argmax = true;
                }
                if tape_present {
                    gpu.verify_mark_graph_gdn_tape(b);
                }
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — One helper for all seven DDTree replay sites. NOT cfg-gated: `spec_step_ddtree_batched` is not gated while the other two are. The `forward_prefill_batch` call is copied argument-for-argument from spec_step_dflash's own fallback (~10861).

```
ANCHOR:
#[cfg(feature = "deltanet")]
pub fn spec_step_ddtree(

REPLACEMENT:
/// Did the forward that was handed this tape actually write `n_steps` rows?
///
/// The only honest answer lives in `GdnTape::captured_rows`, stamped by
/// `forward_prefill_chunk`. A caller-side copy of `prefill_batch_pbs_eligible`
/// is wrong in BOTH directions: eligibility depends on `n` (which differs at
/// every DDTree verify), the tree verify adds a `kv_asym2_tree` term the DFlash
/// predicate lacks, and `forward_prefill_batch_single_chunk_captured_opts`
/// (the hipGraph entry point) writes the tape without consulting the predicate
/// at all.
fn gdn_tape_replay_ok(captured_rows: usize, n_steps: usize) -> bool {
    captured_rows >= n_steps
}

/// Advance `target.dn_state` by `n_steps` tokens of `tokens` (starting at
/// `start_pos`), from the pre-verify DeltaNet snapshot the caller has just
/// restored.
///
/// Replays the captured GDN innovation tape when the verify forward actually
/// wrote one. When it did not — `forward_prefill_batch_with_pbs_opts` declines
/// the batched path for an unbatchable KV tier or weight dtype and silently
/// runs the tape-less per-token loop — re-run the target over the committed
/// tokens instead. That is the same `FullPrefill` fallback `spec_step_dflash`
/// takes when `verify_populates_tape` is false.
#[allow(clippy::too_many_arguments)]
fn replay_committed_dn_state(
    gpu: &mut Gpu,
    target: &mut ModelSlot,
    tape: &GdnTape,
    tokens: &[u32],
    start_pos: usize,
    n_steps: usize,
) -> HipResult<SpecRollbackReplayKind> {
    debug_assert!(
        tokens.len() >= n_steps,
        "replay_committed_dn_state: tokens {} < n_steps {n_steps}",
        tokens.len()
    );
    if gdn_tape_replay_ok(tape.captured_rows, n_steps) {
        tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            n_steps,
        )?;
        return Ok(SpecRollbackReplayKind::GdnTape);
    }
    hipfire_rdna::kernel_trace::record_fallback(
        "ddtree rollback: full-prefill re-run, verify captured no GDN tape",
        &format!(
            "captured_rows={} n_steps={n_steps} start_pos={start_pos}",
            tape.captured_rows
        ),
    );
    qwen35::forward_prefill_batch(
        gpu,
        &target.weights,
        &target.config,
        &tokens[..n_steps],
        start_pos,
        &mut target.kv_cache,
        &mut target.dn_state,
        &target.scratch,
        None,
        None,
        None,
        None,
    )?;
    
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — spec_step_ddtree (~11683). `tape_block[..accept_len+1] == committed[..accept_len+1]` on all three of its arms (full-B top-1 chain under `topk1_is_committed_prefix`, the `accept_len == 0` single-seed case, and the explicit committed-prefix arm).

```
ANCHOR:
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    gdn_tape.replay_gdn(
        gpu,
        &target.weights,
        &target.config,
        &mut target.dn_state,
        accept_len + 1,
    )?;
    // Target state is now at position + accept_len + 1. Bonus token's state
    // is deferred to next cycle's block[0], matching spec_step_dflash.

REPLACEMENT:
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    let rollback_replay = replay_committed_dn_state(
        gpu,
        target,
        gdn_tape,
        &tape_block,
        position,
        accept_len + 1,
    )?;
    // Target state is now at position + accept_len + 1. Bonus token's state
    // is deferred to next cycle's block[0], matching spec_step_dflash.
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — spec_step_ddtree return (~11710): report the kind that actually ran, so `hipfire-detect`'s rollback_parity counters (crates/hipfire-detect/src/rollback.rs) stop claiming gdn_tape on a full-prefill cycle.

```
ANCHOR:
        rollback_replay: SpecRollbackReplayKind::GdnTape,
        verify_graph_mode: tape_verify.verify_graph_mode,

REPLACEMENT:
        rollback_replay,
        verify_graph_mode: tape_verify.verify_graph_mode,
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — spec_step_ddtree_batched fast path (~12046). `fast_tape_ok` implies `spine_accept`, so committed[..accept_len+1] is exactly the linear prefix the tape rows correspond to.

```
ANCHOR:
    let hidden_rows_written;
    if fast_tape_ok {
        // Tape already captured in tree verify. Restore + replay directly.
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = big_n;

REPLACEMENT:
    let hidden_rows_written;
    let rollback_replay;
    if fast_tape_ok {
        // Tape already captured in tree verify. Restore + replay directly.
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        rollback_replay = replay_committed_dn_state(
            gpu,
            target,
            gdn_tape,
            &committed[..accept_len + 1],
            position,
            accept_len + 1,
        )?;
        hidden_rows_written = big_n;
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — spec_step_ddtree_batched Path B (~12271). `gather_accepted` only permutes rows in place, so `captured_rows` still describes them.

```
ANCHOR:
        // ── (c) Replay GDN tape on the committed-order tape.
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            n_positions,
        )?;
        hidden_rows_written = big_n;

REPLACEMENT:
        // ── (c) Replay GDN tape on the committed-order tape.
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        rollback_replay = replay_committed_dn_state(
            gpu,
            target,
            gdn_tape,
            &committed[..n_positions],
            position,
            n_positions,
        )?;
        hidden_rows_written = big_n;
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — spec_step_ddtree_batched default slow path (~12304), whose 2nd verify is non-tree and therefore can silently decline (the tree verify itself cannot — the decline branch asserts `tree_verify.is_none()`).

```
ANCHOR:
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        gdn_tape.replay_gdn(
            gpu,
            &target.weights,
            &target.config,
            &mut target.dn_state,
            accept_len + 1,
        )?;
        hidden_rows_written = tape_block.len();

REPLACEMENT:
        target_snap.restore_to(&mut target.dn_state, gpu)?;
        rollback_replay = replay_committed_dn_state(
            gpu,
            target,
            gdn_tape,
            &tape_block,
            position,
            accept_len + 1,
        )?;
        hidden_rows_written = tape_block.len();
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — spec_step_ddtree_batched return (~12353). If this anchor is not unique, disambiguate by the surrounding `accepted: accept_len,` line in that function's tail.

```
ANCHOR:
        rollback_replay: SpecRollbackReplayKind::GdnTape,
        verify_graph_mode: verify_out.verify_graph_mode,

REPLACEMENT:
        rollback_replay,
        verify_graph_mode: verify_out.verify_graph_mode,
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — Path C main-chain replay (~12544). Path C is the MOST exposed of the three: `HIPFIRE_DDTREE_PATH_C_VERIFY_GRAPH` defaults to false (speculative.rs:298-314), so it always takes `VerifyGraphPolicy::Disabled` and therefore always the eligibility-gated route.

```
ANCHOR:
    // ── 8. Drive DN state to "main-end" via tape replay. Same as Phase 1.
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    gdn_tape.replay_gdn(
        gpu,
        &target.weights,
        &target.config,
        &mut target.dn_state,
        accepted_main + 1,
    )?;

REPLACEMENT:
    // ── 8. Drive DN state to "main-end" via tape replay. Same as Phase 1.
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    let mut rollback_replay = replay_committed_dn_state(
        gpu,
        target,
        gdn_tape,
        &verify_tokens,
        position,
        accepted_main + 1,
    )?;
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — Path C step 2.1, rewind to the branch parent's pre-state (~12627).

```
ANCHOR:
            if accepted_main > 0 {
                gdn_tape.replay_gdn(
                    gpu,
                    &target.weights,
                    &target.config,
                    &mut target.dn_state,
                    accepted_main,
                )?;
            }

REPLACEMENT:
            if accepted_main > 0 {
                rollback_replay = replay_committed_dn_state(
                    gpu,
                    target,
                    gdn_tape,
                    &verify_tokens,
                    position,
                    accepted_main,
                )?;
            }
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — Path C step 3, branch-accept replay (~12707). Uses the BRANCH re-capture's rows, so tokens are `branch_chain_tokens` at `branch_start_pos`, not the main chain.

```
ANCHOR:
                gdn_tape.replay_gdn(
                    gpu,
                    &target.weights,
                    &target.config,
                    &mut target.dn_state,
                    1 + accepted_branch,
                )?;

REPLACEMENT:
                rollback_replay = replay_committed_dn_state(
                    gpu,
                    target,
                    gdn_tape,
                    &branch_chain_tokens,
                    branch_start_pos,
                    1 + accepted_branch,
                )?;
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — spec_step_ddtree_path_c return (~12818). Verify the `verify_graph_mode` field name against the file before applying — if it differs, anchor on the `SpecRollbackReplayKind::GdnTape` line at 12818 by line number.

```
ANCHOR:
        rollback_replay: SpecRollbackReplayKind::GdnTape,
        verify_graph_mode: main_verify_out.verify_graph_mode,

REPLACEMENT:
        rollback_replay,
        verify_graph_mode: main_verify_out.verify_graph_mode,
```

**`crates/hipfire-arch-qwen35/src/mtp_spec.rs`** — SIBLING BUG, same class: `tape_captured` (2727) consults `prefill_batch_pbs_eligible` but omits the `!kv_f32 && !kv_asym2_tree` term the forward adds at prefill_batch.rs:6472, so on an fp32-KV dense trunk it replays a tape the forward never wrote. The `else` branch here is already the correct full-trunk fallback. Leave the `tape_captured` use at 2736 alone — passing the tape is harmless.

```
ANCHOR:
        if tape_captured {
            // The batched verify populated the tape this cycle — cheap GDN-only replay.

REPLACEMENT:
        if state.trunk_gdn_tape.captured_rows >= advance {
            // The batched verify populated the tape this cycle — cheap GDN-only replay.
```

**`crates/hipfire-arch-qwen35/src/mtp_compose.rs`** — SIBLING BUG (~593): this path gates only on MoE-ness (503-511), which is a strictly weaker condition than the forward's. Its `else` branch is already the correct forward_prefill_batch fallback.

```
ANCHOR:
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    if let Some(tape) = gdn_tape_opt.as_deref() {
        tape.replay_gdn(

REPLACEMENT:
    target_snap.restore_to(&mut target.dn_state, gpu)?;
    if let Some(tape) = gdn_tape_opt
        .as_deref()
        .filter(|t| t.captured_rows >= accept_len + 1)
    {
        tape.replay_gdn(
```

**`crates/hipfire-arch-qwen35/src/mtp_compose.rs`** — SIBLING BUG (~1304): same MoE-only gate; the existing `else` already re-prefills `committed[..n_replay]`, so widening the condition routes an uncaptured tape into it.

```
ANCHOR:
        if accept_mtp == 0 {
            tape.replay_gdn(

REPLACEMENT:
        if accept_mtp == 0 && tape.captured_rows >= accept_dflash + 1 {
            tape.replay_gdn(
```

**`crates/hipfire-arch-qwen35/src/speculative.rs`** — The one runnable check, no GPU: it is the predicate that decides replay-vs-reforward. Sits with the existing `dflash_gdn_tape_replay_uses_actual_verify_eligibility` test in the same module.

```
ANCHOR:
    #[test]
    fn dflash_serial_rollback_replay_is_conservative_default() {

REPLACEMENT:
    #[test]
    fn gdn_tape_replay_needs_rows_the_forward_actually_wrote() {
        // A tape the forward filled for this cycle's block.
        assert!(gdn_tape_replay_ok(8, 4));
        assert!(gdn_tape_replay_ok(4, 4));
        // The bug: a declined batched forward leaves the counter at 0 while
        // base_position and Option::is_some both still say "tape present".
        assert!(!gdn_tape_replay_ok(0, 4));
        // A short capture (Path C branch re-capture) cannot cover a longer
        // replay than it wrote.
        assert!(!gdn_tape_replay_ok(3, 4));
    }

    #[test]
    fn dflash_serial_rollback_replay_is_conservative_default() {
```

