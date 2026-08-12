# Outstanding work from the feat/calibrate-session review

Status: working doc, 2026-08-10. Live tracker — update the checkboxes as items
land, and delete the file once it is empty.

## Where this came from

A twelve-dimension review of `feat/calibrate-session` (43 commits, 116 files)
produced 39 raw findings, deduplicated to 32, each then put through two
independent adversarial verifiers. Most collapsed on reachability; what survived
is below, plus work that surfaced while landing the chunk-table PR (#245,
merged 2026-08-10).

Two facts shape the priorities:

- **PR #202 was closed without merging.** The branch still exists on origin, so
  nothing is lost, but the verified fixes on it are not landing by themselves.
- **Master absorbed the M6 work through a different route**, taking the branch
  as of `d9c2f0f28` — which predates every fix listed here.

## P0 — master is broken right now

### [ ] 1. ZAYA ragged calibration hard-errors on master

`origin/master` carries the `capture_by_id` guard but **none** of the six
narrowed call sites, so any ZAYA streamed calibration over a ragged corpus dies:

```
InvalidCapture("capture 1 input row stride 2730 != width 2048
                (input has 8192 elements for 3 rows; narrow it to n*k first)")
```

8192 is the 4x2048 scratch and `n` is 3 — a three-row narrow slice.

The six dense `capture_by_id` calls in `forward_position_slice` pass the whole
`max_slice_rows`-sized scratch while `n` is the slice's actual row count, so
`capture_by_id` derives a row stride of `max_slice_rows * width / s` and reads
rows `1..s-1` from stale scratch. Before the guard landed this was **silent**
Hessian and imatrix contamination; now it is a loud failure, which is better but
still broken.

Fix: narrow each capture input to `s` rows, matching what qwen35's
`prefill_batch.rs` already does and what the same function already does for
`hidden` and `router_rows`. Verified on GPU with a negative control (revert the
six sites and the job fails; restore and it passes).

Do **not** port two things from `feat/calibrate-session`:

- the `CoarseQ4Row` renumber — master already ships 48
- the original guard — it required `row_stride == k` unconditionally, which
  would reject gemma3/gemma4/cohere2. Master's `n == 1`-aware version is correct.

The parity job itself is **already on master** (`3657b90e8`) — landed without
the fix it was written to catch, and at its first revision, so it still carries
two bugs of its own: `--model` accepts an `.hfq` that streamed calibration can
never read, and the verdict keys off the comparator's exit status, which cannot
distinguish float reassociation from corruption. Both fixes come across with
this change.

Done when: the six sites are narrowed on master and the parity job's own two
fixes are in.

**GPU re-proof deferred.** The identical six-site change was verified on GPU on
`feat/calibrate-session`, with a negative control (revert the sites, the job
fails at the guard; restore, it passes). The only new variable here is master's
guard, which is strictly *more* permissive than the one that verification ran
against — `n == 1 ? row_stride >= k : row_stride == k` versus an unconditional
`==` — and the fix makes every dense site pass exactly `n*k`, so it satisfies
the strict arm. It cannot be rejected by a weaker predicate. Re-run
`benchmarks/calib/zaya-ragged-slice-parity.sh` when the GPU frees up to close
this out properly; the box was busy with a 122B `oq4.25++` LDLQ quantize.

## P1 — silent wrongness

### [ ] 2. Attention LDS under-allocation produces wrong output

`crates/hipfire-rdna/src/dispatch/attention.rs`. `GRAPH_CTX_CAP.min(max_seq)`
under-allocates `scores[]` whenever `active_stream` is set — which is ordinary
speculative-decode and EP setup, not a graph-capture flag.

Exposure is narrower than it first looks: qwen3.5 with an operator-chosen
`max_seq` between 8193 and 16128, the default fp32 KV, after DFlash/MTP has run
once. Below 8193 the sizing is byte-identical to before; above ~16128 the old
code failed the launch outright, so that configuration was never usable.

Ranked first among the P1s because the failure is *silent wrong output* rather
than an error. Fix belongs either at the dispatch site (reject when
`seq_len_hint > effective_seq`) or by gating on capture mode rather than
`active_stream`.

### [ ] 3. Abort ignored on the qwen35-vl and dots-ocr paths

`crates/hipfire-serving-core/src/generate_vl.rs`. Both run hand-rolled decode
loops that reach none of the three cancellation hooks, so an abort is accepted
and then ignored. gemma3-vl/medgemma is fine — it delegates to the generic
`decode_loop`. Not a regression (abort was a dead wire variant before), but the
client now gets no reply where it used to get an explanatory error.

### [ ] 4. `cancel::clear()` drops an abort for a still-queued request

`crates/hipfire-daemon/src/main.rs`. The executor clears the pending-stop slot as
it takes up each frame, so an abort that arrived while its target was still
queued is wiped the moment that request starts.

Found independently by five of the twelve reviewers, which is why it is here
despite low reachability today (nothing in-repo passes `--listen`). The id guard
in `stop_kind` already prevents the stale-abort case the clear was written for,
so the clear can probably just go.

### [ ] 5. `collector-status.md` documents a corpus the tree does not ship

`docs/calibration/collector-status.md:96` claims 2713 records at
english 38 / code 21 / chinese 15.5 / japanese 15.5 / math 10. The shipped
`benchmarks/calib/calib-multi-labelled.jsonl` is 1449 records at
35 / 20 / 10 / 10 / 25, at every prefix length. The section's headline 5.27x
math-lift figure was measured on a corpus that is not in the tree.

Docs-only, no code impact — but it is the only description an operator has.

## P2 — unreviewed and unwatched

### [ ] 6. The M6 induction port has never been reviewed

Roughly 7,000 lines — the load-bearing induction Python ported to Rust, plus
calibrate/induction as a daemon session — landed on `feat/calibrate-session`
*after* the twelve-reviewer sweep ran, and is now on master. Nobody has looked
at it.

### [ ] 7. A failing test nobody runs

`variable_layout_formats_have_no_fixed_block_bytes` fails on master:
`block_bytes(Oq2G256)` returns `Some(66)` while the test lists Oq2G256 as
variable-layout. Both contributing commits (`88683d270`, `4ca8f4e18`) are
ancestors of master.

It goes unnoticed because `cargo test -p hipfire-quant-format` is not in
`tests/no-gpu-ci.sh`. Either the test is stale (the oq2/W2A8 commit gave Oq2 a
real fixed geometry and the list was not updated) or `Some(66)` is wrong and Oq2
is row-dependent like `Oq8G256RowPadded`. Resolving it needs someone who knows
which; guessing changes on-disk sizing.

Whichever way it goes, add the crate to the gate so the next one is caught.

## Housekeeping

### [ ] 8. Prune stale worktrees

`git worktree list` carries a dangling `m6-budget-and-locks` entry, marked
locked, whose directory was deleted mid-session, plus several `prunable` agent
worktrees. `git worktree prune` clears the prunable ones; the locked entry needs
unlocking first. `/home/sadara/hf-worktrees/chunk-table` can go now that #245 has
merged.

## Deferred by choice

Not bugs, recorded so they are not rediscovered:

- **`UNIT` reduction is not worth doing.** Measured across ZAYA1-8B's 2483
  tensors, exactly one exceeds it (`embed_tokens`, 1074 MB) and the second
  largest is 16.8 MB, so the effective repair unit is already the tensor at
  8.4 MB median.
- **Signing the `.hfa` index.** The chunk table is shaped for it: sign the
  stored index bytes verbatim (the trailer makes them a contiguous range), and
  no whole-archive SHA is then needed. Blocked on nothing but a decision about
  key management.
- **hfq to safetensors export.** Would have saved a two-hour detour when a
  `.hfq` turned out to be unusable as a `calibrate --model` source. Belongs in
  `hipfire-coexistence`. Note that canonicalization is many-to-one and
  index-collapsing, so a faithful export has to reconstruct the raw Megatron
  half-layer split.
- **Content-defined chunking** instead of fixed windows, for cross-version
  dedup that fixed windows cannot give.
