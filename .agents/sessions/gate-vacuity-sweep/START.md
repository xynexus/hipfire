# Session: sweep the gates asking "can this fail?"

**Blocked on:** nothing. **Est:** one session. **Value:** three independent
instances in a single night says systemic, not coincidence.

## Objective

For every gate in `tests/` and every assertion added to guard a behaviour, answer
one question: **if the thing it guards broke, would this go red?** Fix or delete
the ones that would not.

## Why — three found in one session, none related

1. **`tiny-spec-gate` demanded coverage it had itself made unreachable.** It set
   `HIPFIRE_DFLASH_CHECKPOINT_ROLLBACK=1` and asserted `replay_checkpoint > 0`,
   but the runtime refuses that path unless DeltaNet state is FP32 and said so in
   a log line the gate never read. `replay_checkpoint` was pinned at 0, so the
   assertion could not pass whatever the code did. **It had been failing on
   `master` since it landed** — and because the harness escalates to the
   coherence battery, which only fails on hard errors, the pre-commit hook still
   printed "Both gates passed". Fixed in `f212ae076` by adding
   `HIPFIRE_DN_STATE_FP16=0`; now reports "checkpoint engaged 31x, output == AR".

2. **The `rel_tol` column in `tests/tiny-state-baselines.txt` is read nowhere.**
   Header advertises `... logit_hash token_hash rel_tol`; every row sets 0;
   `grep -c '\$6'` on the gate returns 0. The parser takes `$4" "$5` and
   string-compares.

3. **A test written this session could not fail.** It sized its input as
   `DEFAULT_CALIB_SEQ_LEN * 3`, so raising the constant grew the input to match
   and the assertion held for any value. Caught only by deliberately reverting the
   constant and finding the test still green. Fixed by using a literal.

The prior art is older than this session: commit `825d3ccfc` added an assertion
precisely because an earlier gate "PASSED with the path firing ZERO times".

## Method

For each gate, in rough order of how much is trusted to it:

1. **Read what it asserts, then read what actually runs.** #1 above was a
   mismatch between an env var the gate set and a precondition the runtime
   required — visible only in stderr.
2. **Break the thing on purpose and confirm red.** Revert the fix, corrupt a
   kernel, flip the constant. A gate that stays green is the finding.
3. **Check for self-reference.** An assertion whose expected value is derived
   from the code under test proves nothing.
4. **Check the escalation path.** A red that escalates into something which only
   fails on hard errors is a red that never blocks. That is how #1 survived.
5. **Check knobs are wired.** Anything that looks like a threshold, tolerance or
   budget — confirm it is read.

## Deliverable

A short report per gate: *what it asserts · what would have to break for it to go
red · whether that was demonstrated*. Fix the cheap ones in place; file the rest.

## Note

`serve_fixture` gained a guard this session that fails when a module table is
registered but nothing paged in — because the paged-vs-resident comparison had
passed while paging nothing. That is the shape to add, not just to look for.
