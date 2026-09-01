# RESULT: a drafter on the `Speculator` seam — 2026-09-02

`NgramSpeculator` is the seam's **first implementor**, and it is proven
end-to-end on real hardware: **48 tokens token-for-token identical to AR** on
`Llama-3.2-1B-Instruct--bf16`, with **τ=6.857** (84 drafts proposed, 54
accepted) — so it is genuinely speculating, not quietly degenerating to AR.

## What landed

- `crates/hipfire-specdecode-dspark/src/ngram_speculator.rs` — `impl Speculator
  for NgramSpeculator`. No forward pass, no kernel, no arch type: drafting is
  `NgramSpec::draft`, verification is `SpecTarget::verify_block`, acceptance is
  the shared `accept_greedy_prefix`. That thinness is the claim — it is what the
  trait doc promised and what the five hand-rolled acceptance loops can collapse
  into.
- `crates/hipfire-arch-llama/examples/ngram_seam_demo.rs` — drives it against a
  real `LlamaBackend` and diffs the stream against plain AR over the same
  weights.

## Design notes worth keeping

**The table-miss path is not a branch.** On a miss the spine is empty, the block
is `[seed]`, and `accept_greedy_prefix(&[], &[p0], eos)` returns one committed
token — plain AR. Writing a separate miss path would have re-introduced a second
acceptance implementation, which is the thing this work exists to remove.

**The window arithmetic is a pure function with tests.** `plan_window(spine,
picks, eos)` is split out of `step` because the index alignment (`picks[i]`
verifies `spine[i]`, because `block = [seed] ++ spine`) is where bugs live and it
needs no GPU. The GPU half of `step` is then three trait calls.

**Grammar is refused, not ignored.** `Speculator::step` takes a grammar that
"constrains both the draft and verify logits", but `SpecTarget::verify_block`
has no grammar argument — there is no way to mask the target's logits through
this seam. Ignoring it would emit constraint-violating tokens while reporting
success, so it returns a named error. **This is a gap in the seam, not in the
n-gram drafter**: the fix is a grammar hook on `verify_block`.

**No `unsafe`.** The first draft reached for pointer reborrows to get the
scratch and target borrowed at once; they are disjoint, and restructuring
`scratch_mut` into `ensure_scratch` (returning `()`) makes the safe version
compile.

## Verification, both directions

7 unit tests, no GPU, over `plan_window`: full accept, partial accept, zero
accept, empty spine, EOS-inside-window, and pick-count mismatch. The losslessness
invariant is one line — `emit == picks[..emit.len()]`, i.e. **every emitted
token is a token the target picked** — plus the length contract.

Two of those tests earn their place by having failed first:

- **EOS.** The invariant was originally written `emit.len() == accepted + 1`.
  When an accepted draft is itself EOS there is no bonus, so the window is one
  shorter. The test caught it; the helper was wrong, not the code.
- **`lossless_invariant_can_fail`** is a `#[should_panic]` negative control: a
  hand-built window emitting a token the target did not pick MUST trip the
  invariant. Without it the invariant could rot into a tautology.

**End-to-end negative control (the one that matters).** Patching `plan_window`
to accept every draft without checking it against the target:

    seam: 48 tokens in 2 windows  proposed=48 accepted=48  tau=24.000
    FAIL: spec stream != AR stream (first difference at Some(1))   rc=1

τ **improved** to 24.0 while the output became wrong — which is exactly why the
bar is token identity and not τ. The demo also refuses to pass when
`proposed == 0`, since AR-equals-AR would otherwise be a vacuous green.

## Correction to the brief

> Verify against a **qwen35** target where the existing chain path gives a
> reference token stream.

qwen35 **does not implement `SpecTarget`**. The three implementors are
`LlamaBackend`, `Gemma3Backend`, and a test double, so llama is the only real
target available to the seam today. Verification was done there instead. Putting
qwen35's `ModelSlot` on `SpecTarget` is its own piece of work, and it is what the
"collapse the five acceptance loops" payoff actually waits on.

## Not done — what the next session picks up

1. **`build_speculator` does not exist.** It is referenced in `spec.rs` doc
   comments as a registry that dispatches on arch/draft kind; nothing constructs
   a `Speculator` in the daemon. Until it exists, this implementor is reachable
   only from the example.
2. **The daemon decode loop does not route through `Speculator`.** That is the
   change that retires the hand-rolled loops, and it should not land without the
   token-identity bar above running against each arch it touches.
3. **A grammar hook on `verify_block`**, so grammar-constrained requests can use
   the seam instead of being refused.
4. **`SpecTarget` for qwen35**, which is what unlocks comparing against the
   existing chain path the brief wanted.
