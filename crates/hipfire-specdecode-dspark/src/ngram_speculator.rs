// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The n-gram drafter on the arch-generic [`Speculator`] seam.
//!
//! This is the seam's first implementor. It exists to prove the shape the trait
//! doc describes — *"let a model-free speculator (n-gram / PLD) drive any arch's
//! target without knowing its internals: the target owns ALL verify mechanics
//! while the speculator owns only policy (drafting + acceptance)"* — and it is
//! deliberately thin, because that thinness IS the claim. There is no forward
//! pass here, no kernel, no arch type: drafting is a table probe
//! ([`NgramSpec::draft`]), verification is [`SpecTarget::verify_block`], and
//! acceptance is the shared [`accept_greedy_prefix`]. Nothing in this file
//! knows whether the target is qwen35, llama, or gemma3.
//!
//! Acceptance is currently implemented five times across the tree (DFlash
//! chain, `spec_step_greedy`, MTP, lfm2moe, DDTree), all computing longest
//! matching prefix + bonus. This is where they can collapse to one.
//!
//! ## The empty-draft case needs no special path
//!
//! When the tables miss, the spine is empty and the block is just `[seed]`.
//! `accept_greedy_prefix(&[], &[p0], eos)` then returns `committed = [p0]`,
//! `accepted = 0` — one token of forward progress, which is exactly plain AR.
//! So the miss path is the general path with `spine.len() == 0`, not a branch.
//! A separate branch here would be a second acceptance implementation, which is
//! the thing this file is meant to remove.
//!
//! ## Grammar is REFUSED, not ignored
//!
//! [`Speculator::step`] takes a grammar that "constrains both the draft and
//! verify logits", but the arch-generic [`SpecTarget::verify_block`] takes no
//! grammar argument — there is no way to mask the target's logits through this
//! seam. Ignoring it would emit tokens that violate the constraint while
//! reporting success, so a grammar-constrained request is refused by name and
//! the caller must route it to AR. That is a real limitation of the seam, and
//! it should be fixed by giving `verify_block` a grammar hook rather than by
//! quietly dropping the mask here.

use crate::spec::{
    accept_greedy_prefix, PrefillOutcome, SpecAdvance, SpecGrammar, SpecScratch, SpecStep,
    SpecTarget, Speculator,
};

/// Everything one acceptance window decides, with no GPU in it.
///
/// Split out of [`Speculator::step`] on purpose: the window arithmetic — where
/// the seed sits in the block, which pick lines up with which draft, what the
/// next seed is, how far to rewind — is where the bugs live, and it is pure. The
/// GPU half of `step` is then three trait calls with nothing to get wrong.
///
/// `picks[i]` is the target's prediction after consuming `block[i]`, where
/// `block = [seed] ++ spine`. So `picks[i]` is what should follow `spine[i-1]`,
/// and `picks` lines up with `spine` at the same index: `spine[i]` is correct
/// iff it equals `picks[i]`. That off-by-one is the whole reason this is a
/// named function with tests rather than three lines inside `step`.
struct Window {
    step: SpecStep,
    /// Accepted DRAFT count — the `accept_len` for `commit_prefix`, which
    /// replays `block[..accept_len + 1]` (the seed plus the accepted drafts).
    accept_len: usize,
}

fn plan_window(spine: &[u32], picks: &[u32], eos: u32) -> Result<Window, String> {
    if picks.len() != spine.len() + 1 {
        return Err(format!(
            "NgramSpeculator: verify_block returned {} picks for a {}-token block",
            picks.len(),
            spine.len() + 1
        ));
    }
    let ga = accept_greedy_prefix(spine, picks, Some(eos));
    let next_seed = *ga
        .committed
        .last()
        .ok_or("NgramSpeculator: committed 0 tokens (would stall the decode loop)")?;
    Ok(Window {
        step: SpecStep::new(
            ga.committed.iter().copied(),
            next_seed,
            spine.len(),
            ga.accepted,
        ),
        accept_len: ga.accepted,
    })
}
use hipfire_rdna::Gpu;
use hipfire_specdecode_ngram::NgramSpec;

pub struct NgramSpeculator {
    ng: NgramSpec,
    /// Arch-specific verify scratch, allocated by the TARGET on first use.
    ///
    /// Lazily created because [`SpecTarget::new_spec_scratch`] needs a target
    /// and a `Gpu`, and the constructor has neither — the speculator is built
    /// before the decode loop borrows either.
    scratch: Option<Box<dyn SpecScratch>>,
    /// Longest spine the tables may propose, so the verify block is bounded at
    /// `1 + max_spine` (seed + spine).
    max_spine: usize,
    ctx_capacity: usize,
}

impl NgramSpeculator {
    pub fn new(ng: NgramSpec, max_spine: usize, ctx_capacity: usize) -> Self {
        Self {
            ng,
            scratch: None,
            max_spine,
            ctx_capacity,
        }
    }

    /// Hand the tables back (they outlive one request — see `NgramState`).
    pub fn into_tables(self) -> NgramSpec {
        self.ng
    }

    fn ensure_scratch(&mut self, gpu: &mut Gpu, target: &mut dyn SpecTarget) -> Result<(), String> {
        if self.scratch.is_none() {
            let block = self.max_spine + 1;
            self.scratch = Some(target.new_spec_scratch(gpu, block)?);
        }
        Ok(())
    }
}

impl Speculator for NgramSpeculator {
    fn prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        prefill_tokens: &[u32],
        prefill_start: usize,
        cache_hit: bool,
        _resume_from: Option<usize>,
        abort: &dyn Fn() -> bool,
    ) -> Result<PrefillOutcome, String> {
        // The TARGET only needs the suffix on a cache hit; the n-gram context
        // needs the WHOLE prompt either way, because drafting reads `hist` and a
        // suffix-only history would draft from a truncated context that the
        // target does not share. Two different spans, deliberately.
        let (fill, start) = if cache_hit {
            (prefill_tokens, prefill_start)
        } else {
            (prompt_tokens, 0)
        };

        self.ng.reset_sequence();
        self.ng.observe(prompt_tokens);

        // `resume_from` is a no-op: the drafter is stateless (a pure table
        // probe over `hist`, which was just rebuilt from the full prompt), so
        // there is no recurrent state to roll back to a checkpoint. This is the
        // same reason `checkpoint`/`rewind_to` keep their default no-ops.
        match target.spec_advance(gpu, fill, start, !cache_hit, abort, None)? {
            SpecAdvance::Ready { last_argmax } => Ok(PrefillOutcome::Ready {
                first_token: last_argmax,
            }),
            SpecAdvance::Aborted => Ok(PrefillOutcome::Aborted),
        }
    }

    fn step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        _emitted: &[u32],
        grammar: Option<&mut dyn SpecGrammar>,
        _temp: f32,
    ) -> Result<SpecStep, String> {
        if grammar.is_some() {
            return Err(
                "NgramSpeculator: grammar-constrained spec decode is not supported \
                        (SpecTarget::verify_block has no grammar hook, so the target's logits \
                        cannot be masked through this seam). Route this request to AR."
                    .to_string(),
            );
        }

        // Draft first, then borrow the target: `draft()` returns a slice
        // borrowed from `self.ng`, and the verify needs `&mut self` for scratch.
        let spine: Vec<u32> = self.ng.draft().map(<[u32]>::to_vec).unwrap_or_default();
        debug_assert!(spine.len() <= self.max_spine);

        let mut block = Vec::with_capacity(spine.len() + 1);
        block.push(seed);
        block.extend_from_slice(&spine);

        let eos = target.eos_token();
        // `ensure_scratch` allocates if needed and returns nothing, so the
        // scratch borrow below comes straight from `self.scratch` and never
        // overlaps the `target` borrow the verify needs.
        self.ensure_scratch(gpu, target)?;
        let scratch = self.scratch.as_mut().expect("ensured above").as_mut();
        let picks = target.verify_block(gpu, &block, position, scratch, None)?;

        let w = plan_window(&spine, &picks, eos)?;

        // Rewind the target to the accepted prefix before anything else reads
        // its state.
        {
            let scratch = self.scratch.as_mut().expect("ensured above").as_mut();
            target.commit_prefix(gpu, &block, w.accept_len, position, scratch)?;
        }

        self.ng.record_acceptance(w.accept_len);
        // Learn from what was actually committed, so `hist` stays exactly
        // "prompt + everything emitted" and the next draft probes the same
        // context the target holds.
        self.ng.observe(&w.step.emit);

        Ok(w.step)
    }

    fn reset(&mut self, _gpu: &mut Gpu) {
        // Session-local only: `reset_sequence` clears the context history and
        // leaves the learned tables, which outlive a request by design (the hot
        // tier carries ~95% of the value and grams only reach disk after
        // `promote_count` observations, so clearing them makes the feature
        // decorative).
        self.ng.reset_sequence();
    }

    fn block_size(&self) -> usize {
        self.max_spine + 1
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn free(self: Box<Self>, gpu: &mut Gpu) {
        if let Some(s) = self.scratch {
            s.free(gpu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: u32 = 999;

    /// The losslessness invariant, in one line: every emitted token is a token
    /// the TARGET picked. `emit == picks[..=accepted]` says exactly that — the
    /// accepted drafts are emitted only because they matched a pick, and the
    /// bonus IS a pick. If this holds for every window, spec decode cannot
    /// change the output relative to AR over the same target.
    fn assert_lossless(w: &Window, picks: &[u32]) {
        let emit = w.step.emit.as_slice();
        assert_eq!(
            emit,
            &picks[..emit.len()],
            "emitted a token the target did not pick"
        );
        // Length contract. Normally the window emits the accepted drafts plus
        // the bonus. When an ACCEPTED DRAFT was itself EOS there is no bonus —
        // `accept_greedy_prefix` stops there — so the window is one shorter.
        // Both cases still rewind by `accept_len`, and `commit_prefix` replays
        // `block[..accept_len + 1]` = seed + accepted drafts, which is where
        // `emit` leaves the stream either way.
        let expect_len = if emit.last() == Some(&EOS) {
            w.step.accepted
        } else {
            w.step.accepted + 1
        };
        assert_eq!(emit.len(), expect_len);
        assert_eq!(w.accept_len, w.step.accepted);
        assert_eq!(*emit.last().unwrap(), w.step.next_seed);
    }

    #[test]
    fn full_accept_emits_spine_plus_bonus() {
        let spine = [10, 11];
        let picks = [10, 11, 12];
        let w = plan_window(&spine, &picks, EOS).unwrap();
        assert_eq!(w.step.emit.as_slice(), &[10, 11, 12]);
        assert_eq!(w.step.accepted, 2);
        assert_eq!(w.step.proposed, 2);
        assert_lossless(&w, &picks);
    }

    #[test]
    fn partial_accept_takes_the_target_token_at_divergence() {
        // spine[1] guessed 11, the target picked 77 → emit the target's token.
        let spine = [10, 11];
        let picks = [10, 77, 12];
        let w = plan_window(&spine, &picks, EOS).unwrap();
        assert_eq!(w.step.emit.as_slice(), &[10, 77]);
        assert_eq!(w.step.accepted, 1);
        assert_eq!(w.step.proposed, 2, "proposed counts the whole spine");
        assert_lossless(&w, &picks);
    }

    #[test]
    fn zero_accept_still_makes_progress() {
        let spine = [10];
        let picks = [55, 56];
        let w = plan_window(&spine, &picks, EOS).unwrap();
        assert_eq!(w.step.emit.as_slice(), &[55]);
        assert_eq!(w.step.accepted, 0);
        assert_lossless(&w, &picks);
    }

    /// A table miss is the general path with an empty spine, not a branch —
    /// one token of progress, i.e. plain AR. If this ever needs its own code
    /// path, that is a second acceptance implementation creeping back in.
    #[test]
    fn empty_spine_is_plain_ar() {
        let picks = [42];
        let w = plan_window(&[], &picks, EOS).unwrap();
        assert_eq!(w.step.emit.as_slice(), &[42]);
        assert_eq!(w.step.accepted, 0);
        assert_eq!(w.step.proposed, 0);
        assert_lossless(&w, &picks);
    }

    #[test]
    fn eos_inside_the_window_stops_there() {
        let spine = [10, EOS, 12];
        let picks = [10, EOS, 12, 13];
        let w = plan_window(&spine, &picks, EOS).unwrap();
        assert_eq!(
            w.step.emit.as_slice(),
            &[10, EOS],
            "must not emit past EOS even when later drafts also match"
        );
        assert_lossless(&w, &picks);
    }

    /// The off-by-one guard. A target returning the wrong number of picks is a
    /// contract violation, and silently zipping a short `picks` against the
    /// spine would accept drafts nobody verified.
    #[test]
    fn pick_count_mismatch_is_an_error() {
        assert!(plan_window(&[10, 11], &[10, 11], EOS).is_err());
        assert!(plan_window(&[10], &[10, 11, 12], EOS).is_err());
    }

    /// Negative control for `assert_lossless` itself: if the plan ever emitted
    /// a token the target did not pick, the invariant must catch it. Built by
    /// hand because no correct `plan_window` can produce it.
    #[test]
    #[should_panic(expected = "emitted a token the target did not pick")]
    fn lossless_invariant_can_fail() {
        let picks = [10, 11, 12];
        let bad = Window {
            step: SpecStep::new([10, 11, 88], 88, 2, 2),
            accept_len: 2,
        };
        assert_lossless(&bad, &picks);
    }
}
