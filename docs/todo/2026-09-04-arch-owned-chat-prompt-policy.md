# TODO — move chat-prompt policy into the arch seam

Status: **done** — see the `feat(prompt): the arch owns whether its prompt renders
through a chat template` commit. Two things below turned out to be wrong; they are
corrected in place and noted at the end.

## Why

Whether a model's prompt is rendered by its Jinja chat template or by the hand-rolled
`prompt_frame::ChatFrame` scaffold is **one architecture's policy, written ten times into
a shared crate as an environment read**.

Every gate site sits inside a qwen35-family function — **and six more sit outside one,
reading the flag with the opposite sense**, which the original survey missed:

| file | functions |
|---|---|
| `hipfire-serving-core/src/generate.rs` | `generate_start`, `generate_multi`, `generate_dflash`, `generate_mtp` |
| `hipfire-serving-core/src/qwen35_prefill.rs` | `qwen35_materialize_batch_prefill_prompt` |

| `hipfire-serving-core/src/generate_arch.rs` | `generate_nemotron`, `generate_zaya`, `generate_llama`, `generate_minimax`, `generate_lfm2moe`, `generate_lfm2moe_dflash` |

`generate_mtp` opens with `use hipfire_arch_qwen35::speculative::{…}`. These are not
generic paths that happen to be arch-sensitive.

The two tables do not spell the same test. The qwen35 five read
`HIPFIRE_JINJA_CHAT == "1"` (opt **in**, default scaffold); the generate_arch six read
`!= "0"` (opt **out**, default Jinja). PR #413 converted only the first five, so
`jinja_chat` under `model_overrides` could not turn a llama model's rendering off at
all.

That is why the llama (arch 0) and zaya (arch 16) families are unaffected by
`HIPFIRE_JINJA_CHAT`: their paths never had the branch. Measured 2026-09-04, same request
with the flag off and on:

| model | arch | `prompt_tokens` off → on |
|---|---|---|
| `MiniCPM5--1B.oq4.25++` | 0 (llama) | 235 → 235 |
| `ZAYA1--8b.oq4++` | 16 (zaya) | 320 → 320 |
| `Qwen3.5-9B--oq4.25++` | 5 (qwen3.5) | **26 → 294** |

## What it costs

**It makes an arch-specific defect look global.** Investigating a Qwen tool-calling
problem, the conclusion reached — twice, and written into `BUGS.md` before being corrected
— was "chat templates do not render". The truth was "qwen35's paths do not render unless a
flag is set". Nothing about a backend reveals which; the only way to know is to read which
functions contain the branch.

**Ten sites are ten chances to drift.** A new qwen35 path added later silently gets the
old behaviour and no test fails.

**There is no way to ask.** `ServingBackend::caps() -> ArchCaps` is documented as
"optional fast-path capabilities the daemon checks instead of branching on `arch_id`" —
the seam exists, and prompt framing is not on it. Serving-core cannot ask a backend
whether it renders its own template.

## Shape of the change

Put the policy on the arch:

- `ChatPromptPolicy { Jinja, PlainScaffold }`, as suggested — the answer is not a
  two-state guess.
- **Not on `ArchCaps`.** `ServingBackend::caps()` has zero callers, and
  `registered_backend` is `None` on every `load_model` path but one, so a field there
  would be unreachable for exactly the families this is about. It went next to the arch
  registry that already enumerates families instead:
  `ModelArchFamily::default_chat_prompt` in `hipfire-model`, folded with the operator's
  setting by `chat_prompt_policy` and resolved once into `LoadedModel.chat_prompt`.
  Move it onto `ArchCaps` when something actually consults `caps()`.
- llama and zaya answer unconditionally; qwen35 answers from its resolved setting.
- Serving-core gets **one** branch where it had eleven, read as `m.renders_jinja()`.
- `jinja_chat` had to become `auto|off|on`. A bool cannot say "the arch decides": it
  defaults to `false`, which silently flips every non-qwen family from Jinja to the
  scaffold — a real regression the original shape would have shipped.

## Ordering — read before starting

PR #413 (`feat/chat-template-config-keys`) touches **five of the eleven sites**, replacing
the env read with `m.jinja_chat` resolved from config (`jinja_chat` / `chat_template_file`,
model-scoped, so `model_overrides` works). Land or rebase on that first, or this
refactor will conflict with every site. After it, `LoadedModel.jinja_chat` becomes the
INPUT to an arch's answer rather than a field ten call sites must remember to consult.

Related: #411 (the log said a template was adopted when it could not reach the prompt) and
#414 (`/v1/responses` dropped the tool calls it had). Both are symptoms of the same thing
being hard to see from outside the arch.

## Not in scope

Flipping the Jinja default. `templates/eval/DURABILITY-2026-06-09.md` records it as "not
yet flip-the-default ready" pending the cache-under-jinja wiring and a `| tojson`
tool-render skew fix. This is about where the decision LIVES, not what it defaults to.

## What the survey got wrong

Recorded because both errors are the same error the todo is about — an arch-specific fact
read as a global one.

1. **"Ten sites" was eleven, in two groups with opposite senses.** The survey found the
   qwen35 five because it grepped for the string PR #413 had introduced. The other six
   still read the environment, spelled `!= "0"`, and so did not match. The measured
   "llama and zaya are unaffected by `HIPFIRE_JINJA_CHAT`" was therefore right about the
   symptom and wrong about the cause: their paths do have the branch, it just already
   defaults the other way, so `=1` changes nothing there.
2. **`ArchCaps` is not reachable.** "The seam exists, and prompt framing is not on it" is
   half true — the seam is declared, but nothing calls `caps()` and almost nothing is
   loaded through `registered_backend`, so putting the answer there would have made it
   unaskable for llama, zaya, and qwen35 alike.

## Verified

gfx1103, before/after binaries, same request, `prefill_tokens`:

| model | arch | default | `jinja_chat=on` | `jinja_chat=off` |
|---|---|---|---|---|
| `qwen3.5-0.8b--oq4++` | 5 (qwen3.5) | 15 → 15 | 19 → 19 | 15 → 15 |
| `MiniCPM5-1B.oq4.25++` | 0 (llama) | 18 → 18 | 18 → 18 | **18 → 15** |

Defaults byte-identical for both families. The bold cell is the setting that used to be
ignored. `HIPFIRE_JINJA_CHAT=0` now also reaches the qwen35 paths, where it did nothing
before.
