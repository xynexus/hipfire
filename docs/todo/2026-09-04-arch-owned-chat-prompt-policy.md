# TODO — move chat-prompt policy into the arch seam

Status: **not started.** Medium. Mechanical once the shape is agreed; the seam it belongs
on already exists.

## Why

Whether a model's prompt is rendered by its Jinja chat template or by the hand-rolled
`prompt_frame::ChatFrame` scaffold is **one architecture's policy, written ten times into
a shared crate as an environment read**.

Every gate site sits inside a qwen35-family function:

| file | functions |
|---|---|
| `hipfire-serving-core/src/generate.rs` | `generate_start`, `generate_multi`, `generate_dflash`, `generate_mtp` |
| `hipfire-serving-core/src/qwen35_prefill.rs` | `qwen35_materialize_batch_prefill_prompt` |

`generate_mtp` opens with `use hipfire_arch_qwen35::speculative::{…}`. These are not
generic paths that happen to be arch-sensitive.

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

- Add prompt framing to `ArchCaps` (a `renders_chat_template: bool`, or better a
  `chat_prompt: ChatPromptPolicy { Jinja, PlainScaffold }` so the answer is not a
  two-state guess).
- llama and zaya answer unconditionally; qwen35 answers from its resolved setting.
- Serving-core gets **one** branch where it now has ten, and the arch owns the decision it
  already effectively makes.

## Ordering — read before starting

PR #413 (`feat/chat-template-config-keys`) touches **all ten sites**, replacing the env
read with `m.jinja_chat` resolved from config (`jinja_chat` / `chat_template_file`,
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
