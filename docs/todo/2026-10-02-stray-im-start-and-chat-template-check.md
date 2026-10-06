# TODO: stray `<|im_start|>` in replies; verify the chat template for both swarm models

Status: PARTLY DONE (2026-10-02; render test 2026-10-06) — templates checked, symptom
handled, per-model render-vs-reference test in place; streamed-reply cleaning still open
(see "Still open").
Date: 2026-10-02
Models: Qwen3.8-27B--oq4.25++ (reasoning roles), Qwen3.6-35B-A3B--oq4.25++ (coder),
both with `jinja_chat: "on"`, `max_seq: 32768` in `~/.hipfire/config.json`.

## What was seen

Corrode swarm turns on the CAE fixture (Responses API, tools declared, multi-turn
tool replay as `function_call` + `function_call_output` items):

| model | reply text (start) | when |
|---|---|---|
| 27B | `<|im_start|>` then `<tool_call>…` | tool step (several, turns 2 and 3) |
| 27B | `<|im_start|>assistant` then `<tool_call>…` | tool step |
| 27B | `<|im_start|>assistant\n\n\n# Per-crate summary …` | final answer, no tool call |
| 35B | `<|im_start|>user` — the whole reply, 1 output token | final answer of a follow-up task |

6 of ~140 replies. The model writing a turn HEADER (`<|im_start|>assistant`,
`<|im_start|>user`) as its first tokens suggests the prompt sometimes ends without
the generation prompt (`<|im_start|>assistant\n`), or ends in a state the template
did not intend (e.g. after a `<|im_end|>` with no new turn opened), so the model
opens the turn itself. The 35B case — it opens a USER turn and stops — fits a prompt
whose last rendered turn is already an assistant turn.

Already done: `206d2a8a1` strips special tokens from the text before a parsed
`<tool_call>` (`parse_inline_tool_calls`). That hides the symptom on tool steps
only; message replies (no tool call) still carry it, and Corrode replays them.

## To find out

1. Render the exact prompts. The requests are captured (Corrode → logging proxy,
   `reqs.jsonl` shape: `{body, resp}`); for each offending reply, render its body
   through the same path the server uses and check:
   - it ends with `<|im_start|>assistant\n` (plus `<think>\n\n</think>\n\n` when
     thinking is off, if the model's template does that),
   - `function_call` / `function_call_output` items render as the template's
     `<tool_call>` / `<tool_response>` turns, in order, with no empty assistant
     turn between a call and its result,
   - a trailing tool result is followed by the generation prompt, not left open.
2. Compare against the model's own `chat_template` (tokenizer_config / GGUF
   metadata) rendered by a reference Jinja (minijinja or Python jinja2) for the
   same messages — byte-identical, both models. The 27B (Qwen3.8) and the 35B
   (Qwen3.6 MoE) ship different templates; check each, not one.
3. Check `reasoning_effort: "none"` (Corrode sends it top level) is honoured on
   the Responses route — whether the empty think block is rendered — and that the
   batch path and the serial path render identically (`session_json` /
   `prefix_hash_preflight` vs the single-request path).
4. Check the stop set: `<|im_end|>` and `<|endoftext|>` stop generation on the
   batch decode path, and `<|im_start|>` as a FIRST token is not silently allowed
   past (a reply that is only a header is a failed generation, not content).
5. Usage on the batch path reported `input_tokens: 0` for the 35B reply above —
   separate known gap (batch-path usage), but note it if the render check touches it.

## Done when

- A test per model renders a captured multi-step tool conversation (tools
  declared, calls + results replayed, trailing tool result) and asserts the prompt
  matches the reference template byte for byte and ends in the generation prompt.
- A CAE swarm turn produces no reply starting with `<|im_start|>` on either model.
- If a stray header still appears with a correct prompt, it is the quantized model,
  not the template: then strip leading special tokens from all message text (not
  only before a tool call) and say so here.

## Findings (2026-10-02)

- `HIPFIRE_DEBUG_RENDER_DIR=<dir>` (new) writes the exact batch-prefill render per
  session (`qwen35_materialize_batch_prefill_prompt`). NB: the daemon runs from the
  `hipfire` CLI binary — rebuild `-p hipfire-cli --bin hipfire`, not only
  `hipfire-daemon`, or the change is not in the running daemon.
- Both offending captured requests re-rendered: the 35B (7 items) and the 27B (38
  items, ~25K tokens). Both well-formed: tools block in the system turn, XML
  `<tool_call><function=…><parameter=…>` assistant turns, `<tool_response>` user
  turns, every turn closed, and the prompt ENDS in the generation prompt
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` (`reasoning_effort: none` ->
  `max_think_tokens = 1` -> `enable_thinking = false`, honoured). So the template is
  not opening the turn for the model.
- Replays of both are clean (the 35B one three times). Every batch path picks
  tokens by argmax, so the header was a greedy near-tie that flipped under that
  run's batch composition — the known fused-vs-serial numerics gap — not a stop or
  speculation bug (the spec path truncates at a terminator inside accepted drafts;
  checked). "1 output token" on the 35B reply is the batch path's whitespace word
  count, not a token count.
- Handled at the reply: `clean_reply_text` (routes/chat.rs) drops a leading
  `<|im_start|>{role}` header and keeps what follows (the 27B's `# Per-crate summary`
  was a real answer behind one), and cuts at a later `<|im_start|>` (a hallucinated
  next turn). Applies to every non-streamed chat/Responses reply.

## Still open

- Streamed replies (`stream: true`) are not cleaned — deltas go out as generated.
- ~~The byte-for-byte render test per model against a reference Jinja render.~~ Done
  (2026-10-06): `qwen_templates_render_like_the_reference` (hipfire-prompt) renders both
  models' own templates over a multi-step tool conversation -- tools declared, two calls
  replayed, a trailing tool result; thinking off and on, and the 27B at effort `low` --
  and matches Python jinja2 (configured as HF's apply_chat_template) byte for byte,
  ending in the generation prompt. Fixtures and `render_reference.py` under
  `crates/hipfire-prompt/tests/fixtures/qwen-templates/`. It found one gap: the
  renderer crate did not declare serde_json/minijinja `preserve_order`, so on its own it
  rendered tool definitions with keys sorted (the served binary had the feature only by
  unification from hipfire-runtime). Declared now.
- Suppressing `<|im_start|>` at the logits (argmax kernels) instead of after the
  fact, if a cleaned-but-empty reply (`<|im_start|>user` -> "") shows up often.
