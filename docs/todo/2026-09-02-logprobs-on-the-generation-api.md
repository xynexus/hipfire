# TODO — expose `logprobs` on the generation API

Status: **not started.** Small. The runtime already produces the data.

## Why

`logprobs` is accepted and silently ignored: the field appears nowhere in
`hipfire-server` or `hipfire-daemon-adapter`, so a request carrying it gets a normal
completion and no indication that the option did nothing. Silent no-ops are the worst
shape for an API option — a client cannot tell the difference between "the model is
uncertain" and "you never got the numbers".

The concrete cost, measured while building the cross-encoder reranker: scoring a
`(query, document)` pair means comparing the `yes` and `no` logits at the answer
position. With only the *sampled* token reachable, that comparison collapses to a
binary verdict — and over four near-identical documents the binary answer is all-yes or
all-no, **0 of 4 uniquely correct**, against **3 of 4** for the same model scored from
logits. The graded score is not a refinement of the yes/no answer; it **is** the signal.

That particular case is now served in-process by `/v1/rerank`, so this is no longer
blocking. It stays worth doing because the general capability is not reranking: scoring
a fixed continuation, calibration and confidence work, classification-by-logit against
an arbitrary causal LM, and perplexity from outside the process all need the same thing,
and today each would need its own endpoint.

## What exists

- `pooling::rerank_yes_no` — softmax over a yes/no logit pair.
- `kld_eval::ChunkScoredForward` / `ScoredWindow` — full-vocabulary logits per scored
  position, implemented by every `SimpleAr` backend. `hipfire-daemon`'s
  `causal_lm_yes_no_rerank` drives exactly this.
- So the runtime already computes and exposes per-position logits internally. What is
  missing is a request field, a top-k reduction, and a response shape.

## What's missing

1. **Accept and honour `logprobs` / `top_logprobs`** on `/v1/chat/completions` and
   `/v1/responses`, or reject them explicitly. Either is fine; silently ignoring is not.
2. **A top-k reduction.** Returning full-vocabulary logits per token is ~150k floats per
   position for Qwen3 — the OpenAI shape (top-k plus the sampled token's own logprob) is
   the right one.
3. **Decide the prompt-side question.** Scoring a *given* continuation (perplexity,
   classification) wants logprobs over prompt tokens, which is a different traversal from
   logprobs over generated tokens. Worth settling before the field is added, because it
   determines whether this is a sampling option or an evaluation mode.
