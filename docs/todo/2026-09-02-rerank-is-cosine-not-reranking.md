# TODO — `/v1/rerank` is cosine, not reranking; the cross-encoder path exists and is unwired

Status: **half delivered, and it always was.** This is not a newly found gap — it is the
unbuilt half of a planned capability, and the scoring function for it is already written.
Read "What exists" and "Where this was already planned" before writing anything.

**Correction to an earlier draft of this entry**, which presented the gap as a discovery.
It is not. `542c5c0e8` ("feat(embeddinggemma): text embedding + reranker support") added
*both* scoring modes to `pooling.rs` in one commit — "cosine reranking, **yes/no-logit
reranking (Qwen3-Reranker true/false pair)**" — so `rerank_yes_no` was written
deliberately, anticipating Qwen3-Reranker, and only the cosine half was wired up.

## Why

A client calling `/v1/rerank` reasonably expects cross-encoder reranking; that is what
the name implies and what the endpoint's position beside `/v1/embeddings` suggests. What
it gets is `rank_by_cosine` over the same bi-encoder — the identical computation the
client could perform itself from `/v1/embeddings`, and one that inherits the bi-encoder's
central limitation.

That limitation is not academic. A single embedding is a blended bag of features, so it
cannot express "matches on two axes at once": a document matching one axis strongly
outranks a document matching two axes weakly. Measured in Corrode
(`docs/harness-architecture.md`, 7e) against four near-identical C++ queue headers that
differ only in producer/consumer multiplicity and progress guarantee:

- **Nine** document representations — filename alone, doc comments, full verbatim source,
  commit messages, hand-written acronym expansion, model-generated notes at two prompt
  styles, identifier glosses — all plateau at **2/4 top-1** under cosine. Adding text
  does not help: filename alone (21 bytes) scores the same as full source (3,363 bytes).
- **Which** file fails is predicted by attribute uniqueness, not by the text. The two
  headers with no attribute unique to them are the two that fail; one of them is missed
  by **9 of 9** representations.
- Decomposing the query into axes and rank-combining lifts it to **3/4** with the same
  embedder and the same documents — evidence that composition, not description, is the
  binding constraint.

A cross-encoder is the standard answer to exactly this shape, and it could not be
measured at all, because none is served. The endpoint's name suggested otherwise, which
is the part worth fixing first: a false negative about a cross-encoder that was never
involved is a worse outcome than no endpoint.

## What exists

- **`pooling::rerank_yes_no`** — the Qwen3-Reranker cross-encoder score:
  `softmax([logit[yes], logit[no]])[0]`, numerically stable, documented with the
  true=9693 / false=2152 token ids. `hipfire-serving-core/src/pooling.rs:182`, unit
  tests at `:254` and `:258`.
- **Zero production callers.** `rerank_yes_no` is referenced only by its own module doc
  and its own tests. The scoring seam is built and unwired.
- `/v1/rerank` routes to `embeddinggemma_rerank` (`hipfire-daemon/src/lib.rs:272`),
  which has two branches — a loaded Qwen3 embedding model, else EmbeddingGemma — and
  both end in `pooling::rank_by_cosine`.
- **The models are already on this host.** `/srv/hipfire/models/` holds
  `Qwen3-Reranker-0.6B.hfa`, `-4B.hfa` and `-8B.hfa`. They are `.hfa`; everything the
  daemon serves is `.hfq`, so they are downloaded and unconverted.
- **No architecture support.** `reranker` matches nothing in `hipfire-model` or
  `hipfire-serving-core` beyond `rerank_yes_no` itself — there is no model kind, no
  loader, and no arch id.

## Demonstrated, not inferred

`Qwen3-Reranker-0.6B` was quantized to `oq8` and served, and the gap is now measured
rather than read out of the source. The quantization route is written up in
[QUANTIZE.md](../QUANTIZE.md#cross-encoder-rerankers-qwen3-reranker).

1. **It is an ordinary Qwen3.** `architectures: ["Qwen3ForCausalLM"]`,
   `model_type: qwen3`, and its `1_LogitScore` sidecar holds
   `{"true_token_id": 9693, "false_token_id": 2152}` — **the same constants
   `rerank_yes_no` already documents.** So the loading path really is the existing Qwen3
   path; it needs a prompt template and a logit read, not a new architecture.
2. **It quantizes and serves.** 595.8M params at oq8, 682 MB, all 310 tensors decode,
   and it appears in `/v1/models` without a daemon restart.
3. **`/v1/rerank` refuses it**: `rerank: loaded model is arch_id=1, expected
   embeddinggemma arch_id=19`.
4. **The generation API cannot substitute.** Driving it through
   `/v1/chat/completions` produces a sensible yes/no, but `logprobs` is silently
   ignored — the token appears nowhere in `hipfire-server` or `hipfire-daemon-adapter`
   — so only the *sampled* token is reachable. That carries no ranking signal: over four
   near-identical documents it answers **all-yes or all-no, 0 of 4 uniquely correct**.

Point 4 is the one worth keeping. The graded score is not a refinement of the yes/no
answer, it **is** the signal; argmax discards all of it. That is precisely why
`rerank_yes_no` takes logits, and it means a client cannot work around the missing
endpoint by prompting the model itself. Exposing `logprobs` on the generation API would
unblock the same measurement from outside, and is a smaller change than the loader.

## Where this was already planned

`docs/plans/2026-06-19-arch-roster-feature-matrix.md` scopes this explicitly, and rates
it cheap:

- Family **E, non-generative heads**: "**Qwen3-Embedding** (0.6/4/8B), **Qwen3-Reranker**
  (0.6/4/8B) — `Qwen3ForCausalLM` backbone, no AR loop | cheap: reuse qwen3 forward, add
  encode→pool / score head" (line 128).
- "Non-generative output heads — `encode → pool` (embedding) and pairwise `score`
  (rerank). The **cheapest** new capability: the qwen3 forward already exists; add a
  pooled/scoring output path + skip the AR loop."
- Scope framing: "the near-term-cheap wins that fit the current engine are **(E)
  embedding/rerank heads** (qwen3 forward exists)".

`docs/plans/2026-06-19-multi-family-master-plan.md` places the same item at **E6**:
"Embedding/rerank `Pool`/`Score` heads (cheap; qwen3 forward exists) validate non-AR
*output*".

So the plan named the model family, the backbone, the missing head and the cost. What
followed delivered `Pool` (for EmbeddingGemma) and left `Score` unbuilt, while writing
`Score`'s arithmetic. The evidence below is what that costs a caller today.

## What's missing

1. **A loading path for the reranker, and routing to it.** Qwen3-Reranker is a Qwen3
   causal LM that answers a yes/no question about a `(query, document)` pair — which is
   why `rerank_yes_no` takes *logits* rather than embeddings. So the work is plausibly
   "load it through the existing Qwen3 path, apply the reranker's prompt template, read
   the final-position logits, call `rerank_yes_no`" rather than a new architecture. That
   should be confirmed against the `.hfa` metadata before it is planned as small.
   Conversion to `.hfq` is a prerequisite either way.
2. **Documentation.** `/v1/rerank` appears **zero times** in `docs/*.md`. It is an
   undocumented endpoint whose behaviour differs from what its name implies.
3. **Self-description in the response.** The reply carries `model` but nothing saying
   *how* it scored. A `mode: "cosine" | "cross-encoder"` field lets a client tell which
   it got instead of inferring it from source — and lets one degrade deliberately rather
   than silently.

## Smallest useful step

(2) and (3) together, without any new model: document that `/v1/rerank` is bi-encoder
cosine today, and report that in the response. That is honest immediately, costs almost
nothing, and stops a client assuming a capability it is not getting — which is the
failure this entry exists to record. (1) is the real feature and can follow.
