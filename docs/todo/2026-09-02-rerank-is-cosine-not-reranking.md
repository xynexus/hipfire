# TODO — `/v1/rerank` is cosine, not reranking; the cross-encoder path exists and is unwired

Status: **not started.** The scoring primitive is already written and tested — see
"What exists" before writing anything.

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
  loader, and no arch id. This is the actual missing piece, and it is larger than the
  scorer that is already written.

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
