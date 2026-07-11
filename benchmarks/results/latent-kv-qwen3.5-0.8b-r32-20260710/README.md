# Qwen3.5-0.8B rank-32 latent-KV Phase 0

Status: full-model result invalidated by
`../latent-kv-evaluator-invalidation-20260711.json`. Runtime, loader, and kernel
implementation did not proceed. Component evidence remains diagnostic-only.

The experiment uses the gated-model exception: reconstruct only the current
attention result through `R_v`, then apply the model's existing sigmoid gate
and `W_o`. Cached values and full context are never reconstructed.

Frozen thresholds before held-out evaluation:

- maximum static-vs-same-cache-oracle KLD delta: 0.05
- maximum static-vs-same-cache-oracle PPL ratio: 1.05

The following historical full-model numbers are invalid and must not be used:

- KLD delta: 0.5205436325120052 (failed)
- PPL ratio: 20.874727220324377 (failed)
- all baseline, static, and oracle logits finite

The component proxy also rejected the static-basis quality hypothesis:
rank-32 attention KLD delta was 0.9174547884985581 and static gated-output
relative error was 1.1363156930219995. Component metrics were not used for
admission.

The measured ReCalKV-style refinement was attempted for all 12 layer/GQA
groups and accepted for none: every candidate reduced raw value reconstruction
error but increased the stacked-`W_o` output error used by the acceptance rule.

`evaluator-snapshot/` preserves the exact evaluator sources hashed by
`plan.json`. The live evaluator differs only by a post-evaluation lint cleanup
that removed a redundant local `torch` import; the held-out experiment was not
resealed or rerun after validation was opened.
