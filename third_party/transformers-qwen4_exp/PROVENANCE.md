# transformers `qwen4_exp` — vendored reference implementation

Reference (oracle) implementation of `Qwen4ExpForConditionalGeneration`, the architecture
behind **Qwen3.8-Flash-Next**. Vendored because `qwen4_exp` is not in the `transformers`
release installed on this box (5.2.0), and because a port needs its oracle pinned — upstream
`main` moves.

- **Source:** `huggingface/transformers`, `src/transformers/models/qwen4_exp/`
- **Revision:** `5f8ab9bb53ec9e0c9329153d18bd825ff1db80f9` (2026-08-28)
- **Fetched:** 2026-08-29
- **License:** Apache-2.0 (same as this repo)

`modular_qwen4_exp.py` is the human-authored source; `modeling_qwen4_exp.py` is generated
from it and is the file that actually runs. Read the modular one, trust the modeling one.

Not production code and not on any build path — Python here is a comparison baseline, which
`AGENTS.md` permits. Consumed by the port scoped in
`docs/plans/2026-08-29-qwen4exp-flash-next-scope.md`.
