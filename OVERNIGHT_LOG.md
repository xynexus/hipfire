# Overnight Log 2026-05-01

## Morning Summary (filled at session end)

(empty until termination)

---

## Append-only timeline

### 2026-05-01T08:20Z | session start

- Branch `overnight/2026-05-01` created from master at `fa93b13`.
- `OVERNIGHT_PLAN.md` written and ready to commit.
- Pre-flight commits already on master:
  - `4e1a7e2` MMQ tri-state toggle.
  - `fa93b13` Windows compile-kernels.ps1 + daemon precompile parity (#112 fixed).
- Hardware available: gfx1100 (7900 XTX) on this box. No Strix Halo, no MI300X (rented), no Vega/CDNA.


### 2026-05-01T08:35Z | #111 REPLICATE

- Started serve on port 11435 (qwen3.5:9b warmup, default model).
- Sent OpenAI v1/chat/completions with single-tool, then multi-tool + stream variants targeting `qwen3.6:27b` (MQ4). Both tested at temp=0 with DFlash auto-loaded (qwen36-27b-dflash-mq4.hfq).
- **Reproduced a malformation**, not the exact one in the reporter's screenshot but the same class:
  - Reporter: `<plain>write</param> {...}` (XML-tag-shaped corruption inside <tool_call>).
  - Mine (multi-tool stream): `{"name": "write", "path": "...", "content": "..."}` instead of `{"name": "write", "arguments": {...}}`. JSON parses fine, but `tc.arguments` is undefined and the existing parseToolCalls (cli/index.ts:1541) does `JSON.stringify(tc.arguments || {})` → "{}", silently dropping all args. The tool call is structurally wrong; the downstream harness gets an empty-arg call and the file is never written. Same-shape failure mode as the reporter.
- Single-tool non-stream call produced clean nested JSON, so the malformation depends on prompt shape (multi-tool, system prompt, longer history). Matches the reporter's report ("agentic harness").
- Suspected layer: MQ4 weight quantization (FWHT rotation shifts P over structured-token positions). Same root cause class as #87. Per Rule 1 (quality regressions are bugs), root cause = quant calibration; ship fix = defensive parser; calibration retrain escalates to MANUAL_REVIEW.
