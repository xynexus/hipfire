# Deferred Feature Requests (morning review)

Net-new capability asks. Acknowledge, do not implement tonight.

| # | Reporter | One-line | Notes |
|---|---|---|---|
| 105 | 0c33 | CPU+GPU split for >VRAM models | llama.cpp parity ask (offload). Real work, not tonight. Touches #76/#77 (KV tiering, NVMe paging). |
| 92  | KotDath | DFlash draft models for MoE variants | In flight (Path C training, task #93 / sidecar pipeline). |
| 76  | kmbandy | Design: 3-tier KV cache (hot/warm/cold) | Design proposal; reference llama.cpp implementation. |
| 77  | kmbandy | Design: NVMe→VRAM demand weight paging | 229B on 32GB. SAM/ReBAR. |
| 58  | self    | Multi-GPU roadmap (PP first, TP follow) | Enh, help wanted. |
| 63  | self    | hipfire chat (interactive TUI)         | Enh, help wanted. |
| 78  | self    | Sliding-window FA from Lucebox PR #26  | Long-context decode lever, ~3.48x projected. Already filed. |
| 60  | self    | Prefill scaling regression vs llama.cpp at pp>=512 | Help wanted; needs comparator harness. |
| 61  | self    | gfx1151 Strix Halo speed baseline      | Hardware-blocked. |
| 70  | self    | gfx908 / MI100 MFMA prefill kernels    | Hardware-blocked (needs CDNA1). |
| 38  | self    | Path D stale-context overlap pipelining | 1-2wk eng, +5-15% projected. |
| 39  | self    | Path C custom DFlash draft training    | Long-running training task. |
| 40  | self    | Phase 3 prompt-shape generalized rewriting | Beyond `\n{3,}`. Good first issue. |
| 41  | self    | DDTree gfx1100 RoPE phase-delta skew   | Research-tagged. |
| 42  | self    | Mutable hipGraph (#80 follow)          | Roadmap. |
| 43  | self    | SSM intermediate persist-write (#72)   | ~1-2% lever. |
| 45  | self    | v0.1.8+ Roadmap (living index)         | Doc, not feature. |
| 89  | self    | DFlash thinking-attractor on A3B drafts | Help wanted; covered by docs in feedback memory. |
| 113 | self    | MQ3 perplexity 9B/27B vs MQ4 vs Q8     | Research; alpha->beta gate. |
| 114 | self    | MQ3 quality collapse on 0.8B / 4B      | Research; sub-9B issue. |
| 115 | self    | Lloyd-MQ3 PR (draft, help wanted)      | Already in flight. |
| 116 | self    | Lloyd-MQ3 ship gates                   | Already in flight. |
