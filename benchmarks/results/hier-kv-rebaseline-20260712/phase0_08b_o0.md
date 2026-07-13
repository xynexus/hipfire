# Phase 0 re-baseline 0.8B (mq4+ candidate, bf16 ref PPL 24.05), offset 0 ctx 2048

| config | PPL | KLD/tok |
|---|---|---|
| plain-kvarn (hier off) | 26.3769 | 0.082880 |
| hier fold=1 hot=64 | 26.5182 | 0.082638 |
| hier fold=4 hot=64 (default) | 30.7514 | 0.320234 |
| hier fold=4 hot=64 2-bit | 31.0537 | 0.331124 |
| hier fold=4 hot=256 (new default) | 28.6815 | 0.208798 |
