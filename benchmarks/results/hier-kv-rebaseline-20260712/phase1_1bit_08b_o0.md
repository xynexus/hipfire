=== parity_kv_hier oracle gate ===
parity_kv_hier: PASS

# Phase 1 1-bit cold probe 0.8B (mq4+, bf16 ref PPL 24.05, offset 0 ctx 2048)

| config | PPL | KLD/tok |
|---|---|---|
| fold=1 hot=64 2-bit | 26.9163 | 0.108828 |
| fold=1 hot=64 1-bit | 191.9419 | 1.959850 |
| fold=4 hot=64 1-bit K | 42.9838 | 0.566032 |
| fold=4 hot=64 1-bit K+V | 42.9838 | 0.566032 |
