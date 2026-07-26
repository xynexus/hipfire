# KV head-to-head: hierarchical vs single-tier kvarn vs asym (0.8B mq4+)

| ctx | kv system | PPL | KLD/tok |
|---|---|---|---|
| 2048 | asym4 | 27.0438 | 0.094664 |
| 2048 | asym3 | 27.8211 | 0.114926 |
| 2048 | kvarn (single-tier) | 26.3769 | 0.082880 |
| 2048 | hier hot=512 2-bit | 27.5422 | 0.153126 |
| 16384 | asym4 | 18.1503 | 0.095957 |
| 16384 | asym3 | 19.2244 | 0.124591 |
| 16384 | kvarn (single-tier) | 17.7081 | 0.085453 |
| 16384 | hier hot=2048 2-bit | 18.4087 | 0.148767 |
