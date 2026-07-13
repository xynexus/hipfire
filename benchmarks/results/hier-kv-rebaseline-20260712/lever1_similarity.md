# Lever 1: hier merge grouping — position vs similarity (0.8B mq4+, PPL/KLD vs bf16)

| ctx | grouping | PPL | KLD/tok |
|---|---|---|---|
| 2048 | position | 27.5422 | 0.153126 |
| 2048 | similarity | 27.6030 | 0.150014 |
| 16384 | position | 18.4223 | 0.146327 |
| 16384 | similarity | 18.6434 | 0.144034 |
