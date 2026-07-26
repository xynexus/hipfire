# Lever 2: per-layer pyramid budgets vs uniform (0.8B mq4+, base fold=4 core=0.125 2-bit)

| ctx | schedule | PPL | KLD/tok |
|---|---|---|---|
| 2048 | uniform | 27.5422 | 0.153126 |
| 2048 | pyramid | 27.6100 | 0.160817 |
| 16384 | uniform | 18.3631 | 0.138399 |
| 16384 | pyramid | 18.7202 | 0.152780 |
