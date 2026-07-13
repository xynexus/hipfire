# T1: hier cold-merge ranked by TriAttn vs vnorm (0.8B mq4+, PPL/KLD vs bf16)

| ctx | hier importance | PPL | KLD/tok |
|---|---|---|---|
| 2048 | vnorm | 27.5420 | 0.153126 |
| 2048 | triattn | 29.6678 | 0.205858 |
| 16384 | vnorm | 18.3771 | 0.145149 |
| 16384 | triattn | 20.2056 | 0.193289 |
