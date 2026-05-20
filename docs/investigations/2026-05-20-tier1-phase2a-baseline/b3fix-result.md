# imatrix divergence report

- oracle: `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf`
- target: `qwen3.5-0.8b.tier1-b3fix.imatrix.gguf`
- threshold: NRMSE ≤ 0.05
- shared: 186 / oracle=186 target=187

## Per-role NRMSE

| role | n | median | max |
|---|---:|---:|---:|
| attn_gate (DeltaNet in_proj_z) | 18 | 0.9120 | 1.0080 |
| attn_k | 6 | 0.8916 | 0.9874 |
| attn_output (FullAttn o_proj) | 6 | 4.8969 | 8.2423 |
| attn_q | 6 | 0.8916 | 0.9874 |
| attn_qkv (DeltaNet in_proj_qkv) | 18 | 0.9120 | 1.0080 |
| attn_v | 6 | 0.8916 | 0.9874 |
| ffn_down (MLP) | 24 | 2.4552 | 30.1312 |
| ffn_gate (MLP) | 24 | 0.9689 | 1.0481 |
| ffn_up (MLP) | 24 | 0.9689 | 1.0481 |
| ssm_alpha (DeltaNet in_proj_a) | 18 | 0.9120 | 1.0080 |
| ssm_beta (DeltaNet in_proj_b) | 18 | 0.9120 | 1.0080 |
| ssm_out (DeltaNet out_proj) | 18 | 0.9998 | 1.0000 |

## Top 30 divergences

| tensor | k | NRMSE | cos_dist | rel_max | rel_p99 |
|---|---:|---:|---:|---:|---:|
| blk.3.ffn_down.weight | 3584 | 30.1312 | 0.8041 | 2893.8572 | 116.6724 |
| blk.5.ffn_down.weight | 3584 | 9.2113 | 0.6313 | 160.1964 | 58.2662 |
| blk.7.attn_output.weight | 2048 | 8.2423 | 0.7435 | 8841.1426 | 445.9104 |
| blk.11.attn_output.weight | 2048 | 7.5797 | 0.8033 | 6404.3833 | 744.9789 |
| blk.17.ffn_down.weight | 3584 | 6.5604 | 0.9442 | 2205.3748 | 70.1805 |
| blk.3.attn_output.weight | 2048 | 5.5022 | 0.9613 | 65875.1328 | 6092.8460 |
| blk.11.ffn_down.weight | 3584 | 5.2080 | 0.8324 | 312.2416 | 38.6291 |
| blk.4.ffn_down.weight | 3584 | 4.6507 | 0.6046 | 194.3200 | 26.7427 |
| blk.10.ffn_down.weight | 3584 | 4.6250 | 0.8170 | 388.2491 | 27.1582 |
| blk.7.ffn_down.weight | 3584 | 4.3489 | 0.7253 | 134.9367 | 28.7843 |
| blk.15.attn_output.weight | 2048 | 4.2916 | 0.8001 | 376562.5000 | 2140.9040 |
| blk.12.ffn_down.weight | 3584 | 4.1387 | 0.8183 | 315.8339 | 35.7184 |
| blk.8.ffn_down.weight | 3584 | 4.1297 | 0.6212 | 108.0623 | 25.2127 |
| blk.20.ffn_down.weight | 3584 | 3.4251 | 0.8511 | 215.8548 | 61.7289 |
| blk.16.ffn_down.weight | 3584 | 3.1610 | 0.8418 | 224.7209 | 43.7842 |
| blk.9.ffn_down.weight | 3584 | 2.6812 | 0.7231 | 92.3777 | 27.7564 |
| blk.0.ffn_down.weight | 3584 | 2.2292 | 0.4922 | 84.2864 | 11.3785 |
| blk.13.ffn_down.weight | 3584 | 2.1399 | 0.8461 | 192.4885 | 36.7362 |
| blk.18.ffn_down.weight | 3584 | 2.1367 | 0.9232 | 374.2741 | 79.4753 |
| blk.23.attn_output.weight | 2048 | 2.1272 | 0.8383 | 32958.5977 | 1409.5573 |
| blk.6.ffn_down.weight | 3584 | 2.0783 | 0.8544 | 170.6101 | 25.9149 |
| blk.15.ffn_down.weight | 3584 | 1.9506 | 0.8642 | 223.1835 | 28.6614 |
| blk.19.attn_output.weight | 2048 | 1.9311 | 0.9378 | 62409.7891 | 1285.8168 |
| blk.2.ffn_down.weight | 3584 | 1.7683 | 0.5629 | 49.6945 | 12.6141 |
| blk.14.ffn_down.weight | 3584 | 1.5205 | 0.9165 | 351.7672 | 30.1774 |
| blk.21.ffn_down.weight | 3584 | 1.4959 | 0.9618 | 317.5335 | 53.8642 |
| blk.1.ffn_down.weight | 3584 | 1.1010 | 0.8279 | 37.7821 | 11.0209 |
| blk.17.ffn_gate.weight | 1024 | 1.0481 | 0.7780 | 22.2533 | 14.9443 |
| blk.17.ffn_up.weight | 1024 | 1.0481 | 0.7780 | 22.2533 | 14.9443 |
| blk.16.ffn_gate.weight | 1024 | 1.0398 | 0.7684 | 27.1789 | 15.5940 |
