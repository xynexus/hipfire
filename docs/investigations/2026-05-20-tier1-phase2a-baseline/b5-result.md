# imatrix divergence report

- oracle: `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf`
- target: `qwen3.5-0.8b.tier1-b5.imatrix.gguf`
- threshold: NRMSE ≤ 0.05
- shared: 186 / oracle=186 target=187

## Per-role NRMSE

| role | n | median | max |
|---|---:|---:|---:|
| attn_gate (DeltaNet in_proj_z) | 18 | 0.9120 | 1.0080 |
| attn_k | 6 | 0.8916 | 0.9874 |
| attn_output (FullAttn o_proj) | 6 | 4.8970 | 8.2431 |
| attn_q | 6 | 0.8916 | 0.9874 |
| attn_qkv (DeltaNet in_proj_qkv) | 18 | 0.9120 | 1.0080 |
| attn_v | 6 | 0.8916 | 0.9874 |
| ffn_down (MLP) | 24 | 2.4552 | 30.1310 |
| ffn_gate (MLP) | 24 | 0.9689 | 1.0481 |
| ffn_up (MLP) | 24 | 0.9689 | 1.0481 |
| ssm_alpha (DeltaNet in_proj_a) | 18 | 0.9120 | 1.0080 |
| ssm_beta (DeltaNet in_proj_b) | 18 | 0.9120 | 1.0080 |
| ssm_out (DeltaNet out_proj) | 18 | 0.9998 | 1.0000 |

## Top 30 divergences

| tensor | k | NRMSE | cos_dist | rel_max | rel_p99 |
|---|---:|---:|---:|---:|---:|
| blk.3.ffn_down.weight | 3584 | 30.1310 | 0.8041 | 2893.9819 | 116.6666 |
| blk.5.ffn_down.weight | 3584 | 9.2113 | 0.6313 | 160.1877 | 58.2904 |
| blk.7.attn_output.weight | 2048 | 8.2431 | 0.7435 | 8836.3486 | 445.9257 |
| blk.11.attn_output.weight | 2048 | 7.5795 | 0.8033 | 6405.5146 | 744.7995 |
| blk.17.ffn_down.weight | 3584 | 6.5626 | 0.9442 | 2206.1313 | 70.1785 |
| blk.3.attn_output.weight | 2048 | 5.5022 | 0.9613 | 65879.9922 | 6091.0869 |
| blk.11.ffn_down.weight | 3584 | 5.2091 | 0.8324 | 312.3380 | 38.6218 |
| blk.4.ffn_down.weight | 3584 | 4.6506 | 0.6047 | 194.3525 | 26.7275 |
| blk.10.ffn_down.weight | 3584 | 4.6248 | 0.8170 | 388.3895 | 27.1640 |
| blk.7.ffn_down.weight | 3584 | 4.3489 | 0.7253 | 134.9421 | 28.8136 |
| blk.15.attn_output.weight | 2048 | 4.2917 | 0.8000 | 375802.9688 | 2141.0046 |
| blk.12.ffn_down.weight | 3584 | 4.1394 | 0.8183 | 316.1810 | 35.7641 |
| blk.8.ffn_down.weight | 3584 | 4.1297 | 0.6212 | 108.0737 | 25.2287 |
| blk.20.ffn_down.weight | 3584 | 3.4265 | 0.8512 | 216.1169 | 61.8075 |
| blk.16.ffn_down.weight | 3584 | 3.1616 | 0.8418 | 224.6645 | 43.8140 |
| blk.9.ffn_down.weight | 3584 | 2.6812 | 0.7230 | 92.3692 | 27.7520 |
| blk.0.ffn_down.weight | 3584 | 2.2292 | 0.4922 | 84.2864 | 11.3785 |
| blk.13.ffn_down.weight | 3584 | 2.1395 | 0.8462 | 192.7523 | 36.7358 |
| blk.18.ffn_down.weight | 3584 | 2.1371 | 0.9232 | 374.5264 | 79.4289 |
| blk.23.attn_output.weight | 2048 | 2.1273 | 0.8383 | 32961.1406 | 1411.4921 |
| blk.6.ffn_down.weight | 3584 | 2.0784 | 0.8544 | 170.6114 | 25.9045 |
| blk.15.ffn_down.weight | 3584 | 1.9508 | 0.8642 | 223.0761 | 28.6829 |
| blk.19.attn_output.weight | 2048 | 1.9316 | 0.9378 | 62391.3672 | 1285.2967 |
| blk.2.ffn_down.weight | 3584 | 1.7683 | 0.5629 | 49.6945 | 12.6141 |
| blk.14.ffn_down.weight | 3584 | 1.5204 | 0.9164 | 351.6410 | 30.1514 |
| blk.21.ffn_down.weight | 3584 | 1.4962 | 0.9618 | 317.2030 | 53.8854 |
| blk.1.ffn_down.weight | 3584 | 1.1010 | 0.8279 | 37.7821 | 11.0209 |
| blk.17.ffn_gate.weight | 1024 | 1.0481 | 0.7781 | 22.2357 | 14.9443 |
| blk.17.ffn_up.weight | 1024 | 1.0481 | 0.7781 | 22.2357 | 14.9443 |
| blk.16.ffn_gate.weight | 1024 | 1.0399 | 0.7684 | 27.1824 | 15.5977 |
