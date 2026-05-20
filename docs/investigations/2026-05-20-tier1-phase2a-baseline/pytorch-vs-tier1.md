# imatrix divergence report

- oracle: `qwen3.5-0.8b.pytorch-oracle.imatrix.gguf`
- target: `qwen3.5-0.8b.tier1-tokparity.imatrix.gguf`
- threshold: NRMSE ≤ 0.01
- shared: 186 / oracle=186 target=187

## Per-role NRMSE

| role | n | median | max |
|---|---:|---:|---:|
| attn_gate (DeltaNet in_proj_z) | 18 | 0.9134 | 1.0112 |
| attn_k | 6 | 0.8939 | 0.9877 |
| attn_output (FullAttn o_proj) | 6 | 4.8824 | 8.2731 |
| attn_q | 6 | 0.8939 | 0.9877 |
| attn_qkv (DeltaNet in_proj_qkv) | 18 | 0.9134 | 1.0112 |
| attn_v | 6 | 0.8939 | 0.9877 |
| ffn_down (MLP) | 24 | 2.5114 | 29.4419 |
| ffn_gate (MLP) | 24 | 0.9687 | 1.0507 |
| ffn_up (MLP) | 24 | 0.9687 | 1.0507 |
| ssm_alpha (DeltaNet in_proj_a) | 18 | 0.9134 | 1.0112 |
| ssm_beta (DeltaNet in_proj_b) | 18 | 0.9134 | 1.0112 |
| ssm_out (DeltaNet out_proj) | 18 | 0.9998 | 1.0000 |

## Top 30 divergences

| tensor | k | NRMSE | cos_dist | rel_max | rel_p99 |
|---|---:|---:|---:|---:|---:|
| blk.3.ffn_down.weight | 3584 | 29.4419 | 0.8065 | 3066.8259 | 118.2071 |
| blk.5.ffn_down.weight | 3584 | 9.2817 | 0.6374 | 200.0100 | 59.2861 |
| blk.7.attn_output.weight | 2048 | 8.2731 | 0.7526 | 12395.2549 | 469.3383 |
| blk.11.attn_output.weight | 2048 | 7.6594 | 0.8068 | 6125.1099 | 764.6337 |
| blk.17.ffn_down.weight | 3584 | 6.4961 | 0.9436 | 2345.7336 | 69.2946 |
| blk.3.attn_output.weight | 2048 | 5.5053 | 0.9619 | 90209.1953 | 7376.2072 |
| blk.11.ffn_down.weight | 3584 | 5.3930 | 0.8255 | 297.7640 | 36.3297 |
| blk.4.ffn_down.weight | 3584 | 4.6697 | 0.6036 | 200.4527 | 27.5378 |
| blk.10.ffn_down.weight | 3584 | 4.5736 | 0.8183 | 390.0537 | 26.2830 |
| blk.7.ffn_down.weight | 3584 | 4.3895 | 0.7194 | 133.0682 | 26.8945 |
| blk.15.attn_output.weight | 2048 | 4.2595 | 0.7999 | 270280.0000 | 2145.1450 |
| blk.8.ffn_down.weight | 3584 | 4.1383 | 0.6170 | 98.2439 | 25.6818 |
| blk.12.ffn_down.weight | 3584 | 4.0214 | 0.8210 | 319.8792 | 35.9885 |
| blk.20.ffn_down.weight | 3584 | 3.2205 | 0.8574 | 210.0391 | 65.2526 |
| blk.16.ffn_down.weight | 3584 | 3.1793 | 0.8445 | 216.5411 | 46.1433 |
| blk.9.ffn_down.weight | 3584 | 2.7795 | 0.7126 | 89.8657 | 27.0437 |
| blk.0.ffn_down.weight | 3584 | 2.2434 | 0.4930 | 87.0308 | 11.6630 |
| blk.13.ffn_down.weight | 3584 | 2.2161 | 0.8414 | 198.1736 | 37.2124 |
| blk.23.attn_output.weight | 2048 | 2.1396 | 0.8394 | 38064.4492 | 1474.6960 |
| blk.18.ffn_down.weight | 3584 | 2.0972 | 0.9239 | 373.7254 | 78.2708 |
| blk.6.ffn_down.weight | 3584 | 2.0849 | 0.8567 | 168.5927 | 25.1772 |
| blk.15.ffn_down.weight | 3584 | 1.9468 | 0.8554 | 232.6688 | 26.5158 |
| blk.19.attn_output.weight | 2048 | 1.9252 | 0.9380 | 78793.8438 | 1766.7587 |
| blk.2.ffn_down.weight | 3584 | 1.7648 | 0.5651 | 51.8825 | 13.0980 |
| blk.14.ffn_down.weight | 3584 | 1.5037 | 0.9157 | 353.1480 | 27.2763 |
| blk.21.ffn_down.weight | 3584 | 1.5007 | 0.9615 | 313.7293 | 54.7832 |
| blk.1.ffn_down.weight | 3584 | 1.1069 | 0.8259 | 35.4221 | 11.4492 |
| blk.17.ffn_gate.weight | 1024 | 1.0507 | 0.7746 | 22.9238 | 14.7847 |
| blk.17.ffn_up.weight | 1024 | 1.0507 | 0.7746 | 22.9238 | 14.7847 |
| blk.16.ffn_gate.weight | 1024 | 1.0418 | 0.7662 | 27.9231 | 15.2573 |
