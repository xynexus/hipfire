# imatrix divergence report

- oracle: `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf`
- target: `qwen3.5-0.8b.tier1-b1.imatrix.gguf`
- threshold: NRMSE ≤ 0.05
- shared: 186 / oracle=186 target=186

## Per-role NRMSE

| role | n | median | max |
|---|---:|---:|---:|
| attn_gate (DeltaNet in_proj_z) | 18 | 0.9120 | 0.9800 |
| attn_k | 6 | 0.9012 | 0.9526 |
| attn_output (FullAttn o_proj) | 6 | 4.5694 | 7.8861 |
| attn_q | 6 | 0.9012 | 0.9526 |
| attn_qkv (DeltaNet in_proj_qkv) | 18 | 0.9120 | 0.9800 |
| attn_v | 6 | 0.9012 | 0.9526 |
| ffn_down (MLP) | 24 | 3.4287 | 41.3785 |
| ffn_gate (MLP) | 24 | 0.9558 | 1.0062 |
| ffn_up (MLP) | 24 | 0.9558 | 1.0062 |
| ssm_alpha (DeltaNet in_proj_a) | 18 | 0.9120 | 0.9800 |
| ssm_beta (DeltaNet in_proj_b) | 18 | 0.9120 | 0.9800 |
| ssm_out (DeltaNet out_proj) | 18 | 55.7074 | 131.6241 |

## Top 30 divergences

| tensor | k | NRMSE | cos_dist | rel_max | rel_p99 |
|---|---:|---:|---:|---:|---:|
| blk.9.ssm_out.weight | 2048 | 131.6241 | 0.8587 | 95128.8750 | 13407.9379 |
| blk.4.ssm_out.weight | 2048 | 111.8075 | 0.8252 | 434884.1250 | 19522.2434 |
| blk.12.ssm_out.weight | 2048 | 98.9206 | 0.9152 | 564840.1250 | 100600.5647 |
| blk.10.ssm_out.weight | 2048 | 96.3698 | 0.9034 | 57365.3047 | 13469.7048 |
| blk.14.ssm_out.weight | 2048 | 90.5056 | 0.8886 | 271382.5000 | 23594.5097 |
| blk.13.ssm_out.weight | 2048 | 80.5779 | 0.9244 | 82423.1406 | 24912.0422 |
| blk.8.ssm_out.weight | 2048 | 74.7008 | 0.8769 | 356511.3750 | 21926.0645 |
| blk.6.ssm_out.weight | 2048 | 65.4610 | 0.9014 | 64503.3711 | 10381.1198 |
| blk.5.ssm_out.weight | 2048 | 57.6243 | 0.8823 | 136446.9844 | 13523.1611 |
| blk.18.ssm_out.weight | 2048 | 53.7905 | 0.7964 | 453192.0312 | 36322.9688 |
| blk.0.ffn_down.weight | 3584 | 41.3785 | 0.9110 | 754.2656 | 34.2254 |
| blk.17.ssm_out.weight | 2048 | 29.7163 | 0.9670 | 145740.7188 | 47857.8175 |
| blk.0.ssm_out.weight | 2048 | 25.7748 | 0.9838 | 382469.7812 | 28288.9137 |
| blk.20.ssm_out.weight | 2048 | 18.5433 | 0.8627 | 1538530.3750 | 23521.5211 |
| blk.21.ssm_out.weight | 2048 | 17.9379 | 0.9025 | 1128594.3750 | 23974.9168 |
| blk.2.ffn_down.weight | 3584 | 17.7190 | 0.8130 | 248.7801 | 51.9628 |
| blk.4.ffn_down.weight | 3584 | 15.2187 | 0.7114 | 486.5982 | 40.1565 |
| blk.16.ssm_out.weight | 2048 | 15.1481 | 0.9177 | 322668.5312 | 33124.1543 |
| blk.1.ssm_out.weight | 2048 | 11.7428 | 0.9821 | 359121.4375 | 25858.2505 |
| blk.3.ffn_down.weight | 3584 | 9.5895 | 0.6726 | 256.6452 | 69.1049 |
| blk.2.ssm_out.weight | 2048 | 8.6403 | 0.9809 | 148400.8906 | 25209.7857 |
| blk.11.attn_output.weight | 2048 | 7.8861 | 0.8893 | 11127.9092 | 1027.8153 |
| blk.5.ffn_down.weight | 3584 | 7.2128 | 0.6141 | 141.6972 | 47.5709 |
| blk.7.ffn_down.weight | 3584 | 7.0655 | 0.7996 | 460.0992 | 33.7292 |
| blk.8.ffn_down.weight | 3584 | 5.8280 | 0.6324 | 245.7188 | 33.4598 |
| blk.9.ffn_down.weight | 3584 | 5.3567 | 0.7768 | 278.4774 | 46.5651 |
| blk.3.attn_output.weight | 2048 | 5.0271 | 0.9769 | 32722.9355 | 2888.7481 |
| blk.1.ffn_down.weight | 3584 | 4.8179 | 0.9487 | 283.5373 | 35.4448 |
| blk.11.ffn_down.weight | 3584 | 4.7902 | 0.8177 | 199.5947 | 32.2750 |
| blk.10.ffn_down.weight | 3584 | 4.7074 | 0.8016 | 226.9188 | 48.5149 |
