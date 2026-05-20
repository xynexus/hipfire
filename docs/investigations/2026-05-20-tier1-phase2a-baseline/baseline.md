# imatrix divergence report

- oracle: `benchmarks/quality-baselines/refs/qwen3.5-0.8b-bf16.imatrix.gguf`
- target: `qwen3.5-0.8b.tier1-v1.imatrix.gguf`
- threshold: NRMSE ≤ 0.05
- shared: 186 / oracle=186 target=186

## Per-role NRMSE

| role | n | median | max |
|---|---:|---:|---:|
| attn_gate (DeltaNet in_proj_z) | 18 | 0.9999 | 1.0000 |
| attn_k | 6 | 0.9999 | 1.0000 |
| attn_output (FullAttn o_proj) | 6 | 0.9999 | 1.0000 |
| attn_q | 6 | 0.9999 | 1.0000 |
| attn_qkv (DeltaNet in_proj_qkv) | 18 | 0.9999 | 1.0000 |
| attn_v | 6 | 0.9999 | 1.0000 |
| ffn_down (MLP) | 24 | 1.0000 | 1.0000 |
| ffn_gate (MLP) | 24 | 0.9999 | 1.0000 |
| ffn_up (MLP) | 24 | 0.9999 | 1.0000 |
| ssm_alpha (DeltaNet in_proj_a) | 18 | 0.9999 | 1.0000 |
| ssm_beta (DeltaNet in_proj_b) | 18 | 0.9999 | 1.0000 |
| ssm_out (DeltaNet out_proj) | 18 | 0.9979 | 1.0000 |

## Top 30 divergences

| tensor | k | NRMSE | cos_dist | rel_max | rel_p99 |
|---|---:|---:|---:|---:|---:|
| blk.22.ffn_down.weight | 3584 | 1.0000 | 0.9523 | 1.0000 | 1.0000 |
| blk.19.ffn_down.weight | 3584 | 1.0000 | 0.9636 | 1.0000 | 1.0000 |
| blk.14.ffn_down.weight | 3584 | 1.0000 | 0.8379 | 1.0000 | 1.0000 |
| blk.13.ffn_down.weight | 3584 | 1.0000 | 0.8128 | 1.0000 | 1.0000 |
| blk.1.ffn_down.weight | 3584 | 1.0000 | 0.8559 | 1.0000 | 1.0000 |
| blk.18.ffn_down.weight | 3584 | 1.0000 | 0.8423 | 1.0000 | 1.0000 |
| blk.0.ffn_down.weight | 3584 | 1.0000 | 0.4881 | 1.0000 | 1.0000 |
| blk.21.ffn_down.weight | 3584 | 1.0000 | 0.9206 | 1.0000 | 1.0000 |
| blk.2.ffn_down.weight | 3584 | 1.0000 | 0.6261 | 1.0000 | 1.0000 |
| blk.23.ffn_down.weight | 3584 | 1.0000 | 0.7913 | 1.0000 | 1.0000 |
| blk.6.ffn_down.weight | 3584 | 1.0000 | 0.7846 | 1.0000 | 1.0000 |
| blk.15.ffn_down.weight | 3584 | 1.0000 | 0.7448 | 1.0000 | 1.0000 |
| blk.3.ffn_down.weight | 3584 | 1.0000 | 0.4818 | 1.0000 | 1.0000 |
| blk.9.ffn_down.weight | 3584 | 1.0000 | 0.6007 | 1.0000 | 1.0000 |
| blk.7.ffn_down.weight | 3584 | 1.0000 | 0.6072 | 1.0000 | 1.0000 |
| blk.17.ffn_down.weight | 3584 | 1.0000 | 0.7440 | 1.0000 | 1.0000 |
| blk.10.ffn_down.weight | 3584 | 1.0000 | 0.6873 | 1.0000 | 1.0000 |
| blk.11.ffn_down.weight | 3584 | 1.0000 | 0.6463 | 1.0000 | 1.0000 |
| blk.12.ffn_down.weight | 3584 | 1.0000 | 0.6731 | 1.0000 | 1.0000 |
| blk.5.ffn_down.weight | 3584 | 1.0000 | 0.4070 | 1.0000 | 1.0000 |
| blk.16.ffn_down.weight | 3584 | 1.0000 | 0.6288 | 1.0000 | 1.0000 |
| blk.4.ffn_down.weight | 3584 | 1.0000 | 0.5983 | 1.0000 | 1.0000 |
| blk.8.ffn_down.weight | 3584 | 1.0000 | 0.4495 | 1.0000 | 1.0000 |
| blk.20.ffn_down.weight | 3584 | 1.0000 | 0.6528 | 1.0000 | 1.0000 |
| blk.2.attn_gate.weight | 1024 | 1.0000 | 0.7812 | 1.0000 | 0.9999 |
| blk.2.attn_qkv.weight | 1024 | 1.0000 | 0.7812 | 1.0000 | 0.9999 |
| blk.2.ssm_alpha.weight | 1024 | 1.0000 | 0.7812 | 1.0000 | 0.9999 |
| blk.2.ssm_beta.weight | 1024 | 1.0000 | 0.7812 | 1.0000 | 0.9999 |
| blk.1.attn_gate.weight | 1024 | 1.0000 | 0.7824 | 1.0000 | 0.9999 |
| blk.1.attn_qkv.weight | 1024 | 1.0000 | 0.7824 | 1.0000 | 0.9999 |
