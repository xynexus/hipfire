# Long-ctx merge penalty 0.8B (mq4+), ctx=16384 hot=512 2-bit, offset 0

bf16 ref PPL=16.1047

| config | PPL | KLD/tok |
|---|---|---|
| hier fold=1 hot=512 | 18.1236 | 0.127820 |
| hier fold=4 hot=512 | 20.0702 | 0.246740 |
| hier fold=8 hot=512 | 20.6363 | 0.266686 |
