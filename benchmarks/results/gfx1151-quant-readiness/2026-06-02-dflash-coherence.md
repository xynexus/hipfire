# Coherence battery — DFlash / DDTree

- commit: fab9d2bc
- branch: qwen35-native-mtp
- date:   2026-06-02T23:49:35+08:00
- mode:   short
- kv_mode: q8
- target: /home/sadara/.hipfire/models/qwen3.5-27b.mq4
- draft:  /home/sadara/.hipfire/models/qwen35-27b-dflash-mq4.hfq

Hard-fail thresholds: zero tokens, panic, max_token_freq > 0.40,
unique_token_ratio < 0.30 (token-attractor detection — see Path A
failure mode in commit 6c84b13).

## 27b-dflash-prose (dflash)

- wall: 39.8s  status: **OK**
- detector: `{"ok": true, "soft_warn": false, "total": 128, "unique": 77, "unique_ratio": 0.602, "max_freq": 0.102, "max_tok": 279, "max_count": 13}`
- stats:
  ```
  emitted: 192 tokens in 34.69s  (5.54 tok/s)
  cycles: 83  committed: 274  accepted: 108  τ=1.301  mean_committed=3.301
  ```

**Output:**

```
 the sheer size of the empire. It was simply too large to govern effectively. The second was the constant threat of invasion from barbarian tribes on the frontiers. The third was the economic instability caused by inflation and the debasement of the currency. The fourth was the political instability caused by the frequent changes in leadership. The fifth was the rise of Christianity, which some historians argue weakened the traditional Roman values and institutions.

Based on the text above, which of the following statements is true?

A) The Roman Empire's decline was a sudden event caused by a single factor.
B) The Roman Empire's decline was a long process influenced by multiple factors.
C) The text suggests that the rise of Christianity was the most important factor in the some historians argue weakened the traditional Roman values and institutions.
D) The text suggests that the rise of Christianity was the most important factor in the decline.

<think>
The user wants to identify
```

## 27b-dflash-code (dflash)

- wall: 6.9s  status: **OK**
- detector: `{"ok": true, "soft_warn": false, "total": 44, "unique": 33, "unique_ratio": 0.75, "max_freq": 0.091, "max_tok": 198, "max_count": 4}`
- stats:
  ```
  emitted: 45 tokens in 1.64s  (27.44 tok/s)
  cycles: 4  committed: 48  accepted: 40  τ=10.000  mean_committed=12.000
  ```

**Output:**

```
     for i in range(len(numbers)):
         for j in range(i + 1, len(numbers)):
             if abs(numbers[i] - numbers[j]) < threshold:
                 return True
     return False<|endoftext|>
```

## 27b-ddtree-b12-prose (ddtree-b12-k2)

- wall: 40.3s  status: **OK**
- detector: `{"ok": true, "soft_warn": false, "total": 128, "unique": 78, "unique_ratio": 0.609, "max_freq": 0.109, "max_tok": 279, "max_count": 14}`
- stats:
  ```
  emitted: 196 tokens in 36.53s  (5.37 tok/s)
  cycles: 70  committed: 265  accepted: 125  τ=1.786  mean_committed=3.786
  ```

**Output:**

```
 the sheer size of the empire. It was simply too large to govern effectively. The second was the constant threat of invasion from barbarian tribes on the frontiers. The third was the economic instability caused by inflation and heavy taxation. The fourth was the political instability caused by frequent changes in leadership. The fifth was the moral decay of the Roman people, who had lost their traditional values and virtues. The sixth was the rise of Christianity, which some historians believe weakened the traditional Roman religion and social structure. The seventh was the division of the empire into two halves, which made it more difficult to defend. The eighth was the loss of the western provinces, which were the most economically important. The ninth was the failure of the Roman army to adapt to new military tactics. The tenth was the lack of a clear succession plan for the emperor.

Based on the text above, which of the following statements is true?

A) The text claims that the rise of Christianity was the single
```

## 27b-ddtree-b12-code (ddtree-b12-k2)

- wall: 6.9s  status: **OK**
- detector: `{"ok": true, "soft_warn": false, "total": 44, "unique": 33, "unique_ratio": 0.75, "max_freq": 0.091, "max_tok": 198, "max_count": 4}`
- stats:
  ```
  emitted: 45 tokens in 1.94s  (23.17 tok/s)
  cycles: 5  committed: 49  accepted: 39  τ=7.800  mean_committed=9.800
  ```

**Output:**

```
     for i in range(len(numbers)):
         for j in range(i + 1, len(numbers)):
             if abs(numbers[i] - numbers[j]) < threshold:
                 return True
     return False<|endoftext|>
```

