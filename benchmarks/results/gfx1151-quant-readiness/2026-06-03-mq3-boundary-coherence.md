# Coherence battery

- commit: fab9d2bc
- branch: qwen35-native-mtp
- date:   2026-06-03T03:21:10+08:00
- mode:   short
- binary: `./target/release/examples/daemon` (md5: `a5366b3795b9adb54555ba1356d69b56`)

Review each output for coherence (fluent English, on-topic, not stuck
in verbatim loops). Hard errors fail the gate; soft output changes do not.

## qwen3.5-0.8b.mq4 — cap

- wall: 0.8s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":12,"tok_s":156.6,"prefill_tokens":21,"prefill_ms":20.2,"prefill_tok_s":1040.7,"decode_tok_s":212.6,"ttft_ms":20.2}`
- prompt: inline (md5: `bbf2d0483ecacaa2eb7296f9c8083f95`)
- prompt_text: "What is the capital of France? Answer in one short sentence."

**Output:**

```
<think>

</think>

The capital of France is Paris.<|im_end|>
```

## qwen3.5-4b.mq4 — code

- wall: 2.9s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":125,"tok_s":62.9,"prefill_tokens":21,"prefill_ms":54.0,"prefill_tok_s":389.2,"decode_tok_s":64.7,"ttft_ms":54.0}`
- prompt: inline (md5: `e2f8e5b972f5cb8560d004062094d8a8`)
- prompt_text: "Write a one-line Python function named square that returns n*n."

**Output:**

```
<think>
The user wants a one-line Python function named `square` that returns n*n. This is straightforward - I need to create a function that takes one parameter and returns its square.

Here's the simplest way:

```python
def square(n): return n * n
```

Or with proper formatting:

```python
def square(n): return n ** 2
```

Both work, but `n * n` is more literal to what was asked. Let me provide the clean version with proper syntax.
</think>

```python
def square(n): return n * n
```<|im_end|>
```

## qwen3.5-9b.mq4 — reason

- wall: 6.4s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":213,"tok_s":42.1,"prefill_tokens":36,"prefill_ms":161.5,"prefill_tok_s":222.9,"decode_tok_s":43.5,"ttft_ms":161.5}`
- prompt: inline (md5: `fcfff9c745d58b218fc88be1639d759d`)
- prompt_text: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Premise: A farmer has 17 sheep.
    *   Event: All but 9 die.
    *   Question: How many are left?
    *   Constraint: Show brief reasoning then state the final number.

2.  **Analyze the Logic:**
    *   "All but X" means everything except X remains or survives, or in this context, everything except X dies (which is a bit ambiguous). Let's re-read carefully: "All but 9 die."
    *   Interpretation A: The total number of sheep is $N$. The number that died is $D$. The number that survived is $S$. So $N = D + S$.
    *   Statement: "All but 9 die." This means the set of sheep that died includes everyone except for the group of 9. In other words, everyone *except* those 9 died.
    *  
```

## qwen3.5-9b.mq4 — tool-call

- wall: 3.7s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":64,"tok_s":28.0,"prefill_tokens":193,"prefill_ms":790.0,"prefill_tok_s":244.3,"decode_tok_s":42.9,"ttft_ms":790.0}`
- prompt: inline (md5: `b9aee39cda7453bd8035ae4dd8c055fb`)
- prompt_text: "What does the file /tmp/fibonacci.c contain?"
- system: `@tool_call_system.txt` (md5: `4ea573cdd55e178d80cf89dfb66db0ef`)

**Output:**

```
<think>
The user is asking me to read the contents of a file located at /tmp/fibonacci.c. I should use the read function to get the contents of this file.
</think>

<tool_call>
{"name": "read", "arguments": {"path": "/tmp/fibonacci.c"}}
</tool_call><|im_end|>
```

## qwen3.5-4b.mq3 — cap-mq3-4b

- wall: 2.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":80,"tok_s":69.6,"prefill_tokens":21,"prefill_ms":48.7,"prefill_tok_s":430.8,"decode_tok_s":72.7,"ttft_ms":48.7}`
- prompt: inline (md5: `bbf2d0483ecacaa2eb7296f9c8083f95`)
- prompt_text: "What is the capital of France? Answer in one short sentence."

**Output:**

```
<think>
The user is asking for the capital of France. This is a factual question with a straightforward answer. The capital of France is Paris (specifically, the city of Paris). I should provide this information directly in one short sentence as requested.

I need to ensure the response is concise and direct, avoiding any unnecessary elaboration or multiple sentences that might violate the "one short sentence" constraint
```

## qwen3.5-9b.mq3 — reason-mq3

- wall: 7.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":300,"tok_s":50.9,"prefill_tokens":36,"prefill_ms":140.1,"prefill_tok_s":256.9,"decode_tok_s":52.2,"ttft_ms":140.1}`
- prompt: inline (md5: `fcfff9c745d58b218fc88be1639d759d`)
- prompt_text: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>
Thinking Process:

1.  **Analyze the Request:** The user is asking a riddle-like question: "A farmer has 17 sheep. All but 9 die. How many are left?" This is a classic riddle that plays with the ambiguity of "All" and "but".

2.  **Identify the Core Logic:** The phrase "All but 9 die" means that exactly *one* of the two groups (the farmer's sheep and the dead ones) is left alive. Wait, no, that's not right either. Let's look at it differently:
    - Total sheep = 17
    - Dead sheep = 9
    - Therefore, alive sheep = Total - Dead = ?

3.  **Resolve Ambiguity:** The trick is in the phrasing: "All but X die" means that *one* of them died? No, wait. Let's look at it differently:
    - Total sheep = 17 (this is fixed).
    - Dead ones (the dead ones) are not the answer to "How many are left?". Wait, no. That's not right either.

4.  **Final Count:** The total number of dead sheep is given as 9. The question asks for the number of *alive* sheep left? Wait, no, that's not right either.

5.  **Final Count:** The total number of dead
```

## qwen3.5-27b.mq3 — cap-mq3-27b

- wall: 3.3s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":12,"tok_s":11.1,"prefill_tokens":21,"prefill_ms":405.0,"prefill_tok_s":51.8,"decode_tok_s":17.6,"ttft_ms":405.0}`
- prompt: inline (md5: `bbf2d0483ecacaa2eb7296f9c8083f95`)
- prompt_text: "What is the capital of France? Answer in one short sentence."

**Output:**

```
<think>

</think>

The capital of France is Paris.<|im_end|>
```

## lfm2.5-8b-a1b.mq4 — lfm2-cap — SKIPPED (model not present)

## lfm2.5-8b-a1b.mq4 — lfm2-reason — SKIPPED (model not present)

## qwen3.5-4b.mq3-lloyd — cap-mq3-lloyd-4b — SKIPPED (model not present)

## qwen3.5-9b.mq3-lloyd — reason-mq3-lloyd-9b — SKIPPED (model not present)

## qwen3.5-4b.mq3-lloyd — long-prefill-mq3-lloyd-4b — SKIPPED (model not present)

## qwen3.5-9b.mq3-lloyd — long-prefill-mq3-lloyd-9b — SKIPPED (model not present)

## qwen3.5-9b.mq4-lloyd — reason-mq4-lloyd-9b — SKIPPED (model not present)

## qwen3.5-9b.q8f16 — long-prefill-q8-9b — SKIPPED (model not present)

## qwen3.5-9b.mq6 — reason-mq6

- wall: 3.3s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":45,"tok_s":24.6,"prefill_tokens":36,"prefill_ms":354.8,"prefill_tok_s":101.5,"decode_tok_s":30.6,"ttft_ms":354.8}`
- prompt: inline (md5: `fcfff9c745d58b218fc88be1639d759d`)
- prompt_text: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>

</think>

If all but 9 die, it means that 9 sheep survived while the rest perished. Therefore, the number of sheep left is simply the number that did not die.

**9**<|im_end|>
```

## qwen3.5-27b.mq6 — cap-mq6-27b

- wall: 12.6s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":80,"tok_s":8.6,"prefill_tokens":21,"prefill_ms":1258.1,"prefill_tok_s":16.7,"decode_tok_s":9.9,"ttft_ms":1258.1}`
- prompt: inline (md5: `bbf2d0483ecacaa2eb7296f9c8083f95`)
- prompt_text: "What is the capital of France? Answer in one short sentence."

**Output:**

```
<think>
Thinking Process:

1.  **Analyze the Request:**
    *   Question: "What is the capital of France?"
    *   Constraint: "Answer in one short sentence."

2.  **Identify the Fact:**
    *   The capital of France is Paris.

3.  **Draft the Answer:**
    *   Draft 1:
```

## qwen3.5-4b.mq3-awq-only — mq3-awq-paris — SKIPPED (model not present)

## qwen3.5-9b.mq4-awq-gptq-f2-lmhead — lmhead-awq-paris — SKIPPED (model not present)

