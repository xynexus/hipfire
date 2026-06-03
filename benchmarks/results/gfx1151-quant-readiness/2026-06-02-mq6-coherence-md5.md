# Coherence battery

- commit: fab9d2bc
- branch: qwen35-native-mtp
- date:   2026-06-02T23:45:39+08:00
- mode:   full
- binary: `./target/release/examples/daemon` (md5: `a5366b3795b9adb54555ba1356d69b56`)

Review each output for coherence (fluent English, on-topic, not stuck
in verbatim loops). Hard errors fail the gate; soft output changes do not.

## qwen3.5-0.8b.mq4 — cap

- wall: 0.9s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":12,"tok_s":149.2,"prefill_tokens":21,"prefill_ms":21.5,"prefill_tok_s":975.2,"decode_tok_s":203.8,"ttft_ms":21.5}`
- prompt: inline (md5: `bbf2d0483ecacaa2eb7296f9c8083f95`)
- prompt_text: "What is the capital of France? Answer in one short sentence."

**Output:**

```
<think>

</think>

The capital of France is Paris.<|im_end|>
```

## qwen3.5-4b.mq4 — code

- wall: 3.2s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":125,"tok_s":57.7,"prefill_tokens":21,"prefill_ms":73.6,"prefill_tok_s":285.1,"decode_tok_s":59.7,"ttft_ms":73.6}`
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

- wall: 6.8s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":213,"tok_s":39.6,"prefill_tokens":36,"prefill_ms":249.0,"prefill_tok_s":144.6,"decode_tok_s":41.5,"ttft_ms":249.0}`
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

- wall: 4.1s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":64,"tok_s":26.2,"prefill_tokens":193,"prefill_ms":912.9,"prefill_tok_s":211.4,"decode_tok_s":41.9,"ttft_ms":912.9}`
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

## qwen3.5-9b.mq3 — reason-mq3

- wall: 7.1s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":300,"tok_s":50.6,"prefill_tokens":36,"prefill_ms":145.6,"prefill_tok_s":247.3,"decode_tok_s":51.9,"ttft_ms":145.6}`
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

- wall: 3.9s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":12,"tok_s":10.3,"prefill_tokens":21,"prefill_ms":457.6,"prefill_tok_s":45.9,"decode_tok_s":17.0,"ttft_ms":457.6}`
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

- wall: 3.2s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":45,"tok_s":25.6,"prefill_tokens":36,"prefill_ms":285.4,"prefill_tok_s":126.1,"decode_tok_s":30.5,"ttft_ms":285.4}`
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

- wall: 13.6s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":80,"tok_s":8.4,"prefill_tokens":21,"prefill_ms":1377.7,"prefill_tok_s":15.2,"decode_tok_s":9.8,"ttft_ms":1377.7}`
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

## qwen3.5-35b-a3b.mq4 — moe-sheep

- wall: 5.1s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":47,"tok_s":30.8,"prefill_tokens":36,"prefill_ms":651.3,"prefill_tok_s":55.3,"decode_tok_s":53.8,"ttft_ms":651.3}`
- prompt: inline (md5: `fcfff9c745d58b218fc88be1639d759d`)
- prompt_text: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>

</think>

The phrase "all but 9" means that every sheep died except for 9. Therefore, the number of sheep remaining is simply the 9 that did not die.

**Final Answer:** 9<|im_end|>
```

## qwen3.6-35b-a3b.mq4 — moe36-sheep

- wall: 18.3s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":756,"tok_s":50.8,"prefill_tokens":36,"prefill_ms":644.6,"prefill_tok_s":55.8,"decode_tok_s":53.1,"ttft_ms":644.6}`
- prompt: inline (md5: `fcfff9c745d58b218fc88be1639d759d`)
- prompt_text: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>
Here's a thinking process:

1.  **Analyze User Input:**
   - Statement: "A farmer has 17 sheep. All but 9 die."
   - Question: "How many are left?"
   - Requirement: "Show brief reasoning then state the final number."

2.  **Identify Key Phrase/Trick:**
   - The phrase "All but 9 die" is a classic riddle/trick question.
   - "All but X" means everything except X happens to the subject. So, if all but 9 die, it means 9 survive (are left alive).

3.  **Calculate/Reason:**
   - Total sheep = 17
   - "All but 9 die" → The number that does NOT die = 9
   - Therefore, the number left alive = 9

4.  **Formulate Response (per requirements):**
   - Need brief reasoning first.
   - Then state the final number clearly.

   Draft: The phrase "all but 9 die" means that every sheep except for nine passed away, leaving exactly nine alive. 
   Final number: 9

5.  **Self-Correction/Verification:**
   - Does it match the requirement? Yes, brief reasoning + final number.
   - Is it logically sound? Yes, classic riddle interpretation is correct. "All but X" = X remain.
   - No extra fluff needed.

6.  **Output Generation:** (matches draft)✅
   "The phrase 'all but 9 die' means that every sheep except for nine passed away, so exactly nine remain alive."
   Final number: 9" -> Wait, I'll format it cleanly as requested.
   Reasoning: The phrase "all but 9 die" means that every sheep except for nine died, leaving exactly nine alive.
   Final number: 9✅
   All steps verified. Output matches response. 
   Proceeds. 
   [Output Generation] -> "The phrase 'all but 9 die' means that every sheep except for nine passed away, so exactly nine remain alive.\n\nFinal number: 9" (or similar) -> I'll keep it concise as requested.✅
   Final check of the prompt: "Show brief reasoning then state the final number." -> I will follow exactly.
   Reasoning: The phrase "all but 9 die" means that every sheep except for nine died, leaving exactly nine alive.
   Final number: 9
   Matches perfectly.✅
   Output matches response. 
   [Done.] 
   *Self-Correction/Refinement during thought*: 
   Wait, is there any ambiguity? "All but 9 die" -> 17 - (17-9) = 9 left. Yes, standard interpretation is correct. Some might overthink it as a math problem (17-9=8), but the phrasing "all but X" is a well-known linguistic construct meaning X remain. I'll stick with 9 and explain clearly.
   All good.✅
   Output Generation. 
   [Response Text]: The phrase "all but 9 die" means that every sheep except for nine passed away, so exactly nine remain alive.  
Final number: 9
   Matches requirement perfectly.✅
   Proceeds. 
   [Output Generation] (done)
</think>

The phrase "all but 9 die" means that every sheep except for nine passed away, leaving exactly nine alive.  

Final number: 9<|im_end|>
```

## qwen3.5-35b-a3b.mq6 — moe-mq6-sheep

- wall: 5.2s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":50,"tok_s":44.0,"prefill_tokens":36,"prefill_ms":157.3,"prefill_tok_s":228.8,"decode_tok_s":51.1,"ttft_ms":157.3}`
- prompt: inline (md5: `fcfff9c745d58b218fc88be1639d759d`)
- prompt_text: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>

</think>

The phrase "all but 9 die" means that every sheep died except for 9 of them. Therefore, the number of sheep remaining is exactly those 9 that did not die.

**Final Answer:** 9<|im_end|>
```

## qwen3.6-35b-a3b.mq6 — moe36-mq6-sheep

- wall: 11.0s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":349,"tok_s":49.2,"prefill_tokens":36,"prefill_ms":154.2,"prefill_tok_s":233.5,"decode_tok_s":50.3,"ttft_ms":154.2}`
- prompt: inline (md5: `fcfff9c745d58b218fc88be1639d759d`)
- prompt_text: "A farmer has 17 sheep. All but 9 die. How many are left? Show brief reasoning then state the final number."

**Output:**

```
<think>
Here's a thinking process:

1.  **Analyze the User Input:**
   - Statement: "A farmer has 17 sheep. All but 9 die. How many are left?"
   - Request: "Show brief reasoning then state the final number."

2.  **Identify Key Phrase/Trick:**
   - The phrase "All but 9 die" is a classic riddle/trick question.
   - "All but X" means everything except X dies, so X survives.
   - Therefore, if all but 9 die, then 9 are left alive.

3.  **Formulate Reasoning (Brief):**
   - The phrase "all but 9 die" means that every sheep except for 9 died.
   - Therefore, the number of sheep remaining is exactly the number that did not die: 9.

4.  **State Final Number:**
   - Final answer: 9

5.  **Check for Consistency/Clarity:**
   - Does it match the request? Yes, brief reasoning + final number.
   - Is it mathematically/logically sound? Yes, standard interpretation of "all but X".
   - No hidden tricks beyond the classic riddle format.

6.  **Draft Response:**
   Reasoning: The phrase "all but 9 die" means that every sheep except for nine died, leaving those nine alive. 
   Final number: 9✅
</think>

Reasoning: The phrase "all but 9 die" means that every sheep except for nine died, so those nine remain alive.  
Final number: 9<|im_end|>
```

## qwen3.6-27b.mq4 — tool-call-27b

- wall: 8.9s  status: **OK**
- stats: `{"type":"done","id":"r1","tokens":28,"tok_s":4.6,"prefill_tokens":193,"prefill_ms":4137.3,"prefill_tok_s":46.6,"decode_tok_s":13.9,"ttft_ms":4137.3}`
- prompt: inline (md5: `b9aee39cda7453bd8035ae4dd8c055fb`)
- prompt_text: "What does the file /tmp/fibonacci.c contain?"
- system: `@tool_call_system.txt` (md5: `4ea573cdd55e178d80cf89dfb66db0ef`)

**Output:**

```
<think>

</think>

<tool_call>
{"name": "read", "arguments": {"path": "/tmp/fibonacci.c"}}
</tool_call><|im_end|>
```

## deepseek-v4-flash.mq2lloyd — deepseek4-cap — SKIPPED (model not present)

## deepseek-v4-flash.mq2lloyd — deepseek4-reason — SKIPPED (model not present)

## deepseek-v4-flash.mq2lloyd — deepseek4-long-prefill — SKIPPED (model not present)

