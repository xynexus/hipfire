# Hybrid Speculative Decoding Architecture

## GPU Main Model + NPU DFlash + CPU DDTree and DSpark

## 1. Purpose

This document defines a heterogeneous speculative decoding architecture in which:

- the **main target model** runs on the GPU;
- **DFlash** runs on an NPU and produces a parallel block proposal;
- **DDTree** runs on the CPU and converts the proposal into a bounded candidate tree;
- **DSpark** runs on the CPU and provides branch-specific causal refinement;
- the GPU verifies the packed tree in one target-model pass.

The design goal is to keep the expensive target model highly utilized while moving low-latency, branch-heavy work onto the NPU and CPU.

The intended division of labour is:

| Device | Component | Primary Role |
|---|---|---|
| GPU | Main model | Authoritative verification and final token distribution |
| NPU | DFlash | Parallel block drafting |
| CPU | DDTree | Candidate-tree construction, scheduling, pruning |
| CPU | DSpark | Causal branch refinement and vocabulary routing |
| CPU cache hierarchy | Shared state | Branch states, router weights, recurrent weights, active vocabulary shards |

---

## 2. High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU: Main Model                             │
│                                                                     │
│  Prefix KV cache ──► target hidden state ──► packed tree verifier   │
│                                              │                      │
│                                              ▼                      │
│                                  accepted path + target token       │
└─────────────────────────────────────────────────────────────────────┘
                         ▲                         │
                         │ verifier input          │ accepted prefix
                         │                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CPU: DDTree + DSpark                        │
│                                                                     │
│  DFlash features ─► DDTree root/trunk ─► DSpark branch refinement  │
│                                          │                          │
│                                          ├─► fork branch state      │
│                                          ├─► score active vocab     │
│                                          ├─► prune low-utility nodes │
│                                          └─► pack ancestor mask     │
└─────────────────────────────────────────────────────────────────────┘
                         ▲
                         │ projected features / top-k distributions
                         │
┌─────────────────────────────────────────────────────────────────────┐
│                         NPU: DFlash                                 │
│                                                                     │
│  target hidden state ─► parallel block proposal                    │
│                         q_i(x_i | h_target)                          │
│                                                                     │
│  Outputs:                                                           │
│  - per-position hidden features                                    │
│  - top-k token candidates                                          │
│  - confidence / entropy estimates                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End Dataflow

### 3.1 Target-model prefix state

The GPU processes the accepted prefix and exposes a compact conditioning representation:

\[
h_{\mathrm{target}} = f_{\mathrm{target}}(x_{1:t})
\]

This representation is passed to DFlash on the NPU.

The transfer should ideally use:

- pinned memory;
- shared unified memory;
- peer-to-peer access;
- or a compact projected representation rather than a full hidden-state tensor.

---

### 3.2 DFlash parallel proposal

DFlash predicts a block of \(B\) future positions in parallel:

\[
q_i(x_i \mid h_{\mathrm{target}}), \qquad i \in \{1,\dots,B\}
\]

For each position, DFlash returns:

\[
\mathcal{D}_i =
\left(
z_i,\,
\operatorname{TopK}(q_i),\,
H(q_i),\,
c_i
\right)
\]

where:

- \(z_i\) is a compact hidden feature;
- \(\operatorname{TopK}(q_i)\) is the top-\(k\) candidate set;
- \(H(q_i)\) is token entropy;
- \(c_i\) is a confidence estimate.

The entropy is:

\[
H(q_i) = -\sum_{v \in V} q_i(v)\log q_i(v)
\]

DFlash does not need to construct the final draft path. Its job is to create a rich proposal field over the next block.

---

### 3.3 DDTree initial tree construction

DDTree builds a bounded candidate tree from the DFlash proposal field.

A simple path score before causal refinement is:

\[
S_{\mathrm{DFlash}}(x_{1:k})
=
\sum_{i=1}^{k}
\log q_i(x_i \mid h_{\mathrm{target}})
\]

This score treats future positions as approximately independent given the prefix. It is useful for generating the initial trunk and identifying likely forks.

The CPU can use best-first expansion:

\[
v^\star
=
\arg\max_{v \in \mathcal{F}}
U(v)
\]

where \(\mathcal{F}\) is the frontier and \(U(v)\) is a utility score.

A basic utility function is:

\[
U(v)
=
P(v)
\cdot
\mathbb{E}[\Delta L(v)]
-
\lambda_{\mathrm{verify}} C_{\mathrm{verify}}(v)
-
\lambda_{\mathrm{cpu}} C_{\mathrm{cpu}}(v)
\]

where:

- \(P(v)\) is branch reach probability;
- \(\mathbb{E}[\Delta L(v)]\) is expected rescued accepted length;
- \(C_{\mathrm{verify}}(v)\) is the GPU verification cost;
- \(C_{\mathrm{cpu}}(v)\) is CPU expansion cost.

---

## 4. DSpark Branch Refinement

### 4.1 Branch-specific recurrent state

Each retained tree node carries a compact DSpark state:

\[
s_v \in \mathbb{R}^{r}
\]

For a practical rank:

\[
r = 256
\]

The child state can be computed from:

\[
s_v
=
F_{\mathrm{DSpark}}
\left(
s_{\operatorname{parent}(v)},
e(x_v),
z_{\operatorname{depth}(v)}
\right)
\]

where:

- \(s_{\operatorname{parent}(v)}\) is the parent branch state;
- \(e(x_v)\) is the embedding of the selected token;
- \(z_{\operatorname{depth}(v)}\) is the DFlash position feature.

This turns DFlash's position-wise proposal into a branch-conditioned proposal:

\[
q_i
\left(
x_i
\mid
h_{\mathrm{target}},
x_{1:i-1}
\right)
\]

instead of:

\[
q_i(x_i \mid h_{\mathrm{target}})
\]

---

### 4.2 Forking

At each retained tree fork, the parent state is logically cloned:

\[
s_{\mathrm{parent}}
\rightarrow
\left\{
s_{\mathrm{child}_1},
s_{\mathrm{child}_2},
\dots,
s_{\mathrm{child}_m}
\right\}
\]

This should not be implemented as one operating-system thread per node.

Instead, each branch should be represented as a compact job:

```text
BranchJob {
    parent_state_index
    token_id
    depth
    path_score
    probability
    priority
}
```

A CPU worker pool processes these jobs in parallel.

The branch state itself is small enough that cloning is effectively cache-local.

---

## 5. CPU State Size

Assume:

\[
r = 256
\]

### 5.1 Persistent state per tree node

For BF16 or FP16 state storage:

\[
256 \times 2\ \mathrm{bytes} = 512\ \mathrm{bytes}
\]

Approximate node record:

| Field | Approximate Size |
|---|---:|
| DSpark recurrent state | 512 B |
| Token ID | 4 B |
| Parent index | 4-8 B |
| Depth and flags | 4-8 B |
| Path score | 4 B |
| Probability / confidence | 4 B |
| Child metadata | 8-16 B |
| Alignment and allocator overhead | 32-64 B |
| **Total** | **576-640 B** |

A useful approximation is:

\[
M_{\mathrm{node}} \approx 640\ \mathrm{bytes}
\]

Then:

\[
M_{\mathrm{tree}}
=
N_{\mathrm{nodes}} \cdot M_{\mathrm{node}}
\]

Examples:

| Nodes | Approximate State |
|---:|---:|
| 128 | 80 KB |
| 512 | 320 KB |
| 1,024 | 640 KB |
| 4,096 | 2.5 MB |
| 16,384 | 10 MB |

The branch-state store therefore fits comfortably inside L3 for practical tree sizes.

---

### 5.2 Per-worker scratch

Assume DFlash feature width:

\[
d = 1024
\]

The recurrent input may be:

\[
u_v =
\left[
s_{\mathrm{parent}};
e(x_v);
z_i
\right]
\]

If both the recurrent state and token embedding are rank \(r\), then:

\[
\dim(u_v) = 2r + d
\]

For \(r=256\) and \(d=1024\):

\[
\dim(u_v)=1536
\]

At BF16:

\[
1536 \times 2 = 3072\ \mathrm{bytes}
\]

A practical worker scratch budget is approximately:

\[
M_{\mathrm{worker}} \approx 5\text{ to }10\ \mathrm{KB}
\]

This should fit in private L1/L2 cache.

---

## 6. DSpark Recurrent Weight Size

A recurrent projection of shape:

\[
r \times (2r+d)
\]

contains:

\[
N_W = r(2r+d)
\]

For:

\[
r=256,\qquad d=1024
\]

\[
N_W = 256(512+1024)
=256 \times 1536
=393{,}216
\]

If the implementation uses three such projections:

\[
N_{\mathrm{total}}
\approx
3 \times 393{,}216
=
1{,}179{,}648
\]

Storage:

| Precision | Approximate Size |
|---|---:|
| FP32 | 4.7 MB |
| BF16 / FP16 | 2.4 MB |
| INT8 | 1.2 MB |
| INT4 | 0.6 MB |

Thus, the complete recurrent transition core can remain resident in L3.

---

## 7. Vocabulary Routing

The full DSpark vocabulary projection is the main potential cache bottleneck.

If the vocabulary has size \(V\), then:

\[
W_{\mathrm{vocab}} \in \mathbb{R}^{r \times V}
\]

For:

\[
r=256,\qquad V=150{,}000
\]

\[
N_{\mathrm{vocab}}
=
256 \times 150{,}000
=
38.4 \times 10^6
\]

Storage:

| Precision | Size |
|---|---:|
| FP32 | 153.6 MB |
| BF16 / FP16 | 76.8 MB |
| INT8 | 38.4 MB |
| INT4 | 19.2 MB |

The vocabulary projection therefore requires dynamic trimming or sharding if the entire CPU-side system is to remain L3-resident.

---

## 8. Learned Vocabulary Shards

### 8.1 Active vocabulary

The active token set should be:

\[
V_t =
V_{\mathrm{core}}
\cup
V_{\mathrm{DFlash}}
\cup
V_{\mathrm{recent}}
\cup
V_{\mathrm{task}}
\cup
\bigcup_{g \in G_t} V_g
\]

where:

- \(V_{\mathrm{core}}\) contains punctuation, whitespace, common tokens, control tokens, digits and fallback tokens;
- \(V_{\mathrm{DFlash}}\) contains DFlash top-\(k\) candidates;
- \(V_{\mathrm{recent}}\) contains recently used lexical neighbours;
- \(V_{\mathrm{task}}\) contains task-pinned groups;
- \(G_t\) is the set of vocabulary shards selected by a learned router.

---

### 8.2 Learned router

The router predicts shard probabilities:

\[
p(g \mid s_v,z_i,c)
\]

where:

- \(s_v\) is the DSpark branch state;
- \(z_i\) is the DFlash feature;
- \(c\) is compact context metadata.

A simple router can be:

\[
h_r
=
\phi
\left(
W_s s_v + W_z z_i + W_c c + b
\right)
\]

\[
p(g)
=
\operatorname{softmax}
\left(
W_g h_r
\right)
\]

The active groups may be selected until cumulative routed probability exceeds threshold \(\tau\):

\[
\sum_{g \in G_t} p(g) \ge \tau
\]

Typical values might be:

\[
\tau \in [0.999,0.9999]
\]

A hard language heuristic, such as disabling Cantonese shards when Cantonese has not appeared recently, can be used as an initial bootstrap. The learned router is superior because it can predict an upcoming language switch before the first token of that language appears.

---

### 8.3 Router training objective

The router should minimize missed target probability mass while also minimizing active vocabulary size.

A useful objective is:

\[
\mathcal{L}_{\mathrm{router}}
=
\mathcal{L}_{\mathrm{draft}}
+
\lambda_{\mathrm{miss}}
\left(
1 -
\sum_{v \in V_t}
p_{\mathrm{teacher}}(v)
\right)
+
\lambda_{\mathrm{size}}
|V_t|
\]

For tree drafting, missed probability should be weighted by branch reach probability:

\[
\mathcal{L}_{\mathrm{tree\ miss}}
=
\sum_{v \in T}
P(v)
\left(
1 -
\sum_{x \in V_v}
p_{\mathrm{teacher}}(x \mid v)
\right)
\]

This makes vocabulary mistakes near the root much more expensive than mistakes deep in a low-probability branch.

---

## 9. Progressive Vocabulary Expansion

Vocabulary evaluation should be an anytime process.

1. Score the core vocabulary.
2. Add DFlash top-\(k\) candidates.
3. Evaluate the highest-probability routed shard.
4. Estimate residual probability or confidence.
5. Activate more shards only if necessary.
6. Fall back to the full vocabulary only in pathological cases.

Let the current active vocabulary be \(V_t^{(j)}\) after \(j\) expansion stages.

Expansion continues while:

\[
R_t^{(j)}
>
\epsilon
\]

where residual uncertainty can be estimated as:

\[
R_t^{(j)}
=
1 -
\sum_{v \in V_t^{(j)}}
\hat p(v)
\]

or through a learned confidence estimator.

This allows predictable branches to evaluate only a few thousand tokens while uncertain language or domain transitions activate larger sets.

---

## 10. Candidate-Restricted Projection

For a candidate set of size \(K\), DSpark only evaluates:

\[
W_{\mathrm{cand}}
\in
\mathbb{R}^{r \times K}
\]

The compute cost per branch is:

\[
C_{\mathrm{cand}}
=
rK
\]

For:

\[
r=256,\qquad K=128
\]

\[
C_{\mathrm{cand}}
=
32{,}768
\]

multiply-accumulate operations per node.

The candidate matrix size is:

| Precision | \(256 \times 128\) |
|---|---:|
| BF16 | 64 KB |
| INT8 | 32 KB |
| INT4 | 16 KB |

This is small enough for L1 or L2 cache.

For efficient gathers, the vocabulary projection should be stored token-major:

```text
token_id -> packed 256-element transition row
```

A second layout may be maintained for larger batched projections.

---

## 11. Tree Utility and Pruning

Each node should be expanded only when its expected value exceeds its cost.

A general expansion criterion is:

\[
\Delta U(v)
=
P(\operatorname{parent}(v))
\cdot
p(v \mid \operatorname{parent}(v))
\cdot
\Delta L(v)
-
\lambda C(v)
\]

Expand only if:

\[
\Delta U(v) > 0
\]

A more complete scheduler can use:

\[
U(T)
=
\mathbb{E}[L_{\mathrm{accepted}} \mid T]
-
\lambda_{\mathrm{gpu}}
C_{\mathrm{verify}}(T)
-
\lambda_{\mathrm{cpu}}
C_{\mathrm{expand}}(T)
-
\lambda_{\mathrm{latency}}
L_{\mathrm{queue}}(T)
\]

This allows the system to change tree width and depth dynamically according to system load.

---

## 12. GPU Verification

The final tree is packed into a linear node sequence with an ancestor-only attention mask.

For nodes \(i\) and \(j\), define:

\[
A_{ij}
=
\begin{cases}
1, & \text{if node } j \text{ is an ancestor of node } i \\
0, & \text{otherwise}
\end{cases}
\]

The GPU verifies every candidate node in one target-model pass.

The accepted path is the deepest path consistent with the target model's decisions.

The target model remains authoritative:

\[
p_{\mathrm{final}} = p_{\mathrm{target}}
\]

The draft system only proposes candidates and affects acceptance efficiency.

---

## 13. Correctness Under Vocabulary Trimming

Vocabulary trimming is applied only to the DSpark drafter.

The target model still evaluates the full vocabulary.

For greedy decoding:

- if the target token is present in the tree, the path may continue;
- if the target token is absent, speculation stops and the target token is emitted.

For stochastic speculative decoding, the truncated draft distribution must be normalized consistently:

\[
q'(x)
=
\begin{cases}
\dfrac{q(x)}
{\sum_{v \in V_t} q(v)},
& x \in V_t \\
0,
& x \notin V_t
\end{cases}
\]

The acceptance calculation must use \(q'\), not the unnormalized original distribution.

Thus, trimming changes acceptance rate but does not need to change the final target distribution.

---

## 14. Cache Layout

A practical CPU cache layout is:

### Shared L3-resident data

- DSpark recurrent weights;
- vocabulary router weights;
- core vocabulary shard;
- hot language and domain shards;
- DFlash block features;
- tree node store;
- shared candidate-row cache.

### Private L2/L1 data

- active branch state;
- per-worker scratch;
- local expansion queue;
- current candidate rows;
- temporary logits.

### Main-memory cold data

- inactive language shards;
- obscure domain shards;
- rare token transition rows;
- fallback full vocabulary matrix.

---

## 15. Example L3 Budget

Assume a 64 MB L3 cache.

| Component | Approximate Size |
|---|---:|
| DSpark recurrent weights, BF16 | 2.4 MB |
| Router weights | 1-2 MB |
| 4,096 branch states | 2.5 MB |
| DFlash block features and metadata | <1 MB |
| Core vocabulary rows, INT8 | 8-16 MB |
| Active language/domain shards, INT8 | 16-24 MB |
| Candidate row cache | 4-8 MB |
| Runtime overhead and queues | 2-4 MB |
| **Total** | **36-59 MB** |

This makes full L3 residency plausible for the active working set.

The design should leave headroom rather than attempting to occupy 100% of L3.

A reasonable target is:

\[
M_{\mathrm{working\ set}}
\le
0.75 M_{\mathrm{L3}}
\]

For 64 MB L3:

\[
M_{\mathrm{working\ set}}
\le
48\ \mathrm{MB}
\]

---

## 16. Device Overlap

The architecture should be pipelined.

```text
Time ─────────────────────────────────────────────────────────────►

GPU:
[verify tree N] [target prefix update] [verify tree N+1] ...

NPU:
      [DFlash N+1]           [DFlash N+2] ...

CPU:
          [DDTree + DSpark N+1]
                                [DDTree + DSpark N+2]
```

The desired critical path is:

\[
T_{\mathrm{step}}
=
\max
\left(
T_{\mathrm{GPU\ verify}},
T_{\mathrm{NPU\ draft}} + T_{\mathrm{CPU\ tree}}
\right)
\]

The CPU and NPU draft pipeline should remain shorter than the GPU verification interval:

\[
T_{\mathrm{NPU\ draft}}
+
T_{\mathrm{CPU\ tree}}
\le
T_{\mathrm{GPU\ verify}}
\]

If this inequality holds, drafting is largely hidden behind GPU work.

---

## 17. Recommended Initial Configuration

A sensible prototype configuration is:

| Parameter | Initial Value |
|---|---:|
| DFlash block length \(B\) | 16 |
| DFlash top-\(k\) per position | 64 |
| DSpark rank \(r\) | 256 |
| DDTree node budget | 128 |
| Maximum tree depth | 16 |
| Maximum branch factor | 4 |
| Core vocabulary | 8k-16k tokens |
| Per-node candidate set \(K\) | 64-256 |
| Vocabulary shards | 128-512 |
| Router cumulative threshold \(\tau\) | 0.999 |
| Branch-state precision | BF16 |
| Vocabulary-row precision | INT8 initially |
| CPU worker count | physical cores or one worker per L2 slice |

---

## 18. Suggested Development Order

### Phase 1: Linear integration

- GPU main model;
- NPU DFlash;
- CPU DSpark on a single path;
- no DDTree branching;
- fixed top-\(k\) candidate vocabulary.

### Phase 2: Forked DSpark

- clone DSpark state at DDTree forks;
- bounded node budget;
- branch pruning;
- packed GPU tree verification.

### Phase 3: Learned vocabulary routing

- hand-built vocabulary shards;
- learned shard router;
- DFlash top-\(k\) escape hatch;
- progressive shard activation.

### Phase 4: Adaptive scheduling

- dynamic tree width and depth;
- GPU-load-aware node budget;
- per-request latency versus throughput policy;
- learned utility estimator.

### Phase 5: Joint optimization

- jointly train DFlash, DSpark and router;
- cluster vocabulary by conditional co-activation;
- optimize tree acceptance per unit of GPU verification cost.

---

## 19. Core Design Summary

The architecture is viable because the three speculative components solve different parts of the problem:

- **DFlash** cheaply predicts a block in parallel.
- **DDTree** preserves multiple likely continuations instead of committing to one path.
- **DSpark** restores branch-specific causal dependence.
- **Learned vocabulary routing** makes DSpark practical on the CPU.
- **The GPU target model** preserves exact output authority.

The critical cache observation is:

\[
M_{\mathrm{branch\ state}}
\ll
M_{\mathrm{L3}}
\]

The main challenge is instead:

\[
M_{\mathrm{full\ vocabulary\ projection}}
\gtrsim
M_{\mathrm{L3}}
\]

Therefore, the system should be designed around:

\[
\text{forked branch state}
+
\text{learned vocabulary shards}
+
\text{candidate-restricted scoring}
\]

The resulting CPU-side active working set can plausibly remain inside L3, allowing the CPU to act as a high-speed branch engine while the GPU and NPU remain occupied with dense tensor work.
