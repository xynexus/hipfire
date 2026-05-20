#!/usr/bin/env python3
"""
Build a deployment-mirror calibration corpus for Qwen3.5/3.6 MQ4 quantization.

Target: 1024 sequences x 2048 tokens = ~2.1M tokens total.

Composition (per task brief):
  - 30% chat dialogs (ChatML framing) ~ 307 sequences
  - 25% code (Python/Rust/HIP)        ~ 256 sequences
  - 25% Wikipedia prose               ~ 256 sequences (slice off wikitext-2)
  - 20% tool-call / agentic JSON      ~ 205 sequences

Output is a single concatenated text file. llama-perplexity chunks it into
n_ctx-sized windows at evaluation time, mirroring the wikitext-2 slice's
convention (no per-sequence sentinel; the chunker is the tokenizer).

We approximate sequence boundaries by writing blank-line separators between
units so a human can visually skim. The tokenizer ignores them but tokenizes
them into newline tokens which is benign.

Deterministic (random seed = 1024). Re-running produces byte-identical output
when the input corpora md5s are stable.

This script intentionally writes "sequence-like" content in concatenated form
to mirror wikitext2-1024s-2048ctx.txt; the actual sequence count is verified
post-hoc by computing target_tokens / 2048.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[4]
SLICE_DIR = REPO_ROOT / "benchmarks" / "quality-baselines" / "slice"
WIKITEXT_SLICE = SLICE_DIR / "wikitext2-1024s-2048ctx.txt"
OUT = SLICE_DIR / "calibration-mix-v1.txt"
BUILD_LOG = SLICE_DIR / "calibration-mix-v1.build.log"

# Hermes Apache-2.0 traces (chat + tool-call).
HERMES_BASE = Path(
    "/mnt/nas/kaden/cache/huggingface/hub/datasets--lambda--hermes-agent-reasoning-traces/"
    "snapshots/b92885e4f0161d4b2536512710e004d4892cac6e/data"
)
HERMES_GLM = HERMES_BASE / "glm-5.1" / "train.parquet"
HERMES_KIMI = HERMES_BASE / "kimi" / "train.parquet"

# Char/token ratios (empirically ~3.5 chars/token for Qwen3.5 on English code+prose).
# We over-write text and let llama-tokenize compute the actual count post-hoc.
CHARS_PER_TOKEN = 3.5
TARGET_TOKENS = 1024 * 2048   # 2_097_152
# 25% prose budget shares the wikitext byte stream untouched; the remaining
# 75% comes from new content. We aim for ~9 MB of new content to comfortably
# cover the token target with margin.

# Per-bucket char budget, calibrated against the empirical Qwen3.5 tokenizer
# chars/token ratios from a dry-run on this exact source mix:
#   wiki  ~ 4.37 chars/tok  (English prose)
#   chat  ~ 4.22 chars/tok  (ChatML dialog incl. think blocks)
#   code  ~ 3.32 chars/tok  (Python / Rust / HIP — denser tokenization)
#   tool  ~ 3.06 chars/tok  (JSON + XML wrappers; highest token density)
#
# Target tokens per bucket (out of 2,097,152 = 1024 * 2048):
#   chat 30% = 629,146 tok -> ~2.65 MB
#   code 25% = 524,288 tok -> ~1.74 MB
#   wiki 25% = 524,288 tok -> ~2.29 MB
#   tool 20% = 419,430 tok -> ~1.29 MB
#
# We pad each by a small margin so the final concatenation comfortably exceeds
# the 2.1M-token target; llama-perplexity chunks the first 1024 windows.
BUDGET_CHARS = {
    "wiki":     int(2.30 * 1024 * 1024),
    "chat":     int(2.85 * 1024 * 1024),
    "code":     int(1.70 * 1024 * 1024),
    "tool":     int(1.37 * 1024 * 1024),
}

SEED = 1024
SEP = "\n\n"  # benign tokenization marker; mirrors WT2's blank-line style


def log(msg: str, *, fh=None) -> None:
    print(msg, file=sys.stderr)
    if fh is not None:
        fh.write(msg + "\n")


# ---------------------------------------------------------------------------
# Bucket assemblers
# ---------------------------------------------------------------------------

def assemble_wiki(budget: int, logf) -> str:
    """Take the first `budget` bytes of the existing wikitext slice."""
    text = WIKITEXT_SLICE.read_text(encoding="utf-8")
    if len(text) < budget:
        log(f"warning: wikitext slice ({len(text)} B) < budget ({budget} B)", fh=logf)
        return text
    # Clip at a paragraph boundary near the budget for cleanliness.
    cut = text.rfind("\n", 0, budget)
    if cut < budget - 4096:
        cut = budget
    sliced = text[:cut]
    log(f"  wiki: clipped to {len(sliced):,} B at byte {cut} (target {budget:,})", fh=logf)
    return sliced


def _load_hermes_rows() -> list[dict]:
    """Load hermes parquet rows; combine glm + kimi."""
    import pyarrow.parquet as pq
    rows: list[dict] = []
    for p in (HERMES_GLM, HERMES_KIMI):
        if not p.exists():
            continue
        t = pq.read_table(p)
        # Sample reasonably; we don't need all 14K rows for ~5 MB of content.
        # Take first 4000 rows from each — enough chat + tool-call coverage.
        n = min(4000, t.num_rows)
        for i in range(n):
            rows.append(t.slice(i, 1).to_pylist()[0])
    return rows


def _format_chatml_turn(role: str, value: str) -> str:
    """Emit a single <|im_start|>role ... <|im_end|> turn."""
    return f"<|im_start|>{role}\n{value}<|im_end|>\n"


def _hermes_row_to_chatml(row: dict, *, include_tools_in_system: bool = False) -> tuple[str, bool]:
    """
    Convert a hermes ShareGPT row to Qwen3.5 ChatML format.

    Returns (text, has_tool_call). The role mapping:
      system -> system
      human  -> user
      gpt    -> assistant
      tool   -> user (Qwen wraps tool responses as user with <tool_response>)

    If `include_tools_in_system` is True, the tools JSON is appended to the
    system message in the Qwen3.5 template style (see chat_template.jinja).
    """
    convs = row.get("conversations", [])
    if not convs:
        return "", False
    has_tool_call = False
    out: list[str] = []
    role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "user"}

    # Optionally bolt tools onto system message in Qwen format.
    if include_tools_in_system and row.get("tools"):
        # Find the system message
        for c in convs:
            if c["from"] == "system":
                # Splice the Qwen tool-call instructions into system content.
                tools_block = (
                    "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
                    + row["tools"]
                    + "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                    + "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>"
                )
                c = {"from": "system", "value": c["value"] + tools_block}
                break

    for c in convs:
        role = role_map.get(c["from"], "user")
        val = c.get("value", "")
        if not val:
            continue
        if "<tool_call>" in val:
            has_tool_call = True
        if c["from"] == "tool":
            val = f"<tool_response>\n{val}\n</tool_response>"
        out.append(_format_chatml_turn(role, val))
    return "".join(out), has_tool_call


def assemble_tool_calls(budget: int, hermes_rows: list[dict], rng: random.Random, logf) -> str:
    """
    Assemble tool-call / agentic JSON sequences.

    Sources:
      1. Qwen3.5 tool-call exemplars from benchmarks/prompts/agentic_*.txt + tool_call_*.txt
      2. Hermes traces with high tool_call density
      3. Synthesized round-trip examples
    """
    out: list[str] = []
    total = 0

    # 1. Existing fixtures: load each agentic / tool_call prompt as-is.
    fixtures_dir = REPO_ROOT / "benchmarks" / "prompts"
    fixtures = [
        ("agentic_hermes_system", _format_chatml_turn("system", (fixtures_dir / "agentic_hermes_system.txt").read_text())),
        ("agentic_pi_system", _format_chatml_turn("system", (fixtures_dir / "agentic_pi_system.txt").read_text())),
        ("tool_call_system", _format_chatml_turn("system", (fixtures_dir / "tool_call_system.txt").read_text())),
        ("agentic_user_read", _format_chatml_turn("user", (fixtures_dir / "agentic_user_read.txt").read_text())),
        ("agentic_user_multistep", _format_chatml_turn("user", (fixtures_dir / "agentic_user_multistep.txt").read_text())),
        ("tool_call_read_file", _format_chatml_turn("user", (fixtures_dir / "tool_call_read_file.txt").read_text())),
    ]
    for name, blob in fixtures:
        out.append(blob)
        total += len(blob)
    log(f"  tool: seeded with {len(fixtures)} fixtures ({total:,} B)", fh=logf)

    # 2. Hermes rows with at least 2 tool_call turns. Shuffle for diversity.
    rng.shuffle(hermes_rows)
    n_used = 0
    for row in hermes_rows:
        if total >= budget:
            break
        # Count tool_calls in the row
        n_tc = sum(1 for c in row.get("conversations", []) if c["from"] == "gpt" and "<tool_call>" in c.get("value", ""))
        if n_tc < 1:
            continue
        text, _ = _hermes_row_to_chatml(row, include_tools_in_system=True)
        if not text:
            continue
        out.append(text + SEP)
        total += len(text) + len(SEP)
        n_used += 1
    log(f"  tool: pulled {n_used} hermes rows", fh=logf)

    # 3. Synthesized minimal round-trips for diversity (short, schema-canonical).
    synth = _synth_tool_call_examples(rng)
    for s in synth:
        if total >= budget * 1.05:
            break
        out.append(s + SEP)
        total += len(s) + len(SEP)
    log(f"  tool: added {len(synth)} synthesized round-trips ({total:,} B total)", fh=logf)

    blob = "".join(out)
    if len(blob) > budget * 1.05:
        # Clip at SEP boundary nearest budget.
        cut = blob.rfind(SEP, 0, budget)
        if cut > 0:
            blob = blob[:cut]
    return blob


def _synth_tool_call_examples(rng: random.Random) -> list[str]:
    """
    Generate canonical Qwen3.5 tool-call round-trips for schema coverage.
    Format follows chat_template.jinja: <tool_call><function=NAME>...</function></tool_call>
    """
    examples: list[str] = []

    # Common system prompt for tool-call exemplars.
    system_with_tools = (
        "You are a helpful assistant with access to the following tools.\n\n"
        "# Tools\n\nYou have access to the following functions:\n\n"
        "<tools>\n"
        '{"type":"function","function":{"name":"read_file","description":"Read a file from disk","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}\n'
        '{"type":"function","function":{"name":"write_file","description":"Write content to a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}\n'
        '{"type":"function","function":{"name":"bash","description":"Execute a bash command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}\n'
        '{"type":"function","function":{"name":"web_search","description":"Search the web for a query","parameters":{"type":"object","properties":{"query":{"type":"string"},"max_results":{"type":"integer"}},"required":["query"]}}}\n'
        '{"type":"function","function":{"name":"compute","description":"Evaluate a math expression","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}}\n'
        "</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>"
    )

    # Round-trip patterns (user question -> assistant tool_call -> tool response -> assistant final).
    scenarios = [
        # (user_q, tool_name, params, tool_response, final_answer)
        (
            "What is in /etc/hostname?",
            "read_file",
            {"path": "/etc/hostname"},
            "k9lin\n",
            "The file /etc/hostname contains the hostname `k9lin`.",
        ),
        (
            "Write `Hello, world` to /tmp/greeting.txt",
            "write_file",
            {"path": "/tmp/greeting.txt", "content": "Hello, world"},
            "wrote 12 bytes",
            "Done. I wrote 12 bytes to `/tmp/greeting.txt`.",
        ),
        (
            "Run `uname -srm` and tell me what it prints.",
            "bash",
            {"command": "uname -srm"},
            "Linux 6.17.0-23-generic x86_64",
            "The kernel is `Linux 6.17.0-23-generic` on `x86_64`.",
        ),
        (
            "How many GPUs are visible to ROCm?",
            "bash",
            {"command": "rocm-smi --showid 2>/dev/null | grep -c GPU"},
            "2",
            "There are 2 GPUs visible to ROCm on this host.",
        ),
        (
            "Search for the term `Qwen3.5 MoE quantization`.",
            "web_search",
            {"query": "Qwen3.5 MoE quantization", "max_results": 5},
            '[{"title":"Qwen3.5-A3B MoE quant overview","url":"https://example.org/qwen35-moe"}]',
            "I found a recent overview of Qwen3.5-A3B quantization at example.org.",
        ),
        (
            "Compute 17 * 23 + 41.",
            "compute",
            {"expression": "17 * 23 + 41"},
            "432",
            "17 * 23 + 41 = 432.",
        ),
        (
            "Read /etc/os-release and tell me the distro.",
            "read_file",
            {"path": "/etc/os-release"},
            "NAME=\"Ubuntu\"\nVERSION=\"24.04 LTS\"\nID=ubuntu",
            "The distribution is Ubuntu 24.04 LTS.",
        ),
        (
            "Check disk usage on /.",
            "bash",
            {"command": "df -h / | tail -1"},
            "/dev/nvme0n1p2  931G  421G  463G  48% /",
            "Disk `/dev/nvme0n1p2` is at 48% usage (421G used of 931G).",
        ),
        (
            "Find all `.rs` files under `src/` and count them.",
            "bash",
            {"command": "find src/ -name '*.rs' | wc -l"},
            "47",
            "There are 47 Rust source files under `src/`.",
        ),
        (
            "Write a 3-line Python script that prints the cube of 3 to /tmp/cube.py.",
            "write_file",
            {"path": "/tmp/cube.py", "content": "n = 3\nprint(n ** 3)\n"},
            "wrote 19 bytes",
            "Wrote a 3-line script to `/tmp/cube.py`.",
        ),
        (
            "Read /proc/cpuinfo and tell me the model name of the first CPU.",
            "read_file",
            {"path": "/proc/cpuinfo"},
            "processor\t: 0\nvendor_id\t: GenuineIntel\ncpu family\t: 6\nmodel\t\t: 158\nmodel name\t: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz",
            "The first CPU is an Intel(R) Core(TM) i7-9700K @ 3.60GHz.",
        ),
        (
            "Compute the determinant of the 2x2 matrix [[3,4],[2,5]].",
            "compute",
            {"expression": "3*5 - 4*2"},
            "7",
            "The determinant is 7.",
        ),
        (
            "Search for `RDNA1 ROCm support` and limit to 3 results.",
            "web_search",
            {"query": "RDNA1 ROCm support", "max_results": 3},
            '[{"title":"RDNA1 unofficial ROCm guides","url":"https://rocm.docs.amd.com"},{"title":"gfx1010 build notes","url":"https://github.com/ROCm/ROCm"}]',
            "I found two relevant references: the official ROCm docs and a gfx1010 build note on the ROCm repo.",
        ),
        (
            "Use bash to find Python files modified in the last day under /home/kaden/projects.",
            "bash",
            {"command": "find /home/kaden/projects -name '*.py' -mtime -1"},
            "/home/kaden/projects/foo/main.py\n/home/kaden/projects/bar/utils.py",
            "Two files were modified in the last day: `foo/main.py` and `bar/utils.py`.",
        ),
        (
            "What does the script /tmp/cube.py output?",
            "bash",
            {"command": "python3 /tmp/cube.py"},
            "27",
            "It prints `27` (the cube of 3).",
        ),
        (
            "Read README.md and summarize its first paragraph.",
            "read_file",
            {"path": "README.md"},
            "# hipfire\n\nRust-native ML inference engine for AMD RDNA GPUs.\nTargets Qwen 3.5/3.6 dense + MoE families with custom HIP kernels.\n",
            "hipfire is a Rust-native ML inference engine for AMD RDNA GPUs targeting Qwen 3.5/3.6 dense and MoE models with custom HIP kernels.",
        ),
        (
            "Compute mean of the list [3.2, 4.1, 5.8, 6.5, 7.0].",
            "compute",
            {"expression": "(3.2 + 4.1 + 5.8 + 6.5 + 7.0) / 5"},
            "5.32",
            "The mean is 5.32.",
        ),
        (
            "Write `pid: $$` to /tmp/pid.txt using a bash command.",
            "bash",
            {"command": "echo \"pid: $$\" > /tmp/pid.txt"},
            "",
            "Wrote the PID line to `/tmp/pid.txt`.",
        ),
        (
            "Search for `Lloyd-Max codebook`, top 4 results.",
            "web_search",
            {"query": "Lloyd-Max codebook quantization", "max_results": 4},
            '[{"title":"Lloyd-Max iterative quantizer derivation","url":"https://en.wikipedia.org/wiki/Lloyd%27s_algorithm"}]',
            "Found a reference describing the Lloyd-Max iterative quantizer on Wikipedia.",
        ),
        (
            "Run `ls -la /etc/hostname` and report owner+permissions.",
            "bash",
            {"command": "ls -la /etc/hostname"},
            "-rw-r--r-- 1 root root 6 May 20 04:11 /etc/hostname",
            "Owner is `root:root`, permissions `-rw-r--r--`, size 6 bytes.",
        ),
    ]

    # Multi-step scenarios (user -> tool_call -> tool -> assistant -> user follow-up -> tool_call -> tool -> assistant).
    multistep = [
        {
            "system": system_with_tools,
            "turns": [
                ("user", "I have a CSV at /tmp/data.csv. First, count its lines, then read the header row."),
                ("assistant", "<tool_call>\n<function=bash>\n<parameter=command>\nwc -l /tmp/data.csv\n</parameter>\n</function>\n</tool_call>"),
                ("tool", "127 /tmp/data.csv"),
                ("assistant", "The file has 127 lines. Now I'll read the header.\n\n<tool_call>\n<function=bash>\n<parameter=command>\nhead -1 /tmp/data.csv\n</parameter>\n</function>\n</tool_call>"),
                ("tool", "id,name,value,timestamp"),
                ("assistant", "The CSV has 127 lines (including the header). The header row is `id,name,value,timestamp`."),
            ],
        },
        {
            "system": system_with_tools,
            "turns": [
                ("user", "Read /etc/os-release, then if the distro is Ubuntu, write `ubuntu-detected` to /tmp/marker.txt."),
                ("assistant", "<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/os-release\n</parameter>\n</function>\n</tool_call>"),
                ("tool", "NAME=\"Ubuntu\"\nVERSION=\"24.04 LTS\"\nID=ubuntu"),
                ("assistant", "The distro is Ubuntu. Writing the marker file.\n\n<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/marker.txt\n</parameter>\n<parameter=content>\nubuntu-detected\n</parameter>\n</function>\n</tool_call>"),
                ("tool", "wrote 16 bytes"),
                ("assistant", "Confirmed Ubuntu 24.04 LTS and wrote the marker to `/tmp/marker.txt`."),
            ],
        },
        {
            "system": system_with_tools,
            "turns": [
                ("user", "What's the area of a circle with radius 7.3? Then compute its circumference."),
                ("assistant", "<tool_call>\n<function=compute>\n<parameter=expression>\n3.141592653589793 * 7.3 * 7.3\n</parameter>\n</function>\n</tool_call>"),
                ("tool", "167.4150523817473"),
                ("assistant", "Area ~= 167.42. Now the circumference.\n\n<tool_call>\n<function=compute>\n<parameter=expression>\n2 * 3.141592653589793 * 7.3\n</parameter>\n</function>\n</tool_call>"),
                ("tool", "45.86725476349606"),
                ("assistant", "Area is approximately 167.42 sq units, circumference is approximately 45.87 units."),
            ],
        },
    ]

    # Error / recovery scenarios.
    errors = [
        {
            "system": system_with_tools,
            "turns": [
                ("user", "Read /this/path/does/not/exist.txt"),
                ("assistant", "<tool_call>\n<function=read_file>\n<parameter=path>\n/this/path/does/not/exist.txt\n</parameter>\n</function>\n</tool_call>"),
                ("tool", "Error: file not found: /this/path/does/not/exist.txt"),
                ("assistant", "The file `/this/path/does/not/exist.txt` does not exist. Could you double-check the path?"),
            ],
        },
        {
            "system": system_with_tools,
            "turns": [
                ("user", "Run `rocm-smi` then tell me clock speeds."),
                ("assistant", "<tool_call>\n<function=bash>\n<parameter=command>\nrocm-smi\n</parameter>\n</function>\n</tool_call>"),
                ("tool", "bash: rocm-smi: command not found"),
                ("assistant", "`rocm-smi` is not installed on this system. If ROCm is installed, the binary may be under `/opt/rocm/bin/`; otherwise install the `rocm-utils` package."),
            ],
        },
        {
            "system": system_with_tools,
            "turns": [
                ("user", "Compute 1/0 for me."),
                ("assistant", "<tool_call>\n<function=compute>\n<parameter=expression>\n1 / 0\n</parameter>\n</function>\n</tool_call>"),
                ("tool", "Error: ZeroDivisionError"),
                ("assistant", "Division by zero is undefined. Did you mean a different denominator?"),
            ],
        },
    ]

    # Render single round-trips.
    for (user_q, tool_name, params, tool_response, final) in scenarios:
        param_block = "\n".join(f"<parameter={k}>\n{v}\n</parameter>" for k, v in params.items())
        text = (
            _format_chatml_turn("system", system_with_tools)
            + _format_chatml_turn("user", user_q)
            + _format_chatml_turn(
                "assistant",
                f"<tool_call>\n<function={tool_name}>\n{param_block}\n</function>\n</tool_call>",
            )
            + _format_chatml_turn("user", f"<tool_response>\n{tool_response}\n</tool_response>")
            + _format_chatml_turn("assistant", final)
        )
        examples.append(text)

    # Render multistep + error scenarios.
    for s in multistep + errors:
        out = _format_chatml_turn("system", s["system"])
        for role, val in s["turns"]:
            mapped_role = "user" if role == "tool" else role
            if role == "tool":
                val = f"<tool_response>\n{val}\n</tool_response>"
            out += _format_chatml_turn(mapped_role, val)
        examples.append(out)

    # Schema-edge cases (nested JSON in parameters, multi-line strings, escaping).
    edge_examples = []
    edge_examples.append(
        _format_chatml_turn("system", system_with_tools)
        + _format_chatml_turn(
            "user",
            "Write a JSON config to /tmp/cfg.json with model=qwen3.5 and ctx=8192.",
        )
        + _format_chatml_turn(
            "assistant",
            "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/cfg.json\n</parameter>\n<parameter=content>\n"
            + json.dumps({"model": "qwen3.5", "ctx": 8192, "kv_mode": "asym3", "dtype": "bf16"}, indent=2)
            + "\n</parameter>\n</function>\n</tool_call>",
        )
        + _format_chatml_turn("user", "<tool_response>\nwrote 81 bytes\n</tool_response>")
        + _format_chatml_turn("assistant", "Wrote the JSON config (81 bytes) to `/tmp/cfg.json`.")
    )
    edge_examples.append(
        _format_chatml_turn("system", system_with_tools)
        + _format_chatml_turn(
            "user",
            "Write a multi-line Python function to /tmp/binsearch.py.",
        )
        + _format_chatml_turn(
            "assistant",
            "<tool_call>\n<function=write_file>\n<parameter=path>\n/tmp/binsearch.py\n</parameter>\n<parameter=content>\n"
            + 'def binary_search(arr, target):\n'
            + '    lo, hi = 0, len(arr) - 1\n'
            + '    while lo <= hi:\n'
            + '        mid = (lo + hi) // 2\n'
            + '        if arr[mid] == target:\n'
            + '            return mid\n'
            + '        if arr[mid] < target:\n'
            + '            lo = mid + 1\n'
            + '        else:\n'
            + '            hi = mid - 1\n'
            + '    return -1\n'
            + "\n</parameter>\n</function>\n</tool_call>",
        )
        + _format_chatml_turn("user", "<tool_response>\nwrote 264 bytes\n</tool_response>")
        + _format_chatml_turn("assistant", "Wrote a standard iterative binary search to `/tmp/binsearch.py`.")
    )
    examples.extend(edge_examples)

    rng.shuffle(examples)
    return examples


def assemble_chat(budget: int, hermes_rows: list[dict], rng: random.Random, logf) -> str:
    """
    Assemble pure chat dialog content.

    Strategy:
      1. From hermes traces: take rows whose gpt turns have NO tool_call (rare
         but present) — pure conversational.
      2. For hermes rows that DO have tool_calls, extract only the
         conversational segments (think blocks + final answers without
         tool_call XML).
      3. Synthesize ~50 short helpful-assistant exchanges covering common
         user-question patterns.
    """
    out: list[str] = []
    total = 0

    # 1+2. Hermes — emit rows but strip <tool_call>...</tool_call> blocks to get
    # the conversational skeleton (think + final answer survive).
    rng.shuffle(hermes_rows)
    tool_call_re = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
    tool_response_re = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)
    n_used = 0
    for row in hermes_rows:
        # Hermes is our primary chat source — synthesized exchanges are only
        # a small flavour boost. Aim to consume ~90% of the budget from real
        # multi-turn dialogue.
        if total >= budget * 0.92:
            break
        convs = row.get("conversations", [])
        # Drop rows that would tokenize too short.
        if len(convs) < 2:
            continue
        text_parts: list[str] = []
        for c in convs:
            role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": None}
            role = role_map.get(c["from"])
            if role is None:
                continue
            val = c.get("value", "")
            # Strip tool_call and tool_response blocks — chat bucket is pure dialog.
            val = tool_call_re.sub("", val)
            val = tool_response_re.sub("", val)
            val = val.strip()
            if not val:
                continue
            text_parts.append(_format_chatml_turn(role, val))
        if not text_parts:
            continue
        text = "".join(text_parts)
        # Don't bloat with rows that ended up empty after stripping.
        if len(text) < 200:
            continue
        out.append(text + SEP)
        total += len(text) + len(SEP)
        n_used += 1
    log(f"  chat: pulled {n_used} hermes rows (tool_call stripped) -> {total:,} B", fh=logf)

    # 3. Synthesize chat dialogs for remaining budget.
    synth = _synth_chat_examples(rng)
    n_synth = 0
    for s in synth:
        if total >= budget:
            break
        out.append(s + SEP)
        total += len(s) + len(SEP)
        n_synth += 1
    log(f"  chat: added {n_synth} synthesized exchanges -> {total:,} B", fh=logf)

    blob = "".join(out)
    if len(blob) > budget * 1.05:
        cut = blob.rfind(SEP, 0, budget)
        if cut > 0:
            blob = blob[:cut]
    return blob


def _synth_chat_examples(rng: random.Random) -> list[str]:
    """
    Synthesize helpful-assistant style exchanges. Mixes ~3 categories:
      - Concept explanation (technical / programming)
      - Step-by-step how-to
      - Open-ended Q&A (analogies, debug-help, follow-ups)

    Each exchange is 2-6 turns to mimic real chats.
    """
    base_system = (
        "You are a helpful, harmless, and honest AI assistant. Answer the user's "
        "questions clearly and concisely. When the question is technical, "
        "include short examples where useful. When uncertain, say so."
    )

    exchanges: list[str] = []

    # Single Q-A — concept explanation.
    qa_concept = [
        ("Can you explain what a Bloom filter is and when you would use one?",
         "A Bloom filter is a space-efficient probabilistic data structure that tells you "
         "whether an element is *probably* in a set or *definitely not* in a set. It works "
         "by hashing the element with several independent hash functions and setting the bits "
         "at those positions in a bit array. To test membership, you check whether all the "
         "corresponding bits are set: if any are clear, the element is definitely absent; if "
         "all are set, the element is probably present (false positives are possible, false "
         "negatives are not).\n\n"
         "You'd use one when you need fast set-membership checks against a huge corpus and you "
         "can tolerate a small false positive rate. Classic examples: pre-checking whether a "
         "URL was already crawled, pre-filtering keys before doing an expensive disk lookup, "
         "or de-duplicating streams. The trade-off is that you can't delete from a standard "
         "Bloom filter (you'd need a counting variant), and tuning involves picking the bit "
         "array size and number of hash functions based on the expected element count and "
         "acceptable false positive rate."),
        ("What is the difference between a process and a thread in operating systems?",
         "A process is an isolated execution unit with its own virtual address space, file "
         "descriptors, signal handlers, and OS-level resources. A thread is an execution unit "
         "inside a process; threads of the same process share the address space, heap, and "
         "open files, but each thread has its own stack and register state.\n\n"
         "Practically: switching between processes is more expensive than switching between "
         "threads, because the OS has to swap the page-table base register and flush parts of "
         "the TLB. Threads communicate via shared memory directly; processes communicate via "
         "pipes, sockets, shared memory segments, or message-passing IPC. Threads can race on "
         "shared data, which requires locks/atomics; processes don't have that problem "
         "between themselves but pay for it on context switches.\n\n"
         "Modern programs use both: a web server might use one process per CPU core (for "
         "isolation and avoiding GIL/GC pauses on a single heap) and a thread pool inside each "
         "process for handling concurrent requests."),
        ("Why does floating-point arithmetic sometimes give surprising results?",
         "Most binary floating-point types can't exactly represent decimal fractions like "
         "`0.1` or `0.2`. They store an approximation that rounds to the nearest "
         "representable value, and small rounding errors accumulate during arithmetic.\n\n"
         "The classic example is `0.1 + 0.2 != 0.3`. Internally, `0.1` is stored as something "
         "like `0.1000000000000000055511151231257827021181583404541015625`, and the sum picks "
         "up a small extra epsilon. When you print the sum, you see `0.30000000000000004`.\n\n"
         "Mitigations: compare with a tolerance instead of `==`; use a decimal type when "
         "exact decimal arithmetic matters (financial code); use rational types for symbolic "
         "math; widen to `f64` from `f32` if your computation is sensitive."),
        ("What is the CAP theorem and how does it relate to distributed databases?",
         "The CAP theorem says that a distributed data store can only guarantee two of three "
         "properties at any one time during a network partition: Consistency (every read sees "
         "the latest write), Availability (every request gets a non-error response), and "
         "Partition tolerance (the system keeps working when network links fail).\n\n"
         "Because partitions are a reality of distributed systems, the practical trade-off is "
         "between C and A *during a partition*. A CP system rejects writes (or reads) on the "
         "minority side of the partition to keep data consistent; an AP system accepts writes "
         "on both sides and reconciles later (eventual consistency).\n\n"
         "Examples: classic relational replicas with synchronous failover lean CP; DNS, "
         "Cassandra, and many NoSQL stores lean AP. Modern systems often offer tunable "
         "consistency per query, blurring the binary classification."),
        ("How does HTTPS actually establish a secure connection?",
         "At a high level, the TLS handshake does three things: agree on a cipher suite, "
         "verify the server identity, and derive shared session keys.\n\n"
         "1. The client sends a `ClientHello` with the supported TLS versions, cipher suites, "
         "   and a random nonce.\n"
         "2. The server picks a cipher suite, sends a `ServerHello` with its own nonce, plus "
         "   its X.509 certificate chain (which contains its public key and is signed by a "
         "   trusted CA).\n"
         "3. The client verifies the certificate chain against its trust store, checks the "
         "   hostname matches the certificate's Subject Alternative Name, and uses an "
         "   ephemeral key exchange (typically ECDHE) to derive a shared secret with the "
         "   server. The secret feeds a KDF that produces the symmetric session keys.\n"
         "4. From then on, application data is encrypted with AES-GCM (or ChaCha20-Poly1305) "
         "   using the session keys; the asymmetric crypto is only used during the handshake.\n\n"
         "TLS 1.3 simplified this further: it uses only AEAD ciphers, removes static-RSA key "
         "exchange, and folds the entire handshake into one round trip (0-RTT for resumed "
         "sessions)."),
        ("Explain memory alignment and why it matters for performance.",
         "Memory alignment means placing an object at an address that is a multiple of its "
         "natural alignment (typically its size, for primitives). For example, an `i64` is "
         "naturally aligned at 8-byte boundaries.\n\n"
         "Why it matters:\n\n"
         "- Hardware: many CPUs require aligned access for SIMD instructions (e.g., AVX-512 "
         "  loads of 64 bytes must be 64-byte aligned for the fastest variant). Unaligned "
         "  vector loads work but are slower; misaligned scalar accesses on some ISAs even "
         "  trap.\n"
         "- Cache: aligning structures to cache-line boundaries (typically 64 bytes) prevents "
         "  *false sharing* where two cores keep invalidating each other's cache lines for "
         "  logically independent data.\n"
         "- Atomics: hardware atomics require natural alignment (or stronger) to guarantee "
         "  atomicity; otherwise the atomic op might be implemented as a slower compare-and-"
         "  swap loop.\n\n"
         "In C/C++ you align with `alignas` or compiler attributes; in Rust with `#[repr(align"
         "(N))]`; the allocator usually returns aligned memory, but custom allocators (e.g. for "
         "GPU buffers) often require explicit padding."),
        ("What's the difference between `git rebase` and `git merge`?",
         "Both integrate changes from one branch into another, but they preserve history "
         "differently.\n\n"
         "`merge` creates a new merge commit that has two parents (your branch tip and the "
         "incoming branch tip). The original commit graph is preserved verbatim, including "
         "the divergent topology. Pros: history is faithful to what actually happened, "
         "non-destructive. Cons: graph can become hard to read with many short-lived branches; "
         "merge commits clutter the log.\n\n"
         "`rebase` rewrites your branch's commits on top of the target branch's tip, creating "
         "new commits with the same diff but different parents. Pros: linear history that's "
         "easy to read, individual commits stay reviewable. Cons: destructive — you've "
         "rewritten history, which is bad on shared branches. Force-pushes after rebase can "
         "lose work for collaborators.\n\n"
         "Rule of thumb: rebase locally on your own feature branch to keep history clean, "
         "then merge into `main` to preserve the integration point."),
        ("What is the L2 cache and why does it matter for ML inference?",
         "The L2 cache sits between L1 (per-core, very fast, small) and L3/LLC (shared, "
         "slower, large). It typically holds 256 KB to a few MB per core. For ML inference, "
         "L2 matters because matmul kernels reuse the same activations and weights across "
         "many output channels. If a tile fits in L2, the same weights are streamed once "
         "from DRAM and reused from L2 for all the work in that tile.\n\n"
         "On AMD RDNA GPUs the analogous cache is the L1 (per WGP) and L2 (shared across "
         "WGPs). Kernels are tuned so that weight tiles fit the L1 budget and activation "
         "tiles fit L2. Poorly tuned tile sizes cause spills back to HBM bandwidth, which is "
         "10-30x slower than L2 read bandwidth.\n\n"
         "Practical tuning: use rocprof to track L2 hit rate; increase tile size until either "
         "register pressure forces spills or the L2 hit rate drops; pick the inflection."),
    ]

    for q, a in qa_concept:
        exchanges.append(
            _format_chatml_turn("system", base_system)
            + _format_chatml_turn("user", q)
            + _format_chatml_turn("assistant", a)
        )

    # Coding help — multi-turn.
    coding_help = [
        [
            ("Why does my Python script print `[1, 2, 3]` when I expected `1 2 3`?",
             "Because `print(your_list)` prints the Python representation of the list, which "
             "includes the square brackets and commas. If you want space-separated, use the "
             "splat operator:\n\n"
             "```python\nnums = [1, 2, 3]\nprint(*nums)        # 1 2 3\nprint(*nums, sep=', ')  # 1, 2, 3\n```\n\n"
             "Or `' '.join(map(str, nums))` if you want a single string."),
            ("Got it. What if my list contains other types, like dicts?",
             "`*nums` will call the default `__repr__` for each element, which for a dict is "
             "`{'k': 'v'}`. If you want a stable format, pre-serialize each element:\n\n"
             "```python\nimport json\nprint(*(json.dumps(x) for x in items))\n```\n\n"
             "That way, each dict becomes a single-line JSON string with double quotes, which "
             "is parseable by most tools."),
        ],
        [
            ("I'm getting a `BorrowError` in Rust when I push to a Vec while iterating over it. "
             "What's the idiomatic fix?",
             "The borrow checker is preventing a real bug: if you push during iteration, you "
             "could reallocate the Vec out from under the iterator and end up with a dangling "
             "reference. Two idiomatic fixes:\n\n"
             "1. Collect what you want to push into a separate Vec, then extend after the loop:\n"
             "   ```rust\n"
             "   let to_add: Vec<_> = v.iter().filter(|x| **x > 0).map(|x| x * 2).collect();\n"
             "   v.extend(to_add);\n"
             "   ```\n"
             "2. Use indices instead of iterators if you need to mutate during traversal:\n"
             "   ```rust\n"
             "   let mut i = 0;\n"
             "   while i < v.len() {\n"
             "       if v[i] > 0 { v.push(v[i] * 2); }\n"
             "       i += 1;\n"
             "   }\n"
             "   ```\n\n"
             "Method 1 is usually clearer; method 2 is needed for truly index-driven traversal."),
            ("Will method 2 cause issues if the Vec reallocates during a push?",
             "Reallocation moves the backing buffer, but you're indexing through `v[i]`, not "
             "holding a borrowed reference across the push. Each `v[i]` access takes a fresh "
             "borrow that ends at the end of the expression, so the borrow checker is happy "
             "and the runtime sees the new buffer on the next iteration.\n\n"
             "The only subtle issue: if the loop appends elements faster than `i` advances, "
             "you may iterate forever. Make sure the condition that triggers a push isn't "
             "true for newly appended elements."),
        ],
        [
            ("My HIP kernel's occupancy on gfx1100 is only 25%. Where do I start?",
             "Look at the four resources that bound occupancy on RDNA:\n\n"
             "1. **VGPRs per thread**: each WGP has 1024 VGPRs total. If your kernel uses 256 "
             "   VGPRs/thread, you can only fit `1024 / 256 = 4` waves * 32 threads = 128 "
             "   threads per WGP active, which is 25% of the 512-thread WGP capacity. Dump "
             "   VGPR count from the .hsaco (see the `gfx-kernel-metadata` skill).\n"
             "2. **SGPRs per wave**: similar math, but it's rarely the binding factor unless "
             "   your kernel passes many scalar arguments.\n"
             "3. **LDS per workgroup**: 64 KB on RDNA3 per WGP. If you use 32 KB of LDS per "
             "   workgroup, you can only fit 2 workgroups per WGP.\n"
             "4. **Workgroup size**: launching with `dim3(64,1,1)` means only 2 waves per "
             "   workgroup, so each workgroup uses 64 lanes out of 1024 available. Increase "
             "   to `(256,1,1)` (8 waves) if VGPRs allow.\n\n"
             "Most often it's VGPRs. The fix is usually `__launch_bounds__` plus avoiding "
             "deep loop unrolling, or restructuring to reuse registers across loop iterations."),
            ("How do I check the VGPR count without recompiling?",
             "If you have the compiled `.hsaco`, dump it with the AMD object file tools:\n\n"
             "```\nclang-offload-bundler --type=o --inputs=/path/to/kernel.hsaco \\\n  --outputs=kernel.bc --unbundle \\\n  --targets=hipv4-amdgcn-amd-amdhsa--gfx1100\nllvm-readelf --notes kernel.bc | grep -E '(vgpr|sgpr|lds|stack)_count'\n```\n\n"
             "The `gfx-kernel-metadata` skill in this repo automates this. It prints the "
             "VGPR/SGPR/LDS/spill counts and computes theoretical occupancy."),
        ],
    ]
    for convo in coding_help:
        text = _format_chatml_turn("system", base_system)
        for q, a in convo:
            text += _format_chatml_turn("user", q) + _format_chatml_turn("assistant", a)
        exchanges.append(text)

    # Step-by-step how-to.
    howto = [
        ("How do I add a new endpoint to a FastAPI app?",
         "Five steps:\n\n"
         "1. Make sure `fastapi` and an ASGI server (`uvicorn`) are in your environment:\n"
         "   `pip install fastapi uvicorn`.\n"
         "2. Open `app/main.py` (or wherever your `FastAPI()` instance lives).\n"
         "3. Add a route function:\n"
         "   ```python\n"
         "   @app.get('/health')\n"
         "   def health():\n"
         "       return {'status': 'ok'}\n"
         "   ```\n"
         "4. If the endpoint takes structured input, define a Pydantic model:\n"
         "   ```python\n"
         "   from pydantic import BaseModel\n"
         "   class EchoIn(BaseModel):\n"
         "       msg: str\n"
         "   @app.post('/echo')\n"
         "   def echo(body: EchoIn):\n"
         "       return {'echoed': body.msg}\n"
         "   ```\n"
         "5. Restart `uvicorn app.main:app --reload` and curl it: `curl localhost:8000/health`."),
        ("Walk me through profiling a slow Rust binary.",
         "Six steps that catch ~90% of perf issues:\n\n"
         "1. **Confirm release mode.** A debug binary can be 10-100x slower. Run "
         "`cargo build --release` and then profile the binary in `target/release/`.\n"
         "2. **Get a flamegraph.** Install `cargo-flamegraph` and run "
         "`cargo flamegraph --bin your_binary`. Open the SVG in a browser. Look for the "
         "widest bar — that's where you're spending time.\n"
         "3. **Check the top function.** If it's something you didn't expect (e.g. a JSON "
         "parser, or a HashMap rehash), you've found a candidate.\n"
         "4. **Look at memory pressure.** Run `perf stat -e cache-misses,page-faults` to see "
         "if the bottleneck is memory not CPU.\n"
         "5. **Microbenchmark the hot loop** with criterion to confirm changes are real "
         "improvements, not noise.\n"
         "6. **Re-flamegraph after each change.** A 'fix' that doesn't change the flamegraph "
         "shape probably didn't help."),
        ("How do I set up reproducible builds in Docker?",
         "Three things matter: pinning the base image digest, pinning dependency versions, and "
         "controlling the build environment.\n\n"
         "1. Pin the base image by digest, not tag:\n"
         "   `FROM debian@sha256:abcdef...`\n"
         "   Tags can move (e.g. `:bookworm` updates monthly); digests are immutable.\n"
         "2. Pin every dependency. For apt: `apt-get install -y package=version`. For "
         "Python: use `pip install -r requirements.lock` where the lockfile pins exact "
         "versions of every transitive dependency. For Node: commit `package-lock.json`. For "
         "Rust: commit `Cargo.lock` and copy it into the image before `cargo build`.\n"
         "3. Set `SOURCE_DATE_EPOCH` to a fixed unix timestamp; many build tools respect it "
         "and produce identical bytes when set. Pass it as a build-arg.\n"
         "4. Avoid `--no-cache` unless you actually want a fresh build; for reproducibility "
         "you want the same intermediate layers each time.\n"
         "5. Verify by building twice on two different machines and comparing the image "
         "digest (`docker inspect $img | jq .Id`). If they match, you're reproducible."),
    ]
    for q, a in howto:
        exchanges.append(
            _format_chatml_turn("system", base_system)
            + _format_chatml_turn("user", q)
            + _format_chatml_turn("assistant", a)
        )

    # Casual / analogy / open-ended.
    casual = [
        ("Can you explain Kubernetes like I'm a curious junior dev who just wrote their first dockerfile?",
         "Sure. Think of Docker as building a self-contained box (an image) that holds your app and "
         "everything it needs. You can run that box on any machine that has Docker installed.\n\n"
         "Kubernetes is the *manager* for a whole warehouse of boxes. You tell it 'I want 5 of "
         "these boxes running at all times,' and Kubernetes:\n\n"
         "- decides which physical/virtual machines have room for them\n"
         "- starts them, watches them, restarts them if they crash\n"
         "- routes traffic to whichever boxes are alive\n"
         "- can swap them out for new versions one at a time so users never see downtime\n\n"
         "The catch: Kubernetes is a *lot* of YAML and concepts (Pods, Deployments, Services, "
         "Ingress, ConfigMaps, ...). You don't need it to ship a docker container — you only "
         "need it when you have many containers across many machines and you want hands-off "
         "scaling/restart/upgrade behavior. For a side project, plain `docker run` or a "
         "single-server tool like Docker Compose is usually enough."),
        ("What's a good analogy for what a CPU cache does?",
         "Think of the CPU as a baker working at a counter, and main memory as the back-of-house "
         "pantry. Each trip to the pantry takes a long time. The L1 cache is the small set of "
         "ingredients on the counter; L2 is the shelf right behind the baker; L3 is the closet "
         "in the kitchen. The baker grabs whatever they touched recently in the smallest available "
         "place, because reaching there is fastest. When the counter fills up, the least-recently-"
         "used item goes back to a shelf to make room.\n\n"
         "Cache 'hits' are when the ingredient you need is already on the counter; cache 'misses' "
         "are when you have to go to the pantry. The hit rate is what determines how fast you bake."),
        ("Why are RNG seeds so important for reproducibility?",
         "Because every 'random' sequence in software is actually a deterministic function of an "
         "initial state. If you fix the initial state (the seed), the entire sequence of 'random' "
         "draws is fixed, so the same program with the same seed produces identical results.\n\n"
         "Without a seed, the RNG typically initializes from the system clock or `/dev/urandom`, "
         "and two runs of the same program see different random sequences — which is great for "
         "real applications but a nightmare for debugging. If a bug only appears 1 run in 1000 "
         "with random seeding, you can't reproduce it deliberately. Set the seed at the start of "
         "your test, then the failing run becomes reproducible.\n\n"
         "Caveats: parallel code can still be nondeterministic even with seeds, because the order "
         "in which threads draw from the RNG depends on the scheduler. Use per-thread RNGs seeded "
         "by `seed + thread_id` to keep each thread reproducible."),
        ("My team keeps fighting about tabs vs spaces. What should I tell them?",
         "Tell them to pick one, write it down in `.editorconfig`, and let the formatter enforce "
         "it on save. The actual choice doesn't matter — the cost of fighting about it does.\n\n"
         "Some languages have de facto conventions: Python is 4-space indent (PEP 8), Go uses tabs "
         "(`gofmt` enforces), Rust uses 4-space (`rustfmt` enforces). When the language has a "
         "convention, follow it; when it doesn't, default to spaces because diffs are more "
         "predictable.\n\n"
         "Anyone arguing past 'we picked it, here's the linter config, move on' is signaling that "
         "they want a different fight than the one they're having. Redirect to a real engineering "
         "decision."),
        ("If I'm new to ML inference, what's the minimum mental model for what's happening?",
         "Four things:\n\n"
         "1. **The model is a function** — billions of parameters arranged so that "
         "   `f(token_ids) -> next_token_logits`. Inference is one forward pass through this "
         "   function per generated token.\n"
         "2. **Most of the time is matrix multiplies.** A transformer layer is dominated by a "
         "   few large matmuls (Q/K/V projections, output projection, two FFN matmuls). The rest "
         "   is small: norm, softmax, residual adds.\n"
         "3. **Memory bandwidth is the bottleneck for decode.** Each generated token reads the "
         "   full model weights once. A 27B model in INT4 is ~14 GB; on a 1 TB/s GPU, the lower "
         "   bound is ~14 ms/token = 71 tok/s. Real engines are within 50-90% of that ceiling.\n"
         "4. **Quantization saves bandwidth.** Lowering weight precision from FP16 to INT4 cuts "
         "   the bytes-per-token in 4x, which (roughly) quadruples decode throughput. The "
         "   trade-off is small accuracy loss, measured as KL divergence vs the FP16 reference."),
    ]
    for q, a in casual:
        exchanges.append(
            _format_chatml_turn("system", base_system)
            + _format_chatml_turn("user", q)
            + _format_chatml_turn("assistant", a)
        )

    rng.shuffle(exchanges)
    return exchanges


def assemble_code(budget: int, rng: random.Random, logf) -> str:
    """
    Assemble code content from production Rust, HIP kernels, Python scripts.

    Target ratio: Python 60% / Rust 25% / HIP 15%.
    Read whole files and concatenate; the tokenizer chunks at n_ctx.
    """
    out: list[str] = []
    total = 0
    # Target byte ratios for code mix per task brief: Python 60% / Rust 25% / HIP 15%.
    # Note: the in-tree scripts/ corpus is only ~610 KB total, so 60% of a
    # ~2 MB code budget cannot all be Python. We pull every script we have,
    # then top up with Rust (5+ MB available, structurally similar tokenization
    # to HIP-C++) to hit the total code budget. Actual realized split is
    # logged below and reported in the README.
    py_budget = int(budget * 0.60)
    rs_budget = int(budget * 0.25)
    hip_budget = int(budget * 0.15)

    # Python — scripts/ and benchmarks/prompts/humaneval_*
    py_files = []
    py_files.extend(sorted((REPO_ROOT / "scripts").glob("*.py")))
    py_files.extend(sorted((REPO_ROOT / "scripts").glob("**/*.py")))
    py_files.extend(sorted((REPO_ROOT / "benchmarks" / "prompts").glob("humaneval_*.txt")))
    py_files.extend(sorted((REPO_ROOT / "benchmarks" / "prompts").glob("lru_cache_*.txt")))
    py_files.extend(sorted((REPO_ROOT / "benchmarks" / "prompts" / "dirty").glob("*.txt")))
    # De-duplicate while preserving order.
    seen_py = set()
    py_files = [p for p in py_files if p not in seen_py and not seen_py.add(p)]
    py_used = 0
    py_total = 0
    rng.shuffle(py_files)
    for p in py_files:
        if py_total >= py_budget:
            break
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if len(text) < 200:
            continue
        # Prepend a comment header indicating provenance for tokenizer realism;
        # the tokenizer just emits comment tokens, no harm.
        header = f"# source: {p.relative_to(REPO_ROOT)}\n\n"
        out.append(header + text + SEP)
        py_total += len(header) + len(text) + len(SEP)
        py_used += 1
    log(f"  code: pulled {py_used} Python files ({py_total:,} B)", fh=logf)

    # Rust — pick mid-sized files from our crates. Walk the whole crates tree
    # so we get good coverage of arches, runtime, kernels, codec, etc.
    rs_files = []
    for p in sorted((REPO_ROOT / "crates").glob("*/src/**/*.rs")):
        rs_files.append(p)
    # Also include top-level examples / tests for variety.
    for p in sorted((REPO_ROOT / "crates").glob("*/examples/*.rs")):
        rs_files.append(p)
    # Filter to mid-size. Upper bound generous to allow whole-file inclusion of
    # representative production modules.
    rs_files = [p for p in rs_files if p.exists() and 1000 < p.stat().st_size < 80000]
    rs_used = 0
    rs_total = 0
    rng.shuffle(rs_files)
    for p in rs_files:
        if rs_total >= rs_budget:
            break
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        header = f"// source: {p.relative_to(REPO_ROOT)}\n\n"
        out.append(header + text + SEP)
        rs_total += len(header) + len(text) + len(SEP)
        rs_used += 1
    log(f"  code: pulled {rs_used} Rust files ({rs_total:,} B)", fh=logf)

    # HIP — mid-sized kernels.
    hip_files = sorted((REPO_ROOT / "kernels" / "src").glob("*.hip"))
    hip_files = [p for p in hip_files if 1500 < p.stat().st_size < 80000]
    hip_used = 0
    hip_total = 0
    rng.shuffle(hip_files)
    for p in hip_files:
        if hip_total >= hip_budget:
            break
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        header = f"// source: {p.relative_to(REPO_ROOT)}\n\n"
        out.append(header + text + SEP)
        hip_total += len(header) + len(text) + len(SEP)
        hip_used += 1
    log(f"  code: pulled {hip_used} HIP kernels ({hip_total:,} B)", fh=logf)

    # If we under-shot the total code budget because Python ran out, top up with
    # extra Rust (5+ MB of source available) until we hit the target.
    code_total = py_total + rs_total + hip_total
    shortfall = budget - code_total
    if shortfall > 50_000:
        # Pick more Rust files not yet emitted.
        already_used = set()
        for p in (out):
            # Header line is "// source: <relpath>\n\n" — scrape paths to dedupe.
            for line in p.splitlines()[:1]:
                if line.startswith("// source: "):
                    already_used.add(line[len("// source: "):])
                elif line.startswith("# source: "):
                    already_used.add(line[len("# source: "):])
        extras = [p for p in rs_files if str(p.relative_to(REPO_ROOT)) not in already_used]
        topped = 0
        rng.shuffle(extras)
        for p in extras:
            if topped >= shortfall:
                break
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            header = f"// source: {p.relative_to(REPO_ROOT)}\n\n"
            out.append(header + text + SEP)
            topped += len(header) + len(text) + len(SEP)
            rs_used += 1
            rs_total += len(header) + len(text) + len(SEP)
        log(f"  code: topped up with extra Rust to cover shortfall ({topped:,} B added)", fh=logf)
        code_total = py_total + rs_total + hip_total
    log(f"  code: total {code_total:,} B "
        f"(py {py_total:,} / rs {rs_total:,} / hip {hip_total:,}; "
        f"{py_total/code_total*100:.1f}/{rs_total/code_total*100:.1f}/{hip_total/code_total*100:.1f}%)",
        fh=logf)
    blob = "".join(out)
    if len(blob) > budget * 1.05:
        cut = blob.rfind(SEP, 0, budget)
        if cut > 0:
            blob = blob[:cut]
    return blob


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    rng = random.Random(SEED)
    with open(BUILD_LOG, "w") as logf:
        log(f"calibration-mix-v1 build (seed={SEED})", fh=logf)
        log("===========================", fh=logf)
        log(f"target: {TARGET_TOKENS:,} tokens ({TARGET_TOKENS/1024/1024:.2f}M)", fh=logf)
        log(f"output: {OUT}", fh=logf)
        log("", fh=logf)

        # Wiki bucket (deterministic; slice off existing slice).
        log("[1/4] wiki bucket", fh=logf)
        wiki = assemble_wiki(BUDGET_CHARS["wiki"], logf)
        log("", fh=logf)

        # Load hermes once.
        log("[?] loading hermes parquet rows", fh=logf)
        hermes_rows = _load_hermes_rows()
        log(f"  hermes: loaded {len(hermes_rows)} rows total", fh=logf)
        log("", fh=logf)

        # Tool-call bucket.
        log("[2/4] tool-call bucket", fh=logf)
        tool = assemble_tool_calls(BUDGET_CHARS["tool"], hermes_rows, rng, logf)
        log(f"  tool: final {len(tool):,} B", fh=logf)
        log("", fh=logf)

        # Chat bucket (re-shuffle hermes rows for new sampling).
        log("[3/4] chat bucket", fh=logf)
        chat = assemble_chat(BUDGET_CHARS["chat"], list(hermes_rows), rng, logf)
        log(f"  chat: final {len(chat):,} B", fh=logf)
        log("", fh=logf)

        # Code bucket.
        log("[4/4] code bucket", fh=logf)
        code = assemble_code(BUDGET_CHARS["code"], rng, logf)
        log(f"  code: final {len(code):,} B", fh=logf)
        log("", fh=logf)

        # Final assembly: shuffle order of *blocks* (per source-class chunk) with a
        # deterministic interleave. We tag each bucket with a list of (class, text)
        # chunks split at SEP, then interleave them so the corpus mixes classes
        # uniformly rather than being block-segmented.
        def split_chunks(blob: str, klass: str) -> list[tuple[str, str]]:
            parts = [p for p in blob.split(SEP) if p.strip()]
            return [(klass, p) for p in parts]

        chunks = (
            split_chunks(wiki, "wiki")
            + split_chunks(chat, "chat")
            + split_chunks(code, "code")
            + split_chunks(tool, "tool")
        )
        log(f"chunks pre-shuffle: wiki={sum(1 for c in chunks if c[0]=='wiki')} "
            f"chat={sum(1 for c in chunks if c[0]=='chat')} "
            f"code={sum(1 for c in chunks if c[0]=='code')} "
            f"tool={sum(1 for c in chunks if c[0]=='tool')}", fh=logf)

        rng.shuffle(chunks)

        # Write to file. SEP between chunks for clean boundary tokens.
        with open(OUT, "w", encoding="utf-8") as f:
            for i, (_, txt) in enumerate(chunks):
                if i > 0:
                    f.write(SEP)
                f.write(txt)

        size_bytes = OUT.stat().st_size
        log(f"\nwrote {OUT}", fh=logf)
        log(f"  size: {size_bytes:,} bytes ({size_bytes/1024/1024:.2f} MB)", fh=logf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
