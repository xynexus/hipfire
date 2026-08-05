# chatglm3-6b-onnx-ryzenai-npu — NPU compute bandwidth

`amd/chatglm3-6b-onnx-ryzenai-npu`, copied locally from `/srv/huggingface`.

This is the *compute*-side companion to the storage/load measurement (cold
5.67 GB/s, warm 20.52 GB/s over the 6.54 GB on-disk set). Decode on a dense
model is memory-bound — every weight is read once per token — so the NPU
question is: **what weight-streaming rate can this NPU sustain, and what token
rate does that imply for this model?**

## What was measured

`tools/npu/flm/dispatch_bw_probe.py` streams host buffers through one dispatch
and reports the achieved rate. Sweeping buffer size and worker count:

| bufs x size | workers | GB/s | vs FLM 46.2 | % of 56.5 roof |
|---|---|---|---|---|
| 16 x 1 MiB | 4 | 33.5 | 0.72x | 59% |
| 16 x 1 MiB | 8 | 38.1 | 0.82x | 67% |
| 16 x 4 MiB | 4 | 42.0 | 0.91x | 74% |
| 16 x 2 MiB | 16 | 44.0 | 0.95x | 78% |
| 16 x 4 MiB | 8 | 52.3 | 1.13x | 93% |
| 16 x 8 MiB | 8 | 53.9 | 1.17x | 95% |
| **16 x 16 MiB** | **8** | **54.9** | **1.19x** | **97%** |

Bandwidth is dominated by **transfer size, not worker count**: 8 workers at
16 MiB beats 16 workers at 2 MiB by 25%. Per-transfer overhead is what the
small-buffer configurations are paying.

`--bufs 48` fails with `ERT_CMD_STATE_ERROR`, so this build does not carry the
`kMaxHostBOs` 16 -> 64 raise that `dispatch_bw_probe.py`'s docstring assumes.
Everything above is therefore within the stock 16-BO limit — the 50-buffer
configuration FLM itself uses was not reachable here.

## What it implies for chatglm3

Geometry from `genai_config.json`: 28 layers, hidden 4096, 32 heads / 2 KV
heads (16:1 GQA), head_size 128, vocab 65024, context 8192, `MatMulNBits`
4-bit.

The per-token weight stream is `dd_metastate_Llm_Token_MatMulNBits_2_0.fconst`
= **3.231 GB**. That this file is the quantized weight set and not a container
is worth checking rather than assuming: the geometry gives ~5.98 B parameters,
which at 4 bits plus g128 fp16 scales/zeros budgets to 3.13 GB — within 3% of
the file. So "every weight once per token" is the right model.

| streaming rate | decode ceiling |
|---|---|
| 54.9 GB/s (measured peak) | **17.0 tok/s** |
| 46.2 GB/s (FLM's rate) | 14.3 tok/s |
| 33.5 GB/s (small buffers) | 10.4 tok/s |

## Caveats

This is a **ceiling from a synthetic streaming probe**, not an end-to-end
decode measurement of chatglm3. It says what the fabric can deliver and what
that permits; it does not say what the RyzenAI ONNX stack achieves, and the
two differ whenever dispatch structure, not bandwidth, is the limit — which is
exactly the gap `dispatch_bw_probe.py` exists to isolate (hipfire's own decode
path delivers ~10 GB/s effective against this same fabric).

An end-to-end number would need onnxruntime with the VitisAI EP, which is not
installed on this machine (no `onnxruntime` in any environment here), and the
RyzenAI ONNX path is Windows-oriented. The NPU driver itself is live
(`amdxdna`, `/dev/accel/accel0`, XRT at `/opt/xilinx/xrt`).
