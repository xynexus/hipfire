#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase-2 runner: the vendored IRON llama_3.2_1b whole-model app, with the
AIE/MLIR LLVM loaded BEFORE torch to avoid the duelling-LLVM llvm::cl segfault.

Must be launched via run_llama.sh (which sets the pinned-toolchain overlay env).
"""
import sys, os, runpy

APP = "/home/sadara/.hipfire/src/third_party/IRON/iron/applications/llama_3.2_1b"
SNAP = "/srv/huggingface/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"

sys.path.insert(0, APP)
os.chdir(APP)

# Load libAIEAggregateCAPI (and its LLVM cl::opt registrations) before torch's
# ROCm libLLVM registers the same options — otherwise torch-first segfaults.
import aie.iron  # noqa: F401,E402

args = sys.argv[1:] or ["--prompt-len", "13", "--num-tokens", "3"]
sys.argv = [f"{APP}/llama_npu.py",
            f"{SNAP}/model.safetensors",
            f"{SNAP}/original/tokenizer.model", *args]
runpy.run_path(f"{APP}/llama_npu.py", run_name="__main__")
