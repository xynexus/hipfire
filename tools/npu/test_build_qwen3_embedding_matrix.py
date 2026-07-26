#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import build_qwen3_embedding_matrix as matrix


class BuildFingerprintTests(unittest.TestCase):
    def test_cached_image_requires_matching_command_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "image"
            output.mkdir()
            (output / "final.xclbin").write_bytes(b"xclbin")
            (output / "insts.bin").write_bytes(b"instructions")
            builder = root / "builder.py"
            kernel = root / "kernel.cc"
            builder.write_text("builder-v1\n")
            kernel.write_text("kernel-v1\n")
            command = ["python", str(builder), "--rows", "256"]
            sources = (builder, kernel)

            matrix.write_build_fingerprint(output, command, sources)
            self.assertTrue(matrix.cached_image_ready(output, matrix.image_ready, command, sources))

            kernel.write_text("kernel-v2\n")
            self.assertFalse(matrix.cached_image_ready(output, matrix.image_ready, command, sources))
            kernel.write_text("kernel-v1\n")
            self.assertFalse(
                matrix.cached_image_ready(
                    output,
                    matrix.image_ready,
                    [*command, "--batch", "2"],
                    sources,
                )
            )

    def test_legacy_or_malformed_stamp_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "image"
            output.mkdir()
            (output / "final.xclbin").write_bytes(b"xclbin")
            (output / "insts.bin").write_bytes(b"instructions")
            source = root / "builder.py"
            source.write_text("builder\n")
            command = ["python", str(source)]

            self.assertFalse(matrix.cached_image_ready(output, matrix.image_ready, command, (source,)))
            (output / matrix.BUILD_FINGERPRINT).write_text(json.dumps({"schema": "wrong"}))
            self.assertFalse(matrix.cached_image_ready(output, matrix.image_ready, command, (source,)))


if __name__ == "__main__":
    unittest.main()
