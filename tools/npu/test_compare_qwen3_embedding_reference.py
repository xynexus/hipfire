#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

import compare_qwen3_embedding_reference as reference


class Qwen3ReferenceTests(unittest.TestCase):
    def test_encode_captures_each_layers_last_real_token(self) -> None:
        class AddLayer(torch.nn.Module):
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return hidden_states + 1.0

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList([AddLayer(), AddLayer()])

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
                del attention_mask
                hidden = torch.stack((input_ids.float(), input_ids.float() * 2), -1)
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(last_hidden_state=hidden)

        embeddings, layers, stages = reference.encode(TinyModel(), [[1, 2], [3]], "cpu", capture_layers=True)

        self.assertEqual(tuple(embeddings.shape), (2, 2))
        self.assertIsNotNone(layers)
        self.assertEqual(stages, {})
        np.testing.assert_allclose(
            layers.numpy(),
            np.array([[[3, 5], [4, 7]], [[4, 6], [5, 8]]], dtype=np.float32),
        )

    def test_encode_padding_invariant_uses_each_document_width(self) -> None:
        class WidthSensitiveModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Identity()])

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
                del attention_mask
                width = input_ids.shape[1]
                hidden = torch.stack((input_ids.float() + width, input_ids.float() * 2), -1)
                return SimpleNamespace(last_hidden_state=self.layers[0](hidden))

        model = WidthSensitiveModel()
        embeddings, layers, stages = reference.encode_padding_invariant(
            model, [[1], [2, 3]], "cpu", capture_layers=True
        )
        first, first_layers, _ = reference.encode(model, [[1]], "cpu", capture_layers=True)
        second, second_layers, _ = reference.encode(model, [[2, 3]], "cpu", capture_layers=True)

        torch.testing.assert_close(embeddings, torch.cat((first, second)))
        torch.testing.assert_close(layers, torch.cat((first_layers, second_layers), dim=1))
        self.assertEqual(stages, {})

    def test_inverse_fwht_round_trips_more_than_one_vector_slab(self) -> None:
        values = np.arange(513 * reference.GROUP, dtype=np.float32).reshape(513, reference.GROUP)
        values = np.sin(values * np.float32(0.0017))
        rotated = np.empty_like(values)
        for offset in range(0, len(values), 256):
            slab = values[offset : offset + 256] * reference.SIGNS_1
            stride = 1
            while stride < reference.GROUP:
                for start in range(0, reference.GROUP, 2 * stride):
                    left = slab[:, start : start + stride].copy()
                    right = slab[:, start + stride : start + 2 * stride].copy()
                    slab[:, start : start + stride] = left + right
                    slab[:, start + stride : start + 2 * stride] = left - right
                stride *= 2
            rotated[offset : offset + len(slab)] = slab * reference.SIGNS_2 / np.float32(16.0)

        restored = reference.inverse_fwht(rotated)
        np.testing.assert_allclose(restored, values, rtol=0.0, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
