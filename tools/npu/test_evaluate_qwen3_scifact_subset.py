#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np

import evaluate_qwen3_scifact_subset as scifact


class SciFactSubsetTests(unittest.TestCase):
    def test_subset_keeps_all_relevant_documents_and_is_stable(self) -> None:
        queries = [{"_id": str(index), "text": str(index)} for index in range(5)]
        corpus = [{"_id": str(index), "title": "", "text": str(index)} for index in range(20)]
        qrels = [
            {"query-id": "1", "corpus-id": "17", "score": "1"},
            {"query-id": "1", "corpus-id": "3", "score": "1"},
            {"query-id": "2", "corpus-id": "8", "score": "1"},
        ]

        first = scifact.select_subset(queries, corpus, qrels, 2, 8, "seed")
        second = scifact.select_subset(queries, corpus, qrels, 2, 8, "seed")

        self.assertEqual(first, second)
        self.assertEqual([row["_id"] for row in first[0]], ["1", "2"])
        self.assertTrue({"3", "8", "17"}.issubset({row["_id"] for row in first[1]}))

    def test_ndcg_at_10_is_one_for_ideal_ranking(self) -> None:
        scores = np.array([0.5, 0.9, 0.1], dtype=np.float32)
        self.assertEqual(scifact.ndcg_at_10(scores, ["a", "b", "c"], {"b"}), 1.0)

    def test_ndcg_at_10_penalizes_a_late_relevant_document(self) -> None:
        scores = np.array([0.9, 0.8, 0.1], dtype=np.float32)
        actual = scifact.ndcg_at_10(scores, ["a", "b", "c"], {"c"})
        self.assertAlmostEqual(actual, 0.5)


if __name__ == "__main__":
    unittest.main()
