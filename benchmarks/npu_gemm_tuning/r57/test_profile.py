#!/usr/bin/env python3

import unittest

from r57_profile import R34_ATTENTION, accounting_for_columns


class R57ProfileTests(unittest.TestCase):
    def test_exact_r34_geometry(self):
        self.assertEqual(R34_ATTENTION.m, 256)
        self.assertEqual(R34_ATTENTION.k, 768)
        self.assertEqual(R34_ATTENTION.n, 1280)
        self.assertEqual(R34_ATTENTION.blocks_per_column, 125)
        self.assertEqual(R34_ATTENTION.bytes_per_column, 2_048_000)

    def test_four_columns_match_runtime_assertions(self):
        accounting = accounting_for_columns(R34_ATTENTION, 4)
        self.assertTrue(accounting.production_exact)
        self.assertEqual(accounting.wire_bytes, 8_192_000)
        self.assertEqual(accounting.nonpadding_bytes, 8_159_360)
        self.assertEqual(accounting.semantic_unique_bytes, 2_558_980)
        self.assertAlmostEqual(accounting.wire_over_unique, 3.2012755168074785)

    def test_scaling_controls_are_labeled_nonproduction(self):
        for columns in (1, 2, 8):
            with self.subTest(columns=columns):
                accounting = accounting_for_columns(R34_ATTENTION, columns)
                self.assertFalse(accounting.production_exact)
                self.assertEqual(
                    accounting.wire_bytes,
                    columns * R34_ATTENTION.bytes_per_column,
                )

    def test_invalid_column_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "columns"):
            accounting_for_columns(R34_ATTENTION, 0)


if __name__ == "__main__":
    unittest.main()
