#!/usr/bin/env python3
"""CPU oracle for R58's signed, low-nibble-first AIE decoder."""

import unittest


def sext4(value: int) -> int:
    value &= 0xF
    return value - 16 if value & 8 else value


def decode_low_high(packed: bytes) -> list[int]:
    return [decoded for byte in packed for decoded in (sext4(byte), sext4(byte >> 4))]


def lane_sums(payload: bytes, *, tile_bytes: int = 16_384, data_bytes: int = 12_288) -> list[int]:
    if len(payload) % tile_bytes:
        raise ValueError("payload must contain complete production tiles")
    totals = [0] * 64
    for tile_offset in range(0, len(payload), tile_bytes):
        data = payload[tile_offset : tile_offset + data_bytes]
        for vector_offset in range(0, len(data), 32):
            for lane, value in enumerate(decode_low_high(data[vector_offset : vector_offset + 32])):
                totals[lane] += value
    return totals


class DecodeOracleTests(unittest.TestCase):
    def test_signed_low_nibble_first_extremes(self):
        self.assertEqual(decode_low_high(bytes([0x87, 0xF0, 0x18])), [7, -8, 0, -1, -8, 1])

    def test_lane_sums_ignore_scale_tail(self):
        tile = bytearray(16_384)
        tile[:32] = bytes([0x87] * 32)
        tile[12_288:] = bytes([0xFF] * (16_384 - 12_288))
        self.assertEqual(lane_sums(bytes(tile)), [7, -8] * 32)

    def test_rejects_partial_tile(self):
        with self.assertRaisesRegex(ValueError, "complete production tiles"):
            lane_sums(bytes(16_383))


if __name__ == "__main__":
    unittest.main()
