#!/usr/bin/env python3
"""Pure geometry and byte accounting for R57 bandwidth-first probes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionProfile:
    name: str
    m: int
    k: int
    n: int
    production_columns: int
    tile_bytes: int
    qkv_blocks_per_column: int
    output_blocks_per_column: int
    norm_blocks_per_column: int
    norm_used_bytes_per_block: int
    qkv_semantic_bytes: int
    output_semantic_bytes: int
    residual_semantic_bytes: int
    norm_semantic_bytes: int

    @property
    def blocks_per_column(self) -> int:
        return (
            self.qkv_blocks_per_column
            + self.output_blocks_per_column
            + self.norm_blocks_per_column
        )

    @property
    def bytes_per_column(self) -> int:
        return self.blocks_per_column * self.tile_bytes

    @property
    def nonpadding_bytes_per_column(self) -> int:
        return (
            (self.qkv_blocks_per_column + self.output_blocks_per_column)
            * self.tile_bytes
            + self.norm_blocks_per_column * self.norm_used_bytes_per_block
        )


@dataclass(frozen=True)
class ProfileAccounting:
    columns: int
    production_exact: bool
    wire_bytes: int
    nonpadding_bytes: int
    semantic_unique_bytes: int

    @property
    def wire_over_unique(self) -> float:
        return self.wire_bytes / self.semantic_unique_bytes

    @property
    def nonpadding_fraction(self) -> float:
        return self.nonpadding_bytes / self.wire_bytes


# Mirrors crates/hipfire-xdna/src/resident_embedding_layer.rs (R34):
#   QKV:   4 active columns * 45 blocks * 16 KiB
#   O:     4 active columns * 72 blocks * 16 KiB
#   state: 4 active columns * 8 blocks * 16 KiB
# The state block's last meaningful byte is epsilon at offset 15,360 + 4.
R34_ATTENTION = ProductionProfile(
    name="embeddinggemma-r34-attention-w8",
    m=256,
    k=768,
    n=1280,
    production_columns=4,
    tile_bytes=16_384,
    qkv_blocks_per_column=45,
    output_blocks_per_column=72,
    norm_blocks_per_column=8,
    norm_used_bytes_per_block=15_364,
    qkv_semantic_bytes=768 * 1280,  # dense signed-byte QKV weights
    output_semantic_bytes=768 * 768 * 2,  # BF16 output projection
    residual_semantic_bytes=256 * 768 * 2,  # BF16 residual rows
    norm_semantic_bytes=2 * 768 * 2 + 4,  # two BF16 norm vectors + epsilon
)


def accounting_for_columns(
    profile: ProductionProfile, columns: int
) -> ProfileAccounting:
    """Account a production run or a column-scaling control.

    Controls above the production column count repeat the production stream set;
    semantic bytes remain unique rather than being inflated by that repetition.
    Controls below it cover the corresponding QKV/residual column fraction, while
    each selected output-projection stream still carries a complete duplicated O.
    """

    if columns <= 0:
        raise ValueError("columns must be positive")

    active = min(columns, profile.production_columns)
    qkv = profile.qkv_semantic_bytes * active // profile.production_columns
    residual = (
        profile.residual_semantic_bytes * active // profile.production_columns
    )
    semantic = qkv + residual
    if active:
        semantic += profile.output_semantic_bytes + profile.norm_semantic_bytes

    return ProfileAccounting(
        columns=columns,
        production_exact=columns == profile.production_columns,
        wire_bytes=columns * profile.bytes_per_column,
        nonpadding_bytes=columns * profile.nonpadding_bytes_per_column,
        semantic_unique_bytes=semantic,
    )
