from __future__ import annotations

import pytest

from aiecost.target import Target, infer_device_key, resolve_target


def test_infer_device_key_covers_xdna_generations() -> None:
    assert infer_device_key("RyzenAI-npu1") == "npu1"
    assert infer_device_key("Phoenix") == "npu1"
    assert infer_device_key("NPU Strix Halo", architecture="aie2p") == "npu2"
    assert infer_device_key("RyzenAI-npu4") == "npu2"


def test_explicit_targets_carry_compile_and_topology_contracts() -> None:
    npu1 = resolve_target("npu1")
    npu2 = resolve_target("npu2")

    assert npu1 == Target(
        key="npu1",
        tile_isa="AIE2",
        target_arch="aie2",
        compute_columns=4,
        compute_cores=16,
    )
    assert npu2 == Target(
        key="npu2",
        tile_isa="AIE2P",
        target_arch="aie2p",
        compute_columns=8,
        compute_cores=32,
    )
    assert npu1.cache_tag == "npu1-aie2"
    assert npu2.cache_tag == "npu2-aie2p"
    assert npu1.runtime_library_name == "AIE2"
    assert npu2.runtime_library_name == "AIE2P"


def test_unknown_device_is_rejected_instead_of_guessed() -> None:
    with pytest.raises(ValueError, match="unsupported NPU identity"):
        infer_device_key("mystery accelerator")


def test_k1_uses_native_vmac_shape_and_sized_output_per_isa() -> None:
    from aiecost.benches.k1_clock import select_plateau, vmac_geometry

    assert vmac_geometry("npu1") == (4, 64, 128, 40)
    assert vmac_geometry("npu2") == (8, 128, 128, 72)
    assert select_plateau({1: {"vmac_per_s": 1.0}, 2: {"vmac_per_s": 1.05}}) == (
        (1, 2),
        1.025,
    )
    assert select_plateau(
        {
            1: {"vmac_per_s": 1.0},
            2: {"vmac_per_s": 2.0},
            4: {"vmac_per_s": 4.0},
            8: {"vmac_per_s": 1.2},
        }
    ) == (None, 4.0)
