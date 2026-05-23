#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from learn_butterfly_mq import (  # noqa: E402
    ButterflyResidualPseudoQuantLinear,
    FullRotationPseudoQuantLinear,
    LearnableSignsPseudoQuantLinear,
    N,
    _select_wrapper_class,
    quantize_mq4g256_signs_ste,
    sign_ste,
)


def test_group_signs_initialize_to_tensor_baseline():
    torch.manual_seed(123)
    weight = torch.randn(4, N * 2, dtype=torch.float32)
    x = torch.randn(3, N * 2, dtype=torch.float32)
    scales = torch.ones(N * 2, dtype=torch.float32)

    tensor_wide = LearnableSignsPseudoQuantLinear(
        weight, None, scales, "fixture.tensor", sign_granularity="tensor",
    )
    group_wide = LearnableSignsPseudoQuantLinear(
        weight, None, scales, "fixture.tensor", sign_granularity="group",
    )

    assert tensor_wide.d1_logits.shape == (N,)
    assert group_wide.d1_logits.shape == (2, N)
    torch.testing.assert_close(group_wide(x), tensor_wide(x), rtol=0.0, atol=0.0)


def test_quant_mode_uniform_is_default_byte_equal():
    torch.manual_seed(234)
    weight = torch.randn(3, N * 2, dtype=torch.float32)
    wrapper = LearnableSignsPseudoQuantLinear(
        weight, None, torch.ones(N * 2), "fixture.tensor", sign_granularity="group",
    )

    d1 = sign_ste(wrapper.d1_logits)
    d2 = sign_ste(wrapper.d2_logits)

    default = quantize_mq4g256_signs_ste(weight, d1, d2)
    explicit = quantize_mq4g256_signs_ste(weight, d1, d2, quant_mode="uniform")

    torch.testing.assert_close(default, explicit, rtol=0.0, atol=0.0)


def test_lloyd_quant_mode_forward_and_backward_work():
    torch.manual_seed(345)
    weight = torch.randn(2, N * 3, dtype=torch.float32)
    wrapper = LearnableSignsPseudoQuantLinear(
        weight, None, torch.ones(N * 3), "fixture.tensor",
        sign_granularity="group", quant_mode="lloyd",
    )
    wrapper.set_optim()
    x = torch.randn(4, N * 3, dtype=torch.float32)

    y = wrapper(x)
    loss = y.pow(2).mean()
    loss.backward()

    assert y.shape == (4, 2)
    assert wrapper.quant_mode == "lloyd"
    assert wrapper.d1_logits.grad is not None
    assert wrapper.d2_logits.grad is not None
    assert torch.isfinite(wrapper.d1_logits.grad).all()
    assert torch.isfinite(wrapper.d2_logits.grad).all()


def test_group_signs_do_not_cross_talk_between_256_wide_groups():
    torch.manual_seed(456)
    weight = torch.randn(3, N * 2, dtype=torch.float32)
    wrapper = LearnableSignsPseudoQuantLinear(
        weight, None, torch.ones(N * 2), "fixture.tensor", sign_granularity="group",
    )

    d1_base = sign_ste(wrapper.d1_logits)
    d2_base = sign_ste(wrapper.d2_logits)
    d1_flip_second = d1_base.clone()
    d1_flip_second[1].mul_(-1.0)

    base = quantize_mq4g256_signs_ste(weight, d1_base, d2_base)
    flipped = quantize_mq4g256_signs_ste(weight, d1_flip_second, d2_base)

    torch.testing.assert_close(flipped[:, :N], base[:, :N], rtol=0.0, atol=0.0)
    assert not torch.equal(flipped[:, N:], base[:, N:])


def test_full_rotation_init_matches_fwht_signs_baseline_byte_equal():
    torch.manual_seed(567)
    weight = torch.randn(4, N * 2, dtype=torch.float32)
    bias = torch.randn(4, dtype=torch.float32)
    scales = torch.rand(N * 2, dtype=torch.float32) + 0.5
    x = torch.randn(3, N * 2, dtype=torch.float32)

    fwht_signs = LearnableSignsPseudoQuantLinear(
        weight, bias, scales, "fixture.tensor", sign_granularity="group",
    )
    full_rotation = FullRotationPseudoQuantLinear(
        weight, bias, scales, "fixture.tensor",
    )

    assert full_rotation.theta.shape == (2, 8, 128)
    assert full_rotation.quant_mode == "uniform"
    assert full_rotation.quant_bits == 4
    torch.testing.assert_close(full_rotation(x), fwht_signs(x), rtol=0.0, atol=0.0)


def test_full_rotation_theta_moves_after_training_step():
    torch.manual_seed(678)
    weight = torch.randn(3, N * 2, dtype=torch.float32)
    scales = torch.ones(N * 2, dtype=torch.float32)
    x = torch.randn(5, N * 2, dtype=torch.float32)
    target = torch.randn(5, 3, dtype=torch.float32)
    wrapper = FullRotationPseudoQuantLinear(
        weight, None, scales, "fixture.tensor",
    )
    wrapper.set_optim()
    initial = wrapper.theta.detach().clone()
    optimizer = torch.optim.SGD([wrapper.theta], lr=0.1, momentum=0.0)

    optimizer.zero_grad(set_to_none=True)
    loss = (wrapper(x).to(torch.float32) - target).pow(2).mean()
    loss.backward()

    assert wrapper.theta.grad is not None
    assert torch.isfinite(wrapper.theta.grad).all()
    assert wrapper.theta.grad.abs().sum() > 0.0
    optimizer.step()

    assert not torch.equal(wrapper.theta.detach(), initial)


def test_rotation_mode_fwht_signs_preserves_legacy_default_byte_equal():
    torch.manual_seed(789)
    weight = torch.randn(2, N, dtype=torch.float32)
    scales = torch.rand(N, dtype=torch.float32) + 0.5
    x = torch.randn(4, N, dtype=torch.float32)
    args = type("Args", (), {"rotation_mode": "fwht-signs", "learnable_signs": False})()

    legacy = ButterflyResidualPseudoQuantLinear(weight, None, scales, "fixture.tensor")
    selected_cls = _select_wrapper_class(args)
    selected = selected_cls(weight, None, scales, "fixture.tensor")

    assert selected_cls is ButterflyResidualPseudoQuantLinear
    torch.testing.assert_close(selected(x), legacy(x), rtol=0.0, atol=0.0)


def test_rotation_mode_fwht_signs_preserves_learnable_signs_byte_equal():
    torch.manual_seed(890)
    weight = torch.randn(2, N * 2, dtype=torch.float32)
    scales = torch.ones(N * 2, dtype=torch.float32)
    x = torch.randn(4, N * 2, dtype=torch.float32)
    args = type("Args", (), {"rotation_mode": "fwht-signs", "learnable_signs": True})()

    legacy = LearnableSignsPseudoQuantLinear(
        weight, None, scales, "fixture.tensor", sign_granularity="group",
    )
    selected_cls = _select_wrapper_class(args)
    selected = selected_cls(
        weight, None, scales, "fixture.tensor", sign_granularity="group",
    )

    assert selected_cls is LearnableSignsPseudoQuantLinear
    torch.testing.assert_close(selected(x), legacy(x), rtol=0.0, atol=0.0)


if __name__ == "__main__":
    test_group_signs_initialize_to_tensor_baseline()
    test_quant_mode_uniform_is_default_byte_equal()
    test_lloyd_quant_mode_forward_and_backward_work()
    test_group_signs_do_not_cross_talk_between_256_wide_groups()
    test_full_rotation_init_matches_fwht_signs_baseline_byte_equal()
    test_full_rotation_theta_moves_after_training_step()
    test_rotation_mode_fwht_signs_preserves_legacy_default_byte_equal()
    test_rotation_mode_fwht_signs_preserves_learnable_signs_byte_equal()
    print("learnable FWHT tests passed")
