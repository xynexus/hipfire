#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from learn_butterfly_mq import (  # noqa: E402
    LearnableSignsPseudoQuantLinear,
    N,
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


if __name__ == "__main__":
    test_group_signs_initialize_to_tensor_baseline()
    test_group_signs_do_not_cross_talk_between_256_wide_groups()
    print("learnable FWHT tests passed")
