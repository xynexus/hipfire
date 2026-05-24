#!/usr/bin/env python3
from __future__ import annotations

import sys
import subprocess
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from learn_butterfly_mq import (  # noqa: E402
    ButterflyResidualPseudoQuantLinear,
    FullRotationPseudoQuantLinear,
    LearnableSignsPseudoQuantLinear,
    N,
    _enable_student_gradient_checkpointing,
    _lloyd_max_dequant_per_group,
    _release_oracle_model,
    _select_wrapper_class,
    _uniform_minmax_dequant_per_group,
    cache_oracle_logits,
    compute_logit_kld,
    quantize_mq4g256_signs_ste,
    sign_ste,
    train_kld_loss,
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


def test_lloyd_subscale_forward_and_backward_work_for_64_and_128():
    torch.manual_seed(346)
    weight = torch.randn(2, N * 2, dtype=torch.float32)
    x = torch.randn(4, N * 2, dtype=torch.float32)

    for subscale_size in (64, 128):
        wrapper = LearnableSignsPseudoQuantLinear(
            weight, None, torch.ones(N * 2), "fixture.tensor",
            sign_granularity="group", quant_mode="lloyd",
            subscale_size=subscale_size,
        )
        wrapper.set_optim()

        y = wrapper(x)
        loss = y.pow(2).mean()
        loss.backward()

        assert y.shape == (4, 2)
        assert wrapper.subscale_size == subscale_size
        assert torch.isfinite(y).all()
        assert wrapper.d1_logits.grad is not None
        assert wrapper.d2_logits.grad is not None
        assert torch.isfinite(wrapper.d1_logits.grad).all()
        assert torch.isfinite(wrapper.d2_logits.grad).all()


def test_subscale_dequant_handles_zero_range_blocks():
    rotated = torch.zeros(2, 2, N, dtype=torch.float32)
    rotated[..., :64] = 3.0
    rotated[..., 64:128] = -1.0
    rotated[..., 128:192] = 2.0
    rotated[..., 192:] = 0.5

    uniform = _uniform_minmax_dequant_per_group(rotated, 4, subscale_size=64)
    lloyd = _lloyd_max_dequant_per_group(rotated, 4, subscale_size=64)

    assert uniform.shape == rotated.shape
    assert lloyd.shape == rotated.shape
    assert torch.isfinite(uniform).all()
    assert torch.isfinite(lloyd).all()
    torch.testing.assert_close(uniform, rotated, rtol=0.0, atol=0.0)
    torch.testing.assert_close(lloyd, rotated, rtol=0.0, atol=0.0)


def test_default_residual_supports_lloyd_subscale64_forward():
    torch.manual_seed(347)
    weight = torch.randn(3, N * 2, dtype=torch.float32)
    scales = torch.ones(N * 2, dtype=torch.float32)
    wrapper = ButterflyResidualPseudoQuantLinear(
        weight, None, scales, "fixture.tensor",
        quant_mode="lloyd", subscale_size=64,
    )
    x = torch.randn(4, N * 2, dtype=torch.float32)

    y = wrapper(x)

    assert y.shape == (4, 3)
    assert wrapper.quant_mode == "lloyd"
    assert wrapper.subscale_size == 64
    assert torch.isfinite(y).all()


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


def test_full_rotation_theta_gets_grad_under_nested_non_reentrant_checkpoint():
    torch.manual_seed(679)
    weight = torch.randn(3, N * 2, dtype=torch.float32)
    scales = torch.ones(N * 2, dtype=torch.float32)
    x = torch.randn(5, N * 2, dtype=torch.float32)
    target = torch.randn(5, 3, dtype=torch.float32)
    wrapper = FullRotationPseudoQuantLinear(
        weight, None, scales, "fixture.tensor",
    )
    wrapper.set_optim()

    y = checkpoint(wrapper, x, use_reentrant=False)
    loss = (y.to(torch.float32) - target).pow(2).mean()
    loss.backward()

    assert wrapper.theta.grad is not None
    assert torch.isfinite(wrapper.theta.grad).all()
    assert wrapper.theta.grad.abs().sum() > 0.0


def test_learnable_signs_get_grads_under_non_reentrant_checkpoint():
    torch.manual_seed(680)
    weight = torch.randn(3, N * 2, dtype=torch.float32)
    scales = torch.ones(N * 2, dtype=torch.float32)
    x = torch.randn(5, N * 2, dtype=torch.float32)
    target = torch.randn(5, 3, dtype=torch.float32)
    wrapper = LearnableSignsPseudoQuantLinear(
        weight, None, scales, "fixture.tensor", sign_granularity="group",
    )
    wrapper.set_optim()

    y = checkpoint(wrapper, x, use_reentrant=False)
    loss = (y.to(torch.float32) - target).pow(2).mean()
    loss.backward()

    assert wrapper.d1_logits.grad is not None
    assert wrapper.d2_logits.grad is not None
    assert torch.isfinite(wrapper.d1_logits.grad).all()
    assert torch.isfinite(wrapper.d2_logits.grad).all()
    assert wrapper.d1_logits.grad.abs().sum() > 0.0
    assert wrapper.d2_logits.grad.abs().sum() > 0.0


def test_enable_student_gradient_checkpointing_disables_cache_before_enable():
    class FakeConfig:
        use_cache = True

    class FakeStudent:
        def __init__(self):
            self.config = FakeConfig()
            self.events = []

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            self.events.append((
                "enable",
                self.config.use_cache,
                gradient_checkpointing_kwargs,
            ))

    student = FakeStudent()

    _enable_student_gradient_checkpointing(student)

    assert student.config.use_cache is False
    assert student.events == [
        ("enable", False, {"use_reentrant": False}),
    ]


def test_grad_checkpoint_help_mentions_large_model_memory_reduction():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "learn_butterfly_mq.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--grad-checkpoint" in result.stdout
    assert "--cache-oracle-logits" in result.stdout
    assert "--subscale-size" in result.stdout
    assert "27B/A3B" in result.stdout
    assert "memory" in result.stdout


def test_train_kld_loss_uses_train_mode_only_when_grad_checkpointing():
    class TinyWrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta_residual = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        def set_optim(self):
            self.theta_residual.requires_grad = True

    class TinyOracle(torch.nn.Module):
        def forward(self, input_ids):
            logits = torch.zeros(1, input_ids.shape[1], 3, device=input_ids.device)
            logits[..., 0] = 2.0
            return SimpleNamespace(logits=logits)

    class TinyStudent(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.forward_training_states = []

        def forward(self, input_ids):
            self.forward_training_states.append(self.training)
            logits = torch.zeros(1, input_ids.shape[1], 3, device=input_ids.device)
            logits[..., 1] = self.wrapped.theta_residual
            return SimpleNamespace(logits=logits)

    wrapped = TinyWrapped()
    student = TinyStudent(wrapped)
    student.eval()

    train_kld_loss(
        wrapped={"fixture.tensor": wrapped},
        oracle=TinyOracle(),
        student=student,
        seqs=[torch.tensor([1, 2])],
        device=torch.device("cpu"),
        n_epochs=1,
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        cosine_floor=0.05,
        grad_clip=None,
        log_interval=1,
        grad_checkpoint=True,
    )

    assert student.forward_training_states == [True]
    assert student.training is False
    assert wrapped.theta_residual.grad is not None


def test_cache_oracle_logits_stores_bf16_cpu_and_allows_release():
    class TinyOracle(torch.nn.Module):
        config = SimpleNamespace(vocab_size=4)

        def forward(self, input_ids):
            base = input_ids.to(torch.float32).unsqueeze(-1)
            offsets = torch.arange(4, dtype=torch.float32, device=input_ids.device)
            return SimpleNamespace(logits=base * 0.01 + offsets)

    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    oracle = TinyOracle()

    cache = cache_oracle_logits(oracle, seqs, torch.device("cpu"))
    oracle = _release_oracle_model(oracle)

    assert oracle is None
    assert sorted(cache) == [0, 1]
    assert cache[0].shape == (3, 4)
    assert cache[1].shape == (2, 4)
    assert cache[0].dtype == torch.bfloat16
    assert cache[0].device.type == "cpu"


def test_compute_logit_kld_cache_matches_live_with_bf16_tolerance():
    class TinyOracle(torch.nn.Module):
        config = SimpleNamespace(vocab_size=5)

        def forward(self, input_ids):
            positions = torch.arange(input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
            vocab = torch.arange(5, dtype=torch.float32, device=input_ids.device)
            logits = input_ids.to(torch.float32).unsqueeze(-1) * 0.02
            logits = logits + positions.view(1, -1, 1) * 0.03 + vocab.view(1, 1, -1) * 0.04
            return SimpleNamespace(logits=logits)

    class TinyStudent(torch.nn.Module):
        def forward(self, input_ids):
            positions = torch.arange(input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
            vocab = torch.arange(5, dtype=torch.float32, device=input_ids.device)
            logits = input_ids.to(torch.float32).unsqueeze(-1) * -0.01
            logits = logits + positions.view(1, -1, 1) * 0.01 + vocab.view(1, 1, -1) * 0.02
            return SimpleNamespace(logits=logits)

    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    device = torch.device("cpu")
    oracle = TinyOracle()
    student = TinyStudent()

    live = compute_logit_kld(oracle, student, seqs, device)
    cache = cache_oracle_logits(oracle, seqs, device)
    cached = compute_logit_kld(None, student, seqs, device, oracle_logit_cache=cache)

    assert abs(live - cached) < 5e-3


def test_train_kld_loss_uses_cached_oracle_logits_and_default_live_path():
    class TinyWrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta_residual = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        def set_optim(self):
            self.theta_residual.requires_grad = True

    class TinyOracle(torch.nn.Module):
        config = SimpleNamespace(vocab_size=3)

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            logits = torch.zeros(1, input_ids.shape[1], 3, device=input_ids.device)
            logits[..., 0] = 1.5
            logits[..., 2] = input_ids.to(torch.float32) * 0.01
            return SimpleNamespace(logits=logits)

    class TinyStudent(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids):
            logits = torch.zeros(1, input_ids.shape[1], 3, device=input_ids.device)
            logits[..., 1] = self.wrapped.theta_residual
            return SimpleNamespace(logits=logits)

    seqs = [torch.tensor([1, 2])]
    device = torch.device("cpu")

    live_wrapped = TinyWrapped()
    live_oracle = TinyOracle()
    live_trace = train_kld_loss(
        wrapped={"fixture.tensor": live_wrapped},
        oracle=live_oracle,
        student=TinyStudent(live_wrapped),
        seqs=seqs,
        device=device,
        n_epochs=1,
        lr=0.0,
        momentum=0.0,
        weight_decay=0.0,
        cosine_floor=0.05,
        grad_clip=None,
        log_interval=1,
    )

    cache_oracle = TinyOracle()
    cache = cache_oracle_logits(cache_oracle, seqs, device)
    cached_wrapped = TinyWrapped()
    forbidden_oracle = TinyOracle()
    cached_trace = train_kld_loss(
        wrapped={"fixture.tensor": cached_wrapped},
        oracle=forbidden_oracle,
        student=TinyStudent(cached_wrapped),
        seqs=seqs,
        device=device,
        n_epochs=1,
        lr=0.0,
        momentum=0.0,
        weight_decay=0.0,
        cosine_floor=0.05,
        grad_clip=None,
        log_interval=1,
        oracle_logit_cache=cache,
    )

    assert live_oracle.calls == 1
    assert cache_oracle.calls == 1
    assert forbidden_oracle.calls == 0
    assert abs(
        live_trace[0]["mean_kld_loss"] - cached_trace[0]["mean_kld_loss"]
    ) < 5e-3


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
    test_lloyd_subscale_forward_and_backward_work_for_64_and_128()
    test_subscale_dequant_handles_zero_range_blocks()
    test_default_residual_supports_lloyd_subscale64_forward()
    test_group_signs_do_not_cross_talk_between_256_wide_groups()
    test_full_rotation_init_matches_fwht_signs_baseline_byte_equal()
    test_full_rotation_theta_moves_after_training_step()
    test_full_rotation_theta_gets_grad_under_nested_non_reentrant_checkpoint()
    test_learnable_signs_get_grads_under_non_reentrant_checkpoint()
    test_enable_student_gradient_checkpointing_disables_cache_before_enable()
    test_grad_checkpoint_help_mentions_large_model_memory_reduction()
    test_train_kld_loss_uses_train_mode_only_when_grad_checkpointing()
    test_cache_oracle_logits_stores_bf16_cpu_and_allows_release()
    test_compute_logit_kld_cache_matches_live_with_bf16_tolerance()
    test_train_kld_loss_uses_cached_oracle_logits_and_default_live_path()
    test_rotation_mode_fwht_signs_preserves_legacy_default_byte_equal()
    test_rotation_mode_fwht_signs_preserves_learnable_signs_byte_equal()
    print("learnable FWHT tests passed")
