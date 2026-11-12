import pytest
import torch
from torch import nn
from torch.nn import functional as F

from quic_wc import QuICWCConfig, QuICWCLinear, SideConfig, prepare_for_inference
from quic_wc.layers.linear import StaleCacheError


def make(sigma_mode="learnable", bias=True, rank=12, stages=2, seed=0):
    cfg = QuICWCConfig(in_features=40, out_features=72, rank=rank,
                       left=SideConfig(base_size=8, num_stages=stages, permutation="stride"),
                       right=SideConfig(base_size=4, num_stages=stages, permutation="random", seed=1),
                       sigma_mode=sigma_mode)
    g = torch.Generator().manual_seed(seed)
    module = QuICWCLinear(cfg, torch.rand(rank, generator=g) + 0.5, torch.randn(72, generator=g) if bias else None)
    with torch.no_grad():
        module.left.theta.normal_(0, 0.3, generator=g)
        module.right.theta.normal_(0, 0.3, generator=g)
    return module


def test_identity_init_reconstructs_selected_coordinates():
    cfg = QuICWCConfig(in_features=40, out_features=72, rank=5)
    sigma = torch.arange(5.0, 0, -1)
    module = QuICWCLinear(cfg, sigma)
    expected = torch.zeros(72, 40)
    expected[module.selection_out, module.selection_in] = sigma
    assert torch.equal(module.reconstruct_weight(), expected)


def test_from_linear_uses_top_singular_values_and_bias():
    linear = nn.Linear(40, 72)
    cfg = QuICWCConfig(in_features=40, out_features=72, rank=10)
    module = QuICWCLinear.from_linear(linear, cfg)
    assert torch.allclose(module.sigma0, torch.linalg.svdvals(linear.weight.detach())[:10])
    assert torch.equal(module.bias, linear.bias.detach())
    with pytest.raises(ValueError):
        QuICWCLinear.from_linear(nn.Linear(41, 72), cfg)


def test_factors_are_orthonormal_and_training_forward_is_differentiable():
    module = make()
    u, v = module.factors()
    assert (u.T @ u - torch.eye(12)).abs().max() < 1e-5 and (v.T @ v - torch.eye(12)).abs().max() < 1e-5
    x = torch.randn(3, 5, 40)
    module.train()
    y = module(x)
    assert torch.allclose(y, F.linear(x, module.reconstruct_weight(), module.bias), atol=1e-6)
    y.square().mean().backward()
    for p in (module.left.theta, module.right.theta, module.sigma_gain):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0


def test_eval_decompresses_once_and_then_only_uses_cache(monkeypatch):
    module = make().eval()
    x = torch.randn(4, 40)
    first = module(x)
    calls = []
    original = module.reconstruct_weight
    monkeypatch.setattr(module, "reconstruct_weight", lambda: calls.append(1) or original())
    for _ in range(3):
        assert torch.equal(module(x), first)
    assert module.decompress_count == 1 and not calls
    assert module.weight_cache.is_contiguous()


def test_cache_invalidation_paths():
    module = make().eval()
    x = torch.randn(4, 40)
    module(x)
    with torch.no_grad():
        module.left.theta.add_(0.1)  # in-place, as an optimizer step
    assert not module.cache_valid
    y = module(x)
    assert module.decompress_count == 2
    assert torch.allclose(y, F.linear(x, module.reconstruct_weight(), module.bias), atol=1e-6)

    other = make(seed=5).state_dict()
    module.load_state_dict(other)
    assert module.weight_cache is None
    module(x)
    module.train()
    assert module.weight_cache is None
    module.eval()
    module.to(torch.float64)
    assert module.weight_cache is None and module.left.theta.dtype == torch.float32
    assert module.sigma0.dtype == torch.float32 and module.bias.dtype == torch.float64
    assert module(x.double()).dtype == torch.float64


def test_strict_cache_refuses_stale_weights():
    module = make()
    module.strict_cache = True
    module.eval()
    with pytest.raises(StaleCacheError):
        module(torch.randn(2, 40))
    prepare_for_inference(nn.Sequential(module))
    module(torch.randn(2, 40))
    with torch.no_grad():
        module.sigma0.mul_(2)
    with pytest.raises(StaleCacheError):
        module(torch.randn(2, 40))


def test_state_dict_is_compact_and_excludes_cache():
    module = make().eval()
    module(torch.randn(2, 40))
    assert set(module.state_dict()) == {"left.theta", "right.theta", "sigma0", "sigma_gain", "bias"}
    assert set(make(sigma_mode="fixed", bias=False).state_dict()) == {"left.theta", "right.theta", "sigma0"}


def test_bfloat16_host_cast_keeps_construction_precision():
    module = make().to(torch.bfloat16).eval()
    assert module.left.theta.dtype == torch.float32 and module.sigma0.dtype == torch.float32
    assert module.sigma_gain.dtype == torch.float32 and module.bias.dtype == torch.bfloat16
    out = module(torch.randn(3, 40, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16 and module.weight_cache.dtype == torch.bfloat16


def test_config_validation_and_parameter_counts():
    with pytest.raises(ValueError):
        QuICWCConfig(in_features=40, out_features=72, rank=41)
    with pytest.raises(ValueError):
        QuICWCConfig(in_features=20, out_features=72, rank=4)  # 36-wide block does not fit 20
    cfg = QuICWCConfig(in_features=1024, out_features=4096, rank=1024,
                       left=SideConfig(num_stages=4, permutation="stride"),
                       right=SideConfig(num_stages=4, permutation="stride"))
    counts = cfg.parameter_counts()
    assert counts["theta_out"] == 4 * 113 * 28 and counts["theta_in"] == 4 * 28 * 28
    assert counts["total"] == counts["theta_out"] + counts["theta_in"] + 1024 + 4096
    assert QuICWCConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_device_move_invalidates_and_matches_cpu():
    module = make().eval()
    x = torch.randn(5, 40)
    cpu_out = module(x)
    module.cuda()
    assert module.weight_cache is None
    gpu_out = module(x.cuda())
    assert gpu_out.device.type == "cuda" and torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-5)
