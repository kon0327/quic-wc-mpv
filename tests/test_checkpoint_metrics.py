import json

import pytest
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from quic_wc import QuICWCConfig, QuICWCLinear, SideConfig
from quic_wc.checkpoint import artifact_bytes, load_compact, save_compact, storage_report
from quic_wc.decomposition import activation_aware_truncation, low_rank_parameter_count, select_rank, thin_svd
from quic_wc.metrics import SecondMoment, quadratic_error, relative_output_error, token_metrics


def trained_like_module():
    cfg = QuICWCConfig(in_features=40, out_features=72, rank=12,
                       left=SideConfig(num_stages=2, permutation="stride"),
                       right=SideConfig(base_size=4, num_stages=3, permutation="random", seed=4),
                       sigma_mode="learnable")
    module = QuICWCLinear(cfg, torch.rand(12) + 0.5, torch.randn(72).to(torch.bfloat16), inference_dtype=torch.bfloat16)
    with torch.no_grad():
        module.left.theta.normal_(0, 0.3)
        module.right.theta.normal_(0, 0.3)
        module.sigma_gain.uniform_(0.5, 1.5)
    return module


def test_compact_roundtrip_and_integrity(tmp_path):
    module = trained_like_module()
    payload = save_compact(module, tmp_path / "a", metadata={"layer": "test"})
    loaded = load_compact(tmp_path / "a")
    assert torch.allclose(loaded.reconstruct_weight(), module.reconstruct_weight(), atol=1e-6)
    x = torch.randn(6, 40, dtype=torch.bfloat16)
    assert torch.equal(loaded.eval()(x), module.eval()(x))
    assert payload["config"] == module.config.to_dict()
    with pytest.raises(FileExistsError):
        save_compact(module, tmp_path / "a")
    blob = (tmp_path / "a" / "layer.safetensors").read_bytes()
    (tmp_path / "a" / "layer.safetensors").write_bytes(blob[:-1] + bytes([blob[-1] ^ 1]))
    with pytest.raises(ValueError):
        load_compact(tmp_path / "a")


def test_storage_report_matches_exported_tensors(tmp_path):
    module = trained_like_module()
    report = storage_report(module.config, bias=True)
    tensors = module.export_tensors()
    assert report["parameters"]["total"] == sum(t.numel() for t in tensors.values())
    tensor_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    assert report["tensor_bytes"] == tensor_bytes
    save_compact(module, tmp_path / "b")
    sizes = artifact_bytes(tmp_path / "b")
    assert sizes["layer.safetensors"] >= tensor_bytes
    assert json.loads((tmp_path / "b" / "quic_wc.json").read_text())["inference_dtype"] == "bfloat16"


def test_svd_convention_and_rank_policies():
    w = torch.randn(30, 20, dtype=torch.float64)
    svd = thin_svd(w, dtype=torch.float64)
    assert svd.V.shape == (20, 20)
    assert torch.allclose((svd.U * svd.S) @ svd.V.T, w, atol=1e-12)
    assert select_rank(svd.S, rank=5) == 5
    energy = torch.cumsum(svd.S**2, 0) / (svd.S**2).sum()
    r = select_rank(svd.S, energy_ratio=0.9)
    assert energy[r - 1] >= 0.9 - 1e-12 and (r == 1 or energy[r - 2] < 0.9)
    assert select_rank(svd.S, relative_threshold=1.0) == 1
    with pytest.raises(ValueError):
        select_rank(svd.S, rank=3, energy_ratio=0.5)
    assert low_rank_parameter_count(30, 20, 4) == 50 * 4 + 30
    with pytest.raises(TypeError):
        thin_svd(w.to(torch.bfloat16), dtype=torch.bfloat16)


def test_activation_aware_truncation_is_optimal_in_weighted_metric():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(30, 20, dtype=torch.float64, generator=g)
    mix = torch.randn(20, 20, dtype=torch.float64, generator=g) * torch.logspace(0, -2, 20, dtype=torch.float64)
    c = mix @ mix.T
    plain = thin_svd(w, torch.float64).truncated(6)
    aware = activation_aware_truncation(w, c, 6)
    assert quadratic_error(aware - w, c) < quadratic_error(plain - w, c)
    assert torch.allclose(activation_aware_truncation(w, torch.eye(20, dtype=torch.float64), 6), plain, atol=1e-10)


def test_second_moment_quadratic_form_equals_token_mse():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(500, 20, dtype=torch.float64, generator=g) + 0.3
    w = torch.randn(30, 20, dtype=torch.float64, generator=g)
    w_hat = w + 0.1 * torch.randn(30, 20, dtype=torch.float64, generator=g)
    b = torch.randn(30, dtype=torch.float64, generator=g)
    stats = SecondMoment(20)
    for chunk in x.split(77):
        stats.update(chunk)
    assert stats.count == 500
    assert torch.allclose(stats.second_moment, x.T @ x / 500)
    token_mse = ((x @ (w_hat - w).T) ** 2).sum(1).mean()
    assert torch.allclose(quadratic_error(w_hat - w, stats.second_moment), token_mse)
    exact = relative_output_error(w_hat, w, stats.second_moment, b, stats.mean)
    sampled = token_metrics(x, w_hat, w, b, compute_dtype=torch.float64)
    assert abs(exact["rel_l2_with_bias"] - sampled["rel_l2_with_bias"]) < 1e-10
    assert abs(exact["rel_l2_no_bias"] - sampled["rel_l2_no_bias"]) < 1e-10
    same = token_metrics(x, w, w, b)
    assert same["cos_with_bias_mean"] > 1 - 1e-6 and same["rel_l2_no_bias"] < 1e-6
    ref = F.linear(x, w, b)
    est = F.linear(x, w_hat, b)
    cos = F.cosine_similarity(ref, est, dim=1).mean()
    assert abs(float(cos) - sampled["cos_with_bias_mean"]) < 1e-10


def test_bias_correction_metrics_agree_between_exact_and_token_paths():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(400, 12, dtype=torch.float64, generator=g) + torch.linspace(-1, 2, 12, dtype=torch.float64)
    w = torch.randn(16, 12, dtype=torch.float64, generator=g)
    w_hat = w + 0.2 * torch.randn(16, 12, dtype=torch.float64, generator=g)
    b = torch.randn(16, dtype=torch.float64, generator=g)
    stats = SecondMoment(12)
    stats.update(x)
    b_hat = b + (w - w_hat) @ stats.mean  # mean-exact bias correction
    exact = relative_output_error(w_hat, w, stats.second_moment, b, stats.mean, bias_hat=b_hat)
    sampled = token_metrics(x, w_hat, w, b, b_hat, compute_dtype=torch.float64)
    est, ref = F.linear(x, w_hat, b_hat), F.linear(x, w, b)
    assert abs(exact["rel_l2_with_bias"] - float((est - ref).norm() / ref.norm())) < 1e-10
    assert abs(exact["rel_l2_no_bias"] - sampled["rel_l2_no_bias"]) < 1e-10
    assert abs(float(F.cosine_similarity(est - b, ref - b, dim=1).mean()) - sampled["cos_no_bias_mean"]) < 1e-10
    fixed = relative_output_error(w_hat, w, stats.second_moment, b, stats.mean)
    assert exact["rel_l2_no_bias"] < fixed["rel_l2_no_bias"]  # correction removes the mean error


def test_bf16_storage_matches_rounded_parameters_and_halves_bytes(tmp_path):
    module = trained_like_module()
    save_compact(module, tmp_path / "fp32")
    payload = save_compact(module, tmp_path / "bf16", storage_dtype=torch.bfloat16)
    assert payload["format_version"] == 2 and payload["storage_dtype"] == "bfloat16"
    loaded = load_compact(tmp_path / "bf16")
    assert loaded.left.theta.dtype == torch.float32  # construction stays FP32
    reference = trained_like_module()
    reference.load_state_dict(module.state_dict())  # same state, then round theta/sigma by hand
    with torch.no_grad():
        reference.left.theta.copy_(module.left.theta.to(torch.bfloat16).float())
        reference.right.theta.copy_(module.right.theta.to(torch.bfloat16).float())
        reference.sigma0.copy_(module.sigma().to(torch.bfloat16).float())
        reference.sigma_gain.fill_(1.0)
    assert torch.equal(loaded.reconstruct_weight(), reference.reconstruct_weight())
    u_hat, v_hat = loaded.factors()
    eye = torch.eye(loaded.rank)
    assert torch.allclose(u_hat.T @ u_hat, eye, atol=1e-5) and torch.allclose(v_hat.T @ v_hat, eye, atol=1e-5)
    report = storage_report(module.config, bias=True, storage_dtype="bfloat16")
    stored = load_file(str(tmp_path / "bf16" / "layer.safetensors"))
    assert report["tensor_bytes"] == sum(t.numel() * t.element_size() for t in stored.values())
    counts = report["parameters"]
    assert report["tensor_bytes"] == 2 * counts["total"]  # theta, sigma and bias all BF16
    assert artifact_bytes(tmp_path / "bf16")["layer.safetensors"] < artifact_bytes(tmp_path / "fp32")["layer.safetensors"]


def test_version1_artifact_still_loads_and_dtype_mismatch_is_rejected(tmp_path):
    module = trained_like_module()
    save_compact(module, tmp_path / "v1")
    manifest_path = tmp_path / "v1" / "quic_wc.json"
    payload = json.loads(manifest_path.read_text())
    payload["format_version"] = 1
    payload.pop("storage_dtype")
    manifest_path.write_text(json.dumps(payload))
    loaded = load_compact(tmp_path / "v1")
    assert torch.allclose(loaded.reconstruct_weight(), module.reconstruct_weight(), atol=1e-6)
    payload["storage_dtype"] = "bfloat16"  # lie about the stored dtype
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_compact(tmp_path / "v1")
    with pytest.raises(ValueError):
        storage_report(module.config, storage_dtype="float64")  # wider than FP32 construction
