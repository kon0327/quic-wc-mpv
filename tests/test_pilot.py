import json

import torch
from safetensors.torch import save_file

from quic_wc.decomposition import activation_aware_svd
from quic_wc.pilot.collect import select_split
from quic_wc.pilot.run import Context, markdown, run_quic


def synthetic_calibration(path, m=72, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(n, n, generator=g, dtype=torch.float64) * torch.logspace(0, -2, n, dtype=torch.float64)
    x_train = torch.randn(3000, n, generator=g, dtype=torch.float64) @ mix.T + 0.5
    x_val = torch.randn(800, n, generator=g, dtype=torch.float64) @ mix.T + 0.5
    path.mkdir()
    save_file({"c_train": x_train.T @ x_train / 3000, "mean_train": x_train.mean(0),
               "c_val": x_val.T @ x_val / 800, "mean_val": x_val.mean(0),
               "x_val": x_val.to(torch.bfloat16), "x_train_sample": x_train[:100].to(torch.bfloat16)},
              str(path / "calibration.safetensors"))
    save_file({"weight": torch.randn(m, n, generator=g).to(torch.bfloat16), "bias": torch.randn(m, generator=g).to(torch.bfloat16)},
              str(path / "teacher_layer.safetensors"))
    manifest = {"layer": {"name": "synthetic", "out_features": m, "in_features": n},
                "checkpoint": {"repo": "synthetic", "revision": "0" * 40},
                "tokens": {"train": 3000, "val": 800}, "checks": {}}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def test_pilot_runner_end_to_end_with_controls(tmp_path):
    ctx = Context(synthetic_calibration(tmp_path / "cal"), "cpu")
    defaults = {"fit": {"steps": 30, "lr": 0.03, "eval_every": 10}}
    side = {"base_size": 4, "orders": [1, 2], "num_stages": 2, "permutation": "stride"}
    specs = [
        {"name": "real", "rank": 20, "left": side},
        {"name": "real_bc", "rank": 20, "left": side, "bias_policy": "corrected"},
        {"name": "pos", "rank": 20, "left": side, "teacher": "positive_control", "teacher_theta_scale": 0.1},
        {"name": "neg", "rank": 20, "left": side, "teacher": "negative_control"},
    ]
    entries = [e for spec in specs for e in run_quic(ctx, spec, defaults, tmp_path / "out")]
    for entry in entries:
        assert entry["fit"]["status"] == "completed"
        assert entry["roundtrip_max_abs"] < 1e-5
        eq = entry["equal_budget"]
        assert eq["r_bytes"] >= eq["r_params"] >= 0 and eq["r_bytes_ceil"] in (eq["r_bytes"], eq["r_bytes"] + 1)
        assert eq["asvd_r_bytes_ceil"]["tensor_bytes"] >= entry["tensor_bytes"] >= eq["asvd_r_bytes"]["tensor_bytes"]
        assert 0 <= entry["metrics"]["exact_val_rel_l2_no_bias"] < 10
        assert (entry["gate_m3"] is None) == (entry["teacher"]["kind"] != "real")
    corrected = entries[1]
    assert corrected["metrics"]["exact_val_rel_l2_no_bias"] <= entries[0]["metrics"]["exact_val_rel_l2_no_bias"] + 0.05
    results = {"layer": ctx.manifest["layer"], "checkpoint": ctx.manifest["checkpoint"],
               "tokens": ctx.manifest["tokens"], "baselines": [], "quic_wc": entries}
    assert "| real |" in markdown(results)


def test_bf16_storage_growth_and_restarts_end_to_end(tmp_path):
    ctx = Context(synthetic_calibration(tmp_path / "cal"), "cpu")
    defaults = {"storage_dtype": "bfloat16", "fit": {"steps": 20, "lr": 0.03, "eval_every": 10, "select_on": "train"}}
    side = {"base_size": 4, "orders": [1, 2], "permutation": "stride", "stage_form": "conjugated"}
    spec = {"name": "grow", "rank": "full", "left": side, "growth": [1, 2, 4], "growth_steps": [20, 10, 10],
            "restarts": {"count": 3, "scale": 0.05, "screen_steps": 5, "seed": 0}}
    entries = run_quic(ctx, spec, defaults, tmp_path / "out")
    assert [e["name"] for e in entries] == ["grow_L1", "grow_L2", "grow_L4"]
    assert [e["stages"] for e in entries] == [[1, 1], [2, 2], [4, 4]]
    assert entries[0]["restarts"]["count"] == 3 and entries[1]["restarts"] is None
    for e in entries:
        assert e["storage_dtype"] == "bfloat16" and e["tensor_bytes"] == 2 * e["parameters"]
        assert e["roundtrip_max_abs"] == 0.0
        assert e["fit"]["best_step"] >= 0
    # growth never makes the train objective worse than the previous size (function-preserving start)
    fits = [e["fit"]["history"] for e in entries]
    for prev, cur in zip(fits, fits[1:]):
        assert cur[0]["train_rel_mse"] <= prev[-1]["train_rel_mse_after"] + 1e-6
    results = {"layer": ctx.manifest["layer"], "checkpoint": ctx.manifest["checkpoint"], "tokens": ctx.manifest["tokens"],
               "baselines": [], "quic_wc": entries}
    from quic_wc.pilot.run import layer_gate
    results["gate_m3"] = layer_gate(entries)
    assert "fixed" in results["gate_m3"] and "Brána M3" in markdown(results)


def test_layer_without_bias_runs_fixed_policy_only(tmp_path):
    cal = synthetic_calibration(tmp_path / "cal")
    from safetensors.torch import load_file
    layer = load_file(str(cal / "teacher_layer.safetensors"))
    save_file({"weight": layer["weight"]}, str(cal / "teacher_layer.safetensors"))
    ctx = Context(cal, "cpu")
    assert not ctx.has_bias and ctx.policies(["fixed", "corrected"]) == ["fixed"]
    spec = {"name": "nb", "rank": 20, "left": {"base_size": 4, "orders": [1, 2], "num_stages": 2, "permutation": "stride"}}
    (entry,) = run_quic(ctx, spec, {"fit": {"steps": 10, "eval_every": 5}}, tmp_path / "out")
    assert entry["parameter_breakdown"]["bias"] == 0
    assert entry["equal_budget"]["asvd_r_bytes"]["tensor_bytes"] == 2 * (72 + 40) * entry["equal_budget"]["r_bytes"]
    import pytest
    with pytest.raises(ValueError):
        run_quic(ctx, spec | {"bias_policy": "corrected"}, {}, tmp_path / "out2")


def test_asvd_cache_never_returns_a_stale_teacher():
    ctx = Context.__new__(Context)
    ctx._asvd_cache = {}
    c = torch.eye(6, dtype=torch.float64) + 0.1
    ctx.objective = {"fixed": (c, c)}
    first = torch.randn(8, 6)
    assert ctx.asvd(first, "fixed") is ctx.asvd(first, "fixed")
    second = torch.randn(8, 6)
    key = (first.data_ptr(), "fixed")
    ctx._asvd_cache[(second.data_ptr(), "fixed")] = ctx._asvd_cache.pop(key)  # simulate address reuse
    got = ctx.asvd(second, "fixed")
    assert torch.allclose(got.truncated(3), activation_aware_svd(second, c).truncated(3))


def test_episode_split_is_deterministic_and_per_task():
    episodes = [{"index": i, "instruction": f"task {i % 3}"} for i in range(30)]
    split = select_split(episodes, per_task=4, val_per_task=1, seed=0)
    assert split == select_split(episodes, per_task=4, val_per_task=1, seed=0)
    assert len(split) == 12 and sum(v == "val" for v in split.values()) == 3
    for task in range(3):
        chosen = [i for i in split if i % 3 == task]
        assert len(chosen) == 4 and sum(split[i] == "val" for i in chosen) == 1


def test_stages_sweep_expands_to_independent_specs():
    from quic_wc.pilot.run import expand_sweeps
    specs = expand_sweeps([{"name": "s", "rank": 8, "stages_sweep": [2, 4], "left": {"base_size": 4, "num_stages": 1}},
                           {"name": "plain", "rank": 8, "left": {}}])
    assert [s["name"] for s in specs] == ["s_L2", "s_L4", "plain"]
    assert specs[1]["left"]["num_stages"] == 4 and specs[1]["right"]["num_stages"] == 4 and specs[1]["sweep_of"] == "s"
    assert "stages_sweep" not in specs[0]


def test_depth_scaled_learning_rate():
    from quic_wc.distillation import FitConfig
    from quic_wc.pilot.run import depth_scaled
    cfg = FitConfig(lr=0.03, sigma_lr=0.06)
    assert depth_scaled(cfg, None, 128) is cfg and depth_scaled(cfg, {"reference_stages": 16}, 8) is cfg
    scaled = depth_scaled(cfg, {"reference_stages": 16, "power": 1.0}, 64)
    assert abs(scaled.lr - 0.0075) < 1e-12 and abs(scaled.sigma_lr - 0.015) < 1e-12
    assert abs(depth_scaled(cfg, {"reference_stages": 16, "power": 0.5}, 64).lr - 0.015) < 1e-12
