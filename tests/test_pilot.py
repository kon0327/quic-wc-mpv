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
    entries = [run_quic(ctx, spec, defaults, tmp_path / "out") for spec in specs]
    for entry in entries:
        assert entry["fit"]["status"] == "completed"
        assert entry["roundtrip_max_abs"] < 1e-5
        assert entry["equal_budget"]["r_bytes"] >= entry["equal_budget"]["r_params"] >= 0
        assert 0 <= entry["metrics"]["exact_val_rel_l2_no_bias"] < 10
    corrected = entries[1]
    assert corrected["metrics"]["exact_val_rel_l2_no_bias"] <= entries[0]["metrics"]["exact_val_rel_l2_no_bias"] + 0.05
    results = {"layer": ctx.manifest["layer"], "checkpoint": ctx.manifest["checkpoint"],
               "tokens": ctx.manifest["tokens"], "baselines": [], "quic_wc": entries}
    assert "| real |" in markdown(results)


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
