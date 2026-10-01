"""Offline checks of CUDA math, local OpenVLA inference and LIBERO rendering.

Uses random miniature OpenVLA weights, not a pretrained policy. No downloads.
"""

import argparse
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
REPORT = {"checks": {}, "versions": {}, "scope": "offline synthetic environment checks"}


def check(name, fn):
    started = time.perf_counter()
    try:
        detail = fn()
        REPORT["checks"][name] = {
            "status": "passed", "seconds": time.perf_counter() - started, "detail": detail,
        }
        print(f"PASS {name}: {detail}", flush=True)
    except Exception:
        REPORT["checks"][name] = {
            "status": "failed", "seconds": time.perf_counter() - started,
            "error": traceback.format_exc(),
        }
        print(f"FAIL {name}:\n{REPORT['checks'][name]['error']}", flush=True)


def versions():
    expected = {
        "torch": "2.2.0", "torchvision": "0.17.0", "torchaudio": "2.2.0",
        "transformers": "4.40.1", "tokenizers": "0.19.1", "timm": "0.9.10",
        "flash-attn": "2.5.5", "peft": "0.11.1", "numpy": "1.26.4",
        "tensorflow": "2.15.0", "robosuite": "1.4.1", "mujoco": "2.3.7",
        "tensorflow-metadata": "1.17.2", "protobuf": "4.21.12",
    }
    assert sys.version_info[:2] == (3, 10), sys.version
    for name, wanted in expected.items():
        actual = importlib.metadata.version(name)
        REPORT["versions"][name] = actual
        assert actual.split("+")[0] == wanted, (name, actual, wanted)
    return REPORT["versions"]


def local_sources():
    import prismatic
    import libero
    for module, folder in [(prismatic, "openvla"), (libero, "libero")]:
        origins = [Path(p).resolve() for p in module.__path__]
        assert origins and all(p.is_relative_to(ROOT / "external" / folder) for p in origins), origins
    return {"prismatic": prismatic.__file__, "libero_namespace_paths": list(libero.__path__)}


def gpu_math():
    import torch
    assert torch.cuda.is_available(), "CUDA unavailable; run outside the restricted sandbox"
    assert torch.cuda.is_bf16_supported(), "BF16 unsupported"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    REPORT["gpu"] = {
        "name": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    x = torch.randn(128, 128, device="cuda")
    assert torch.isfinite(x @ x.T).all()
    bf = x.to(torch.bfloat16)
    assert torch.isfinite(bf @ bf.T).all()
    weight = torch.randn(96, 64, device="cuda")
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    relative_error = ((u * s) @ vh - weight).norm() / weight.norm()
    assert relative_error < 1e-5, relative_error
    p = (torch.randn(8, 8, device="cuda") * 0.01).requires_grad_()
    skew = p - p.T
    eye = torch.eye(8, device="cuda", dtype=p.dtype)
    a = torch.linalg.solve((eye + skew).T, (eye - skew).T).T
    indices = torch.tensor(list(itertools.combinations(range(8), 2)), device="cuda")
    minors = a[indices][:, :, indices].permute(0, 2, 1, 3)
    compound = torch.linalg.det(minors)
    direct = torch.block_diag(a, compound)
    orth_error = (direct.T @ direct - torch.eye(36, device="cuda")).abs().max()
    assert orth_error < 1e-5, orth_error
    (direct * torch.randn_like(direct)).sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
    identity_p = torch.zeros(8, 8, device="cuda", requires_grad=True)
    identity_skew = identity_p - identity_p.T
    identity_a = torch.linalg.solve((eye + identity_skew).T, (eye - identity_skew).T).T
    identity_minors = identity_a[indices][:, :, indices].permute(0, 2, 1, 3)
    identity_c = torch.linalg.det(identity_minors)
    identity_probe = torch.randn_like(identity_c)
    compound_grad = torch.autograd.grad((identity_c * identity_probe).sum(), identity_p,
                                      retain_graph=True)[0]
    reference_c = (identity_minors[..., 0, 0] * identity_minors[..., 1, 1]
                   - identity_minors[..., 0, 1] * identity_minors[..., 1, 0])
    reference_grad = torch.autograd.grad((reference_c * identity_probe).sum(), identity_p,
                                       retain_graph=True)[0]
    assert torch.isfinite(reference_grad).all() and reference_grad.norm() > 0
    singular_gradient_error = (compound_grad - reference_grad).abs().max().item()
    if singular_gradient_error > 1e-5:
        REPORT.setdefault("limitations", []).append(
            "torch.linalg.det C2 backward at exact identity differs from the explicit 2x2 formula; "
            "use the explicit formula for order 2 or avoid identity initialization of C2."
        )
    # Direct sum includes C1, which provides a gradient even when det's singular
    # backward returns zero for C2 at exact identity on PyTorch 2.2/CUDA.
    ((identity_c * identity_probe).sum() + (identity_a * torch.randn_like(identity_a)).sum()).backward()
    assert identity_p.grad is not None and torch.isfinite(identity_p.grad).all()
    assert identity_p.grad.norm() > 0
    # Record the CUDA determinant limitation; the random-init C2 gradient above is tested.
    assert torch.isfinite(compound_grad).all()
    torch.cuda.synchronize()
    return {**REPORT["gpu"], "svd_relative_error": relative_error.item(),
            "orthogonality_max_error": orth_error.item(), "gradient_norm": p.grad.norm().item(),
            "identity_init_direct_sum_gradient_norm": identity_p.grad.norm().item(),
            "identity_init_compound_only_gradient_norm": compound_grad.norm().item(),
            "identity_init_explicit_c2_gradient_norm": reference_grad.norm().item(),
            "identity_init_det_gradient_max_error": singular_gradient_error}


def flash_attention():
    import torch
    from flash_attn import flash_attn_func
    q, k, v = [torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.bfloat16,
                         requires_grad=True) for _ in range(3)]
    output = flash_attn_func(q, k, v, causal=True)
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
        is_causal=True,
    ).transpose(1, 2)
    max_error = (output.float() - reference).abs().max().item()
    assert max_error < 0.03, max_error
    output.float().square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    torch.cuda.synchronize()
    return {"shape": list(output.shape), "max_error_vs_fp32_sdpa": max_error,
            "backward": "finite"}


def tensorflow_cpu():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    y = tf.image.resize(tf.ones((8, 8, 3)), (16, 16), method="lanczos3", antialias=True)
    assert tuple(y.shape) == (16, 16, 3)
    assert bool(tf.reduce_all(tf.math.is_finite(y)))
    return {"resize_shape": list(y.shape), "visible_gpus": len(tf.config.get_visible_devices("GPU"))}


def openvla_synthetic():
    import numpy as np
    import torch
    from PIL import Image
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor

    cfg = OpenVLAConfig(
        vision_backbone_id="clip-vit-l", llm_backbone_id="llama2-7b-pure",
        norm_stats={"smoke": {"action": {"q01": [-1.0] * 7, "q99": [1.0] * 7}}},
        text_config={"vocab_size": 32064, "hidden_size": 128, "intermediate_size": 256,
                     "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 4, "max_position_embeddings": 512,
                     "pad_token_id": 32000, "bos_token_id": 1, "eos_token_id": 2},
    )
    cfg.timm_model_ids = ["vit_tiny_patch16_224"]
    cfg.timm_override_act_layers = [None]
    cfg._attn_implementation = "flash_attention_2"
    cfg.text_config._attn_implementation = "flash_attention_2"
    cfg.torch_dtype = torch.bfloat16
    cfg.text_config.torch_dtype = torch.bfloat16
    model = OpenVLAForActionPrediction(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    image_processor = PrismaticImageProcessor(interpolations=["bicubic"])
    image = Image.fromarray(np.full((256, 256, 3), 127, dtype=np.uint8))
    pixels = image_processor(images=image, return_tensors="pt")["pixel_values"].to(
        device="cuda", dtype=torch.bfloat16,
    )
    ids = torch.tensor([[1, 100, 29871]], device="cuda")
    captures = []
    handle = model.vision_backbone.featurizer.blocks[0].mlp.fc1.register_forward_hook(
        lambda module, inputs, output: captures.append((tuple(inputs[0].shape), tuple(output.shape)))
    )
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, pixel_values=pixels, return_dict=True)
            assert torch.isfinite(out.logits).all()
            actions = model.predict_action(input_ids=ids, pixel_values=pixels,
                                           unnorm_key="smoke", do_sample=False)
        assert actions.shape == (7,) and np.isfinite(actions).all(), actions
        assert len(captures) == 2, captures
        backend = type(model.language_model.model.layers[0].self_attn).__name__
        assert "FlashAttention" in backend, backend
        torch.cuda.synchronize()
        return {"weights": "random miniature model; no policy-quality evaluation",
                "logits_shape": list(out.logits.shape), "action_shape": list(actions.shape),
                "vision_hook_shapes": captures, "attention_class": backend}
    finally:
        handle.remove()
        del model
        torch.cuda.empty_cache()


def libero_render():
    import numpy as np
    import yaml
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = ROOT / "external/libero/libero/libero"
    paths = {
        "benchmark_root": str(benchmark_root), "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"), "assets": str(benchmark_root / "assets"),
        "datasets": str(ROOT / "artifacts/datasets/libero"),
    }
    Path(paths["datasets"]).mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(paths))
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    env = OffScreenRenderEnv(
        bddl_file_name=str(benchmark_root / "bddl_files" / task.problem_folder / task.bddl_file),
        camera_heights=128, camera_widths=128,
    )
    try:
        env.seed(7)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(3):
            obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
        frame = obs["agentview_image"]
        assert frame.shape == (128, 128, 3), frame.shape
        assert np.isfinite(frame).all() and frame.std() > 0, "Empty render"
        return {"task": task.name, "frame_shape": list(frame.shape),
                "frame_std": float(frame.std()), "render_backend": os.environ.get("MUJOCO_GL"),
                "simulation_steps": 3}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/environment/gpu-smoke.json")
    checks = [
        ("versions", versions), ("local_sources", local_sources), ("cuda_math", gpu_math),
        ("flash_attention", flash_attention), ("tensorflow_preprocessing", tensorflow_cpu),
        ("openvla_synthetic_inference", openvla_synthetic), ("libero_render", libero_render),
    ]
    parser.add_argument("--only", nargs="+", choices=[name for name, _ in checks],
                        help="Run selected checks when diagnosing a failure")
    args = parser.parse_args()
    REPORT.update({"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "project_root": str(ROOT)})
    # The inner LIBERO package prompts on first import; create its config before any imports.
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in checks:
        if args.only is None or name in args.only:
            check(name, fn)
    REPORT["passed"] = all(item["status"] == "passed" for item in REPORT["checks"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(REPORT, indent=2) + "\n")
    print(f"Report: {args.report}", flush=True)
    return 0 if REPORT["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
