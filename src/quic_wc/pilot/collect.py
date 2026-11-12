"""Calibration statistics for one Linear layer of the OpenVLA DINOv2 featurizer.

Reads only the featurizer tensors from the HF safetensors shards (no 7B model load), runs the
featurizer in BF16 exactly like OpenVLA (``get_intermediate_layers(n={len(blocks)-2})``) on
LIBERO RLDS frames with the LIBERO evaluation preprocessing, and stores FP64 second moments of
the layer input split by episodes, token samples and a manifest.

Preprocessing (experiments/robot/libero + openvla_utils, center_crop=True): RLDS JPEG 256x256
(already in training orientation, not rotated again) -> TF lanczos3 resize to 224 with
antialias, round+clip to uint8 -> TF center crop of area 0.9 and bilinear resize back to 224
-> PIL -> processor branch 0 (TVF resize/center-crop to 224, to_tensor, normalize with the
checkpoint's tvf_normalize_params[0]) -> BF16.

Usage: python -m quic_wc.pilot.collect --checkpoint <hf snapshot> --dataset <tfds dir> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ..metrics import SecondMoment

DEFAULT_LAYER = "vision_backbone.featurizer.blocks.0.mlp.fc1"
FEATURIZER_PREFIX = "vision_backbone.featurizer."


def sha256_tensor(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous()
    if data.dtype == torch.bfloat16:
        data = data.view(torch.int16)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def checkpoint_revision(snapshot: Path) -> str:
    return snapshot.resolve().name


def load_featurizer(snapshot: Path, device: str):
    """timm DINOv2 featurizer with OpenVLA weights (LayerScale scale_factor -> gamma)."""
    import timm
    from safetensors import safe_open

    config = json.loads((snapshot / "config.json").read_text())
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [k for k in index if k.startswith(FEATURIZER_PREFIX)]
    state, shards = {}, sorted({index[k] for k in keys})
    for shard in shards:
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                if index[key] == shard:
                    name = key[len(FEATURIZER_PREFIX):].replace(".scale_factor", ".gamma")
                    state[name] = handle.get_tensor(key)
    model = timm.create_model(config["timm_model_ids"][0], pretrained=False, num_classes=0,
                              img_size=config["image_sizes"][0], act_layer=config["timm_override_act_layers"][0])
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    return model, {"timm_model_id": config["timm_model_ids"][0], "image_size": config["image_sizes"][0],
                   "shards": shards, "tensors": len(keys)}


def read_layer(snapshot: Path, layer: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for suffix in ("weight", "bias"):
        key = f"{layer}.{suffix}"
        if key in index:
            with safe_open(str(snapshot / index[key]), framework="pt", device="cpu") as handle:
                out[suffix] = handle.get_tensor(key)
    if "weight" not in out:
        raise KeyError(f"{layer}.weight not found in checkpoint index")
    return out


class Preprocessor:
    def __init__(self, snapshot: Path) -> None:
        import tensorflow as tf
        import torchvision.transforms.functional as TVF

        self.tf, self.TVF = tf, TVF
        cfg = json.loads((snapshot / "preprocessor_config.json").read_text())
        if cfg.get("image_resize_strategy") != "resize-naive" or cfg.get("tvf_do_letterbox"):
            raise ValueError("unexpected image processor configuration")
        self.resize = dict(cfg["tvf_resize_params"][0])
        self.resize["size"] = list(self.resize["size"])
        self.resize["interpolation"] = TVF.InterpolationMode.BICUBIC if self.resize["interpolation"] == 3 else None
        if self.resize["interpolation"] is None:
            raise ValueError("unexpected interpolation mode")
        self.crop = cfg["tvf_crop_params"][0]
        self.normalize = cfg["tvf_normalize_params"][0]
        self.description = {"rlds_image": "jpeg 256x256 (training orientation)",
                            "resize": "tf lanczos3 224 antialias, round, clip uint8",
                            "center_crop": "tf crop_and_resize area 0.9 -> 224 (bilinear)",
                            "processor_branch0": {"resize": cfg["tvf_resize_params"][0], "crop": self.crop,
                                                  "normalize": self.normalize},
                            "model_dtype": "bfloat16"}

    def __call__(self, jpeg_bytes: list[bytes]) -> torch.Tensor:
        from PIL import Image

        tf, TVF = self.tf, self.TVF
        out = []
        for data in jpeg_bytes:
            img = tf.io.decode_jpeg(data, channels=3)
            img = tf.image.resize(img, (224, 224), method="lanczos3", antialias=True)
            img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
            img = tf.image.convert_image_dtype(img, tf.float32)
            side = float(0.9 ** 0.5)
            off = (1.0 - side) / 2
            img = tf.image.crop_and_resize(img[None], [[off, off, off + side, off + side]], [0], (224, 224))[0]
            img = tf.image.convert_image_dtype(tf.clip_by_value(img, 0, 1), tf.uint8, saturate=True)
            pil = Image.fromarray(img.numpy()).convert("RGB")
            pil = TVF.resize(pil, **self.resize)
            pil = TVF.center_crop(pil, **self.crop)
            out.append(TVF.normalize(TVF.to_tensor(pil), **self.normalize))
        return torch.stack(out)


def list_episodes(dataset_dir: Path) -> tuple[Any, list[dict[str, Any]]]:
    import tensorflow_datasets as tfds

    builder = tfds.builder_from_directory(str(dataset_dir))
    decoders = {"steps": {"observation": {"image": tfds.decode.SkipDecoding(),
                                          "wrist_image": tfds.decode.SkipDecoding()}}}
    ds = builder.as_dataset(split="train", shuffle_files=False, decoders=decoders)
    episodes = []
    for index, episode in enumerate(ds):
        steps = next(iter(episode["steps"].batch(100_000)))
        instructions = {s.decode() for s in steps["language_instruction"].numpy().tolist()}
        if len(instructions) != 1:
            raise ValueError(f"episode {index} has {len(instructions)} instructions")
        episodes.append({"index": index, "instruction": instructions.pop(),
                         "length": int(steps["is_first"].shape[0]),
                         "file_path": episode["episode_metadata"]["file_path"].numpy().decode()})
    return (builder, ds), episodes


def select_split(episodes: list[dict[str, Any]], per_task: int, val_per_task: int, seed: int) -> dict[int, str]:
    by_task = defaultdict(list)
    for ep in episodes:
        by_task[ep["instruction"]].append(ep["index"])
    rng = random.Random(seed)
    split = {}
    for task in sorted(by_task):
        chosen = rng.sample(sorted(by_task[task]), per_task)
        for i, index in enumerate(chosen):
            split[index] = "val" if i < val_per_task else "train"
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--val-per-task", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--train-token-sample", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.layer.startswith(FEATURIZER_PREFIX):
        raise ValueError("this collector supports layers of vision_backbone.featurizer only")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # TF must not reserve PyTorch GPU memory
    import tensorflow_datasets as tfds
    import timm

    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    device = "cuda"
    snapshot = args.checkpoint.resolve()
    featurizer, featurizer_info = load_featurizer(snapshot, device)
    layer_name = args.layer[len(FEATURIZER_PREFIX):]
    target = featurizer.get_submodule(layer_name)
    reference = read_layer(snapshot, args.layer)
    if not torch.equal(target.weight.detach().cpu(), reference["weight"]):
        raise RuntimeError("loaded featurizer weight differs from checkpoint tensor")
    weight_bf16, bias_bf16 = target.weight.detach(), target.bias.detach()
    w32, b32 = weight_bf16.float(), bias_bf16.float()
    in_features = weight_bf16.shape[1]

    (builder, _), episodes = list_episodes(args.dataset)
    split = select_split(episodes, args.episodes_per_task, args.val_per_task, args.seed)
    t_list = time.perf_counter() - t0
    print(f"listed {len(episodes)} episodes in {t_list:.1f}s; selected {len(split)}", flush=True)

    preprocess = Preprocessor(snapshot)
    stats = {"train": SecondMoment(in_features, device), "val": SecondMoment(in_features, device)}
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, output):
        captured["x"] = inputs[0].detach()
        captured["y"] = output.detach()

    handle = target.register_forward_hook(hook)
    val_x, train_x, hook_check = [], [], {"max_abs_bf16_vs_module": 0.0}
    floor = {"err": 0.0, "ref": 0.0, "cos_sum": 0.0, "cos_min": 1.0, "tokens": 0}
    decoders = {"steps": {"observation": {"image": tfds.decode.SkipDecoding(),
                                          "wrist_image": tfds.decode.SkipDecoding()}}}
    ds = builder.as_dataset(split="train", shuffle_files=False, decoders=decoders)
    frames_used: dict[int, list[int]] = {}
    tokens_per_image = None
    n_blocks = len(featurizer.blocks)
    try:
        for index, episode in enumerate(ds):
            if index not in split:
                continue
            steps = next(iter(episode["steps"].batch(100_000)))
            images = steps["observation"]["image"].numpy().tolist()
            chosen = list(range(0, len(images), args.frame_stride))
            frames_used[index] = chosen
            part = split[index]
            for start in range(0, len(chosen), args.batch_size):
                batch_idx = chosen[start:start + args.batch_size]
                pixels = preprocess([images[i] for i in batch_idx]).to(device=device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    featurizer.get_intermediate_layers(pixels, n={n_blocks - 2})
                x = captured["x"].reshape(-1, in_features)
                y_module = captured["y"].reshape(x.shape[0], -1)
                tokens_per_image = captured["x"].shape[1]
                with torch.inference_mode():
                    y_bf16 = F.linear(x, weight_bf16, bias_bf16)
                    hook_check["max_abs_bf16_vs_module"] = max(hook_check["max_abs_bf16_vs_module"],
                                                              float((y_bf16.float() - y_module.float()).abs().max()))
                stats[part].update(x)
                if part == "val":
                    val_x.append(x.cpu())
                    with torch.inference_mode():
                        y32 = F.linear(x.float(), w32, b32)
                        cos = F.cosine_similarity(y_module.float(), y32, dim=1)
                    floor["err"] += float(((y_module.float() - y32).double() ** 2).sum())
                    floor["ref"] += float((y32.double() ** 2).sum())
                    floor["cos_sum"] += float(cos.double().sum())
                    floor["cos_min"] = min(floor["cos_min"], float(cos.min()))
                    floor["tokens"] += x.shape[0]
                else:
                    train_x.append(x[:: max(1, args.frame_stride)].cpu())
            print(f"episode {index} ({part}): {len(chosen)} frames", flush=True)
    finally:
        handle.remove()

    x_val = torch.cat(val_x)
    x_train = torch.cat(train_x)
    if x_train.shape[0] > args.train_token_sample:
        keep = torch.randperm(x_train.shape[0], generator=torch.Generator().manual_seed(args.seed))[: args.train_token_sample]
        x_train = x_train[keep.sort().values]

    from safetensors.torch import save_file

    tensors = {
        "c_train": stats["train"].second_moment.cpu(), "mean_train": stats["train"].mean.cpu(),
        "c_val": stats["val"].second_moment.cpu(), "mean_val": stats["val"].mean.cpu(),
        "x_val": x_val.contiguous(), "x_train_sample": x_train.contiguous(),
    }
    save_file(tensors, str(args.out / "calibration.safetensors"))
    save_file({"weight": weight_bf16.cpu().contiguous(), "bias": bias_bf16.cpu().contiguous()},
              str(args.out / "teacher_layer.safetensors"))
    dataset_manifest = json.loads((args.dataset.parent / "manifest.json").read_text())
    selected = [dict(ep, split=split[ep["index"]], frames=len(frames_used.get(ep["index"], [])))
                for ep in episodes if ep["index"] in split]
    manifest = {
        "schema": "quic_wc.calibration/v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint": {"path": str(snapshot), "revision": checkpoint_revision(snapshot),
                       "repo": snapshot.parent.parent.name.replace("models--", "").replace("--", "/")},
        "layer": {"name": args.layer, "out_features": weight_bf16.shape[0], "in_features": in_features,
                  "dtype": str(weight_bf16.dtype).replace("torch.", ""), "bias": True,
                  "weight_sha256": sha256_tensor(weight_bf16), "bias_sha256": sha256_tensor(bias_bf16)},
        "featurizer": featurizer_info,
        "dataset": {"path": str(args.dataset), "logical_name": dataset_manifest.get("logical_name"),
                    "source": dataset_manifest.get("canonical_locator"),
                    "prepared_content_digest": dataset_manifest.get("prepared_content_digest"),
                    "episodes_total": len(episodes)},
        "selection": {"seed": args.seed, "episodes_per_task": args.episodes_per_task,
                      "val_per_task": args.val_per_task, "frame_stride": args.frame_stride,
                      "split_rule": "per task: seeded sample, first val_per_task -> val", "episodes": selected},
        "tokens": {"per_image": tokens_per_image, "train": stats["train"].count, "val": stats["val"].count,
                   "x_val_stored": x_val.shape[0], "x_train_sample_stored": x_train.shape[0],
                   "x_train_sample_rule": f"every {args.frame_stride}th token of train batches, then seeded subsample"},
        "preprocessing": preprocess.description,
        "checks": {**hook_check,
                   "bf16_module_vs_fp32_reference_val": {
                       "rel_l2_with_bias": (floor["err"] / floor["ref"]) ** 0.5,
                       "cos_with_bias_mean": floor["cos_sum"] / floor["tokens"], "cos_with_bias_min": floor["cos_min"]}},
        "versions": {"torch": torch.__version__, "timm": timm.__version__, "tensorflow": tf.__version__,
                     "tensorflow_datasets": tfds.__version__, "python": platform.python_version(),
                     "cuda_device": torch.cuda.get_device_name(0)},
        "seconds": time.perf_counter() - t0,
        "files": {name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
                  for name in ("calibration.safetensors", "teacher_layer.safetensors")},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("layer", "tokens", "checks", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
