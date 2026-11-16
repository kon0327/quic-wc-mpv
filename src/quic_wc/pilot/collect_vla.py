"""Calibration statistics for several Linear layers of the full OpenVLA model (vision + LLM).

Loads ``OpenVLAForActionPrediction`` from the local snapshot with the class of the project's
``external/openvla`` checkout (BF16, flash-attention 2, as in ``experiments/robot/openvla_utils.get_vla``)
and queries it frame by frame exactly like the LIBERO evaluation: the TF image pipeline of
``collect.Preprocessor`` (lanczos3 224, center crop 0.9), the official ``PrismaticProcessor`` with
the prompt ``"In: What action should the robot take to {task}?\\nOut:"`` and ``predict_action``
(greedy, ``do_sample=False``). Forward hooks therefore see the vision tokens, the prefill tokens
and the decode steps of the action tokens. Episode split and frame selection are the same as in
``collect`` (same seed and stride give the same frames).

For every hooked layer the output directory ``<out>/<layer>`` gets the same files as ``collect``
(``calibration.safetensors``, ``teacher_layer.safetensors``, ``manifest.json``), so
``quic_wc.pilot.run`` works unchanged. ``<out>/teacher_actions.safetensors`` stores the greedy
action tokens and dataset actions per frame (reference for later end-to-end checks).

Usage: python -m quic_wc.pilot.collect_vla --checkpoint <hf snapshot> --dataset <tfds dir> --out <dir>
       --layers <name> [<name> ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ..metrics import SecondMoment
from .collect import Preprocessor, checkpoint_revision, list_episodes, read_layer, select_split, sha256_tensor

PROMPT = "In: What action should the robot take to {task}?\nOut:"


def short_name(layer: str) -> str:
    return (layer.replace("vision_backbone.fused_featurizer.", "siglip.").replace("vision_backbone.featurizer.", "dinov2.")
            .replace("language_model.model.", "llm.").replace(".", "_"))


class LayerStats:
    def __init__(self, name: str, module: torch.nn.Linear, device: str, train_stride: int) -> None:
        self.name, self.module = name, module
        self.in_features = module.in_features
        self.stats = {"train": SecondMoment(self.in_features, device), "val": SecondMoment(self.in_features, device)}
        self.val_x: list[torch.Tensor] = []
        self.train_x: list[torch.Tensor] = []
        self.train_stride = train_stride
        self._train_offset = 0  # global token counter so 1-token decode calls are strided like prefill tokens
        self.check = {"max_abs_bf16_vs_module": 0.0}
        self.floor = {"err": 0.0, "ref": 0.0, "cos_sum": 0.0, "cos_min": 1.0, "tokens": 0}
        self.calls = {"train": 0, "val": 0}
        self.part = "train"
        w = module.weight.detach()
        self.w32 = w.float()
        self.b32 = None if module.bias is None else module.bias.detach().float()

    def hook(self, _module, inputs, output) -> None:
        x = inputs[0].detach().reshape(-1, self.in_features)
        y = output.detach().reshape(x.shape[0], -1)
        part = self.part
        self.calls[part] += 1
        with torch.inference_mode():
            y_ref = F.linear(x, self.module.weight, self.module.bias)
            self.check["max_abs_bf16_vs_module"] = max(self.check["max_abs_bf16_vs_module"],
                                                       float((y_ref.float() - y.float()).abs().max()))
        self.stats[part].update(x)
        if part == "val":
            self.val_x.append(x.cpu())
            with torch.inference_mode():
                y32 = F.linear(x.float(), self.w32, self.b32)
                cos = F.cosine_similarity(y.float(), y32, dim=1)
            self.floor["err"] += float(((y.float() - y32).double() ** 2).sum())
            self.floor["ref"] += float((y32.double() ** 2).sum())
            self.floor["cos_sum"] += float(cos.double().sum())
            self.floor["cos_min"] = min(self.floor["cos_min"], float(cos.min()))
            self.floor["tokens"] += x.shape[0]
        else:
            first = (-self._train_offset) % self.train_stride
            self.train_x.append(x[first:: self.train_stride].cpu())
            self._train_offset += x.shape[0]


def sample_rows(x: torch.Tensor, limit: int, seed: int) -> torch.Tensor:
    if x.shape[0] <= limit:
        return x
    keep = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))[:limit]
    return x[keep.sort().values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--val-per-task", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--val-token-sample", type=int, default=32768)
    parser.add_argument("--train-token-sample", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn", default="flash_attention_2")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # TF must not reserve PyTorch GPU memory
    import tensorflow_datasets as tfds
    import transformers
    from transformers import AutoTokenizer
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    t0 = time.perf_counter()
    torch.manual_seed(args.seed)
    device = "cuda"
    snapshot = args.checkpoint.resolve()
    vla = OpenVLAForActionPrediction.from_pretrained(snapshot, attn_implementation=args.attn, torch_dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True).to(device).eval()
    processor = PrismaticProcessor(PrismaticImageProcessor.from_pretrained(snapshot), AutoTokenizer.from_pretrained(snapshot))
    unnorm_key = next(iter(vla.norm_stats))
    action_dim = vla.get_action_dim(unnorm_key)

    layers: dict[str, LayerStats] = {}
    handles = []
    for name in args.layers:
        module = vla.get_submodule(name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"{name} is not nn.Linear")
        reference = read_layer(snapshot, name)
        if not torch.equal(module.weight.detach().cpu(), reference["weight"]):
            raise RuntimeError(f"{name}: loaded weight differs from the checkpoint tensor")
        layers[name] = LayerStats(name, module, device, args.frame_stride)
        handles.append(module.register_forward_hook(layers[name].hook))

    generated: list[torch.Tensor] = []
    original_generate = vla.generate

    def recording_generate(*g_args, **g_kwargs):
        out = original_generate(*g_args, **g_kwargs)
        generated.append(out[0, -action_dim:].detach().cpu())
        return out

    vla.generate = recording_generate  # predict_action calls self.generate; record its token ids

    (builder, _), episodes = list_episodes(args.dataset)
    split = select_split(episodes, args.episodes_per_task, args.val_per_task, args.seed)
    print(f"listed {len(episodes)} episodes; selected {len(split)}; load+list {time.perf_counter() - t0:.0f}s", flush=True)
    preprocess = Preprocessor(snapshot)
    decoders = {"steps": {"observation": {"image": tfds.decode.SkipDecoding(),
                                          "wrist_image": tfds.decode.SkipDecoding()}}}
    ds = builder.as_dataset(split="train", shuffle_files=False, decoders=decoders)
    frames: list[dict[str, Any]] = []
    actions_gt: list[torch.Tensor] = []
    prompt_tokens: dict[str, int] = {}
    try:
        for index, episode in enumerate(ds):
            if index not in split:
                continue
            steps = next(iter(episode["steps"].batch(100_000)))
            images = steps["observation"]["image"].numpy().tolist()
            actions = torch.from_numpy(steps["action"].numpy())
            task = steps["language_instruction"].numpy()[0].decode()
            part = split[index]
            prompt = PROMPT.format(task=task.lower())
            for layer in layers.values():
                layer.part = part
            chosen = list(range(0, len(images), args.frame_stride))
            for frame in chosen:
                inputs = processor(prompt, preprocess.pil_image(images[frame])).to(device, dtype=torch.bfloat16)
                prompt_tokens[task] = int(inputs["input_ids"].shape[1])
                with torch.inference_mode():
                    vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
                frames.append({"episode": index, "frame": frame, "split": part})
                actions_gt.append(actions[frame])
            print(f"episode {index} ({part}): {len(chosen)} frames, {time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        vla.generate = original_generate

    from safetensors.torch import save_file

    if len(generated) != len(frames):
        raise RuntimeError(f"recorded {len(generated)} generations for {len(frames)} frames")
    index = torch.tensor([[f["episode"], f["frame"], int(f["split"] == "val")] for f in frames], dtype=torch.int64)
    save_file({"generated_action_tokens": torch.stack(generated), "dataset_actions": torch.stack(actions_gt).float(),
               "episode_frame_isval": index}, str(args.out / "teacher_actions.safetensors"))
    dataset_manifest = json.loads((args.dataset.parent / "manifest.json").read_text())
    selected = [dict(ep, split=split[ep["index"]]) for ep in episodes if ep["index"] in split]
    common = {
        "checkpoint": {"path": str(snapshot), "revision": checkpoint_revision(snapshot),
                       "repo": snapshot.parent.parent.name.replace("models--", "").replace("--", "/")},
        "dataset": {"path": str(args.dataset), "logical_name": dataset_manifest.get("logical_name"),
                    "source": dataset_manifest.get("canonical_locator"),
                    "prepared_content_digest": dataset_manifest.get("prepared_content_digest"),
                    "episodes_total": len(episodes)},
        "selection": {"seed": args.seed, "episodes_per_task": args.episodes_per_task, "val_per_task": args.val_per_task,
                      "frame_stride": args.frame_stride, "split_rule": "per task: seeded sample, first val_per_task -> val",
                      "episodes": selected, "frames": len(frames)},
        "model_query": {"loader": "prismatic.extern.hf.modeling_prismatic.OpenVLAForActionPrediction (external/openvla)",
                        "attn_implementation": args.attn, "dtype": "bfloat16", "prompt": PROMPT,
                        "call": "predict_action(do_sample=False), batch 1", "unnorm_key": unnorm_key,
                        "action_dim": action_dim, "prompt_tokens_per_task": prompt_tokens},
        "preprocessing": {"tf": "jpeg -> tf lanczos3 224 antialias, round, clip uint8 -> crop_and_resize area 0.9",
                          "processor": "PrismaticImageProcessor.from_pretrained(snapshot) (both branches)"},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "tensorflow": tf.__version__,
                     "tensorflow_datasets": tfds.__version__, "python": platform.python_version(),
                     "cuda_device": torch.cuda.get_device_name(0)},
    }
    summary = {}
    for name, layer in layers.items():
        target = args.out / short_name(name)
        target.mkdir()
        x_val = sample_rows(torch.cat(layer.val_x), args.val_token_sample, args.seed)
        x_train = sample_rows(torch.cat(layer.train_x), args.train_token_sample, args.seed + 1)
        st = layer.stats
        save_file({"c_train": st["train"].second_moment.cpu(), "mean_train": st["train"].mean.cpu(),
                   "c_val": st["val"].second_moment.cpu(), "mean_val": st["val"].mean.cpu(),
                   "x_val": x_val.contiguous(), "x_train_sample": x_train.contiguous()},
                  str(target / "calibration.safetensors"))
        teacher = {"weight": layer.module.weight.detach().cpu().contiguous()}
        if layer.module.bias is not None:
            teacher["bias"] = layer.module.bias.detach().cpu().contiguous()
        save_file(teacher, str(target / "teacher_layer.safetensors"))
        w = teacher["weight"]
        fl = layer.floor
        manifest = {
            "schema": "quic_wc.calibration/v2", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "layer": {"name": name, "out_features": w.shape[0], "in_features": w.shape[1],
                      "dtype": str(w.dtype).replace("torch.", ""), "bias": "bias" in teacher,
                      "weight_sha256": sha256_tensor(w),
                      "bias_sha256": sha256_tensor(teacher["bias"]) if "bias" in teacher else None},
            **common,
            "tokens": {"train": st["train"].count, "val": st["val"].count, "x_val_stored": x_val.shape[0],
                       "x_train_sample_stored": x_train.shape[0], "hook_calls": layer.calls,
                       "x_val_rule": f"seeded subsample of all val tokens to <= {args.val_token_sample}",
                       "x_train_sample_rule": f"every {args.frame_stride}th train token (global count), seeded subsample"},
            "checks": {**layer.check, "bf16_module_vs_fp32_reference_val": {
                "rel_l2_with_bias": (fl["err"] / fl["ref"]) ** 0.5, "cos_with_bias_mean": fl["cos_sum"] / fl["tokens"],
                "cos_with_bias_min": fl["cos_min"]}},
            "seconds": time.perf_counter() - t0,
        }
        manifest["files"] = {n: hashlib.sha256((target / n).read_bytes()).hexdigest()
                             for n in ("calibration.safetensors", "teacher_layer.safetensors")}
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summary[name] = {"dir": str(target), "shape": list(w.shape), "tokens": manifest["tokens"], "checks": manifest["checks"]}
    (args.out / "summary.json").write_text(json.dumps({"layers": summary, **common, "seconds": time.perf_counter() - t0},
                                                      indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
