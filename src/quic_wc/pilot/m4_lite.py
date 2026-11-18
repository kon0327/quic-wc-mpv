"""M4-lite: compressed layers inside the full OpenVLA — action tokens, logits and in-situ layer error.

The model is loaded and queried exactly as in ``collect_vla`` (BF16, flash-attention 2, TF image pipeline,
official processor, ``predict_action`` greedy) on the *validation* frames of the calibration run (same
episodes and frames). Each variant replaces one or more ``nn.Linear`` layers:

* ``quic``: the BF16 QuIC-WC artifact is loaded with ``load_compact`` and swapped in as a ``QuICWCLinear``
  module, decompressed once (``strict_cache``); the forward is ``F.linear`` on the cached dense weight —
  the deployment path of the plan (decompress at load, dense inference).
* ``asvd``: the activation-aware rank-r factors (BF16, as in the pilot) multiplied into the dense weight.

Measured per variant (reference = the unmodified model in the same run):

* free-running greedy decoding (``predict_action``): agreement of the 7 action tokens with the reference
  (per token and all 7), with the dataset action tokens (reference model: 83.6 % in the calibration run),
  mean |Δ| of the normalized actions and the gripper flip rate;
* teacher-forced logits: one forward on prompt + reference action tokens; KL(reference || variant) of the
  next-token distribution at the 7 action positions and top-1 agreement — no error propagation through
  diverging decodes, so it is the most sensitive and stable metric;
* in-situ layer error: relL2 of the replaced layer's output against the original weight on the inputs it
  actually sees (= ``exact_val_rel_l2_no_bias`` of the pilot for a single replaced layer; with several
  replacements later layers see shifted inputs).

  plan: python -m quic_wc.pilot.m4_lite plan --results <dir> [<dir> ...] --out <plan.json>
  run:  python -m quic_wc.pilot.m4_lite run --checkpoint <snapshot> --dataset <tfds dir> --plan <plan.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .collect_vla import PROMPT, short_name

EMPTY_TOKEN = 29871
METRIC = "exact_val_rel_l2_no_bias"


# ----------------------------------------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------------------------------------
def plan(args: argparse.Namespace) -> None:
    """Single-layer QuIC-WC variants of every real-teacher entry, their equal-byte ASVD (r_bytes_ceil) and the
    combined replacement of all given layers at the smallest and largest common sweep index."""
    variants: list[dict[str, Any]] = [{"name": "reference", "replacements": []}]
    per_layer: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    seen_asvd: set[tuple[str, int, str]] = set()
    for directory in args.results:
        res = json.loads((directory / "results.json").read_text())
        layer, cal = res["layer"]["name"], res["calibration"]
        short = short_name(layer)
        for e in sorted(res["quic_wc"], key=lambda e: (e["bias_policy"], e["byte_ratio"])):
            if e["teacher"]["kind"] != "real" or e.get("family") not in args.families:
                continue
            policy, ref = e["bias_policy"], e["equal_budget"]["asvd_r_bytes_ceil"]
            quic = {"layer": layer, "kind": "quic", "artifact": str(directory / "artifacts" / e["name"]),
                    "calibration": cal, "policy": policy, "byte_ratio": e["byte_ratio"],
                    "pilot_rel_l2": e["metrics"][METRIC], "stages": e["stages"]}
            asvd = {"layer": layer, "kind": "asvd", "rank": ref["rank"], "calibration": cal, "policy": policy,
                    "byte_ratio": ref["byte_ratio"], "pilot_rel_l2": ref["metrics"][METRIC]}
            variants.append({"name": f"{short}:quic:{e['name']}", "replacements": [quic]})
            if (layer, ref["rank"], policy) not in seen_asvd:
                seen_asvd.add((layer, ref["rank"], policy))
                variants.append({"name": f"{short}:asvd:{policy}:r{ref['rank']}", "replacements": [asvd]})
            if policy == "fixed":
                per_layer.setdefault(layer, []).append((quic, asvd))
    if len(per_layer) > 1:  # smallest and largest budget of every layer at once
        for tag, index in (("min", 0), ("max", -1)):
            for kind, pos in (("quic", 0), ("asvd", 1)):
                reps = [pairs[index][pos] for pairs in per_layer.values()]
                variants.append({"name": f"all{len(reps)}:{kind}:{tag}", "replacements": reps})
    variants.append({"name": "reference_end", "replacements": []})
    args.out.write_text(json.dumps({"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "sources": [str(d) for d in args.results], "variants": variants}, indent=2) + "\n")
    print(f"{len(variants)} variants -> {args.out}")


# ----------------------------------------------------------------------------------------------------------
# replacements
# ----------------------------------------------------------------------------------------------------------
def asvd_weights(variants: list[dict[str, Any]], device: str) -> dict[tuple[str, int, str], dict[str, torch.Tensor]]:
    """Deployable BF16 ASVD weights (and corrected biases) computed before the model occupies the GPU."""
    from .run import Context, bf16, svd_weight

    wanted: dict[str, set[tuple[int, str]]] = {}
    for v in variants:
        for r in v["replacements"]:
            if r["kind"] == "asvd":
                wanted.setdefault(r["calibration"], set()).add((r["rank"], r["policy"]))
    out = {}
    for cal, items in wanted.items():
        ctx = Context(Path(cal), device)
        layer = ctx.manifest["layer"]["name"]
        for rank, policy in sorted(items):
            w = bf16(svd_weight(ctx.asvd(ctx.weight, policy), rank))
            entry = {"weight": w.to(torch.bfloat16).cpu()}
            b_hat = ctx.bias_hat(w, ctx.weight, policy)
            if b_hat is not None:
                entry["bias"] = b_hat.to(torch.bfloat16).cpu()
            entry["check_rel_l2"] = torch.tensor(ctx.evaluate(w, ctx.weight, policy)[METRIC])
            out[(layer, rank, policy)] = entry
        del ctx
        torch.cuda.empty_cache()
    return out


class InSitu:
    """relL2 of a replaced layer's output against the original weight on the inputs it actually sees."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        self.weight, self.bias = weight, bias
        self.err = self.ref = 0.0
        self.active = False

    def hook(self, _module, inputs, output) -> None:
        if not self.active:
            return
        x = inputs[0].reshape(-1, self.weight.shape[1])
        y = output.reshape(x.shape[0], -1).float()
        y_nb = F.linear(x, self.weight).float()
        y_ref = y_nb if self.bias is None else y_nb + self.bias.float()
        self.err += float(((y - y_ref).double() ** 2).sum())
        self.ref += float((y_nb.double() ** 2).sum())

    def value(self) -> float:
        return (self.err / self.ref) ** 0.5 if self.ref else float("nan")


class Swap:
    """Applies the replacements of one variant and restores the original modules/weights."""

    def __init__(self, vla: torch.nn.Module, replacements: list[dict[str, Any]],
                 asvd: dict[tuple[str, int, str], dict[str, torch.Tensor]]) -> None:
        self.vla, self.replacements, self.asvd = vla, replacements, asvd
        self.restore: list[Any] = []
        self.monitors: dict[str, InSitu] = {}
        self.quic_modules: dict[str, Any] = {}

    def __enter__(self) -> "Swap":
        from ..checkpoint import load_compact

        for r in self.replacements:
            name = r["layer"]
            parent_name, _, child = name.rpartition(".")
            parent = self.vla.get_submodule(parent_name)
            original = getattr(parent, child)
            monitor = InSitu(original.weight.detach(), None if original.bias is None else original.bias.detach())
            if r["kind"] == "quic":
                module = load_compact(r["artifact"], device=original.weight.device).eval()
                module.decompress(torch.bfloat16)
                module.strict_cache = True  # a second decompression would raise
                setattr(parent, child, module)
                self.quic_modules[name] = module
                handle = module.register_forward_hook(monitor.hook)
                self.restore.append(("module", parent, child, original, handle))
            else:
                monitor.weight = original.weight.detach().clone()  # the in-place copy below overwrites it
                monitor.bias = None if original.bias is None else original.bias.detach().clone()
                rep = self.asvd[(name, r["rank"], r["policy"])]
                original.weight.data.copy_(rep["weight"].to(original.weight.device))
                if "bias" in rep:
                    original.bias.data.copy_(rep["bias"].to(original.bias.device))
                handle = original.register_forward_hook(monitor.hook)
                self.restore.append(("weight", original, monitor.weight, monitor.bias, handle))
            self.monitors[name] = monitor
        return self

    def __exit__(self, *exc) -> None:
        for item in reversed(self.restore):
            if item[0] == "module":
                _, parent, child, original, handle = item
                handle.remove()
                setattr(parent, child, original)
            else:
                _, module, weight, bias, handle = item
                handle.remove()
                module.weight.data.copy_(weight)
                if bias is not None:
                    module.bias.data.copy_(bias)
        self.restore.clear()


# ----------------------------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------------------------
def dataset_tokens(actions: torch.Tensor, stats: dict[str, Any], vocab_size: int, bins: int = 256) -> torch.Tensor:
    """ActionTokenizer: normalize with q01/q99 (masked dims), clip to [-1, 1], digitize, map to vocab tail."""
    a = actions.numpy().astype(np.float64)
    q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
    mask = np.array(stats.get("mask", np.ones_like(q01, dtype=bool)))
    norm = np.where(mask, 2 * (a - q01) / (q99 - q01 + 1e-8) - 1, a)
    norm = np.clip(norm, -1, 1)
    return torch.from_numpy(vocab_size - np.digitize(norm, np.linspace(-1, 1, bins))).long()


def normalized_actions(tokens: torch.Tensor, vla) -> torch.Tensor:
    disc = np.clip(vla.vocab_size - tokens.numpy() - 1, 0, vla.bin_centers.shape[0] - 1)
    return torch.from_numpy(vla.bin_centers[disc]).float()


def evaluate(vla, frames: list[dict[str, Any]], unnorm_key: str, action_dim: int, swap: Swap,
             reference: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    t0 = time.perf_counter()
    generated: list[torch.Tensor] = []
    original_generate = vla.generate

    def recording_generate(*g_args, **g_kwargs):
        out = original_generate(*g_args, **g_kwargs)
        generated.append(out[0, -action_dim:].detach().cpu())
        return out

    vla.generate = recording_generate
    try:
        with torch.inference_mode():
            for f in frames:
                vla.predict_action(input_ids=f["input_ids"].cuda(), attention_mask=f["attention_mask"].cuda(),
                                   pixel_values=f["pixel_values"].cuda(), unnorm_key=unnorm_key, do_sample=False)
    finally:
        vla.generate = original_generate
    t_free = time.perf_counter() - t0
    tokens = torch.stack(generated)
    ref_tokens = tokens if reference is None else reference["tokens"]
    # Teacher forcing on the reference tokens; the in-situ monitors see exactly prefill + 6 decode tokens.
    for m in swap.monitors.values():
        m.active = True
    logp, argmax = [], []
    with torch.inference_mode():
        for f, ref in zip(frames, ref_tokens):
            ids = f["input_ids"]
            if ids[0, -1] != EMPTY_TOKEN:
                ids = torch.cat([ids, torch.tensor([[EMPTY_TOKEN]])], dim=1)
            ids = torch.cat([ids, ref[None, :-1]], dim=1).cuda()
            out = vla(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=f["pixel_values"].cuda())
            lp = torch.log_softmax(out.logits[0, -action_dim:].float(), dim=-1)
            argmax.append(lp.argmax(-1).cpu())
            logp.append(lp.cpu())
    for m in swap.monitors.values():
        m.active = False
    logp_t, argmax_t = torch.stack(logp), torch.stack(argmax)
    metrics: dict[str, Any] = {"seconds_free_running": t_free, "seconds_total": time.perf_counter() - t0,
                               "in_situ_rel_l2": {k: m.value() for k, m in swap.monitors.items()},
                               "decompress_count": {k: int(m.decompress_count) for k, m in swap.quic_modules.items()}}
    if reference is not None:
        ref_lp = reference["logp"]
        kl = (ref_lp.exp() * (ref_lp - logp_t)).sum(-1)  # [frames, 7]
        same = tokens == ref_tokens
        a_var, a_ref = normalized_actions(tokens, vla), reference["norm_actions"]
        metrics |= {
            "free_token_agree_ref": float(same.float().mean()), "free_all7_agree_ref": float(same.all(1).float().mean()),
            "free_token_agree_ref_per_dim": same.float().mean(0).tolist(),
            "free_norm_action_abs_diff": float((a_var - a_ref).abs().mean()),
            "free_gripper_flip_rate": float((torch.sign(a_var[:, -1]) != torch.sign(a_ref[:, -1])).float().mean()),
            "tf_top1_agree_ref": float((argmax_t == ref_tokens).float().mean()),
            "tf_kl_mean": float(kl.mean()), "tf_kl_median": float(kl.median()),
            "tf_kl_p95": float(kl.flatten().quantile(0.95)), "tf_kl_max": float(kl.max()),
        }
    metrics["free_token_agree_dataset"] = float((tokens == frames_dataset_tokens(frames)).float().mean())
    return metrics, {"tokens": tokens, "logp": logp_t, "tf_argmax": argmax_t}


def frames_dataset_tokens(frames: list[dict[str, Any]]) -> torch.Tensor:
    return torch.stack([f["dataset_tokens"] for f in frames])


def load_frames(args: argparse.Namespace, processor, preprocess, vla, unnorm_key: str,
                calibration_root: Path) -> list[dict[str, Any]]:
    import tensorflow_datasets as tfds
    from safetensors.torch import load_file

    from .collect import list_episodes, select_split

    summary = json.loads((calibration_root / "summary.json").read_text())
    sel = summary["selection"]
    (builder, _), episodes = list_episodes(args.dataset)
    split = select_split(episodes, sel["episodes_per_task"], sel["val_per_task"], sel["seed"])
    teacher = load_file(str(calibration_root / "teacher_actions.safetensors"))
    is_val = teacher["episode_frame_isval"][:, 2] == 1
    ref_index = teacher["episode_frame_isval"][is_val][:, :2]
    ref_tokens = teacher["generated_action_tokens"][is_val]
    decoders = {"steps": {"observation": {"image": tfds.decode.SkipDecoding(), "wrist_image": tfds.decode.SkipDecoding()}}}
    ds = builder.as_dataset(split="train", shuffle_files=False, decoders=decoders)
    stats = vla.get_action_stats(unnorm_key)
    frames: list[dict[str, Any]] = []
    for index, episode in enumerate(ds):
        if split.get(index) != "val":
            continue
        steps = next(iter(episode["steps"].batch(100_000)))
        images = steps["observation"]["image"].numpy().tolist()
        actions = torch.from_numpy(steps["action"].numpy())
        task = steps["language_instruction"].numpy()[0].decode()
        prompt = PROMPT.format(task=task.lower())
        for frame in range(0, len(images), sel["frame_stride"]):
            inputs = processor(prompt, preprocess.pil_image(images[frame]))
            frames.append({"episode": index, "frame": frame, "input_ids": inputs["input_ids"],
                           "attention_mask": inputs["attention_mask"],
                           "pixel_values": inputs["pixel_values"].to(torch.bfloat16),
                           "dataset_tokens": dataset_tokens(actions[frame][None], stats, vla.vocab_size)[0]})
    got = torch.tensor([[f["episode"], f["frame"]] for f in frames])
    if not torch.equal(got, ref_index):
        raise RuntimeError("validation frames differ from the calibration run")
    for f, tok in zip(frames, ref_tokens):
        f["calibration_tokens"] = tok
    return frames


def run(args: argparse.Namespace) -> None:
    if args.out.exists() and (args.out / "results.json").exists():
        raise FileExistsError(f"{args.out}/results.json exists")
    args.out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.plan.read_text())
    variants = spec["variants"]
    if args.only:
        variants = [v for v in variants if v["name"] in args.only or v["name"].startswith("reference")]
    t0 = time.perf_counter()
    asvd = asvd_weights(variants, "cuda")
    print(f"ASVD weights: {len(asvd)} in {time.perf_counter() - t0:.0f}s", flush=True)

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from transformers import AutoTokenizer
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    from .collect import Preprocessor, checkpoint_revision

    snapshot = args.checkpoint.resolve()
    vla = OpenVLAForActionPrediction.from_pretrained(snapshot, attn_implementation="flash_attention_2",
                                                     torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda().eval()
    processor = PrismaticProcessor(PrismaticImageProcessor.from_pretrained(snapshot), AutoTokenizer.from_pretrained(snapshot))
    unnorm_key = next(iter(vla.norm_stats))
    action_dim = vla.get_action_dim(unnorm_key)
    calibration_root = Path(next(r["calibration"] for v in variants for r in v["replacements"])).parent
    frames = load_frames(args, processor, Preprocessor(snapshot), vla, unnorm_key, calibration_root)
    if args.max_frames:
        frames = frames[: args.max_frames]  # smoke tests only
    print(f"{len(frames)} validation frames; model+data {time.perf_counter() - t0:.0f}s", flush=True)

    from safetensors.torch import save_file

    results: dict[str, Any] = {"schema": "quic_wc.m4_lite/v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               "plan": str(args.plan), "checkpoint": {"path": str(snapshot),
                                                                      "revision": checkpoint_revision(snapshot)},
                               "frames": len(frames), "max_frames": args.max_frames, "action_dim": action_dim, "variants": []}
    reference = None
    (args.out / "tokens").mkdir(exist_ok=True)
    for v in variants:
        with Swap(vla, v["replacements"], asvd) as swap:
            metrics, data = evaluate(vla, frames, unnorm_key, action_dim, swap,
                                     None if v["name"] == "reference" else reference)
        if v["name"] == "reference":
            reference = {"tokens": data["tokens"], "logp": data["logp"],
                         "norm_actions": normalized_actions(data["tokens"], vla)}
            cal = torch.stack([f["calibration_tokens"] for f in frames])
            metrics["reference_vs_calibration_token_agree"] = float((data["tokens"] == cal).float().mean())
            metrics["tf_top1_agree_ref"] = float((data["tf_argmax"] == data["tokens"]).float().mean())
        for r in v["replacements"]:
            if r["kind"] == "asvd":
                r["check_rel_l2"] = float(asvd[(r["layer"], r["rank"], r["policy"])]["check_rel_l2"])
        save_file({"tokens": data["tokens"], "tf_argmax": data["tf_argmax"]},
                  str(args.out / "tokens" / f"{v['name'].replace(':', '__')}.safetensors"))
        results["variants"].append({"name": v["name"], "replacements": v["replacements"], "metrics": metrics})
        short = {k: (round(x, 4) if isinstance(x, float) else x) for k, x in metrics.items()
                 if k in ("free_token_agree_ref", "free_all7_agree_ref", "tf_top1_agree_ref", "tf_kl_mean",
                          "free_token_agree_dataset", "in_situ_rel_l2", "seconds_total")}
        print(f"{v['name']}: {short}", flush=True)
        (args.out / "results.partial.json").write_text(json.dumps(results, indent=2) + "\n")
    results["seconds"] = time.perf_counter() - t0
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "results.partial.json").unlink(missing_ok=True)
    (args.out / "results.md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))


def markdown(results: dict[str, Any]) -> str:
    lines = ["# M4-lite — komprimované vrstvy v celém OpenVLA", "",
             f"{results['frames']} validačních snímků (LIBERO-10, stejné jako kalibrace). Reference = nezměněný model "
             "ve stejném běhu. `free` = greedy `predict_action`; `tf` = teacher forcing na referenčních tokenech; "
             "`in-situ relL2` = chyba nahrazené vrstvy na vstupech, které v modelu skutečně vidí.", "",
             "| varianta | poměr bajtů | relL2 pilot | in-situ relL2 | tf top-1 | tf KL průměr | tf KL p95 | "
             "free tokeny = ref | free všech 7 | free = dataset | |Δa| norm. | gripper flip |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def f(x, d=4):
        return "—" if x is None else f"{x:.{d}f}"

    for v in results["variants"]:
        m, reps = v["metrics"], v["replacements"]
        ratio = "; ".join(f"{r['byte_ratio']:.4f}" for r in reps) or "1"
        pilot = "; ".join(f"{r['pilot_rel_l2']:.4f}" for r in reps) or "—"
        insitu = "; ".join(f"{x:.4f}" for x in m["in_situ_rel_l2"].values()) or "—"
        lines.append(f"| {v['name']} | {ratio} | {pilot} | {insitu} | {f(m.get('tf_top1_agree_ref'))} | "
                     f"{f(m.get('tf_kl_mean'), 5)} | {f(m.get('tf_kl_p95'), 5)} | {f(m.get('free_token_agree_ref'))} | "
                     f"{f(m.get('free_all7_agree_ref'))} | {f(m.get('free_token_agree_dataset'))} | "
                     f"{f(m.get('free_norm_action_abs_diff'))} | {f(m.get('free_gripper_flip_rate'))} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--families", nargs="+", default=["sweep", "sweep_bc", "sweep_r1024"])
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--dataset", type=Path, required=True)
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--only", nargs="*", default=None, help="run only these variants (plus the references)")
    r.add_argument("--max-frames", type=int, default=0, help="use only the first N validation frames (smoke test)")
    args = parser.parse_args()
    plan(args) if args.command == "plan" else run(args)


if __name__ == "__main__":
    main()
