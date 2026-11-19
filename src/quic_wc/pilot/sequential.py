"""Sequential calibration of a chain of DINOv2 layers (experiment 2026-11-19).

Independent per-layer fits see the inputs of the unchanged model; inside a compressed group every layer gets
inputs shifted by the errors of the layers before it (group experiment 2026-11-18). Here the layers are
calibrated and compressed one after another in forward order: layer k is calibrated on the inputs of a copy of
the featurizer in which layers 0..k-1 are already replaced by their compressed versions, then fitted, and its
deployable weight is installed in that copy before layer k+1 is calibrated.

Targets (``--target``):

- ``input``: classic sequential calibration (GPTQ-style). The teacher stays ``W, b``; only the input statistics
  come from the compressed chain, so the layer reproduces the original layer on shifted inputs.
- ``output``: asymmetric calibration. The target is the output of the *unchanged* model, ``W x_ref + b``, for
  the paired input ``x_c`` of the compressed chain (same frame, same token). With centred moments
  ``S_cc = E[x̄_c x̄_cᵀ]`` and ``S_rc = E[x̄_ref x̄_cᵀ]`` the error of any ``Ŵ`` with the corrected bias is
  ``‖Ŵ − W_t‖²_{S_cc} + const`` where ``W_t = W S_rc S_cc⁻¹`` and the teacher bias ``b_t = b + W μ_ref − W_t μ_c``
  matches the means. The ridge shrinks the *correction* towards the original weight,
  ``W_t = W + W (S_rc − S_cc)(S_cc + λI)⁻¹`` (λ = ``ridge`` · tr S_cc / n; λ = 0 is exact), so directions the
  data do not constrain keep ``W`` and identical inputs give ``W_t = W`` exactly. ``W_t, b_t`` are written as
  the teacher of the calibration, so ``quic_wc.pilot.run`` (QuIC-WC and ASVD) works unchanged.

Only layers of ``vision_backbone.featurizer`` (DINOv2) are supported: their inputs depend on the image only, so
the featurizer alone is run (``get_intermediate_layers(n={len(blocks)-2})`` exactly as inside OpenVLA, batch 1,
pixel values from the official processor). A forward hook stops the forward pass after the hooked layer.

Every calibration directory has the files of ``collect_vla`` plus ``x_val_ref`` (unchanged-model inputs
paired with ``x_val``) and diagnostics in ``manifest.json["sequential"]``. ``<out>/chain.json`` is the list of
replacements in the format of ``m4_lite`` and allows resuming an interrupted run.

  chain: python -m quic_wc.pilot.sequential chain --checkpoint <snapshot> --dataset <tfds dir> --out <dir>
         --layers <name> ... --method quic|asvd --target input|output [--grid <json> --stages 16]
         [--asvd-ranks-from <m4 plan> --asvd-variant <name>] [--reference-calibration <dir> ...]
  plan:  python -m quic_wc.pilot.sequential plan --chains <dir> ... --baseline-plan <m4 plan>
         --baseline-variants <name> ... --fill-from <variant> --out <plan.json>
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ..metrics import SecondMoment
from .collect_vla import PROMPT, short_name

PREFIX = "vision_backbone.featurizer."
RIDGE_GRID = (0.0, 1e-6, 1e-4, 1e-2)


class _Stop(Exception):
    """Raised by the capture hook to end the forward pass after the hooked layer."""


class CrossMoment:
    """Streaming FP64 ``sum x_ref x_cᵀ`` for paired token rows."""

    def __init__(self, dim: int, device: str) -> None:
        self.sum_outer = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        self.count = 0

    @torch.no_grad()
    def update(self, x_ref: torch.Tensor, x_c: torch.Tensor) -> None:
        self.sum_outer.addmm_(x_ref.double().T, x_c.double())
        self.count += x_ref.shape[0]

    @property
    def moment(self) -> torch.Tensor:
        return self.sum_outer / max(self.count, 1)


def output_target(weight: torch.Tensor, bias: torch.Tensor | None, c_cc: torch.Tensor, mu_c: torch.Tensor,
                  c_rc: torch.Tensor, mu_r: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher ``(W_t, b_t)`` of the asymmetric objective from uncentred moments ``E[x_c x_cᵀ]``,
    ``E[x_ref x_cᵀ]`` and the means (all FP64). ``ridge`` is relative to the mean eigenvalue of ``S_cc``."""
    w = weight.double()
    s_cc = c_cc - torch.outer(mu_c, mu_c)
    s_rc = c_rc - torch.outer(mu_r, mu_c)
    n = s_cc.shape[0]
    lam = ridge * float(torch.trace(s_cc)) / n
    reg = s_cc + lam * torch.eye(n, dtype=s_cc.dtype, device=s_cc.device)
    # W_t = W + W (S_rc - S_cc) reg^-1  <=>  reg (W_t - W)ᵀ = (S_rc - S_cc)ᵀ Wᵀ  (reg symmetric)
    w_t = w + torch.linalg.solve(reg, (s_rc - s_cc).T @ w.T).T
    b = torch.zeros(w.shape[0], dtype=torch.float64, device=w.device) if bias is None else bias.double()
    b_t = b + w @ mu_r - w_t @ mu_c
    return w_t, b_t


def rel_l2(y: torch.Tensor, y_ref: torch.Tensor, norm: torch.Tensor) -> float:
    return float(((y.double() - y_ref.double()) ** 2).sum() / (norm.double() ** 2).sum()) ** 0.5


def paired_sample(n: int, limit: int, seed: int) -> torch.Tensor:
    if n <= limit:
        return torch.arange(n)
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:limit].sort().values


# ----------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------

def load_pixels(args: argparse.Namespace, processor, preprocess) -> list[dict[str, Any]]:
    """Frames of ``collect_vla`` (same split, order and stride) with their DINOv2 pixel values (CPU, BF16)."""
    import tensorflow_datasets as tfds

    from .collect import list_episodes, select_split

    (builder, _), episodes = list_episodes(args.dataset)
    split = select_split(episodes, args.episodes_per_task, args.val_per_task, args.seed)
    decoders = {"steps": {"observation": {"image": tfds.decode.SkipDecoding(),
                                          "wrist_image": tfds.decode.SkipDecoding()}}}
    ds = builder.as_dataset(split="train", shuffle_files=False, decoders=decoders)
    frames = []
    for index, episode in enumerate(ds):
        if index not in split:
            continue
        steps = next(iter(episode["steps"].batch(100_000)))
        images = steps["observation"]["image"].numpy().tolist()
        task = steps["language_instruction"].numpy()[0].decode()
        prompt = PROMPT.format(task=task.lower())
        for frame in range(0, len(images), args.frame_stride):
            pixels = processor(prompt, preprocess.pil_image(images[frame]))["pixel_values"]
            frames.append({"episode": index, "frame": frame, "split": split[index],
                           "pixels": pixels[:, :3].to(torch.bfloat16).contiguous()})
    return frames


# ----------------------------------------------------------------------------------------------------------
# chain
# ----------------------------------------------------------------------------------------------------------

class Capture:
    def __init__(self) -> None:
        self.x: torch.Tensor | None = None

    def hook(self, _module, inputs, _output) -> None:
        self.x = inputs[0].detach()
        raise _Stop


def run_featurizer(featurizer, pixels: torch.Tensor, n_blocks: int) -> None:
    try:
        with torch.inference_mode():
            featurizer.get_intermediate_layers(pixels, n={n_blocks - 2})
    except _Stop:
        return
    raise RuntimeError("hooked layer was not reached")


def collect_layer(args, name: str, ref, comp, frames, n_blocks: int, out_dir: Path) -> dict[str, Any]:
    """Paired statistics of one layer (unchanged ``ref`` vs compressed chain ``comp``) and its calibration dir."""
    from safetensors.torch import save_file

    from .collect import checkpoint_revision, sha256_tensor

    sub = name[len(PREFIX):]
    m_ref, m_comp = ref.get_submodule(sub), comp.get_submodule(sub)
    dim = m_ref.in_features
    dev = "cuda"
    cap_r, cap_c = Capture(), Capture()
    handles = [m_ref.register_forward_hook(cap_r.hook), m_comp.register_forward_hook(cap_c.hook)]
    st_c = {"train": SecondMoment(dim, dev), "val": SecondMoment(dim, dev)}
    st_r = {"train": SecondMoment(dim, dev), "val": SecondMoment(dim, dev)}
    cross = CrossMoment(dim, dev)
    val_c, val_r, train_x = [], [], []
    offset, calls = 0, {"train": 0, "val": 0}
    try:
        for f in frames:
            px = f["pixels"].to(dev)
            run_featurizer(ref, px, n_blocks)
            run_featurizer(comp, px, n_blocks)
            xr, xc = cap_r.x.reshape(-1, dim), cap_c.x.reshape(-1, dim)
            part = f["split"]
            calls[part] += 1
            st_c[part].update(xc)
            st_r[part].update(xr)
            if part == "val":
                val_c.append(xc.cpu())
                val_r.append(xr.cpu())
            else:
                cross.update(xr, xc)
                first = (-offset) % args.frame_stride
                train_x.append(xc[first:: args.frame_stride].cpu())
                offset += xc.shape[0]
    finally:
        for h in handles:
            h.remove()

    w, b = m_ref.weight.detach().float(), None if m_ref.bias is None else m_ref.bias.detach().float()
    c_cc, mu_c = st_c["train"].second_moment, st_c["train"].mean
    keep = paired_sample(sum(x.shape[0] for x in val_c), args.val_token_sample, args.seed)
    x_val, x_val_ref = torch.cat(val_c)[keep], torch.cat(val_r)[keep]
    x_tr = torch.cat(train_x)
    x_tr = x_tr[paired_sample(x_tr.shape[0], args.train_token_sample, args.seed + 1)]

    # diagnostics on the stored paired validation tokens: error against the unchanged model's output
    xv, xvr = x_val.to(dev).float(), x_val_ref.to(dev).float()
    y_ref_nb = F.linear(xvr, w)
    y_ref = y_ref_nb if b is None else y_ref_nb + b
    diag: dict[str, Any] = {"input_shift_rel_l2_val": rel_l2(xv, xvr, xvr),
                            "original_layer_vs_reference_output": rel_l2(F.linear(xv, w, b), y_ref, y_ref_nb)}
    lin = {}
    for ridge in RIDGE_GRID:
        try:
            w_r, b_r = output_target(w.to(dev), b if b is None else b.to(dev), c_cc, mu_c, cross.moment,
                                     st_r["train"].mean, ridge)
            lin[str(ridge)] = rel_l2(F.linear(xv.double(), w_r, b_r), y_ref, y_ref_nb)
        except RuntimeError as exc:  # singular without ridge
            lin[str(ridge)] = f"failed: {exc}"[:80]
    diag["linear_target_vs_reference_output_by_ridge"] = lin

    if args.target == "output":
        w_t, b_t = output_target(w.to(dev), b if b is None else b.to(dev), c_cc, mu_c, cross.moment,
                                 st_r["train"].mean, args.ridge)
        teacher = {"weight": w_t.float().cpu().contiguous(), "bias": b_t.float().cpu().contiguous()}
        if b is None:
            raise ValueError("--target output needs a layer with bias (corrected bias policy)")
    else:
        teacher = {"weight": m_ref.weight.detach().cpu().contiguous()}
        if m_ref.bias is not None:
            teacher["bias"] = m_ref.bias.detach().cpu().contiguous()

    target = out_dir / "calibration" / short_name(name)
    target.mkdir(parents=True, exist_ok=True)
    save_file({"c_train": c_cc.cpu(), "mean_train": mu_c.cpu(), "c_val": st_c["val"].second_moment.cpu(),
               "mean_val": st_c["val"].mean.cpu(), "x_val": x_val.contiguous(), "x_train_sample": x_tr.contiguous(),
               "x_val_ref": x_val_ref.contiguous()}, str(target / "calibration.safetensors"))
    save_file(teacher, str(target / "teacher_layer.safetensors"))
    ref_check = {}
    if args.reference_calibration:
        from safetensors.torch import load_file
        for root in args.reference_calibration:
            path = root / short_name(name) / "calibration.safetensors"
            if path.exists():
                old = load_file(str(path))
                ref_check = {"source": str(path),
                             "c_train_max_abs_diff": float((old["c_train"].to(dev) - st_r["train"].second_moment).abs().max()),
                             "c_train_max_abs": float(old["c_train"].abs().max()),
                             "c_val_max_abs_diff": float((old["c_val"].to(dev) - st_r["val"].second_moment).abs().max())}
    manifest = {
        "schema": "quic_wc.calibration/v2", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "layer": {"name": name, "out_features": w.shape[0], "in_features": dim, "dtype": "bfloat16",
                  "bias": b is not None, "weight_sha256": sha256_tensor(m_ref.weight.detach()),
                  "teacher": "original" if args.target == "input" else "asymmetric output target (float32)"},
        "sequential": {"target": args.target, "ridge": args.ridge if args.target == "output" else None,
                       "method": args.method, "upstream_replaced": args.upstream, "diagnostics": diag,
                       "reference_path_vs_original_calibration": ref_check},
        "selection": {"seed": args.seed, "episodes_per_task": args.episodes_per_task,
                      "val_per_task": args.val_per_task, "frame_stride": args.frame_stride, "frames": len(frames)},
        "tokens": {"train": st_c["train"].count, "val": st_c["val"].count, "x_val_stored": x_val.shape[0],
                   "x_train_sample_stored": x_tr.shape[0], "hook_calls": calls},
        "checkpoint": {"path": str(args.checkpoint.resolve()), "revision": checkpoint_revision(args.checkpoint),
                       "repo": args.checkpoint.resolve().parent.parent.name.replace("models--", "").replace("--", "/")},
        "checks": {"note": "sequential calibration: see sequential.diagnostics"},
    }
    manifest["files"] = {n: hashlib.sha256((target / n).read_bytes()).hexdigest()
                         for n in ("calibration.safetensors", "teacher_layer.safetensors")}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"dir": target, "diag": diag, "ref_check": ref_check}


def fit_quic(args, name: str, calibration: Path, out_dir: Path) -> dict[str, Any]:
    fit_dir = out_dir / "fits" / short_name(name)
    if not (fit_dir / "results.json").exists():
        log = out_dir / "fits" / f"{short_name(name)}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as fh:
            subprocess.run([sys.executable, "-m", "quic_wc.pilot.run", "--calibration", str(calibration),
                            "--grid", str(args.grid), "--out", str(fit_dir)], check=True, stdout=fh,
                           stderr=subprocess.STDOUT)
    res = json.loads((fit_dir / "results.json").read_text())
    from .summarize import METRIC
    entry = next(e for e in res["quic_wc"] if e["teacher"]["kind"] == "real" and e["stages"][0] == args.stages)
    return {"layer": name, "kind": "quic", "artifact": str(fit_dir / "artifacts" / entry["name"]),
            "calibration": str(calibration), "policy": entry["bias_policy"], "byte_ratio": entry["byte_ratio"],
            "pilot_rel_l2": entry["metrics"][METRIC], "stages": entry["stages"]}


def deployable(rep: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
    """Dense BF16 weight and bias exactly as ``m4_lite.Swap`` deploys the replacement."""
    if rep["kind"] == "quic":
        from ..checkpoint import load_compact
        module = load_compact(rep["artifact"], device="cuda").eval()
        w = module.decompress(torch.bfloat16)
        return w.detach(), module._cached_bias, None
    from .run import Context, bf16, svd_weight
    from .summarize import METRIC
    ctx = Context(Path(rep["calibration"]), "cuda")
    w = bf16(svd_weight(ctx.asvd(ctx.weight, rep["policy"]), rep["rank"]))
    b_hat = ctx.bias_hat(w, ctx.weight, rep["policy"])
    check = ctx.evaluate(w, ctx.weight, rep["policy"])[METRIC]
    return w.to(torch.bfloat16), None if b_hat is None else b_hat.to(torch.bfloat16), check


def install(comp, name: str, w: torch.Tensor, b: torch.Tensor | None) -> None:
    m = comp.get_submodule(name[len(PREFIX):])
    m.weight.data.copy_(w.to(m.weight.device, m.weight.dtype))
    if b is not None:
        m.bias.data.copy_(b.to(m.bias.device, m.bias.dtype))


def chain(args: argparse.Namespace) -> None:
    for name in args.layers:
        if not name.startswith(PREFIX):
            raise ValueError(f"only {PREFIX}* layers are supported: {name}")
    if args.method == "quic" and (args.grid is None or args.stages is None):
        raise ValueError("--method quic needs --grid and --stages")
    asvd_ranks = {}
    if args.method == "asvd":
        plan = json.loads(args.asvd_ranks_from.read_text())
        v = next(v for v in plan["variants"] if v["name"] == args.asvd_variant)
        asvd_ranks = {r["layer"]: r for r in v["replacements"] if r["kind"] == "asvd"}
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "chain.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": {k: str(v) for k, v in vars(args).items()},
        "replacements": []}

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from transformers import AutoTokenizer
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    from .collect import Preprocessor

    t0 = time.perf_counter()
    snapshot = args.checkpoint.resolve()
    vla = OpenVLAForActionPrediction.from_pretrained(snapshot, attn_implementation="flash_attention_2",
                                                     torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    ref = vla.vision_backbone.featurizer.to("cuda").eval()
    del vla
    comp = copy.deepcopy(ref)
    n_blocks = len(ref.blocks)
    processor = PrismaticProcessor(PrismaticImageProcessor.from_pretrained(snapshot), AutoTokenizer.from_pretrained(snapshot))
    frames = load_pixels(args, processor, Preprocessor(snapshot))
    if args.max_frames:  # smoke tests only
        frames = [f for f in frames if f["split"] == "train"][: args.max_frames] + \
                 [f for f in frames if f["split"] == "val"][: args.max_frames]
    print(f"{len(frames)} frames, setup {time.perf_counter() - t0:.0f}s", flush=True)

    done = {r["layer"]: r for r in state["replacements"]}
    args.upstream = []
    for name in args.layers:
        t1 = time.perf_counter()
        if name in done:
            rep = done[name]
            w, b, _ = deployable(rep)
            install(comp, name, w, b)
            args.upstream.append(name)
            print(f"reinstalled {name}", flush=True)
            continue
        info = collect_layer(args, name, ref, comp, frames, n_blocks, args.out)
        t_collect = time.perf_counter() - t1
        if args.method == "quic":
            rep = fit_quic(args, name, info["dir"], args.out)
        else:
            src = asvd_ranks[name]
            rep = {"layer": name, "kind": "asvd", "rank": src["rank"], "calibration": str(info["dir"]),
                   "policy": src["policy"], "byte_ratio": src["byte_ratio"]}
        w, b, check = deployable(rep)
        if check is not None:
            rep["pilot_rel_l2"] = check
        # error of the installed layer against the unchanged model's output on the paired validation tokens
        from safetensors.torch import load_file
        cal = load_file(str(info["dir"] / "calibration.safetensors"))
        m_ref = ref.get_submodule(name[len(PREFIX):])
        xv, xvr = cal["x_val"].cuda().float(), cal["x_val_ref"].cuda().float()
        y_ref_nb = F.linear(xvr, m_ref.weight.float())
        y_ref = y_ref_nb + m_ref.bias.float() if m_ref.bias is not None else y_ref_nb
        y_hat = F.linear(xv, w.float(), None if b is None else b.float())
        y_loc_nb = F.linear(xv, m_ref.weight.float())
        y_loc = y_loc_nb + m_ref.bias.float() if m_ref.bias is not None else y_loc_nb
        rep["sequential"] = {**info["diag"], "installed_vs_reference_output": rel_l2(y_hat, y_ref, y_ref_nb),
                             "installed_vs_original_layer_same_input": rel_l2(y_hat, y_loc, y_loc_nb),
                             "reference_check": info["ref_check"], "seconds_collect": t_collect,
                             "seconds_total": time.perf_counter() - t1}
        install(comp, name, w, b)
        args.upstream.append(name)
        state["replacements"].append(rep)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        s = rep["sequential"]
        print(f"{name}: shift {s['input_shift_rel_l2_val']:.4f} orig→ref {s['original_layer_vs_reference_output']:.4f} "
              f"installed→ref {s['installed_vs_reference_output']:.4f} local {rep.get('pilot_rel_l2', float('nan')):.4f} "
              f"collect {t_collect:.0f}s total {s['seconds_total']:.0f}s", flush=True)
    state["seconds"] = time.perf_counter() - t0
    state_path.write_text(json.dumps(state, indent=2) + "\n")


def plan(args: argparse.Namespace) -> None:
    """m4_lite plan: reference, selected baseline variants, one group variant per chain (layers missing in the
    chain are taken from ``--fill-from``), reference at the end."""
    base = json.loads(args.baseline_plan.read_text())
    byname = {v["name"]: v for v in base["variants"]}
    variants = [{"name": "reference", "replacements": []}]
    variants += [copy.deepcopy(byname[n]) for n in args.baseline_variants]
    fill = byname[args.fill_from]["replacements"] if args.fill_from else []
    for directory in args.chains:
        reps = json.loads((directory / "chain.json").read_text())["replacements"]
        have = {r["layer"] for r in reps}
        extra = [r for r in fill if r["layer"] not in have]
        variants.append({"name": f"seq:{directory.name}", "replacements": reps + extra,
                         "filled_from_baseline": [r["layer"] for r in extra]})
    variants.append({"name": "reference_end", "replacements": []})
    args.out.write_text(json.dumps({"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "sources": [str(d) for d in args.chains], "variants": variants}, indent=2) + "\n")
    print(f"{len(variants)} variants -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("chain")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--dataset", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--layers", nargs="+", required=True)
    c.add_argument("--method", choices=("quic", "asvd"), required=True)
    c.add_argument("--target", choices=("input", "output"), required=True)
    c.add_argument("--grid", type=Path)
    c.add_argument("--stages", type=int)
    c.add_argument("--asvd-ranks-from", type=Path)
    c.add_argument("--asvd-variant")
    c.add_argument("--ridge", type=float, default=1e-4, help="relative damping of the correction (fixed before the run)")
    c.add_argument("--reference-calibration", type=Path, nargs="*", default=[])
    c.add_argument("--episodes-per-task", type=int, default=5)
    c.add_argument("--val-per-task", type=int, default=1)
    c.add_argument("--frame-stride", type=int, default=10)
    c.add_argument("--val-token-sample", type=int, default=32768)
    c.add_argument("--train-token-sample", type=int, default=32768)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--max-frames", type=int, default=0, help="per split; smoke tests only")
    p = sub.add_parser("plan")
    p.add_argument("--chains", type=Path, nargs="+", required=True)
    p.add_argument("--baseline-plan", type=Path, required=True)
    p.add_argument("--baseline-variants", nargs="*", default=[])
    p.add_argument("--fill-from")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    chain(args) if args.command == "chain" else plan(args)


if __name__ == "__main__":
    main()
