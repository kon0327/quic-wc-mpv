"""Distribution-shift check: saved QuIC-WC artifacts and SVD/ASVD baselines on other LIBERO suites.

All compressed weights are fitted on the original calibration (train split); here they are only
evaluated on every collected token (train + val statistics; stored token samples) of other
calibration sets of the same layer. QuIC-WC is loaded from its compact artifact (including its
stored bias), so the check also exercises the deployable format.

Usage: python -m quic_wc.pilot.shift --calibration <orig> --results <grid dir> --shift <dir> [<dir> ...] --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import load_compact
from ..metrics import relative_output_error, token_metrics
from .run import BF16, Context, bf16, svd_weight


def load_shift(path: Path, device: str) -> dict[str, Any]:
    from safetensors.torch import load_file

    manifest = json.loads((path / "manifest.json").read_text())
    cal = load_file(str(path / "calibration.safetensors"))
    n_tr, n_va = manifest["tokens"]["train"], manifest["tokens"]["val"]
    total = n_tr + n_va
    return {"name": manifest["dataset"]["logical_name"], "tokens_total": total,
            "weight_sha256": manifest["layer"]["weight_sha256"],
            "c": ((cal["c_train"] * n_tr + cal["c_val"] * n_va) / total).to(device),
            "mean": ((cal["mean_train"] * n_tr + cal["mean_val"] * n_va) / total).to(device),
            "x": torch.cat([cal["x_val"], cal["x_train_sample"]]).to(device)}


def evaluate(ctx: Context, shift: dict[str, Any], weight_hat: torch.Tensor, bias_hat: torch.Tensor | None) -> dict[str, float]:
    exact = relative_output_error(weight_hat, ctx.weight, shift["c"], ctx.bias, shift["mean"], bias_hat=bias_hat)
    tokens = token_metrics(shift["x"], weight_hat, ctx.weight, ctx.bias, bias_hat)
    return {"exact_rel_l2_no_bias": exact["rel_l2_no_bias"], "cos_no_bias_mean": tokens["cos_no_bias_mean"],
            "cos_no_bias_p05": tokens["cos_no_bias_p05"], "cos_with_bias_mean": tokens["cos_with_bias_mean"],
            "tokens_sampled": tokens["tokens"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--shift", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="*", default=[4, 8, 12, 16, 25, 32, 50, 64, 99, 128, 256])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.out / "shift_results.json").exists():
        raise FileExistsError("shift_results.json exists; use a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ctx = Context(args.calibration, args.device)
    shifts = [load_shift(p, args.device) for p in args.shift]
    for s in shifts:
        if s["weight_sha256"] != ctx.manifest["layer"]["weight_sha256"]:
            raise ValueError(f"{s['name']}: calibration belongs to a different layer weight")
    grid = json.loads((args.results / "results.json").read_text())
    in_dist = {e["name"]: e for e in grid["quic_wc"]}
    rows: list[dict[str, Any]] = []
    for policy in ("fixed", "corrected"):
        for method in ("svd", "asvd"):
            for rank in [0] + args.ranks if method == "svd" else args.ranks:
                svd = ctx.svd if method == "svd" else ctx.asvd(ctx.weight, policy)
                w_hat = svd_weight(svd, rank)
                b_hat = ctx.bias_hat(w_hat, ctx.weight, policy)
                row = {"name": f"{'bias_only' if rank == 0 else method}_r{rank}_{policy}", "kind": "baseline",
                       "method": "bias_only" if rank == 0 else method, "rank": rank, "bias_policy": policy,
                       "byte_ratio": ((ctx.out_features + ctx.in_features) * rank + ctx.out_features)
                       / (ctx.out_features * ctx.in_features + ctx.out_features),
                       "in_distribution": ctx.evaluate(w_hat, None, policy)["exact_val_rel_l2_no_bias"]}
                row |= {s["name"]: evaluate(ctx, s, w_hat, b_hat) for s in shifts}
                rows.append(row)
    for name, entry in in_dist.items():
        if entry["teacher"]["kind"] != "real":
            continue
        module = load_compact(args.results / "artifacts" / name, device=args.device)
        with torch.no_grad():
            w_hat = bf16(module.reconstruct_weight())
        b_hat = module.bias.to(torch.float32) if entry["bias_policy"] == "corrected" else None
        row = {"name": name, "kind": "quic_wc", "bias_policy": entry["bias_policy"], "parameters": entry["parameters"],
               "byte_ratio": entry["byte_ratio"], "byte_ratio_theta_bf16": 2 * entry["parameters"]
               / (2 * (ctx.out_features * ctx.in_features + ctx.out_features)),
               "in_distribution": entry["metrics"]["exact_val_rel_l2_no_bias"],
               "in_distribution_reloaded": ctx.evaluate(w_hat, None, "fixed", deploy_rounding=False)[
                   "exact_val_rel_l2_no_bias"] if b_hat is None else None}
        row |= {s["name"]: evaluate(ctx, s, w_hat, b_hat) for s in shifts}
        rows.append(row)
    out = {"schema": "quic_wc.shift_results/v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "calibration": str(args.calibration), "results": str(args.results),
           "shifts": [{"name": s["name"], "path": str(p), "tokens_total": s["tokens_total"],
                       "tokens_sampled": int(s["x"].shape[0])} for s, p in zip(shifts, args.shift)],
           "rows": rows, "seconds": time.perf_counter() - t0}
    (args.out / "shift_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    names = [s["name"] for s in shifts]
    lines = ["| varianta | bias | poměr bajtů | relL2 LIBERO-10 val | " + " | ".join(f"relL2 {n} | cos_nb {n}" for n in names) + " |",
             "|---|---|---:|---:|" + "---:|---:|" * len(names)]
    for row in rows:
        cells = " | ".join(f"{row[n]['exact_rel_l2_no_bias']:.4f} | {row[n]['cos_no_bias_mean']:.4f}" for n in names)
        lines.append(f"| {row['name']} | {row['bias_policy']} | {row['byte_ratio']:.4f} | {row['in_distribution']:.4f} | {cells} |")
    (args.out / "shift_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
