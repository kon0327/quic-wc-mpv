"""Re-export fitted QuIC-WC artifacts in a lower storage dtype and re-evaluate them (no refitting).

For every entry of a finished grid, the FP32 artifact is loaded, theta/sigma are rounded to
``--storage-dtype``, a corrected bias is recomputed from the *rounded* weight, and the new
artifact is written, reloaded and evaluated exactly like ``quic_wc.pilot.run`` (including the
equal-byte SVD/ASVD references and gate M3). ``metrics_unrounded`` must reproduce the original
grid metrics; the script checks that and records the difference.

Usage: python -m quic_wc.pilot.requantize --calibration <dir> --results <grid dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import load_compact
from ..decomposition import thin_svd
from .run import Context, build_teacher, finalize, layer_gate, low_rank, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, nargs="+",
                        help="grid result directories; later directories override entries of the same name")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--storage-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if (args.out / "results.json").exists():
        raise FileExistsError(f"{args.out}/results.json exists; use a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ctx = Context(args.calibration, args.device)
    sources: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for directory in args.results:
        grid_results = json.loads((directory / "results.json").read_text())
        specs = {s["name"]: s for s in grid_results["grid"]["experiments"]}
        for entry in grid_results["quic_wc"]:
            sources[entry["name"]] = (directory, entry, specs[entry["name"]])
    results: dict[str, Any] = {
        "schema": "quic_wc.pilot_results/v3", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "derived_from": [str(d) for d in args.results], "storage_dtype": args.storage_dtype,
        "calibration": str(args.calibration), "layer": ctx.manifest["layer"], "checkpoint": ctx.manifest["checkpoint"],
        "tokens": ctx.manifest["tokens"], "has_bias": ctx.has_bias, "torch": torch.__version__,
        "baselines": [], "quic_wc": [], "consistency": []}
    for policy in ctx.policies(["fixed", "corrected"]):
        for rank in (12, 25, 49, 50, 99, 128):
            results["baselines"].append(low_rank(ctx, "asvd", rank, ctx.weight, policy, ctx.svd)[1])
    for name, (directory, entry, spec) in sources.items():
        module = load_compact(directory / "artifacts" / name, device=ctx.device)
        config = module.config
        teacher, teacher_info = build_teacher(ctx, spec, config)
        teacher_svd = ctx.svd if teacher_info["kind"] == "real" else thin_svd(teacher)
        policy = entry["bias_policy"]
        extra = {"fit": entry["fit"], "fit_config": entry["fit"].get("config"), "source": str(directory / "artifacts" / name)}
        new = finalize(ctx, spec, name, module, teacher, teacher_info, policy, args.storage_dtype, args.out, extra,
                       teacher_svd)
        original = entry["metrics"]["exact_val_rel_l2_no_bias"]
        reproduced = new["metrics_unrounded"]["exact_val_rel_l2_no_bias"]
        check = {"name": name, "original_fp32": original, "reproduced_fp32": reproduced,
                 "abs_diff": abs(original - reproduced), "ok": abs(original - reproduced) <= args.tolerance,
                 "original_theta_bf16_in_memory": entry.get("metrics_theta_bf16", {}).get("exact_val_rel_l2_no_bias"),
                 "storage_dtype_from_artifact": new["metrics"]["exact_val_rel_l2_no_bias"]}
        results["consistency"].append(check)
        results["quic_wc"].append(new)
        g = new["gate_m3"]
        print(f"{name}: fp32 {original:.4f} (repro {reproduced:.4f}) -> {args.storage_dtype} "
              f"{check['storage_dtype_from_artifact']:.4f}, bytes {new['byte_ratio']:.4f}"
              + (f" | ASVD r={g['asvd_rank']} {g['asvd']:.4f} M3 {'pass' if g['pass'] else 'fail'}" if g else ""),
              flush=True)
    results["gate_m3"] = layer_gate(results["quic_wc"])
    results["all_consistent"] = all(c["ok"] for c in results["consistency"])
    results["seconds"] = time.perf_counter() - t0
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.out / "results.md").write_text(markdown(results), encoding="utf-8")
    print(f"all_consistent={results['all_consistent']}")


if __name__ == "__main__":
    main()
