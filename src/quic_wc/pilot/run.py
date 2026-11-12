"""Pilot M2/M3 for one layer: SVD/ASVD baselines and QuIC-WC distillation with controls.

Every candidate is evaluated in its *deployable* form: SVD factors rounded to BF16 (the dense
baseline dtype); QuIC-WC reconstructed in FP32 from FP32 parameters and cast to BF16 like the
inference cache; any corrected bias rounded to BF16. Metrics use validation episodes only: exact
output errors from the validation second moment and per-token cosine on the stored validation
tokens. Fitting and ASVD use the train statistics only.

Bias policies: "fixed" keeps the teacher bias (instructions/phase3.md). "corrected" stores
``b_hat = b + (W - W_hat) mu_train`` instead (same storage), which removes the mean output error;
the fit then minimises the error of the train-centred inputs.

Equal-budget references for each QuIC-WC entry: SVD and ASVD (activation-aware SVD) with the
largest rank whose stored parameters (``r_params``) or tensor bytes (``r_bytes``, BF16 factors vs
FP32 theta) do not exceed the QuIC-WC entry, with the same teacher and bias policy.

Usage: python -m quic_wc.pilot.run --calibration <dir> --grid configs/<grid>.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import artifact_bytes, load_compact, save_compact, storage_report
from ..config import QuICWCConfig, SideConfig
from ..decomposition import activation_aware_svd, low_rank_parameter_count, thin_svd
from ..distillation import FitConfig, fit
from ..layers.linear import QuICWCLinear
from ..metrics import relative_output_error, token_metrics

BF16 = torch.bfloat16
BIAS_POLICIES = ("fixed", "corrected")


def bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(BF16).to(torch.float32)


class Context:
    def __init__(self, calibration: Path, device: str) -> None:
        from safetensors.torch import load_file

        self.manifest = json.loads((calibration / "manifest.json").read_text())
        cal = load_file(str(calibration / "calibration.safetensors"))
        layer = load_file(str(calibration / "teacher_layer.safetensors"))
        self.device = device
        self.weight = layer["weight"].to(device=device, dtype=torch.float32)  # exact BF16 values
        self.bias = layer["bias"].to(device=device, dtype=torch.float32)
        self.c_train = cal["c_train"].to(device)
        self.mean_train = cal["mean_train"].to(device)
        self.c_val = cal["c_val"].to(device)
        self.mean_val = cal["mean_val"].to(device)
        self.x_val = cal["x_val"].to(device)
        self.out_features, self.in_features = self.weight.shape
        mu, mv = self.mean_train, self.mean_val
        # Objectives per bias policy (train) and the matching validation error matrices.
        self.objective = {"fixed": (self.c_train, self.c_val),
                          "corrected": (self.c_train - torch.outer(mu, mu),
                                        self.c_val - torch.outer(mv, mu) - torch.outer(mu, mv) + torch.outer(mu, mu))}
        self.svd = thin_svd(self.weight)
        self._asvd_cache: dict[tuple, Any] = {}

    def asvd(self, teacher: torch.Tensor, policy: str):
        # The cached entry keeps a reference to its teacher: a freed tensor's address could
        # otherwise be reused by the next teacher and return a stale decomposition.
        key = (teacher.data_ptr(), policy)
        cached = self._asvd_cache.get(key)
        if cached is None or cached[0] is not teacher:
            cached = (teacher, activation_aware_svd(teacher, self.objective[policy][0]))
            self._asvd_cache[key] = cached
        return cached[1]

    def bias_hat(self, weight_hat: torch.Tensor, teacher: torch.Tensor, policy: str) -> torch.Tensor | None:
        if policy == "fixed":
            return None
        return bf16(self.bias.double() + (teacher.double() - weight_hat.double()) @ self.mean_train)

    def evaluate(self, weight_hat: torch.Tensor, teacher: torch.Tensor | None = None, policy: str = "fixed",
                 deploy_rounding: bool = True) -> dict[str, float]:
        teacher = self.weight if teacher is None else teacher
        deployed = bf16(weight_hat) if deploy_rounding else weight_hat.detach().float()
        b_hat = self.bias_hat(deployed, teacher, policy)
        exact = relative_output_error(deployed, teacher, self.c_val, self.bias, self.mean_val, bias_hat=b_hat)
        tokens = token_metrics(self.x_val, deployed, teacher, self.bias, b_hat)
        fro = float((deployed.double() - teacher.double()).norm() / teacher.double().norm())
        return {"exact_val_" + k: v for k, v in exact.items()} | tokens | {"weight_rel_fro": fro}


def svd_weight(svd, rank: int) -> torch.Tensor:
    """Rank-r factors stored as BF16 ``A = U_r diag(s_r)``, ``B = V_r``."""
    if rank == 0:
        return torch.zeros(svd.U.shape[0], svd.V.shape[0], device=svd.U.device)
    return bf16(svd.U[:, :rank] * svd.S[:rank]) @ bf16(svd.V[:, :rank]).T


def low_rank(ctx: Context, method: str, rank: int, teacher: torch.Tensor, policy: str,
             teacher_svd=None) -> tuple[torch.Tensor, dict[str, Any]]:
    if method == "svd" or rank == 0:
        weight_hat = svd_weight(teacher_svd or thin_svd(teacher), rank)
    else:
        weight_hat = svd_weight(ctx.asvd(teacher, policy), rank)
    m, n = ctx.out_features, ctx.in_features
    params, dense = low_rank_parameter_count(m, n, rank), m * n + m
    return weight_hat, {"method": method, "rank": rank, "bias_policy": policy, "parameters": params,
                        "parameter_ratio": params / dense, "tensor_bytes": 2 * params, "byte_ratio": params / dense,
                        "metrics": ctx.evaluate(weight_hat, teacher, policy)}


def negative_teacher(ctx: Context, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    m, n, k = ctx.out_features, ctx.in_features, ctx.svd.S.numel()
    u = torch.linalg.qr(torch.randn(m, k, generator=g, dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(n, k, generator=g, dtype=torch.float64))[0]
    weight = (u * ctx.svd.S.double().cpu()) @ v.T
    return bf16(weight.to(ctx.device))  # BF16-representable like the real layer


def side(spec: dict[str, Any]) -> SideConfig:
    return SideConfig.from_dict(spec)


def build_teacher(ctx: Context, spec: dict[str, Any], config: QuICWCConfig) -> tuple[torch.Tensor, dict[str, Any]]:
    kind = spec.get("teacher", "real")
    info: dict[str, Any] = {"kind": kind}
    if kind == "real":
        return ctx.weight, info
    seed = spec.get("teacher_seed", 1)
    if kind == "positive_control":
        tspec = spec.get("teacher_config", {})
        tcfg = config.with_updates(sigma_mode="fixed", **{k: side(v) for k, v in tspec.items() if k in ("left", "right")})
        gen = torch.Generator(device="cpu").manual_seed(seed)
        tmod = QuICWCLinear(tcfg, ctx.svd.S[: config.rank].clone(), device=ctx.device)
        scale = spec.get("teacher_theta_scale", 0.3)
        with torch.no_grad():
            tmod.left.theta.copy_(scale * torch.randn(tmod.left.theta.shape, generator=gen))
            tmod.right.theta.copy_(scale * torch.randn(tmod.right.theta.shape, generator=gen))
            teacher = bf16(tmod.reconstruct_weight())
        return teacher, info | {"theta_scale": scale, "seed": seed, "config": tcfg.to_dict()}
    if kind == "negative_control":
        return negative_teacher(ctx, seed), info | {
            "construction": "random orthonormal bases with the real singular values", "seed": seed}
    raise ValueError(f"unknown teacher {kind!r}")


def equal_budget(ctx: Context, teacher: torch.Tensor, policy: str, params: int, tensor_bytes: int,
                 teacher_svd) -> dict[str, Any]:
    m, n = ctx.out_features, ctx.in_features
    ranks = {"r_params": max(0, (params - m) // (m + n)), "r_bytes": max(0, (tensor_bytes // 2 - m) // (m + n))}
    out: dict[str, Any] = dict(ranks)
    for label, rank in ranks.items():
        for method in ("svd", "asvd"):
            _, entry = low_rank(ctx, method, rank, teacher, policy, teacher_svd)
            out[f"{method}_{label}"] = entry
    return out


def run_quic(ctx: Context, spec: dict[str, Any], defaults: dict[str, Any], out: Path) -> dict[str, Any]:
    torch.manual_seed(spec.get("seed", 0))
    m, n = ctx.out_features, ctx.in_features
    config = QuICWCConfig(in_features=n, out_features=m, rank=spec["rank"], left=side(spec["left"]),
                          right=side(spec.get("right", spec["left"])), selection=spec.get("selection", "spread"),
                          sigma_mode=spec.get("sigma_mode", "learnable"))
    policy = spec.get("bias_policy", defaults.get("bias_policy", "fixed"))
    if policy not in BIAS_POLICIES:
        raise ValueError(f"bias_policy must be one of {BIAS_POLICIES}")
    teacher, teacher_info = build_teacher(ctx, spec, config)
    c_fit, c_sel = ctx.objective[policy]
    if spec.get("objective_override") == "identity":  # isotropic diagnostic (weight-space fit)
        c_fit = c_sel = torch.eye(n, dtype=c_fit.dtype, device=c_fit.device)
    sigma0 = torch.linalg.svdvals(teacher)[: config.rank]
    student = QuICWCLinear(config, sigma0, ctx.bias.to(BF16), inference_dtype=BF16, device=ctx.device)
    fit_cfg = FitConfig(**(defaults.get("fit", {}) | spec.get("fit", {})))
    result = fit(student, teacher, c_fit.float(), c_val=c_sel.float(), cfg=fit_cfg, log_every=spec.get("log_every"))
    with torch.no_grad():
        w_hat = student.reconstruct_weight()
        metrics = ctx.evaluate(w_hat, teacher, policy)
        # Storage variant: theta and sigma rounded to BF16 before reconstruction.
        saved = [student.left.theta.detach().clone(), student.right.theta.detach().clone(), student.sigma0.clone()]
        student.left.theta.copy_(bf16(saved[0]))
        student.right.theta.copy_(bf16(saved[1]))
        student.sigma0.copy_(bf16(student.sigma()) / (student.sigma_gain if student.sigma_gain is not None else 1))
        metrics_bf16 = ctx.evaluate(student.reconstruct_weight(), teacher, policy)
        student.left.theta.copy_(saved[0])
        student.right.theta.copy_(saved[1])
        student.sigma0.copy_(saved[2])
        b_hat = ctx.bias_hat(bf16(w_hat), teacher, policy)
        if b_hat is not None:
            student.bias.copy_(b_hat.to(BF16))
    report = storage_report(config, bias=True)
    counts = report["parameters"]
    bytes_theta_bf16 = 2 * counts["total"]
    art_dir = out / "artifacts" / spec["name"]
    save_compact(student, art_dir, metadata={"pilot": spec["name"], "teacher": teacher_info, "bias_policy": policy,
                                             "layer": ctx.manifest["layer"], "fit": fit_cfg.to_dict()})
    reloaded = load_compact(art_dir, device=ctx.device)
    roundtrip = float((reloaded.reconstruct_weight() - w_hat).abs().max())
    teacher_svd = ctx.svd if teacher_info["kind"] == "real" else thin_svd(teacher)
    return {"name": spec["name"], "kind": "quic_wc", "description": spec.get("description", ""),
            "teacher": teacher_info, "bias_policy": policy, "objective_override": spec.get("objective_override"),
            "config": config.to_dict(), "parameters": counts["total"], "parameter_breakdown": counts,
            "parameter_ratio": report["parameter_ratio"], "tensor_bytes": report["tensor_bytes"],
            "byte_ratio": report["byte_ratio"], "tensor_bytes_theta_bf16": bytes_theta_bf16,
            "artifact_bytes": artifact_bytes(art_dir), "roundtrip_max_abs": roundtrip,
            "fit": result.to_dict() | {"config": fit_cfg.to_dict()}, "metrics": metrics,
            "metrics_theta_bf16": metrics_bf16,
            "equal_budget": equal_budget(ctx, teacher, policy, counts["total"], report["tensor_bytes"], teacher_svd)}


def fmt(v: float | None, digits: int = 4) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


def markdown(results: dict[str, Any]) -> str:
    lay, ck = results["layer"], results["checkpoint"]
    lines = ["# Pilot QuIC-WC — výsledky", "",
             f"Vrstva `{lay['name']}` ({lay['out_features']}×{lay['in_features']}), checkpoint "
             f"`{ck['repo']}@{ck['revision'][:12]}`; validační tokeny: {results['tokens']['val']}.", "",
             "Metriky na validačních epizodách, váhy v nasaditelné podobě (BF16). `relL2` = přesná relativní L2 chyba "
             "výstupu vztažená k výstupu bez biasu; `cos_nb` / `cos_b` = průměrná kosinová podobnost tokenů bez / s biasem; "
             "`p05` = 5. percentil `cos_nb`. Poměr bajtů je vůči husté BF16 vrstvě včetně biasu.", "",
             "## Baseline", "", "| metoda | bias | rank | parametry | poměr bajtů | relL2 | cos_nb | p05 | cos_b |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in results["baselines"]:
        mt = e["metrics"]
        lines.append(f"| {e['method']} | {e['bias_policy']} | {e['rank']} | {e['parameters']} | {fmt(e['byte_ratio'], 3)} | "
                     f"{fmt(mt['exact_val_rel_l2_no_bias'])} | {fmt(mt['cos_no_bias_mean'])} | "
                     f"{fmt(mt['cos_no_bias_p05'])} | {fmt(mt['cos_with_bias_mean'])} |")
    lines += ["", "## QuIC-WC", "",
              "| varianta | učitel | bias | θ | parametry | poměr bajtů | relL2 | cos_nb | p05 | cos_b | relL2 θ-BF16 | "
              "ASVD r_params (r: relL2) | ASVD r_bytes (r: relL2) | SVD r_params (r: relL2) | stav @ best | s |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|"]
    for e in results["quic_wc"]:
        mt, eq, pb = e["metrics"], e["equal_budget"], e["parameter_breakdown"]

        def ref(key: str) -> str:
            return f"{eq[key]['rank']}: {fmt(eq[key]['metrics']['exact_val_rel_l2_no_bias'])}"

        lines.append(f"| {e['name']} | {e['teacher']['kind']} | {e['bias_policy']} | {pb['theta_out'] + pb['theta_in']} | "
                     f"{e['parameters']} | {fmt(e['byte_ratio'], 4)} | {fmt(mt['exact_val_rel_l2_no_bias'])} | "
                     f"{fmt(mt['cos_no_bias_mean'])} | {fmt(mt['cos_no_bias_p05'])} | {fmt(mt['cos_with_bias_mean'])} | "
                     f"{fmt(e['metrics_theta_bf16']['exact_val_rel_l2_no_bias'])} | {ref('asvd_r_params')} | "
                     f"{ref('asvd_r_bytes')} | {ref('svd_r_params')} | {e['fit']['status']} @ {e['fit']['best_step']} | "
                     f"{e['fit']['seconds']:.0f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*", default=None, help="run only these QuIC-WC experiment names")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    if (args.out / "results.json").exists():
        raise FileExistsError(f"{args.out}/results.json exists; use a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    grid = json.loads(args.grid.read_text())
    t0 = time.perf_counter()
    ctx = Context(args.calibration, args.device)
    s = ctx.svd.S.double()
    results: dict[str, Any] = {
        "schema": "quic_wc.pilot_results/v2", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calibration": str(args.calibration), "grid_file": str(args.grid), "grid": grid,
        "layer": ctx.manifest["layer"], "checkpoint": ctx.manifest["checkpoint"], "tokens": ctx.manifest["tokens"],
        "bf16_floor": ctx.manifest["checks"], "torch": torch.__version__,
        "spectrum": {"singular_values_head": s[:8].tolist(),
                     "weight_energy_at_rank": {str(r): float((s[:r] ** 2).sum() / (s ** 2).sum())
                                               for r in (16, 64, 256, 512, 768)},
                     "c_train_eigen_fraction_top": {str(k): float(v) for k, v in zip(
                         (1, 4, 16, 64), (torch.linalg.eigvalsh(ctx.c_train).flip(0).cumsum(0)
                                          / torch.trace(ctx.c_train))[[0, 3, 15, 63]].tolist())},
                     "mean_fraction_of_trace": float(ctx.mean_train @ ctx.mean_train / torch.trace(ctx.c_train))},
        "baselines": [], "quic_wc": [],
    }
    if not args.skip_baselines:
        for policy in grid.get("bias_policies", ["fixed"]):
            results["baselines"].append(low_rank(ctx, "svd", 0, ctx.weight, policy, ctx.svd)[1] | {"method": "bias_only"})
            for method in ("svd", "asvd"):
                for rank in grid.get(f"{method}_ranks", []):
                    results["baselines"].append(low_rank(ctx, method, rank, ctx.weight, policy, ctx.svd)[1])
        print(f"baselines done in {time.perf_counter() - t0:.1f}s", flush=True)
    for spec in grid["experiments"]:
        if args.only and spec["name"] not in args.only:
            continue
        t1 = time.perf_counter()
        entry = run_quic(ctx, spec, grid.get("defaults", {}), args.out)
        results["quic_wc"].append(entry)
        mt, eq = entry["metrics"], entry["equal_budget"]
        print(f"{spec['name']}: relL2 {mt['exact_val_rel_l2_no_bias']:.4f} cos_nb {mt['cos_no_bias_mean']:.4f} "
              f"params {entry['parameters']} | ASVD r={eq['r_bytes']} "
              f"{eq['asvd_r_bytes']['metrics']['exact_val_rel_l2_no_bias']:.4f} "
              f"[{entry['fit']['status']} best@{entry['fit']['best_step']}] {time.perf_counter() - t1:.0f}s", flush=True)
        (args.out / "results.partial.json").write_text(json.dumps(results, indent=2) + "\n")
    results["seconds"] = time.perf_counter() - t0
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.out / "results.md").write_text(markdown(results), encoding="utf-8")
    partial = args.out / "results.partial.json"
    if partial.exists():
        partial.unlink()
    print(markdown(results))


if __name__ == "__main__":
    main()
