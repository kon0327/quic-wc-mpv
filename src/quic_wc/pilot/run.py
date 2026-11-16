"""Pilot M2/M3 for one layer: SVD/ASVD baselines and QuIC-WC distillation with controls.

Every candidate is evaluated in its *deployable* form: SVD factors rounded to BF16 (the dense
baseline dtype); QuIC-WC reloaded from its compact artifact (theta/sigma in ``storage_dtype``,
reconstructed in FP32 and cast to BF16 like the inference cache); any corrected bias rounded to
BF16. Metrics use validation episodes only: exact output errors from the validation second moment
and per-token cosine on the stored validation tokens. Fitting and ASVD use the train statistics.

Bias policies: "fixed" keeps the teacher bias (instructions/phase3.md). "corrected" stores
``b_hat = b + (W - W_hat) mu_train`` instead (same storage), which removes the mean output error;
the fit then minimises the error of the train-centred inputs.

Equal-budget references for each QuIC-WC entry (same teacher and bias policy): SVD and ASVD
(activation-aware, whitened SVD) with the largest rank whose stored parameters (``r_params``) or
tensor bytes (``r_bytes``, BF16 factors) do not exceed the QuIC-WC entry, and the smallest rank
whose bytes are at least as large (``r_bytes_ceil``).

Gate M3 (decided 2026-11-12): a real-layer entry passes when its validation relL2 without bias is
lower than that of ASVD at ``r_bytes_ceil`` (ASVD gets at least as many tensor bytes), at a byte
ratio <= ``GATE_MAX_BYTE_RATIO``. A layer passes when some entry passes for that bias policy.

Spec extensions: ``storage_dtype`` (theta/sigma storage, default float32), ``restarts``
(``{"count", "scale", "screen_steps", "seed"}``: screen random inits on the train objective,
continue the best), ``growth`` (list of stage counts for both sides; needs
``stage_form="conjugated"``; every size is fitted, exported and reported as its own entry),
``checkpoint_segments`` (activation recomputation for large layers; results are unchanged),
``stages_sweep`` (independent from-scratch fits for each listed stage count), ``lr_depth_scaling``
(``{"reference_stages", "power"}``: LR (and sigma LR) times ``min(1, (L_ref/L)^power)``).

Usage: python -m quic_wc.pilot.run --calibration <dir> --grid configs/<grid>.json --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import QUANTIZED, artifact_bytes, load_compact, save_compact, storage_report, storage_tensors
from ..config import QuICWCConfig, SideConfig
from ..decomposition import activation_aware_svd, low_rank_parameter_count, thin_svd
from ..distillation import FitConfig, fit
from ..growth import grow_stages, perturb_
from ..layers.linear import QuICWCLinear
from ..metrics import relative_output_error, token_metrics

BF16 = torch.bfloat16
BIAS_POLICIES = ("fixed", "corrected")
GATE_MAX_BYTE_RATIO = 0.25


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
        self.has_bias = "bias" in layer
        self.bias = layer["bias"].to(device=device, dtype=torch.float32) if self.has_bias else None
        self.c_train = cal["c_train"].to(device)
        self.mean_train = cal["mean_train"].to(device)
        self.c_val = cal["c_val"].to(device)
        self.mean_val = cal["mean_val"].to(device)
        self.x_val = cal["x_val"].to(device)
        self.out_features, self.in_features = self.weight.shape
        self.dense_parameters = self.out_features * self.in_features + (self.out_features if self.has_bias else 0)
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

    def policies(self, requested: list[str]) -> list[str]:
        """Bias policies applicable to this layer ("corrected" needs a layer with bias)."""
        return [p for p in requested if self.has_bias or p == "fixed"]

    def check_policy(self, policy: str) -> None:
        if policy not in BIAS_POLICIES:
            raise ValueError(f"bias_policy must be one of {BIAS_POLICIES}")
        if policy == "corrected" and not self.has_bias:
            raise ValueError("bias_policy 'corrected' needs a layer with bias")

    def bias_hat(self, weight_hat: torch.Tensor, teacher: torch.Tensor, policy: str) -> torch.Tensor | None:
        if policy == "fixed":
            return None
        return bf16(self.bias.double() + (teacher.double() - weight_hat.double()) @ self.mean_train)

    def factor_bytes(self, rank: int) -> int:
        """Tensor bytes of BF16 rank-r factors plus the BF16 bias (if the layer has one)."""
        return 2 * low_rank_parameter_count(self.out_features, self.in_features, rank, bias=self.has_bias)

    def evaluate(self, weight_hat: torch.Tensor, teacher: torch.Tensor | None = None, policy: str = "fixed",
                 deploy_rounding: bool = True) -> dict[str, float]:
        teacher = self.weight if teacher is None else teacher
        deployed = bf16(weight_hat) if deploy_rounding else weight_hat.detach().float()
        b_hat = self.bias_hat(deployed, teacher, policy)
        exact = relative_output_error(deployed, teacher, self.c_val, self.bias, self.mean_val, bias_hat=b_hat)
        tokens = token_metrics(self.x_val, deployed, teacher, self.bias, b_hat)
        fro = float((deployed.double() - teacher.double()).norm() / teacher.double().norm())
        return {"exact_val_" + k: v for k, v in exact.items()} | tokens | {"weight_rel_fro": fro}

    @property
    def max_rank(self) -> int:
        return min(self.out_features, self.in_features)


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
    params = low_rank_parameter_count(m, n, rank, bias=ctx.has_bias)
    return weight_hat, {"method": method, "rank": rank, "bias_policy": policy, "parameters": params,
                        "parameter_ratio": params / ctx.dense_parameters, "tensor_bytes": ctx.factor_bytes(rank),
                        "byte_ratio": params / ctx.dense_parameters, "metrics": ctx.evaluate(weight_hat, teacher, policy)}


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
    bias_params = m if ctx.has_bias else 0
    r_bytes = min(ctx.max_rank, max(0, (tensor_bytes // 2 - bias_params) // (m + n)))
    ceil = r_bytes if ctx.factor_bytes(r_bytes) >= tensor_bytes else min(ctx.max_rank, r_bytes + 1)
    ranks = {"r_params": min(ctx.max_rank, max(0, (params - bias_params) // (m + n))), "r_bytes": r_bytes,
             "r_bytes_ceil": ceil}
    out: dict[str, Any] = dict(ranks)
    done: dict[tuple[str, int], dict[str, Any]] = {}
    for label, rank in ranks.items():
        for method in ("svd", "asvd"):
            if (method, rank) not in done:
                done[(method, rank)] = low_rank(ctx, method, rank, teacher, policy, teacher_svd)[1]
            out[f"{method}_{label}"] = done[(method, rank)]
    return out


def gate_m3(entry: dict[str, Any]) -> dict[str, Any] | None:
    if entry["teacher"]["kind"] != "real":
        return None
    asvd = entry["equal_budget"]["asvd_r_bytes_ceil"]
    quic_err = entry["metrics"]["exact_val_rel_l2_no_bias"]
    asvd_err = asvd["metrics"]["exact_val_rel_l2_no_bias"]
    eligible = entry["byte_ratio"] <= GATE_MAX_BYTE_RATIO
    return {"metric": "exact_val_rel_l2_no_bias", "asvd_rank": asvd["rank"], "asvd_tensor_bytes": asvd["tensor_bytes"],
            "quic_tensor_bytes": entry["tensor_bytes"], "quic": quic_err, "asvd": asvd_err,
            "relative_margin": 1 - quic_err / asvd_err if asvd_err > 0 else None,
            "eligible": eligible, "pass": bool(eligible and quic_err < asvd_err)}


def layer_gate(entries: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"definition": f"relL2 bez biasu < ASVD (r_bytes_ceil, BF16), poměr bajtů <= {GATE_MAX_BYTE_RATIO}"}
    for policy in BIAS_POLICIES:
        gated = [e for e in entries if e.get("gate_m3") and e["bias_policy"] == policy]
        if gated:
            best = max(gated, key=lambda e: e["gate_m3"]["relative_margin"] if e["gate_m3"]["eligible"] else -math.inf)
            out[policy] = {"pass": any(e["gate_m3"]["pass"] for e in gated), "entries": len(gated),
                           "passing": [e["name"] for e in gated if e["gate_m3"]["pass"]], "best": best["name"],
                           "best_margin": best["gate_m3"]["relative_margin"]}
    return out


def expand_sweeps(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``stages_sweep: [L1, L2, ...]`` -> independent from-scratch specs ``<name>_L<k>`` (both sides)."""
    out = []
    for spec in specs:
        if "stages_sweep" not in spec:
            out.append(spec)
            continue
        if spec.get("growth"):
            raise ValueError("use either stages_sweep or growth")
        for stages in spec["stages_sweep"]:
            sides = {k: dict(spec.get(k, spec["left"]), num_stages=stages) for k in ("left", "right")}
            item = {k: v for k, v in spec.items() if k != "stages_sweep"} | sides
            out.append(item | {"name": f"{spec['name']}_L{stages}", "sweep_of": spec["name"]})
    return out


def make_config(ctx: Context, spec: dict[str, Any], stages: int | None = None) -> QuICWCConfig:
    left = dict(spec["left"])
    right = dict(spec.get("right", spec["left"]))
    if stages is not None:
        left["num_stages"] = right["num_stages"] = stages
    rank = spec["rank"] if spec["rank"] != "full" else ctx.max_rank
    return QuICWCConfig(in_features=ctx.in_features, out_features=ctx.out_features, rank=rank, left=side(left),
                        right=side(right), selection=spec.get("selection", "spread"),
                        sigma_mode=spec.get("sigma_mode", "learnable"))


@torch.no_grad()
def round_to_storage(module: QuICWCLinear, storage_dtype: str) -> None:
    """Replace theta/sigma by their storage-dtype values (gain folded), exactly as after reload."""
    tensors = storage_tensors(module, storage_dtype)
    dtype = module.config.dtype
    module.load_exported({k: (v.to(dtype) if k in QUANTIZED else v).to(module.sigma0.device)
                          for k, v in tensors.items()})


def screen_restarts(ctx: Context, make_student, teacher: torch.Tensor, c_fit: torch.Tensor, fit_cfg: FitConfig,
                    restarts: dict[str, Any]) -> tuple[QuICWCLinear, dict[str, Any]]:
    """Short fits from identity (restart 0) and random inits; keep the best on the train objective."""
    gen = torch.Generator().manual_seed(restarts.get("seed", 0))
    screen_cfg = FitConfig(**(fit_cfg.to_dict() | {"steps": restarts["screen_steps"], "restore_best": False,
                                                  "betas": tuple(fit_cfg.betas)}))
    candidates = []
    for k in range(restarts["count"]):
        student = make_student()
        if k > 0:
            perturb_(student, restarts["scale"], gen)
        result = fit(student, teacher, c_fit.float(), cfg=screen_cfg)
        train_now = result.history[-1].get("train_rel_mse_after", result.final_train_rel_mse)
        candidates.append((train_now, k, student))
    candidates.sort(key=lambda c: c[0])
    info = {"count": restarts["count"], "scale": restarts["scale"], "screen_steps": restarts["screen_steps"],
            "screen_train_rel_mse": {str(k): v for v, k, _ in sorted(candidates, key=lambda c: c[1])},
            "chosen": candidates[0][1]}
    return candidates[0][2], info


def finalize(ctx: Context, spec: dict[str, Any], name: str, student: QuICWCLinear, teacher: torch.Tensor,
             teacher_info: dict[str, Any], policy: str, storage_dtype: str, out: Path, extra: dict[str, Any],
             teacher_svd) -> dict[str, Any]:
    """Round to storage, set the corrected bias, export, reload and evaluate the reloaded artifact."""
    student = copy.deepcopy(student)
    config = student.config
    with torch.no_grad():
        metrics_unrounded = ctx.evaluate(student.reconstruct_weight(), teacher, policy)
        round_to_storage(student, storage_dtype)
        w_rounded = student.reconstruct_weight()
        b_hat = ctx.bias_hat(bf16(w_rounded), teacher, policy)
        if b_hat is not None:
            student.bias.copy_(b_hat.to(BF16))
    art_dir = out / "artifacts" / name
    save_compact(student, art_dir, storage_dtype=storage_dtype,
                 metadata={"pilot": name, "teacher": teacher_info, "bias_policy": policy,
                           "layer": ctx.manifest["layer"], "fit": extra.get("fit_config")})
    reloaded = load_compact(art_dir, device=ctx.device)
    with torch.no_grad():
        w_reloaded = reloaded.reconstruct_weight()
        roundtrip = float((w_reloaded - w_rounded).abs().max())
        if policy == "corrected" and not torch.equal(reloaded.bias.float(), b_hat):
            raise RuntimeError("stored corrected bias differs from the evaluated one")
        metrics = ctx.evaluate(w_reloaded, teacher, policy)
    report = storage_report(config, bias=ctx.has_bias, storage_dtype=storage_dtype)
    counts = report["parameters"]
    entry = {"name": name, "kind": "quic_wc", "description": spec.get("description", ""), "spec": spec["name"],
             "family": spec.get("sweep_of", spec["name"]),
             "teacher": teacher_info, "bias_policy": policy, "objective_override": spec.get("objective_override"),
             "config": config.to_dict(), "stages": [config.left.num_stages, config.right.num_stages],
             "storage_dtype": report["storage_dtype"], "parameters": counts["total"], "parameter_breakdown": counts,
             "parameter_ratio": report["parameter_ratio"], "tensor_bytes": report["tensor_bytes"],
             "byte_ratio": report["byte_ratio"], "artifact_bytes": artifact_bytes(art_dir),
             "roundtrip_max_abs": roundtrip, "metrics": metrics, "metrics_unrounded": metrics_unrounded} | extra
    entry["equal_budget"] = equal_budget(ctx, teacher, policy, counts["total"], report["tensor_bytes"], teacher_svd)
    entry["gate_m3"] = gate_m3(entry)
    return entry


def depth_scaled(fit_cfg: FitConfig, scaling: dict[str, Any] | None, stages: int) -> FitConfig:
    """``lr * min(1, (reference_stages / L) ** power)``; ``L`` = deeper side (2026-11-12 finding)."""
    if not scaling:
        return fit_cfg
    factor = min(1.0, (scaling["reference_stages"] / stages) ** scaling.get("power", 1.0))
    if factor == 1.0:
        return fit_cfg
    changes = {"lr": fit_cfg.lr * factor, "betas": tuple(fit_cfg.betas)}
    if fit_cfg.sigma_lr is not None:
        changes["sigma_lr"] = fit_cfg.sigma_lr * factor
    return FitConfig(**(fit_cfg.to_dict() | changes))


def run_quic(ctx: Context, spec: dict[str, Any], defaults: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    torch.manual_seed(spec.get("seed", 0))
    growth = spec.get("growth")
    config = make_config(ctx, spec, growth[0] if growth else None)
    policy = spec.get("bias_policy", defaults.get("bias_policy", "fixed"))
    ctx.check_policy(policy)
    storage_dtype = spec.get("storage_dtype", defaults.get("storage_dtype", "float32"))
    teacher, teacher_info = build_teacher(ctx, spec, config)
    teacher_svd = ctx.svd if teacher_info["kind"] == "real" else thin_svd(teacher)
    c_fit, c_sel = ctx.objective[policy]
    if spec.get("objective_override") == "identity":  # isotropic diagnostic (weight-space fit)
        c_fit = c_sel = torch.eye(ctx.in_features, dtype=c_fit.dtype, device=c_fit.device)
    sigma0 = torch.linalg.svdvals(teacher)[: config.rank]
    bias0 = ctx.bias.to(BF16) if ctx.has_bias else None

    def make_student() -> QuICWCLinear:
        module = QuICWCLinear(config, sigma0, bias0, inference_dtype=BF16, device=ctx.device)
        return module.set_checkpointing(spec.get("checkpoint_segments", 0))

    fit_cfg = FitConfig(**(defaults.get("fit", {}) | spec.get("fit", {})))
    scaling = spec.get("lr_depth_scaling", defaults.get("lr_depth_scaling"))
    if not growth:
        fit_cfg = depth_scaled(fit_cfg, scaling, max(config.left.num_stages, config.right.num_stages))
    restart_info = None
    if spec.get("restarts"):
        student, restart_info = screen_restarts(ctx, make_student, teacher, c_fit, fit_cfg, spec["restarts"])
    else:
        student = make_student()
    entries = []
    for phase, stages in enumerate(growth or [None]):
        t0 = time.perf_counter()
        if phase > 0:
            student = grow_stages(student, left=stages, right=stages)
        phase_cfg = fit_cfg
        if growth and spec.get("growth_steps"):
            phase_cfg = FitConfig(**(fit_cfg.to_dict() | {"steps": spec["growth_steps"][phase],
                                                         "betas": tuple(fit_cfg.betas)}))
        if growth:
            phase_cfg = depth_scaled(phase_cfg, scaling, stages)
        result = fit(student, teacher, c_fit.float(), c_val=c_sel.float(), cfg=phase_cfg, log_every=spec.get("log_every"))
        name = spec["name"] if not growth else f"{spec['name']}_L{stages}"
        extra = {"fit": result.to_dict() | {"config": phase_cfg.to_dict()}, "fit_config": phase_cfg.to_dict(),
                 "growth_phase": phase if growth else None, "restarts": restart_info if phase == 0 else None}
        entry = finalize(ctx, spec, name, student, teacher, teacher_info, policy, storage_dtype, out, extra, teacher_svd)
        entry["seconds_total_phase"] = time.perf_counter() - t0
        entries.append(entry)
    return entries


def fmt(v: float | None, digits: int = 4) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


def markdown(results: dict[str, Any]) -> str:
    lay, ck = results["layer"], results["checkpoint"]
    bias_note = "včetně biasu" if results.get("has_bias", True) else "(vrstva bez biasu)"
    lines = ["# Pilot QuIC-WC — výsledky", "",
             f"Vrstva `{lay['name']}` ({lay['out_features']}×{lay['in_features']}), checkpoint "
             f"`{ck['repo']}@{ck['revision'][:12]}`; validační tokeny: {results['tokens']['val']}.", "",
             "Metriky na validačních epizodách, váhy v nasaditelné podobě (BF16; QuIC-WC načtený z artefaktu). "
             "`relL2` = přesná relativní L2 chyba výstupu vztažená k výstupu bez biasu; `cos_nb` / `cos_b` = průměrná "
             "kosinová podobnost tokenů bez / s biasem; `p05` = 5. percentil `cos_nb`. Poměr bajtů je vůči husté BF16 "
             f"vrstvě {bias_note}. ASVD ≤B / ≥B = největší rank s nejvýše stejnými / nejmenší rank s alespoň stejnými "
             "bajty. M3 = relL2 QuIC-WC < ASVD ≥B při poměru bajtů ≤ " + str(GATE_MAX_BYTE_RATIO) + ".", "",
             "## Baseline", "", "| metoda | bias | rank | parametry | poměr bajtů | relL2 | cos_nb | p05 | cos_b |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in results["baselines"]:
        mt = e["metrics"]
        lines.append(f"| {e['method']} | {e['bias_policy']} | {e['rank']} | {e['parameters']} | {fmt(e['byte_ratio'], 3)} | "
                     f"{fmt(mt['exact_val_rel_l2_no_bias'])} | {fmt(mt['cos_no_bias_mean'])} | "
                     f"{fmt(mt['cos_no_bias_p05'])} | {fmt(mt['cos_with_bias_mean'])} |")
    lines += ["", "## QuIC-WC", "",
              "| varianta | učitel | bias | L (out/in) | uložení θ | parametry | poměr bajtů | relL2 | relL2 nezaokr. | cos_nb | "
              "p05 | ASVD ≤B (r: relL2) | ASVD ≥B (r: relL2) | M3 (rezerva) | best @ | s |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|"]
    for e in results["quic_wc"]:
        mt, eq = e["metrics"], e["equal_budget"]

        def ref(key: str) -> str:
            return f"{eq[key]['rank']}: {fmt(eq[key]['metrics']['exact_val_rel_l2_no_bias'])}"

        gate = e.get("gate_m3")
        gate_cell = "—" if not gate else (f"{'ano' if gate['pass'] else 'ne'} ({fmt(gate['relative_margin'], 3)})"
                                          + ("" if gate["eligible"] else " mimo rozpočet"))
        lines.append(f"| {e['name']} | {e['teacher']['kind']} | {e['bias_policy']} | {e['stages'][0]}/{e['stages'][1]} | "
                     f"{e['storage_dtype']} | {e['parameters']} | {fmt(e['byte_ratio'], 4)} | "
                     f"{fmt(mt['exact_val_rel_l2_no_bias'])} | {fmt(e['metrics_unrounded']['exact_val_rel_l2_no_bias'])} | "
                     f"{fmt(mt['cos_no_bias_mean'])} | {fmt(mt['cos_no_bias_p05'])} | {ref('asvd_r_bytes')} | "
                     f"{ref('asvd_r_bytes_ceil')} | {gate_cell} | {e['fit']['status']} @ {e['fit']['best_step']} | "
                     f"{e['fit']['seconds']:.0f} |")
    gate = results.get("gate_m3", {})
    if gate:
        lines += ["", "## Brána M3", "", f"Definice: {gate['definition']}.", ""]
        for policy in BIAS_POLICIES:
            if policy in gate:
                g = gate[policy]
                lines.append(f"- `{policy}`: {'PROŠLA' if g['pass'] else 'NEPROŠLA'} — prošlo {len(g['passing'])}/"
                             f"{g['entries']} variant; nejlepší rezerva {fmt(g['best_margin'], 3)} (`{g['best']}`).")
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
        "schema": "quic_wc.pilot_results/v3", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "calibration": str(args.calibration), "grid_file": str(args.grid), "grid": grid,
        "layer": ctx.manifest["layer"], "checkpoint": ctx.manifest["checkpoint"], "tokens": ctx.manifest["tokens"],
        "bf16_floor": ctx.manifest["checks"], "torch": torch.__version__, "has_bias": ctx.has_bias,
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
        for policy in ctx.policies(grid.get("bias_policies", ["fixed"])):
            results["baselines"].append(low_rank(ctx, "svd", 0, ctx.weight, policy, ctx.svd)[1] | {"method": "bias_only"})
            for method in ("svd", "asvd"):
                for rank in grid.get(f"{method}_ranks", []):
                    results["baselines"].append(low_rank(ctx, method, rank, ctx.weight, policy, ctx.svd)[1])
        print(f"baselines done in {time.perf_counter() - t0:.1f}s", flush=True)
    for spec in expand_sweeps(grid["experiments"]):
        if args.only and spec["name"] not in args.only and spec.get("sweep_of") not in args.only:
            continue
        policy = spec.get("bias_policy", grid.get("defaults", {}).get("bias_policy", "fixed"))
        if policy not in ctx.policies([policy]):
            print(f"{spec['name']}: skipped (bias policy {policy!r} needs a layer with bias)", flush=True)
            continue
        for entry in run_quic(ctx, spec, grid.get("defaults", {}), args.out):
            results["quic_wc"].append(entry)
            mt, eq = entry["metrics"], entry["equal_budget"]
            print(f"{entry['name']}: relL2 {mt['exact_val_rel_l2_no_bias']:.4f} cos_nb {mt['cos_no_bias_mean']:.4f} "
                  f"bytes {entry['byte_ratio']:.4f} | ASVD r={eq['r_bytes_ceil']} "
                  f"{eq['asvd_r_bytes_ceil']['metrics']['exact_val_rel_l2_no_bias']:.4f} "
                  f"[{entry['fit']['status']} best@{entry['fit']['best_step']}] {entry['seconds_total_phase']:.0f}s",
                  flush=True)
            (args.out / "results.partial.json").write_text(json.dumps(results, indent=2) + "\n")
    results["gate_m3"] = layer_gate(results["quic_wc"])
    results["seconds"] = time.perf_counter() - t0
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.out / "results.md").write_text(markdown(results), encoding="utf-8")
    partial = args.out / "results.partial.json"
    if partial.exists():
        partial.unlink()
    print(markdown(results))


if __name__ == "__main__":
    main()
