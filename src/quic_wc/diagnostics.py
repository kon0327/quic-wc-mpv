"""M1 numerical report: precision, gradients at identity and cost of the reconstruction.

Usage: ``python -m quic_wc.diagnostics --out artifacts/m1/diagnostics.json [--device cuda]``
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from .config import QuICWCConfig, SideConfig
from .layers.linear import QuICWCLinear
from .math.cayley import cayley_from_params, skew_size
from .math.compound import compound_matrix, compound_matrix_reference
from .math.orthogonal import BlockLayout, QuICOrthogonal, select_columns


def _orthogonality(n: int, scale: float, dtype: torch.dtype, device: str) -> float:
    g = torch.Generator().manual_seed(0)
    theta = (scale * torch.randn(64, skew_size(n), generator=g, dtype=torch.float64)).to(device=device, dtype=dtype)
    a = cayley_from_params(theta, n)
    return float((a.transpose(-1, -2) @ a - torch.eye(n, dtype=dtype, device=device)).abs().max())


def _compound_error(n: int, order: int, dtype: torch.dtype) -> float:
    g = torch.Generator().manual_seed(1)
    base = cayley_from_params(0.7 * torch.randn(skew_size(n), generator=g, dtype=torch.float64), n)
    got = compound_matrix(base.to(dtype), order).double()
    return float((got - compound_matrix_reference(base, order)).abs().max())


def _identity_gradients(n: int, order: int) -> dict[str, float]:
    """Gradient norm of a fixed random linear functional of C_k(A(theta)) at theta = 0."""
    out = {}
    weights = None
    for method in ("explicit_polynomial", "torch_linalg_det"):
        theta = torch.zeros(skew_size(n), dtype=torch.float64, requires_grad=True)
        a = cayley_from_params(theta, n)
        if method == "explicit_polynomial":
            comp = compound_matrix(a, order)
        else:
            idx = torch.combinations(torch.arange(n), order)
            comp = torch.linalg.det(a[idx[:, None, :, None], idx[None, :, None, :]])
        if weights is None:
            weights = torch.randn(comp.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
        (comp * weights).sum().backward()
        out[method] = float(theta.grad.norm())
    # Central finite difference of the same functional as an independent reference.
    eps, fd = 1e-6, torch.zeros(skew_size(n), dtype=torch.float64)
    for i in range(skew_size(n)):
        step = torch.zeros(skew_size(n), dtype=torch.float64)
        step[i] = eps
        plus = (compound_matrix(cayley_from_params(step, n), order) * weights).sum()
        minus = (compound_matrix(cayley_from_params(-step, n), order) * weights).sum()
        fd[i] = (plus - minus) / (2 * eps)
    out["finite_difference"] = float(fd.norm())
    return out


def _layer_cost(device: str, stages: int, rank: int, repeats: int = 5) -> dict[str, float]:
    cfg = QuICWCConfig(in_features=1024, out_features=4096, rank=rank,
                       left=SideConfig(num_stages=stages, permutation="stride"),
                       right=SideConfig(num_stages=stages, permutation="stride"), sigma_mode="learnable")
    module = QuICWCLinear(cfg, torch.ones(rank, device=device), torch.zeros(4096, device=device))
    with torch.no_grad():
        module.left.theta.normal_(0, 0.5)
        module.right.theta.normal_(0, 0.5)
    u, v = module.factors()
    eye = torch.eye(rank, device=device)
    result = {"stages": stages, "rank": rank,
              "orthonormality_U_max_abs": float((u.T @ u - eye).abs().max()),
              "orthonormality_V_max_abs": float((v.T @ v - eye).abs().max()),
              "theta_parameters": module.left.theta.numel() + module.right.theta.numel()}
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for label, backward in (("reconstruct_forward_s", False), ("reconstruct_forward_backward_s", True)):
        times = []
        for _ in range(repeats + 1):
            sync()
            t0 = time.perf_counter()
            w = module.reconstruct_weight()
            if backward:
                w.square().sum().backward()
            sync()
            times.append(time.perf_counter() - t0)
        result[label] = sorted(times[1:])[len(times[1:]) // 2]
    return result


def run(device: str) -> dict:
    report = {"torch": torch.__version__, "python": platform.python_version(), "device": device,
              "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None}
    report["cayley_orthogonality_max_abs"] = [
        {"n": n, "theta_scale": s, "dtype": str(dt).replace("torch.", ""), "error": _orthogonality(n, s, dt, device)}
        for n in (4, 8, 16) for s in (0.5, 3.0) for dt in (torch.float32, torch.float64)]
    report["compound_vs_reference_max_abs"] = [
        {"n": 8, "order": k, "dtype": str(dt).replace("torch.", ""), "error": _compound_error(8, k, dt)}
        for k in (1, 2, 3) for dt in (torch.float32, torch.float64)]
    report["identity_gradient_norms"] = [{"n": n, "order": k, **_identity_gradients(n, k)} for n in (4, 8) for k in (2, 3)]
    lay = BlockLayout.build(36, 8, (1, 2))
    q = QuICOrthogonal(lay, 3, "stride", dtype=torch.float64)
    report["structure"] = {"layout_4096_n8_K12": {"blocks": BlockLayout.build(4096, 8, (1, 2)).num_blocks,
                                                  "remainder": BlockLayout.build(4096, 8, (1, 2)).remainder},
                           "layout_1024_n8_K12": {"blocks": BlockLayout.build(1024, 8, (1, 2)).num_blocks,
                                                  "remainder": BlockLayout.build(1024, 8, (1, 2)).remainder},
                           "example_parameters_36_dim_3_stages": q.parameter_count,
                           "selection_spread_first_8_of_4096": select_columns(BlockLayout.build(4096, 8, (1, 2)), 8).tolist()}
    report["layer_4096x1024_cost"] = [_layer_cost(device, s, r) for s, r in ((1, 1024), (4, 1024), (16, 1024), (64, 1024))]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = run(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
