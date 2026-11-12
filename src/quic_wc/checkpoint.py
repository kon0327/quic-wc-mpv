"""Compact export/load of one QuIC-WC layer and exact storage accounting.

The artifact holds only ``theta_out``, ``theta_in``, ``sigma`` (gain folded in) and ``bias`` in
``layer.safetensors`` plus ``quic_wc.json`` (config, conventions, tensor SHA256, metadata).
Selections and permutations are regenerated deterministically from the config. No original
``W``, ``U`` or ``V`` is stored; the dense cache is never part of the artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import CAYLEY_CONVENTION, FORMAT_VERSION, QuICWCConfig
from .layers.linear import QuICWCLinear

TENSOR_FILE = "layer.safetensors"
MANIFEST_FILE = "quic_wc.json"
BYTES = {torch.float64: 8, torch.float32: 4, torch.bfloat16: 2, torch.float16: 2}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def save_compact(module: QuICWCLinear, directory: str | Path, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from safetensors.torch import save_file

    tensors = module.export_tensors()
    if any(not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("refusing to export non-finite tensors")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if (target / TENSOR_FILE).exists() or (target / MANIFEST_FILE).exists():
        raise FileExistsError(f"artifact already exists in {target}")
    save_file(tensors, str(target / TENSOR_FILE))
    payload = {
        "format_version": FORMAT_VERSION,
        "config": module.config.to_dict(),
        "conventions": {"cayley": CAYLEY_CONVENTION, "compound_index_order": "ascending_lexicographic",
                        "weight": "W_hat = (Q_out E_out) diag(sigma) (Q_in E_in)^T; Y = X W_hat^T + b"},
        "bias": module.bias is not None,
        "inference_dtype": str(module.inference_dtype).replace("torch.", "") if module.inference_dtype else None,
        "tensor_sha256": hashlib.sha256((target / TENSOR_FILE).read_bytes()).hexdigest(),
        "metadata": dict(metadata or {}),
    }
    (target / MANIFEST_FILE).write_bytes(_canonical(payload) + b"\n")
    return payload


def load_compact(directory: str | Path, device: torch.device | str = "cpu") -> QuICWCLinear:
    from safetensors.torch import load_file

    source = Path(directory)
    payload = json.loads((source / MANIFEST_FILE).read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported QuIC-WC artifact version")
    if hashlib.sha256((source / TENSOR_FILE).read_bytes()).hexdigest() != payload["tensor_sha256"]:
        raise ValueError("tensor SHA256 mismatch")
    config = QuICWCConfig.from_dict(payload["config"])
    tensors = load_file(str(source / TENSOR_FILE), device=str(device))
    dtype_name = payload.get("inference_dtype")
    module = QuICWCLinear(config, tensors["sigma"], tensors.get("bias"),
                          inference_dtype=getattr(torch, dtype_name) if dtype_name else None, device=device)
    module.load_exported(tensors)
    return module


def storage_report(config: QuICWCConfig, *, bias: bool = True, bias_dtype: torch.dtype = torch.bfloat16,
                   dense_dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    """Scalar counts and tensor bytes of the deployable state versus the dense layer."""
    counts = config.parameter_counts(bias)
    cbytes = BYTES[config.dtype]
    tensor_bytes = ((counts["theta_out"] + counts["theta_in"] + counts["sigma"]) * cbytes
                    + counts["bias"] * BYTES[bias_dtype])
    dense_bytes = config.out_features * config.in_features * BYTES[dense_dtype] + (
        config.out_features * BYTES[bias_dtype] if bias else 0)
    return {
        "parameters": counts,
        "dense_parameters": config.dense_parameter_count(bias),
        "tensor_bytes": tensor_bytes,
        "dense_tensor_bytes": dense_bytes,
        "parameter_ratio": counts["total"] / config.dense_parameter_count(bias),
        "byte_ratio": tensor_bytes / dense_bytes,
    }


def artifact_bytes(directory: str | Path) -> dict[str, int]:
    source = Path(directory)
    sizes = {name: (source / name).stat().st_size for name in (TENSOR_FILE, MANIFEST_FILE)}
    sizes["total"] = sum(sizes.values())
    return sizes
