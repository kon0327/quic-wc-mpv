"""Compact export/load of one QuIC-WC layer and exact storage accounting.

The artifact holds only ``theta_out``, ``theta_in``, ``sigma`` (gain folded in) and ``bias`` in
``layer.safetensors`` plus ``quic_wc.json`` (config, conventions, tensor SHA256, metadata).
Selections and permutations are regenerated deterministically from the config. No original
``W``, ``U`` or ``V`` is stored; the dense cache is never part of the artifact.

Storage dtype (format version 2): ``theta`` and ``sigma`` may be stored in a lower precision than
the construction dtype (``storage_dtype``, e.g. BF16). Loading upcasts them exactly to the
construction dtype; the orthogonal factors are always rebuilt in FP32/FP64 from the rounded
parameters, so they stay exactly orthogonal up to construction round-off. The bias keeps its own
dtype. Version-1 artifacts (theta/sigma in construction dtype, no ``storage_dtype``) still load.
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
STORAGE_DTYPES = {"float64": torch.float64, "float32": torch.float32, "bfloat16": torch.bfloat16,
                  "float16": torch.float16}
READABLE_VERSIONS = (1, 2)
QUANTIZED = ("theta_out", "theta_in", "sigma")


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def resolve_storage_dtype(config: QuICWCConfig, storage_dtype: torch.dtype | str | None) -> torch.dtype:
    if storage_dtype is None:
        return config.dtype
    dtype = STORAGE_DTYPES.get(storage_dtype) if isinstance(storage_dtype, str) else storage_dtype
    if dtype not in STORAGE_DTYPES.values():
        raise ValueError(f"storage_dtype must be one of {sorted(STORAGE_DTYPES)}")
    if BYTES[dtype] > BYTES[config.dtype]:
        raise ValueError("storage_dtype cannot be wider than the construction dtype")
    return dtype


def storage_tensors(module: QuICWCLinear, storage_dtype: torch.dtype | str | None = None) -> dict[str, torch.Tensor]:
    """Exported tensors with theta/sigma rounded to the storage dtype (bias unchanged)."""
    dtype = resolve_storage_dtype(module.config, storage_dtype)
    tensors = module.export_tensors()
    return {k: (v.to(dtype).contiguous() if k in QUANTIZED else v) for k, v in tensors.items()}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def save_compact(module: QuICWCLinear, directory: str | Path, metadata: Mapping[str, Any] | None = None, *,
                 storage_dtype: torch.dtype | str | None = None) -> dict[str, Any]:
    from safetensors.torch import save_file

    dtype = resolve_storage_dtype(module.config, storage_dtype)
    tensors = storage_tensors(module, dtype)
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
        "storage_dtype": dtype_name(dtype),
        "inference_dtype": dtype_name(module.inference_dtype) if module.inference_dtype else None,
        "tensor_sha256": hashlib.sha256((target / TENSOR_FILE).read_bytes()).hexdigest(),
        "metadata": dict(metadata or {}),
    }
    (target / MANIFEST_FILE).write_bytes(_canonical(payload) + b"\n")
    return payload


def load_compact(directory: str | Path, device: torch.device | str = "cpu") -> QuICWCLinear:
    from safetensors.torch import load_file

    source = Path(directory)
    payload = json.loads((source / MANIFEST_FILE).read_text(encoding="utf-8"))
    if payload.get("format_version") not in READABLE_VERSIONS:
        raise ValueError("unsupported QuIC-WC artifact version")
    if hashlib.sha256((source / TENSOR_FILE).read_bytes()).hexdigest() != payload["tensor_sha256"]:
        raise ValueError("tensor SHA256 mismatch")
    config = QuICWCConfig.from_dict(payload["config"])
    tensors = load_file(str(source / TENSOR_FILE), device=str(device))
    stored = STORAGE_DTYPES[payload.get("storage_dtype", config.construction_dtype)]
    for name in QUANTIZED:
        if tensors[name].dtype != stored:
            raise ValueError(f"tensor {name} is {tensors[name].dtype}, manifest says {stored}")
        tensors[name] = tensors[name].to(config.dtype)  # exact upcast
    dtype_name = payload.get("inference_dtype")
    module = QuICWCLinear(config, tensors["sigma"], tensors.get("bias"),
                          inference_dtype=getattr(torch, dtype_name) if dtype_name else None, device=device)
    module.load_exported(tensors)
    return module


def storage_report(config: QuICWCConfig, *, bias: bool = True, bias_dtype: torch.dtype = torch.bfloat16,
                   dense_dtype: torch.dtype = torch.bfloat16,
                   storage_dtype: torch.dtype | str | None = None) -> dict[str, Any]:
    """Scalar counts and tensor bytes of the deployable state versus the dense layer."""
    counts = config.parameter_counts(bias)
    dtype = resolve_storage_dtype(config, storage_dtype)
    cbytes = BYTES[dtype]
    tensor_bytes = ((counts["theta_out"] + counts["theta_in"] + counts["sigma"]) * cbytes
                    + counts["bias"] * BYTES[bias_dtype])
    dense_bytes = config.out_features * config.in_features * BYTES[dense_dtype] + (
        config.out_features * BYTES[bias_dtype] if bias else 0)
    return {
        "storage_dtype": dtype_name(dtype),
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
