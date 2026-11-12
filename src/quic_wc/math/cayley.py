"""Cayley parametrisation of SO(n) by minimal skew-symmetric coordinates.

Convention (instructions/phase1.md): ``A = (I - S)(I + S)^{-1}`` with ``S = -S^T``.
``I - S`` and ``(I + S)^{-1}`` commute, so ``A = solve(I + S, I - S)``; no explicit inverse
is formed. ``S`` is stored by its ``n(n-1)/2`` strictly upper-triangular entries, which is the
concept's ``S = P - P^T`` with an upper-triangular ``P``. Zero parameters give ``A = I``.
The image is SO(n) without the matrices that have eigenvalue -1.
"""

from __future__ import annotations

import torch
from torch import Tensor

SUPPORTED_DTYPES = (torch.float32, torch.float64)


def skew_size(n: int) -> int:
    """Number of free coordinates of an ``n x n`` skew-symmetric matrix."""
    if type(n) is not int or n < 1:
        raise ValueError("matrix size must be a positive integer")
    return n * (n - 1) // 2


def check_construction_dtype(tensor: Tensor) -> None:
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise TypeError(
            f"QuIC construction requires float32 or float64, got {tensor.dtype}; "
            "Cayley, compound minors and SVD must not run in half precision"
        )


def skew_from_params(theta: Tensor, n: int) -> Tensor:
    """Map ``(..., n(n-1)/2)`` upper-triangular coordinates to skew matrices ``(..., n, n)``."""
    if theta.shape[-1] != skew_size(n):
        raise ValueError(f"expected {skew_size(n)} skew coordinates for n={n}, got {theta.shape[-1]}")
    rows, cols = torch.triu_indices(n, n, 1, device=theta.device)
    skew = theta.new_zeros(*theta.shape[:-1], n, n)
    skew[..., rows, cols] = theta
    skew[..., cols, rows] = -theta
    return skew


def params_from_skew(skew: Tensor) -> Tensor:
    """Inverse of :func:`skew_from_params` for an exactly skew-symmetric input."""
    n = skew.shape[-1]
    rows, cols = torch.triu_indices(n, n, 1, device=skew.device)
    return skew[..., rows, cols]


def cayley(skew: Tensor) -> Tensor:
    """Return ``(I - S)(I + S)^{-1}`` for a batch of skew-symmetric matrices."""
    check_construction_dtype(skew)
    if skew.ndim < 2 or skew.shape[-1] != skew.shape[-2]:
        raise ValueError("skew must contain square matrices")
    eye = torch.eye(skew.shape[-1], dtype=skew.dtype, device=skew.device)
    with torch.autocast(device_type=skew.device.type, enabled=False):
        return torch.linalg.solve(eye + skew, eye - skew)


def cayley_from_params(theta: Tensor, n: int) -> Tensor:
    return cayley(skew_from_params(theta, n))


def inverse_cayley(orthogonal: Tensor) -> Tensor:
    """Skew ``S = (I + A)^{-1}(I - A)`` with ``cayley(S) = A``; requires ``A`` without eigenvalue -1."""
    check_construction_dtype(orthogonal)
    eye = torch.eye(orthogonal.shape[-1], dtype=orthogonal.dtype, device=orthogonal.device)
    with torch.autocast(device_type=orthogonal.device.type, enabled=False):
        skew = torch.linalg.solve(eye + orthogonal, eye - orthogonal)
    return 0.5 * (skew - skew.transpose(-1, -2))
