"""Compound matrices ``[C_k(A)]_{I,J} = det(A[I, J])`` over ascending lexicographic k-subsets.

Orders 2 and 3 use explicit determinant polynomials. ``torch.linalg.det`` returns a zero or
otherwise wrong gradient for singular minors (PyTorch 2.2, observed at the identity base in
audit/2026-10-01-gpu-environment.md); most minors of the identity are singular, so the
explicit formulas are mandatory for training from identity. Higher orders fall back to
``torch.linalg.det`` only on explicit request. The dtype of the base is preserved (FP64 checks
are real FP64, unlike the local compound-PEFT, which casts minors to FP32).

Useful identities (tested): ``C_k(AB) = C_k(A) C_k(B)`` (Cauchy-Binet), ``C_k(A^T) = C_k(A)^T``,
so ``C_k`` of an orthogonal matrix is orthogonal.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Iterable

import torch
from torch import Tensor


@lru_cache(maxsize=None)
def _combination_tuple(n: int, order: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations(range(n), order))


def combinations(n: int, order: int, device: torch.device | str | None = None) -> Tensor:
    """``(C(n, k), k)`` long tensor of ascending lexicographic k-subsets of ``range(n)``."""
    if type(n) is not int or type(order) is not int or n < 1 or not 1 <= order <= n:
        raise ValueError(f"invalid combination request n={n}, order={order}")
    return torch.tensor(_combination_tuple(n, order), dtype=torch.long, device=device)


def compound_size(n: int, order: int) -> int:
    return math.comb(n, order)


def compound_dim(n: int, orders: Iterable[int]) -> int:
    """Size of the direct sum ``C_{k1}(A) (+) C_{k2}(A) (+) ...`` for an ``n x n`` base."""
    return sum(math.comb(n, k) for k in orders)


def _gather(base: Tensor, idx: Tensor, row: int, col: int) -> Tensor:
    # Entry (row, col) of every k x k minor, as a (..., C, C) matrix indexed by subset pairs.
    return base[..., idx[:, row, None], idx[None, :, col]]


def compound_matrix(base: Tensor, order: int, *, indices: Tensor | None = None,
                    allow_generic_det: bool = False) -> Tensor:
    """Return ``C_order(base)`` with the batch dimensions of ``base``.

    ``order == 0`` gives a ``1 x 1`` one on the base device and dtype. ``order == n`` equals
    ``det(base)``; for ``n > 3`` it uses ``torch.linalg.det``, whose gradient is reliable for the
    nonsingular (e.g. orthogonal) full matrix. Orders ``4 <= k < n`` require
    ``allow_generic_det=True`` because singular minors make their gradients unreliable.
    """
    if base.ndim < 2 or base.shape[-1] != base.shape[-2]:
        raise ValueError("base must contain square matrices")
    if not base.is_floating_point():
        raise TypeError("base must be floating point")
    n = base.shape[-1]
    if type(order) is not int or not 0 <= order <= n:
        raise ValueError(f"compound order must be an integer in [0, {n}], got {order!r}")
    if order == 0:
        return base.new_ones(*base.shape[:-2], 1, 1)
    if order == 1:
        return base
    if order > 3:
        if order == n:
            return torch.linalg.det(base)[..., None, None]
        if not allow_generic_det:
            raise ValueError(
                "orders 4..n-1 need torch.linalg.det, whose gradient is unreliable at singular minors; "
                "pass allow_generic_det=True for diagnostics only"
            )
        idx = combinations(n, order, base.device) if indices is None else indices
        minors = base[..., idx[:, None, :, None], idx[None, :, None, :]]
        return torch.linalg.det(minors)
    idx = combinations(n, order, base.device) if indices is None else indices
    if idx.shape != (math.comb(n, order), order):
        raise ValueError("indices do not match the requested compound order")
    a, b = _gather(base, idx, 0, 0), _gather(base, idx, 0, 1)
    d, e = _gather(base, idx, 1, 0), _gather(base, idx, 1, 1)
    if order == 2:
        return a * e - b * d
    c, f = _gather(base, idx, 0, 2), _gather(base, idx, 1, 2)
    g, h, i = _gather(base, idx, 2, 0), _gather(base, idx, 2, 1), _gather(base, idx, 2, 2)
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def compound_matrix_reference(base: Tensor, order: int) -> Tensor:
    """Independent loop over minors with FP64 determinants (tests and diagnostics only)."""
    if base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("reference expects a single square matrix")
    n = base.shape[0]
    if order == 0:
        return torch.ones(1, 1, dtype=torch.float64, device=base.device)
    subsets = list(itertools.combinations(range(n), order))
    out = torch.empty(len(subsets), len(subsets), dtype=torch.float64, device=base.device)
    source = base.detach().to(torch.float64)
    for r, rows in enumerate(subsets):
        for c, cols in enumerate(subsets):
            out[r, c] = torch.linalg.det(source[list(rows)][:, list(cols)])
    return out
