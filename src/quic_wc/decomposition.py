"""SVD baselines with explicit rank policies.

Convention: ``W = U diag(s) V^T`` with ``U (m x k)``, ``V (n x k)`` (``V``, never ``Vh``). Thin
factors have orthonormal columns; truncation keeps the leading ``r`` singular triplets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SVDResult:
    U: Tensor
    S: Tensor
    V: Tensor

    def truncated(self, rank: int) -> Tensor:
        if not 0 <= rank <= self.S.numel():
            raise ValueError("rank out of range")
        return (self.U[:, :rank] * self.S[:rank]) @ self.V[:, :rank].transpose(0, 1)


def thin_svd(weight: Tensor, dtype: torch.dtype = torch.float32) -> SVDResult:
    """Thin SVD in FP32 (default) or FP64; half precision is rejected by design."""
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("SVD must run in float32 or float64")
    u, s, vh = torch.linalg.svd(weight.to(dtype), full_matrices=False)
    return SVDResult(u, s, vh.transpose(0, 1))


def select_rank(singular_values: Tensor, *, rank: int | None = None, energy_ratio: float | None = None,
                relative_threshold: float | None = None) -> int:
    """Exactly one policy: fixed ``rank``; smallest rank whose squared singular values reach
    ``energy_ratio`` of the total; or count of values ``>= relative_threshold * s_max``."""
    given = [p is not None for p in (rank, energy_ratio, relative_threshold)]
    if sum(given) != 1:
        raise ValueError("specify exactly one of rank, energy_ratio, relative_threshold")
    s = singular_values.detach().to(torch.float64)
    if rank is not None:
        if type(rank) is not int or not 1 <= rank <= s.numel():
            raise ValueError("rank out of range")
        return rank
    if energy_ratio is not None:
        if not 0 < energy_ratio <= 1:
            raise ValueError("energy_ratio must be in (0, 1]")
        energy = torch.cumsum(s**2, 0) / torch.sum(s**2)
        return int(torch.searchsorted(energy, torch.tensor(energy_ratio, dtype=torch.float64) - 1e-12).item()) + 1
    if not 0 < relative_threshold <= 1:
        raise ValueError("relative_threshold must be in (0, 1]")
    return int((s >= relative_threshold * s[0]).sum().item())


def low_rank_parameter_count(out_features: int, in_features: int, rank: int, bias: bool = True) -> int:
    """Stored scalars of a rank-r factorisation ``(U diag(s)) V^T`` plus bias."""
    return (out_features + in_features) * rank + (out_features if bias else 0)


def psd_power(matrix: Tensor, power: float, *, rel_floor: float = 1e-8) -> Tensor:
    """Symmetric power of a PSD matrix with eigenvalues clamped at ``rel_floor * max``."""
    evals, evecs = torch.linalg.eigh(matrix)
    floor = evals.max().clamp_min(0) * rel_floor
    evals = evals.clamp_min(floor)
    return (evecs * evals.pow(power)) @ evecs.transpose(0, 1)


def activation_aware_truncation(weight: Tensor, second_moment: Tensor, rank: int, *,
                                rel_floor: float = 1e-8) -> Tensor:
    """Rank-r ``W_r`` minimising ``tr((W_r - W) C (W_r - W)^T)`` (whitened SVD).

    With ``C = R^2`` (R the PSD square root), the optimum is ``[W R]_r R^{-1}``. Eigenvalues of C
    below ``rel_floor * max`` are clamped, so for a singular C the result is approximately optimal.
    Storage equals an ordinary rank-r factorisation: ``(U_M s_M)`` and ``R^{-1} V_M``.
    """
    svd = activation_aware_svd(weight, second_moment, rel_floor=rel_floor)
    return svd.truncated(rank).to(weight.dtype if weight.is_floating_point() else torch.float32)


def activation_aware_svd(weight: Tensor, second_moment: Tensor, *, rel_floor: float = 1e-8) -> SVDResult:
    """Factors with ``truncated(r) == [W R]_r R^{-1}`` for every r (FP64).

    ``U``/``S`` come from the SVD of ``W R``; ``V = R^{-1} V_M`` is *not* orthonormal.
    """
    dtype = torch.float64
    root = psd_power(second_moment.to(dtype), 0.5, rel_floor=rel_floor)
    inv_root = psd_power(second_moment.to(dtype), -0.5, rel_floor=rel_floor)
    svd = thin_svd(weight.to(dtype) @ root, dtype=dtype)
    return SVDResult(svd.U, svd.S, inv_root @ svd.V)
