"""Structured orthogonal transforms ``Q = S_L ... S_1`` built from compound blocks.

One stage is ``S = B P``: a fixed row permutation ``P`` followed by a block-diagonal ``B``.
Every block of ``B`` is ``C_{k1}(A_t) (+) C_{k2}(A_t) (+) ... (+) I`` for its own Cayley base
``A_t in SO(n)``; coordinates not covered by whole blocks form an identity remainder. The
transform is exactly ``dim x dim`` and orthogonal by construction, so selected columns form an
orthonormal thin factor (no row truncation of a larger matrix).

Consequence of Cauchy-Binet: stages with the *same* block grouping multiply to a single stage
(``C_k(A) C_k(B) = C_k(AB)``). Extra stages therefore add capacity only together with a
permutation that changes the grouping ("cyclic", "stride", "random").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .cayley import SUPPORTED_DTYPES, cayley, skew_from_params, skew_size
from .compound import combinations, compound_dim, compound_matrix

PERMUTATIONS = ("identity", "cyclic", "stride", "random")
SELECTIONS = ("spread", "leading")
MAX_ORDER = 3


def validate_orders(orders: Sequence[int]) -> tuple[int, ...]:
    orders = tuple(orders)
    if not orders or any(type(k) is not int or not 1 <= k <= MAX_ORDER for k in orders):
        raise ValueError(f"orders must be a nonempty selection of integers 1..{MAX_ORDER}")
    if tuple(sorted(set(orders))) != orders:
        raise ValueError("orders must be unique and ascending")
    return orders


@dataclass(frozen=True)
class BlockLayout:
    """Block structure of one stage of a ``dim x dim`` transform."""

    dim: int
    base_size: int
    orders: tuple[int, ...]
    block_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", validate_orders(self.orders))
        if type(self.base_size) is not int or self.base_size < max(2, max(self.orders)):
            raise ValueError("base_size must be >= 2 and >= the largest order")
        if type(self.dim) is not int or type(self.block_size) is not int:
            raise ValueError("dim and block_size must be integers")
        if self.block_size < self.compound_dim:
            raise ValueError(f"block_size={self.block_size} < compound dimension {self.compound_dim}")
        if self.block_size > self.dim:
            raise ValueError(f"block_size={self.block_size} exceeds dim={self.dim}")

    @classmethod
    def build(cls, dim: int, base_size: int, orders: Sequence[int], block_size: int | None = None) -> "BlockLayout":
        orders = validate_orders(orders)
        return cls(dim, base_size, orders, compound_dim(base_size, orders) if block_size is None else block_size)

    @property
    def compound_sizes(self) -> tuple[int, ...]:
        return tuple(math.comb(self.base_size, k) for k in self.orders)

    @property
    def compound_dim(self) -> int:
        return compound_dim(self.base_size, self.orders)

    @property
    def num_blocks(self) -> int:
        return self.dim // self.block_size

    @property
    def remainder(self) -> int:
        return self.dim - self.num_blocks * self.block_size

    @property
    def params_per_block(self) -> int:
        return skew_size(self.base_size)

    @property
    def params_per_stage(self) -> int:
        return self.num_blocks * self.params_per_block

    @property
    def active_coordinates(self) -> int:
        """Coordinates rotated by compound blocks in one stage (the rest is identity)."""
        return self.num_blocks * self.compound_dim


def make_permutation(kind: str, dim: int, stage: int, *, block_size: int, seed: int = 0) -> Tensor:
    """Row permutation applied before block stage ``stage`` (stage 0 is always the identity).

    ``(P x)[j] = x[perm[j]]``. "cyclic" rotates by half a block per stage (brick-wall pattern,
    local mixing); "stride" interleaves coordinates so each new block draws from different
    previous blocks (Monarch-style transpose, global mixing); "random" is a seeded permutation.
    Permutations are regenerated from (kind, dim, stage, block_size, seed); they are not stored.
    """
    if kind not in PERMUTATIONS:
        raise ValueError(f"unknown permutation {kind!r}; choose from {PERMUTATIONS}")
    if stage == 0 or kind == "identity":
        return torch.arange(dim)
    if kind == "cyclic":
        return torch.roll(torch.arange(dim), -max(1, block_size // 2))
    if kind == "stride":
        index = torch.arange(dim)
        key = (index % block_size) * dim + index // block_size
        return torch.argsort(key)
    generator = torch.Generator().manual_seed(int(seed) * 1_000_003 + stage)
    return torch.randperm(dim, generator=generator)


def select_columns(layout: BlockLayout, rank: int, rule: str = "spread") -> Tensor:
    """Indices of the ``rank`` identity columns forming ``E`` in the thin factor ``Q E``.

    "spread" walks positions inside blocks (compound coordinates first, C1 before C2) round-robin
    across blocks, then the identity remainder; "leading" takes the first ``rank`` columns.
    """
    if rule not in SELECTIONS:
        raise ValueError(f"unknown selection rule {rule!r}; choose from {SELECTIONS}")
    if type(rank) is not int or not 1 <= rank <= layout.dim:
        raise ValueError(f"rank must be in [1, {layout.dim}]")
    if rule == "leading":
        return torch.arange(rank)
    blocks, size = layout.num_blocks, layout.block_size
    order = (torch.arange(size)[:, None] + size * torch.arange(blocks)[None, :]).reshape(-1)
    order = torch.cat([order, torch.arange(blocks * size, layout.dim)])
    return order[:rank].clone()


class QuICOrthogonal(nn.Module):
    """Learnable ``Q = S_L ... S_1`` with minimal skew parameters ``theta[L, blocks, n(n-1)/2]``."""

    def __init__(self, layout: BlockLayout, num_stages: int = 1, permutation: str = "identity", seed: int = 0,
                 *, dtype: torch.dtype = torch.float32, device: torch.device | str | None = None) -> None:
        super().__init__()
        if type(num_stages) is not int or num_stages < 1:
            raise ValueError("num_stages must be a positive integer")
        if dtype not in SUPPORTED_DTYPES:
            raise TypeError("construction dtype must be float32 or float64")
        if layout.num_blocks < 1:
            raise ValueError("layout has no compound block")
        self.layout = layout
        self.num_stages = num_stages
        self.permutation = permutation
        self.seed = seed
        self.construction_dtype = dtype
        self.theta = nn.Parameter(torch.zeros(num_stages, layout.num_blocks, layout.params_per_block,
                                              dtype=dtype, device=device))
        perms = torch.stack([make_permutation(permutation, layout.dim, i, block_size=layout.block_size, seed=seed)
                             for i in range(num_stages)])
        self.register_buffer("perms", perms if device is None else perms.to(device), persistent=False)
        self._identity_perm = [bool(torch.equal(p, torch.arange(layout.dim))) for p in perms]
        for k in layout.orders:
            if k > 1:
                self.register_buffer(f"indices_{k}", combinations(layout.base_size, k, device), persistent=False)

    def extra_repr(self) -> str:
        lay = self.layout
        return (f"dim={lay.dim}, base={lay.base_size}, orders={lay.orders}, block={lay.block_size}, "
                f"blocks={lay.num_blocks}, remainder={lay.remainder}, stages={self.num_stages}, "
                f"permutation={self.permutation}")

    def _apply(self, fn: Any, *args: Any, **kwargs: Any) -> "QuICOrthogonal":
        super()._apply(fn, *args, **kwargs)
        # A host-wide .to(bfloat16) must not lower the construction precision of theta.
        if self.theta.dtype != self.construction_dtype:
            self.theta.data = self.theta.data.to(self.construction_dtype)
            if self.theta.grad is not None:
                self.theta.grad.data = self.theta.grad.data.to(self.construction_dtype)
        return self

    def bases(self) -> Tensor:
        """Cayley bases ``A[L, blocks, n, n]``."""
        return cayley(skew_from_params(self.theta, self.layout.base_size))

    def stage_matrices(self) -> list[Tensor]:
        """Compound matrices per order, each ``[L, blocks, C(n,k), C(n,k)]``."""
        with torch.autocast(device_type=self.theta.device.type, enabled=False):
            base = self.bases()
            return [compound_matrix(base, k, indices=getattr(self, f"indices_{k}", None)) for k in self.layout.orders]

    def _block_stage(self, x: Tensor, mats: list[Tensor]) -> Tensor:
        lay = self.layout
        cols = x.shape[1]
        covered = lay.num_blocks * lay.block_size
        head = x[:covered].reshape(lay.num_blocks, lay.block_size, cols)
        pieces, start = [], 0
        for size, matrix in zip(lay.compound_sizes, mats):
            pieces.append(torch.bmm(matrix, head[:, start:start + size]))
            start += size
        if start < lay.block_size:
            pieces.append(head[:, start:])
        rotated = torch.cat(pieces, dim=1).reshape(covered, cols)
        return torch.cat([rotated, x[covered:]], dim=0) if lay.remainder else rotated

    def apply(self, x: Tensor) -> Tensor:
        """Return ``Q @ x`` for ``x`` of shape ``(dim, cols)`` in the construction dtype."""
        if x.ndim != 2 or x.shape[0] != self.layout.dim:
            raise ValueError(f"expected a ({self.layout.dim}, cols) operand")
        if x.dtype != self.construction_dtype:
            raise TypeError("operand must use the construction dtype")
        mats = self.stage_matrices()
        with torch.autocast(device_type=x.device.type, enabled=False):
            for stage in range(self.num_stages):
                if not self._identity_perm[stage]:
                    x = x.index_select(0, self.perms[stage])
                x = self._block_stage(x, [m[stage] for m in mats])
        return x

    def thin(self, selection: Tensor) -> Tensor:
        """Orthonormal thin factor ``Q E = Q[:, selection]``."""
        eye = torch.eye(self.layout.dim, dtype=self.construction_dtype, device=self.theta.device)
        return self.apply(eye.index_select(1, selection.to(self.theta.device)))

    def matrix(self) -> Tensor:
        """Dense ``Q`` (tests and small diagnostics)."""
        return self.apply(torch.eye(self.layout.dim, dtype=self.construction_dtype, device=self.theta.device))

    @property
    def parameter_count(self) -> int:
        return self.theta.numel()
