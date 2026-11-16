"""Structured orthogonal transforms ``Q = S_L ... S_1`` built from compound blocks.

One stage is ``S = B P``: a fixed row permutation ``P`` followed by a block-diagonal ``B``.
Every block of ``B`` is ``C_{k1}(A_t) (+) C_{k2}(A_t) (+) ... (+) I`` for its own Cayley base
``A_t in SO(n)``; coordinates not covered by whole blocks form an identity remainder. The
transform is exactly ``dim x dim`` and orthogonal by construction, so selected columns form an
orthonormal thin factor (no row truncation of a larger matrix).

Consequence of Cauchy-Binet: stages with the *same* block grouping multiply to a single stage
(``C_k(A) C_k(B) = C_k(AB)``). Extra stages therefore add capacity only together with a
permutation that changes the grouping ("cyclic", "stride", "random").

Stage forms. "chain" (format v1 default): ``Q = B_L P_L ... B_2 P_2 B_1`` (``P_1 = I``).
"conjugated": ``Q = (R_L^T B_L R_L) ... (R_1^T B_1 R_1)`` with cumulative permutations
``R_l = P_l ... P_1``. Since ``B_L P_L ... B_1 = R_L (R_L^T B_L R_L) ... (R_1^T B_1 R_1)``, both
forms span the same family up to the fixed row permutation ``R_L``. Only the conjugated form has
``Q = I`` at ``theta = 0`` for every ``L``, so stages can be appended (applied last) without
changing the transform: this is what progressive stage growth needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .cayley import SUPPORTED_DTYPES, cayley, skew_from_params, skew_size
from .compound import combinations, compound_dim, compound_matrix

PERMUTATIONS = ("identity", "cyclic", "stride", "random")
STAGE_FORMS = ("chain", "conjugated")
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


def stage_permutations(kind: str, dim: int, num_stages: int, *, block_size: int, seed: int = 0,
                       form: str = "chain") -> Tensor:
    """Gather indices per stage: ``P_l`` ("chain") or the cumulative ``R_l = P_l ... P_1`` ("conjugated").

    ``(R x)[j] = x[r[j]]`` with ``r_l = r_{l-1}[p_l]``. Stage ``l`` depends only on its index, so a
    longer transform extends a shorter one with the same stages.
    """
    if form not in STAGE_FORMS:
        raise ValueError(f"unknown stage form {form!r}; choose from {STAGE_FORMS}")
    perms = [make_permutation(kind, dim, i, block_size=block_size, seed=seed) for i in range(num_stages)]
    if form == "conjugated":
        for i in range(1, num_stages):
            perms[i] = perms[i - 1][perms[i]]
    return torch.stack(perms)


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
                 *, stage_form: str = "chain", dtype: torch.dtype = torch.float32,
                 device: torch.device | str | None = None) -> None:
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
        self.stage_form = stage_form
        self.construction_dtype = dtype
        # Runtime memory option (not part of the config or state): recompute stage activations in
        # ``checkpoint_segments`` segments during backward instead of storing all of them.
        self.checkpoint_segments = 0
        self.theta = nn.Parameter(torch.zeros(num_stages, layout.num_blocks, layout.params_per_block,
                                              dtype=dtype, device=device))
        # Both forms run the chain stages ``B_l P_l``; the conjugated form then applies ``R_L^T`` once
        # (``Q_conj = R_L^T Q_chain``), which is cheaper than conjugating every stage.
        perms = stage_permutations(permutation, layout.dim, num_stages, block_size=layout.block_size, seed=seed)
        self.register_buffer("perms", perms if device is None else perms.to(device), persistent=False)
        self._identity_perm = [bool(torch.equal(p, torch.arange(layout.dim))) for p in perms]
        final = None
        if stage_form == "conjugated":
            cumulative = stage_permutations(permutation, layout.dim, num_stages, block_size=layout.block_size,
                                            seed=seed, form="conjugated")[-1]
            if not torch.equal(cumulative, torch.arange(layout.dim)):
                final = torch.argsort(cumulative)
        self.register_buffer("final_perm", final if final is None or device is None else final.to(device),
                             persistent=False)
        for k in layout.orders:
            if k > 1:
                self.register_buffer(f"indices_{k}", combinations(layout.base_size, k, device), persistent=False)

    def extra_repr(self) -> str:
        lay = self.layout
        return (f"dim={lay.dim}, base={lay.base_size}, orders={lay.orders}, block={lay.block_size}, "
                f"blocks={lay.num_blocks}, remainder={lay.remainder}, stages={self.num_stages}, "
                f"permutation={self.permutation}, form={self.stage_form}")

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

    def stage_blocks(self) -> Tensor:
        """Full block matrices ``[L, blocks, block_size, block_size]`` = ``C_k1 (+) C_k2 (+) ... (+) I``."""
        mats = self.stage_matrices()
        lay = self.layout
        if len(mats) == 1 and lay.compound_dim == lay.block_size:
            return mats[0]
        with torch.autocast(device_type=self.theta.device.type, enabled=False):
            out = mats[0].new_zeros(self.num_stages, lay.num_blocks, lay.block_size, lay.block_size)
            start = 0
            for size, matrix in zip(lay.compound_sizes, mats):
                out[..., start:start + size, start:start + size] = matrix
                start += size
            if start < lay.block_size:
                out[..., start:, start:] = torch.eye(lay.block_size - start, dtype=out.dtype, device=out.device)
        return out

    def _block_stage(self, x: Tensor, blocks: Tensor) -> Tensor:
        lay = self.layout
        covered = lay.num_blocks * lay.block_size
        head = x[:covered].reshape(lay.num_blocks, lay.block_size, x.shape[1])
        rotated = torch.bmm(blocks, head).reshape(covered, x.shape[1])
        return torch.cat([rotated, x[covered:]], dim=0) if lay.remainder else rotated

    def apply(self, x: Tensor) -> Tensor:
        """Return ``Q @ x`` for ``x`` of shape ``(dim, cols)`` in the construction dtype."""
        if x.ndim != 2 or x.shape[0] != self.layout.dim:
            raise ValueError(f"expected a ({self.layout.dim}, cols) operand")
        if x.dtype != self.construction_dtype:
            raise TypeError("operand must use the construction dtype")
        blocks = self.stage_blocks()
        segments = self.checkpoint_segments
        if segments and self.num_stages > 1 and torch.is_grad_enabled() and self.theta.requires_grad:
            size = math.ceil(self.num_stages / min(segments, self.num_stages))
            for lo in range(0, self.num_stages, size):
                x = checkpoint(self._run_stages, x, lo, min(self.num_stages, lo + size), blocks, use_reentrant=False)
        else:
            x = self._run_stages(x, 0, self.num_stages, blocks)
        return x if self.final_perm is None else x.index_select(0, self.final_perm)

    def _run_stages(self, x: Tensor, lo: int, hi: int, blocks: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            for stage in range(lo, hi):
                if not self._identity_perm[stage]:
                    x = x.index_select(0, self.perms[stage])
                x = self._block_stage(x, blocks[stage])
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
