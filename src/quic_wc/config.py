"""Configuration and exact parameter accounting of a QuIC-WC linear layer.

``W in R^{m x n}`` (out x in), ``Y = X W^T + b``. The compressed weight is
``W_hat = U_hat diag(sigma) V_hat^T`` with ``U_hat = Q_out E_out`` (m x r) and
``V_hat = Q_in E_in`` (n x r); ``Q_out``/``Q_in`` are structured QuIC transforms (left = output
side, right = input side). ``r`` is the SVD-like rank; it is unrelated to the compound-PEFT ``r``
(number of blocks).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

import torch

from .math.orthogonal import PERMUTATIONS, SELECTIONS, BlockLayout, validate_orders

CONSTRUCTION_DTYPES = {"float32": torch.float32, "float64": torch.float64}
SIGMA_MODES = ("fixed", "learnable")
CAYLEY_CONVENTION = "A=(I-S)(I+S)^-1, S strictly-upper minimal coordinates"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SideConfig:
    base_size: int = 8
    orders: tuple[int, ...] = (1, 2)
    block_size: int | None = None
    num_stages: int = 1
    permutation: str = "identity"
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "orders", validate_orders(self.orders))
        if self.permutation not in PERMUTATIONS:
            raise ValueError(f"permutation must be one of {PERMUTATIONS}")
        if type(self.num_stages) is not int or self.num_stages < 1:
            raise ValueError("num_stages must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

    def layout(self, dim: int) -> BlockLayout:
        return BlockLayout.build(dim, self.base_size, self.orders, self.block_size)

    def parameter_count(self, dim: int) -> int:
        return self.num_stages * self.layout(dim).params_per_stage

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["orders"] = list(self.orders)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SideConfig":
        data = dict(data)
        data["orders"] = tuple(data.get("orders", (1, 2)))
        return cls(**data)


@dataclass(frozen=True)
class QuICWCConfig:
    in_features: int
    out_features: int
    rank: int
    left: SideConfig = field(default_factory=SideConfig)
    right: SideConfig = field(default_factory=SideConfig)
    selection: str = "spread"
    sigma_mode: str = "fixed"
    construction_dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("in_features", "out_features", "rank"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.rank > min(self.in_features, self.out_features):
            raise ValueError("rank cannot exceed min(in_features, out_features)")
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        if self.sigma_mode not in SIGMA_MODES:
            raise ValueError(f"sigma_mode must be one of {SIGMA_MODES}")
        if self.construction_dtype not in CONSTRUCTION_DTYPES:
            raise ValueError("construction_dtype must be float32 or float64")
        # Validate both layouts eagerly so invalid shapes fail before any allocation.
        self.left.layout(self.out_features)
        self.right.layout(self.in_features)

    @property
    def dtype(self) -> torch.dtype:
        return CONSTRUCTION_DTYPES[self.construction_dtype]

    def parameter_counts(self, bias: bool = True) -> dict[str, int]:
        """Stored scalars of the deployable state (sigma gain folded into sigma on export)."""
        counts = {
            "theta_out": self.left.parameter_count(self.out_features),
            "theta_in": self.right.parameter_count(self.in_features),
            "sigma": self.rank,
            "bias": self.out_features if bias else 0,
        }
        counts["total"] = sum(counts.values())
        return counts

    def dense_parameter_count(self, bias: bool = True) -> int:
        return self.out_features * self.in_features + (self.out_features if bias else 0)

    def with_updates(self, **changes: Any) -> "QuICWCConfig":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features, "out_features": self.out_features, "rank": self.rank,
            "left": self.left.to_dict(), "right": self.right.to_dict(), "selection": self.selection,
            "sigma_mode": self.sigma_mode, "construction_dtype": self.construction_dtype,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuICWCConfig":
        data = dict(data)
        data["left"] = SideConfig.from_dict(data.get("left", {}))
        data["right"] = SideConfig.from_dict(data.get("right", {}))
        return cls(**data)
