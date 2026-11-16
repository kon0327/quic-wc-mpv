"""``QuICWCLinear``: a linear layer whose weight is reconstructed from QuIC thin factors.

Training mode always differentiates through the reconstruction. Eval mode uses a dense weight
cache built once without gradients and then calls only ``F.linear``. The cache is a
non-persistent buffer (never serialised) and is invalidated by ``.to()``/dtype changes,
``train()``, ``load_state_dict()`` and by any in-place change of the stored state
(tensor version counters), so it can never silently correspond to other parameters.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import QuICWCConfig
from ..math.orthogonal import QuICOrthogonal, select_columns


class StaleCacheError(RuntimeError):
    pass


class QuICWCLinear(nn.Module):
    def __init__(self, config: QuICWCConfig, sigma: Tensor, bias: Tensor | None = None, *,
                 inference_dtype: torch.dtype | None = None, device: torch.device | str | None = None,
                 strict_cache: bool = False) -> None:
        super().__init__()
        self.config = config
        self.in_features, self.out_features, self.rank = config.in_features, config.out_features, config.rank
        dtype = config.dtype
        if sigma.shape != (config.rank,):
            raise ValueError(f"sigma must have shape ({config.rank},)")
        if bias is not None and bias.shape != (config.out_features,):
            raise ValueError(f"bias must have shape ({config.out_features},)")
        device = sigma.device if device is None else torch.device(device)
        lc, rc = config.left, config.right
        self.left = QuICOrthogonal(lc.layout(self.out_features), lc.num_stages, lc.permutation, lc.seed,
                                   stage_form=lc.stage_form, dtype=dtype, device=device)
        self.right = QuICOrthogonal(rc.layout(self.in_features), rc.num_stages, rc.permutation, rc.seed,
                                    stage_form=rc.stage_form, dtype=dtype, device=device)
        self.register_buffer("selection_out", select_columns(self.left.layout, self.rank, config.selection).to(device),
                             persistent=False)
        self.register_buffer("selection_in", select_columns(self.right.layout, self.rank, config.selection).to(device),
                             persistent=False)
        self.register_buffer("sigma0", sigma.detach().to(device=device, dtype=dtype).clone())
        if config.sigma_mode == "learnable":
            self.sigma_gain = nn.Parameter(torch.ones(self.rank, dtype=dtype, device=device))
        else:
            self.sigma_gain = None
        self.register_buffer("bias", None if bias is None else bias.detach().to(device).clone())
        self.register_buffer("weight_cache", None, persistent=False)
        self.inference_dtype = inference_dtype
        self.strict_cache = strict_cache
        self._cached_bias: Tensor | None = None
        self._cache_key: tuple | None = None
        self.decompress_count = 0

    # ----- construction helpers -------------------------------------------------------------
    @classmethod
    def from_linear(cls, linear: nn.Linear, config: QuICWCConfig, *, sigma: Tensor | None = None,
                    inference_dtype: torch.dtype | None = None, **kwargs: Any) -> "QuICWCLinear":
        """Initialise sigma from the top singular values of ``linear.weight`` (FP32 SVD)."""
        if (config.out_features, config.in_features) != tuple(linear.weight.shape):
            raise ValueError("config dimensions do not match the linear layer")
        weight = linear.weight.detach()
        if sigma is None:
            sigma = torch.linalg.svdvals(weight.to(torch.float32))[: config.rank]
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(config, sigma, bias, inference_dtype=inference_dtype or weight.dtype,
                   device=weight.device, **kwargs)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, "
                f"sigma_mode={self.config.sigma_mode}, bias={self.bias is not None}")

    def set_checkpointing(self, segments: int) -> "QuICWCLinear":
        """Recompute stage activations in ``segments`` segments per side during backward (0 = off)."""
        if type(segments) is not int or segments < 0:
            raise ValueError("segments must be a non-negative integer")
        self.left.checkpoint_segments = self.right.checkpoint_segments = segments
        return self

    # ----- reconstruction ---------------------------------------------------------------------
    def sigma(self) -> Tensor:
        return self.sigma0 if self.sigma_gain is None else self.sigma0 * self.sigma_gain

    def factors(self) -> tuple[Tensor, Tensor]:
        """Orthonormal thin factors ``U_hat (m x r)`` and ``V_hat (n x r)`` in construction dtype."""
        return self.left.thin(self.selection_out), self.right.thin(self.selection_in)

    def reconstruct_weight(self) -> Tensor:
        """Differentiable ``W_hat = U_hat diag(sigma) V_hat^T`` in construction dtype."""
        u_hat, v_hat = self.factors()
        return (u_hat * self.sigma()) @ v_hat.transpose(0, 1)

    # ----- cache --------------------------------------------------------------------------------
    def _state_tensors(self) -> list[Tensor]:
        tensors = [self.left.theta, self.right.theta, self.sigma0]
        if self.sigma_gain is not None:
            tensors.append(self.sigma_gain)
        if self.bias is not None:
            tensors.append(self.bias)
        return tensors

    def _state_key(self, dtype: torch.dtype) -> tuple:
        return (dtype,) + tuple((t.device, t.dtype, t.data_ptr(), t._version) for t in self._state_tensors())

    def invalidate_cache(self) -> None:
        self.weight_cache = None
        self._cached_bias = None
        self._cache_key = None

    @property
    def cache_valid(self) -> bool:
        return (self.weight_cache is not None and self._cache_key is not None
                and self._cache_key == self._state_key(self.weight_cache.dtype))

    @torch.no_grad()
    def decompress(self, dtype: torch.dtype | None = None) -> Tensor:
        """Build the dense inference weight once (FP32/FP64 math, then cast and make contiguous)."""
        dtype = dtype or self.inference_dtype or (self.bias.dtype if self.bias is not None else self.config.dtype)
        weight = self.reconstruct_weight().to(dtype).contiguous()
        self.weight_cache = weight
        self._cached_bias = None if self.bias is None else self.bias.to(dtype)
        self._cache_key = self._state_key(dtype)
        self.decompress_count += 1
        return weight

    def _apply(self, fn: Any, *args: Any, **kwargs: Any) -> "QuICWCLinear":
        super()._apply(fn, *args, **kwargs)
        dtype = self.config.dtype
        if self.sigma0.dtype != dtype:
            self.sigma0 = self.sigma0.to(dtype)
        if self.sigma_gain is not None and self.sigma_gain.dtype != dtype:
            self.sigma_gain.data = self.sigma_gain.data.to(dtype)
        self.invalidate_cache()
        return self

    def train(self, mode: bool = True) -> "QuICWCLinear":
        super().train(mode)
        if mode:
            self.invalidate_cache()
        return self

    def _load_from_state_dict(self, *args: Any, **kwargs: Any) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self.invalidate_cache()

    # ----- forward --------------------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected {self.in_features} input features, got {x.shape[-1]}")
        if self.training:
            weight = self.reconstruct_weight().to(x.dtype)
            bias = None if self.bias is None else self.bias.to(x.dtype)
            return F.linear(x, weight, bias)
        if not (self.cache_valid and self.weight_cache.dtype == x.dtype):
            if self.strict_cache:
                raise StaleCacheError("weight cache is missing or stale; call decompress()/prepare_for_inference()")
            self.decompress(x.dtype)
        return F.linear(x, self.weight_cache, self._cached_bias)

    # ----- export ---------------------------------------------------------------------------------
    def export_tensors(self) -> dict[str, Tensor]:
        """Minimal deployable state; the learnable sigma gain is folded into sigma."""
        tensors = {
            "theta_out": self.left.theta.detach(),
            "theta_in": self.right.theta.detach(),
            "sigma": self.sigma().detach(),
        }
        if self.bias is not None:
            tensors["bias"] = self.bias.detach()
        return {k: v.cpu().contiguous() for k, v in tensors.items()}

    @torch.no_grad()
    def load_exported(self, tensors: dict[str, Tensor]) -> None:
        expected = {"theta_out", "theta_in", "sigma"} | ({"bias"} if self.bias is not None else set())
        if set(tensors) != expected:
            raise ValueError(f"exported tensors {sorted(tensors)} do not match expected {sorted(expected)}")
        for name, target in (("theta_out", self.left.theta), ("theta_in", self.right.theta), ("sigma", self.sigma0)):
            if tensors[name].shape != target.shape or tensors[name].dtype != target.dtype:
                raise ValueError(f"tensor {name} has shape/dtype {tuple(tensors[name].shape)}/{tensors[name].dtype}, "
                                 f"expected {tuple(target.shape)}/{target.dtype}")
            target.copy_(tensors[name])
        if self.sigma_gain is not None:
            self.sigma_gain.fill_(1.0)
        if self.bias is not None:
            if tensors["bias"].shape != self.bias.shape:
                raise ValueError("bias shape mismatch")
            self.bias.copy_(tensors["bias"])
        self.invalidate_cache()


def prepare_for_inference(model: nn.Module, dtype: torch.dtype | None = None) -> list[str]:
    """Switch QuIC-WC layers to eval and decompress each exactly once on its current device."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, QuICWCLinear):
            module.eval()
            module.decompress(dtype)
            names.append(name)
    return names


def quic_wc_layers(model: nn.Module) -> Iterable[tuple[str, QuICWCLinear]]:
    return ((n, m) for n, m in model.named_modules() if isinstance(m, QuICWCLinear))
