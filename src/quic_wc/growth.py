"""Progressive stage growth and random restarts for ``QuICWCLinear``.

``grow_stages`` appends identity stages (``theta = 0``) to the *end* of the stage product (they are
applied last). With the "conjugated" stage form a zero stage is exactly the identity, so the grown
layer reconstructs the same weight as before; its new parameters still get non-zero gradients
(explicit compound polynomials, see ``math.compound``). The chain form would change the transform
by a permutation, so growth refuses it.

Growing a fitted layer through ``L_1 < L_2 < ...`` stages yields a deployable layer at every size;
this is how the per-layer choice of ``L`` (adaptive ``L``) is made from one error-vs-bytes curve.
"""

from __future__ import annotations

import torch

from .layers.linear import QuICWCLinear


@torch.no_grad()
def grow_stages(module: QuICWCLinear, *, left: int | None = None, right: int | None = None) -> QuICWCLinear:
    """New layer with ``left``/``right`` total stages; existing stages and sigma/bias are copied."""
    cfg = module.config
    sides = {"left": (cfg.left, left, module.left), "right": (cfg.right, right, module.right)}
    updates = {}
    for name, (side_cfg, target, _) in sides.items():
        target = side_cfg.num_stages if target is None else target
        if target < side_cfg.num_stages:
            raise ValueError(f"{name}: cannot shrink from {side_cfg.num_stages} to {target} stages")
        if target > side_cfg.num_stages and side_cfg.stage_form != "conjugated":
            raise ValueError(f"{name}: growth needs stage_form='conjugated' (chain stages are not identity at theta=0)")
        updates[name] = type(side_cfg)(**{**side_cfg.to_dict(), "orders": side_cfg.orders, "num_stages": target})
    new_cfg = cfg.with_updates(**updates)
    grown = QuICWCLinear(new_cfg, module.sigma0.clone(), None if module.bias is None else module.bias.clone(),
                         inference_dtype=module.inference_dtype, device=module.sigma0.device,
                         strict_cache=module.strict_cache)
    for name, (side_cfg, _, old) in sides.items():
        getattr(grown, name).theta[: side_cfg.num_stages].copy_(old.theta)
    if module.sigma_gain is not None:
        grown.sigma_gain.copy_(module.sigma_gain)
    grown.left.checkpoint_segments = module.left.checkpoint_segments
    grown.right.checkpoint_segments = module.right.checkpoint_segments
    grown.train(module.training)
    return grown


@torch.no_grad()
def perturb_(module: QuICWCLinear, scale: float, generator: torch.Generator) -> QuICWCLinear:
    """Add ``N(0, scale^2)`` noise to all theta (restart initialisation); returns the module."""
    for theta in (module.left.theta, module.right.theta):
        noise = torch.randn(theta.shape, generator=generator, dtype=theta.dtype)
        theta.add_(scale * noise.to(theta.device))
    return module
