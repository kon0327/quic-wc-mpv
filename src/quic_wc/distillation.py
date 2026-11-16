"""Activation distillation of a ``QuICWCLinear`` towards a dense teacher layer.

The objective is the exact full-calibration-set output MSE (fixed shared bias), normalised by
the teacher output energy: ``tr(dW C dW^T) / tr(W C W^T)`` with ``dW = W_hat - W``. This equals
the minibatch MSE of instructions/phase3.md in expectation, without minibatch noise and without
iterating over stored activations. AdamW, warmup + cosine schedule, global gradient clipping;
a non-finite loss or gradient stops the run before the optimizer step (no corrupted export).

``restore_best`` keeps the state with the lowest selection objective seen at evaluation steps:
the validation objective (``select_on="val"``, v1 default) or the train objective
(``select_on="train"``), which keeps validation data out of every model choice.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor

from .layers.linear import QuICWCLinear
from .metrics import quadratic_error


@dataclass(frozen=True)
class FitConfig:
    steps: int = 1000
    lr: float = 1e-2
    sigma_lr: float | None = None
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    max_grad_norm: float = 1.0
    eval_every: int = 100
    restore_best: bool = True
    select_on: str = "val"

    def __post_init__(self) -> None:
        if self.select_on not in ("val", "train"):
            raise ValueError("select_on must be 'val' or 'train'")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["betas"] = list(self.betas)
        return out


@dataclass
class FitResult:
    status: str
    steps_completed: int
    best_step: int
    best_val_rel_mse: float
    final_train_rel_mse: float
    seconds: float
    peak_memory_bytes: int | None
    history: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lr_lambda(cfg: FitConfig):
    def schedule(step: int) -> float:
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        span = max(1, cfg.steps - cfg.warmup_steps)
        progress = min(1.0, (step - cfg.warmup_steps) / span)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return schedule


def fit(module: QuICWCLinear, weight: Tensor, c_train: Tensor, *, c_val: Tensor | None = None,
        cfg: FitConfig = FitConfig(), log_every: int | None = None) -> FitResult:
    device, dtype = module.left.theta.device, module.config.dtype
    w = weight.to(device=device, dtype=dtype)
    c_tr = c_train.to(device=device, dtype=dtype)
    c_va = None if c_val is None else c_val.to(device=device, dtype=dtype)
    with torch.no_grad():
        denom_tr = quadratic_error(w, c_tr)
        denom_va = None if c_va is None else quadratic_error(w, c_va)
    groups = [{"params": [module.left.theta, module.right.theta], "lr": cfg.lr}]
    if module.sigma_gain is not None:
        groups.append({"params": [module.sigma_gain], "lr": cfg.sigma_lr or cfg.lr})
    params = [p for g in groups for p in g["params"]]
    optimizer = torch.optim.AdamW(groups, betas=cfg.betas, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(cfg))
    module.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    history: list[dict[str, float]] = []
    best = (math.inf, 0, copy.deepcopy(module.state_dict()))
    status, loss_value, step = "completed", math.nan, 0

    def evaluate() -> tuple[float, float]:
        """(train, val) relative objectives of the current state; val falls back to train."""
        with torch.no_grad():
            delta = module.reconstruct_weight() - w
            train = float(quadratic_error(delta, c_tr) / denom_tr)
            return train, (train if c_va is None else float(quadratic_error(delta, c_va) / denom_va))

    def selection(train: float, val: float) -> float:
        return val if cfg.select_on == "val" else train

    initial_train, initial = evaluate()
    history.append({"step": 0, "train_rel_mse": initial_train, "val_rel_mse": initial, "lr": cfg.lr,
                    "grad_norm": float("nan"), "seconds": 0.0})
    best = (selection(initial_train, initial), 0, copy.deepcopy(module.state_dict()))
    best_val = initial
    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = quadratic_error(module.reconstruct_weight() - w, c_tr) / denom_tr
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            status = "nonfinite_loss"
            break
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        if not math.isfinite(grad_norm):
            status = "nonfinite_gradient"
            break
        lr_now = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            train_now, val = evaluate()  # after the step (loss_value is the pre-step train objective)
            history.append({"step": step, "train_rel_mse": loss_value, "val_rel_mse": val, "lr": lr_now,
                            "grad_norm": grad_norm, "seconds": time.perf_counter() - start,
                            "train_rel_mse_after": train_now})
            if not (math.isfinite(val) and math.isfinite(train_now)):
                status = "nonfinite_validation"
                break
            if selection(train_now, val) < best[0]:
                best = (selection(train_now, val), step, copy.deepcopy(module.state_dict()))
                best_val = val
            if log_every and step % log_every == 0:
                print(f"step {step:5d} train {loss_value:.5f} val {val:.5f} |g| {grad_norm:.3g} lr {lr_now:.2e}",
                      flush=True)
    else:
        step = cfg.steps
    completed = step if status == "completed" else step - 1
    if cfg.restore_best:
        module.load_state_dict(best[2])
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return FitResult(status, completed, best[1], best_val, loss_value, time.perf_counter() - start, peak, history)
