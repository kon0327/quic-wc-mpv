"""Activation statistics and output-space metrics of a compressed linear layer.

For a fixed bias, the distillation MSE of ``Y = X W^T + b`` is an exact quadratic form of the
uncentred input second moment ``C = E[x x^T]``:
``E ||x (W_hat - W)^T||^2 = tr((W_hat - W) C (W_hat - W)^T)``. The full calibration set is
therefore represented by ``C`` (n x n) and ``mu = E[x]``; per-token metrics such as cosine
similarity still need real token samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SecondMoment:
    """Streaming FP64 accumulator of ``sum x x^T``, ``sum x`` and the token count."""

    dim: int
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        self.sum_outer = torch.zeros(self.dim, self.dim, dtype=torch.float64, device=self.device)
        self.sum = torch.zeros(self.dim, dtype=torch.float64, device=self.device)
        self.count = 0

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        x = x.reshape(-1, self.dim).to(device=self.sum.device, dtype=torch.float64)
        self.sum_outer.addmm_(x.transpose(0, 1), x)
        self.sum.add_(x.sum(0))
        self.count += x.shape[0]

    @property
    def second_moment(self) -> Tensor:
        return self.sum_outer / max(self.count, 1)

    @property
    def mean(self) -> Tensor:
        return self.sum / max(self.count, 1)


def quadratic_error(delta: Tensor, second_moment: Tensor) -> Tensor:
    """``tr(delta C delta^T)`` = mean over tokens of ``||x delta^T||^2``."""
    return ((delta @ second_moment) * delta).sum()


def output_energy(weight: Tensor, second_moment: Tensor, bias: Tensor | None = None,
                  mean: Tensor | None = None) -> Tensor:
    """``E ||W x + b||^2 = tr(W C W^T) + 2 b^T W mu + ||b||^2``."""
    energy = quadratic_error(weight, second_moment)
    if bias is not None:
        if mean is None:
            raise ValueError("mean is required with bias")
        energy = energy + 2 * bias @ (weight @ mean) + bias @ bias
    return energy


def relative_output_error(weight_hat: Tensor, weight: Tensor, second_moment: Tensor,
                          bias: Tensor | None = None, mean: Tensor | None = None,
                          bias_hat: Tensor | None = None) -> dict[str, float]:
    """Exact relative L2 output errors over a whole token set.

    The output error is ``e = (W_hat - W) x + (b_hat - b)`` (``b_hat = b`` unless given), so
    ``E||e||^2 = tr(D C D^T) + 2 db^T D mu + ||db||^2``. It is normalised by the teacher output
    energy without bias (``rel_l2_no_bias``) and with bias (``rel_l2_with_bias``).
    """
    dt = torch.float64
    c = second_moment.to(dt)
    delta = weight_hat.to(dt) - weight.to(dt)
    err = quadratic_error(delta, c)
    if bias_hat is not None:
        if bias is None or mean is None:
            raise ValueError("bias_hat requires the teacher bias and the token mean")
        db = bias_hat.to(dt) - bias.to(dt)
        err = err + 2 * db @ (delta @ mean.to(dt)) + db @ db
    out = {"rel_l2_no_bias": float(torch.sqrt(err.clamp_min(0) / quadratic_error(weight.to(dt), c)))}
    if bias is not None:
        total = output_energy(weight.to(dt), c, bias.to(dt), mean.to(dt))
        out["rel_l2_with_bias"] = float(torch.sqrt(err.clamp_min(0) / total))
    return out


@torch.no_grad()
def token_metrics(x: Tensor, weight_hat: Tensor, weight: Tensor, bias: Tensor | None = None,
                  bias_hat: Tensor | None = None, *, chunk: int = 8192, eps: float = 1e-12,
                  compute_dtype: torch.dtype = torch.float32) -> dict[str, float]:
    """Per-token cosine (mean and lower quantiles), relative L2 and MSE, with and without bias.

    "with bias" compares the real outputs ``x W_hat^T + b_hat`` and ``x W^T + b``; "no bias"
    subtracts the *teacher* bias from both, so a bias correction ``b_hat - b`` stays in the
    estimate and the metric remains the actual output error relative to the pre-bias output.
    """
    device = weight.device
    w = weight.to(device=device, dtype=compute_dtype)
    w_hat = weight_hat.to(device=device, dtype=compute_dtype)
    b = None if bias is None else bias.to(device=device, dtype=compute_dtype)
    b_hat = b if bias_hat is None else bias_hat.to(device=device, dtype=compute_dtype)
    shift = None if bias_hat is None else b_hat - b
    cos_b, cos_nb = [], []
    acc = {"err_b": 0.0, "ref_b": 0.0, "err_nb": 0.0, "ref_nb": 0.0}
    zero_ref = 0
    for start in range(0, x.shape[0], chunk):
        xc = x[start:start + chunk].to(device=device, dtype=compute_dtype)
        y_nb, yh_nb = xc @ w.T, xc @ w_hat.T
        if shift is not None:
            yh_nb = yh_nb + shift
        y_b = y_nb if b is None else y_nb + b
        yh_b = yh_nb if b is None else yh_nb + b
        for ref, est, store, key in ((y_b, yh_b, cos_b, "b"), (y_nb, yh_nb, cos_nb, "nb")):
            ref_norm, est_norm = ref.norm(dim=1), est.norm(dim=1)
            store.append(((ref * est).sum(1) / (ref_norm * est_norm).clamp_min(eps)).double().cpu())
            acc[f"err_{key}"] += float(((est - ref).double() ** 2).sum())
            acc[f"ref_{key}"] += float((ref.double() ** 2).sum())
        zero_ref += int((y_b.norm(dim=1) <= eps).sum())
    tokens, width = x.shape[0], weight.shape[0]
    out: dict[str, float] = {"tokens": tokens, "zero_norm_reference_tokens": zero_ref}
    for name, values in (("with_bias", torch.cat(cos_b)), ("no_bias", torch.cat(cos_nb))):
        q = torch.quantile(values, torch.tensor([0.01, 0.05, 0.5], dtype=torch.float64))
        out[f"cos_{name}_mean"] = float(values.mean())
        out[f"cos_{name}_p01"], out[f"cos_{name}_p05"], out[f"cos_{name}_p50"] = (float(v) for v in q)
        out[f"cos_{name}_min"] = float(values.min())
    out["rel_l2_with_bias"] = (acc["err_b"] / max(acc["ref_b"], eps)) ** 0.5
    out["rel_l2_no_bias"] = (acc["err_nb"] / max(acc["ref_nb"], eps)) ** 0.5
    out["mse_with_bias"] = acc["err_b"] / (tokens * width)
    return out
