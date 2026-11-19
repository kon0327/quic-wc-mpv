import torch

from quic_wc.pilot.sequential import CrossMoment, output_target, paired_sample


def test_output_target_matches_asymmetric_objective():
    """With the corrected bias, ‖Ŵx_c + b̂ − (Wx_r + b)‖² = ‖Ŵ − W_t‖²_{S_cc} + const for every Ŵ."""
    g = torch.Generator().manual_seed(0)
    n, d_in, d_out = 4000, 6, 4
    x_r = torch.randn(n, d_in, generator=g, dtype=torch.float64) * torch.tensor([3, 2, 1, 1, .5, .2]) + 0.7
    x_c = x_r @ (torch.eye(d_in) + 0.3 * torch.randn(d_in, d_in, generator=g)).double() + 0.1 * torch.randn(
        n, d_in, generator=g, dtype=torch.float64) - 0.2
    w, b = torch.randn(d_out, d_in, generator=g, dtype=torch.float64), torch.randn(d_out, generator=g, dtype=torch.float64)
    cross = CrossMoment(d_in, "cpu")
    cross.update(x_r, x_c)
    c_cc, mu_c, mu_r = x_c.T @ x_c / n, x_c.mean(0), x_r.mean(0)
    w_t, b_t = output_target(w, b, c_cc, mu_c, cross.moment, mu_r, ridge=0.0)
    s_cc = c_cc - torch.outer(mu_c, mu_c)
    y_ref = x_r @ w.T + b

    def true_err(w_hat):
        b_hat = b_t + (w_t - w_hat) @ mu_c  # quic_wc.pilot.run corrected bias with teacher (W_t, b_t)
        return float(((x_c @ w_hat.T + b_hat - y_ref) ** 2).sum(1).mean())

    def quad(w_hat):
        d = w_hat - w_t
        return float(torch.trace(d @ s_cc @ d.T))

    consts = [true_err(wh) - quad(wh) for wh in (w, w_t, torch.randn(d_out, d_in, generator=g, dtype=torch.float64))]
    assert max(consts) - min(consts) < 1e-9
    # W_t is the least-squares optimum: no other weight does better on the true objective
    assert true_err(w_t) <= true_err(w) + 1e-12


def test_output_target_identity_when_inputs_equal():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(500, 5, generator=g, dtype=torch.float64) + 1
    w, b = torch.randn(3, 5, generator=g, dtype=torch.float64), torch.randn(3, generator=g, dtype=torch.float64)
    cross = CrossMoment(5, "cpu")
    cross.update(x, x)
    w_t, b_t = output_target(w, b, x.T @ x / 500, x.mean(0), cross.moment, x.mean(0), ridge=0.0)
    assert torch.allclose(w_t, w, atol=1e-10) and torch.allclose(b_t, b, atol=1e-10)


def test_paired_sample_sorted_and_deterministic():
    a, b = paired_sample(100, 10, 3), paired_sample(100, 10, 3)
    assert torch.equal(a, b) and torch.equal(a, a.sort().values) and len(a) == 10
    assert torch.equal(paired_sample(5, 10, 3), torch.arange(5))


def test_ridge_keeps_original_weight_in_unconstrained_directions():
    """Inputs confined to a subspace: with a ridge the correction lives in that subspace only, W_t = W elsewhere."""
    g = torch.Generator().manual_seed(2)
    n = 2000
    basis = torch.zeros(3, 5, dtype=torch.float64)
    basis[0, 0] = basis[1, 1] = basis[2, 2] = 1.0  # dims 3, 4 never excited
    x_r = torch.randn(n, 3, generator=g, dtype=torch.float64) @ basis
    x_c = (torch.randn(n, 3, generator=g, dtype=torch.float64) * 0.1 + x_r[:, :3] * 1.2) @ basis
    w, b = torch.randn(4, 5, generator=g, dtype=torch.float64), torch.randn(4, generator=g, dtype=torch.float64)
    cross = CrossMoment(5, "cpu")
    cross.update(x_r, x_c)
    w_t, _ = output_target(w, b, x_c.T @ x_c / n, x_c.mean(0), cross.moment, x_r.mean(0), ridge=1e-6)
    assert torch.allclose(w_t[:, 3:], w[:, 3:], atol=1e-12)
    assert not torch.allclose(w_t[:, :3], w[:, :3], atol=1e-3)
