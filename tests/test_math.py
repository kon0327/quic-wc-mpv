import itertools

import pytest
import torch

from quic_wc.math.cayley import cayley, cayley_from_params, inverse_cayley, params_from_skew, skew_from_params, skew_size
from quic_wc.math.compound import combinations, compound_dim, compound_matrix, compound_matrix_reference


def random_orthogonal(n, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    theta = 0.5 * torch.randn(skew_size(n), generator=g, dtype=dtype)
    return cayley_from_params(theta, n)


def test_skew_roundtrip_and_symmetry():
    theta = torch.randn(3, skew_size(5), dtype=torch.float64)
    skew = skew_from_params(theta, 5)
    assert torch.equal(skew, -skew.transpose(-1, -2))
    assert torch.equal(params_from_skew(skew), theta)
    with pytest.raises(ValueError):
        skew_from_params(theta[..., :-1], 5)


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_cayley_is_special_orthogonal(dtype, tol):
    theta = torch.randn(7, skew_size(8), dtype=dtype)
    a = cayley_from_params(theta, 8)
    eye = torch.eye(8, dtype=dtype)
    assert a.dtype == dtype
    assert (a.transpose(-1, -2) @ a - eye).abs().max() < tol
    assert (torch.linalg.det(a.double()) - 1).abs().max() < 1e3 * tol


def test_cayley_identity_convention_and_inverse():
    assert torch.equal(cayley_from_params(torch.zeros(skew_size(6), dtype=torch.float64), 6), torch.eye(6, dtype=torch.float64))
    theta = 0.3 * torch.randn(skew_size(6), dtype=torch.float64)
    skew = skew_from_params(theta, 6)
    eye = torch.eye(6, dtype=torch.float64)
    expected = (eye - skew) @ torch.linalg.inv(eye + skew)  # phase1.md formula
    a = cayley(skew)
    assert torch.allclose(a, expected, atol=1e-13)
    assert torch.allclose(params_from_skew(inverse_cayley(a)), theta, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cayley_rejects_half_precision(dtype):
    with pytest.raises(TypeError):
        cayley(torch.zeros(4, 4, dtype=dtype))


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_compound_matches_reference_for_all_supported_orders(n):
    base = torch.randn(n, n, dtype=torch.float64)  # general, not orthogonal
    for order in range(0, n + 1):
        if 4 <= order < n:
            with pytest.raises(ValueError):
                compound_matrix(base, order)
            got = compound_matrix(base, order, allow_generic_det=True)
        else:
            got = compound_matrix(base, order)
        ref = compound_matrix_reference(base, order)
        assert got.shape == ref.shape == (compound_dim(n, [order]),) * 2
        assert got.dtype == torch.float64
        assert torch.allclose(got, ref, atol=1e-12, rtol=0), (n, order)


def test_compound_order_bounds_and_device_dtype():
    base = torch.randn(2, 4, 4, dtype=torch.float32)
    one = compound_matrix(base, 0)
    assert one.shape == (2, 1, 1) and one.dtype == torch.float32 and one.device == base.device
    for bad in (-1, 5, 1.0):
        with pytest.raises(ValueError):
            compound_matrix(base, bad)
    with pytest.raises(ValueError):
        combinations(3, 4)


def test_combination_order_is_lexicographic():
    assert combinations(4, 2).tolist() == [list(c) for c in itertools.combinations(range(4), 2)]


@pytest.mark.parametrize("order", [1, 2, 3])
def test_cauchy_binet_transpose_and_orthogonality(order):
    a, b = random_orthogonal(6, 1), random_orthogonal(6, 2)
    m = torch.randn(6, 6, dtype=torch.float64)
    assert torch.allclose(compound_matrix(m @ a, order), compound_matrix(m, order) @ compound_matrix(a, order), atol=1e-11)
    assert torch.allclose(compound_matrix(m.T, order), compound_matrix(m, order).T, atol=1e-12)
    c = compound_matrix(a @ b, order)
    assert (c.T @ c - torch.eye(c.shape[0], dtype=torch.float64)).abs().max() < 1e-12


@pytest.mark.parametrize("order", [2, 3])
def test_explicit_compound_gradcheck_at_identity_and_random(order):
    # Identity has many singular minors; the explicit polynomials stay exactly differentiable there.
    for base in (torch.eye(5, dtype=torch.float64), random_orthogonal(5, 3)):
        base = base.clone().requires_grad_(True)
        assert torch.autograd.gradcheck(lambda x: compound_matrix(x, order), (base,), eps=1e-6, atol=1e-8)


def test_identity_gradient_of_second_compound_is_nonzero_through_cayley():
    theta = torch.zeros(skew_size(6), dtype=torch.float64, requires_grad=True)
    weights = torch.randn(compound_dim(6, [2]), compound_dim(6, [2]), dtype=torch.float64,
                          generator=torch.Generator().manual_seed(0))
    (compound_matrix(cayley_from_params(theta, 6), 2) * weights).sum().backward()
    assert theta.grad.abs().max() > 1e-3


def test_full_chain_gradcheck_cayley_compound():
    theta = (0.2 * torch.randn(2, skew_size(4), dtype=torch.float64)).requires_grad_(True)

    def chain(t):
        a = cayley_from_params(t, 4)
        return torch.cat([compound_matrix(a, 1).flatten(1), compound_matrix(a, 2).flatten(1),
                          compound_matrix(a, 3).flatten(1)], dim=1)

    assert torch.autograd.gradcheck(chain, (theta,), eps=1e-6, atol=1e-8)
    zero = torch.zeros(2, skew_size(4), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(chain, (zero,), eps=1e-6, atol=1e-8)
