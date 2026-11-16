import pytest
import torch

from quic_wc.math.cayley import cayley_from_params, inverse_cayley, params_from_skew
from quic_wc.math.compound import compound_matrix
from quic_wc.math.orthogonal import PERMUTATIONS, BlockLayout, QuICOrthogonal, make_permutation, select_columns


def randomize(module, scale=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.theta.copy_(scale * torch.randn(module.theta.shape, generator=g, dtype=module.theta.dtype))
    return module


def dense_reference(module):
    """Independent dense construction with block_diag and explicit permutation matrices."""
    lay = module.layout
    dt = module.theta.dtype
    q = torch.eye(lay.dim, dtype=dt)
    for stage in range(module.num_stages):
        perm = module.perms[stage]
        p = torch.zeros(lay.dim, lay.dim, dtype=dt)
        p[torch.arange(lay.dim), perm] = 1.0
        blocks = []
        for t in range(lay.num_blocks):
            a = cayley_from_params(module.theta[stage, t].detach(), lay.base_size)
            parts = [compound_matrix(a, k) for k in lay.orders]
            pad = lay.block_size - lay.compound_dim
            if pad:
                parts.append(torch.eye(pad, dtype=dt))
            blocks.append(torch.block_diag(*parts))
        if lay.remainder:
            blocks.append(torch.eye(lay.remainder, dtype=dt))
        q = torch.block_diag(*blocks) @ p @ q
    return q


CASES = [
    dict(dim=36, base=8, orders=(1, 2), block=None, stages=1, perm="identity"),
    dict(dim=50, base=4, orders=(1, 2), block=12, stages=3, perm="stride"),
    dict(dim=47, base=4, orders=(1, 2, 3), block=None, stages=2, perm="cyclic"),
    dict(dim=40, base=5, orders=(2,), block=None, stages=4, perm="random"),
    dict(dim=33, base=6, orders=(1,), block=None, stages=3, perm="stride"),
]


@pytest.mark.parametrize("case", CASES)
def test_apply_matches_dense_reference_and_is_orthogonal(case):
    lay = BlockLayout.build(case["dim"], case["base"], case["orders"], case["block"])
    module = randomize(QuICOrthogonal(lay, case["stages"], case["perm"], seed=3, dtype=torch.float64))
    q = module.matrix()
    assert torch.allclose(q, dense_reference(module), atol=1e-12)
    assert (q.T @ q - torch.eye(lay.dim, dtype=torch.float64)).abs().max() < 1e-12
    sel = select_columns(lay, min(17, lay.dim))
    thin = module.thin(sel)
    assert torch.allclose(thin, q[:, sel], atol=1e-13)
    if case["perm"] != "identity":  # conjugated form: Q_conj = R_L^T Q_chain with cumulative R_L
        conj = QuICOrthogonal(lay, case["stages"], case["perm"], seed=3, stage_form="conjugated", dtype=torch.float64)
        with torch.no_grad():
            conj.theta.copy_(module.theta)
        cumulative = torch.arange(lay.dim)
        for perm in module.perms:
            cumulative = cumulative[perm]
        r = torch.zeros(lay.dim, lay.dim, dtype=torch.float64)
        r[torch.arange(lay.dim), cumulative] = 1.0
        assert torch.allclose(conj.matrix(), r.T @ dense_reference(module), atol=1e-12)


def test_float32_orthonormality_threshold():
    lay = BlockLayout.build(4096, 8, (1, 2))
    module = randomize(QuICOrthogonal(lay, 4, "stride"), scale=0.5)
    thin = module.thin(select_columns(lay, 512))
    err = (thin.T @ thin - torch.eye(512)).abs().max()
    assert err < 1e-5, float(err)


def test_identity_permutation_stages_collapse_to_one_stage():
    # Cauchy-Binet: C_k(A2) C_k(A1) = C_k(A2 A1) blockwise, so same-grouping stages add nothing.
    lay = BlockLayout.build(36, 8, (1, 2))
    two = randomize(QuICOrthogonal(lay, 2, "identity", dtype=torch.float64), scale=0.2)
    one = QuICOrthogonal(lay, 1, "identity", dtype=torch.float64)
    a = cayley_from_params(two.theta.detach(), 8)  # [2, 1, 8, 8]
    with torch.no_grad():
        one.theta[0] = params_from_skew(inverse_cayley(a[1] @ a[0]))
    assert torch.allclose(one.matrix(), two.matrix(), atol=1e-11)


@pytest.mark.parametrize("kind", PERMUTATIONS)
def test_permutations_are_valid_and_deterministic(kind):
    for stage in range(3):
        perm = make_permutation(kind, 50, stage, block_size=12, seed=7)
        assert torch.equal(torch.sort(perm).values, torch.arange(50))
        assert torch.equal(perm, make_permutation(kind, 50, stage, block_size=12, seed=7))
    assert torch.equal(make_permutation(kind, 50, 0, block_size=12), torch.arange(50))


def test_stride_permutation_spreads_blocks():
    perm = make_permutation("stride", 48, 1, block_size=12)
    first_new_block = perm[:12]
    assert len(set((first_new_block // 12).tolist())) == 4  # draws from all old blocks


def test_selection_rules():
    lay = BlockLayout.build(80, 4, (1, 2), block_size=12)  # 6 blocks, compound dim 10, remainder 8
    sel = select_columns(lay, 30)
    assert len(set(sel.tolist())) == 30 and sel.max() < 80
    assert all((i % 12) < 10 for i in sel.tolist())  # compound coordinates come first
    assert torch.equal(select_columns(lay, 80).sort().values, torch.arange(80))
    assert torch.equal(select_columns(lay, 5, "leading"), torch.arange(5))
    with pytest.raises(ValueError):
        select_columns(lay, 81)


def test_layout_validation():
    with pytest.raises(ValueError):
        BlockLayout.build(100, 8, (1, 2), block_size=30)  # compound dim 36 > 30
    with pytest.raises(ValueError):
        BlockLayout.build(20, 8, (1, 2))  # block larger than dim
    with pytest.raises(ValueError):
        BlockLayout.build(100, 8, (2, 1))
    lay = BlockLayout.build(4096, 8, (1, 2))
    assert (lay.num_blocks, lay.remainder, lay.params_per_stage) == (113, 28, 113 * 28)


def test_theta_keeps_construction_dtype_under_host_cast():
    module = QuICOrthogonal(BlockLayout.build(36, 8, (1, 2)), 2, "stride")
    module.to(torch.bfloat16)
    assert module.theta.dtype == torch.float32


def test_nonzero_gradient_from_identity_for_weighted_loss():
    lay = BlockLayout.build(72, 8, (1, 2))
    module = QuICOrthogonal(lay, 2, "stride", dtype=torch.float64)
    weights = torch.randn(72, 72, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    (module.matrix() * weights).sum().backward()
    per_block = module.theta.grad.abs().amax(dim=-1)
    assert torch.isfinite(module.theta.grad).all()
    assert (per_block > 1e-6).all()


def test_gradcheck_structured_transform():
    lay = BlockLayout.build(14, 4, (1, 2), block_size=None)  # block 10, remainder 4
    module = randomize(QuICOrthogonal(lay, 2, "stride", dtype=torch.float64), scale=0.3)
    sel = select_columns(lay, 6)
    theta = module.theta.detach().clone().requires_grad_(True)

    def wrapped(t):  # route a plain tensor through the module's own construction
        saved = module.theta
        try:
            del module.theta
            module.theta = t
            return module.thin(sel)
        finally:
            del module.theta
            module.theta = saved

    assert torch.autograd.gradcheck(wrapped, (theta,), eps=1e-6, atol=1e-8)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu():
    lay = BlockLayout.build(300, 8, (1, 2))
    cpu = randomize(QuICOrthogonal(lay, 3, "random", seed=2))
    gpu = randomize(QuICOrthogonal(lay, 3, "random", seed=2)).cuda()
    sel = select_columns(lay, 100)
    assert torch.allclose(cpu.thin(sel), gpu.thin(sel).cpu(), atol=2e-5)
