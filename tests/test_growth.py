import pytest
import torch

from quic_wc import QuICWCConfig, QuICWCLinear, SideConfig
from quic_wc.growth import grow_stages, perturb_
from quic_wc.math.orthogonal import BlockLayout, QuICOrthogonal, stage_permutations


@pytest.mark.parametrize("kind", ["stride", "cyclic", "random"])
def test_conjugated_form_equals_chain_up_to_fixed_row_permutation(kind):
    torch.manual_seed(0)
    layout = BlockLayout.build(50, 4, (1, 2))
    chain = QuICOrthogonal(layout, 5, kind, seed=3, dtype=torch.float64)
    conj = QuICOrthogonal(layout, 5, kind, seed=3, stage_form="conjugated", dtype=torch.float64)
    with torch.no_grad():
        chain.theta.normal_(0, 0.5)
        conj.theta.copy_(chain.theta)
    r_last = stage_permutations(kind, 50, 5, block_size=layout.block_size, seed=3, form="conjugated")[-1]
    assert torch.allclose(chain.matrix(), conj.matrix()[r_last], atol=1e-12)
    q = conj.matrix()
    assert torch.allclose(q.T @ q, torch.eye(50, dtype=torch.float64), atol=1e-12)


def test_conjugated_zero_theta_is_identity_for_any_depth():
    layout = BlockLayout.build(70, 8, (1, 2))
    for stages in (1, 3, 7):
        q = QuICOrthogonal(layout, stages, "stride", stage_form="conjugated", dtype=torch.float64).matrix()
        assert torch.equal(q, torch.eye(70, dtype=torch.float64))


def make(stages_left, stages_right, form="conjugated"):
    cfg = QuICWCConfig(in_features=40, out_features=72, rank=16, sigma_mode="learnable",
                       left=SideConfig(base_size=4, num_stages=stages_left, permutation="stride", stage_form=form),
                       right=SideConfig(base_size=4, num_stages=stages_right, permutation="random", seed=2,
                                        stage_form=form))
    return QuICWCLinear(cfg, torch.rand(16) + 0.5, torch.randn(72))


def test_growth_preserves_weight_and_new_stages_get_gradients():
    module = make(2, 3)
    perturb_(module, 0.3, torch.Generator().manual_seed(1))
    with torch.no_grad():
        module.sigma_gain.uniform_(0.8, 1.2)
    grown = grow_stages(module, left=5, right=4)
    assert grown.config.left.num_stages == 5 and grown.config.right.num_stages == 4
    assert torch.allclose(grown.reconstruct_weight(), module.reconstruct_weight(), atol=1e-6)
    assert torch.equal(grown.left.theta[:2], module.left.theta) and not grown.left.theta[2:].any()
    target = torch.randn(72, 40)
    ((grown.reconstruct_weight() - target) ** 2).sum().backward()
    assert grown.left.theta.grad[2:].abs().sum() > 0 and grown.right.theta.grad[3:].abs().sum() > 0


def test_growth_rejects_chain_form_and_shrinking():
    with pytest.raises(ValueError):
        grow_stages(make(2, 2, form="chain"), left=3)
    with pytest.raises(ValueError):
        grow_stages(make(3, 3), left=2)
    same = grow_stages(make(2, 2, form="chain"))  # no growth requested -> allowed
    assert same.config.left.num_stages == 2


def test_checkpointing_gives_identical_gradients_and_survives_growth():
    module = make(5, 3)
    perturb_(module, 0.2, torch.Generator().manual_seed(4))
    target = torch.randn(72, 40)

    def grads(m):
        m.zero_grad()
        ((m.reconstruct_weight() - target) ** 2).sum().backward()
        return m.left.theta.grad.clone(), m.right.theta.grad.clone()

    plain = grads(module)
    module.set_checkpointing(2)
    ckpt = grads(module)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(plain, ckpt))
    assert grow_stages(module, left=6).left.checkpoint_segments == 2
    with torch.no_grad():  # without grad the plain path runs; same weight as with checkpointing off
        w_ckpt = module.reconstruct_weight()
        module.set_checkpointing(0)
        assert torch.equal(w_ckpt, module.reconstruct_weight())
