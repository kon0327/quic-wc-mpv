import numpy as np
import torch
from torch import nn

from quic_wc.pilot.m4_lite import InSitu, Swap, dataset_tokens


def test_dataset_tokens_match_action_tokenizer_rule():
    stats = {"q01": [-1.0, 0.0], "q99": [1.0, 2.0], "mask": [True, False]}
    actions = torch.tensor([[0.0, 1.0], [1.0, -1.0], [-2.0, 0.3]])
    tokens = dataset_tokens(actions, stats, vocab_size=32000)
    bins = np.linspace(-1, 1, 256)
    # OpenVLA normalization (data_utils.py): 2 (x - q01) / (q99 - q01 + 1e-8) - 1 on masked dims, then clip
    norm = np.array([[2 * 1.0 / (2.0 + 1e-8) - 1, 1.0], [2 * 2.0 / (2.0 + 1e-8) - 1, -1.0], [-1.0, 0.3]])
    assert torch.equal(tokens, torch.from_numpy(32000 - np.digitize(norm, bins)).long())


def test_in_situ_rel_l2_without_bias_in_the_denominator():
    w, b = torch.eye(3), torch.ones(3)
    mon = InSitu(w, b)
    mon.active = True
    x = torch.tensor([[1.0, 0.0, 0.0]])
    mon.hook(None, (x,), x @ w.T + b + torch.tensor([[0.5, 0.0, 0.0]]))
    assert abs(mon.value() - 0.5) < 1e-12


def test_swap_replaces_and_restores_weights():
    model = nn.Sequential(nn.Linear(4, 3))
    original_w, original_b = model[0].weight.detach().clone(), model[0].bias.detach().clone()
    rep = {"weight": torch.zeros(3, 4), "bias": torch.full((3,), 2.0)}
    replacements = [{"layer": "0", "kind": "asvd", "rank": 1, "policy": "corrected"}]
    x = torch.randn(5, 4)
    with Swap(model, replacements, {("0", 1, "corrected"): rep}) as swap:
        swap.monitors["0"].active = True
        assert torch.allclose(model(x), torch.full((5, 3), 2.0))
        assert swap.monitors["0"].value() > 0
    assert torch.equal(model[0].weight, original_w) and torch.equal(model[0].bias, original_b)
