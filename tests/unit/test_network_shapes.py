import torch

from src.agents.muzero.networks import MuZeroNet


def test_network_inference_shapes():
    net = MuZeroNet(channels=16, res_blocks=1, latent_dim=32, action_dim=7)
    obs = torch.zeros(1, 3, 6, 7)

    policy_logits, value, latent = net.initial_inference(obs)

    assert policy_logits.shape == (1, 7)
    assert value.shape == (1,)
    assert latent.shape == (1, 32)

    action = torch.tensor([3])
    policy_logits2, value2, reward2, latent_next = net.recurrent_inference(
        latent, action
    )

    assert policy_logits2.shape == (1, 7)
    assert value2.shape == (1,)
    assert reward2.shape == (1,)
    assert latent_next.shape == (1, 32)
    assert torch.all(value2.abs() <= 1.0)
    assert torch.all(reward2.abs() <= 1.0)
