import numpy as np

from src.agents.muzero.replay import GameHistory, ReplayBuffer


def make_policy(action_space: int, idx: int) -> list[float]:
    vec = [0.0 for _ in range(action_space)]
    vec[idx] = 1.0
    return vec


def test_sample_batch_masks_terminal_suffix():
    buffer = ReplayBuffer(capacity=10, action_space=7, unroll_steps=3)

    history = GameHistory()
    obs = np.zeros((3, 6, 7), dtype=np.float32)
    history.add_step(obs, player=1, to_move=1, policy=make_policy(7, 3), action=3, reward=1.0)
    history.winner = 1
    buffer.add_game(history)

    batch = buffer.sample_batch(1)
    mask = batch["mask"][0].numpy()
    assert mask[0] == 1.0
    assert np.allclose(mask[1:], 0.0)

    root_policy = batch["target_policy"][0, 0].numpy()
    assert np.isclose(root_policy.sum(), 1.0)
    assert root_policy[3] == 1.0
