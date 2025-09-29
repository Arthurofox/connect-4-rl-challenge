import numpy as np
import pytest

from src.agents.random_policy import RandomAgent
from src.env import ConnectFourEnv


def test_env_reset_returns_empty_board():
    env = ConnectFourEnv(random_first_player=False)
    obs, info = env.reset()
    assert obs.shape == (6, 7)
    assert np.all(obs == 0)
    assert info["current_player"] == 1


def test_invalid_move_penalizes_agent():
    env = ConnectFourEnv(random_first_player=False)
    env.reset()
    # Fill column 0 manually.
    for _ in range(env.board.rows):
        env.board.drop(0)
    obs, reward, terminated, _, info = env.step(0)
    assert terminated is True
    assert reward < 0
    assert info.get("invalid_move") is True
    assert info["reward_breakdown"]["invalid_move"] == pytest.approx(env.invalid_move_penalty)
    assert obs.shape == (6, 7)


def test_step_reward_shaping_applied():
    env = ConnectFourEnv(random_first_player=False, reward_step=-0.05, opponent=RandomAgent(seed=0))
    env.reset(seed=123)
    _, reward, terminated, _, info = env.step(3)
    assert info["reward_breakdown"].get("step") == pytest.approx(-0.05)
    assert terminated in (True, False)
    assert reward <= 0.0
