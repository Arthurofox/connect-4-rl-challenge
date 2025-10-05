from src.agents.base import Agent
from src.agents.random_policy import RandomAgent
from src.agents.sb3_agent import SB3Agent
from src.match import play_match


class FixedColumnAgent(Agent):
    def __init__(self, column: int) -> None:
        super().__init__(name=f"fixed-{column}")
        self.column = column

    def select_action(self, board):
        return self.column


def test_random_agents_complete_match():
    agent_one = RandomAgent(seed=0)
    agent_two = RandomAgent(seed=1)

    result = play_match(agent_one, agent_two, games=4)

    assert result.total_games == 4
    assert result.agent_one_wins + result.agent_two_wins + result.draws == 4


class FixedColumnAgent(Agent):
    def __init__(self, column: int) -> None:
        super().__init__(name=f"fixed-{column}")
        self.column = column

    def select_action(self, board):
        return self.column


def test_invalid_move_gives_opponent_win():
    agent_one = FixedColumnAgent(0)
    agent_two = FixedColumnAgent(0)

    result = play_match(agent_one, agent_two, games=1, swap_start=False)

    assert result.agent_two_wins == 1
    assert result.games[0].winner == 2


def test_load_sb3_agent_cpu_default(tmp_path, monkeypatch):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.env import ConnectFourEnv

    env = DummyVecEnv([lambda: ConnectFourEnv(random_first_player=False)])
    model = PPO(
        "MlpPolicy", env, n_steps=16, batch_size=16, n_epochs=1, gamma=0.9, verbose=0
    )
    model.learn(total_timesteps=16)
    checkpoint = tmp_path / "ppo_test.zip"
    model.save(checkpoint)

    agent = SB3Agent.load(checkpoint)
    assert agent.name == "sb3-ppo"
