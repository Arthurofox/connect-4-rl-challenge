from __future__ import annotations

from typing import List

from src.agents.alphabeta import AlphaBetaAgent
from src.agents.muzero import MuZeroAgent, MuZeroConfig
from src.agents.random_policy import RandomAgent
from src.match import play_match
from src.board import ConnectFourBoard


def test_alphabeta_center_opening() -> None:
    agent = AlphaBetaAgent(depth=7)
    board = ConnectFourBoard()
    move = agent.select_action(board)
    assert move == 3


def test_opponent_receives_turns_trace() -> None:
    logs: List[dict] = []
    agent = RandomAgent(seed=42)
    opponent = AlphaBetaAgent(depth=5)
    actor_metadata = {
        "agent": {"algo": getattr(agent, "name", "agent"), "depth": None, "sims": 0, "stochastic": False},
        "opponent": {"algo": "alphabeta", "depth": 5, "sims": 0, "stochastic": False},
    }
    play_match(
        agent,
        opponent,
        games=1,
        swap_start=False,
        start_player="agent",
        trace_fn=logs.append,
        actor_metadata=actor_metadata,
    )
    assert any(entry.get("actor") == "opponent" and entry.get("algo") == "alphabeta" for entry in logs)


def test_muzero_greedy_vs_alphabeta_not_perfect() -> None:
    config = MuZeroConfig.from_dict(
        {
            "seed": 7,
            "model": {"channels": 16, "res_blocks": 1, "latent_dim": 32, "unroll_steps": 1},
            "mcts": {"simulations": 0},
        }
    )
    agent = MuZeroAgent(config)
    opponent = AlphaBetaAgent(depth=5)
    actor_metadata = {
        "agent": {"algo": "muzero", "depth": None, "sims": 0, "stochastic": False},
        "opponent": {"algo": "alphabeta", "depth": 5, "sims": 0, "stochastic": False},
    }
    result = play_match(
        agent,
        opponent,
        games=20,
        swap_start=True,
        actor_metadata=actor_metadata,
    )
    assert result.win_rate_agent_one < 0.9


def test_alphabeta_depth7_beats_random() -> None:
    agent = AlphaBetaAgent(depth=7)
    opponent = RandomAgent(seed=123)
    actor_metadata = {
        "agent": {"algo": "alphabeta", "depth": 7, "sims": 0, "stochastic": False},
        "opponent": {"algo": "random", "depth": None, "sims": 0, "stochastic": False},
    }
    result = play_match(
        agent,
        opponent,
        games=20,
        swap_start=True,
        actor_metadata=actor_metadata,
    )
    assert result.win_rate_agent_one >= 0.9
