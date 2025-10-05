from __future__ import annotations

from final_eval.arena import fight
from src.agents.alphabeta import AlphaBetaAgent


def test_arena_alpha_beta_smoke() -> None:
    agent_one = AlphaBetaAgent(depth=3)
    agent_two = AlphaBetaAgent(depth=3)
    summary = fight(agent_one, agent_two, games=4, swap_first=True, sims1=0, sims2=0)
    played = summary["wins_agent1"] + summary["wins_agent2"] + summary["draws"]
    assert played == 4
