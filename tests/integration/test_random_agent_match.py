from src.agents.random_policy import RandomAgent
from src.match import play_match


def test_random_agents_complete_match():
    agent_one = RandomAgent(seed=0)
    agent_two = RandomAgent(seed=1)

    result = play_match(agent_one, agent_two, games=4)

    assert result.total_games == 4
    assert result.agent_one_wins + result.agent_two_wins + result.draws == 4
