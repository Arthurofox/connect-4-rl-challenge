import numpy as np
from connect4_env import Connect4Env
from rl_agent import DQNAgent

# ======================================================
# 🧩 Agent vs Agent Arena
# ======================================================
def battle(agent1, agent2, n_games=50, render=False):
    wins_1, wins_2, draws = 0, 0, 0
    for g in range(n_games):
        env = Connect4Env()
        obs = env.reset(starting_player=1 if g % 2 == 0 else -1)
        done = False
        while not done:
            player = obs["player"]
            a = agent1.act(obs, eps=0.0) if player == 1 else agent2.act(obs, eps=0.0)
            obs, r, done, info = env.step(a)
            if render:
                env.render()
                print()
        if "win" in info:
            if r == 1 and player == 1:
                wins_1 += 1
            else:
                wins_2 += 1
        else:
            draws += 1
    print("===================================")
    print(f"Games played: {n_games}")
    print(f"Agent 1 wins: {wins_1}")
    print(f"Agent 2 wins: {wins_2}")
    print(f"Draws: {draws}")
    print(f"Win rate Agent 1: {wins_1 / n_games:.2%}")
    print(f"Win rate Agent 2: {wins_2 / n_games:.2%}")
    print("===================================")
    return wins_1, wins_2, draws


if __name__ == "__main__":
    print("\n⚔️  Connect-4 RL Arena — Agent vs Agent\n")

    # Load your trained agent
    agent_me = DQNAgent()
    agent_me.load("runs/Roumouz_final_best.pt")

    # Load opponent agent (Arthur's model)
    agent_arthur = DQNAgent()
    agent_arthur.load("runs/arthur_final.pt")  # change to the correct file name

    # Run the battle
    battle(agent_me, agent_arthur, n_games=100, render=False)
