# Final Evaluation Battleground

Keep the main repo untouched and drop the evaluation tools here.

## Setup
- Install Streamlit once inside your env:
  ```bash
  uv pip install streamlit
  ```

## Streamlit UI
Launch the clickable arena and point it at your checkpoints:
```bash
streamlit run final_eval/ui_streamlit.py -- \
  --muzero artifacts/released/muzero_final.pt \
  --dqn /path/to/friend_dqn.pt \
  --device cpu --sims 160
```

## Headless Arena
Run MuZero vs DQN (200 games, swap who starts each time):
```bash
python -m final_eval.arena \
  --agent1 muzero --ckpt1 artifacts/released/muzero_final.pt \
  --agent2 dqn --ckpt2 /path/to/friend_dqn.pt \
  --games 200 --swap-first --device cpu --sims1 160
```

Pit MuZero against AlphaBeta depth 7:
```bash
python -m final_eval.arena \
  --agent1 muzero --ckpt1 artifacts/released/muzero_final.pt \
  --agent2 alphabeta --depth2 7 --games 200 --swap-first --device cpu --sims1 160
```

## Notes
- tau = 0 for MuZero in evaluation, no Dirichlet noise applied.
- Illegal moves are masked by each agent; any remaining illegal action forfeits the game.
- Standard 6x7 Connect-Four board with columns indexed 0..6.
