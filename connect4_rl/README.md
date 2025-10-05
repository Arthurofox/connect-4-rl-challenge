
MODEL FORMAT
--------------------------------------------------------
- Format: PyTorch checkpoint (.pt)
- File name: runs/Roumouz_final_best.pt
- The model can be reloaded using:
    agent = DQNAgent()
    agent.load("runs/Roumouz_final_best.pt")

--------------------------------------------------------
APPROACH
--------------------------------------------------------
Algorithm: Deep Q-Network (DQN) with self-play and heuristic sparring.

The agent learns by playing against itself and occasionally against a heuristic opponent 
(30% of the time) that prioritizes winning and blocking moves.  
Training also uses data augmentation by horizontally flipping the board, 
which effectively doubles the experience replay data.

Input representation:
- 3 channels: current player pieces, opponent pieces, and player indicator.  
- Board size: 6 rows × 7 columns.

Network architecture:
- Convolutional layer 1: 96 filters, kernel size 4  
- Convolutional layer 2: 96 filters, kernel size 3  
- Fully connected layer: 384 neurons  
- Output: 7 Q-values (one per column)  
- Activation: ReLU  
- Optimization: Adam (lr = 1e-3)  
- Loss: Smooth L1 (Huber loss)

--------------------------------------------------------
TRAINING SETUP
--------------------------------------------------------
- Episodes: ~9,000 (until policy stabilization)
- Replay buffer: 100,000 transitions
- Batch size: 512
- Discount factor (γ): 0.995
- Target network soft update (τ): 0.01
- Exploration (ε): 1.0 → 0.01 (decay 0.9995)
- Opponent: 70% self-play, 30% heuristic
- Hardware: MacBook Air M2 (local only)
- Total training time: ~3 hours

--------------------------------------------------------
RESULTS
--------------------------------------------------------
Evaluation vs Random Agent (100 games):
- Wins: 84
- Losses: 16
- Draws: 0
- Win rate: 84.0%
--------------------------------------------------------
ARENA — AGENT VS AGENT MATCHES
--------------------------------------------------------
File: arena.py

Run head-to-head matches between two agents (e.g., my trained model vs Arthur's model).

Usage:
    python arena.py

You can modify the file to specify different checkpoints:
    agent_me.load("runs/Roumouz_final_best.pt")
    agent_arthur.load("runs/Arthur_final.pt")

The script reports:
- Wins, losses, and draws for both agents
- Win rates over N games
- Optional board rendering for visual debugging (render=True)



FILES INCLUDED
--------------------------------------------------------
1. connect4_env.py          — custom Connect-Four environment
2. rl_agent.py              — DQN agent implementation and training loop
3. runs/Roumouz_final_best.pt        — final trained model checkpoint
4. README.txt               — documentation and submission note
5. requirements.txt — list of dependencies
