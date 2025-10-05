import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
from tqdm import trange
from connect4_env import Connect4Env

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ================================================================
# 🧠 Q-Network
# ================================================================
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self._init_flatten_size()
        self.fc1 = nn.Linear(self.flat_size, 384)
        self.fc2 = nn.Linear(384, 7)

    def _init_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 6, 7)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            self.flat_size = x.numel()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ================================================================
# 💾 Replay Buffer
# ================================================================
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    def add(self, s, a, r, s2, done, mask2):
        self.buffer.append((s, a, r, s2, done, mask2))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

# ================================================================
# 🧮 Helper functions
# ================================================================
def flip_obs(obs):
    b = obs["board"][:, ::-1].copy()
    return {"board": b, "player": obs["player"], "mask": obs["mask"][::-1].copy()}

def flip_action(a):
    return 6 - int(a)

# simple heuristic opponent
def heuristic_act(obs):
    board, player, mask = obs["board"], obs["player"], obs["mask"]
    legal = np.where(mask > 0)[0]
    from connect4_env import Connect4Env
    # 1. win immediately
    for c in legal:
        e = Connect4Env(); e.board = board.copy(); e.player = player
        _, _, done, info = e.step(int(c))
        if done and "win" in info: return int(c)
    # 2. block opponent
    for c in legal:
        e = Connect4Env(); e.board = board.copy(); e.player = player
        e.step(int(c))
        opp_legal = np.where(e.valid_actions() > 0)[0]
        for oc in opp_legal:
            t = e.clone()
            _, _, done, info = t.step(int(oc))
            if done and "win" in info: return int(c)
    # 3. prefer center
    for c in [3,2,4,1,5,0,6]:
        if c in legal: return int(c)
    return int(np.random.choice(legal))

# ================================================================
# 🤖 DQN Agent
# ================================================================
class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.995, tau=0.01, batch_size=512):
        self.net = QNetwork().to(DEVICE)
        self.tgt = QNetwork().to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer = ReplayBuffer()

    def _obs_to_tensor(self, obs):
        b = obs["board"]; p = obs["player"]
        cur = (b == p).astype(np.float32)
        opp = (b == -p).astype(np.float32)
        pid = np.full_like(cur, 1.0 if p == 1 else 0.0, dtype=np.float32)
        x = np.stack([cur, opp, pid])
        return torch.tensor(x[None], dtype=torch.float32, device=DEVICE)

    def act(self, obs, eps=0.1):
        mask = obs["mask"]
        legal = np.where(mask > 0)[0]
        if np.random.rand() < eps:
            return int(np.random.choice(legal))
        with torch.no_grad():
            q = self.net(self._obs_to_tensor(obs)).cpu().numpy()[0]
            q[mask == 0] = -1e9
            return int(np.argmax(q))

    def learn(self):
        if len(self.buffer) < self.batch_size: return
        s, a, r, s2, done, mask2 = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        s2 = torch.tensor(s2, dtype=torch.float32, device=DEVICE)
        a = torch.tensor(a, dtype=torch.long, device=DEVICE)
        r = torch.tensor(r, dtype=torch.float32, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float32, device=DEVICE)
        mask2 = torch.tensor(mask2, dtype=torch.float32, device=DEVICE)

        q = self.net(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            q2_live = self.net(s2)
            q2_live[mask2 == 0] = -1e9
            a2 = q2_live.argmax(1)
            q2_tgt = self.tgt(s2)
            q2_tgt[mask2 == 0] = -1e9
            next_q = q2_tgt.gather(1, a2[:, None]).squeeze(1)
            target = r + (1 - done) * self.gamma * next_q

        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        # soft update
        with torch.no_grad():
            for p, tp in zip(self.net.parameters(), self.tgt.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.net.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.net.load_state_dict(ckpt["state_dict"])
        self.tgt.load_state_dict(self.net.state_dict())

# ================================================================
# 🎯 Evaluation vs random baseline
# ================================================================
def eval_vs_random(agent, games=50):
    from random import choice
    wins = losses = draws = 0
    for g in range(games):
        env = Connect4Env()
        obs = env.reset(starting_player=1 if g % 2 == 0 else -1)
        done = False
        while not done:
            if obs["player"] == 1:
                a = agent.act(obs, eps=0.0)
            else:
                legal = np.where(obs["mask"] > 0)[0]
                a = int(choice(legal))
            obs, r, done, info = env.step(a)
        if "win" in info:
            if r == 1 and obs["player"] == -1: wins += 1
            else: losses += 1
        else:
            draws += 1
    wr = wins / games
    print(f"🎯 Eval vs Random — Wins: {wins}, Losses: {losses}, Draws: {draws} | Win Rate: {wr:.1%}")
    return wr

# ================================================================
# 🚀 Training Loop
# ================================================================
def train(
    episodes=3000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.9995,
    save_path="runs/dqn_final.pt"
):
    agent = DQNAgent()
    env = Connect4Env()
    eps = eps_start
    total_rewards = []
    best_wr = -1.0
    best_path = save_path.replace(".pt", "_best.pt")

    print(f"\n🚀 Starting training on {DEVICE} for {episodes} episodes...\n")

    for ep in trange(1, episodes + 1, desc="Training", ncols=80):
        obs = env.reset(starting_player=1 if random.random() < 0.5 else -1)
        done = False
        ep_reward = 0

        while not done:
            # player +1 = our agent, player -1 = heuristic/self-play
            if obs["player"] == 1:
                a = agent.act(obs, eps)
            else:
                a = heuristic_act(obs) if random.random() < 0.30 else agent.act(obs, eps)

            nxt, r, done, info = env.step(a)
            s = agent._obs_to_tensor(obs).squeeze(0).cpu().numpy()
            s2 = agent._obs_to_tensor(nxt).squeeze(0).cpu().numpy()
            agent.buffer.add(s, a, r, s2, done, nxt["mask"])

            # --- symmetry augmentation ---
            obs_f = flip_obs(obs)
            nxt_f = flip_obs(nxt)
            sf  = agent._obs_to_tensor(obs_f).squeeze(0).cpu().numpy()
            s2f = agent._obs_to_tensor(nxt_f).squeeze(0).cpu().numpy()
            af  = flip_action(a)
            agent.buffer.add(sf, af, r, s2f, done, nxt_f["mask"])
            # -------------------------------

            agent.learn()
            obs = nxt
            ep_reward += r

        total_rewards.append(ep_reward)
        eps = max(eps * eps_decay, eps_end)

        if ep % 50 == 0:
            avg_r = np.mean(total_rewards[-50:])
            print(f"Ep {ep:4d} | ε={eps:.3f} | avg_r={avg_r:+.3f} | buffer={len(agent.buffer)}")

        if ep % 500 == 0:
            wr = eval_vs_random(agent, games=50)
            if wr > best_wr:
                best_wr = wr
                agent.save(best_path)
                print(f"💾 New best checkpoint saved to {best_path}")

    agent.save(save_path)
    print(f"\n✅ Training finished! Final model saved to {save_path}")
    print(f"Best win rate during training: {best_wr:.1%}")
    return agent

# ================================================================
# 🧪 Standalone evaluation helper
# ================================================================
def test_vs_random(agent, games=100):
    from random import choice
    wins = losses = draws = 0
    for g in range(games):
        env = Connect4Env()
        obs = env.reset(starting_player=1 if g % 2 == 0 else -1)
        done = False
        while not done:
            if obs["player"] == 1:
                a = agent.act(obs, eps=0.0)
            else:
                legal = np.where(obs["mask"] > 0)[0]
                a = int(choice(legal))
            obs, r, done, info = env.step(a)
        if "win" in info:
            if r == 1 and obs["player"] == -1: wins += 1
            else: losses += 1
        else:
            draws += 1
    print(f"\n🎯 Evaluation vs Random Agent ({games} games)")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"✅ Win Rate: {wins / games:.2%}\n")
