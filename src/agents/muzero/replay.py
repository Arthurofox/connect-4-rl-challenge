"""Experience replay buffer for MuZero self-play."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch


@dataclass
class GameHistory:
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    policies: List[List[float]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    players: List[int] = field(default_factory=list)
    to_moves: List[int] = field(default_factory=list)
    winner: int = 0

    def add_step(
        self,
        observation: np.ndarray,
        player: int,
        to_move: int,
        policy: Sequence[float],
        action: int,
        reward: float,
    ) -> None:
        self.observations.append(observation)
        self.players.append(player)
        self.to_moves.append(int(to_move))
        self.policies.append(list(policy))
        self.actions.append(int(action))
        self.rewards.append(float(reward))

    def game_length(self) -> int:
        return len(self.actions)

    def value_target(self, index: int) -> float:
        if self.winner == 0:
            return 0.0
        winner_sign = 1 if self.winner == 1 else -1
        if index < len(self.to_moves):
            to_move = self.to_moves[index]
            return 1.0 if to_move == winner_sign else -1.0
        player = self.players[index]
        player_sign = 1 if player == 1 else -1
        return 1.0 if player_sign == winner_sign else -1.0


class ReplayBuffer:
    def __init__(self, capacity: int, action_space: int, unroll_steps: int) -> None:
        self.capacity = capacity
        self.action_space = action_space
        self.unroll_steps = unroll_steps
        self.buffer: List[GameHistory] = []
        self.num_positions = 0

    def __len__(self) -> int:
        return self.num_positions

    def add_game(self, game: GameHistory) -> None:
        self.buffer.append(game)
        self.num_positions += game.game_length()
        while self.num_positions > self.capacity and self.buffer:
            removed = self.buffer.pop(0)
            self.num_positions -= removed.game_length()

    def sample_batch(self, batch_size: int) -> dict:
        if not self.buffer:
            raise ValueError("Replay buffer is empty")
        batch_obs = []
        batch_actions = []
        batch_target_policies = []
        batch_target_values = []
        batch_target_rewards = []
        batch_masks = []

        for _ in range(batch_size):
            game = random.choice(self.buffer)
            game_len = game.game_length()
            idx = random.randint(0, max(0, game_len - 1))
            obs = game.observations[idx]
            batch_obs.append(obs)

            actions = []
            target_policies = []
            target_values = []
            target_rewards = []
            mask = []

            root_policy = (
                game.policies[idx]
                if idx < game_len
                else [1.0 / self.action_space] * self.action_space
            )
            target_policies.append(root_policy)
            target_values.append(game.value_target(idx))

            zero_policy = [0.0] * self.action_space
            for k in range(self.unroll_steps):
                current = idx + k
                if current < game_len:
                    actions.append(game.actions[current])
                    target_rewards.append(game.rewards[current])
                    mask.append(1.0)
                    next_index = current + 1
                    if next_index < game_len:
                        target_policies.append(game.policies[next_index])
                        target_values.append(game.value_target(next_index))
                    else:
                        target_policies.append(zero_policy)
                        target_values.append(0.0)
                else:
                    actions.append(0)
                    target_rewards.append(0.0)
                    target_policies.append(zero_policy)
                    target_values.append(0.0)
                    mask.append(0.0)

            batch_actions.append(actions)
            batch_target_policies.append(target_policies)
            batch_target_values.append(target_values)
            batch_target_rewards.append(target_rewards)
            batch_masks.append(mask)

        return {
            "obs": torch.tensor(np.stack(batch_obs), dtype=torch.float32),
            "actions": torch.tensor(batch_actions, dtype=torch.int64),
            "target_policy": torch.tensor(batch_target_policies, dtype=torch.float32),
            "target_value": torch.tensor(batch_target_values, dtype=torch.float32),
            "target_reward": torch.tensor(batch_target_rewards, dtype=torch.float32),
            "mask": torch.tensor(batch_masks, dtype=torch.float32),
        }


__all__ = ["ReplayBuffer", "GameHistory"]
