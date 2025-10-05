"""Gymnasium environment wrapping the Connect-Four board."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding

from src.agents.base import Agent
from src.agents.random_policy import RandomAgent
from src.board import ConnectFourBoard, InvalidMoveError


class ConnectFourEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        *,
        opponent: Optional[Agent] = None,
        reward_win: float = 1.0,
        reward_draw: float = 0.0,
        reward_loss: float = -1.0,
        invalid_move_penalty: float = -1.0,
        reward_step: float = 0.0,
        random_first_player: bool = True,
        swap_start_probability: float = 0.5,
    ) -> None:
        super().__init__()
        self.board = ConnectFourBoard()
        self.opponent = opponent or RandomAgent()
        self.reward_win = reward_win
        self.reward_draw = reward_draw
        self.reward_loss = reward_loss
        self.invalid_move_penalty = invalid_move_penalty
        self.reward_step = reward_step
        self.random_first_player = random_first_player
        self.swap_start_probability = swap_start_probability
        self.np_random, _ = seeding.np_random(None)

        self._observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.board.rows, self.board.columns),
            dtype=np.float32,
        )
        self._action_space = spaces.Discrete(self.board.columns)

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Discrete:
        return self._action_space

    def _get_obs(self) -> np.ndarray:
        view = self.board.as_player_view(player=1)
        return np.asarray(view, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.board.reset()
        if (
            self.random_first_player
            and self.np_random.random() < self.swap_start_probability
        ):
            self.board.current_player = 2
            # Opponent makes an opening move before the agent acts.
            _, _, _, _ = self._opponent_turn(initial_move=True)

        observation = self._get_obs()
        info = {"current_player": 1}
        return observation, info

    def step(self, action: int):
        terminated = False
        truncated = False
        reward = 0.0
        reward_breakdown: Dict[str, float] = {}
        info: Dict[str, Any] = {"opponent_move": None}

        try:
            result = self.board.drop(action)
        except InvalidMoveError:
            reward = self.invalid_move_penalty
            reward_breakdown["invalid_move"] = self.invalid_move_penalty
            terminated = True
            info["invalid_move"] = True
            info["reward_breakdown"] = reward_breakdown
            return self._get_obs(), reward, terminated, truncated, info

        if result.winner == 1:
            reward += self.reward_win
            reward_breakdown["win"] = self.reward_win
            terminated = True
        elif result.board_full:
            reward += self.reward_draw
            reward_breakdown["draw"] = self.reward_draw
            terminated = True
        else:
            if self.reward_step:
                reward += self.reward_step
                reward_breakdown["step"] = (
                    reward_breakdown.get("step", 0.0) + self.reward_step
                )
            opp_reward, terminated, opp_breakdown, opponent_move = self._opponent_turn()
            reward += opp_reward
            reward_breakdown.update(opp_breakdown)
            info["opponent_move"] = opponent_move

        info["reward_breakdown"] = reward_breakdown
        return self._get_obs(), reward, terminated, truncated, info

    def _opponent_turn(
        self,
        *,
        initial_move: bool = False,
    ) -> Tuple[float, bool, Dict[str, float], Optional[int]]:
        opponent_player = self.board.current_player
        move = self.opponent.select_action(self.board)
        result = self.board.drop(move)
        breakdown: Dict[str, float] = {}
        reward = 0.0

        if result.winner == opponent_player:
            if initial_move:
                reward += self.reward_draw
                breakdown["draw"] = self.reward_draw
            else:
                reward += self.reward_loss
                breakdown["loss"] = self.reward_loss
            return reward, True, breakdown, move
        if result.board_full:
            reward += self.reward_draw
            breakdown["draw"] = self.reward_draw
            return reward, True, breakdown, move

        return reward, False, breakdown, move

    def render(self):  # type: ignore[override]
        print(self.board.render())

    def close(self):
        return None

    def set_reward_scheme(self, **kwargs: float) -> None:
        if "reward_win" in kwargs:
            self.reward_win = float(kwargs["reward_win"])
        if "reward_draw" in kwargs:
            self.reward_draw = float(kwargs["reward_draw"])
        if "reward_loss" in kwargs:
            self.reward_loss = float(kwargs["reward_loss"])
        if "invalid_move_penalty" in kwargs:
            self.invalid_move_penalty = float(kwargs["invalid_move_penalty"])
        if "reward_step" in kwargs:
            self.reward_step = float(kwargs["reward_step"])

    def update_opponent(self, opponent: Agent) -> None:
        self.opponent = opponent

    def apply_curriculum(
        self,
        *,
        swap_start_probability: Optional[float] = None,
        reward_overrides: Optional[Dict[str, float]] = None,
        opponent: Optional[Agent] = None,
    ) -> None:
        if swap_start_probability is not None:
            self.swap_start_probability = float(
                np.clip(swap_start_probability, 0.0, 1.0)
            )
        if reward_overrides:
            self.set_reward_scheme(**reward_overrides)
        if opponent is not None:
            self.update_opponent(opponent)


__all__ = ["ConnectFourEnv"]
