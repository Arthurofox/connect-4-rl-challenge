"""Stable-Baselines3 agent wrapper for Connect-Four."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Type

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.agents.base import Agent


_SB3_LOADERS = {
    "ppo": PPO,
}


class SB3Agent(Agent):
    def __init__(
        self,
        model: BaseAlgorithm,
        *,
        deterministic: bool = True,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "sb3")
        self.model = model
        self.deterministic = deterministic

    def select_action(self, board) -> int:  # type: ignore[override]
        player = board.current_player
        observation = np.asarray(board.as_player_view(player), dtype=np.float32)
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return int(action)

    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        algorithm: str = "ppo",
        deterministic: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ) -> "SB3Agent":
        loader: Optional[Type[BaseAlgorithm]] = _SB3_LOADERS.get(algorithm.lower())
        if loader is None:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        model = loader.load(path, device=device, **kwargs)
        return cls(model=model, deterministic=deterministic, name=f"sb3-{algorithm}")


__all__ = ["SB3Agent"]
