"""Baseline agent that selects uniformly random legal moves."""

from __future__ import annotations

import random
from typing import Optional

from src.agents.base import Agent
from src.board import ConnectFourBoard


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(name="random")
        self._rng = random.Random(seed)

    def select_action(self, board: ConnectFourBoard) -> int:
        moves = board.valid_moves()
        if not moves:
            raise RuntimeError("No available moves for RandomAgent")
        return self._rng.choice(moves)


__all__ = ["RandomAgent"]
