"""Agent interfaces for Connect-Four policies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.board import ConnectFourBoard


class Agent(ABC):
    """Abstract agent API used by training and evaluation pipelines."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    def on_game_start(self, board: ConnectFourBoard) -> None:
        """Hook called before the first move of each game."""

    def on_game_end(self, board: ConnectFourBoard, winner: Optional[int]) -> None:
        """Hook called after the game finishes."""

    @abstractmethod
    def select_action(self, board: ConnectFourBoard) -> int:
        """Return the column to play for the current board state."""

    def save(self, path: str) -> None:
        """Persist the agent to disk. Override when agents are stateful."""

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "Agent":
        """Restore an agent from disk. Override when agents are stateful."""
        raise NotImplementedError("load must be implemented on stateful agents")


__all__ = ["Agent"]
