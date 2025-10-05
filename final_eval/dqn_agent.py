from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from src.board import ConnectFourBoard

from .friend_qnet import QNet


LegalMask = np.ndarray


def _legal_moves_mask(board: ConnectFourBoard) -> LegalMask:
    mask = np.zeros(board.columns, dtype=bool)
    for column in board.valid_moves():
        mask[column] = True
    return mask


def _ensure_numpy(array: torch.Tensor) -> np.ndarray:
    return array.detach().cpu().numpy()


class DQNAgent:
    """Lightweight DQN policy wrapper that masks illegal actions before argmax."""

    name = "dqn"

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "DQNAgent":
        checkpoint = torch.load(Path(path), map_location=device)
        model = cls._materialize_model(checkpoint, device)
        return cls(model, device=device)

    @staticmethod
    def _materialize_model(payload: Any, device: str) -> torch.nn.Module:
        if isinstance(payload, torch.nn.Module):
            payload.to(device)
            payload.eval()
            return payload
        state_dict = _extract_state_dict(payload)
        model = QNet()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def _encode(self, board: ConnectFourBoard) -> torch.Tensor:
        planes = board.encode_planes(board.current_player)
        tensor = torch.from_numpy(planes).float().unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def select_action(self, board: ConnectFourBoard, training: bool = False) -> int:  # noqa: ARG002
        mask = _legal_moves_mask(board)
        if not mask.any():
            raise RuntimeError("DQNAgent: no legal moves available")
        obs = self._encode(board)
        q_values = self.model(obs)
        q_values = _ensure_numpy(q_values.squeeze(0))
        masked = np.where(mask, q_values, -1e9)
        action = int(masked.argmax())
        return action


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if "model" in payload and isinstance(payload["model"], Mapping):
            return payload["model"]  # type: ignore[return-value]
        if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
            return payload["state_dict"]  # type: ignore[return-value]
        return payload  # type: ignore[return-value]
    raise TypeError(
        "Unsupported checkpoint format for DQNAgent: expected Module or Mapping"
    )
