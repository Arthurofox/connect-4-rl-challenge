from __future__ import annotations

import torch

from final_eval.dqn_agent import DQNAgent
from src.board import ConnectFourBoard


class DummyQ(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Shape: [batch, 7]; highest value in column 6 to ensure masking is required.
        return torch.tensor([[10.0, 9.0, 8.0, -5.0, 7.0, 6.0, 12.0]], dtype=torch.float32)


def _blocking_board() -> ConnectFourBoard:
    board = ConnectFourBoard()
    board._grid = [  # type: ignore[attr-defined]
        [1, 1, 2, 0, 2, 1, 2],
        [2, 2, 1, 1, 1, 2, 1],
        [1, 1, 2, 2, 2, 1, 2],
        [2, 2, 1, 1, 1, 2, 1],
        [1, 1, 2, 2, 2, 1, 2],
        [2, 2, 1, 1, 1, 2, 1],
    ]
    board.current_player = 1
    return board


def test_dqn_masks_illegal_columns() -> None:
    agent = DQNAgent(DummyQ())
    board = _blocking_board()
    assert board.valid_moves() == [3]
    action = agent.select_action(board, training=False)
    assert action == 3
