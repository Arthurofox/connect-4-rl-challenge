from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from src.agents.muzero.mcts import MCTS
from src.board import ConnectFourBoard


ROWS = 6
COLUMNS = 7
ACTION_SPACE = 7


def make_board_from_moves(moves: Tuple[int, ...]) -> ConnectFourBoard:
    board = ConnectFourBoard()
    for column in moves:
        board.drop(column)
    return board


def has_connect(grid: list[list[int]]) -> bool:
    rows = len(grid)
    cols = len(grid[0])
    for r in range(rows):
        for c in range(cols):
            player = grid[r][c]
            if player == 0:
                continue
            if c + 3 < cols and all(grid[r][c + i] == player for i in range(4)):
                return True
            if r + 3 < rows and all(grid[r + i][c] == player for i in range(4)):
                return True
            if c + 3 < cols and r + 3 < rows and all(
                grid[r + i][c + i] == player for i in range(4)
            ):
                return True
            if c - 3 >= 0 and r + 3 < rows and all(
                grid[r + i][c - i] == player for i in range(4)
            ):
                return True
    return False


def generate_mask_grid() -> list[list[int]]:
    patterns = (
        [1, 2, 1, 2, 1, 2],
        [2, 1, 2, 1, 2, 1],
    )
    for mask in range(1 << (COLUMNS - 1)):
        grid = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]
        for col in range(COLUMNS - 1):
            pattern = patterns[(mask >> col) & 1]
            for row in range(ROWS):
                grid[row][col] = pattern[row]
        if not has_connect(grid):
            return grid
    raise RuntimeError("Failed to construct a legal grid without a connect-four")


class FakeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_dim = ACTION_SPACE
        self.rows = ROWS
        self.columns = COLUMNS
        self.latent_dim = self.rows * self.columns + 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _board_from_obs(self, obs: torch.Tensor) -> ConnectFourBoard:
        perspective = 1 if float(obs[2, 0, 0]) > 0 else 2
        opponent = 3 - perspective
        board = ConnectFourBoard()
        for r in range(self.rows):
            for c in range(self.columns):
                if obs[0, r, c] > 0.5:
                    board._grid[r][c] = perspective
                elif obs[1, r, c] > 0.5:
                    board._grid[r][c] = opponent
                else:
                    board._grid[r][c] = 0
        board.current_player = perspective
        return board

    def _encode_board(self, board: ConnectFourBoard, device: torch.device) -> torch.Tensor:
        data = []
        for r in range(self.rows):
            for c in range(self.columns):
                data.append(float(board._grid[r][c]))
        data.append(float(board.current_player))
        return torch.tensor(data, dtype=torch.float32, device=device)

    def _apply_and_copy(self, board: ConnectFourBoard, column: int) -> Tuple[ConnectFourBoard, float]:
        clone = board.copy()
        current = clone.current_player
        result = clone.drop(column)
        if result.winner is None:
            reward = 0.0
        elif result.winner == current:
            reward = 1.0
        else:
            reward = -1.0
        return clone, reward

    def _policy_logits(self, board: ConnectFourBoard) -> torch.Tensor:
        logits = torch.full((self.action_dim,), -1e9)
        for col in range(self.columns):
            if board._grid[0][col] != 0:
                continue
            hypo, _ = self._apply_and_copy(board, col)
            if hypo.check_winner() == board.current_player:
                logits[col] = 12.0
            else:
                logits[col] = -4.0
        return logits

    def _value_estimate(self, board: ConnectFourBoard) -> float:
        winner = board.check_winner()
        if winner == board.current_player:
            return 1.0
        if winner != 0:
            return -1.0
        for col in range(self.columns):
            if board._grid[0][col] != 0:
                continue
            hypo, _ = self._apply_and_copy(board, col)
            if hypo.check_winner() == board.current_player:
                return 1.0
        opponent = 3 - board.current_player
        for col in range(self.columns):
            if board._grid[0][col] != 0:
                continue
            clone = board.copy()
            clone.current_player = opponent
            result = clone.drop(col)
            if result.winner == opponent:
                return -1.0
        return 0.0

    # ------------------------------------------------------------------
    # MuZero network API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def initial_inference(self, obs: torch.Tensor):
        obs = obs.squeeze(0)
        board = self._board_from_obs(obs)
        device = obs.device
        policy = self._policy_logits(board).to(device).unsqueeze(0)
        value = torch.tensor([self._value_estimate(board)], dtype=torch.float32, device=device)
        latent = self._encode_board(board, device).unsqueeze(0)
        return policy, value, latent

    @torch.no_grad()
    def recurrent_inference(self, latent: torch.Tensor, action: torch.Tensor):
        device = latent.device
        data = latent.squeeze(0).cpu().numpy()
        board = ConnectFourBoard()
        idx = 0
        for r in range(self.rows):
            for c in range(self.columns):
                board._grid[r][c] = int(data[idx])
                idx += 1
        board.current_player = int(data[-1])
        column = int(action.item())
        current = board.current_player
        result = board.drop(column)
        if result.winner == current:
            reward = 1.0
        elif result.winner is not None:
            reward = -1.0
        else:
            reward = 0.0
        policy = self._policy_logits(board).to(device).unsqueeze(0)
        value = torch.tensor([self._value_estimate(board)], dtype=torch.float32, device=device)
        latent_next = self._encode_board(board, device).unsqueeze(0)
        return policy, value, torch.tensor([reward], dtype=torch.float32, device=device), latent_next

    def training_initial_inference(self, obs: torch.Tensor):  # pragma: no cover - unused in tests
        return self.initial_inference(obs)

    def training_recurrent_inference(self, latent: torch.Tensor, action: torch.Tensor):  # pragma: no cover - unused in tests
        policy, value, reward, latent_next = self.recurrent_inference(latent, action)
        return policy, value, reward, latent_next

    # Convenience for tests
    def greedy_action(self, board: ConnectFourBoard) -> int:
        logits = self._policy_logits(board)
        legal = [col for col in range(self.columns) if board._grid[0][col] == 0]
        if not legal:
            return 0
        best = logits[legal[0]]
        best_col = legal[0]
        for col in legal:
            if logits[col] > best:
                best = logits[col]
                best_col = col
        return best_col


def make_mcts(fake_net: FakeNet) -> MCTS:
    return MCTS(
        fake_net,
        c_puct=2.0,
        discount=1.0,
        action_space=ACTION_SPACE,
        device=torch.device("cpu"),
    )


def test_eval_has_no_dirichlet_and_tau_zero():
    net = FakeNet()
    mcts = make_mcts(net)
    board = ConnectFourBoard()
    info = mcts.debug_self_check(
        board.copy(), board.current_player, simulations=1, temperature=0.0
    )
    assert info.used_dirichlet is False
    assert math.isclose(info.temperature, 0.0, abs_tol=1e-6)


def test_mcts_prefers_forced_win_in_one():
    net = FakeNet()
    mcts = make_mcts(net)
    # Player 1 can win vertically in column 0
    board = make_board_from_moves((0, 1, 0, 1, 0, 1))
    info = mcts.debug_self_check(
        board.copy(), board.current_player, simulations=64, temperature=0.0
    )
    q_values = info.q_values.tolist()
    visit_counts = info.visit_counts.tolist()
    winning_moves = []
    current_player = board.current_player
    for col in board.valid_moves():
        clone = board.copy()
        result = clone.drop(col)
        if result.winner == current_player:
            winning_moves.append(col)
    chosen = int(np.argmax(visit_counts))
    assert winning_moves, "Expected at least one immediate winning move"
    assert chosen in winning_moves
    best_q = max(q_values)
    assert np.isclose(q_values[chosen], best_q, atol=1e-4)


def test_mcts_masks_full_columns():
    net = FakeNet()
    mcts = make_mcts(net)
    board = ConnectFourBoard()
    grid = generate_mask_grid()
    for r in range(ROWS):
        for c in range(COLUMNS):
            board._grid[r][c] = grid[r][c]
    board.current_player = 1
    assert board.check_winner() == 0
    info = mcts.debug_self_check(
        board.copy(), board.current_player, simulations=16, temperature=0.0
    )
    legal = info.legal.tolist()
    assert legal.count(1.0) == 1
    only_legal = legal.index(1.0)
    visit_counts = info.visit_counts.tolist()
    assert all(
        count == 0.0 for idx, count in enumerate(visit_counts) if idx != only_legal
    )
    assert int(np.argmax(visit_counts)) == only_legal


def test_mcts_agrees_with_net_on_easy_tactic():
    net = FakeNet()
    mcts = make_mcts(net)
    board = make_board_from_moves((0, 1, 0, 1, 0, 1))
    greedy = net.greedy_action(board.copy())
    info = mcts.debug_self_check(
        board.copy(), board.current_player, simulations=32, temperature=0.0
    )
    chosen = int(np.argmax(info.visit_counts.tolist()))
    winning_moves = []
    current_player = board.current_player
    for col in board.valid_moves():
        clone = board.copy()
        result = clone.drop(col)
        if result.winner == current_player:
            winning_moves.append(col)
    assert greedy in winning_moves
    assert chosen == greedy
