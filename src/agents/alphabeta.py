"""Alpha-beta Connect-Four agent using bitboards."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

from src.agents.base import Agent
from src.board import ConnectFourBoard

ROWS = 6
COLUMNS = 7
MOVE_ORDER = [3, 4, 2, 5, 1, 6, 0]


def bit_index(column: int, row: int) -> int:
    return column * ROWS + row


def winning_position(bitboard: int) -> bool:
    # Vertical
    m = bitboard & (bitboard >> 1)
    if m & (m >> 2):
        return True
    # Horizontal
    m = bitboard & (bitboard >> ROWS)
    if m & (m >> (2 * ROWS)):
        return True
    # Diagonal /
    m = bitboard & (bitboard >> (ROWS - 1))
    if m & (m >> (2 * (ROWS - 1))):
        return True
    # Diagonal \
    m = bitboard & (bitboard >> (ROWS + 1))
    if m & (m >> (2 * (ROWS + 1))):
        return True
    return False


class AlphaBetaAgent(Agent):
    def __init__(self, depth: int = 7) -> None:
        super().__init__(name="alphabeta")
        self.depth = depth
        self.tt: Dict[Tuple[int, int, int], int] = {}

    def select_action(self, board: ConnectFourBoard, **_: object) -> int:  # type: ignore[override]
        position, opponent = self._bitboards_from_board(board)
        mask = position | opponent
        best_score = -float("inf")
        best_move = -1
        alpha = -float("inf")
        beta = float("inf")
        for col in MOVE_ORDER:
            bit = self._lowest_empty_bit(mask, col)
            if bit == 0:
                continue
            new_position = position | bit
            if winning_position(new_position):
                return col
            score = -self._negamax(
                opponent, new_position, mask | bit, self.depth - 1, -beta, -alpha
            )
            if score > best_score:
                best_score = score
                best_move = col
            alpha = max(alpha, score)
        if best_move == -1:
            legal = board.valid_moves()
            if not legal:
                raise RuntimeError("AlphaBetaAgent: no legal moves available")
            return legal[0]
        return best_move

    def _bitboards_from_board(self, board: ConnectFourBoard) -> Tuple[int, int]:
        p1, p2 = board.to_bitboards()
        if board.current_player == 1:
            return p1, p2
        return p2, p1

    def _lowest_empty_bit(self, mask: int, column: int) -> int:
        for row in range(ROWS):
            bit = 1 << bit_index(column, row)
            if mask & bit == 0:
                return bit
        return 0

    def _negamax(
        self,
        position: int,
        opponent: int,
        mask: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        key = (position, opponent, depth)
        if key in self.tt:
            return self.tt[key]

        if winning_position(opponent):
            return -1.0
        if depth == 0 or mask == (1 << (ROWS * COLUMNS)) - 1:
            return 0.0

        value = -float("inf")
        for col in MOVE_ORDER:
            bit = self._lowest_empty_bit(mask, col)
            if bit == 0:
                continue
            new_position = position | bit
            if winning_position(new_position):
                self.tt[key] = 1.0
                return 1.0
            score = -self._negamax(
                opponent, new_position, mask | bit, depth - 1, -beta, -alpha
            )
            if score > value:
                value = score
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        self.tt[key] = value
        return value


__all__ = ["AlphaBetaAgent"]
