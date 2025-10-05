"""Core Connect-Four board mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

ROWS = 6
COLUMNS = 7
CONNECT = 4


class InvalidMoveError(ValueError):
    """Raised when an agent attempts to play an illegal column."""


@dataclass
class MoveResult:
    winner: Optional[int]
    board_full: bool


class ConnectFourBoard:
    """Stateful Connect-Four board supporting turn-based play."""

    def __init__(self, rows: int = ROWS, columns: int = COLUMNS) -> None:
        self.rows = rows
        self.columns = columns
        self._grid: List[List[int]] = [[0 for _ in range(columns)] for _ in range(rows)]
        self.current_player: int = 1

    def reset(self) -> None:
        self._grid = [[0 for _ in range(self.columns)] for _ in range(self.rows)]
        self.current_player = 1

    @property
    def to_move(self) -> int:
        return 1 if self.current_player == 1 else -1

    def copy(self) -> "ConnectFourBoard":
        clone = ConnectFourBoard(self.rows, self.columns)
        clone._grid = self.copy_grid()
        clone.current_player = self.current_player
        return clone

    def copy_grid(self) -> List[List[int]]:
        return [row.copy() for row in self._grid]

    def valid_moves(self) -> List[int]:
        return [col for col in range(self.columns) if self._grid[0][col] == 0]

    def is_full(self) -> bool:
        return all(cell != 0 for cell in self._grid[0])

    def drop(self, column: int) -> MoveResult:
        if column not in range(self.columns):
            raise InvalidMoveError(
                f"Column {column} outside valid range 0-{self.columns - 1}"
            )
        if self._grid[0][column] != 0:
            raise InvalidMoveError(f"Column {column} is full")

        target_row = self._find_row(column)
        self._grid[target_row][column] = self.current_player

        winner = self._detect_winner(target_row, column)
        board_full = self.is_full()
        self.current_player = 3 - self.current_player
        return MoveResult(winner=winner, board_full=board_full)

    def _find_row(self, column: int) -> int:
        for row in range(self.rows - 1, -1, -1):
            if self._grid[row][column] == 0:
                return row
        raise InvalidMoveError(f"Column {column} is full")

    def _detect_winner(self, row: int, column: int) -> Optional[int]:
        player = self._grid[row][column]
        if player == 0:
            return None

        directions = [
            (0, 1),  # horizontal
            (1, 0),  # vertical
            (1, 1),  # diagonal \
            (1, -1),  # diagonal /
        ]

        for delta_row, delta_col in directions:
            count = (
                1
                + self._line_length(row, column, delta_row, delta_col)
                + self._line_length(row, column, -delta_row, -delta_col)
            )
            if count >= CONNECT:
                return player
        return None

    def _line_length(
        self, row: int, column: int, delta_row: int, delta_col: int
    ) -> int:
        player = self._grid[row][column]
        total = 0
        current_row, current_col = row + delta_row, column + delta_col
        while 0 <= current_row < self.rows and 0 <= current_col < self.columns:
            if self._grid[current_row][current_col] != player:
                break
            total += 1
            current_row += delta_row
            current_col += delta_col
        return total

    def as_player_view(self, player: int) -> List[List[int]]:
        """Return a perspective copy with player pieces as 1 and opponent as -1."""
        opponent = 3 - player
        view: List[List[int]] = []
        for row in self._grid:
            view.append(
                [1 if cell == player else -1 if cell == opponent else 0 for cell in row]
            )
        return view

    def check_winner(self) -> int:
        for row in range(self.rows):
            for col in range(self.columns):
                player = self._grid[row][col]
                if player == 0:
                    continue
                if self._has_connect(row, col, player):
                    return player
        return 0

    def _has_connect(self, row: int, col: int, player: int) -> bool:
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while (
                0 <= r < self.rows
                and 0 <= c < self.columns
                and self._grid[r][c] == player
            ):
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while (
                0 <= r < self.rows
                and 0 <= c < self.columns
                and self._grid[r][c] == player
            ):
                count += 1
                r -= dr
                c -= dc
            if count >= CONNECT:
                return True
        return False

    def encode_planes(self, perspective: int) -> np.ndarray:
        grid = np.array(self._grid, dtype=np.int8)
        player_board = (grid == perspective).astype(np.float32)
        opponent_board = ((grid != 0) & (grid != perspective)).astype(np.float32)
        to_move_value = 1.0 if perspective == 1 else -1.0
        to_move_plane = np.full(
            (self.rows, self.columns), to_move_value, dtype=np.float32
        )
        stacked = np.stack([player_board, opponent_board, to_move_plane], axis=0)
        return stacked

    def to_bitboards(self) -> Tuple[int, int]:
        p1 = 0
        p2 = 0
        for c in range(self.columns):
            for r in range(self.rows):
                bit_index = c * self.rows + r
                cell = self._grid[self.rows - 1 - r][c]
                if cell == 1:
                    p1 |= 1 << bit_index
                elif cell == 2:
                    p2 |= 1 << bit_index
        return p1, p2

    def render(self) -> str:
        rows = ["|" + " ".join(str(cell) for cell in row) + "|" for row in self._grid]
        footer = " " + " ".join(str(idx) for idx in range(self.columns))
        return "\n".join(rows + [footer])

    def __iter__(self) -> Iterable[List[int]]:
        return iter(self._grid)


__all__ = [
    "CONNECT",
    "COLUMNS",
    "ROWS",
    "ConnectFourBoard",
    "InvalidMoveError",
    "MoveResult",
]
