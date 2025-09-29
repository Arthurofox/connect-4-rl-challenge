"""Core Connect-Four board mechanics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

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

    def copy_grid(self) -> List[List[int]]:
        return [row.copy() for row in self._grid]

    def valid_moves(self) -> List[int]:
        return [col for col in range(self.columns) if self._grid[0][col] == 0]

    def is_full(self) -> bool:
        return all(cell != 0 for cell in self._grid[0])

    def drop(self, column: int) -> MoveResult:
        if column not in range(self.columns):
            raise InvalidMoveError(f"Column {column} outside valid range 0-{self.columns - 1}")
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
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal \
            (1, -1),  # diagonal /
        ]

        for delta_row, delta_col in directions:
            count = 1 + self._line_length(row, column, delta_row, delta_col) + self._line_length(
                row, column, -delta_row, -delta_col
            )
            if count >= CONNECT:
                return player
        return None

    def _line_length(self, row: int, column: int, delta_row: int, delta_col: int) -> int:
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
        """Return a perspective copy where the current player is always 1."""
        multiplier = 1 if player == 1 else -1
        return [[cell * multiplier for cell in row] for row in self._grid]

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
