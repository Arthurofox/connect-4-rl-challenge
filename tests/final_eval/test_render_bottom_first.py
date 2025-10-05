from __future__ import annotations

import numpy as np

from src.board import ConnectFourBoard


def test_render_bottom_first_stacks_from_bottom() -> None:
    board = ConnectFourBoard()
    board.drop(3)
    board.drop(3)

    grid = board.render_grid_bottom_first()
    assert grid.shape == (6, 7)
    assert grid[0, 3] in (1, 2)
    assert grid[1, 3] in (1, 2)
    assert grid[0, 3] != grid[1, 3]
    assert np.all(grid[2:, 3] == 0), "Cells above the placed discs should be empty"
