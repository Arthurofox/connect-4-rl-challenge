import numpy as np
import pytest

from src.board import ConnectFourBoard, InvalidMoveError


def test_drop_disc_places_piece_and_switches_player():
    board = ConnectFourBoard()

    result = board.drop(3)

    assert board.copy_grid()[-1][3] == 1
    assert result.winner is None
    assert board.current_player == 2


def test_invalid_move_raises_error():
    board = ConnectFourBoard()
    with pytest.raises(InvalidMoveError):
        board.drop(-1)


def test_vertical_win_detection():
    board = ConnectFourBoard()
    for _ in range(3):
        board.drop(0)
        board.drop(1)
    result = board.drop(0)
    assert result.winner == 1


def test_encode_planes_to_move_plane():
    board = ConnectFourBoard()
    planes_p1 = board.encode_planes(1)
    planes_p2 = board.encode_planes(2)

    assert planes_p1.shape == (3, board.rows, board.columns)
    assert np.allclose(planes_p1[2], 1.0)
    assert np.allclose(planes_p2[2], -1.0)
