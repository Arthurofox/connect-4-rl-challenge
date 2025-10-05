import numpy as np
import torch

from src.agents.muzero.mcts import MCTS
from src.agents.muzero.networks import MuZeroNet
from src.board import ConnectFourBoard


def fill_column(board: ConnectFourBoard, column: int) -> None:
    other = (column + 1) % board.columns
    for idx in range(board.rows):
        board.drop(column)
        if idx != board.rows - 1:
            board.drop(other)


def test_mcts_masks_full_column():
    board = ConnectFourBoard()
    fill_column(board, 0)

    net = MuZeroNet(channels=16, res_blocks=1, latent_dim=32, action_dim=7).to(
        torch.device("cpu")
    )
    mcts = MCTS(
        net, c_puct=1.5, discount=1.0, action_space=7, device=torch.device("cpu")
    )

    policy, _ = mcts.search(
        board.copy(),
        board.current_player,
        simulations=5,
        temperature=1.0,
        add_noise=False,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
    )

    assert np.isclose(policy[0], 0.0)
    assert np.argmax(policy) != 0
