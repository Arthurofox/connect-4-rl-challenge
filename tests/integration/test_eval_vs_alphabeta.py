from pathlib import Path

from src.agents.alphabeta import AlphaBetaAgent
from src.agents.muzero import (
    MuZeroAgent,
    MuZeroConfig,
    SelfPlayConfig,
    MCTSConfig,
    ModelConfig,
    TrainHyperParams,
    LoggingConfig,
    EvalConfig,
    resolve_device,
)
from src.match import play_match


def make_eval_config(tmp_path: Path) -> MuZeroConfig:
    return MuZeroConfig(
        algo="muzero",
        seed=1,
        device="cpu",
        self_play=SelfPlayConfig(
            games_per_iter=1,
            max_moves=12,
            temperature_moves=1,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.25,
        ),
        mcts=MCTSConfig(simulations=5, c_puct=1.5, value_discount=1.0),
        model=ModelConfig(channels=16, res_blocks=1, latent_dim=32, unroll_steps=2),
        train=TrainHyperParams(
            batch_size=4,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            epochs_per_iter=1,
            replay_capacity=256,
            warm_start_steps=0,
            amp=False,
        ),
        logging=LoggingConfig(
            tensorboard=str(tmp_path / "tb"),
            ckpt_dir=str(tmp_path / "ck"),
            save_every_iters=5,
        ),
        eval=EvalConfig(games=10, opponent="alphabeta", opponent_depth=3),
    )


def test_checkpoint_loads_and_plays(tmp_path):
    config = make_eval_config(tmp_path)
    agent = MuZeroAgent(config)
    checkpoint = tmp_path / "muzero.pt"
    agent.save_checkpoint(checkpoint)
    agent.writer.close()

    loaded = MuZeroAgent.from_checkpoint(
        checkpoint, device=resolve_device(config.device)
    )
    loaded.config.mcts.simulations = 5
    opponent = AlphaBetaAgent(depth=3)

    result = play_match(loaded, opponent, games=10)
    assert result.total_games == 10
    loaded.writer.close()
