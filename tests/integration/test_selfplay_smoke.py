from pathlib import Path

from src.agents.muzero import (
    MuZeroAgent,
    MuZeroConfig,
    SelfPlayConfig,
    MCTSConfig,
    ModelConfig,
    TrainHyperParams,
    LoggingConfig,
    EvalConfig,
)


def make_test_config(tmp_path: Path) -> MuZeroConfig:
    return MuZeroConfig(
        algo="muzero",
        seed=0,
        device="cpu",
        self_play=SelfPlayConfig(
            games_per_iter=5,
            max_moves=12,
            temperature_moves=2,
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
            replay_capacity=512,
            warm_start_steps=0,
            amp=False,
        ),
        logging=LoggingConfig(
            tensorboard=str(tmp_path / "tensorboard"),
            ckpt_dir=str(tmp_path / "ckpts"),
            save_every_iters=10,
        ),
        eval=EvalConfig(games=4, opponent="random", opponent_depth=3),
    )


def test_selfplay_generates_replay(tmp_path):
    config = make_test_config(tmp_path)
    agent = MuZeroAgent(config)

    games, _ = agent._generate_self_play_games()
    for game in games:
        agent.replay.add_game(game)

    assert len(agent.replay) > 0
    agent.writer.close()
