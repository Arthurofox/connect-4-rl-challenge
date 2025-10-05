"""Training entry point for Connect-Four agents."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import yaml
from alive_progress import alive_bar
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.random_policy import RandomAgent
from src.agents.sb3_agent import SB3Agent
from src.agents.alphabeta import AlphaBetaAgent
from src.agents.muzero import (
    MuZeroAgent,
    MuZeroConfig,
    load_config as load_muzero_config,
)
from src.callbacks import AliveProgressCallback, SelfPlayCallback
from src.env import ConnectFourEnv
from src.match import MatchResult, play_match

DEFAULT_EPISODES = 1000
DEFAULT_OPPONENT = "random"
DEFAULT_TIMESTEPS = 50_000
DEFAULT_EVAL_GAMES = 20
DEFAULT_NUM_ENVS = 1
DEFAULT_REWARD_WIN = 1.0
DEFAULT_REWARD_DRAW = 0.0
DEFAULT_REWARD_LOSS = -1.0
DEFAULT_INVALID_MOVE = -1.0
DEFAULT_REWARD_STEP = 0.0
DEFAULT_SWAP_START_PROB = 0.5
DEFAULT_SELF_PLAY_UPDATE_INTERVAL = 20000
DEFAULT_SELF_PLAY_POOL_SIZE = 5
DEFAULT_SELF_PLAY_WARMUP = 5000
SUPPORTED_ALGOS = {"random", "ppo", "muzero"}


@dataclass
class TrainConfig:
    episodes: int = DEFAULT_EPISODES
    timesteps: int = DEFAULT_TIMESTEPS
    algo: str = "random"
    eval_games: int = DEFAULT_EVAL_GAMES
    num_envs: int = DEFAULT_NUM_ENVS
    seed: Optional[int] = None
    opponent: str = DEFAULT_OPPONENT
    tensorboard_log: Optional[Path] = None
    save_path: Optional[Path] = None
    load_path: Optional[Path] = None
    reset_num_timesteps: bool = True
    random_first_player: bool = True
    swap_start_probability: float = DEFAULT_SWAP_START_PROB
    reward_win: float = DEFAULT_REWARD_WIN
    reward_draw: float = DEFAULT_REWARD_DRAW
    reward_loss: float = DEFAULT_REWARD_LOSS
    invalid_move_penalty: float = DEFAULT_INVALID_MOVE
    reward_step: float = DEFAULT_REWARD_STEP
    self_play: bool = False
    self_play_update_interval: int = DEFAULT_SELF_PLAY_UPDATE_INTERVAL
    self_play_pool_size: int = DEFAULT_SELF_PLAY_POOL_SIZE
    self_play_warmup_steps: int = DEFAULT_SELF_PLAY_WARMUP
    self_play_deterministic_snapshots: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainConfig":
        tensorboard = payload.get("tensorboard_log")
        save_path = payload.get("save_path")
        load_path = payload.get("load_path")
        rewards = payload.get("rewards", {})
        return cls(
            episodes=int(payload.get("episodes", DEFAULT_EPISODES)),
            timesteps=int(payload.get("timesteps", DEFAULT_TIMESTEPS)),
            algo=str(payload.get("algo", "random")).lower(),
            eval_games=int(payload.get("eval_games", DEFAULT_EVAL_GAMES)),
            num_envs=int(payload.get("num_envs", DEFAULT_NUM_ENVS)),
            seed=payload.get("seed"),
            opponent=str(payload.get("opponent", DEFAULT_OPPONENT)),
            tensorboard_log=Path(tensorboard) if tensorboard else None,
            save_path=Path(save_path) if save_path else None,
            load_path=Path(load_path) if load_path else None,
            reset_num_timesteps=bool(payload.get("reset_num_timesteps", True)),
            random_first_player=bool(payload.get("random_first_player", True)),
            swap_start_probability=float(
                payload.get("swap_start_probability", DEFAULT_SWAP_START_PROB)
            ),
            reward_win=float(rewards.get("win", DEFAULT_REWARD_WIN)),
            reward_draw=float(rewards.get("draw", DEFAULT_REWARD_DRAW)),
            reward_loss=float(rewards.get("loss", DEFAULT_REWARD_LOSS)),
            invalid_move_penalty=float(
                rewards.get("invalid_move", DEFAULT_INVALID_MOVE)
            ),
            reward_step=float(rewards.get("step", DEFAULT_REWARD_STEP)),
            self_play=bool(payload.get("self_play", False)),
            self_play_update_interval=int(
                payload.get(
                    "self_play_update_interval", DEFAULT_SELF_PLAY_UPDATE_INTERVAL
                )
            ),
            self_play_pool_size=int(
                payload.get("self_play_pool_size", DEFAULT_SELF_PLAY_POOL_SIZE)
            ),
            self_play_warmup_steps=int(
                payload.get("self_play_warmup_steps", DEFAULT_SELF_PLAY_WARMUP)
            ),
            self_play_deterministic_snapshots=bool(
                payload.get("self_play_deterministic_snapshots", True)
            ),
        )


def load_any_config(path: Path) -> tuple[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a mapping at the top level")
    algo = str(payload.get("algo", "random")).lower()
    if algo == "muzero":
        return algo, load_muzero_config(path)
    config = TrainConfig.from_dict(payload)
    if config.algo not in {"random", "ppo"}:
        raise ValueError(
            f"Unsupported algo '{config.algo}'. Supported: ['random', 'ppo', 'muzero']"
        )
    return config.algo, config


def build_opponent(name: str, seed: Optional[int]) -> RandomAgent:
    if name != "random":
        raise ValueError(f"Unsupported opponent '{name}'. Only 'random' is available")
    return RandomAgent(seed=seed)


def make_env_factory(config: TrainConfig, base_seed: Optional[int]):
    def _factory(idx: int):
        def _init():
            env_seed = (base_seed or 0) + idx
            opponent = build_opponent(config.opponent, env_seed + 100)
            env = ConnectFourEnv(
                opponent=opponent,
                reward_win=config.reward_win,
                reward_draw=config.reward_draw,
                reward_loss=config.reward_loss,
                invalid_move_penalty=config.invalid_move_penalty,
                reward_step=config.reward_step,
                random_first_player=config.random_first_player,
                swap_start_probability=config.swap_start_probability,
            )
            env.reset(seed=env_seed)
            return env

        return _init

    return _factory


def train_with_ppo(config: TrainConfig) -> SB3Agent:
    env_factory = make_env_factory(config, config.seed)
    env_fns = [env_factory(i) for i in range(config.num_envs)]
    vec_env = DummyVecEnv(env_fns)

    if config.load_path:
        if not config.load_path.exists():
            raise FileNotFoundError(f"Load checkpoint not found: {config.load_path}")
        model = PPO.load(str(config.load_path), env=vec_env)
        if config.tensorboard_log:
            model.tensorboard_log = str(config.tensorboard_log)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            seed=config.seed,
            tensorboard_log=(
                str(config.tensorboard_log) if config.tensorboard_log else None
            ),
        )

    initial_opponent = build_opponent(config.opponent, (config.seed or 0) + 1000)
    vec_env.env_method("update_opponent", initial_opponent)

    callbacks = []

    with (
        TemporaryDirectory() as snapshot_dir,
        alive_bar(
            config.timesteps, title="Training", bar="smooth", spinner="dots_waves2"
        ) as bar,
    ):
        callbacks.append(AliveProgressCallback(bar, config.timesteps))
        if config.self_play:
            callbacks.append(
                SelfPlayCallback(
                    snapshot_dir=Path(snapshot_dir),
                    initial_opponents=[initial_opponent],
                    update_interval=config.self_play_update_interval,
                    pool_size=config.self_play_pool_size,
                    warmup_steps=config.self_play_warmup_steps,
                    deterministic_snapshot=config.self_play_deterministic_snapshots,
                    seed=config.seed,
                )
            )
        callback_obj = None
        if callbacks:
            callback_obj = (
                callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
            )
        model.learn(
            total_timesteps=config.timesteps,
            reset_num_timesteps=config.reset_num_timesteps,
            callback=callback_obj,
        )
    if config.save_path:
        config.save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(config.save_path))
    return SB3Agent(model=model, deterministic=True, name="ppo-agent")


def run_muzero_training(config: MuZeroConfig, args: argparse.Namespace) -> MatchResult:
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.tensorboard_log is not None:
        config.logging.tensorboard = str(args.tensorboard_log)
    if args.save_checkpoint is not None:
        config.logging.ckpt_dir = str(Path(args.save_checkpoint).parent)
    if args.mcts_simulations is not None:
        config.mcts.simulations = args.mcts_simulations
    if args.temperature_moves is not None:
        config.self_play.temperature_moves = args.temperature_moves
    if args.eval_games is not None:
        config.eval.games = args.eval_games

    if args.num_workers and args.num_workers > 1:
        print(
            "[warn] --num-workers not yet implemented; running single-process self-play"
        )

    iterations = args.iters or 1
    save_every = args.save_every or config.logging.save_every_iters
    save_path = (
        args.save_checkpoint or Path(config.logging.ckpt_dir) / "muzero_latest.pt"
    )

    gate_games = max(0, args.gate_games or 0)
    gate_opponent = args.gate_opponent.lower() if args.gate_opponent else "none"
    gate_depth = args.gate_depth
    gate_sims = 160
    leaderboard_path = Path("artifacts/released/leaderboard.jsonl")
    best_gate_scores: Dict[str, Optional[float]] = {"random": None, "gate": None}

    def record_leaderboard_entry(
        iteration: int,
        opponent_name: str,
        depth: Optional[int],
        result: MatchResult,
    ) -> None:
        leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        wins = result.agent_one_wins
        draws = result.draws
        losses = result.agent_two_wins
        entry = {
            "iter": iteration,
            "sims": gate_sims,
            "opponent": opponent_name,
            "depth": depth,
            "wdl": f"{wins}-{draws}-{losses}",
            "win_rate": round(result.win_rate_agent_one, 4),
        }
        with leaderboard_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def maybe_copy_best(source: Path) -> Path:
        target = source.with_name("muzero_best.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_yaml = source.with_suffix(".yaml")
        if source_yaml.exists():
            shutil.copy2(source_yaml, target.with_suffix(".yaml"))
        return target

    def gating_callback(agent_obj: MuZeroAgent, iteration: int, ckpt_path: Path) -> None:
        if gate_games <= 0:
            return
        original_training_mode = agent_obj.net.training
        agent_obj.net.eval()
        sims_backup = agent_obj.config.mcts.simulations
        stochastic_backup = agent_obj.eval_stochastic
        agent_obj.eval_stochastic = False
        try:
            agent_obj.config.mcts.simulations = gate_sims
            random_result = play_match(
                agent_obj,
                RandomAgent(seed=123 + iteration),
                games=gate_games,
                swap_start=True,
            )
            random_score = random_result.win_rate_agent_one
            if (
                best_gate_scores["random"] is None
                or random_score >= (best_gate_scores["random"] or 0.0) + 0.03
            ):
                best_gate_scores["random"] = random_score
                maybe_copy_best(ckpt_path)
                record_leaderboard_entry(iteration, "random", None, random_result)

            if gate_opponent != "none":
                if gate_opponent == "alphabeta":
                    gate_opp = AlphaBetaAgent(depth=gate_depth)
                else:
                    gate_opp = RandomAgent(seed=321 + iteration)
                gate_result = play_match(
                    agent_obj,
                    gate_opp,
                    games=gate_games,
                    swap_start=True,
                )
                gate_score = gate_result.win_rate_agent_one
                key = "gate"
                if (
                    best_gate_scores[key] is None
                    or gate_score >= (best_gate_scores[key] or 0.0) + 0.03
                ):
                    best_gate_scores[key] = gate_score
                    maybe_copy_best(ckpt_path)
                    depth_value = gate_depth if gate_opponent == "alphabeta" else None
                    record_leaderboard_entry(
                        iteration,
                        gate_opponent,
                        depth_value,
                        gate_result,
                    )
        finally:
            agent_obj.config.mcts.simulations = sims_backup
            agent_obj.eval_stochastic = stochastic_backup
            if original_training_mode:
                agent_obj.net.train()

    agent = MuZeroAgent(config)
    if args.load_checkpoint is not None:
        agent.load_checkpoint(args.load_checkpoint, overwrite_config=True)

    print(
        f"[MuZero] seed={config.seed} device={agent.device.type} sims={config.mcts.simulations} iterations={iterations}"
    )

    agent.train_iterations(
        iterations=iterations,
        save_path=save_path,
        save_every=save_every,
        resume=args.resume,
        gating_callback=gating_callback,
    )
    agent.save_checkpoint(save_path)

    opponent_name = config.eval.opponent.lower()
    if opponent_name == "alphabeta":
        opponent = AlphaBetaAgent(depth=config.eval.opponent_depth)
    elif opponent_name == "random":
        opponent = RandomAgent(seed=123)
    elif opponent_name == "muzero":
        opponent = agent
    else:
        opponent = RandomAgent(seed=123)

    agent.eval_stochastic = False
    result = play_match(agent, opponent, games=config.eval.games, render=args.render)
    if result.illegal_moves:
        print(f"[eval] illegal_moves={result.illegal_moves}")

    agent.writer.flush()
    agent.writer.close()
    return result


def run_training(config: TrainConfig, *, render: bool = False) -> MatchResult:
    if config.algo == "random":
        agent = RandomAgent(seed=config.seed)
        baseline = RandomAgent(seed=(config.seed or 0) + 1)
        games = max(config.eval_games, 1)
        return play_match(agent, baseline, games=games, render=render)

    if config.algo == "ppo":
        agent = train_with_ppo(config)
        baseline = build_opponent(config.opponent, (config.seed or 0) + 5)
        return play_match(agent, baseline, games=config.eval_games, render=render)

    raise ValueError(f"Unsupported algo '{config.algo}'")


def parse_args(argv: Optional[Any] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Connect-Four agent")
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML config file path"
    )
    parser.add_argument(
        "--episodes", type=int, help="Number of episodes to run (random baseline)"
    )
    parser.add_argument(
        "--timesteps", type=int, help="Number of timesteps for RL training"
    )
    parser.add_argument(
        "--algo", choices=sorted(SUPPORTED_ALGOS), help="Training algorithm"
    )
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Print board states during evaluation matches",
    )
    parser.add_argument(
        "--tensorboard-log", type=Path, help="TensorBoard log directory"
    )
    parser.add_argument(
        "--save-checkpoint", type=Path, help="Where to save the trained model"
    )
    parser.add_argument(
        "--load-checkpoint", type=Path, help="Checkpoint to resume training from"
    )
    parser.add_argument(
        "--eval-games", type=int, help="Number of evaluation games after training"
    )
    parser.add_argument("--num-envs", type=int, help="Parallel env count for PPO")
    parser.add_argument(
        "--reward-win", type=float, help="Reward value for agent victories"
    )
    parser.add_argument(
        "--reward-loss", type=float, help="Reward value when the opponent wins"
    )
    parser.add_argument("--reward-draw", type=float, help="Reward value for draws")
    parser.add_argument(
        "--reward-step", type=float, help="Per-step reward shaping value"
    )
    parser.add_argument(
        "--invalid-move-penalty", type=float, help="Penalty applied on illegal actions"
    )
    parser.add_argument(
        "--swap-start-probability",
        type=float,
        help="Probability opponent opens when random starts enabled",
    )
    parser.add_argument(
        "--random-first-player",
        action=argparse.BooleanOptionalAction,
        help="Enable random opening swaps",
    )
    parser.add_argument(
        "--self-play",
        action=argparse.BooleanOptionalAction,
        help="Enable self-play via opponent snapshots",
    )
    parser.add_argument(
        "--self-play-update-interval",
        type=int,
        help="Timesteps between opponent snapshot updates",
    )
    parser.add_argument(
        "--self-play-pool-size",
        type=int,
        help="Maximum opponents to keep in the self-play pool",
    )
    parser.add_argument(
        "--self-play-warmup-steps",
        type=int,
        help="Timesteps before taking the first snapshot",
    )
    parser.add_argument(
        "--self-play-deterministic-snapshots",
        action=argparse.BooleanOptionalAction,
        help="Use deterministic actions for snapshot opponents",
    )
    parser.add_argument(
        "--gate-opponent",
        choices=["alphabeta", "random", "none"],
        default="alphabeta",
        help="Opponent to use for gating evaluations (set to 'none' to disable second gate run)",
    )
    parser.add_argument(
        "--gate-depth",
        type=int,
        default=5,
        help="Search depth for gate opponent when alphabeta is selected",
    )
    parser.add_argument(
        "--gate-games",
        type=int,
        default=50,
        help="Number of games per gating evaluation batch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training without resetting timesteps",
    )
    parser.add_argument(
        "--iters", type=int, help="Number of training iterations (MuZero)"
    )
    parser.add_argument(
        "--mcts-simulations", type=int, help="Override MuZero MCTS simulations"
    )
    parser.add_argument(
        "--temperature-moves",
        type=int,
        help="Moves using temperature sampling for MuZero",
    )
    parser.add_argument(
        "--device", type=str, help="Device override for MuZero (auto|cpu|mps|cuda)"
    )
    parser.add_argument(
        "--save-every", type=int, help="Save checkpoint every N iterations (MuZero)"
    )
    parser.add_argument(
        "--num-workers", type=int, help="Number of self-play workers (MuZero)"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Any] = None) -> None:
    args = parse_args(argv)
    algo_from_config, config_obj = load_any_config(args.config)
    algo = args.algo or algo_from_config

    if algo == "muzero":
        if not isinstance(config_obj, MuZeroConfig):
            config_obj = load_muzero_config(args.config)
        if args.tensorboard_log is not None:
            config_obj.logging.tensorboard = str(args.tensorboard_log)
        if args.save_checkpoint is not None:
            config_obj.logging.ckpt_dir = str(Path(args.save_checkpoint).parent)
        result = run_muzero_training(config_obj, args)
        output = {
            "algo": "muzero",
            "config": asdict(config_obj),
            "result": {
                "agent_one_wins": result.agent_one_wins,
                "agent_two_wins": result.agent_two_wins,
                "draws": result.draws,
                "total_games": result.total_games,
            },
        }
        print(json.dumps(output, indent=2))
        return

    if not isinstance(config_obj, TrainConfig):
        raise ValueError("Expected PPO/random config for non-MuZero training")
    config = config_obj
    if args.episodes is not None:
        config.episodes = args.episodes
    if args.timesteps is not None:
        config.timesteps = args.timesteps
    if args.algo is not None:
        config.algo = args.algo
    if args.seed is not None:
        config.seed = args.seed
    if args.tensorboard_log is not None:
        config.tensorboard_log = args.tensorboard_log
    if args.save_checkpoint is not None:
        config.save_path = args.save_checkpoint
    if args.load_checkpoint is not None:
        config.load_path = args.load_checkpoint
    if args.eval_games is not None:
        config.eval_games = args.eval_games
    if args.num_envs is not None:
        config.num_envs = args.num_envs
    if args.reward_win is not None:
        config.reward_win = args.reward_win
    if args.reward_loss is not None:
        config.reward_loss = args.reward_loss
    if args.reward_draw is not None:
        config.reward_draw = args.reward_draw
    if args.reward_step is not None:
        config.reward_step = args.reward_step
    if args.invalid_move_penalty is not None:
        config.invalid_move_penalty = args.invalid_move_penalty
    if args.swap_start_probability is not None:
        config.swap_start_probability = max(0.0, min(1.0, args.swap_start_probability))
    if args.random_first_player is not None:
        config.random_first_player = args.random_first_player
    if args.self_play is not None:
        config.self_play = args.self_play
    if args.self_play_update_interval is not None:
        config.self_play_update_interval = args.self_play_update_interval
    if args.self_play_pool_size is not None:
        config.self_play_pool_size = args.self_play_pool_size
    if args.self_play_warmup_steps is not None:
        config.self_play_warmup_steps = args.self_play_warmup_steps
    if args.self_play_deterministic_snapshots is not None:
        config.self_play_deterministic_snapshots = (
            args.self_play_deterministic_snapshots
        )

    config.reset_num_timesteps = not args.resume

    summary = run_training(config, render=args.render)
    output = {
        "config": {
            "episodes": config.episodes,
            "timesteps": config.timesteps,
            "algo": config.algo,
            "eval_games": config.eval_games,
            "num_envs": config.num_envs,
            "seed": config.seed,
            "opponent": config.opponent,
            "tensorboard_log": (
                str(config.tensorboard_log) if config.tensorboard_log else None
            ),
            "save_path": str(config.save_path) if config.save_path else None,
            "load_path": str(config.load_path) if config.load_path else None,
            "reward_win": config.reward_win,
            "reward_loss": config.reward_loss,
            "reward_draw": config.reward_draw,
            "reward_step": config.reward_step,
            "invalid_move_penalty": config.invalid_move_penalty,
            "random_first_player": config.random_first_player,
            "swap_start_probability": config.swap_start_probability,
            "self_play": config.self_play,
            "self_play_update_interval": config.self_play_update_interval,
            "self_play_pool_size": config.self_play_pool_size,
            "self_play_warmup_steps": config.self_play_warmup_steps,
            "self_play_deterministic_snapshots": config.self_play_deterministic_snapshots,
        },
        "result": {
            "agent_one_wins": summary.agent_one_wins,
            "agent_two_wins": summary.agent_two_wins,
            "draws": summary.draws,
            "total_games": summary.total_games,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
