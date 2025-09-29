"""Training entry point for Connect-Four agents."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.random_policy import RandomAgent
from src.agents.sb3_agent import SB3Agent
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
SUPPORTED_ALGOS = {"random", "ppo"}


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
            swap_start_probability=float(payload.get("swap_start_probability", DEFAULT_SWAP_START_PROB)),
            reward_win=float(rewards.get("win", DEFAULT_REWARD_WIN)),
            reward_draw=float(rewards.get("draw", DEFAULT_REWARD_DRAW)),
            reward_loss=float(rewards.get("loss", DEFAULT_REWARD_LOSS)),
            invalid_move_penalty=float(rewards.get("invalid_move", DEFAULT_INVALID_MOVE)),
            reward_step=float(rewards.get("step", DEFAULT_REWARD_STEP)),
        )


def load_config(path: Path) -> TrainConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a mapping at the top level")
    config = TrainConfig.from_dict(payload)
    if config.algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported algo '{config.algo}'. Supported: {sorted(SUPPORTED_ALGOS)}")
    return config


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
            tensorboard_log=str(config.tensorboard_log) if config.tensorboard_log else None,
        )

    model.learn(total_timesteps=config.timesteps, reset_num_timesteps=config.reset_num_timesteps)
    if config.save_path:
        config.save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(config.save_path))
    return SB3Agent(model=model, deterministic=True, name="ppo-agent")


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
    parser.add_argument("--config", type=Path, required=True, help="YAML config file path")
    parser.add_argument("--episodes", type=int, help="Number of episodes to run (random baseline)")
    parser.add_argument("--timesteps", type=int, help="Number of timesteps for RL training")
    parser.add_argument("--algo", choices=sorted(SUPPORTED_ALGOS), help="Training algorithm")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--render", action="store_true", help="Print board states during evaluation matches")
    parser.add_argument("--tensorboard-log", type=Path, help="TensorBoard log directory")
    parser.add_argument("--save-checkpoint", type=Path, help="Where to save the trained model")
    parser.add_argument("--load-checkpoint", type=Path, help="Checkpoint to resume training from")
    parser.add_argument("--eval-games", type=int, help="Number of evaluation games after training")
    parser.add_argument("--num-envs", type=int, help="Parallel env count for PPO")
    parser.add_argument("--reward-win", type=float, help="Reward value for agent victories")
    parser.add_argument("--reward-loss", type=float, help="Reward value when the opponent wins")
    parser.add_argument("--reward-draw", type=float, help="Reward value for draws")
    parser.add_argument("--reward-step", type=float, help="Per-step reward shaping value")
    parser.add_argument("--invalid-move-penalty", type=float, help="Penalty applied on illegal actions")
    parser.add_argument("--swap-start-probability", type=float, help="Probability opponent opens when random starts enabled")
    parser.add_argument("--random-first-player", action=argparse.BooleanOptionalAction, help="Enable random opening swaps")
    parser.add_argument("--resume", action="store_true", help="Continue training without resetting timesteps")
    return parser.parse_args(argv)


def main(argv: Optional[Any] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
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
            "tensorboard_log": str(config.tensorboard_log) if config.tensorboard_log else None,
            "save_path": str(config.save_path) if config.save_path else None,
            "load_path": str(config.load_path) if config.load_path else None,
            "reward_win": config.reward_win,
            "reward_loss": config.reward_loss,
            "reward_draw": config.reward_draw,
            "reward_step": config.reward_step,
            "invalid_move_penalty": config.invalid_move_penalty,
            "random_first_player": config.random_first_player,
            "swap_start_probability": config.swap_start_probability,
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
