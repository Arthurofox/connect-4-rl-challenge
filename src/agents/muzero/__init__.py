"""MuZero agent exports and configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from .agent import MuZeroAgent, resolve_device


@dataclass
class SelfPlayConfig:
    games_per_iter: int
    max_moves: int
    temperature_moves: int
    dirichlet_alpha: float
    dirichlet_eps: float
    mirror_openings: bool

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SelfPlayConfig":
        return cls(
            games_per_iter=int(payload.get("games_per_iter", 200)),
            max_moves=int(payload.get("max_moves", 42)),
            temperature_moves=int(payload.get("temperature_moves", 6)),
            dirichlet_alpha=float(payload.get("dirichlet_alpha", 0.3)),
            dirichlet_eps=float(payload.get("dirichlet_eps", 0.25)),
            mirror_openings=bool(payload.get("mirror_openings", False)),
        )


@dataclass
class MCTSConfig:
    simulations: int
    c_puct: float
    value_discount: float

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MCTSConfig":
        return cls(
            simulations=int(payload.get("simulations", 200)),
            c_puct=float(payload.get("c_puct", 2.5)),
            value_discount=float(payload.get("value_discount", 1.0)),
        )


@dataclass
class ModelConfig:
    channels: int
    res_blocks: int
    latent_dim: int
    unroll_steps: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelConfig":
        return cls(
            channels=int(payload.get("channels", 64)),
            res_blocks=int(payload.get("res_blocks", 6)),
            latent_dim=int(payload.get("latent_dim", 128)),
            unroll_steps=int(payload.get("unroll_steps", 4)),
        )


@dataclass
class TrainHyperParams:
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    epochs_per_iter: int
    replay_capacity: int
    warm_start_steps: int
    amp: bool
    policy_weight: float
    value_weight: float
    reward_weight: float

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainHyperParams":
        return cls(
            batch_size=int(payload.get("batch_size", 256)),
            lr=float(payload.get("lr", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            grad_clip=float(payload.get("grad_clip", 1.0)),
            epochs_per_iter=int(payload.get("epochs_per_iter", 200)),
            replay_capacity=int(payload.get("replay_capacity", 100_000)),
            warm_start_steps=int(payload.get("warm_start_steps", 5000)),
            amp=bool(payload.get("amp", True)),
            policy_weight=float(payload.get("policy_weight", 1.0)),
            value_weight=float(payload.get("value_weight", 1.0)),
            reward_weight=float(payload.get("reward_weight", 0.5)),
        )


@dataclass
class LoggingConfig:
    tensorboard: str
    ckpt_dir: str
    save_every_iters: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LoggingConfig":
        return cls(
            tensorboard=str(payload.get("tensorboard", "artifacts/tensorboard/muzero")),
            ckpt_dir=str(payload.get("ckpt_dir", "artifacts/released")),
            save_every_iters=int(payload.get("save_every_iters", 5)),
        )


@dataclass
class EvalConfig:
    games: int
    opponent: str
    opponent_depth: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvalConfig":
        return cls(
            games=int(payload.get("games", 200)),
            opponent=str(payload.get("opponent", "alphabeta")),
            opponent_depth=int(payload.get("opponent_depth", 9)),
        )


@dataclass
class MuZeroConfig:
    algo: str
    seed: int
    device: str
    self_play: SelfPlayConfig
    mcts: MCTSConfig
    model: ModelConfig
    train: TrainHyperParams
    logging: LoggingConfig
    eval: EvalConfig

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MuZeroConfig":
        return cls(
            algo=str(payload.get("algo", "muzero")),
            seed=int(payload.get("seed", 42)),
            device=str(payload.get("device", "auto")),
            self_play=SelfPlayConfig.from_dict(payload.get("self_play", {})),
            mcts=MCTSConfig.from_dict(payload.get("mcts", {})),
            model=ModelConfig.from_dict(payload.get("model", {})),
            train=TrainHyperParams.from_dict(payload.get("train", {})),
            logging=LoggingConfig.from_dict(payload.get("logging", {})),
            eval=EvalConfig.from_dict(payload.get("eval", {})),
        )


def load_config(path: Path) -> MuZeroConfig:
    if not path.exists():
        raise FileNotFoundError(f"MuZero config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("MuZero config root must be a mapping")
    return MuZeroConfig.from_dict(data)


__all__ = [
    "MuZeroAgent",
    "MuZeroConfig",
    "SelfPlayConfig",
    "MCTSConfig",
    "ModelConfig",
    "TrainHyperParams",
    "LoggingConfig",
    "EvalConfig",
    "resolve_device",
    "load_config",
]
