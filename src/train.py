"""Training entry point for Connect-Four agents."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.agents.random_policy import RandomAgent
from src.match import MatchResult, play_match

DEFAULT_EPISODES = 1000
DEFAULT_OPPONENT = "random"


@dataclass
class TrainConfig:
    episodes: int = DEFAULT_EPISODES
    seed: Optional[int] = None
    opponent: str = DEFAULT_OPPONENT

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainConfig":
        return cls(
            episodes=int(payload.get("episodes", DEFAULT_EPISODES)),
            seed=payload.get("seed"),
            opponent=str(payload.get("opponent", DEFAULT_OPPONENT)),
        )


def load_config(path: Path) -> TrainConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a mapping at the top level")
    return TrainConfig.from_dict(payload)


def run_training(config: TrainConfig) -> MatchResult:
    """Placeholder training loop: plays self-play matches for evaluation."""
    agent = RandomAgent(seed=config.seed)
    baseline = RandomAgent(seed=(config.seed or 0) + 1)
    games = max(config.episodes // 10, 1)
    return play_match(agent, baseline, games=games)


def parse_args(argv: Optional[Any] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Connect-Four agent")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file path")
    parser.add_argument("--episodes", type=int, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, help="Random seed override")
    return parser.parse_args(argv)


def main(argv: Optional[Any] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.episodes is not None:
        config.episodes = args.episodes
    if args.seed is not None:
        config.seed = args.seed

    summary = run_training(config)
    output = {
        "config": config.__dict__,
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
