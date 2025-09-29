"""CLI to evaluate a frozen agent checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from src.agents.random_policy import RandomAgent
from src.agents.sb3_agent import SB3Agent
from src.match import play_match

SUPPORTED_ALGOS = ["random", "ppo"]


def parse_args(argv: Optional[Any] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Connect-Four agent")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint to evaluate",
    )
    parser.add_argument(
        "--algo",
        choices=SUPPORTED_ALGOS,
        default="ppo",
        help="Algorithm used to train the checkpoint",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of evaluation games to run",
    )
    parser.add_argument(
        "--baseline",
        choices=["random"],
        default="random",
        help="Baseline opponent to compare against",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Print board states for each evaluation game",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    return parser.parse_args(argv)


def build_agent(algo: str, checkpoint: Path, deterministic: bool):
    if algo == "random":
        return RandomAgent()
    if algo == "ppo":
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return SB3Agent.load(checkpoint, algorithm="ppo", deterministic=deterministic)
    raise ValueError(f"Unsupported algo '{algo}'")


def main(argv: Optional[Any] = None) -> None:
    args = parse_args(argv)

    deterministic = not args.stochastic
    agent = build_agent(args.algo, args.checkpoint, deterministic)
    baseline = RandomAgent(seed=42)

    result = play_match(agent, baseline, games=args.games, render=args.render)

    output = {
        "checkpoint": str(args.checkpoint),
        "algo": args.algo,
        "games": args.games,
        "agent_one_wins": result.agent_one_wins,
        "agent_two_wins": result.agent_two_wins,
        "draws": result.draws,
        "win_rate_agent_one": result.win_rate_agent_one,
        "deterministic": deterministic,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
