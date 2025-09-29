"""CLI to evaluate a frozen agent checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from src.agents.random_policy import RandomAgent
from src.match import play_match


def parse_args(argv: Optional[Any] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Connect-Four agent")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint to evaluate",
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
    return parser.parse_args(argv)


def main(argv: Optional[Any] = None) -> None:
    args = parse_args(argv)

    # Placeholder: load the target agent once training checkpoints are defined.
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    target_agent = RandomAgent()
    baseline = RandomAgent(seed=42)
    result = play_match(target_agent, baseline, games=args.games)

    output = {
        "checkpoint": str(args.checkpoint),
        "games": args.games,
        "agent_one_wins": result.agent_one_wins,
        "agent_two_wins": result.agent_two_wins,
        "draws": result.draws,
        "win_rate_agent_one": result.win_rate_agent_one,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
