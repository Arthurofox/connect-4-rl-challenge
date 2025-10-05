"""CLI to evaluate a frozen agent checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.agents.random_policy import RandomAgent
from src.agents.sb3_agent import SB3Agent
from src.agents.alphabeta import AlphaBetaAgent
from src.agents.muzero import MuZeroAgent, resolve_device
from src.match import play_match

SUPPORTED_ALGOS = ["random", "ppo", "muzero", "alphabeta"]
SUPPORTED_OPPONENTS = ["random", "alphabeta", "muzero", "dqn"]


def parse_args(argv: Optional[Any] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Connect-Four agent")
    parser.add_argument(
        "--checkpoint",
        type=Path,
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
        "--opponent",
        choices=SUPPORTED_OPPONENTS,
        default="alphabeta",
        help="Opponent to evaluate against",
    )
    parser.add_argument(
        "--opponent-depth",
        type=int,
        default=9,
        help="Depth for alpha-beta opponent",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        help="Checkpoint to load when the opponent is DQN or MuZero",
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
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        help="Override MuZero MCTS simulations during evaluation",
    )
    parser.add_argument(
        "--temperature-moves",
        type=int,
        help="Moves using temperature-based sampling for MuZero",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for MuZero evaluation",
    )
    parser.add_argument(
        "--debug-mcts",
        action="store_true",
        help="Dump root-level MCTS diagnostics during evaluation",
    )
    parser.add_argument(
        "--trace-moves",
        action="store_true",
        help="Log each move as JSON during evaluation",
    )
    parser.add_argument(
        "--debug-opponent",
        action="store_true",
        help="Print opponent wiring information before matches",
    )
    parser.add_argument(
        "--swap-first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alternate which player moves first across games",
    )
    parser.add_argument(
        "--start-player",
        choices=["agent", "opponent"],
        default="agent",
        help="Player to start when swap-first is disabled",
    )
    parser.add_argument(
        "--agent-depth",
        type=int,
        default=7,
        help="Depth for AlphaBeta when used as the main evaluated agent",
    )
    return parser.parse_args(argv)


def build_agent(args: argparse.Namespace):
    if args.algo == "random":
        return RandomAgent()
    if args.algo == "ppo":
        if args.checkpoint is None or not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        return SB3Agent.load(
            args.checkpoint, algorithm="ppo", deterministic=not args.stochastic
        )
    if args.algo == "muzero":
        if args.checkpoint is None or not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        device = resolve_device(args.device)
        agent = MuZeroAgent.from_checkpoint(args.checkpoint, device=device)
        agent.eval_stochastic = args.stochastic
        if args.mcts_simulations is not None:
            agent.config.mcts.simulations = args.mcts_simulations
        if args.temperature_moves is not None:
            agent.policy_temperature_moves = args.temperature_moves
        agent.debug_mcts = args.debug_mcts
        return agent
    if args.algo == "alphabeta":
        return AlphaBetaAgent(depth=args.agent_depth)
    raise ValueError(f"Unsupported algo '{args.algo}'")


def build_opponent(args: argparse.Namespace):
    if args.opponent == "random":
        return RandomAgent(seed=123)
    if args.opponent == "alphabeta":
        return AlphaBetaAgent(depth=args.opponent_depth)
    if args.opponent == "muzero":
        checkpoint = args.opponent_checkpoint or args.checkpoint
        if checkpoint is None:
            raise FileNotFoundError("Opponent MuZero requires a valid checkpoint path")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Opponent MuZero checkpoint not found: {checkpoint_path}")
        device = resolve_device(args.device)
        opponent = MuZeroAgent.from_checkpoint(checkpoint_path, device=device)
        opponent.eval_stochastic = False
        if hasattr(opponent, "debug_mcts"):
            opponent.debug_mcts = False
        return opponent
    if args.opponent == "dqn":
        if args.opponent_checkpoint is None or not args.opponent_checkpoint.exists():
            raise FileNotFoundError("DQN opponent requires --opponent-checkpoint")
        from final_eval.dqn_agent import DQNAgent

        device = resolve_device(args.device)
        return DQNAgent.load_from_checkpoint(args.opponent_checkpoint, device=str(device))
    raise ValueError(f"Unsupported opponent '{args.opponent}'")


def main(argv: Optional[Any] = None) -> None:
    args = parse_args(argv)

    agent = build_agent(args)
    opponent = build_opponent(args)
    assert agent is not opponent, "Agent and opponent must be distinct objects"

    swap_first = bool(args.swap_first)
    start_player = "agent" if swap_first else args.start_player

    def describe(policy: Any) -> str:
        name = getattr(policy, "name", policy.__class__.__name__)
        if name == "muzero":
            sims_val = args.mcts_simulations
            if sims_val is None:
                sims_val = getattr(policy, "sims", None)
            return f"MuZero(sims={sims_val})"
        if name == "dqn":
            return "DQN"
        depth_val = getattr(policy, "depth", None)
        if depth_val is not None:
            return f"AlphaBeta(depth={depth_val})"
        return name.capitalize()

    if args.debug_opponent:
        print(f"[Eval] agent={describe(agent)}; opponent={describe(opponent)}")
        print(f"[Eval] swap_first={swap_first} start_player={start_player}")

    agent_sims = args.mcts_simulations
    if getattr(agent, "name", None) == "muzero" and agent_sims is None:
        agent_sims = getattr(agent, "sims", None)

    actor_metadata = {
        "agent": {
            "algo": getattr(agent, "name", "agent"),
            "depth": getattr(agent, "depth", None),
            "sims": agent_sims,
            "stochastic": args.stochastic,
        },
        "opponent": {
            "algo": getattr(opponent, "name", "opponent"),
            "depth": getattr(opponent, "depth", None),
            "sims": getattr(opponent, "sims", None),
            "stochastic": False,
        },
    }
    if actor_metadata["agent"]["sims"] is None:
        actor_metadata["agent"]["sims"] = 0
    if actor_metadata["opponent"]["sims"] is None:
        actor_metadata["opponent"]["sims"] = 0

    trace_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    if args.trace_moves:
        def trace_logger(entry: Dict[str, Any]) -> None:
            print(json.dumps(entry))

        trace_callback = trace_logger

    result = play_match(
        agent,
        opponent,
        games=args.games,
        swap_start=swap_first,
        render=args.render,
        start_player=start_player,
        trace_fn=trace_callback,
        actor_metadata=actor_metadata,
    )
    if result.illegal_moves:
        print(f"[eval] illegal_moves={result.illegal_moves}")

    output = {
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "algo": args.algo,
        "games": args.games,
        "opponent": args.opponent,
        "opponent_name": getattr(opponent, "name", args.opponent),
        "opponent_depth": getattr(opponent, "depth", None),
        "agent_one_wins": result.agent_one_wins,
        "agent_two_wins": result.agent_two_wins,
        "draws": result.draws,
        "win_rate_agent_one": result.win_rate_agent_one,
        "stochastic": args.stochastic,
        "debug_mcts": args.debug_mcts,
        "swap_first": swap_first,
        "start_player": start_player,
        "mcts_simulations": agent_sims,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
