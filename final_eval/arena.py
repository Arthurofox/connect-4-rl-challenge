from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import Any, Dict

from src.board import ConnectFourBoard, InvalidMoveError
from src.agents.alphabeta import AlphaBetaAgent
from src.agents.muzero.agent import MuZeroAgent, resolve_device

from .dqn_agent import DQNAgent


AgentLike = Any


def _load_muzero(checkpoint: str | Path, device: str) -> MuZeroAgent:
    if not checkpoint:
        raise ValueError("MuZero checkpoint path is required")
    resolved = resolve_device(device)
    loader = getattr(MuZeroAgent, "load_from_checkpoint", None)
    if callable(loader):
        agent = loader(checkpoint, device=resolved)
    else:
        agent = MuZeroAgent.from_checkpoint(Path(checkpoint), device=resolved)
    agent.eval_stochastic = False
    if hasattr(agent, "debug_mcts"):
        agent.debug_mcts = False
    return agent


def _load_dqn(checkpoint: str | Path, device: str) -> DQNAgent:
    if not checkpoint:
        raise ValueError("DQN checkpoint path is required")
    return DQNAgent.load_from_checkpoint(checkpoint, device=device)


def _load_alphabeta(depth: int | None) -> AlphaBetaAgent:
    return AlphaBetaAgent(depth=depth or 7)


def make_agent(kind: str, *, checkpoint: str | None, depth: int | None, device: str) -> AgentLike:
    if kind == "muzero":
        return _load_muzero(checkpoint or "", device)
    if kind == "dqn":
        return _load_dqn(checkpoint or "", device)
    if kind == "alphabeta":
        return _load_alphabeta(depth)
    raise ValueError(f"Unsupported agent kind '{kind}'")


def _accepts_kw(agent: AgentLike, key: str) -> bool:
    from inspect import signature

    try:
        sig = signature(agent.select_action)
    except (TypeError, ValueError):
        return False
    return key in sig.parameters


def _call_agent(agent: AgentLike, board: ConnectFourBoard, sims: int | None) -> int:
    name = getattr(agent, "name", agent.__class__.__name__).lower()
    if name == "muzero":
        kwargs: Dict[str, Any] = {"training": False}
        if sims is not None:
            if _accepts_kw(agent, "mcts_simulations"):
                kwargs["mcts_simulations"] = sims
            else:
                kwargs["simulations"] = sims
        return agent.select_action(board, **kwargs)
    try:
        return agent.select_action(board, training=False)
    except TypeError:
        return agent.select_action(board)


def fight(
    agent1: AgentLike,
    agent2: AgentLike,
    *,
    games: int = 200,
    swap_first: bool = True,
    sims1: int | None = 160,
    sims2: int | None = 160,
) -> Dict[str, int]:
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0

    for index in range(games):
        board = ConnectFourBoard()
        agent1_starts = not swap_first or index % 2 == 0
        slots = (agent1, agent2) if agent1_starts else (agent2, agent1)
        token_owner: Dict[int, AgentLike] = {1: slots[0], 2: slots[1]}

        while True:
            player = board.current_player
            actor = slots[player - 1]
            sims = sims1 if id(actor) == id(agent1) else sims2
            try:
                move = _call_agent(actor, board, sims)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Agent '{actor}' failed to produce a move") from exc
            try:
                result = board.drop(move)
            except InvalidMoveError:
                if id(actor) == id(agent1):
                    wins_agent2 += 1
                else:
                    wins_agent1 += 1
                break

            if result.winner is not None:
                winner_agent = token_owner[result.winner]
                if id(winner_agent) == id(agent1):
                    wins_agent1 += 1
                else:
                    wins_agent2 += 1
                break

            if result.board_full:
                draws += 1
                break

    return {
        "wins_agent1": wins_agent1,
        "wins_agent2": wins_agent2,
        "draws": draws,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless evaluation arena")
    parser.add_argument("--agent1", choices=["muzero", "dqn", "alphabeta"], required=True)
    parser.add_argument("--agent2", choices=["muzero", "dqn", "alphabeta"], required=True)
    parser.add_argument("--ckpt1", type=str)
    parser.add_argument("--ckpt2", type=str)
    parser.add_argument("--depth1", type=int)
    parser.add_argument("--depth2", type=int)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument(
        "--swap-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Swap which agent moves first each game",
    )
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--sims1", type=int, default=160)
    parser.add_argument("--sims2", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    agent1 = make_agent(
        args.agent1,
        checkpoint=args.ckpt1,
        depth=args.depth1,
        device=args.device,
    )
    agent2 = make_agent(
        args.agent2,
        checkpoint=args.ckpt2,
        depth=args.depth2,
        device=args.device,
    )

    summary = fight(
        agent1,
        agent2,
        games=args.games,
        swap_first=bool(args.swap_first),
        sims1=args.sims1 if args.sims1 >= 0 else None,
        sims2=args.sims2 if args.sims2 >= 0 else None,
    )

    total_games = args.games
    payload: Dict[str, Any] = {
        "agent1": args.agent1,
        "agent2": args.agent2,
        "games": total_games,
        "swap_first": bool(args.swap_first),
        "wins_agent1": summary["wins_agent1"],
        "wins_agent2": summary["wins_agent2"],
        "draws": summary["draws"],
        "win_rate_agent1": summary["wins_agent1"] / total_games if total_games else 0.0,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
