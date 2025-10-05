"""Utilities to run head-to-head Connect-Four matches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.agents.base import Agent
from src.board import ConnectFourBoard, InvalidMoveError


RenderCallback = Callable[[ConnectFourBoard, int, int, int], None]
TraceCallback = Callable[[Dict[str, Any]], None]


@dataclass
class GameLog:
    moves: List[int]
    winner: Optional[int]
    illegal_move: bool = False


@dataclass
class MatchResult:
    agent_one_wins: int
    agent_two_wins: int
    draws: int
    games: List[GameLog]
    illegal_moves: int = 0

    @property
    def total_games(self) -> int:
        return self.agent_one_wins + self.agent_two_wins + self.draws

    @property
    def win_rate_agent_one(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.agent_one_wins / self.total_games


def play_single_game(
    agent_one: Agent,
    agent_two: Agent,
    board: Optional[ConnectFourBoard] = None,
    *,
    render: bool = False,
    render_fn: Optional[RenderCallback] = None,
    labels: Tuple[str, str] = ("agent_one", "agent_two"),
    trace_cb: Optional[TraceCallback] = None,
    actor_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> GameLog:
    board = board or ConnectFourBoard()
    board.reset()
    agent_one.on_game_start(board)
    agent_two.on_game_start(board)

    if render:
        print("== New Game ==")
        print(board.render())
        print()

    agents: Tuple[Agent, Agent] = (agent_one, agent_two)
    moves: List[int] = []
    illegal = False
    meta = actor_metadata or {}

    def emit(move_index: int, player: int, column: int) -> None:
        if not render:
            return
        if render_fn is not None:
            render_fn(board, move_index, player, column)
            return
        print(f"Move {move_index}: Player {player} -> column {column}")
        print(board.render())
        print()

    while True:
        player = board.current_player
        index = (player - 1) % 2
        current_agent = agents[index]
        label = labels[index]
        info = meta.get(label, {})
        if getattr(current_agent, "name", None) == "muzero":
            sims_override = info.get("sims") if info else None
            if sims_override is None:
                sims_override = getattr(current_agent, "sims", None)
            stochastic = info.get("stochastic", False) if info else False
            move = current_agent.select_action(
                board,
                training=False,
                stochastic=stochastic,
                simulations=sims_override,
            )
        else:
            try:
                move = current_agent.select_action(board)
            except TypeError:
                move = current_agent.select_action(board, training=False)
        try:
            result = board.drop(move)
        except InvalidMoveError:
            moves.append(move)
            illegal = True
            winner = 3 - player
            if render:
                print(f"Invalid move: Player {player} -> column {move}")
                print(board.render())
                print()
                print(f"Winner: Player {winner}")
                print()
            agent_one.on_game_end(board, winner)
            agent_two.on_game_end(board, winner)
            if trace_cb is not None:
                trace_cb(
                    {
                        "turn": len(moves) - 1,
                        "actor": label,
                        "algo": info.get("algo", getattr(current_agent, "name", label)),
                        "depth": info.get("depth"),
                        "sims": info.get("sims"),
                        "move": move,
                    }
                )
            return GameLog(moves=moves, winner=winner, illegal_move=True)

        moves.append(move)
        emit(len(moves), player, move)
        if trace_cb is not None:
            trace_cb(
                {
                    "turn": len(moves) - 1,
                    "actor": label,
                    "algo": info.get("algo", getattr(current_agent, "name", label)),
                    "depth": info.get("depth"),
                    "sims": info.get("sims"),
                    "move": move,
                }
            )
        if result.winner is not None or result.board_full:
            agent_one.on_game_end(board, result.winner)
            agent_two.on_game_end(board, result.winner)
            if render:
                if result.winner is not None:
                    print(f"Winner: Player {result.winner}")
                else:
                    print("Result: Draw")
                print()
            return GameLog(moves=moves, winner=result.winner, illegal_move=illegal)


def play_match(
    agent_one: Agent,
    agent_two: Agent,
    games: int,
    swap_start: bool = True,
    *,
    render: bool = False,
    render_fn: Optional[RenderCallback] = None,
    start_player: str = "agent",
    trace_fn: Optional[TraceCallback] = None,
    actor_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> MatchResult:
    board = ConnectFourBoard()
    agent_one_wins = 0
    agent_two_wins = 0
    draws = 0
    illegal_moves = 0
    logs: List[GameLog] = []

    start = start_player.lower()
    if start not in {"agent", "opponent"}:
        raise ValueError("start_player must be 'agent' or 'opponent'")

    def label_pair(agent_first: bool) -> Tuple[str, str]:
        return ("agent", "opponent") if agent_first else ("opponent", "agent")

    if swap_start:
        order: Iterable[Tuple[Agent, Agent, Tuple[str, str]]] = [
            (agent_one, agent_two, label_pair(True))
            if i % 2 == 0
            else (agent_two, agent_one, label_pair(False))
            for i in range(games)
        ]
    else:
        agent_first = start == "agent"
        order = [
            (agent_one, agent_two, label_pair(True))
            if agent_first
            else (agent_two, agent_one, label_pair(False))
        ] * games

    for index, (first, second, labels) in enumerate(order, start=1):
        if render and swap_start:
            print(f"=== Game {index} ===")
        log = play_single_game(
            first,
            second,
            board,
            render=render,
            render_fn=render_fn,
            labels=labels,
            trace_cb=trace_fn,
            actor_metadata=actor_metadata,
        )
        logs.append(log)
        if log.illegal_move:
            illegal_moves += 1
        if log.winner == 1:
            winner_agent = first
        elif log.winner == 2:
            winner_agent = second
        else:
            winner_agent = None

        if winner_agent is agent_one:
            agent_one_wins += 1
        elif winner_agent is agent_two:
            agent_two_wins += 1
        else:
            draws += 1

    return MatchResult(
        agent_one_wins=agent_one_wins,
        agent_two_wins=agent_two_wins,
        draws=draws,
        games=logs,
        illegal_moves=illegal_moves,
    )


__all__ = [
    "GameLog",
    "MatchResult",
    "RenderCallback",
    "TraceCallback",
    "play_match",
    "play_single_game",
]
