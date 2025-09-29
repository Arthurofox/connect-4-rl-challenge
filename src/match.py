"""Utilities to run head-to-head Connect-Four matches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

from src.agents.base import Agent
from src.board import ConnectFourBoard


RenderCallback = Callable[[ConnectFourBoard, int, int, int], None]


@dataclass
class GameLog:
    moves: List[int]
    winner: Optional[int]


@dataclass
class MatchResult:
    agent_one_wins: int
    agent_two_wins: int
    draws: int
    games: List[GameLog]

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
        current_agent = agents[(player - 1) % 2]
        move = current_agent.select_action(board)
        result = board.drop(move)
        moves.append(move)
        emit(len(moves), player, move)
        if result.winner is not None or result.board_full:
            agent_one.on_game_end(board, result.winner)
            agent_two.on_game_end(board, result.winner)
            if render:
                if result.winner is not None:
                    print(f"Winner: Player {result.winner}")
                else:
                    print("Result: Draw")
                print()
            return GameLog(moves=moves, winner=result.winner)


def play_match(
    agent_one: Agent,
    agent_two: Agent,
    games: int,
    swap_start: bool = True,
    *,
    render: bool = False,
    render_fn: Optional[RenderCallback] = None,
) -> MatchResult:
    board = ConnectFourBoard()
    agent_one_wins = 0
    agent_two_wins = 0
    draws = 0
    logs: List[GameLog] = []

    if swap_start:
        order: Iterable[Tuple[Agent, Agent]] = [
            (agent_one, agent_two) if i % 2 == 0 else (agent_two, agent_one) for i in range(games)
        ]
    else:
        order = [(agent_one, agent_two)] * games

    for index, (first, second) in enumerate(order, start=1):
        if render and swap_start:
            print(f"=== Game {index} ===")
        log = play_single_game(
            first,
            second,
            board,
            render=render,
            render_fn=render_fn,
        )
        logs.append(log)
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
    )


__all__ = ["GameLog", "MatchResult", "RenderCallback", "play_match", "play_single_game"]
