"""Utilities to run head-to-head Connect-Four matches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from src.agents.base import Agent
from src.board import ConnectFourBoard


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


def play_single_game(agent_one: Agent, agent_two: Agent, board: Optional[ConnectFourBoard] = None) -> GameLog:
    board = board or ConnectFourBoard()
    board.reset()
    agent_one.on_game_start(board)
    agent_two.on_game_start(board)

    agents: Tuple[Agent, Agent] = (agent_one, agent_two)
    moves: List[int] = []

    while True:
        current_agent = agents[(board.current_player - 1) % 2]
        move = current_agent.select_action(board)
        result = board.drop(move)
        moves.append(move)
        if result.winner is not None or result.board_full:
            agent_one.on_game_end(board, result.winner)
            agent_two.on_game_end(board, result.winner)
            return GameLog(moves=moves, winner=result.winner)


def play_match(
    agent_one: Agent,
    agent_two: Agent,
    games: int,
    swap_start: bool = True,
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

    for first, second in order:
        log = play_single_game(first, second, board)
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


__all__ = ["GameLog", "MatchResult", "play_match", "play_single_game"]
