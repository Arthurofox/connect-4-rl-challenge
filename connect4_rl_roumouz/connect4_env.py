import numpy as np
from typing import Optional, Tuple, Dict, Any

ROWS, COLS = 6, 7
CONNECT_N = 4

class Connect4Env:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.player = 1
        self.last_move: Optional[Tuple[int,int]] = None
        self.moves_played = 0

    def reset(self, starting_player: int = 1) -> Dict[str, Any]:
        self.board[:] = 0
        self.player = 1 if starting_player == 1 else -1
        self.last_move = None
        self.moves_played = 0
        return self._obs()

    def valid_actions(self) -> np.ndarray:
        return (self.board[0] == 0).astype(np.float32)

    def step(self, action: int):
        if action < 0 or action >= COLS or self.board[0, action] != 0:
            return self._obs(), -1.0, True, {"illegal": True}

        r = ROWS - 1
        while r >= 0 and self.board[r, action] != 0:
            r -= 1
        self.board[r, action] = self.player
        self.last_move = (r, action)
        self.moves_played += 1

        if self._is_winner(r, action):
            return self._obs(), 1.0, True, {"win": True}

        if self.moves_played == ROWS * COLS:
            return self._obs(), 0.0, True, {"draw": True}

        self.player *= -1
        return self._obs(), 0.0, False, {}

    def _obs(self) -> Dict[str, Any]:
        return {"board": self.board.copy(), "player": int(self.player), "mask": self.valid_actions()}

    def _is_winner(self, r: int, c: int) -> bool:
        p = self.board[r, c]
        if p == 0:
            return False
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in dirs:
            cnt = 1
            rr, cc = r - dr, c - dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr, cc] == p:
                cnt += 1; rr -= dr; cc -= dc
            rr, cc = r + dr, c + dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and self.board[rr, cc] == p:
                cnt += 1; rr += dr; cc += dc
            if cnt >= CONNECT_N:
                return True
        return False

    def clone(self):
        env = Connect4Env()
        env.board = self.board.copy()
        env.player = self.player
        env.last_move = self.last_move
        env.moves_played = self.moves_played
        return env

    def render(self):
        symbols = {1: "X", -1: "O", 0: "."}
        print("\n  0 1 2 3 4 5 6")
        for r in range(ROWS):
            row = " ".join(symbols[int(v)] for v in self.board[r])
            print(f"{r} {row}")
        print(f"Next to move: {'X' if self.player == 1 else 'O'}")
