"""Custom callbacks supporting progress reporting and self-play."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import random

from stable_baselines3.common.callbacks import BaseCallback

from src.agents.base import Agent
from src.agents.sb3_agent import SB3Agent


class AliveProgressCallback(BaseCallback):
    """Drives an ``alive_progress`` bar using SB3 training updates."""

    def __init__(self, bar, total_timesteps: int) -> None:  # type: ignore[anno-var]
        super().__init__()
        self._bar = bar
        self._total = total_timesteps
        self._count = 0

    def _on_step(self) -> bool:
        current = min(self._total, getattr(self.model, "num_timesteps", self._count))
        while self._count < current:
            self._bar()
            self._count += 1
        return True


@dataclass
class _Snapshot:
    path: Optional[Path]
    agent: Agent


class SelfPlayCallback(BaseCallback):
    """Periodically snapshot the learner and refresh opponents for self-play."""

    def __init__(
        self,
        *,
        snapshot_dir: Path,
        initial_opponents: List[Agent],
        update_interval: int,
        pool_size: int,
        warmup_steps: int = 0,
        deterministic_snapshot: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.pool_size = max(pool_size, 1)
        self.update_interval = max(update_interval, 1)
        self.warmup_steps = max(warmup_steps, 0)
        self.deterministic_snapshot = deterministic_snapshot
        self._rng = random.Random(seed)
        self._pool: List[_Snapshot] = [
            _Snapshot(path=None, agent=opponent) for opponent in initial_opponents
        ]
        self._next_snapshot = (
            self.warmup_steps if self.warmup_steps > 0 else self.update_interval
        )

    def _choose(self) -> Agent:
        if len(self._pool) == 1:
            return self._pool[0].agent
        return self._rng.choice(self._pool).agent

    def _prune_if_needed(self) -> None:
        while len(self._pool) > self.pool_size:
            snapshot = self._pool.pop(0)
            if snapshot.path is not None:
                snapshot.path.unlink(missing_ok=True)

    def _on_training_start(self) -> None:
        opponent = self._choose()
        self.training_env.env_method("update_opponent", opponent, indices=None)  # type: ignore[attr-defined]

    def _on_step(self) -> bool:
        current = getattr(self.model, "num_timesteps", 0)
        if current >= self._next_snapshot:
            snapshot_path = self.snapshot_dir / f"snapshot_{current}.zip"
            self.model.save(snapshot_path)
            agent = SB3Agent.load(
                snapshot_path, deterministic=self.deterministic_snapshot
            )
            self._pool.append(_Snapshot(snapshot_path, agent))
            self._prune_if_needed()
            opponent = self._choose()
            self.training_env.env_method("update_opponent", opponent, indices=None)
            self._next_snapshot += self.update_interval
        return True


__all__ = ["AliveProgressCallback", "SelfPlayCallback"]
