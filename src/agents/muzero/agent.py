"""MuZero agent implementation with self-play training."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from alive_progress import alive_bar

from src.agents.base import Agent
from src.board import ConnectFourBoard
from src.match import play_match
from src.agents.random_policy import RandomAgent

from .mcts import MCTS, RootDebugInfo
from .networks import MuZeroNet
from .replay import GameHistory, ReplayBuffer

if TYPE_CHECKING:  # pragma: no cover
    from . import MuZeroConfig
    from typing import Protocol

    class GatingCallback(Protocol):
        def __call__(
            self, agent: "MuZeroAgent", iteration: int, save_path: Path
        ) -> None:
            ...

ACTION_SPACE = 7


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


@dataclass
class TrainingState:
    iteration: int = 0
    total_self_play_moves: int = 0


class MuZeroAgent(Agent):
    def __init__(
        self, config: MuZeroConfig, device: Optional[torch.device] = None
    ) -> None:
        super().__init__(name="muzero")
        self.config = config
        self.device = device or resolve_device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        self.net = MuZeroNet(
            channels=config.model.channels,
            res_blocks=config.model.res_blocks,
            latent_dim=config.model.latent_dim,
            action_dim=ACTION_SPACE,
        ).to(self.device)
        self.optimizer = AdamW(
            self.net.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        if self.device.type == "mps":
            for module in self.net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module._forward_hooks.clear()
        self.scheduler = None
        self.replay = ReplayBuffer(
            capacity=config.train.replay_capacity,
            action_space=ACTION_SPACE,
            unroll_steps=config.model.unroll_steps,
        )
        self.writer = SummaryWriter(config.logging.tensorboard)
        self.training_state = TrainingState()
        self.use_amp = config.train.amp and self.device.type in {"cuda", "mps"}
        self.policy_temperature_moves = config.self_play.temperature_moves
        self.random_generator = random.Random(config.seed)
        self.global_step = 0
        self.eval_stochastic = False
        self.net.eval()
        self.policy_weight = config.train.policy_weight
        self.value_weight = config.train.value_weight
        self.reward_weight = config.train.reward_weight
        self.debug_mcts = False
        self.last_mcts_debug: Optional[RootDebugInfo] = None
        self.sims = config.mcts.simulations

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def select_action(
        self,
        board: ConnectFourBoard,
        training: bool = False,
        move_index: int = 0,
        stochastic: bool = False,
        override_temperature: Optional[float] = None,
        simulations: Optional[int] = None,
        return_policy: bool = False,
    ) -> int | tuple[int, List[float]]:
        mcts = MCTS(
            self.net,
            c_puct=self.config.mcts.c_puct,
            discount=self.config.mcts.value_discount,
            action_space=ACTION_SPACE,
            device=self.device,
        )
        sims = self.config.mcts.simulations if simulations is None else simulations
        eval_mode = not training

        if eval_mode:
            temperature = 0.0
        elif override_temperature is not None:
            temperature = override_temperature
        elif training:
            temperature = 1.0 if move_index < self.policy_temperature_moves else 1e-3
        elif stochastic or self.eval_stochastic:
            temperature = 1.0 if move_index < self.policy_temperature_moves else 1e-3
        else:
            temperature = 0.0

        self.sims = sims
        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            policy, root_value = mcts.search(
                board.copy(),
                board.current_player,
                simulations=sims,
                temperature=temperature,
                add_noise=training,
                dirichlet_alpha=self.config.self_play.dirichlet_alpha,
                dirichlet_eps=self.config.self_play.dirichlet_eps,
                eval_mode=eval_mode,
            )
        if was_training:
            self.net.train()
        legal = board.valid_moves()
        policy_array = np.array(policy, dtype=np.float64)
        mask = np.zeros_like(policy_array)
        if legal:
            mask[legal] = policy_array[legal]
            total = mask.sum()
            if total <= 1e-8:
                mask[legal] = 1.0 / len(legal)
            else:
                mask /= total
        else:
            mask[:] = 1.0 / ACTION_SPACE

        if training:
            effective_stochastic = True
        else:
            effective_stochastic = stochastic or self.eval_stochastic
        if eval_mode:
            effective_stochastic = False
        if effective_stochastic:
            action = int(np.random.choice(np.arange(ACTION_SPACE), p=mask))
        else:
            if legal:
                best_idx = int(np.argmax(mask[legal]))
                action = int(legal[best_idx])
            else:
                action = int(np.argmax(mask))
        self.last_mcts_debug = mcts.last_root_debug
        if (
            eval_mode
            and self.debug_mcts
            and self.last_mcts_debug is not None
            and sims > 0
        ):
            debug = self.last_mcts_debug
            payload = {
                "to_move": int(debug.to_move),
                "legal": debug.legal.cpu().int().tolist(),
                "priors": [round(float(x), 4) for x in debug.priors.cpu().tolist()],
                "visit_counts": [int(round(float(x))) for x in debug.visit_counts.cpu().tolist()],
                "q_values": [round(float(x), 4) for x in debug.q_values.cpu().tolist()],
                "chosen": int(debug.chosen),
                "root_value_net": round(float(debug.root_value_net), 4),
            }
            print(json.dumps(payload))
        if return_policy:
            return action, mask.tolist(), float(root_value)
        return action

    # ------------------------------------------------------------------
    # Self-play & Training
    # ------------------------------------------------------------------
    def train_iterations(
        self,
        iterations: int,
        save_path: Optional[Path] = None,
        save_every: Optional[int] = None,
        resume: bool = False,
        extra_overrides: Optional[Dict[str, float]] = None,
        gating_callback: Optional["GatingCallback"] = None,
    ) -> None:
        save_every = save_every or self.config.logging.save_every_iters
        if iterations <= 0:
            return
        with alive_bar(
            iterations,
            title="MuZero Training",
            bar="smooth",
            spinner="dots_waves2",
        ) as bar:
            for _ in range(iterations):
                self.training_state.iteration += 1
                self.net.eval()
                histories, sp_stats = self._generate_self_play_games()
                for game in histories:
                    self.replay.add_game(game)

                bar.text = (
                    f"iter={self.training_state.iteration} "
                    f"replay={len(self.replay)} wins={sp_stats['wins']}"
                )

                self.writer.add_scalar(
                    "self_play/wins", sp_stats["wins"], self.training_state.iteration
                )
                self.writer.add_scalar(
                    "self_play/draws", sp_stats["draws"], self.training_state.iteration
                )
                self.writer.add_scalar(
                    "self_play/losses",
                    sp_stats["losses"],
                    self.training_state.iteration,
                )
                self.writer.add_scalar(
                    "replay/positions", len(self.replay), self.training_state.iteration
                )
                metrics = sp_stats.get("metrics", {})
                if "avg_length" in metrics:
                    self.writer.add_scalar(
                        "self_play/avg_length",
                        metrics["avg_length"],
                        self.training_state.iteration,
                    )
                if "avg_entropy" in metrics:
                    self.writer.add_scalar(
                        "self_play/root_entropy",
                        metrics["avg_entropy"],
                        self.training_state.iteration,
                    )
                if "avg_root_value" in metrics:
                    self.writer.add_scalar(
                        "self_play/root_value",
                        metrics["avg_root_value"],
                        self.training_state.iteration,
                    )

                if len(self.replay) >= self.config.train.warm_start_steps:
                    self.net.train()
                    losses = self._optimize_network()
                    for key, value in losses.items():
                        self.writer.add_scalar(f"train/{key}", value, self.global_step)
                else:
                    self.writer.add_scalar(
                        "train/skipped", 1, self.training_state.iteration
                    )

                if save_path and self.training_state.iteration % save_every == 0:
                    self.save_checkpoint(save_path)
                    if gating_callback is not None:
                        gating_callback(self, self.training_state.iteration, save_path)

                bar()
        self.writer.flush()

    def _generate_self_play_games(self) -> Tuple[List[GameHistory], Dict[str, int]]:
        games: List[GameHistory] = []
        wins = draws = losses = 0
        total_length = 0
        entropy_sum = 0.0
        root_value_sum = 0.0
        total_roots = 0
        mirror_probability = 0.5 if self.config.self_play.mirror_openings else 0.0
        for _ in range(self.config.self_play.games_per_iter):
            board = ConnectFourBoard()
            history = GameHistory()
            mirror_game = (
                mirror_probability > 0.0
                and self.random_generator.random() < mirror_probability
            )
            for move in range(self.config.self_play.max_moves):
                player = board.current_player
                to_move = 1 if player == 1 else -1
                obs = board.encode_planes(player).astype(np.float32)
                if mirror_game:
                    obs = obs[:, :, ::-1]
                action, policy, root_value = self.select_action(
                    board,
                    training=True,
                    move_index=move,
                    simulations=self.config.mcts.simulations,
                    return_policy=True,
                )
                policy_array = np.array(policy, dtype=np.float64)
                entropy_sum += float(
                    -np.sum(policy_array * np.log(policy_array + 1e-8))
                )
                root_value_sum += root_value
                total_roots += 1
                result = board.drop(action)
                reward = 0.0
                if result.winner is not None:
                    if result.winner == player:
                        reward = 1.0
                    else:
                        reward = -1.0
                store_policy = policy
                store_action = action
                if mirror_game:
                    store_policy = list(reversed(policy))
                    store_action = ACTION_SPACE - 1 - action
                history.add_step(obs, player, to_move, store_policy, store_action, reward)
                if result.winner is not None or result.board_full:
                    history.winner = result.winner or 0
                    break
            if history.winner == 1:
                wins += 1
            elif history.winner == 2:
                losses += 1
            else:
                draws += 1
            total_length += history.game_length()
            games.append(history)
            self.training_state.total_self_play_moves += history.game_length()
        averages = {}
        if games:
            averages["avg_length"] = total_length / len(games)
        if total_roots > 0:
            averages["avg_entropy"] = entropy_sum / total_roots
            averages["avg_root_value"] = root_value_sum / total_roots
        return games, {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "metrics": averages,
        }

    def _optimize_network(self) -> Dict[str, float]:
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "reward_loss": 0.0,
            "entropy": 0.0,
        }
        batches = self.config.train.epochs_per_iter
        for _ in range(batches):
            batch = self.replay.sample_batch(self.config.train.batch_size)
            losses, entropy = self._train_batch(batch)
            stats["policy_loss"] += losses["policy"]
            stats["value_loss"] += losses["value"]
            stats["reward_loss"] += losses["reward"]
            stats["entropy"] += entropy
            self.global_step += 1
        for key in stats:
            stats[key] /= batches
        return stats

    def _train_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], float]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        target_policy = batch["target_policy"].to(self.device)
        target_value = batch["target_value"].to(self.device)
        target_reward = batch["target_reward"].to(self.device)
        mask = batch["mask"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            policy_logits, value, latent = self.net.training_initial_inference(obs)
            policy_loss = self._policy_loss(policy_logits, target_policy[:, 0])
            value_loss = F.mse_loss(value, target_value[:, 0])
            entropy = (
                -(
                    F.softmax(policy_logits, dim=-1)
                    * F.log_softmax(policy_logits, dim=-1)
                )
                .sum(dim=-1)
                .mean()
            )

            reward_loss_total = torch.tensor(0.0, device=self.device)

            for step in range(self.config.model.unroll_steps):
                logits, v, r, latent = self.net.training_recurrent_inference(
                    latent, actions[:, step]
                )
                step_mask = mask[:, step]
                policy_step = self._policy_loss(
                    logits, target_policy[:, step + 1], reduction="none"
                )
                value_step = F.mse_loss(v, target_value[:, step + 1], reduction="none")
                reward_step = F.mse_loss(r, target_reward[:, step], reduction="none")
                mask_sum = step_mask.sum().clamp_min(1e-6)
                policy_loss += (policy_step * step_mask).sum() / mask_sum
                value_loss += (value_step * step_mask).sum() / mask_sum
                reward_loss_total += (reward_step * step_mask).sum() / mask_sum

            reward_loss = reward_loss_total
            loss = (
                self.policy_weight * policy_loss
                + self.value_weight * value_loss
                + self.reward_weight * reward_loss
            )

        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.train.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return (
            {
                "policy": float(policy_loss.detach().cpu()),
                "value": float(value_loss.detach().cpu()),
                "reward": float(reward_loss.detach().cpu()),
            },
            float(entropy.detach().cpu()),
        )

    def _policy_loss(
        self, logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target * log_probs).sum(dim=-1)
        if reduction == "none":
            return loss
        return loss.mean()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls, path: Path, device: Optional[torch.device] = None
    ) -> "MuZeroAgent":
        payload = torch.load(path, map_location=device if device is not None else "cpu")
        from . import MuZeroConfig

        cfg_dict = payload.get("config", {})
        config = MuZeroConfig.from_dict(cfg_dict)
        agent = cls(config, device=device)
        agent.load_checkpoint(path, overwrite_config=True)
        return agent

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "iteration": self.training_state.iteration,
            "buffer": self._serialize_buffer(),
        }
        torch.save(payload, path)
        config_path = path.with_suffix(".yaml")
        with config_path.open("w", encoding="utf-8") as handle:
            import yaml

            yaml.safe_dump(asdict(self.config), handle)

    def load_checkpoint(self, path: Path, *, overwrite_config: bool = False) -> None:
        payload = torch.load(path, map_location=self.device)
        if overwrite_config and "config" in payload:
            from . import MuZeroConfig as _MuZeroConfig

            self.config = _MuZeroConfig.from_dict(payload["config"])
        self.net.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        scheduler_state = payload.get("scheduler")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        self.training_state.iteration = payload.get("iteration", 0)
        self._load_buffer(payload.get("buffer", []))
        self.net.to(self.device)
        self.net.eval()

    def _serialize_buffer(self) -> List[Dict[str, List[float]]]:
        data: List[Dict[str, List[float]]] = []
        for game in self.replay.buffer:
            data.append(
                {
                    "observations": [obs.tolist() for obs in game.observations],
                    "actions": game.actions,
                    "policies": game.policies,
                    "rewards": game.rewards,
                    "players": game.players,
                    "winner": game.winner,
                }
            )
        return data

    def _load_buffer(self, data: Iterable[Dict[str, List[float]]]) -> None:
        self.replay.buffer.clear()
        self.replay.num_positions = 0
        for entry in data:
            game = GameHistory()
            game.observations = [
                np.array(x, dtype=np.float32) for x in entry.get("observations", [])
            ]
            game.actions = list(entry.get("actions", []))
            game.policies = [list(p) for p in entry.get("policies", [])]
            game.rewards = list(entry.get("rewards", []))
            game.players = list(entry.get("players", []))
            game.winner = int(entry.get("winner", 0))
            self.replay.add_game(game)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def play_against(
        self, opponent: Agent, games: int, render: bool = False
    ) -> Dict[str, int]:
        result = play_match(self, opponent, games=games, swap_start=True, render=render)
        return {
            "wins": result.agent_one_wins,
            "losses": result.agent_two_wins,
            "draws": result.draws,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _sample_from_policy(self, policy: List[float]) -> int:
        return int(np.random.choice(np.arange(ACTION_SPACE), p=np.array(policy)))


__all__ = ["MuZeroAgent", "resolve_device"]
