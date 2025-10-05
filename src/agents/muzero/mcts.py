"""Monte Carlo Tree Search implementation for MuZero."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.board import ConnectFourBoard

from .networks import MuZeroNet


@dataclass
class Node:
    prior: float
    player: int
    latent: torch.Tensor
    reward: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    is_terminal: bool = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def softmax_temperature(visit_counts: List[int], temperature: float) -> List[float]:
    if temperature <= 1e-2:
        best = max(visit_counts)
        policy = [1.0 if c == best else 0.0 for c in visit_counts]
        total = sum(policy)
        return [p / total if total else 1.0 / len(policy) for p in policy]
    exps = [c ** (1.0 / temperature) for c in visit_counts]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(exps)] * len(exps)
    return [x / total for x in exps]


@dataclass
class RootDebugInfo:
    to_move: int
    legal: torch.Tensor
    priors: torch.Tensor
    visit_counts: torch.Tensor
    q_values: torch.Tensor
    chosen: int
    root_value_net: float
    used_dirichlet: bool
    temperature: float


class MCTS:
    def __init__(
        self,
        net: MuZeroNet,
        c_puct: float,
        discount: float,
        action_space: int = 7,
        device: torch.device | None = None,
    ) -> None:
        self.net = net
        self.c_puct = c_puct
        self.discount = discount
        self.action_space = action_space
        self.device = device or torch.device("cpu")
        self.last_root_debug: RootDebugInfo | None = None
        self.used_dirichlet = False
        self.temperature = 0.0
        self._root_value_net = 0.0

    def _legal_actions(self, board: ConnectFourBoard) -> List[int]:
        return board.valid_moves()

    def _mask_policy(
        self, logits: torch.Tensor, board: ConnectFourBoard
    ) -> torch.Tensor:
        tensor = torch.as_tensor(logits, dtype=torch.float32)
        if tensor.ndim > 1:
            tensor = tensor.squeeze(0)
        legal = self._legal_actions(board)
        if not legal:
            return torch.zeros(
                self.action_space, dtype=torch.float32, device=tensor.device
            )
        mask = torch.full(
            (self.action_space,), float("-inf"), dtype=torch.float32, device=tensor.device
        )
        idx = torch.tensor(legal, dtype=torch.long, device=tensor.device)
        mask[idx] = tensor[idx]
        probs = F.softmax(mask, dim=-1)
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            uniform = torch.zeros(
                self.action_space, dtype=torch.float32, device=tensor.device
            )
            uniform[idx] = 1.0 / len(legal)
            return uniform
        return probs

    def _expand_root(self, board: ConnectFourBoard, player: int) -> Node:
        obs = torch.from_numpy(board.encode_planes(player)).unsqueeze(0).to(self.device)
        policy_logits, value, latent = self.net.initial_inference(obs)
        policy = self._mask_policy(policy_logits[0], board)
        root = Node(prior=1.0, player=player, latent=latent[0])
        legal = self._legal_actions(board)
        if not legal:
            root.is_terminal = True
            self._root_value_net = value.item()
            return root
        for action in legal:
            child = Node(
                prior=policy[action].item(),
                player=3 - player,
                latent=torch.zeros_like(latent[0]),
            )
            root.children[action] = child
        root.value_sum = value.item()
        root.visit_count = 1
        self._root_value_net = value.item()
        return root

    def _visit_count_policy(self, node: Node, temperature: float) -> List[float]:
        counts = torch.zeros(self.action_space, dtype=torch.float32)
        for action, child in node.children.items():
            counts[action] = child.visit_count
        legal = [action for action in node.children.keys()]
        if not legal:
            if counts.sum() == 0:
                return [1.0 / self.action_space] * self.action_space
            probs = counts / counts.sum()
            return probs.tolist()
        if temperature <= 1e-6:
            policy = [0.0] * self.action_space
            best_action = max(legal, key=lambda a: counts[a])
            policy[best_action] = 1.0
            return policy
        counts = counts.pow(1.0 / temperature)
        total = counts.sum().item()
        if total <= 0.0 or not np.isfinite(total):
            uniform = [0.0] * self.action_space
            for action in legal:
                uniform[action] = 1.0 / len(legal)
            return uniform
        probs = (counts / total).tolist()
        return probs

    def search(
        self,
        board: ConnectFourBoard,
        player: int,
        simulations: int,
        temperature: float,
        add_noise: bool,
        dirichlet_alpha: float,
        dirichlet_eps: float,
        eval_mode: bool = False,
    ) -> Tuple[List[float], float]:
        root = self._expand_root(board, player)
        self.used_dirichlet = False
        self.temperature = temperature
        if add_noise and root.children:
            self._apply_root_noise(root, dirichlet_alpha, dirichlet_eps)
            self.used_dirichlet = True

        for _ in range(simulations):
            board_copy = board.copy()
            self._simulate(root, board_copy, player)

        policy = self._visit_count_policy(root, temperature)
        visit_counts = torch.zeros(self.action_space, dtype=torch.float32)
        q_values = torch.zeros(self.action_space, dtype=torch.float32)
        priors = torch.zeros(self.action_space, dtype=torch.float32)
        legal_mask = torch.zeros(self.action_space, dtype=torch.float32)
        for action, child in root.children.items():
            visit_counts[action] = float(child.visit_count)
            child_q = float(child.value())
            q_values[action] = float(child.reward + self.discount * -child_q)
            priors[action] = float(child.prior)
            legal_mask[action] = 1.0
        chosen = int(np.argmax(policy)) if policy else -1
        self.last_root_debug = RootDebugInfo(
            to_move=player,
            legal=legal_mask,
            priors=priors,
            visit_counts=visit_counts,
            q_values=q_values,
            chosen=chosen,
            root_value_net=float(self._root_value_net),
            used_dirichlet=self.used_dirichlet,
            temperature=self.temperature,
        )
        if eval_mode:
            assert not self.used_dirichlet, "Dirichlet noise must be disabled during evaluation"
            assert abs(self.temperature) <= 1e-6, "Temperature must be 0 during evaluation"
        return policy, root.value()

    def _simulate(self, root: Node, board: ConnectFourBoard, root_player: int) -> None:
        node = root
        search_path: List[Node] = [node]
        current_player = root_player

        while True:
            if node.is_terminal or not node.children:
                value = self._terminal_value(board, current_player)
                break
            action, child = self._select_child(node)
            board.drop(action)
            node = child
            current_player = child.player
            search_path.append(node)
            if child.visit_count == 0:
                value = self._expand_child(search_path[-2], child, action, board)
                break

        self._backup(search_path, value)

    def _backup(self, path: List[Node], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            # Values are stored from the perspective of the player to act at each node.
            # After we step to the parent the perspective flips, so we negate.
            assert isinstance(value, float) or isinstance(value, (int, np.floating))
            value = node.reward + self.discount * -value

    def _terminal_value(self, board: ConnectFourBoard, player_to_move: int) -> float:
        winner = board.check_winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == player_to_move else -1.0

    def _expand_child(
        self, parent: Node, child: Node, action: int, board: ConnectFourBoard
    ) -> float:
        policy_logits, value, reward, latent_next = self.net.recurrent_inference(
            parent.latent.unsqueeze(0), torch.tensor([action], device=self.device)
        )
        child.reward = reward.item()
        child.latent = latent_next[0]
        child.is_terminal = board.is_full() or board.check_winner() != 0
        if not child.is_terminal:
            policy = self._mask_policy(policy_logits[0], board)
            legal = self._legal_actions(board)
            child.children = {}
            for a in legal:
                child.children[a] = Node(
                    prior=policy[a].item(),
                    player=3 - child.player,
                    latent=torch.zeros_like(child.latent),
                )
        return value.item()

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)
        best_score = -float("inf")
        best_action = -1
        best_child: Node | None = None
        for action, child in node.children.items():
            child_q = child.value()
            q = child.reward + self.discount * -child_q
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        assert best_child is not None
        return best_action, best_child

    def _apply_root_noise(self, root: Node, alpha: float, eps: float) -> None:
        distribution = torch.distributions.Dirichlet(
            torch.full((len(root.children),), alpha)
        )
        noise = distribution.sample()
        for (action, child), noise_val in zip(root.children.items(), noise):
            child.prior = child.prior * (1 - eps) + noise_val.item() * eps

    def debug_self_check(
        self,
        board: ConnectFourBoard,
        player: int,
        *,
        simulations: int,
        temperature: float = 0.0,
    ) -> RootDebugInfo:
        self.search(
            board,
            player,
            simulations=simulations,
            temperature=temperature,
            add_noise=False,
            dirichlet_alpha=1.0,
            dirichlet_eps=0.0,
        )
        assert self.last_root_debug is not None
        info = self.last_root_debug
        illegal = (info.legal == 0) & (info.priors != 0)
        if illegal.any():
            raise AssertionError("Illegal moves received positive prior mass in MCTS")
        assert info.used_dirichlet is False
        return info


__all__ = ["MCTS", "Node", "RootDebugInfo", "softmax_temperature"]
