"""Neural networks backing the MuZero agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return F.relu(out)


class RepresentationNet(nn.Module):
    def __init__(
        self, input_channels: int, channels: int, res_blocks: int, latent_dim: int
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(res_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.stem(obs)
        x = self.body(x)
        x = self.head(x)
        return torch.tanh(x)


class DynamicsNet(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim + action_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([latent, action], dim=-1)
        x = F.relu(self.fc1(x))
        new_latent = torch.tanh(self.fc2(x))
        reward = torch.tanh(self.reward_head(new_latent))
        return new_latent, reward.squeeze(-1)


class PredictionNet(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_logits = self.policy_head(latent)
        value = torch.tanh(self.value_head(latent)).squeeze(-1)
        return policy_logits, value


class MuZeroNet(nn.Module):
    """Wrapper combining representation, dynamics, and prediction heads."""

    def __init__(
        self, channels: int, res_blocks: int, latent_dim: int, action_dim: int
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.representation = RepresentationNet(3, channels, res_blocks, latent_dim)
        self.dynamics = DynamicsNet(latent_dim, action_dim)
        self.prediction = PredictionNet(latent_dim, action_dim)

    def encode_action(self, actions: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(actions.long(), num_classes=self.action_dim)
        return one_hot.to(actions.device).float()

    @torch.no_grad()
    def initial_inference(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.representation(obs)
        policy_logits, value = self.prediction(latent)
        return policy_logits, value, latent

    @torch.no_grad()
    def recurrent_inference(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_encoded = self.encode_action(action)
        latent_next, reward = self.dynamics(latent, action_encoded)
        policy_logits, value = self.prediction(latent_next)
        return policy_logits, value, reward, latent_next

    def training_initial_inference(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.representation(obs)
        policy_logits, value = self.prediction(latent)
        return policy_logits, value, latent

    def training_recurrent_inference(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_encoded = self.encode_action(action)
        latent_next, reward = self.dynamics(latent, action_encoded)
        policy_logits, value = self.prediction(latent_next)
        return policy_logits, value, reward, latent_next


@dataclass
class NetworkOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    latent: torch.Tensor
