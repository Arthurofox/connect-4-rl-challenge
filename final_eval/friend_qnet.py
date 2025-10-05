from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class QNet(nn.Module):
    """
    Default Q-network matching the reference DQN architecture.

    Replace this with your friend's exact model if needed; otherwise this serves
    as the expected loader for state_dict checkpoints.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=4, padding=1)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self._flatten_size = self._infer_flatten_size()
        self.fc1 = nn.Linear(self._flatten_size, 384)
        self.fc2 = nn.Linear(384, 7)

    def _infer_flatten_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 3, 6, 7)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
