"""Politique apprise de bout en bout, incluant l'allumage du moteur."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from rocket_quentin.common.env import ACTION_SIZES, OBSERVATION_SIZES
from rocket_quentin.common.fuel import MIN_THRUST_FRACTION

MIN_POWERED_COMMAND = 2 * MIN_THRUST_FRACTION - 1


class LandingPolicy(nn.Module):
    def __init__(self, phase: int):
        super().__init__()
        size = OBSERVATION_SIZES[phase] + 1
        self.phase = phase
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size))
        self.network = nn.Sequential(
            nn.Linear(size, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, ACTION_SIZES[phase] + 1),
        )

    @staticmethod
    def features(observation: torch.Tensor) -> torch.Tensor:
        height = observation[..., 2:3].clamp_min(0).log1p()
        return torch.cat((observation, height), dim=-1)

    def set_statistics(self, observations: torch.Tensor) -> None:
        features = self.features(observations)
        self.mean.copy_(features.mean(dim=0))
        self.std.copy_(features.std(dim=0).clamp_min(0.1))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        inputs = ((self.features(observation) - self.mean) / self.std).clamp(-12, 12)
        output = self.network(inputs)
        return torch.cat((output[..., :1], output[..., 1:].tanh()), dim=-1)

    @torch.inference_mode()
    def action(self, observation: np.ndarray) -> list[float]:
        margin, *action = self(torch.from_numpy(observation)).tolist()
        action[0] = max(MIN_POWERED_COMMAND, action[0]) if margin >= 0 else -1.0
        return action

    def save(self, path: Path) -> None:
        torch.save({"phase": self.phase, "weights": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        policy = cls(checkpoint["phase"])
        policy.load_state_dict(checkpoint["weights"])
        return policy.eval()
