"""Signal d'entraînement fondé sur la progression vers le pont."""

from collections.abc import Callable
from math import hypot

import gymnasium as gym
import numpy as np

SAC_DISCOUNT = 0.995


def landing_potential(observation: np.ndarray) -> float:
    """Attribue un score entre −20 et 0 à partir de la hauteur et du décalage."""
    x, y, height = observation[:3]
    height = max(0.0, float(height))
    distance = hypot(x, y)
    return -10.0 * (height / (height + 1.0) + distance / (distance + 2.0))


class ProgressReward(gym.Wrapper):
    def __init__(self, env: gym.Env, potential: Callable = landing_potential):
        super().__init__(env)
        self._potential_fn = potential

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._potential = self._potential_fn(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Après une troncature, SAC estime encore la valeur du dernier état.
        next_potential = 0.0 if terminated else self._potential_fn(observation)
        shaped_reward = reward + SAC_DISCOUNT * next_potential - self._potential
        self._potential = next_potential
        return observation, shaped_reward, terminated, truncated, info


def approach_potential(observation: np.ndarray) -> float:
    """Valorise à proximité du pont les états compatibles avec une pose douce."""
    height = max(0.0, float(observation[2]))
    influence = 2.0 / (2.0 + height)
    speed = np.linalg.norm(observation[3:6])
    rotation = np.linalg.norm(observation[6:9])
    deck_up = observation[18:21] if len(observation) >= 29 else (0.0, 0.0, 1.0)
    tilt = 1.0 - float(np.clip(observation[12:15] @ deck_up, -1.0, 1.0))
    return float(
        landing_potential(observation)
        - influence * (12.0 * speed / (speed + 2.0) + 6.0 * rotation / (rotation + 0.25) + 12.0 * tilt)
    )
