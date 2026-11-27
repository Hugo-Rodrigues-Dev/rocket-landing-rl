"""Signal d'entraînement fondé sur la progression vers le pont."""

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
    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._potential = landing_potential(observation)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Après une troncature, SAC estime encore la valeur du dernier état.
        next_potential = 0.0 if terminated else landing_potential(observation)
        shaped_reward = reward + SAC_DISCOUNT * next_potential - self._potential
        self._potential = next_potential
        return observation, shaped_reward, terminated, truncated, info
