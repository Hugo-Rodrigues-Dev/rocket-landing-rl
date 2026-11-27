"""Vent horizontal, traînée et couples du RCS de phase 4."""

from math import cos, sin, sqrt, tau

import numpy as np

DRAG_FACTOR = 0.5 * 1.225 * 1.0 * 10.0
MAX_GUST_M_S = 3.0
RCS_TORQUE_LIMITS = (1000.0, 1000.0, 100.0)


def drag_force(relative_air_velocity: np.ndarray) -> np.ndarray:
    return -DRAG_FACTOR * np.linalg.norm(relative_air_velocity) * relative_air_velocity


class Wind:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        direction = rng.uniform(0, tau)
        self.mean = rng.uniform(0, 8) * np.array([cos(direction), sin(direction)])
        self.gust = np.zeros(2)
        self.velocity = np.array([*self.mean, 0.0])

    def advance(self, interval: float) -> None:
        self.gust *= 1 - interval
        self.gust += 1.5 * sqrt(interval) * self.rng.standard_normal(2)
        norm = np.linalg.norm(self.gust)
        if norm > MAX_GUST_M_S:
            self.gust *= MAX_GUST_M_S / norm
        self.velocity[:2] = self.mean + self.gust
