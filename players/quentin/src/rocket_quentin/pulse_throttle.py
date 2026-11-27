"""Adapte la poussée d'une politique historique au moteur de la phase 2."""

from collections.abc import Callable

import numpy as np

from rocket_quentin.fuel import FUEL_OBSERVATION_INDEX, MIN_THRUST_FRACTION

DRY_MASS_KG = 1000.0


class PulseThrottle:
    def __init__(self, policy: Callable):
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self._credit = 0.0

    def __call__(self, observation: np.ndarray) -> list[float]:
        throttle, tilt_x, tilt_y = self.policy(observation)
        mass_ratio = 1 + float(observation[FUEL_OBSERVATION_INDEX]) / DRY_MASS_KG
        requested = float(np.clip(0.5 * (throttle + 1) * mass_ratio, 0, 1))
        if requested == 0 or observation[-4:].any():
            self.reset()
            return [-1.0, tilt_x, tilt_y]

        self._credit += requested
        if requested >= MIN_THRUST_FRACTION:
            delivered = float(np.clip(self._credit, MIN_THRUST_FRACTION, 1.0))
        elif self._credit >= MIN_THRUST_FRACTION / 2:
            delivered = MIN_THRUST_FRACTION
        else:
            delivered = 0.0
        self._credit -= delivered
        return [2 * delivered - 1, tilt_x, tilt_y]
