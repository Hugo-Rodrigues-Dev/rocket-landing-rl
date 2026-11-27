"""Guidage vers le centre du pont en coordonnées relatives."""

import numpy as np

from rocket_quentin.dagger.pilot_attitude import attitude_action
from rocket_quentin.dagger.pilot_vertical import GRAVITY_M_S2

POSITION_GAIN_S_INV = 0.4
SPEED_GAIN_S_INV = 1.1
MAX_HORIZONTAL_SPEED_M_S = 5.0
MAX_HORIZONTAL_ACCELERATION_M_S2 = 2.5


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    """Borne la longueur d'un vecteur en conservant sa direction."""
    norm = np.linalg.norm(vector)
    return vector * (limit / norm) if norm > limit else vector


def lateral_action(observation: np.ndarray, drag_acceleration_xy: np.ndarray | None = None) -> list[float]:
    """Vise le centre du pont, freine la dérive et commande la descente."""
    position_xy = observation[:2]
    velocity_xy = observation[3:5]
    target_speed = _limit_norm(-POSITION_GAIN_S_INV * position_xy, MAX_HORIZONTAL_SPEED_M_S)
    target_acceleration = _limit_norm(
        SPEED_GAIN_S_INV * (target_speed - velocity_xy), MAX_HORIZONTAL_ACCELERATION_M_S2
    )
    if drag_acceleration_xy is not None:
        target_acceleration -= drag_acceleration_xy

    # Pour accélérer dans le plan horizontal, le pilote incline l'axe de la poussée.
    target_up = np.array([*target_acceleration, GRAVITY_M_S2])
    target_up /= np.linalg.norm(target_up)
    return attitude_action(observation, target_up)
