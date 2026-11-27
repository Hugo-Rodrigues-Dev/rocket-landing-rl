"""Pilote de référence pour une pose sur un pont oscillant."""

import numpy as np

from rocket_quentin.pilot_attitude import attitude_action
from rocket_quentin.pilot_lateral import lateral_action
from rocket_quentin.pilot_vertical import GRAVITY_M_S2

APPROACH_HEIGHT_M = 1.5


def wave_reference(observation: np.ndarray, drag_acceleration_xy: np.ndarray | None = None) -> list[float]:
    height = observation[2]
    if height >= APPROACH_HEIGHT_M:
        return lateral_action(observation, drag_acceleration_xy)

    weight = float(np.clip(1 - height / APPROACH_HEIGHT_M, 0, 1))
    direction = observation[18:21].copy()
    # Privilégie progressivement l'alignement au pont à mesure que le contact approche.
    lateral = np.clip(-0.3 * observation[:2] - 0.8 * observation[3:5], -0.9, 0.9)
    if drag_acceleration_xy is not None:
        lateral -= drag_acceleration_xy
    direction[:2] += (1 - weight) * lateral / GRAVITY_M_S2
    direction /= np.linalg.norm(direction)

    relative = observation.copy()
    body_x, body_z = observation[9:12], observation[12:15]
    body = np.column_stack((body_x, np.cross(body_z, body_x), body_z))
    relative[6:9] -= body.T @ observation[21:24]
    # Augmente le freinage pour réduire la vitesse visée au toucher de 0,15 m/s.
    relative[5] -= 0.15 * weight
    return attitude_action(relative, direction)
