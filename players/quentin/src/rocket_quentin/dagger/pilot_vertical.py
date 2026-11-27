"""Pilote de référence pour une fusée droite et centrée sur le pont."""

import numpy as np

GRAVITY_M_S2 = 9.81
MAX_DESCENT_M_S = 12.0
TOUCHDOWN_TARGET_M_S = 0.6
ALTITUDE_GAIN_S_INV = 0.55
VELOCITY_GAIN_S_INV = 3.0


def vertical_action(observation: np.ndarray) -> list[float]:
    """Choisit la poussée à partir de la hauteur, de la vitesse et des contacts."""
    if observation[-4:].any():
        return [-1.0, 0.0, 0.0]

    altitude = max(0.0, float(observation[2]))
    vertical_speed = float(observation[5])
    target_speed = -min(MAX_DESCENT_M_S, TOUCHDOWN_TARGET_M_S + ALTITUDE_GAIN_S_INV * altitude)
    desired_acceleration = VELOCITY_GAIN_S_INV * (target_speed - vertical_speed)

    # Dans cet environnement, une commande de 0 compense le poids.
    # Chaque unité de commande ajoute ou retranche 9,81 m/s² à cette accélération nulle.
    throttle = float(np.clip(desired_acceleration / GRAVITY_M_S2, -1.0, 1.0))
    return [throttle, 0.0, 0.0]
