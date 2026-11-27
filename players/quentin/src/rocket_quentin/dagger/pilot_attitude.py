"""Ajoute le redressement aux commandes du pilote vertical."""

import numpy as np

from rocket_quentin.dagger.pilot_vertical import vertical_action

ORIENTATION_GAIN = 8.0
ANGULAR_VELOCITY_GAIN = 4.5


def attitude_action(observation: np.ndarray, target_up=(0.0, 0.0, 1.0)) -> list[float]:
    """Oriente l'axe vertical de la fusée vers un vecteur unitaire dans le monde."""
    action = vertical_action(observation)
    if action[0] == -1.0:
        return action

    # Les axes x et z observés sont exprimés dans le repère du monde.
    body_x = observation[9:12]
    body_z = observation[12:15]
    body_y = np.cross(body_z, body_x)

    # La direction cible projetée sur x et y donne l'erreur de rotation dans le repère fusée.
    angular_error = np.array([-np.dot(body_y, target_up), np.dot(body_x, target_up)])
    angular_velocity = observation[6:8]
    correction = ORIENTATION_GAIN * angular_error - ANGULAR_VELOCITY_GAIN * angular_velocity

    # La tuyère est sous le centre de masse : un angle positif crée un couple négatif.
    action[1:] = np.clip(-correction, -1.0, 1.0).tolist()
    return action
