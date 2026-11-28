"""Pilote de référence pour les départs complets, à partir des 19 observations."""

import numpy as np

from rocket_hugo.env import GIMBAL_LIMIT_RAD
from rocket_hugo.vertical import GRAVITY


# Borne la norme en conservant la direction, pour éviter une préférence entre les axes.
def _limit_norm(vector, limit):
    return vector * min(1.0, limit / max(np.linalg.norm(vector), 1e-12))


# Convertit l'état mesuré en poussée et en deux commandes de tuyère.
# Le guidage propose d'abord un déplacement, puis une rotation pour l'obtenir.
def pilot_action(observation):
    position = np.asarray(observation[:3], dtype=np.float64) * 100
    velocity = np.asarray(observation[3:6], dtype=np.float64) * 20
    height = position[2]
    if height < 0.025 or np.any(observation[15:19]):
        return np.array((-1.0, 0.0, 0.0))

    # Plus on est près du pont, plus la vitesse de descente visée est faible.
    descent_speed = -min(10.0, max(0.45, 0.65 * height))
    vertical_acceleration = np.clip((descent_speed - velocity[2]) / 0.65, -4.0, 6.0)
    lateral_velocity = _limit_norm(-0.32 * position[:2], 5.0)
    lateral_acceleration = _limit_norm((lateral_velocity - velocity[:2]) / 1.4, 2.3)
    thrust_world = np.array((*lateral_acceleration, GRAVITY + vertical_acceleration))

    # Exprime la direction de poussée souhaitée dans le repère de la fusée.
    body_x, body_z = observation[6:9], observation[9:12]
    body_y = np.cross(body_z, body_x)
    desired_up = np.array((body_x @ thrust_world, body_y @ thrust_world, body_z @ thrust_world))
    desired_up /= np.linalg.norm(desired_up)
    # Le produit vectoriel donne l'axe de rotation ; le terme de vitesse amortit la rotation.
    angular_acceleration = 6.0 * np.array((-desired_up[1], desired_up[0])) - 4.5 * observation[12:14] * 2

    # La poussée est appliquée à 5 m du centre de masse. Les deux signes de tuyère
    # sont opposés aux signes des couples demandés (voir scene.xml et env._apply_force).
    thrust_per_kg = np.clip(thrust_world[2] / max(float(body_z[2]), 0.5), 0, 2 * GRAVITY)
    gimbal = -angular_acceleration * (8.395833 / 5) / max(thrust_per_kg, 1.0)
    return np.clip((thrust_per_kg / GRAVITY - 1, *(gimbal / GIMBAL_LIMIT_RAD)), -1, 1)
