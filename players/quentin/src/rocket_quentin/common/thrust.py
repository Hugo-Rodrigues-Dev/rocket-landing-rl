"""Poussée du moteur dans la simulation MuJoCo."""

from math import cos, radians, sin

import mujoco
import numpy as np

MAX_THRUST_N = 19_620.0
MAX_GIMBAL_DEG = 10.0


def apply_thrust(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    thrust_n: float,
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
) -> None:
    """Écrit la force et le couple du moteur avant un pas de physique.

    Les angles sont mesurés autour des axes x et y de la fusée. Réappeler
    la fonction avant chaque pas pour suivre l'orientation de la fusée.
    """
    thrust = np.clip(thrust_n, 0.0, MAX_THRUST_N)
    angle_x = radians(np.clip(tilt_x_deg, -MAX_GIMBAL_DEG, MAX_GIMBAL_DEG))
    angle_y = radians(np.clip(tilt_y_deg, -MAX_GIMBAL_DEG, MAX_GIMBAL_DEG))

    # La poussée pointe vers +z dans le repère fusée quand les angles sont nuls.
    # On tourne ce vecteur autour de x, puis autour de y.
    direction = np.array([sin(angle_y) * cos(angle_x), -sin(angle_x), cos(angle_y) * cos(angle_x)])
    force_local = thrust * direction

    rocket_id = model.body("rocket").id
    lever_local = model.site("nozzle").pos - model.body_ipos[rocket_id]
    torque_local = np.cross(lever_local, force_local)

    # MuJoCo attend la force et le couple au centre de masse, dans le repère monde.
    # Le quaternion de l'articulation libre donne l'orientation du pas courant.
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, data.joint("rocket_free").qpos[3:])
    body_to_world = rotation.reshape(3, 3)
    data.xfrc_applied[rocket_id, :3] = body_to_world @ force_local
    data.xfrc_applied[rocket_id, 3:] = body_to_world @ torque_local
