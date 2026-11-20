"""Le moteur de la fusée : poussée orientable par une tuyère (gimbal) sur deux axes.

Unités physiques ici (newtons, radians). La conversion depuis les actions de l'agent
([-1, 1]) se fera dans l'environnement.

Conventions, dans le repère de la fusée :
- la tuyère pivote d'abord de gimbal[0] autour de x, puis de gimbal[1] autour de y ;
- la poussée vaut T · u avec u = Rx(gimbal[0]) · Ry(gimbal[1]) · e_z (vers la pointe à 0°) ;
- elle s'applique au pivot de la tuyère (site « moteur »), sous le centre de masse,
  donc gimbal[0] > 0 fait tourner la fusée autour de -x, et gimbal[1] > 0 autour de -y.
"""

import mujoco
import numpy as np

T_MAX = 2 * 1000 * 9.81  # N, deux fois le poids à vide (contrat)
GIMBAL_MAX = np.radians(10)  # débattement de la tuyère (contrat)

_NOZZLE_HALF = 0.15  # demi-longueur de la tuyère (m)
_FLAME_RADIUS = 0.25
_FLAME_MAX_HALF = 1.6  # demi-longueur de la flamme à pleine poussée (m)


def gimbal_quat(gimbal: np.ndarray) -> np.ndarray:
    """Quaternion de la rotation Rx(gimbal[0]) · Ry(gimbal[1])."""
    qx = np.array([np.cos(gimbal[0] / 2), np.sin(gimbal[0] / 2), 0.0, 0.0])
    qy = np.array([np.cos(gimbal[1] / 2), 0.0, np.sin(gimbal[1] / 2), 0.0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qx, qy)
    return q


def thrust_direction(gimbal: np.ndarray) -> np.ndarray:
    """Direction unitaire de la poussée dans le repère de la fusée."""
    u = np.zeros(3)
    mujoco.mju_rotVecQuat(u, np.array([0.0, 0.0, 1.0]), gimbal_quat(gimbal))
    return u


class Engine:
    """Applique la poussée à la fusée et met à jour le décor (tuyère, flamme)."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.body = model.body("fusee").id
        self.site = model.site("moteur").id
        self.nozzle = model.geom("tuyere").id
        self.flame = model.geom("flamme").id
        self.pivot = model.site_pos[self.site].copy()  # pivot de la tuyère, repère fusée
        # MuJoCo marque ces géométries « orientées comme le corps » et ignorerait alors geom_quat.
        model.geom_sameframe[[self.nozzle, self.flame]] = 0
        self._thrust_body = np.zeros(3)  # poussée commandée, repère fusée

    def command(self, thrust: float, gimbal) -> None:
        """Fixe la commande (une fois par décision de l'agent) et met à jour le décor."""
        thrust = float(np.clip(thrust, 0.0, T_MAX))
        gimbal = np.clip(np.asarray(gimbal, dtype=float), -GIMBAL_MAX, GIMBAL_MAX)
        u = thrust_direction(gimbal)
        self._thrust_body = thrust * u
        self._update_visuals(gimbal, u, thrust / T_MAX)

    def push(self, data: mujoco.MjData) -> None:
        """Applique la commande courante. À appeler avant chaque mj_step, car la fusée tourne."""
        # Force dans le repère du monde, appliquée au pivot : MuJoCo l'attend au centre de masse,
        # on ajoute donc le couple r × F (r = pivot - centre de masse).
        force = data.xmat[self.body].reshape(3, 3) @ self._thrust_body
        lever = data.site_xpos[self.site] - data.xipos[self.body]
        data.xfrc_applied[self.body, :3] = force
        data.xfrc_applied[self.body, 3:] = np.cross(lever, force)

    def apply(self, data: mujoco.MjData, thrust: float, gimbal) -> None:
        """Raccourci : command() puis push()."""
        self.command(thrust, gimbal)
        self.push(data)

    def _update_visuals(self, gimbal: np.ndarray, u: np.ndarray, throttle: float) -> None:
        q = gimbal_quat(gimbal)
        self.model.geom_quat[self.nozzle] = q
        self.model.geom_pos[self.nozzle] = self.pivot - u * _NOZZLE_HALF
        half = max(throttle * _FLAME_MAX_HALF, 1e-3)
        self.model.geom_quat[self.flame] = q
        self.model.geom_size[self.flame, 1] = half
        self.model.geom_pos[self.flame] = self.pivot - u * (2 * _NOZZLE_HALF + _FLAME_RADIUS + half)
        self.model.geom_rgba[self.flame, 3] = 0.85 if throttle > 0.01 else 0.0
